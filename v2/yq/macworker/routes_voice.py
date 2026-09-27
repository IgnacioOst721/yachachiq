"""VOICE endpoints of the Mac worker (CONTRACTS.md §3.1).

POST /asr        multipart audio (WAV), lang ("auto" | code), prompt -> Transcript
POST /lid        multipart audio -> {"candidates": [[code, prob], ...]}
POST /translate  JSON {"text","src","tgt"} -> {"text","engine"}
POST /tts        JSON {"text","lang","voice"?} -> audio/wav
GET  /voice/info configured models and what is loaded (debugging)
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from yq.common import languages
from yq.common.contracts import Transcript, to_dict
from yq.macworker.modelmgr import models as _default_models
from yq.macworker.models import voice_router as vr
from yq.voice import audio as au
from yq.voice import settings

log = logging.getLogger("yq.voice.routes")
router = APIRouter()
MODELS = _default_models          # replaced by setup(); tests may use their own manager
LID_MODEL = "mms-lid-4017"
_asr_lock = threading.Lock()      # one visitor at a time; MLX/torch models are not re-entrant


# --- model registration ----------------------------------------------------------------------
def register_models(models) -> None:
    from yq.macworker.models import voice_lid, voice_omni, voice_translate, voice_whisper
    for name, size in voice_whisper.SIZE_GB.items():
        models.register(name, loader=lambda n=name: voice_whisper.load(n), size_gb=size,
                        unloader=voice_whisper.unload, kind="asr", repo=voice_whisper.REPOS[name])
    for name, (card, size) in voice_omni.CARDS.items():
        models.register(name, loader=lambda n=name: voice_omni.load(n), size_gb=size,
                        unloader=voice_omni.unload, kind="asr", card=card)
    models.register(LID_MODEL, loader=voice_lid.load, size_gb=voice_lid.SIZE_GB, kind="lid", repo=voice_lid.REPO)
    for eng, repo in voice_translate.REPOS.items():
        models.register("mt-" + eng, loader=lambda e=eng: voice_translate.load(e),
                        size_gb=voice_translate.SIZE_GB[eng], kind="translation", repo=repo)


def setup(jobs, models) -> None:
    global MODELS
    MODELS = models
    register_models(models)


def _tts_model(iso3: str):
    from yq.macworker.models import voice_tts
    name = "mms-tts-" + iso3
    if name not in MODELS.registered():
        MODELS.register(name, loader=lambda: voice_tts.load(iso3), size_gb=voice_tts.SIZE_GB, kind="tts")
    return MODELS.get(name)


# --- ASR -------------------------------------------------------------------------------------
def whisper_lid(x: np.ndarray) -> list:
    m = MODELS.get(settings.MAC_WHISPER_MODEL)
    out = []
    for code, p in m.detect_language(x)[:8]:
        c = languages.resolve(code)
        if c:
            out.append([c, round(float(p), 4)])
    return out


def mms_lid(x: np.ndarray) -> list:
    allowed = {l.iso639_3 for l in languages.all() if l.asr}
    m = MODELS.get(LID_MODEL)
    out = []
    for iso3, p in m.predict(x, allowed_iso3=allowed, top=8):
        c = languages.resolve(iso3)
        if c:
            out.append([c, p])
    return out


def _safe(fn, x) -> list:
    try:
        return fn(x)
    except Exception as e:
        log.warning("%s failed: %s", getattr(fn, "__name__", "lid"), e)
        return []


def run_engine(engine: str, x: np.ndarray, lang: str, prompt: str) -> Transcript:
    L = languages.get(lang) if lang else None
    dur = round(len(x) / 16000.0, 2)
    m = MODELS.get(engine)
    if engine.startswith("whisper"):
        wcode = L.whisper_code if L and L.whisper_code else None
        r = m.transcribe(x, language=wcode, prompt=prompt)
        detected = languages.resolve(r.get("language") or "") or (L.code if L else "")
        return Transcript(text=r["text"], lang=L.code if (L and wcode) else detected, engine=engine + "@mac",
                          confidence=float(r["confidence"]), duration_s=dur,
                          segments=[{"start": s["start"], "end": s["end"], "text": s["text"],
                                     "confidence": s["confidence"]} for s in r["segments"]])
    ocode = L.engines.get("omniasr") if L else None
    r = m.transcribe(x, lang=ocode)
    conf = r.get("confidence")
    return Transcript(text=r["text"], lang=L.code if L else "", engine=engine + "@mac",
                      confidence=round(float(conf), 3) if conf is not None else 0.5, duration_s=dur,
                      segments=[{"start": 0.0, "end": dur, "text": r["text"],
                                 "confidence": round(float(conf), 3) if conf is not None else 0.5}])


def transcribe(x: np.ndarray, lang: str = "auto", prompt: str = "") -> Transcript:
    whisper, omni = settings.MAC_WHISPER_MODEL, settings.MAC_OMNI_MODEL
    avail = set(MODELS.registered())
    t0 = time.time()
    if not lang or lang == "auto":
        wl = _safe(whisper_lid, x) if whisper in avail else []
        ml = _safe(mms_lid, x) if (settings.MAC_MMS_LID and LID_MODEL in avail) else []
        plan = vr.plan_auto(wl, ml, whisper, omni, settings.LID_WHISPER_MIN, settings.LID_MMS_MIN, avail)
    else:
        known = settings.MAC_OMNI_MODEL_KNOWN or omni
        if known not in avail:                     # precise model not installed here: the CTC one
            known = omni
        plan = vr.plan_for_language(lang, whisper, known, avail)
        if known != omni and known in plan.engines and omni in avail:    # CTC 1B right after, as fallback
            plan.engines.insert(plan.engines.index(known) + 1, omni)
    if not plan.engines:
        raise HTTPException(422, "no speech recognizer for language %r (%s)" % (lang, plan.reason))
    best, errors = None, []
    for eng in plan.engines:
        try:
            t = run_engine(eng, x, plan.lang, prompt)
        except Exception as e:
            log.warning("asr %s failed: %s", eng, e)
            errors.append("%s: %s" % (eng, e))
            continue
        if best is None or (t.text and t.confidence > best.confidence) or not best.text:
            best = t
        if t.text and t.confidence >= settings.ASR_RETRY_MIN_CONF:
            break
    if best is None:
        raise HTTPException(503, "every ASR engine failed: %s" % "; ".join(errors))
    best.lang_candidates = plan.candidates
    log.info("asr %s lang=%s conf=%.2f in %.1f s (%s)", best.engine, best.lang, best.confidence,
             time.time() - t0, plan.reason)
    return best


def _read_audio(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    try:
        return au.load_audio(data)
    except Exception as e:
        raise HTTPException(400, "audio must be a PCM WAV file (%s)" % e)


@router.post("/asr")
def asr_endpoint(audio: UploadFile = File(...), lang: str = Form("auto"), prompt: str = Form("")):
    x = _read_audio(audio)
    if lang and lang != "auto" and not languages.get(lang):
        raise HTTPException(400, "unknown language code %r" % lang)
    code = "auto" if (not lang or lang == "auto") else languages.resolve(lang)
    with _asr_lock:
        return to_dict(transcribe(x, code, prompt))


@router.post("/lid")
def lid_endpoint(audio: UploadFile = File(...)):
    x = _read_audio(audio)
    avail = set(MODELS.registered())
    with _asr_lock:
        wl = _safe(whisper_lid, x) if settings.MAC_WHISPER_MODEL in avail else []
        ml = _safe(mms_lid, x) if (settings.MAC_MMS_LID and LID_MODEL in avail) else []
    return {"candidates": vr.merge_lid(wl, ml)}


# --- translation -----------------------------------------------------------------------------
class TranslateIn(BaseModel):
    text: str
    src: str
    tgt: str


@router.post("/translate")
def translate_endpoint(req: TranslateIn):
    from yq.macworker.models import voice_translate as vt
    try:
        return vt.translate_detail(req.text, req.src, req.tgt, get_model=lambda e: MODELS.get("mt-" + e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# --- speech synthesis ------------------------------------------------------------------------
class TtsIn(BaseModel):
    text: str
    lang: str
    voice: str = ""


def synthesize(text: str, lang: str, voice: str = "") -> tuple:
    """(audio, rate, engine) or raises LookupError."""
    from yq.macworker.models import voice_tts
    L = languages.get(lang)
    if L is None:
        raise LookupError("unknown language %r" % lang)
    keys = [voice] if voice else []
    if not voice and L.engines.get("mms_tts"):
        x, r = _tts_model(L.engines["mms_tts"]).synthesize(text)
        return x, r, "mms-tts-" + L.engines["mms_tts"]
    from yq.voice import tts as jtts
    keys += jtts.candidate_voices(L.code)
    for k in keys:
        try:
            x, r = voice_tts.PiperMac(k).synthesize(text)
            return x, r, "piper:" + k
        except Exception:
            continue
    raise LookupError("no voice for %s" % L.code)


@router.post("/tts")
def tts_endpoint(req: TtsIn):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    try:
        with _asr_lock:
            x, rate, eng = synthesize(req.text, req.lang, req.voice)
    except LookupError as e:
        raise HTTPException(404, str(e))
    return Response(content=au.to_wav_bytes(x, rate), media_type="audio/wav", headers={"X-TTS-Engine": eng})


@router.get("/voice/info")
def info():
    return {"whisper": settings.MAC_WHISPER_MODEL, "omni": settings.MAC_OMNI_MODEL, "mms_lid": settings.MAC_MMS_LID,
            "loaded": [m for m in MODELS.loaded()], "languages": languages.meta().get("counts", {})}
