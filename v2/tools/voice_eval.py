"""Measure speech-recognition accuracy and speed on the M4 (writes the docs table).

    .venvs/voice/bin/python tools/voice_eval_data.py --n 80          # once, needs internet
    .venvs/voice/bin/python -m yq.common.heavylock \\
        .venvs/voice/bin/python tools/voice_eval.py --sets fleurs_es,cv_qxp --engines whisper-large-v3,omniasr-ctc-1b

For every (set, engine): corpus WER and CER on normalised text (yq.voice.textnorm),
real-time factor (processing seconds / audio seconds, model load excluded),
load time and peak memory (MLX peak for Whisper; child RSS + MPS allocation for
Omnilingual). The language is given to the engine (the kiosk asks the visitor
to pick it); `--lid` also measures automatic language identification.
Results go to ~/yq-data/eval/voice/asr_<date>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.common import config  # noqa: E402

DEFAULT_SETS = "fleurs_es,fleurs_en,fleurs_pt,cv_qxp,chanka_quy"
DEFAULT_ENGINES = "whisper-large-v3,whisper-large-v3-turbo,omniasr-ctc-300m,omniasr-ctc-1b,omniasr-llm-1b"


class Engine:
    def __init__(self, name: str):
        self.name = name
        self.peak_gb = 0.0
        t0 = time.time()
        if name.startswith("whisper"):
            import mlx.core as mx
            from yq.macworker.models import voice_whisper
            self.mx = mx
            (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()
            self.m = voice_whisper.load(name)
        else:
            from yq.macworker.models import voice_omni
            self.m = voice_omni.load(name)
        self.load_s = time.time() - t0

    def run(self, x, lang_code: str):
        from yq.common import languages
        L = languages.get(lang_code)
        if self.name.startswith("whisper"):
            r = self.m.transcribe(x, language=L.whisper_code or None)
            get_peak = getattr(self.mx, "get_peak_memory", None) or self.mx.metal.get_peak_memory
            self.peak_gb = max(self.peak_gb, get_peak() / 1e9)
            return r["text"], r["confidence"]
        r = self.m.transcribe(x, lang=L.engines.get("omniasr"))
        self.peak_gb = max(self.peak_gb, float(r.get("mem_gb") or 0.0))
        return r["text"], r.get("confidence")

    def close(self):
        if self.name.startswith("omni"):
            self.m.close()
        self.m = None


def evaluate(engine: Engine, rows: list) -> dict:
    import jiwer
    from yq.voice import audio as au
    from yq.voice.textnorm import normalize
    refs, hyps, confs, proc, audio_s, examples = [], [], [], 0.0, 0.0, []
    for i, r in enumerate(rows):
        x = au.load_audio(r["path"])
        t0 = time.time()
        text, conf = engine.run(x, r["lang"])
        dt = time.time() - t0
        if i > 0:                       # first call warms caches; not timed
            proc += dt
            audio_s += len(x) / 16000.0
        refs.append(normalize(r["text"]))
        hyps.append(normalize(text))
        if conf is not None:
            confs.append(float(conf))
        if i < 3:
            examples.append({"ref": r["text"], "hyp": text})
        if (i + 1) % 10 == 0:
            print("      %d/%d  %.1f s audio in %.1f s" % (i + 1, len(rows), audio_s, proc), flush=True)
    pairs = [(a, b) for a, b in zip(refs, hyps) if a]
    refs, hyps = [p[0] for p in pairs], [p[1] if p[1] else "∅" for p in pairs]
    return {"wer": round(100 * jiwer.wer(refs, hyps), 1), "cer": round(100 * jiwer.cer(refs, hyps), 1),
            "rtf": round(proc / max(audio_s, 1e-6), 3), "n": len(rows),
            "audio_min": round(sum(r["seconds"] for r in rows) / 60, 1),
            "mean_conf": round(sum(confs) / len(confs), 3) if confs else None, "examples": examples}


def eval_lid(sets: dict) -> list:
    """Automatic language identification: Whisper LID + MMS-LID -> router decision."""
    from yq.macworker.models import voice_lid, voice_router as vr, voice_whisper
    from yq.common import languages
    from yq.voice import audio as au, settings
    wm = voice_whisper.load(settings.MAC_WHISPER_MODEL)
    lid = voice_lid.load()
    allowed = {l.iso639_3 for l in languages.all() if l.asr}
    def group(code):
        L = languages.get(code)
        return "quechua" if L and "quechua" in (L.name_en + " " + " ".join(L.aliases)).lower() else code

    out = []
    for name, rows in sets.items():
        ok = ok_w = ok_m = ok_g = 0
        conf = {}
        for r in rows:
            x = au.load_audio(r["path"])
            wl = [[languages.resolve(c) or c, p] for c, p in wm.detect_language(x)[:8]]
            ml = [[languages.resolve(c) or c, p] for c, p in lid.predict(x, allowed_iso3=allowed, top=8)]
            plan = vr.plan_auto(wl, ml, settings.MAC_WHISPER_MODEL, settings.MAC_OMNI_MODEL,
                                settings.LID_WHISPER_MIN, settings.LID_MMS_MIN)
            ok += plan.lang == r["lang"]
            ok_g += group(plan.lang) == group(r["lang"])
            ok_w += bool(wl) and wl[0][0] == r["lang"]
            ok_m += bool(ml) and ml[0][0] == r["lang"]
            conf[plan.lang] = conf.get(plan.lang, 0) + 1
        n = len(rows)
        res = {"set": name, "lang": rows[0]["lang"], "n": n, "router_acc": round(100 * ok / n, 1),
               "router_group_acc": round(100 * ok_g / n, 1),
               "whisper_top1": round(100 * ok_w / n, 1), "mms_top1": round(100 * ok_m / n, 1),
               "decided": sorted(conf.items(), key=lambda kv: -kv[1])[:4]}
        print("LID", res, flush=True)
        out.append(res)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=DEFAULT_SETS)
    ap.add_argument("--engines", default=DEFAULT_ENGINES)
    ap.add_argument("--n", type=int, default=0, help="limit utterances per set (0 = all prepared)")
    ap.add_argument("--lid", action="store_true", help="also measure automatic language identification")
    ap.add_argument("--skip-whisper-on", default="cv_qxp,chanka_quy,cv_ayr",
                    help="sets where Whisper is not run (it does not know the language)")
    a = ap.parse_args()
    sys.path.insert(0, str(V2 / "tools"))
    from voice_eval_data import load_set
    sets = {s: load_set(s)[: a.n or None] for s in a.sets.split(",")}
    skip = set(a.skip_whisper_on.split(","))
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "machine": "MacBook M4 16 GB", "results": [], "lid": []}
    d = config.DATA_DIR / "eval" / "voice"
    d.mkdir(parents=True, exist_ok=True)
    path = d / ("asr_%s.json" % time.strftime("%Y%m%d_%H%M"))
    for eng_name in a.engines.split(","):
        todo = [s for s in sets if not (eng_name.startswith("whisper") and s in skip)]
        if not todo:
            continue
        print("== loading", eng_name, flush=True)
        try:
            eng = Engine(eng_name)
        except Exception as e:
            print("   cannot load %s: %s" % (eng_name, e), flush=True)
            continue
        for s in todo:
            res = evaluate(eng, sets[s])
            res.update({"set": s, "lang": sets[s][0]["lang"], "engine": eng_name, "load_s": round(eng.load_s, 1),
                        "peak_gb": round(eng.peak_gb, 2)})
            out["results"].append(res)
            print("   %-11s %-24s WER %5.1f  CER %5.1f  RTF %.3f  peak %.1f GB" %
                  (s, eng_name, res["wer"], res["cer"], res["rtf"], res["peak_gb"]), flush=True)
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        eng.close()
        del eng
        import gc
        gc.collect()
    if a.lid:
        out["lid"] = eval_lid(sets)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n| Conjunto | Idioma | Motor | WER % | CER % | RTF | Memoria pico GB |\n|---|---|---|---|---|---|---|")
    for r in out["results"]:
        print("| %s | %s | %s | %.1f | %.1f | %.3f | %.1f |" % (r["set"], r["lang"], r["engine"], r["wer"], r["cer"],
                                                             r["rtf"], r["peak_gb"]))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
