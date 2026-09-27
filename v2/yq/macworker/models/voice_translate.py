"""Machine translation on the Mac with CTranslate2 (int8, CPU).

Engines (both open; chosen per language pair by `pick_engine`, see docs/voice.md
for the measured comparison):
  nllb    NLLB-200 distilled 1.3B  OpenNMT/nllb-200-distilled-1.3B-ct2-int8  (weights CC-BY-NC-4.0)
  madlad  MADLAD-400 3B MT         Nextcloud-AI/madlad400-3b-mt-ct2-int8    (weights Apache-2.0)

Python API (CONTRACTS.md §3.1):
    from yq.macworker.models.voice_translate import translate
    translate("Había una vez un cóndor", "spa_Latn", "quy_Latn") -> str
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

log = logging.getLogger("yq.voice.translate")

REPOS = {
    "nllb": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
    "madlad": "Nextcloud-AI/madlad400-3b-mt-ct2-int8",
}
SIZE_GB = {"nllb": 2.0, "madlad": 3.4}   # nllb: measured process RSS ~1.8-2.4 GB; madlad: estimate (2.95 GB file)
# Engine preference (first that supports both languages wins). Decided from the
# measured chrF on FLORES-200 devtest (tools/voice_eval_mt.py, docs/voice.md).
PREFERENCE = os.environ.get("YQ_MT_PREFERENCE", "nllb,madlad").split(",")
MAX_CHARS = 400        # longer inputs are split into sentences


class CT2Translator:
    def __init__(self, engine: str):
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        self.engine = engine
        try:
            path = snapshot_download(REPOS[engine], local_files_only=True)
        except Exception:
            path = snapshot_download(REPOS[engine])
        threads = int(os.environ.get("YQ_MT_THREADS", "4"))   # 8 threads thrash when other jobs share the CPU
        self.translator = ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=threads)
        if engine == "madlad":      # T5 SentencePiece; same pieces as the HF tokenizer, without its warnings
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=os.path.join(path, "spiece.model"))
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        self._lock = threading.Lock()

    def translate_batch(self, texts: list, src_code: str, tgt_code: str, beam: int = 4) -> list:
        tok = self.tokenizer
        with self._lock:
            if self.engine == "nllb":
                tok.src_lang = src_code
                src = [tok.convert_ids_to_tokens(tok.encode(t)) for t in texts]
                res = self.translator.translate_batch(src, target_prefix=[[tgt_code]] * len(src), beam_size=beam,
                                                      max_decoding_length=_max_len(src), repetition_penalty=1.1)
                outs = [r.hypotheses[0][1:] for r in res]
            else:   # madlad: "<2xx> text" + </s>
                src = [self.sp.encode("<2%s> %s" % (tgt_code, t), out_type=str) + ["</s>"] for t in texts]
                res = self.translator.translate_batch(src, beam_size=beam, max_decoding_length=_max_len(src),
                                                      repetition_penalty=1.1)
                return [self.sp.decode_pieces([p for p in r.hypotheses[0] if p not in ("</s>", "<pad>")]).strip()
                        for r in res]
        return [tok.decode(tok.convert_tokens_to_ids(o), skip_special_tokens=True).strip() for o in outs]


def _max_len(batch: list) -> int:
    """Output token limit: generous (agglutinative languages are longer) but stops runaway repetition."""
    return int(min(512, 16 + 3 * max(len(t) for t in batch)))


def load(engine: str) -> CT2Translator:
    return CT2Translator(engine)


def engine_codes(src: str, tgt: str, engine: str) -> Optional[tuple]:
    """(src_code, tgt_code) in the engine's own codes, or None when unsupported."""
    from yq.common import languages
    s, t = languages.get(src), languages.get(tgt)
    if not s or not t:
        return None
    if engine == "nllb":
        a, b = s.engines.get("nllb"), t.engines.get("nllb")
    else:
        a, b = s.engines.get("madlad"), t.engines.get("madlad")
    return (a, b) if a and b else None


def pick_engine(src: str, tgt: str, available: Optional[list] = None) -> Optional[str]:
    for eng in PREFERENCE:
        eng = eng.strip()
        if available is not None and eng not in available:
            continue
        if eng in REPOS and engine_codes(src, tgt, eng):
            return eng
    return None


def split_sentences(text: str, max_chars: int = MAX_CHARS) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?¡¿;:…])\s+", text)
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                out.append(cur)
            while len(p) > max_chars:           # no punctuation: cut at a space
                cut = p.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                out.append(p[:cut])
                p = p[cut:].strip()
            cur = p
    if cur:
        out.append(cur)
    return out


def translate_detail(text: str, src: str, tgt: str, engine: Optional[str] = None, get_model=None) -> dict:
    """{"text","engine"}. `get_model(engine)` returns a loaded CT2Translator (model manager)."""
    from yq.common import languages
    s, t = languages.resolve(src) or src, languages.resolve(tgt) or tgt
    text = (text or "").strip()
    if not text or s == t:
        return {"text": text, "engine": "same"}
    from . import voice_terms
    src_orig, tgt_orig = s, t
    engine = engine or pick_engine(s, t)
    # close varieties (Cusco/Puno Quechua...) go through NLLB as Ayacucho Quechua instead of MADLAD,
    # which invents text for them (voice_terms.PROXY)
    ps, pt = voice_terms.proxy(s), voice_terms.proxy(t)
    if engine in (None, "madlad") and (ps, pt) != (s, t) and engine_codes(ps, pt, "nllb"):
        s, t, engine = ps, pt, "nllb"
    if not engine:
        raise ValueError("no translation engine supports %s -> %s" % (s, t))
    codes = engine_codes(s, t, engine)
    if not codes:
        raise ValueError("%s does not support %s -> %s" % (engine, s, t))
    if get_model is None:
        from yq.macworker.modelmgr import models
        register()
        get_model = lambda e: models.get("mt-" + e)   # noqa: E731
    model = get_model(engine)
    pieces = split_sentences(text)
    outs = model.translate_batch(pieces, codes[0], codes[1])
    out = voice_terms.postedit(text, src_orig, " ".join(o for o in outs if o), tgt_orig)
    if voice_terms.degenerate(text, out):
        raise ValueError("translation %s -> %s with %s came out as garbage; not used" % (src_orig, tgt_orig, engine))
    from . import andean_lexicon           # NLLB loses animals/places in Quechua ("atuq" -> "un grupo")
    fixed = andean_lexicon.llm_postedit(text, src_orig, out, tgt_orig)
    postedited = fixed != out
    out = fixed
    return {"text": out, "engine": engine, "verified": engine == "nllb", "postedit": "glossary+llm" if postedited else "",
            "via": s if s != src_orig else (t if t != tgt_orig else "")}


def translate(text: str, src: str, tgt: str) -> str:
    """Contract API for other Mac domains (ART, BOX-ANALYSIS)."""
    return translate_detail(text, src, tgt)["text"]


def register() -> None:
    from yq.macworker.modelmgr import models
    for eng in REPOS:
        if "mt-" + eng not in models.registered():
            models.register("mt-" + eng, loader=lambda e=eng: load(e), size_gb=SIZE_GB[eng],
                            repo=REPOS[eng], kind="translation")
