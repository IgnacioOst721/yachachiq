"""Write the latest measurements (~/yq-data/eval/voice/*.json) into docs/voice.md.

    .venvs/voice/bin/python tools/voice_report.py

Replaces the text between <!-- EVAL:ASR --> ... <!-- /EVAL:ASR --> (and MT, LID)
so the table in the docs is always the one really measured.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.common import config  # noqa: E402

DOC = V2 / "docs" / "voice.md"
SET_ES = {"fleurs_es": "FLEURS español (es-419)", "fleurs_en": "FLEURS inglés (en-US)",
          "fleurs_pt": "FLEURS portugués (pt-BR)", "cv_qxp": "Common Voice quechua de Puno (qxp)",
          "chanka_quy": "Quechua chanka (quy), lectura"}


def latest(prefix: str):
    """Merge every run: a later run overrides the same (set, engine) / pair; LID from the last run that has it."""
    files = sorted((config.DATA_DIR / "eval" / "voice").glob(prefix + "_*.json"))
    if not files:
        return None, ""
    merged, lid, n, dates = {}, [], None, []
    for f in files:
        d = json.loads(f.read_text())
        dates.append(d.get("date", ""))
        for r in d.get("results", []):
            key = (r.get("set"), r.get("engine")) if prefix == "asr" else (r.get("pair"), r.get("engine"))
            merged[key] = r
        if d.get("lid"):
            lid = d["lid"]
        n = d.get("n", n)
    out = {"results": list(merged.values()), "lid": lid, "n": n, "date": ", ".join(x for x in dates if x)}
    return out, ", ".join(f.name for f in files)


def asr_table(d: dict, name: str) -> str:
    rows = ["| Datos | Motor | WER % | CER % | RTF | Memoria pico | Carga |", "|---|---|---|---|---|---|---|"]
    for r in sorted(d["results"], key=lambda r: (list(SET_ES).index(r["set"]) if r["set"] in SET_ES else 99,
                                                  r["wer"])):
        rows.append("| %s (%d frases, %.0f min) | %s | %.1f | %.1f | %.3f | %.1f GB | %.0f s |" % (
            SET_ES.get(r["set"], r["set"]), r["n"], r["audio_min"], r["engine"], r["wer"], r["cer"], r["rtf"],
            r["peak_gb"], r["load_s"]))
    rows.append("\n_Archivo: `~/yq-data/eval/voice/%s` (%s)._" % (name, d["date"]))
    return "\n".join(rows)


def lid_table(d: dict) -> str:
    if not d.get("lid"):
        return "_(sin medición de detección automática)_"
    rows = ["| Datos | Acierto del ruteo (idioma exacto) | Acierto “familia” (cualquier quechua) | Whisper top-1 | MMS-LID top-1 | Qué decidió |",
            "|---|---|---|---|---|---|"]
    for r in d["lid"]:
        rows.append("| %s | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %s |" % (
            SET_ES.get(r["set"], r["set"]), r["router_acc"], r.get("router_group_acc", r["router_acc"]),
            r["whisper_top1"], r["mms_top1"], ", ".join("%s×%d" % (c or "?", n) for c, n in r["decided"])))
    return "\n".join(rows)


def mt_table(d: dict, name: str) -> str:
    rows = ["| Par | NLLB-200 1.3B chrF++ | MADLAD-400 3B chrF++ | NLLB s/frase | MADLAD s/frase |", "|---|---|---|---|---|"]
    by = {}
    for r in d["results"]:
        by.setdefault(r["pair"], {})[r["engine"]] = r
    for pair, e in by.items():
        n, m = e.get("nllb"), e.get("madlad")
        rows.append("| %s | %s | %s | %s | %s |" % (
            pair.replace(">", " → "), "%.1f" % n["chrf++"] if n else "–", "%.1f" % m["chrf++"] if m else "–",
            "%.2f" % n["s_per_sentence"] if n else "–", "%.2f" % m["s_per_sentence"] if m else "–"))
    rows.append("\n_FLORES-200 devtest, %s frases por par. Archivos: `~/yq-data/eval/voice/%s` (%s)._" % (
        d["n"], name, d["date"]))
    return "\n".join(rows)


def put(doc: str, key: str, body: str) -> str:
    pat = re.compile(r"<!-- EVAL:%s -->.*?<!-- /EVAL:%s -->" % (key, key), re.S)
    block = "<!-- EVAL:%s -->\n%s\n<!-- /EVAL:%s -->" % (key, body, key)
    if not pat.search(doc):
        raise SystemExit("marker EVAL:%s missing in %s" % (key, DOC))
    return pat.sub(lambda m: block, doc)


def main() -> int:
    doc = DOC.read_text(encoding="utf-8")
    asr, an = latest("asr")
    mt, mn = latest("mt")
    if asr:
        doc = put(doc, "ASR", asr_table(asr, an))
        doc = put(doc, "LID", lid_table(asr))
    if mt:
        doc = put(doc, "MT", mt_table(mt, mn))
    DOC.write_text(doc, encoding="utf-8")
    print("updated", DOC, "asr:", an or "-", "mt:", mn or "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
