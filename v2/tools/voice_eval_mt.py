"""Compare translation engines (NLLB-200 1.3B vs MADLAD-400 3B, CTranslate2 int8)
on FLORES-200 devtest (CC-BY-SA-4.0) for the pairs Yachachiq needs.

    .venvs/voice/bin/python -m yq.common.heavylock .venvs/voice/bin/python tools/voice_eval_mt.py --n 100

Metric: chrF++ (sacrebleu, word_order=2) and BLEU; speed = seconds per sentence
on the M4 CPU. Results: ~/yq-data/eval/voice/mt_<date>.json + a markdown table.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.common import config  # noqa: E402

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
PAIRS = ["spa_Latn>quy_Latn", "quy_Latn>spa_Latn", "spa_Latn>ayr_Latn", "ayr_Latn>spa_Latn",
         "spa_Latn>eng_Latn", "eng_Latn>spa_Latn", "quy_Latn>eng_Latn", "ayr_Latn>eng_Latn"]


def flores(lang: str, n: int) -> list:
    tgz = config.DATA_DIR / "cache" / "voice_eval" / "flores200_dataset.tar.gz"
    if not tgz.exists():
        import requests
        tgz.parent.mkdir(parents=True, exist_ok=True)
        tgz.write_bytes(requests.get(FLORES_URL, timeout=600).content)
    with tarfile.open(tgz) as tar:
        f = tar.extractfile("./flores200_dataset/devtest/%s.devtest" % lang) or \
            tar.extractfile("flores200_dataset/devtest/%s.devtest" % lang)
        return f.read().decode("utf-8").splitlines()[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--engines", default="nllb,madlad")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    a = ap.parse_args()
    import sacrebleu
    from yq.macworker.models import voice_translate as vt
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "n": a.n, "dataset": "FLORES-200 devtest", "results": []}
    for eng in a.engines.split(","):
        t0 = time.time()
        model = vt.load(eng)
        load_s = time.time() - t0
        for pair in a.pairs.split(","):
            src, tgt = pair.split(">")
            codes = vt.engine_codes(src, tgt, eng)
            if not codes:
                print(eng, pair, "unsupported")
                continue
            srcs, refs = flores(src, a.n), flores(tgt, a.n)
            t0 = time.time()
            hyps = []
            for i in range(0, len(srcs), 8):
                hyps += model.translate_batch(srcs[i:i + 8], codes[0], codes[1])
            dt = time.time() - t0
            chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            r = {"engine": eng, "pair": pair, "chrf++": round(chrf, 1), "bleu": round(bleu, 1),
                 "s_per_sentence": round(dt / len(srcs), 3), "load_s": round(load_s, 1),
                 "example": {"src": srcs[0], "ref": refs[0], "hyp": hyps[0]}}
            out["results"].append(r)
            print("%-7s %-20s chrF++ %5.1f  BLEU %5.1f  %.2f s/sent" % (eng, pair, chrf, bleu, dt / len(srcs)),
                  flush=True)
        del model
    d = config.DATA_DIR / "eval" / "voice"
    d.mkdir(parents=True, exist_ok=True)
    path = d / ("mt_%s.json" % time.strftime("%Y%m%d_%H%M"))
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n| Par | Motor | chrF++ | BLEU | s/frase |\n|---|---|---|---|---|")
    for r in sorted(out["results"], key=lambda r: (r["pair"], r["engine"])):
        print("| %s | %s | %.1f | %.1f | %.2f |" % (r["pair"].replace(">", " → "), r["engine"], r["chrf++"], r["bleu"],
                                                 r["s_per_sentence"]))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
