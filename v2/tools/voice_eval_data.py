"""Prepare small evaluation sets for tools/voice_eval.py (needs internet once).

Each set becomes ~/yq-data/cache/voice_eval/<set>/ with 16 kHz WAV files and a
manifest.jsonl {"wav","text","lang","source"}. Sets:

  fleurs_es / fleurs_en / fleurs_pt   google/fleurs test split (CC-BY-4.0), one reading per
                                      sentence, streamed from the start of test.tar.gz (no 600 MB download)
  cv_qxp        Common Voice Quechua (Puno/Collao, qxp; CC0) clips, from the test split of the
                HF mirror Epiph0nE/quechua-puno-tts (file names common_voice_qxp_*)
  chanka_quy    josemercado/quechua-chanka-asr (Ayacucho/Chanka Quechua, read speech;
                LICENSE NOT STATED: used only to measure, never redistributed)

    .venvs/voice/bin/python tools/voice_eval_data.py --n 80
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.common import config  # noqa: E402

ROOT = config.DATA_DIR / "cache" / "voice_eval"
FLEURS = {"fleurs_es": ("es_419", "spa_Latn"), "fleurs_en": ("en_us", "eng_Latn"), "fleurs_pt": ("pt_br", "por_Latn")}
HF = "https://huggingface.co/datasets/"
PARQUET = {
    "cv_qxp": (HF + "Epiph0nE/quechua-puno-tts/resolve/main/data/test-00000-of-00001.parquet",
               "quechua_puno_tts_test.parquet", "transcription", "qxp_Latn", "Common Voice qxp (CC0) via Epiph0nE"),
    "chanka_quy": (HF + "josemercado/quechua-chanka-asr/resolve/main/data/train-00000-of-00001.parquet",
                   "quechua_chanka_asr.parquet", "TEXTO", "quy_Latn", "josemercado/quechua-chanka-asr (license n/a)"),
}


def _write(set_dir: Path, rows: list) -> None:
    with open(set_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  %s: %d utterances, %.1f min" % (set_dir.name, len(rows), sum(r["seconds"] for r in rows) / 60))


def fleurs(name: str, n: int) -> None:
    import requests
    from yq.voice import audio as au
    cfg, lang = FLEURS[name]
    d = ROOT / name
    if (d / "manifest.jsonl").exists():
        print("  have", name)
        return
    d.mkdir(parents=True, exist_ok=True)
    tsv = requests.get(HF + "google/fleurs/resolve/main/data/%s/test.tsv" % cfg, timeout=60).text
    meta = {}
    for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        meta[row[1]] = (row[0], row[2])            # file -> (sentence id, raw transcription)
    rows, seen = [], set()
    url = HF + "google/fleurs/resolve/main/data/%s/audio/test.tar.gz" % cfg
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                base = Path(m.name).name
                if not m.isfile() or base not in meta or meta[base][0] in seen:
                    continue
                sid, text = meta[base]
                x = au.load_audio(tar.extractfile(m).read())
                au.save_wav(x, d / base)
                seen.add(sid)
                rows.append({"wav": base, "text": text, "lang": lang, "source": "google/fleurs %s test" % cfg,
                             "seconds": round(len(x) / 16000, 2)})
                if len(rows) >= n:
                    break
    _write(d, rows)


def parquet_set(name: str, n: int) -> None:
    import pyarrow.parquet as pq
    import requests
    from yq.voice import audio as au
    url, fname, col, lang, source = PARQUET[name]
    d = ROOT / name
    if (d / "manifest.jsonl").exists():
        print("  have", name)
        return
    d.mkdir(parents=True, exist_ok=True)
    local = ROOT / fname
    if not local.exists():
        print("  downloading", url)
        local.write_bytes(requests.get(url, timeout=3600).content)
    t = pq.read_table(local)
    # spread the sample over the whole file (different speakers), deterministic
    step = max(1, t.num_rows // n)
    rows = []
    for i in range(0, t.num_rows, step):
        r = t.slice(i, 1).to_pylist()[0]
        text = (r[col] or "").strip()
        if not text:
            continue
        x = au.load_audio(r["audio"]["bytes"])
        if len(x) < 8000:
            continue
        wav = "%s_%04d.wav" % (name, i)
        au.save_wav(x, d / wav)
        rows.append({"wav": wav, "text": text, "lang": lang, "source": source, "seconds": round(len(x) / 16000, 2),
                     "orig": r["audio"].get("path", "")})
        if len(rows) >= n:
            break
    _write(d, rows)


def load_set(name: str) -> list:
    d = ROOT / name
    rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["path"] = str(d / r["wav"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--sets", default=",".join(list(FLEURS) + list(PARQUET)))
    a = ap.parse_args()
    for s in a.sets.split(","):
        print("set", s, flush=True)
        (fleurs if s in FLEURS else parquet_set)(s, a.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
