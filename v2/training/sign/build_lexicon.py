"""Build the word-completion lexicons shipped in yq/sign/models/ (run once, needs internet).

    .venvs/sign/bin/python training/sign/build_lexicon.py

Sources (downloaded into training/sign/data/lexicon/):
  English  https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt
  Spanish  https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt
  (FrequencyWords by Hermit Dave: code MIT, word lists CC BY-SA 4.0; built from OpenSubtitles 2018)
Quechua: no frequency list with a clear license was found, so lexicon_que.tsv is made from our
own story_words.tsv only (see docs/licenses_sign.md).

Output: lexicon_eng.tsv and lexicon_spa.tsv (word<TAB>count, top N purely alphabetic words,
at least 2 letters except a/y/o/e/u/i), lexicon_que.tsv.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
OUT = V2 / "yq" / "sign" / "models"
DATA = HERE / "data" / "lexicon"
URLS = {
    "eng": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt",
    "spa": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt",
}
FILES = {"eng": "en_50k.txt", "spa": "es_50k.txt"}
TOP_N = 30000
SINGLE_OK = {"eng": {"a", "i"}, "spa": {"a", "y", "o", "e", "u"}}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for code, url in URLS.items():
        src = DATA / FILES[code]
        if not src.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, src)
        rows = []
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            w, c = parts[0].strip().lower(), int(parts[1])
            if not w.isalpha() or (len(w) == 1 and w not in SINGLE_OK[code]):
                continue
            rows.append((w, c))
            if len(rows) >= TOP_N:
                break
        dst = OUT / ("lexicon_%s.tsv" % code)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("# FrequencyWords 2018 (Hermit Dave), OpenSubtitles, CC BY-SA 4.0; top %d words\n" % len(rows))
            for w, c in rows:
                fh.write("%s\t%d\n" % (w, c))
        print("wrote", dst, len(rows), "words", "%.0f KB" % (dst.stat().st_size / 1024))
    # Quechua from our own story vocabulary
    que = []
    for line in (OUT / "story_words.tsv").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0] == "que":
            que.append(p[1])
    dst = OUT / "lexicon_que.tsv"
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("# Southern Quechua story words written by the Yachachiq team (see story_words.tsv)\n")
        for w in que:
            fh.write("%s\t%d\n" % (w, 1000))
    print("wrote", dst, len(que), "words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
