"""Download every voice model so the robot works offline (run with internet, once).

    cd ~/yachachiq/v2
    .venvs/voice/bin/python tools/voice_download.py --mac          # on the MacBook (~22 GB)
    .venvs/voice/bin/python tools/voice_download.py --jetson       # on the Jetson (~2.5 GB)
    .venvs/voice/bin/python tools/voice_download.py --list         # what would be downloaded

Big files are fetched with several parallel HTTP ranges (the Meta CDN limits
each connection) and resumed when interrupted. Hugging Face repos go to the
normal HF cache (~/.cache/huggingface), Omnilingual checkpoints to fairseq2's
cache (~/.cache/fairseq2/assets/<sha1(url)[:24]>/), exactly where the
libraries look for them, so nothing touches the internet at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

OMNI_BASE = "https://dl.fbaipublicfiles.com/mms/"
MAC_OMNI = ["omniASR_tokenizer_written_v2.model", "omniASR-CTC-1B-v2.pt", "omniASR-CTC-300M-v2.pt",
            "omniASR-LLM-1B-v2.pt"]
MAC_HF = [
    ("mlx-community/whisper-large-v3-mlx", None),
    ("mlx-community/whisper-large-v3-turbo", None),
    ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", None),
    ("Nextcloud-AI/madlad400-3b-mt-ct2-int8", None),
    ("facebook/mms-lid-4017", ["*.json", "*.safetensors"]),
]
# MMS-TTS voices fetched for the Mac: Peruvian languages that have one + the big ones
MAC_TTS_EXTRA = ["spa", "eng", "por", "fra", "deu", "ita"]
JETSON_FW = ["dropbox-dash/faster-whisper-large-v3-turbo", "small"]   # faster-whisper models (CTranslate2)
JETSON_PIPER_LANGS = ["spa_Latn", "eng_Latn", "por_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "cmn_Hans",
                      "jpn_Jpan", "rus_Cyrl", "arb_Arab", "hin_Deva", "tur_Latn", "nld_Latn", "pol_Latn"]


def fairseq2_asset_path(url: str) -> Path:
    """Where fairseq2 0.6 caches `url` (see fairseq2/assets/download_manager.py)."""
    h = hashlib.sha1(url.encode()).hexdigest()[:24]
    return Path.home() / ".cache" / "fairseq2" / "assets" / h / url.rsplit("/", 1)[1]


def fetch_parallel(url: str, dest: Path, parts: int = 8, timeout: float = 30.0) -> Path:
    """Download `url` to `dest` with `parts` parallel ranges, resumable, with progress."""
    import requests
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print("  have", dest)
        return dest
    head = requests.head(url, allow_redirects=True, timeout=timeout)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", 0))
    tmpdir = dest.parent.parent / (dest.parent.name + ".partial")
    tmpdir.mkdir(parents=True, exist_ok=True)
    if size <= 0 or head.headers.get("Accept-Ranges", "bytes") != "bytes":
        parts = 1
    step = -(-size // parts) if size else 0
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(parts)] if size else [(0, -1)]
    done = [0] * len(ranges)
    errors: list = []

    def worker(i, lo, hi):
        part = tmpdir / ("part%02d" % i)
        for attempt in range(20):
            have = part.stat().st_size if part.exists() else 0
            done[i] = have
            if hi >= 0 and lo + have > hi:
                return
            try:
                hdr = {"Range": "bytes=%d-%s" % (lo + have, hi if hi >= 0 else "")} if size else {}
                with requests.get(url, headers=hdr, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(part, "ab") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                            done[i] += len(chunk)
                if hi < 0 or lo + part.stat().st_size > hi:
                    return
            except Exception as e:
                time.sleep(min(30, 2 ** attempt))
                if attempt == 19:
                    errors.append("part %d: %s" % (i, e))

    threads = [threading.Thread(target=worker, args=(i, lo, hi), daemon=True) for i, (lo, hi) in enumerate(ranges)]
    for t in threads:
        t.start()
    t0, last = time.time(), 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        if time.time() - last > 15:
            last = time.time()
            got = sum(done)
            print("  %s %.0f/%.0f MB (%.1f MB/s)" % (dest.name, got / 1e6, size / 1e6,
                                                    got / 1e6 / max(1e-3, time.time() - t0)), flush=True)
    if errors:
        raise RuntimeError("download failed: %s" % errors)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as out:
        for i in range(len(ranges)):
            out.write((tmpdir / ("part%02d" % i)).read_bytes())
    if size and tmp.stat().st_size != size:
        raise RuntimeError("size mismatch for %s: %d != %d" % (dest, tmp.stat().st_size, size))
    tmp.replace(dest)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  done", dest, "%.0f MB" % (size / 1e6))
    return dest


def hf(repo: str, patterns=None) -> str:
    from huggingface_hub import snapshot_download
    print("HF", repo, flush=True)
    return snapshot_download(repo, allow_patterns=patterns)


def mac(list_only: bool = False) -> None:
    from yq.common import languages
    omni = [(OMNI_BASE + n, fairseq2_asset_path(OMNI_BASE + n)) for n in MAC_OMNI]
    tts = sorted({l.engines["mms_tts"] for l in languages.peru() if l.engines.get("mms_tts")} | set(MAC_TTS_EXTRA))
    if list_only:
        for u, p in omni:
            print("omni", u, "->", p)
        for r, _ in MAC_HF:
            print("hf  ", r)
        print("mms-tts", len(tts), "voices:", " ".join(tts))
        return
    for r, pat in MAC_HF:
        hf(r, pat)
    for u, p in omni:
        print("OMNI", u, flush=True)
        fetch_parallel(u, p)
    for code in tts:
        hf("facebook/mms-tts-" + code)


def jetson(list_only: bool = False) -> None:
    from yq.voice import settings, tts
    vad = settings.vad_model_path()
    if list_only:
        print("vad", settings.VAD_MODEL_URL, "->", vad)
        print("faster-whisper", JETSON_FW, "->", settings.whisper_cache())
        print("piper", [tts.candidate_voices(l)[:1] for l in JETSON_PIPER_LANGS], "->", settings.piper_dir())
        return
    if not vad.exists():
        fetch_parallel(settings.VAD_MODEL_URL, vad, parts=1)
    for lang in JETSON_PIPER_LANGS:
        key = tts.ensure_voice(lang)
        print("piper", lang, key, flush=True)
    try:
        from faster_whisper import WhisperModel   # noqa: F401
        from faster_whisper.utils import download_model
        for name in JETSON_FW:
            print("faster-whisper", name, flush=True)
            download_model(name, cache_dir=str(settings.whisper_cache()))
    except ImportError:
        print("faster-whisper not installed here: skipping its models (install requirements/voice-jetson.txt)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mac", action="store_true")
    ap.add_argument("--jetson", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--url", help="download one URL with parallel ranges")
    ap.add_argument("--dest")
    ap.add_argument("--parts", type=int, default=8)
    a = ap.parse_args()
    if a.url:
        fetch_parallel(a.url, Path(a.dest or fairseq2_asset_path(a.url)), parts=a.parts)
        return 0
    if not (a.mac or a.jetson):
        ap.print_help()
        return 2
    if a.mac:
        mac(a.list)
    if a.jetson:
        jetson(a.list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
