"""Where the sign-language models live and how to download them (once, with internet).

At runtime everything is read from disk (offline). Layout under
``config.MODELS_DIR / "sign"``::

    pose/<name>/end2end.onnx          RTMW / RTMPose / YOLOX ONNX files (OpenMMLab, Apache-2.0)
    letters_<lang>.onnx + .json       our fingerspelling classifiers (also shipped in yq/sign/models/)
    words_<lang>.onnx + .json         our isolated-word classifiers
    recordings/                       what `python -m yq.sign.record` writes

Download (repeatable, documented in docs/sign.md):

    python -m yq.sign.modelstore download            # default set (~420 MB of zips)
    python -m yq.sign.modelstore download rtmw-m     # one model
    python -m yq.sign.modelstore list
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

from yq.common import config

PACKAGE_MODELS = Path(__file__).resolve().parent / "models"

_OPENMMLAB = "https://download.openmmlab.com/mmpose/v1/projects/"
# rtmlib's own fallback mirror of the same files (github.com/Tau-J/rtmlib, tools/file.py)
_HF_MIRROR = "https://huggingface.co/Tau-J/RTMPose/resolve/main/"

# name -> (path under projects/, input size (w, h), what it is)
POSE_MODELS = {
    # RTMW-m: distilled from RTMW-l ("dw-l-m"), whole body 133 kpts. rtmlib "lightweight" mode.
    "rtmw-m": ("rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip", (192, 256),
               "RTMW-m wholebody 133 kpts, 256x192"),
    # RTMW-l: distilled from RTMW-x ("dw-x-l"). rtmlib "balanced" mode.
    "rtmw-l": ("rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip", (192, 256),
               "RTMW-l wholebody 133 kpts, 256x192"),
    "rtmw-l-384": ("rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip", (288, 384),
                   "RTMW-l wholebody 133 kpts, 384x288 (rtmlib 'performance' pose model)"),
    # Person detector (only used to (re)acquire the visitor; then we track from keypoints).
    "yolox-tiny": ("rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416),
                   "YOLOX-tiny person detector (Human-Art)"),
    # Optional hand refinement for fingerspelling: RTMPose-m trained on 5 hand datasets.
    "rtmpose-hand": ("rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
                     (256, 256), "RTMPose-m hand 21 kpts, 256x256"),
}
DEFAULT_SET = ("rtmw-m", "rtmw-l", "yolox-tiny", "rtmpose-hand")


def sign_dir() -> Path:
    return Path(config.MODELS_DIR) / "sign"


def pose_dir() -> Path:
    return sign_dir() / "pose"


def recordings_dir() -> Path:
    return sign_dir() / "recordings"


def pose_model_path(name: str) -> Optional[Path]:
    """Path of the downloaded ONNX file for `name`, or None when missing."""
    d = pose_dir() / name
    if not d.is_dir():
        return None
    found = sorted(d.rglob("*.onnx"))
    return found[0] if found else None


def find_model(filename: str) -> Optional[Path]:
    """Our own trained model: MODELS_DIR/sign first (retrained), then the versioned copy."""
    for base in (sign_dir(), PACKAGE_MODELS):
        p = base / filename
        if p.exists():
            return p
    return None


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    import requests
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        last = -1
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(100 * done / total)
                    if pct // 10 != last:
                        last = pct // 10
                        print("  %3d%%  %.0f/%.0f MB" % (pct, done / 1e6, total / 1e6), flush=True)


def download(name: str, force: bool = False) -> Path:
    """Download and unzip one pose model into pose/<name>/. Returns the .onnx path."""
    rel, _size, desc = POSE_MODELS[name]
    have = pose_model_path(name)
    if have and not force:
        print("[ok] %s already at %s" % (name, have))
        return have
    out = pose_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="yq-sign-"))
    zpath = tmp / "model.zip"
    print("[get] %s (%s)" % (name, desc), flush=True)
    err = None
    for attempt in range(3):
        for base in (_OPENMMLAB, _HF_MIRROR):
            try:
                _download(base + rel, zpath)
                err = None
                break
            except Exception as e:  # server down / flaky network -> try the mirror, then retry
                err = e
                print("  failed from %s: %s" % (base, e), flush=True)
        if err is None:
            break
        time.sleep(5 * (attempt + 1))
    if err is not None:
        raise RuntimeError("could not download %s: %s" % (name, err))
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    (out / "source.json").write_text(json.dumps({"name": name, "url": _OPENMMLAB + rel, "sha256_zip": sha,
                                                 "description": desc}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    p = pose_model_path(name)
    if p is None:
        raise RuntimeError("zip for %s had no .onnx inside" % name)
    print("[ok] %s -> %s (zip sha256 %s)" % (name, p, sha[:16]))
    return p


def status() -> dict:
    return {n: (str(pose_model_path(n)) if pose_model_path(n) else None) for n in POSE_MODELS}


def main(argv: list) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "list":
        for n, p in status().items():
            print("%-14s %s" % (n, p or "(missing)"))
        return 0
    if argv[0] == "download":
        names = argv[1:] or list(DEFAULT_SET)
        if names == ["all"]:
            names = list(POSE_MODELS)
        for n in names:
            download(n, force="--force" in argv)
        return 0
    print("unknown command %r" % argv[0])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
