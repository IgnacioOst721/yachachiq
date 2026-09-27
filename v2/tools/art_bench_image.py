"""Benchmark the image generators on this Mac: seconds per image, peak memory, samples.

    cd ~/yachachiq/v2
    .venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_image.py \
        --backends z-image-turbo flux2-klein-4b --out /tmp/art_bench

Writes <out>/<backend>_<i>.png and prints one JSON line per image plus a summary.
Each backend is unloaded before the next one so the numbers do not overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCENES = [
    "An Andean condor with wide open wings flying above snowy mountains, a small adobe village below",
    "A girl wearing a chullo and a poncho walking with her llama on a mountain path, stone terraces behind",
    "A red fox and a moon above a lake with reed boats, stars in the sky",
    "An old grandmother weaving a textile on a backstrap loom in front of her adobe house, a dog sleeping nearby",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["z-image-turbo", "flux2-klein-4b"])
    ap.add_argument("--out", default="/tmp/art_bench")
    ap.add_argument("--n", type=int, default=len(SCENES))
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import mlx.core as mx
    from yq.macworker.modelmgr import models
    from yq.macworker.models import art_image, art_style
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for b in a.backends:
        mx.reset_peak_memory()
        t0 = time.time()
        times = []
        for i, scene in enumerate(SCENES[: a.n]):
            prompt = art_style.compose_prompt(scene, b)
            t = time.time()
            img = art_image.generate(prompt, a.width, a.height, seed=a.seed + i, backend=b,
                                     negative=art_style.negative(b))
            dt = time.time() - t
            times.append(dt)
            img.save(out / ("%s_%d.png" % (b, i)))
            print(json.dumps({"backend": b, "i": i, "seconds": round(dt, 1)}), flush=True)
        summary[b] = {"first_incl_load_s": round(times[0], 1),
                      "median_s": round(sorted(times[1:] or times)[len(times[1:] or times) // 2], 1),
                      "total_s": round(time.time() - t0, 1),
                      "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        print(json.dumps({"summary": b, **summary[b]}), flush=True)
        models.unload_all()
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
