"""Measure pose-model speed on this machine (Mac: CPU vs CoreML; Jetson: TensorRT vs CUDA).

    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python tools/sign_benchmark.py
    .venvs/sign/bin/python tools/sign_benchmark.py --backends tensorrt cuda --models rtmw-m rtmw-l   # Jetson

Prints median latency (ms) and fps per model/backend, plus the full pipeline
(PoseEstimator.process with tracking + One Euro + hand refine) on a test image.
The first TensorRT run builds the engine (minutes); it is cached in MODELS_DIR/sign/trt_cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.sign import modelstore, rtm  # noqa: E402

SAMPLE = V2 / "training" / "sign" / "data" / "samples" / "astronaut.png"


def bench(fn, n: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cpu", "coreml"])
    ap.add_argument("--models", nargs="+", default=["rtmw-m", "rtmw-l", "rtmw-l-384", "rtmpose-hand", "yolox-tiny"])
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--image", default=str(SAMPLE))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import cv2
    img = cv2.imread(args.image) if Path(args.image).exists() else np.random.randint(0, 255, (800, 1280, 3), np.uint8)
    img = cv2.resize(img, (1280, 800)) if img.shape[:2] != (800, 1280) else img
    cache = modelstore.sign_dir() / "trt_cache"
    res = {}
    for be in args.backends:
        for name in args.models:
            p = modelstore.pose_model_path(name)
            if p is None:
                print("skip %s (not downloaded)" % name)
                continue
            wh = modelstore.POSE_MODELS[name][1]
            try:
                if name.startswith("yolox"):
                    m = rtm.PersonDetector(p, wh, be, cache)
                    ms = bench(lambda: m(img), args.n)
                    prov = m.sess.get_providers()[0]
                else:
                    m = rtm.TopDownPose(p, wh, be, cache)
                    box = np.array([200, 50, 1080, 800], np.float32)
                    ms = bench(lambda: m(img, box), args.n)
                    prov = m.providers[0]
            except Exception as e:
                print("%s/%s failed: %s" % (be, name, e))
                continue
            res["%s/%s" % (be, name)] = {"ms": round(ms, 2), "fps": round(1000 / ms, 1), "provider": prov}
            print("%-8s %-13s %7.2f ms  %6.1f fps  (%s)" % (be, name, ms, 1000 / ms, prov), flush=True)
    # full pipeline
    from yq.sign.pose import PoseEstimator
    for be in args.backends:
        for model in ("rtmw-m", "rtmw-l"):
            if modelstore.pose_model_path(model) is None:
                continue
            for refine in ("off", "letters"):
                est = PoseEstimator(model=model, backend=be, hand_refine=refine)
                est.refine_side = "right" if refine != "off" else None
                t = [0.0]

                def step():
                    t[0] += 1 / 30
                    est.process(img, t[0])
                ms = bench(step, args.n)
                key = "pipeline/%s/%s/refine=%s" % (be, model, refine)
                res[key] = {"ms": round(ms, 2), "fps": round(1000 / ms, 1)}
                print("%-40s %7.2f ms  %6.1f fps" % (key, ms, 1000 / ms), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
