"""Measure VLM accuracy for the drawing check: present AND absent elements on labelled pictures.

    .venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_vlm.py \
        --vlms qwen3-vl-8b qwen3.5-9b --dir /tmp/art_bench --out /tmp/vlm_bench.json

Pictures are the ones tools/art_bench_image.py makes (<backend>_<i>.png, scenes in the same order).
For each picture we ask about the things that are really drawn (should be true) and about
plausible things that are NOT drawn (should be false); accuracy counts both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# scene index -> (present, absent), labelled by looking at the generated pictures
TRUTH = {
    0: (["condor", "mountains", "houses"], ["llama", "river", "person"]),
    1: (["girl", "llama", "stone walls"], ["condor", "dog", "river"]),
    2: (["fox", "moon", "lake", "boat"], ["condor", "person", "sun"]),
    3: (["person", "loom", "dog", "house"], ["llama", "cat", "condor"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlms", nargs="+", default=["qwen3-vl-8b", "qwen3.5-9b"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="/tmp/art_vlm_bench.json")
    a = ap.parse_args()
    import mlx.core as mx
    from PIL import Image
    from yq.macworker.modelmgr import models
    from yq.macworker.models import art_verify, vlm
    pics = sorted(Path(a.dir).glob("*_[0-3].png"))
    res = {}
    for key in a.vlms:
        vlm.VLM = key
        mx.reset_peak_memory()
        right = total = 0
        times, rows = [], []
        for p in pics:
            i = int(p.stem.rsplit("_", 1)[1])
            present, absent = TRUTH[i]
            t0 = time.time()
            v = art_verify.verify(Image.open(p), present + absent)
            dt = time.time() - t0
            times.append(dt)
            ok = sum(1 for e in present if v["verified"].get(e)) + sum(1 for e in absent if not v["verified"].get(e))
            right += ok
            total += len(present) + len(absent)
            rows.append({"image": p.name, "seconds": round(dt, 1), "correct": ok, "of": len(present) + len(absent),
                         "verified": v["verified"], "line_art": v["line_art"], "text": v["text"],
                         "frame": v["frame"], "problems": v["problems"]})
            print(json.dumps({"vlm": key, **rows[-1]}), flush=True)
        res[key] = {"accuracy": round(right / max(1, total), 3), "median_s": round(sorted(times)[len(times) // 2], 1),
                    "first_s": round(times[0], 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "rows": rows}
        print(json.dumps({"summary": key, **{k: v for k, v in res[key].items() if k != "rows"}}), flush=True)
        models.unload_all()
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
