"""Compare LLMs on story planning (and optionally VLMs on picture checking).

    cd ~/yachachiq/v2
    .venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_bench_llm.py \
        --llms qwen3-8b qwen2.5-7b qwen3.5-9b --out /tmp/llm_bench.json
    ... --vlms qwen3-vl-8b qwen3.5-9b --images a.png b.png     (VLM part)

For every story in tools/art_stories.py it records seconds, whether the JSON was valid first
time, the elements and scene; the expected things are listed next to them so a human can
judge faithfulness and cultural accuracy (scored by hand in docs/art.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def bench_llm(key: str, stories: list) -> dict:
    import mlx.core as mx
    from yq.macworker.models import art_plan, llm
    llm.LLM = key
    mx.reset_peak_memory()
    rows = []
    t_load = time.time()
    llm.chat([{"role": "user", "content": "Hola"}], max_tokens=5, key=key)       # load
    load_s = time.time() - t_load
    for st in stories:
        t0 = time.time()
        try:
            p, es, en = art_plan.plan(st["text"], st["lang"], st.get("text_es", ""), "")
            row = {"id": st["id"], "ok": True, "seconds": round(time.time() - t0, 1), "title": p.title,
                   "title_es": p.title_es, "elements": p.elements, "expect": st["expect"],
                   "scene": art_plan_scene(p.prompt), "notes": p.cultural_notes, "text_en": en[:200]}
        except Exception as e:
            row = {"id": st["id"], "ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)[:300]}
        rows.append(row)
        print(json.dumps({"llm": key, **row}, ensure_ascii=False), flush=True)
    return {"load_s": round(load_s, 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "rows": rows}


def art_plan_scene(prompt: str) -> str:
    from yq.macworker.models import art_style
    return art_style.scene_of(prompt)


def bench_vlm(key: str, images: list, elements: list) -> dict:
    import mlx.core as mx
    from yq.macworker.models import art_verify, vlm
    vlm.VLM = key
    mx.reset_peak_memory()
    rows = []
    for im in images:
        t0 = time.time()
        v = art_verify.verify(__import__("PIL.Image", fromlist=["Image"]).open(im), elements)
        rows.append({"image": im, "seconds": round(time.time() - t0, 1), "verified": v["verified"],
                     "line_art": v["line_art"], "text": v["text"], "frame": v["frame"], "note": v["note"]})
        print(json.dumps({"vlm": key, **rows[-1]}, ensure_ascii=False), flush=True)
    return {"peak_gb": round(mx.get_peak_memory() / 1e9, 2), "rows": rows}


def main() -> int:
    from art_stories import STORIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", nargs="*", default=[])
    ap.add_argument("--vlms", nargs="*", default=[])
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--elements", nargs="*", default=["condor", "mountain", "llama", "girl", "fox", "moon"])
    ap.add_argument("--stories", nargs="*", default=[])
    ap.add_argument("--out", default="/tmp/art_llm_bench.json")
    a = ap.parse_args()
    from yq.macworker.modelmgr import models
    stories = [s for s in STORIES if not a.stories or s["id"] in a.stories]
    res = {"llm": {}, "vlm": {}}
    for k in a.llms:
        res["llm"][k] = bench_llm(k, stories)
        models.unload_all()
    for k in a.vlms:
        res["vlm"][k] = bench_vlm(k, a.images, a.elements)
        models.unload_all()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
