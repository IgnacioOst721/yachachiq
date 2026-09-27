"""Retrieval + VLM on a small held-out sample (ART's VLM must be installed; run inside the heavy lock)."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable


def vlm_eval(index, ids: list, groups: dict, log: Callable = print) -> dict:
    from PIL import Image
    from .catalog import public_reference
    from .identify import ask_vlm, build_identification, decide, vlm_prompt
    stats = defaultdict(lambda: [0, 0])
    t0 = time.time()
    answered = 0
    for k, i in enumerate(ids):
        it = index.items[i]
        excl = groups[(it.get("museum"), (it.get("title") or "").strip().lower())] | {i}
        d = decide(index.vector(i), index, None, exclude=excl, use_llm_context=False)
        refs = [public_reference(index.items[j], s) for j, s in d["neighbors"][:8]]
        img = Image.open(index.root / it["thumb"]).convert("RGB")
        cand = dict(d["votes"])
        cand["culture"] = sorted(d["scores"].items(), key=lambda x: -x[1])
        vl = ask_vlm([img], vlm_prompt([], refs, cand, None))
        answered += int(vl is not None)
        ident = build_identification(d, index, vl, [], None)
        for name, truth, pred in (("culture/retrieval", it["culture_norm"], d["best"]),
                                  ("culture/retrieval+vlm", it["culture_norm"], ident["culture"]),
                                  ("material_cls/retrieval+vlm", it.get("material_cls"), ident.get("material_class"))):
            if truth:
                stats[name][0] += 1
                stats[name][1] += int(truth == pred)
        log("vlm eval %d/%d (%.1f s/object)" % (k + 1, len(ids), (time.time() - t0) / (k + 1)))
    return {"n": len(ids), "vlm_answered": answered,
            "top1": {n: round(s[1] / max(1, s[0]), 4) for n, s in stats.items()}}
