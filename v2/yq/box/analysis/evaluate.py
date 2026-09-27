"""Leave-one-object-out evaluation of the identification on held-out catalog objects.

For each sampled catalog object, its own embedding is the query and the object itself plus every
record of the same museum with the same title (duplicates, sets, fragments) is excluded. Measures
top-1/top-3 accuracy for culture, fine material, material class and object type, with and without
the visitor context (correct context derived from the object's own metadata, and a wrong one),
and fits the confidence calibration table (vote share -> measured accuracy).
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from . import settings
from .context import CULTURE_ZONES, ZONES

REGION_TEXT = {"Mesoamerica": "Mexico", "West Mexico": "Mexico", "Northern Andes": "Ecuador", "Isthmo-Colombian": "Costa Rica",
               "Egypt": "Egipto", "Mediterranean": "Grecia", "Near East": "Iraq", "China": "China", "Japan": "Japon",
               "South Asia": "India", "West Africa": "Nigeria", "Central Africa": "Congo", "Europe": "Francia",
               "Islamic": "Iran"}


def correct_context(item: dict) -> Optional[dict]:
    c, r = item.get("culture_norm"), item.get("region_norm")
    if c in CULTURE_ZONES and CULTURE_ZONES[c]:
        return {"region_hint": CULTURE_ZONES[c][0]}
    if r in REGION_TEXT:
        return {"found_where": REGION_TEXT[r], "region_hint": "otro_pais"}
    return None


def wrong_context(item: dict, rng: random.Random) -> Optional[dict]:
    c, r = item.get("culture_norm"), item.get("region_norm")
    if c in CULTURE_ZONES and CULTURE_ZONES[c]:
        bad = [z for z in ZONES if z not in CULTURE_ZONES[c]]
        opts = [{"region_hint": z} for z in bad] + [{"found_where": t, "region_hint": "otro_pais"} for t in ("Mexico", "Egipto", "China")]
        return rng.choice(opts)
    opts = [{"found_where": t, "region_hint": "otro_pais"} for reg, t in REGION_TEXT.items() if reg != r]
    opts += [{"region_hint": z} for z in ("costa_norte", "costa_sur", "sierra_sur")]
    return rng.choice(opts)


def _group_excl(index) -> dict:
    groups = defaultdict(set)
    for i, it in index.items.items():
        groups[(it.get("museum"), (it.get("title") or "").strip().lower())].add(i)
    return groups


def leave_one_out(model_key: Optional[str] = None, root=None, n: int = 2000, seed: int = 0, log: Callable = print,
                  vlm_n: int = 0, save_calibration: bool = True) -> dict:
    from .catalog import CatalogIndex
    from .identify import decide
    key = model_key or settings.EMBED_MODEL
    index = CatalogIndex.load(key, root, mmap=False)
    rng = random.Random(seed)
    pool = [i for i in index.ids if index.items[i].get("culture_norm")]
    sample = rng.sample(pool, min(n, len(pool)))
    groups = _group_excl(index)
    stats = defaultdict(lambda: [0, 0, 0])           # name -> [n, top1, top3]
    calib = []
    t0 = time.time()
    for k, i in enumerate(sample):
        it = index.items[i]
        excl = groups[(it.get("museum"), (it.get("title") or "").strip().lower())] | {i}
        q = index.vector(i)
        runs = {"none": None, "correct": correct_context(it), "wrong": wrong_context(it, rng)}
        for label, ctx in runs.items():
            if label != "none" and ctx is None:
                continue
            d = decide(q, index, ctx, exclude=excl, use_llm_context=False)
            ranked = [c for c, _s in sorted(d["scores"].items(), key=lambda x: -x[1])]
            _acc(stats, "culture/" + label, it["culture_norm"], ranked)
            if label == "none" and runs["correct"] is not None:
                _acc(stats, "culture/none_same_subset_as_correct", it["culture_norm"], ranked)
            if label == "none":
                calib.append((d["scores"].get(ranked[0], 0.0) if ranked else 0.0, bool(ranked and ranked[0] == it["culture_norm"])))
                v = d["votes"]
                for attr, field in (("material", "material_norm"), ("material_cls", "material_cls"), ("type", "type_norm"),
                                    ("type_cls", "type_cls"), ("region", "region_norm")):
                    if it.get(field):
                        _acc(stats, attr, it[field], [x for x, _s in v[attr]])
                if it.get("region_norm") == "Andes":
                    _acc(stats, "culture/andes_only", it["culture_norm"], ranked)
        if (k + 1) % 250 == 0:
            log("eval %d/%d (%.1f/s)" % (k + 1, len(sample), (k + 1) / (time.time() - t0)))
    summary = {name: {"n": s[0], "top1": round(s[1] / max(1, s[0]), 4), "top3": round(s[2] / max(1, s[0]), 4)}
               for name, s in sorted(stats.items())}
    table = _calibration_table(calib)
    if save_calibration and table:
        import json
        (CatalogIndex.index_dir(key, root) / "calibration.json").write_text(json.dumps(table, indent=1))
    res = {"model": key, "catalog_size": len(index), "sampled": len(sample), "seed": seed, "summary": summary,
           "calibration": table, "protocol": "leave-one-object-out; excluded the object and same-museum same-title records"}
    if vlm_n:
        from .evaluate_vlm import vlm_eval
        res["vlm"] = vlm_eval(index, sample[:vlm_n], groups, log)
    return res


def _acc(stats, name, truth, ranked) -> None:
    s = stats[name]
    s[0] += 1
    s[1] += int(bool(ranked) and ranked[0] == truth)
    s[2] += int(truth in ranked[:3])


def _calibration_table(pairs: list, bins: int = 10) -> Optional[dict]:
    if len(pairs) < 50:
        return None
    p = np.array([a for a, _b in pairs])
    y = np.array([b for _a, b in pairs], float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p <= hi)
        if sel.sum() >= 10:
            xs.append(float(p[sel].mean()))
            ys.append(float(y[sel].mean()))
    ys = list(np.maximum.accumulate(ys))            # monotone (isotonic-like)
    return {"x": [0.0] + xs + [1.0], "y": [0.0] + ys + [max(ys[-1], min(0.97, ys[-1] + 0.05))], "n": len(pairs)}
