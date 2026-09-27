"""Identification: type, material, culture, period from photos + measurements, fully offline.

1. Query images: the best views of the object, cropped and put on a light museum-like background.
2. Retrieval: SigLIP 2 embeddings vs the offline museum catalog (catalog.py) -> k nearest objects,
   similarity-weighted votes for culture, material, object type and region (image only).
3. Visitor context (context.py): bounded soft prior on the culture votes (never creates a culture).
4. VLM (ART's yq.macworker.models.vlm.ask, optional): photos + measured facts + the retrieved
   references' metadata -> strict JSON, validated against the controlled vocabularies.
5. Confidence: retrieval vote share calibrated on held-out catalog objects (calibration.json),
   adjusted by agreement between retrieval and VLM. `similar` only ever lists real catalog entries.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

import numpy as np

from . import settings

log = logging.getLogger("yq.box.identify")
from .taxonomy import (CULTURE_BY_NAME, MATERIAL_BY_NAME, MATERIAL_CLASSES_ES, OBJECT_TYPE_BY_NAME, REGIONS_ES,
                       TYPE_CLASSES_ES, format_period)

TAU = 0.03                        # softmax temperature on cosine similarity for the votes
BG = (228, 228, 226)              # light grey background, like most museum photos


def prepare_query(img: np.ndarray, mask: Optional[np.ndarray], out_px: int = 448, margin: float = 0.12) -> np.ndarray:
    """Crop the object, replace the black box interior by a light background, pad to a square."""
    import cv2
    if mask is None or mask.sum() < 50:
        from .masks import object_mask
        mask, _ = object_mask(img)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * margin))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
    comp = (img.astype(np.float32) * soft + np.array(BG, np.float32) * (1 - soft)).astype(np.uint8)
    canvas = np.full((side, side, 3), BG, np.uint8)
    sx0, sy0 = cx - side // 2, cy - side // 2
    ix0, iy0 = max(0, sx0), max(0, sy0)
    ix1, iy1 = min(img.shape[1], sx0 + side), min(img.shape[0], sy0 + side)
    canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = comp[iy0:iy1, ix0:ix1]
    return cv2.resize(canvas, (out_px, out_px), interpolation=cv2.INTER_AREA)


def _weights(sims: np.ndarray) -> np.ndarray:
    return np.exp((sims - sims.max()) / TAU)


def vote(neighbors: list, items: dict) -> dict:
    """neighbors: [(id, sim)]. Returns {attr: [(value, share), ...]} for culture/material/type/region + period."""
    sims = np.array([s for _i, s in neighbors])
    w = _weights(sims)
    fields = {"culture": "culture_norm", "material": "material_norm", "material_cls": "material_cls",
              "type": "type_norm", "type_cls": "type_cls", "region": "region_norm"}
    out = {}
    for name, key in fields.items():
        acc = defaultdict(float)
        tot = 0.0
        for (i, _s), wi in zip(neighbors, w):
            v = items[i].get(key)
            if v:
                acc[v] += wi
                tot += wi
        out[name] = sorted(((v, a / tot) for v, a in acc.items()), key=lambda x: -x[1]) if tot > 0 else []
        out[name + "_coverage"] = float(tot / w.sum()) if w.sum() > 0 else 0.0
    out["top_sim"] = float(sims.max()) if len(sims) else 0.0
    return out


def period_for(culture: Optional[str], neighbors: list, items: dict) -> tuple:
    """Weighted 20-80 % range of the neighbours' dates of that culture (fallback: curated range)."""
    sims = np.array([s for _i, s in neighbors])
    w = _weights(sims)
    b, e, ww = [], [], []
    for (i, _s), wi in zip(neighbors, w):
        it = items[i]
        if culture and it.get("culture_norm") != culture:
            continue
        if isinstance(it.get("date_begin"), (int, float)) and isinstance(it.get("date_end"), (int, float)):
            b.append(it["date_begin"])
            e.append(it["date_end"])
            ww.append(wi)
    if len(b) >= 2:
        def wq(x, q):
            o = np.argsort(x)
            c = np.cumsum(np.array(ww)[o]) / sum(ww)
            return int(np.array(x)[o][min(np.searchsorted(c, q), len(x) - 1)])
        lo, hi = wq(b, 0.2), wq(e, 0.8)
        if lo <= hi:
            return format_period(lo, hi), (lo, hi)
    c = CULTURE_BY_NAME.get(culture or "")
    return (c["dates"] if c else ""), None


def _name_es(kind: str, value: Optional[str]) -> str:
    if not value:
        return "desconocido"
    if kind == "culture":
        return CULTURE_BY_NAME.get(value, {}).get("name_es", value)
    if kind == "material":
        return MATERIAL_BY_NAME.get(value, {}).get("name_es", value)
    if kind == "material_cls":
        return MATERIAL_CLASSES_ES.get(value, value)
    if kind == "type":
        return OBJECT_TYPE_BY_NAME.get(value, {}).get("name_es", value)
    if kind == "type_cls":
        return TYPE_CLASSES_ES.get(value, value)
    if kind == "region":
        return REGIONS_ES.get(value, value)
    return value


VLM_FIELDS = {"object_type": str, "material": str, "culture": str, "period": str, "region": str, "confidence": (int, float),
              "evidence_es": list, "description_es": str}


def vlm_prompt(facts: list, refs: list, cand: dict, context_hint: Optional[dict]) -> str:
    lines = ["You are an archaeologist. Identify the object in the photos (taken inside a black analysis box, "
             "several sides). Measured facts (trust them):"]
    lines += ["- " + f for f in facts]
    lines.append("Most similar objects in an offline museum catalog (real records, similarity 0-1):")
    for r in refs:
        lines.append("- %.2f | %s | culture: %s | date: %s | medium: %s | %s" % (
            r["score"], r["title"][:70], r.get("culture_raw") or r["culture"], r["date"][:40], r["medium"][:50], r["museum"]))
    lines.append("Retrieval vote (culture: share): " + ", ".join("%s %.2f" % cv for cv in cand.get("culture", [])[:5]))
    lines.append("Retrieval vote (material): " + ", ".join("%s %.2f" % cv for cv in cand.get("material", [])[:4]))
    if context_hint and context_hint.get("given"):
        lines.append("Lugar reportado por el visitante; puede ser incorrecto: %r (zona: %s, país: %s). Use it only to "
                     "decide between candidates that the photos support." % (context_hint.get("text", ""),
                                                                         context_hint.get("zone"), context_hint.get("country")))
    lines.append('Reply with ONE JSON object: {"object_type": English noun phrase, "material": English, "culture": '
                 'English culture name, "period": e.g. "100-800 CE", "region": English, "confidence": 0-1, '
                 '"evidence_es": [3-5 short reasons in simple Spanish], "description_es": 2-3 sentences in simple '
                 'Spanish for a museum visitor}. If unsure, say so and lower the confidence. Never invent references.')
    return "\n".join(lines)


def ask_vlm(images: list, prompt: str) -> Optional[dict]:
    """ART's VLM (contract §3.2). None when it is not available or does not answer valid JSON."""
    if not settings.USE_VLM:
        return None
    try:
        from yq.macworker.models import vlm
    except Exception as e:
        log.warning("VLM not importable here (%s): identification uses the catalog only", e)
        return None
    import json
    for attempt in range(2):
        try:
            text = vlm.ask(images, prompt + ("" if attempt == 0 else "\nONLY the JSON object, nothing else."),
                           max_tokens=settings.VLM_MAX_TOKENS)
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception as e:                 # log it: a silent failure hid a bug on 2026-09-27
            log.warning("VLM identification attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)
            continue
        if isinstance(obj, dict) and all(isinstance(obj.get(k), t) for k, t in VLM_FIELDS.items() if k in ("culture", "material")):
            return obj
    return None


def calibrate(p: float, table: Optional[dict]) -> float:
    """Map a raw vote share to the accuracy measured on held-out catalog objects (piecewise linear)."""
    if not table or not table.get("x"):
        return float(p) * 0.85
    return float(np.interp(p, table["x"], table["y"]))


def load_calibration(model_key: str, root=None) -> Optional[dict]:
    import json
    from .catalog import CatalogIndex
    p = CatalogIndex.index_dir(model_key, root) / "calibration.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def decide(q_embs: np.ndarray, index, context: Optional[dict] = None, exclude: Optional[set] = None,
           k: Optional[int] = None, use_llm_context: bool = True) -> dict:
    """Retrieval + context prior (no VLM). Returns the pieces identify() and the evaluation need."""
    from .context import apply_prior, compatibility, parse_context
    from .taxonomy import CULTURE_BY_NAME as CB
    nb = index.search(q_embs, k=k or settings.KNN, exclude=exclude)
    v = vote(nb, index.items)
    img_scores = dict(v["culture"])
    hint = parse_context(context, use_llm=use_llm_context) if context else {"given": False, "confidence": 0.0}
    regions = {c: CB.get(c, {}).get("region") for c in img_scores}
    ctx_scores, mult = apply_prior(img_scores, regions, hint, settings.CONTEXT_MAX_BOOST) if hint.get("confidence") \
        else (img_scores, {c: 1.0 for c in img_scores})
    best_img = max(img_scores, key=img_scores.get) if img_scores else None
    best_ctx = max(ctx_scores, key=ctx_scores.get) if ctx_scores else None
    k_img = compatibility(best_img, regions.get(best_img), hint) if best_img and hint.get("confidence") else None
    return {"neighbors": nb, "votes": v, "image_scores": img_scores, "scores": ctx_scores, "multipliers": mult,
            "hint": hint, "best_image": best_img, "best": best_ctx, "k_image": k_img}


def _canon_culture(name: str, cand: list) -> Optional[str]:
    from .taxonomy import match_culture
    c = match_culture(name or "")
    return c["name"] if c else None


def build_identification(d: dict, index, vlm: Optional[dict], facts: list, table: Optional[dict],
                         similar_dir=None, engine: str = "") -> dict:
    from .catalog import public_reference
    from .context import effect_text
    from .taxonomy import match_material, match_type
    v, items = d["votes"], index.items
    scores = sorted(d["scores"].items(), key=lambda x: -x[1])
    culture, p = (scores[0] if scores else (None, 0.0))
    agree = None
    if vlm:
        vc = _canon_culture(vlm.get("culture", ""), scores)
        support = d["scores"].get(vc, 0.0) if vc else 0.0
        agree = vc == culture
        if vc and not agree and support >= 0.5 * p:
            culture, p = vc, support            # the VLM breaks a near tie among supported candidates
    conf = calibrate(p, table)
    if agree is True:
        conf = min(0.97, conf + 0.1 * (1 - conf) + 0.05)
    elif agree is False:
        conf *= 0.75
    mat = (v["material"][0][0] if v["material"] else None)
    mcls = (v["material_cls"][0][0] if v["material_cls"] else None)
    typ = (v["type"][0][0] if v["type"] else None)
    if vlm:
        mv, tv = match_material(vlm.get("material", "")), match_type(vlm.get("object_type", ""))
        if mv and (mv["cls"] == mcls or not mat):
            mat, mcls = mv["name"], mv["cls"]
        if tv and (not typ or tv["cls"] == OBJECT_TYPE_BY_NAME.get(typ, {}).get("cls")):
            typ = tv["name"]
    period, _rng = period_for(culture, d["neighbors"], items)
    region = CULTURE_BY_NAME.get(culture or "", {}).get("region") or (v["region"][0][0] if v["region"] else "")
    alts = [{"culture": c, "culture_es": _name_es("culture", c), "period": period_for(c, d["neighbors"], items)[0],
             "confidence": round(calibrate(s, table), 3),
             "why": "%.0f %% de los objetos más parecidos del catálogo" % (100 * d["image_scores"].get(c, 0))}
            for c, s in scores[1:4] if s >= 0.05]
    evidence = ["Se parece a objetos del catálogo de museos atribuidos a %s (%.0f %% del voto)."
                % (_name_es("culture", culture), 100 * p)] if culture else []
    evidence += list((vlm or {}).get("evidence_es") or [])[:5]
    evidence += [f for f in facts if f.startswith("densidad") or f.startswith("UV") or f.startswith("térmica")][:2]
    sim = []
    seen = set()
    for i, s in d["neighbors"]:
        rec = items[i]
        key = (rec.get("museum"), (rec.get("title") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        img = ""
        if similar_dir is not None:
            import shutil
            from pathlib import Path
            dst = Path(similar_dir) / (i.replace(":", "_") + ".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(index.root / rec["thumb"], dst)
                img = "%s/%s" % (dst.parent.name, dst.name)
            except (OSError, AttributeError, KeyError):
                img = ""
        sim.append(public_reference(rec, s, img))
        if len(sim) >= settings.SHOW_SIMILAR:
            break
    img_only_c = d["best_image"]
    image_only = {"culture": img_only_c, "period": period_for(img_only_c, d["neighbors"], items)[0],
                  "material": mat, "confidence": round(calibrate(d["image_scores"].get(img_only_c, 0.0), table), 3)}
    names = {c: _name_es("culture", c) for c in d["image_scores"]}
    effect = effect_text(d["hint"], img_only_c, culture, names, d["k_image"])
    desc = (vlm or {}).get("description_es") or (
        "Por su forma y decoración se parece a objetos %s de %s, probablemente de %s. Es una estimación automática: "
        "un especialista puede confirmarla." % (_name_es("culture", culture), period or "época desconocida",
                                                  _name_es("material", mat)) if culture else
        "No se encontró una cultura clara: el objeto no se parece lo suficiente a los del catálogo.")
    return {"object_type": typ or "", "object_type_es": _name_es("type", typ), "material": mat or "",
            "material_es": _name_es("material", mat), "culture": culture or "", "period": period, "region": region or "",
            "confidence": round(float(conf), 3), "alternatives": alts, "evidence": evidence, "similar": sim,
            "description_es": desc, "engine": engine, "context_effect_es": effect, "image_only": image_only,
            "material_class": mcls or "", "retrieval": {"top_similarity": round(v["top_sim"], 4), "knn": len(d["neighbors"]),
                                                        "culture_votes": [[c, round(s, 3)] for c, s in scores[:6]]}}


def facts_from_measurements(measurements: list, findings: Optional[list] = None, notes: str = "") -> list:
    """Short English/Spanish fact lines for the VLM and the evidence list."""
    m = {x["name"]: x for x in measurements or [] if isinstance(x, dict)}
    out = []
    if "height" in m:
        out.append("size: height %.0f mm, width %.0f mm, depth %.0f mm" % (
            m["height"]["value"], m.get("width", {}).get("value", 0), m.get("depth", {}).get("value", 0)))
    if "mass" in m:
        out.append("mass: %.0f g" % m["mass"]["value"])
    if "density_apparent" in m:
        d = m["density_apparent"]
        out.append("densidad aparente %.2f ± %.2f g/cm3 (masa / volumen exterior; baja si es hueco)" % (d["value"], d["uncertainty"]))
    for f in findings or []:
        if f.get("analysis") == "uv" and f.get("severity") == "notable":
            out.append("UV: " + f.get("title_es", ""))
        if f.get("analysis") == "thermal" and f.get("severity") == "notable":
            out.append("térmica: " + f.get("title_es", ""))
    if notes:
        out.append("notes: " + str(notes)[:200])
    return out


def mock_identification(reason: str) -> dict:
    return {"object_type": "", "object_type_es": "desconocido", "material": "", "material_es": "desconocido", "culture": "",
            "period": "", "region": "", "confidence": 0.0, "alternatives": [], "evidence": [], "similar": [],
            "description_es": reason, "engine": "unavailable", "context_effect_es": "", "image_only": None}


def identify(images: list, measurements: Optional[list] = None, findings: Optional[list] = None, notes: str = "",
             context: Optional[dict] = None, out_dir=None, masks: Optional[list] = None, model_key: Optional[str] = None,
             catalog_root=None, use_vlm: Optional[bool] = None) -> dict:
    """images: RGB uint8 arrays or paths (object photos). Returns an Identification as dict."""
    from yq.common import config
    from .catalog import CatalogIndex
    from .scanio import load_image
    key = model_key or settings.EMBED_MODEL
    if config.mock("box_analysis"):
        r = mock_identification("Resultado simulado (modo de prueba YQ_MOCK): no se usó ningún modelo.")
        r["engine"] = "mock"
        return r
    if not CatalogIndex.exists(key, catalog_root) and not model_key:
        built = CatalogIndex.available(catalog_root)
        if built:                          # query with the SAME model the catalog was embedded with
            log.info("catalog index for %s missing; using %s", key, built[0])
            key = built[0]
    if not CatalogIndex.exists(key, catalog_root):
        return mock_identification("No se pudo identificar: falta el catálogo de museos en esta computadora "
                                   "(tools/box_analysis_build_catalog.py).")
    index = CatalogIndex.load(key, catalog_root)
    imgs = [load_image(im) if not isinstance(im, np.ndarray) else im for im in images]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return mock_identification("No hay fotos para identificar el objeto.")
    queries = [prepare_query(im, masks[k] if masks and k < len(masks) else None) for k, im in enumerate(imgs)]
    from .embed import get_embedder
    try:
        embedder = get_embedder(key)
    except ImportError:
        return mock_identification("La identificación se hace en la Mac (aquí no está instalado el modelo).")
    q = embedder.embed(queries)
    d = decide(q, index, context)
    facts = facts_from_measurements(measurements or [], findings, notes)
    vl = None
    if use_vlm if use_vlm is not None else settings.USE_VLM:
        refs = [__import__("yq.box.analysis.catalog", fromlist=["public_reference"]).public_reference(index.items[i], s)
                for i, s in d["neighbors"][:8]]
        vl = ask_vlm(queries[:4], vlm_prompt(facts, refs, d["votes"] | {"culture": sorted(d["scores"].items(), key=lambda x: -x[1])},
                                             d["hint"]))
    table = load_calibration(key, catalog_root)
    from pathlib import Path
    sim_dir = Path(out_dir) / "similar" if out_dir else None
    engine = "%s+knn%s%s" % (key, "+vlm" if vl else "", "+context" if d["hint"].get("confidence") else "")
    ident = build_identification(d, index, vl, facts, table, sim_dir, engine)
    if out_dir:
        import cv2
        for k, qi in enumerate(queries[:4]):
            cv2.imwrite(str(Path(out_dir) / ("identify_query%d.jpg" % k)), qi[..., ::-1])
    return ident
