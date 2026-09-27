"""UV fluorescence (365 nm LED): zones that glow differently from the rest of the object.

fluorescence = (uv - dark) / exposure, divided by the visible reflectance (COB light, same camera)
so that shading and dark paint do not look like "less fluorescence"; then clustered in CIELAB over
superpixels. Zones whose colour/brightness differs clearly from the main surface and that are big
enough are reported with cautious wording (only a specialist can confirm what they are).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .masks import linear, object_mask

MAX_WIDTH = 1600
MIN_ZONE_FRACTION = 0.004        # zones smaller than 0.4 % of the visible object are ignored
MIN_DELTA_E = 12.0               # CIELAB distance to the main surface
MIN_RATIO = 1.6                  # or brightness ratio (brighter or darker) to the main surface
SHADING_AREA_FRACTION = 0.2      # same-hue regions larger than this are lighting, not a different zone


def _load(scan, key, size=None):
    import cv2
    from .scanio import load_image
    p = scan.uv.get(key)
    if p is None:
        return None
    im = load_image(p)
    if im is None:
        return None
    if size is not None and (im.shape[1], im.shape[0]) != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return im


def fluorescence_image(uv: np.ndarray, visible: np.ndarray, dark: Optional[np.ndarray], mask: np.ndarray,
                       exposure: dict) -> tuple:
    """-> (F (H,W,3) linear fluorescence normalised by reflectance, white-balance gains (3,))."""
    e_uv = float(exposure.get("uv_exposure_us") or 1.0)
    e_vis = float(exposure.get("visible_exposure_us") or 1.0)
    e_dark = float(exposure.get("dark_exposure_us") or e_uv)
    U = linear(uv)
    if dark is not None:
        U = np.clip(U - linear(dark) * (e_uv / e_dark), 0, None)
    U = U / e_uv
    Vv = linear(visible) / e_vis
    # grey-world white balance of the visible image on the object (the COB is ~neutral, the camera WB is locked)
    med = np.array([np.median(Vv[..., c][mask]) for c in range(3)]) + 1e-9
    gains = med.mean() / med
    Y = np.einsum("hwc,c->hw", Vv * gains, [0.2126, 0.7152, 0.0722])
    k = 0.05 * float(np.median(Y[mask])) + 1e-9
    F = (U * gains) / (Y + k)[..., None]
    return F, gains


def _lab(rgb_lin: np.ndarray) -> np.ndarray:
    import cv2
    x = np.clip(rgb_lin, 0, 1).astype(np.float32)
    return cv2.cvtColor(x, cv2.COLOR_RGB2Lab)


def segment_zones(F: np.ndarray, mask: np.ndarray, n_segments: int = 400) -> dict:
    """Superpixels (SLIC) + k-means in CIELAB. Returns labels (H,W; -1 outside), the main cluster and zone stats."""
    import cv2
    from skimage.segmentation import slic
    scale = float(np.percentile(np.linalg.norm(F[mask], axis=1), 99)) + 1e-9
    Fn = np.clip(F / scale, 0, 1)
    lab = _lab(Fn)
    sp = slic(Fn, n_segments=n_segments, compactness=8.0, mask=mask, start_label=0, channel_axis=-1)
    ids = np.unique(sp[mask])
    feats = np.array([lab[sp == i].mean(axis=0) for i in ids], np.float32)
    area = np.array([(sp == i).sum() for i in ids], np.float64)
    cv2.setRNGSeed(0)                               # same scan -> same zones, every run
    best = None
    for k in (2, 3, 4):
        if len(ids) < k * 2:
            break
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _c, lab_k, cent = cv2.kmeans(feats, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        lab_k = lab_k.ravel()
        inertia = float(sum(area[lab_k == j] @ np.sum((feats[lab_k == j] - cent[j]) ** 2, axis=1) for j in range(k)))
        score = inertia * (1.0 + 0.35 * k)          # prefer few clusters unless they explain much more
        if best is None or score < best[0]:
            best = (score, k, lab_k, cent)
    labels = np.full(mask.shape, -1, np.int32)
    if best is None:
        labels[mask] = 0
        return {"labels": labels, "main": 0, "zones": [], "lab": lab, "scale": scale}
    _s, k, lab_k, cent = best
    for i, j in zip(ids, lab_k):
        labels[sp == i] = j
    tot = float(mask.sum())
    areas = [float((labels == j).sum()) for j in range(k)]
    main = int(np.argmax(areas))
    L_main = float(np.median(np.linalg.norm(F[labels == main], axis=1)))

    def _stats(j):
        dE = float(np.linalg.norm(cent[j] - cent[main]))
        Lj = float(np.median(np.linalg.norm(F[labels == j], axis=1)))
        return dE, (Lj + 1e-9) / (L_main + 1e-9)

    # Clusters that are the same surface as the main one: same hue (small dE) and either a small
    # brightness change or a LARGE area. A same-hue region covering a big part of the object with a
    # brightness change is the object's shaded/lit side (illumination falloff the visible-light
    # normalization did not fully remove), not a different material.
    surface = {main}
    for j in range(k):
        if j == main:
            continue
        dE, ratio = _stats(j)
        same_hue = dE < MIN_DELTA_E
        if same_hue and (1 / MIN_RATIO < ratio < MIN_RATIO or areas[j] / tot > SHADING_AREA_FRACTION):
            surface.add(j)
    # Everything else is anomalous. One physical zone often spans two clusters (its fluorescence
    # varies with the surface angle), so zones are the connected components of the UNION of the
    # anomalous clusters, after a small closing that joins pieces split by superpixel boundaries.
    anomalous = mask & ~np.isin(labels, list(surface))
    anomalous = cv2.morphologyEx(anomalous.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    anomalous &= mask
    zones = []
    n, cc, st, _ = cv2.connectedComponentsWithStats(anomalous.astype(np.uint8), 8)
    for c in range(1, n):
        a = float(st[c, cv2.CC_STAT_AREA])
        if a / tot < MIN_ZONE_FRACTION:
            continue
        zm = cc == c
        inside = [j for j in range(k) if j not in surface and (labels[zm] == j).any()]
        if not inside:
            continue
        dom = max(inside, key=lambda j: int((labels[zm] == j).sum()))
        dE = max(_stats(j)[0] for j in inside)
        Lz = float(np.median(np.linalg.norm(F[zm], axis=1)))
        x, y, w, h = (int(st[c, q]) for q in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        zones.append({"cluster": dom, "clusters": inside, "component": c, "area_px": int(a), "area_fraction": a / tot,
                      "delta_e": dE, "brightness_ratio": (Lz + 1e-9) / (L_main + 1e-9), "bbox": [x, y, w, h],
                      "_mask": zm})
    zones.sort(key=lambda z: -z["area_px"])
    return {"labels": labels, "main": main, "surface": sorted(surface), "zones": zones, "lab": lab, "scale": scale}


def _zone_text(z: dict) -> tuple:
    r = z["brightness_ratio"]
    kind = "más brillante" if r >= 1 else "más oscura"
    title = "Zona con fluorescencia distinta"
    detail = ("Zona %s bajo luz UV (%.1f veces, diferencia de color ΔE %.0f; %.1f %% de la superficie visible). "
              "Puede ser una restauración, un adhesivo, un consolidante o una pintura moderna"
              % (kind, r if r >= 1 else 1 / r, z["delta_e"], 100 * z["area_fraction"]))
    if r < 1:
        detail += ", o una zona con suciedad o pátina diferente"
    detail += ". Esto es solo una pista: confirmarlo con un especialista en conservación."
    return title, detail


def overlay(visible: np.ndarray, seg: dict, mask: np.ndarray) -> np.ndarray:
    """Visible photo in grey with each distinct zone outlined in magenta and numbered."""
    import cv2
    g = cv2.cvtColor(visible, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    img = (img * 0.8).astype(np.uint8)
    for k, z in enumerate(seg["zones"]):
        m = z["_mask"].astype(np.uint8)
        tint = np.zeros_like(img)
        tint[m > 0] = (255, 0, 200)
        img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 0, 220), 2)
        x, y, w, h = z["bbox"]
        cv2.putText(img, str(k + 1), (x + w // 2 - 6, y + h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def analyze_uv(scan, cal, out_dir: Path, mesh_verts=None, mesh_faces=None, prefix: str = "uv_") -> dict:
    import cv2
    out_dir = Path(out_dir)
    exposure = scan.uv.get("exposure") or {}
    vis = _load(scan, "visible")
    h0, w0 = vis.shape[:2]
    s = min(1.0, MAX_WIDTH / float(w0))
    size = (int(round(w0 * s)), int(round(h0 * s)))
    vis = cv2.resize(vis, size, interpolation=cv2.INTER_AREA) if s < 1 else vis
    uvi = _load(scan, "uv", size)
    dark = _load(scan, "dark", size)
    cam_name = str(exposure.get("camera") or "A").upper()
    deg = float(exposure.get("platter_deg", 0.0))
    cam = cal.view(cam_name, deg, size[0], size[1])
    if mesh_verts is not None:
        from .render import depth_map
        mask = np.isfinite(depth_map(cam, mesh_verts, mesh_faces))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    else:
        mask, _ = object_mask(vis, None, cam)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if mask.sum() < 500:
        raise ValueError("no se encontró el objeto en las fotos UV")
    F, gains = fluorescence_image(uvi, vis, dark, mask, exposure)
    seg = segment_zones(F, mask)
    fl = np.clip(F / seg["scale"], 0, 1) * mask[..., None]
    from .rti_maps import to_srgb8
    arts = {"uv_fluorescence": prefix + "fluorescence.png", "uv_overlay": prefix + "overlay.png"}
    cv2.imwrite(str(out_dir / arts["uv_fluorescence"]), to_srgb8(fl)[..., ::-1])
    cv2.imwrite(str(out_dir / arts["uv_overlay"]), overlay(vis, seg, mask)[..., ::-1])
    findings = []
    for k, z in enumerate(seg["zones"]):
        title, detail = _zone_text(z)
        findings.append({"analysis": "uv", "title_es": "%s (%d)" % (title, k + 1), "detail_es": detail,
                         "severity": "notable", "image": arts["uv_overlay"], "bbox_px": z["bbox"]})
    if not findings:
        findings.append({"analysis": "uv", "title_es": "Fluorescencia UV uniforme",
                         "detail_es": "Bajo luz UV la superficie brilla de forma pareja: no se ven zonas con "
                                      "fluorescencia distinta (restauraciones o adhesivos modernos suelen verse así). "
                                      "Ojo: algunas restauraciones no fluorescen.",
                         "severity": "info", "image": arts["uv_overlay"]})
    info = {"camera": cam_name, "platter_deg": deg, "zones": len(seg["zones"]), "white_balance": gains.tolist(),
            "_seg": seg, "_mask": mask}
    return {"artifacts": arts, "findings": findings, "info": info}
