"""Mass, silhouettes, visual hull and the measurements derived from them (with uncertainties)."""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from yq.common.contracts import Measurement

from .calib import BoxCalibration
from .carve import Hull, carve, hull_mesh, mesh_area, mesh_volume
from .masks import object_mask
from .scanio import Scan, load_image

MASK_MAX_WIDTH = 2400          # silhouettes are computed on images at most this wide (speed; sub-pixel SDF keeps accuracy)
REFINE_MAX_WIDTH = 1200        # photo-consistency refinement works on gray images at most this wide
PRINT_SCALE_REL = 0.002        # relative scale uncertainty of the printed calibration board (after checking with calipers)
NOMINAL_POSE_MM = 3.0          # 1-sigma error of the CAD nominal geometry when the box is not calibrated

HOLLOW_NOTE_ES = ("Densidad aparente = masa / volumen de la envolvente exterior. Si el objeto es hueco (una vasija, "
                  "una botella, una figura hueca) la densidad aparente sale MUCHO menor que la del material, porque el "
                  "volumen incluye el aire del interior. Solo en piezas macizas se parece a la densidad del material.")


def mass_measurement(weight: Optional[dict], warnings: list) -> Optional[Measurement]:
    if not weight or weight.get("grams") is None:
        warnings.append("No hay peso (weight.json): no se puede calcular la densidad.")
        return None
    g = float(weight["grams"])
    sigma = float(weight.get("sigma_g") or 0.0)
    samples = weight.get("samples") or []
    if len(samples) >= 3:
        sigma = max(sigma, float(np.std(samples, ddof=1) / math.sqrt(len(samples))))
    sigma = max(sigma, 0.5)                 # HX711 + 5 kg cell: never claim better than 0.5 g
    note = ""
    if not weight.get("stable", True):
        note = "La balanza no se estabilizó; el peso puede tener más error."
        warnings.append(note)
        sigma = max(sigma, 2.0)
    return Measurement("mass", round(g, 1), "g", round(sigma, 1), "celda de carga 5 kg + HX711", note)


def silhouettes(scan: Scan, cal: BoxCalibration, cameras=None, on_progress: Optional[Callable] = None) -> list:
    """[(photo, camera_at_angle, mask, info)] for every usable photogrammetry photo."""
    out = []
    bgs = {}
    photos = [p for p in scan.photos if cameras is None or p.camera in cameras]
    for k, ph in enumerate(photos):
        img = load_image(ph.path)
        if img is None:
            scan.warnings.append("No se pudo leer %s." % ph.path.name)
            continue
        h, w = img.shape[:2]
        scale = min(1.0, MASK_MAX_WIDTH / float(w))
        if scale < 1.0:
            import cv2
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        if ph.camera not in bgs:
            bp = scan.backgrounds.get(ph.camera)
            bg = load_image(bp) if bp else None
            if bg is not None and bg.shape[:2] != img.shape[:2]:
                import cv2
                bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            bgs[ph.camera] = bg
        cam = cal.view(ph.camera, ph.platter_deg, img.shape[1], img.shape[0])
        m, info = object_mask(img, bgs[ph.camera], cam)
        import cv2
        gs = min(1.0, REFINE_MAX_WIDTH / float(img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gs < 1.0:
            gray = cv2.resize(gray, (int(round(img.shape[1] * gs)), int(round(img.shape[0] * gs))), interpolation=cv2.INTER_AREA)
        info["gray"] = gray
        info["gray_cam"] = cam.scaled(gray.shape[1], gray.shape[0]) if gs < 1.0 else cam
        info["image_path"] = str(ph.path)
        if info["area_px"] < 200:
            scan.warnings.append("En %s casi no se ve el objeto; se ignora esa foto." % ph.path.name)
            continue
        if info["touches_border"]:
            scan.warnings.append("En %s el objeto toca el borde de la foto: puede estar cortado." % ph.path.name)
        out.append((ph, cam, m, info))
        if on_progress:
            on_progress((k + 1) / max(1, len(photos)))
    if not any(v is not None for v in bgs.values()):
        scan.warnings.append("No hay fotos del plato vacío (background_cam*.jpg): la silueta se separa solo por brillo; "
                             "objetos muy oscuros pueden medirse peor.")
    return out


def _footprint(verts: np.ndarray) -> tuple:
    """(width = long side, depth = short side, angle) of the minimum-area rectangle of the XY footprint."""
    import cv2
    xy = verts[:, :2].astype(np.float32)
    (_cx, _cy), (a, b), ang = cv2.minAreaRect(xy)
    return max(a, b), min(a, b), ang


def robust_height(verts: np.ndarray, cell: float = 1.0) -> float:
    """Top of the object ignoring single-vertex spikes: max of the 3x3-median-filtered height map."""
    from scipy.ndimage import median_filter
    ij = np.floor(verts[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    H = np.zeros(tuple(ij.max(axis=0) + 1), np.float32)
    np.maximum.at(H, (ij[:, 0], ij[:, 1]), verts[:, 2].astype(np.float32))
    return float(median_filter(H, size=3).max())


def _angle_step(degs: list) -> float:
    a = np.unique(np.round(np.mod(degs, 360.0), 3))
    if len(a) < 2:
        return 90.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(np.max(gaps))


def geometry(views: list, cal: BoxCalibration, voxel: Optional[float] = None, on_progress=None, refine: bool = True,
             log=None) -> dict:
    """Visual hull (+ photo-consistency carving) and measurements. views from `silhouettes`."""
    cams = [(cam, m, i.get("offset")) for _ph, cam, m, i in views]
    mm_per_px = float(np.median([np.linalg.norm(c[0].center - [0, 0, 60]) / c[0].focal for c in cams]))
    if voxel is None:
        voxel = float(np.clip(1.6 * mm_per_px, 0.5, 1.5))
    hull: Hull = carve(cams, voxel=voxel, on_progress=(lambda f: on_progress(0.6 * f)) if on_progress else None)
    verts, faces, normals = hull_mesh(hull)
    hull_verts = verts
    refine_info = {}
    if refine:
        from .refine import refine_surface
        gviews = [(i["gray_cam"], i["gray"]) for _ph, _c, _m, i in views if "gray" in i]
        g_mm = float(np.median([np.linalg.norm(c.center - [0, 0, 60]) / c.focal for c, _g in gviews]))
        verts, refine_info = refine_surface(verts, faces, normals, gviews, g_mm, log=log)
        if on_progress:
            on_progress(1.0)
    V = mesh_volume(verts, faces)
    A = mesh_area(verts, faces)
    height = robust_height(verts)
    width, depth, ang = _footprint(verts)
    reproj = max(cal.reproj_px.values()) if cal.reproj_px else 0.0
    sig_cal = reproj * mm_per_px if cal.calibrated() else NOMINAL_POSE_MM
    sig_mask = 0.5 * mm_per_px
    sig_vox = voxel / math.sqrt(12) * 0.5
    sig_s = math.sqrt(sig_cal ** 2 + sig_mask ** 2 + sig_vox ** 2)        # one surface position, 1 sigma

    # model terms measured on synthetic scans: silhouettes over-estimate flat faces between view directions
    # (worse with bigger angle steps) and flat tops that photo-consistency could not refine (textureless)
    step = _angle_step([ph.platter_deg for ph, _c, _m, _i in views])
    sig_face = 0.012 * max(width, 1.0) * (step / 15.0)
    top_cov = refine_info.get("top_coverage", 0.0) if refine else 0.0
    roof = max(0.0, float(hull_verts[:, 2].max()) - height)
    sig_top = 0.5 + (1.0 - top_cov) * roof * 0.5

    def lin(L, extra=0.0):
        return math.sqrt(2 * sig_s ** 2 + (PRINT_SCALE_REL * L) ** 2 + extra ** 2)
    sig_V = math.sqrt((A * sig_s) ** 2 + (3 * PRINT_SCALE_REL * V) ** 2 + (0.01 * V * step / 15.0) ** 2)
    method = "casco visual %s(%d vistas, vóxel %.2f mm)" % ("+ fotoconsistencia " if refine else "", hull.n_views, voxel)
    ms = [Measurement("height", round(height, 1), "mm", round(lin(height, sig_top), 1), method, "altura sobre el plato"),
          Measurement("width", round(width, 1), "mm", round(lin(width, sig_face), 1), method, "ancho máximo (lado largo de la huella)"),
          Measurement("depth", round(depth, 1), "mm", round(lin(depth, sig_face), 1), method, "profundidad (lado corto de la huella)"),
          Measurement("volume_envelope", round(V / 1000.0, 1), "cm3", round(sig_V / 1000.0, 1), method,
                      "volumen de la envolvente exterior (incluye huecos y concavidades)"),
          Measurement("surface_area", round(A / 100.0, 1), "cm2", round(A * 2 * sig_s / (100.0 * max(1e-6, math.sqrt(A / 4 / math.pi))), 1),
                      method, "área de la superficie de la envolvente")]
    V_hull = mesh_volume(hull_verts, faces)
    return {"hull": hull, "verts": verts, "faces": faces, "normals": normals, "measurements": ms, "volume_mm3": V,
            "volume_hull_mm3": V_hull, "hull_height_mm": float(hull_verts[:, 2].max()),
            "refine": {k: v for k, v in refine_info.items() if k != "displacement"},
            "sigma_surface_mm": sig_s, "mm_per_px": mm_per_px, "voxel": voxel, "footprint_angle_deg": float(ang)}


def density(mass: Optional[Measurement], volume_cm3: Optional[Measurement]) -> Optional[Measurement]:
    if mass is None or volume_cm3 is None or volume_cm3.value <= 0:
        return None
    rho = mass.value / volume_cm3.value
    rel = math.sqrt((mass.uncertainty / max(mass.value, 1e-9)) ** 2 + (volume_cm3.uncertainty / volume_cm3.value) ** 2)
    return Measurement("density_apparent", round(rho, 3), "g/cm3", round(rho * rel, 3), "masa / volumen de la envolvente",
                       HOLLOW_NOTE_ES)


def density_finding(rho: Optional[Measurement]) -> Optional[dict]:
    """Plain-language comparison of the apparent density with typical materials."""
    if rho is None:
        return None
    from .taxonomy import DENSITY_G_CM3, MATERIAL_CLASSES_ES
    lo, hi = rho.value - 2 * rho.uncertainty, rho.value + 2 * rho.uncertainty
    fits = [MATERIAL_CLASSES_ES[k] for k, (a, b) in DENSITY_G_CM3.items() if k not in ("other",) and b >= lo and a <= hi]
    if rho.value < 1.2:
        detail = ("La densidad aparente es baja (%.2f ± %.2f g/cm³). Lo más probable es que el objeto sea hueco "
                  "(vasija, botella, figura hueca) o de un material liviano (madera, fibra, hueso poroso)."
                  % (rho.value, rho.uncertainty))
    else:
        detail = ("La densidad aparente es %.2f ± %.2f g/cm³. Si el objeto es macizo, es compatible con: %s. "
                  "Si es hueco, el material real es más denso que esto." % (rho.value, rho.uncertainty, ", ".join(fits) or "ningún material típico"))
    return {"analysis": "measure", "title_es": "Densidad aparente", "detail_es": detail, "severity": "info", "image": ""}
