"""Run RTI on a scan folder: load, per-pixel geometry, photometric stereo, PTM, maps, findings."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .calib import BoxCalibration
from .geometry import nominal_leds, rotz, unit
from .masks import linear, object_mask
from .rti import MAX_WIDTH, export_web_ptm, fit_ptm, eval_ptm, per_pixel_lights, photometric_stereo
from .rti_maps import colormap, curvature, frankot_chellappa, local_relief, normals_png, shade_normals, to_srgb8
from .scanio import Scan, load_image

MAX_CONDITION = 25.0      # photometric-stereo normal matrix condition number above which RTI is not trusted


def _lights_object_frame(scan: Scan, cal: BoxCalibration, deg: float) -> tuple:
    """Light positions (M,3) in the object frame at the RTI platter angle, their indices and source."""
    idx = sorted(scan.rti["images"])
    src = "cad"
    pos = {}
    if cal.lights and cal.lights[0].get("source") not in (None, "cad"):
        pos = {int(l["index"]): np.asarray(l["position_mm"], float) for l in cal.lights}
        src = "calibration"
    elif scan.rti.get("lights", {}).get("lights"):
        pos = {int(l["index"]): np.asarray(l["position_mm"], float) for l in scan.rti["lights"]["lights"] if l.get("position_mm")}
        src = "scan"
    nom = nominal_leds()
    P = np.array([pos.get(i, nom.get(i, np.array([0, 0, 200.0]))) for i in idx])
    inten = [next((float(l.get("intensity", 1.0)) for l in cal.lights if int(l["index"]) == i), 1.0) for i in idx]
    R = rotz(-cal.direction * deg)                  # box frame -> object frame at this platter angle
    return P @ R.T, idx, src, np.array(inten)


def _surface_points(cam, mesh_verts, mesh_faces, mask, fallback_z: float = 40.0) -> np.ndarray:
    """(H,W,3) object-frame point seen by every pixel: mesh depth when available, else a plane at the object."""
    import cv2
    H, W = mask.shape
    vv, uu = np.mgrid[0:H, 0:W]
    o, d = cam.rays(np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64))
    zaxis = cam.R[2]
    if mesh_verts is not None:
        from .render import depth_map
        z = depth_map(cam, mesh_verts, mesh_faces)
        bad = ~np.isfinite(z)
        if bad.any() and (~bad).any():
            idx = cv2.distanceTransformWithLabels(bad.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)[1]
            src = np.zeros(idx.max() + 1, np.float32)
            src[idx[~bad]] = z[~bad]
            z = np.where(bad, src[idx], z)
        depth_along = z.ravel() / np.maximum(d @ zaxis, 1e-6)
    else:
        target = np.array([0, 0, fallback_z])
        zc = float((target - o) @ zaxis)
        depth_along = zc / np.maximum(d @ zaxis, 1e-6)
    return (o[None] + depth_along[:, None] * d).reshape(H, W, 3)


def depth_normals(pts: np.ndarray, cam, smooth: float = 1.5) -> np.ndarray:
    """Unit normals (object frame) of the surface points map (H,W,3), oriented towards the camera."""
    import cv2
    P = np.stack([cv2.GaussianBlur(pts[..., k].astype(np.float32), (0, 0), smooth) for k in range(3)], axis=-1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2
    dv[1:-1] = (P[2:] - P[:-2]) / 2
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    view = cam.center[None, None] - P
    flip = np.sum(n * view, axis=-1) < 0
    return np.where(flip[..., None], -n, n)


def analyze_rti(scan: Scan, cal: BoxCalibration, out_dir: Path, mesh_verts: Optional[np.ndarray] = None,
                mesh_faces: Optional[np.ndarray] = None, prefix: str = "rti_") -> dict:
    import cv2
    out_dir = Path(out_dir)
    meta = scan.rti.get("lights") or {}
    cam_name = str(meta.get("camera") or cal.rti_camera or "B").upper()
    deg = float(meta.get("platter_deg", 0.0))
    idx = sorted(scan.rti["images"])
    raw = [load_image(scan.rti["images"][i]) for i in idx]
    amb = load_image(scan.rti["ambient"]) if scan.rti.get("ambient") else None
    h0, w0 = raw[0].shape[:2]
    s = min(1.0, MAX_WIDTH / float(w0))
    size = (int(round(w0 * s)), int(round(h0 * s)))
    rs = lambda im: cv2.resize(im, size, interpolation=cv2.INTER_AREA) if s < 1.0 else im
    raw = [rs(im) for im in raw]
    amb = rs(amb) if amb is not None else None
    cam = cal.view(cam_name, deg, size[0], size[1])
    lin = np.stack([linear(im) for im in raw])                      # (M,H,W,3)
    if amb is not None:
        lin = np.clip(lin - linear(amb)[None], 0, None)
    sat = np.stack([im.max(axis=2) >= 250 for im in raw])
    # object mask: mesh projection when available, else brightness of the max-over-lights image
    if mesh_verts is not None:
        from .render import depth_map
        mask = np.isfinite(depth_map(cam, mesh_verts, mesh_faces))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    else:
        mx = np.max(np.stack(raw), axis=0)
        mask, _ = object_mask(mx, None, cam)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError("no se encontró el objeto en las fotos RTI")
    x0, y0 = max(0, xs.min() - 8), max(0, ys.min() - 8)
    x1, y1 = min(size[0], xs.max() + 9), min(size[1], ys.max() + 9)
    mask = mask[y0:y1, x0:x1]
    lin = lin[:, y0:y1, x0:x1]
    sat = sat[:, y0:y1, x0:x1]
    K = cam.K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    from .geometry import Camera
    ccam = Camera(cam.name, K, cam.R, cam.t, x1 - x0, y1 - y0, cam.dist, deg)
    pts = _surface_points(ccam, mesh_verts, mesh_faces, mask)
    Lpos, idx, lsrc, inten = _lights_object_frame(scan, cal, deg)
    P = pts[mask]
    Ldir, E = per_pixel_lights(Lpos, P, inten)
    Y = np.einsum("mhwc,c->mhw", lin, [0.2126, 0.7152, 0.0722])[:, mask]
    noise = 3.0 / 255.0 / 12.92
    ymax = Y.max(axis=0)
    valid = (Y > np.maximum(0.05 * ymax[None], noise)) & ~sat[:, mask]
    prior = None
    if mesh_verts is not None:
        prior = depth_normals(pts, ccam)[mask]
    n_w, rho, used, cond = photometric_stereo(Y, Ldir, E, valid, n_prior=prior)
    # where the raking LEDs cannot constrain the normal (tops of tall objects: lights below the tangent
    # plane), keep the 3D-model normal instead of an unreliable estimate
    lit = (np.einsum("mpi,pi->mp", Ldir, np.nan_to_num(prior if prior is not None else n_w)) > 0.15).sum(axis=0)
    weak = (cond > MAX_CONDITION) | (lit < 3)
    if prior is not None:
        n_w = np.where(weak[:, None], prior, n_w)
    else:
        n_w = np.where(weak[:, None], np.nan, n_w)
    good = np.isfinite(n_w[:, 0])
    nz = np.nan_to_num(n_w)
    shading = np.clip(np.einsum("mpi,pi->mp", Ldir, nz), 0, None)
    rgb = lin[:, mask]                                               # (M,P,3)
    J = rgb / np.maximum(E[..., None], 1e-6)
    wv = used[..., None] * shading[..., None]
    alb = (wv * J).sum(axis=0) / np.maximum((wv * shading[..., None]).sum(axis=0), 1e-9)
    Rc = ccam.R
    def to_img(v):
        c = v @ Rc.T
        return np.stack([c[..., 0], -c[..., 1], -c[..., 2]], axis=-1)
    n_img = np.zeros(mask.shape + (3,))
    n_img[..., 2] = 1
    n_img[mask] = np.where(good[:, None], to_img(nz), [0, 0, 1])
    l_img = to_img(Ldir)                                              # (M,P,3)
    target = np.where(rho[None] > 1e-9, (Y / np.maximum(E, 1e-6)) / np.maximum(rho[None], 1e-9), 0)
    coef_p = fit_ptm(target, l_img[..., 0], l_img[..., 1], valid=~sat[:, mask])
    coef = np.zeros(mask.shape + (6,))
    coef[mask] = coef_p
    albedo = np.zeros(mask.shape + (3,))
    albedo[mask] = np.clip(alb / max(np.percentile(alb[good].max(axis=1), 99) if good.any() else 1.0, 1e-9), 0, 1)
    fit_lights = [(float(np.median(l_img[m, :, 0])), float(np.median(l_img[m, :, 1]))) for m in range(len(idx))]
    files = export_web_ptm(coef, albedo, mask, n_img, out_dir, fit_lights, [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], prefix)
    # maps
    mm_px = float(np.median(np.linalg.norm(P - ccam.center, axis=1)) / ccam.focal)
    hgt = frankot_chellappa(n_img[..., 0], n_img[..., 1], n_img[..., 2], mask) * mm_px
    rel = local_relief(hgt, mask, sigma_px=max(3.0, 2.0 / mm_px))
    curv = curvature(n_img[..., 0], n_img[..., 1], mask)
    arts = {"rti_ptm": files["json"], "rti_normals": files["normals"], "rti_albedo": files["albedo"]}
    outs = {"rti_relief": ("relief.png", colormap(rel, mask)), "rti_curvature": ("curvature.png", colormap(curv, mask, cmap="JET")),
            "rti_specular": ("specular.png", to_srgb8(shade_normals(n_img, np.full(mask.shape + (3,), 0.35), (-0.5, 0.6, 0.62), mask,
                                                                   spec=0.9, shininess=30)))}
    for k, az in enumerate((0, 90, 180, 270)):
        lu, lv = 0.9 * np.cos(np.radians(az)), 0.9 * np.sin(np.radians(az))
        Lm = np.clip(eval_ptm(coef, lu, lv), 0, 2)
        outs["rti_relight_%d" % az] = ("relight_%03d.png" % az, to_srgb8(albedo * Lm[..., None]))
    for key, (name, img) in outs.items():
        cv2.imwrite(str(out_dir / (prefix + name)), img[..., ::-1])
        arts[key] = prefix + name
    finding = relief_finding(rel, mask, mm_px, prefix + "relief.png")
    info = {"camera": cam_name, "platter_deg": deg, "lights": lsrc, "valid_normals": float(good.mean()),
            "prior": prior is not None, "rti_constrained_fraction": float((~weak).mean()), "median_condition": float(np.median(cond[np.isfinite(cond)])) if np.isfinite(cond).any() else None,
            "median_lights_used": float(np.median(used.sum(axis=0))), "mm_per_px": mm_px,
            "_normals_world": n_w, "_mask": mask, "_crop": (x0, y0), "_cam": ccam, "_points": P}
    return {"artifacts": arts, "findings": [finding] if finding else [], "info": info}


def relief_finding(rel: np.ndarray, mask: np.ndarray, mm_px: float, image: str) -> Optional[dict]:
    v = rel[mask]
    if v.size < 100:
        return None
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    strong = np.abs(v) > max(0.12, 4 * mad)
    frac = float(strong.mean())
    if frac < 0.003:
        detail = "El relieve fino es suave: no se ven incisiones ni marcas de herramienta claras con esta iluminación."
        sev = "info"
    else:
        detail = ("El mapa de relieve muestra marcas finas (%.1f %% de la superficie visible): pueden ser incisiones, "
                  "decoración grabada, marcas de herramienta o desgaste. Mírelas en el visor RTI moviendo la luz." % (100 * frac))
        sev = "notable"
    return {"analysis": "rti", "title_es": "Relieve fino (RTI)", "detail_es": detail, "severity": sev, "image": image}
