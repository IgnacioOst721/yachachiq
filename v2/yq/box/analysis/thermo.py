"""Active (long-pulse) thermography with the FLIR Lepton 3.5: what lies just under the surface.

The halogen heats the object for a few seconds, then the camera films it cooling. Where there is
a crack, void, delamination, inclusion or a repair of a different material just under the surface,
heat escapes differently and that spot cools at a different pace than its surroundings.
Pipeline: clean the sequence (repeated/frozen frames, flat-field-correction offset jumps), subtract
the pre-heating baseline, then per pixel on the cooling part:
  * TSR (Shepard 2003): log(dT) vs log(t) fitted with a degree-5 polynomial -> smooth derivatives,
  * PPT (Maldague & Marinetti 1996): phase of the Fourier transform of the cooling curve,
and compare every pixel with its own neighbourhood (so uneven halogen heating does not matter).
"""
from __future__ import annotations

import numpy as np


def clean_sequence(seq: np.ndarray, times: np.ndarray, noise_c: float = 0.02) -> tuple:
    """Drop repeated frames (Lepton sends the same frame again at 27 Hz / during FFC) and remove the
    offset jump after a flat-field correction. Returns (seq, times, info)."""
    seq = np.asarray(seq, np.float64)
    times = np.asarray(times, np.float64)
    keep = np.ones(len(seq), bool)
    for k in range(1, len(seq)):
        if np.max(np.abs(seq[k] - seq[k - 1])) < 1e-6 or np.median(np.abs(seq[k] - seq[k - 1])) < 1e-4 * noise_c:
            keep[k] = False
    s, t = seq[keep], times[keep]
    # offset jumps: median frame-to-frame change is smooth in time except at an FFC
    step = np.median((s[1:] - s[:-1]).reshape(len(s) - 1, -1), axis=1) if len(s) > 2 else np.zeros(0)
    jumps = []
    if len(step) > 6:
        from scipy.ndimage import median_filter
        base = median_filter(step, size=7, mode="nearest")
        mad = 1.4826 * np.median(np.abs(step - base)) + 1e-6
        for k in np.nonzero(np.abs(step - base) > max(8 * mad, 3 * noise_c))[0]:
            gap = t[k + 1] - t[k]
            dt_typ = np.median(np.diff(t))
            if gap > 1.5 * dt_typ or abs(step[k] - base[k]) > 6 * noise_c:
                jumps.append((int(k + 1), float(step[k] - base[k])))
    for k, off in jumps:
        s[k:] -= off
    return s.astype(np.float32), t, {"dropped_frames": int((~keep).sum()), "offset_jumps": jumps}


def local_contrast(x: np.ndarray, mask: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """(x - Gaussian-weighted local mean inside the mask) / robust global scale; 0 outside."""
    import cv2
    m = mask.astype(np.float32)
    xm = np.where(mask, x, 0).astype(np.float32)
    mean = cv2.GaussianBlur(xm, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)
    d = np.where(mask, x - mean, 0)
    scale = 1.4826 * np.median(np.abs(d[mask])) + 1e-9
    return (d / scale).astype(np.float32)


def thermal_mask(dT_peak: np.ndarray, erode: int = 2) -> np.ndarray:
    """Object pixels: those that warmed clearly more than the box interior."""
    import cv2
    v = dT_peak[np.isfinite(dT_peak)]
    thr = max(0.4, 0.25 * float(np.percentile(v, 99)))
    m = (dT_peak > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    from .masks import fill_holes
    m = fill_holes(m > 0).astype(np.uint8)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m > 0


def tsr(dT: np.ndarray, t: np.ndarray, mask: np.ndarray, degree: int = 5, eval_times=None) -> dict:
    """Thermographic Signal Reconstruction on the cooling frames. dT (T,H,W) > 0, t (T,) seconds after
    heating stopped. Returns fitted log-derivatives d1, d2 at eval_times, shape (E,H,W)."""
    lt = np.log(t)
    P = dT[:, mask].astype(np.float64)
    ok = np.all(P > 1e-3, axis=0)
    Y = np.log(np.clip(P, 1e-3, None))
    V = np.vander(lt, degree + 1)
    coef, *_ = np.linalg.lstsq(V, Y, rcond=None)                    # (deg+1, P)
    eval_times = np.asarray(eval_times if eval_times is not None else np.geomspace(t[0] * 1.5, t[-1] / 1.5, 6))
    le = np.log(eval_times)
    dc1 = coef[:-1] * np.arange(degree, 0, -1)[:, None]
    dc2 = dc1[:-1] * np.arange(degree - 1, 0, -1)[:, None]
    E1 = np.vander(le, degree)
    E2 = np.vander(le, degree - 1)
    out1 = np.zeros((len(le),) + mask.shape, np.float32)
    out2 = np.zeros_like(out1)
    out1[:, mask] = np.where(ok[None], E1 @ dc1, 0)
    out2[:, mask] = np.where(ok[None], E2 @ dc2, 0)
    fit = V @ coef
    rms = float(np.sqrt(np.mean((fit - Y)[:, ok] ** 2))) if ok.any() else 0.0
    return {"d1": out1, "d2": out2, "eval_times": eval_times, "fit_rms_log": rms}


def ppt(dT: np.ndarray, t: np.ndarray, mask: np.ndarray, n_bins: int = 5) -> dict:
    """Pulsed Phase Thermography: phase of the FFT of the (uniformly resampled) cooling curves."""
    tu = np.linspace(t[0], t[-1], len(t))
    P = dT[:, mask].astype(np.float64)
    if not np.allclose(tu, t):
        P = _interp_cols(tu, t, P)
    F = np.fft.rfft(P - P[-1:], axis=0)
    freqs = np.fft.rfftfreq(len(tu), d=(tu[1] - tu[0]))
    phase = np.zeros((n_bins,) + mask.shape, np.float32)
    phase[:, mask] = np.angle(F[1:n_bins + 1])
    return {"phase": phase, "freqs": freqs[1:n_bins + 1]}


def _interp_cols(tu, t, P):
    idx = np.clip(np.searchsorted(t, tu) - 1, 0, len(t) - 2)
    w = ((tu - t[idx]) / (t[idx + 1] - t[idx]))[:, None]
    return P[idx] * (1 - w) + P[idx + 1] * w


def anomaly_maps(seq: np.ndarray, times: np.ndarray, meta: dict) -> dict:
    """Clean + baseline + TSR + PPT. Returns maps (H,W) of robust local contrast and the object mask."""
    seq, times, clean = clean_sequence(seq, times)
    on = float(meta.get("heat_on_s", 0.0))
    off = float(meta.get("heat_off_s", on + 10.0))
    pre = times < on - 0.05
    if pre.sum() >= 2:
        base = np.median(seq[pre], axis=0)
    else:
        base = seq[0]
    dT = seq - base[None]
    k_off = int(np.searchsorted(times, off))
    peak = dT[max(0, k_off - 2):k_off + 2].mean(axis=0)
    mask = thermal_mask(peak)
    cool = times > off + 0.3
    tc = times[cool] - off
    dc = dT[cool]
    if len(tc) < 12 or mask.sum() < 20:
        raise ValueError("secuencia térmica demasiado corta o el objeto casi no se calentó")
    # normalise by the heat each pixel received (uneven halogen spot) before comparing neighbours
    norm = np.where(mask, np.maximum(peak, 1e-3), 1.0)
    T = tsr(dc / norm[None], tc, mask)
    P = ppt(dc / norm[None], tc, mask)
    late = dc[-max(3, len(tc) // 10):].mean(axis=0) / norm      # heat still trapped at the end
    maps = {"late": local_contrast(late, mask)}
    for i in range(T["d1"].shape[0]):
        maps["tsr_d1_%d" % i] = local_contrast(T["d1"][i], mask)
    for i in range(P["phase"].shape[0]):
        maps["ppt_%d" % i] = local_contrast(P["phase"][i], mask)
    return {"maps": maps, "mask": mask, "peak": peak, "clean": clean, "tsr": T, "ppt": P, "times_cool": tc,
            "dT_end": dc[-1]}


def detect_anomalies(maps: dict, mask: np.ndarray, z_thr: float = 5.0, min_px: int = 6) -> list:
    """Regions where several independent maps agree. Confidence grows with agreement and contrast."""
    import cv2
    tsr_keys = [k for k in maps if k.startswith("tsr")]
    ppt_keys = [k for k in maps if k.startswith("ppt")]
    z_tsr = np.max(np.abs(np.stack([maps[k] for k in tsr_keys])), axis=0) if tsr_keys else 0
    z_ppt = np.max(np.abs(np.stack([maps[k] for k in ppt_keys])), axis=0) if ppt_keys else 0
    z_late = np.abs(maps["late"])
    votes = (z_tsr > z_thr).astype(int) + (z_ppt > z_thr).astype(int) + (z_late > z_thr).astype(int)
    score = np.where(mask, np.maximum.reduce([z_tsr, z_ppt, z_late]), 0)
    cand = ((votes >= 2) & mask).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    out = []
    for c in range(1, n):
        a = int(stats[c, cv2.CC_STAT_AREA])
        if a < min_px:
            continue
        reg = lab == c
        zmax = float(score[reg].max())
        agree = float(votes[reg].mean()) / 3.0
        conf = float(np.clip(0.35 * agree * 3 / 2 + 0.25 * np.tanh((zmax - z_thr) / 5.0) + 0.1 * np.tanh(a / 40.0), 0, 0.9))
        out.append({"u": float(cents[c][0]), "v": float(cents[c][1]), "area_px": a, "z_max": zmax,
                    "agreement": agree, "confidence": round(conf, 2), "warmer_late": bool(maps["late"][reg].mean() > 0),
                    "_mask": reg})
    out.sort(key=lambda r: -r["confidence"])
    return out, score


def _colorize(x: np.ndarray, mask: np.ndarray, lo: float, hi: float, cmap: str = "INFERNO", scale: int = 4) -> np.ndarray:
    import cv2
    u = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((u * 255).astype(np.uint8), getattr(cv2, "COLORMAP_" + cmap))[:, :, ::-1]
    img = np.where(mask[..., None], img, (img * 0.35).astype(np.uint8))
    return cv2.resize(img, (x.shape[1] * scale, x.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def _finding(a: dict, k: int, image: str) -> dict:
    kind = ("el calor quedó atrapado más tiempo (típico de un hueco, grieta o despegue bajo la superficie)"
            if a["warmer_late"] else
            "el calor se fue más rápido (típico de una inclusión o reparación de un material que conduce mejor)")
    level = "alta" if a["confidence"] >= 0.6 else ("media" if a["confidence"] >= 0.4 else "baja")
    return {"analysis": "thermal", "title_es": "Posible anomalía bajo la superficie (%d)" % (k + 1),
            "detail_es": ("En esta zona %s. Puede ser una grieta, un hueco, una inclusión, una reparación o un cambio de "
                          "material. Confianza %s (%.0f %%). La termografía solo ve unos pocos milímetros bajo la "
                          "superficie y los bordes, el color y el brillo del objeto también pueden producir falsas alarmas."
                          % (kind, level, 100 * a["confidence"])),
            "severity": "notable" if a["confidence"] >= 0.4 else "info", "image": image,
            "thermal_uv": [round(a["u"], 1), round(a["v"], 1)], "confidence": a["confidence"]}


def analyze_thermal(scan, cal, out_dir, prefix: str = "thermal_") -> dict:
    import cv2
    from pathlib import Path
    out_dir = Path(out_dir)
    seq = np.load(scan.thermal["sequence"])
    times = np.load(scan.thermal["times"])
    meta = scan.thermal.get("meta") or {}
    if seq.ndim != 3 or len(seq) != len(times):
        raise ValueError("thermal/sequence.npy y times.npy no coinciden")
    res = anomaly_maps(seq, times, meta)
    anomalies, score = detect_anomalies(res["maps"], res["mask"])
    mask = res["mask"]
    arts = {"thermal_heating": prefix + "heating.png", "thermal_anomaly": prefix + "anomaly.png",
            "thermal_phase": prefix + "phase.png"}
    peak = res["peak"]
    cv2.imwrite(str(out_dir / arts["thermal_heating"]),
                _colorize(peak, mask | (peak > 0), 0, float(np.percentile(peak[mask], 99)))[..., ::-1])
    an = _colorize(score, mask, 0, max(10.0, float(score.max())), "TURBO")
    for k, a in enumerate(anomalies):
        cnts, _ = cv2.findContours(cv2.resize(a["_mask"].astype(np.uint8), (score.shape[1] * 4, score.shape[0] * 4),
                                              interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(an, cnts, -1, (255, 255, 255), 2)
        cv2.putText(an, str(k + 1), (int(a["u"] * 4) + 12, int(a["v"] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / arts["thermal_anomaly"]), an[..., ::-1])
    ph = res["ppt"]["phase"][0]
    cv2.imwrite(str(out_dir / arts["thermal_phase"]),
                _colorize(ph, mask, *np.percentile(ph[mask], [1, 99]), "TWILIGHT")[..., ::-1])
    reg = overlay_on_visible(scan, cal, score, mask, anomalies, out_dir, prefix)
    if reg:
        arts["thermal_overlay"] = reg
    findings = [_finding(a, k, arts["thermal_anomaly"]) for k, a in enumerate(anomalies[:5])]
    if not findings:
        findings.append({"analysis": "thermal", "title_es": "Sin anomalías térmicas claras",
                         "detail_es": "El objeto se enfrió de forma pareja: no se ven grietas, huecos ni reparaciones "
                                      "cerca de la superficie en la cara que mira a la cámara térmica.",
                         "severity": "info", "image": arts["thermal_anomaly"]})
    info = {"clean": res["clean"], "object_px": int(mask.sum()), "anomalies": len(anomalies),
            "tsr_fit_rms_log": res["tsr"]["fit_rms_log"], "max_heating_c": float(np.percentile(peak[mask], 99)),
            "_anomalies": anomalies, "_score": score, "_mask": mask}
    return {"artifacts": arts, "findings": findings, "info": info}


def overlay_on_visible(scan, cal, score, mask, anomalies, out_dir, prefix) -> str:
    """Warp the anomaly map onto the visible photo with the thermal->camera homography (thermal_reg.json),
    when that registration exists and the matching photo is in the scan. Returns the file name or ''."""
    import cv2
    reg = (cal.thermal or {}).get("homography") if cal.thermal else None
    if not reg or not reg.get("H"):
        return ""
    cam, deg = str(reg.get("to_camera", "A")).upper(), float(reg.get("platter_deg", 0.0))
    ph = min((p for p in scan.photos if p.camera == cam), key=lambda p: abs(((p.platter_deg - deg) + 180) % 360 - 180),
             default=None)
    if ph is None:
        return ""
    from .scanio import load_image
    vis = load_image(ph.path)
    H = np.asarray(reg["H"], float)
    ref_w, ref_h = reg.get("image_size", [vis.shape[1], vis.shape[0]])
    S = np.diag([vis.shape[1] / float(ref_w), vis.shape[0] / float(ref_h), 1.0])
    Hs = S @ H
    col = _colorize(score, mask, 0, max(10.0, float(score.max())), "TURBO", scale=1)
    warped = cv2.warpPerspective(col, Hs, (vis.shape[1], vis.shape[0]))
    wm = cv2.warpPerspective(mask.astype(np.uint8), Hs, (vis.shape[1], vis.shape[0])) > 0
    out = vis.copy()
    out[wm] = (0.45 * vis[wm] + 0.55 * warped[wm]).astype(np.uint8)
    name = prefix + "overlay.png"
    cv2.imwrite(str(out_dir / name), out[..., ::-1])
    return name
