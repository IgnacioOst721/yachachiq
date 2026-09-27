"""The capture stages of a scan, each writing its part of the §4 folder."""
from __future__ import annotations

import json
import shutil

import numpy as np

from . import settings as S
from .cameras import capture_still, open_camera
from .layout import photo_name
from .scanctx import ScanContext


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def _camera_meta(cam) -> dict:
    c = cam.controls
    return {"id": cam.device, "resolution": list(cam.resolution), "focus": c.get("focus"),
            "exposure_us": c.get("exposure_us"), "gain": c.get("gain"), "wb_k": c.get("wb_k"),
            "lock_warnings": (getattr(cam, "lock_info", None) or {}).get("warnings", [])}


def stage_weight(ctx: ScanContext) -> dict:
    box = ctx.box
    ctx.progress("weighing", 0.0, "Pesando el objeto…")
    box.all_off()
    box.motor(False)                      # no chopper hum on the load cell
    ctx.sleep(S.SETTLE_BEFORE_WEIGH_S)
    w = box.weigh()
    box.motor(True)
    data = {"grams": round(float(w["grams"]), 2), "sigma_g": round(float(w["sigma_g"]), 3),
            "samples": w.get("samples", []), "tare_g": None, "tare_raw": w.get("tare_raw"),
            "stable": bool(w.get("stable")), "calibration_factor": w.get("factor"),
            "raw_mean": w.get("raw_mean"), "n": S.WEIGH_SAMPLES, "stable_g": S.WEIGH_STABLE_G,
            "note": "tare_g is null: the tare is the empty platter reading (tare_raw, ADC counts)"}
    if not data["stable"]:
        ctx.warn("La balanza no se estabilizó: el peso puede tener un error mayor.")
    if data["grams"] > S.MAX_OBJECT_G:
        ctx.warn("El objeto pesa más de %.0f g (límite de la balanza)." % S.MAX_OBJECT_G)
    _write_json(ctx.folder / "weight.json", data)
    ctx.progress("weighing", 1.0, "Peso: %.1f g" % data["grams"], grams=data["grams"])
    return data


def stage_photogrammetry(ctx: ScanContext, profile: str) -> dict:
    box, stops = ctx.box, S.PROFILES[profile]["stops"]
    out = ctx.folder / "photogrammetry"
    out.mkdir(parents=True, exist_ok=True)
    poses = {}
    step = 360.0 / stops
    box.all_off()
    ctx.require_doors("photogrammetry")
    # one COB session for both passes: its cool-down (0.5x on-time) would stall a second one
    ctx.light("cob", 1.0, max_ms=S.FIRMWARE_DEFAULTS["max_on_ms"]["cob"], stage="photogrammetry")
    try:
        _photo_passes(ctx, stops, step, out, poses)
    finally:
        box.light("cob", 0)
    ctx.rotate_to(720.0, "photogrammetry")      # back to 0 mod 360, continuing forward
    box.zero()
    _write_json(out / "poses.json", poses)
    _copy_backgrounds(ctx, out)
    return poses


def _photo_passes(ctx: ScanContext, stops: int, step: float, out, poses: dict) -> None:
    for ci, cam_name in enumerate(("A", "B")):     # one camera streams at a time (USB bandwidth)
        cam = open_camera(cam_name, ctx.scene_fn)
        try:
            ctx.cameras_meta[cam_name] = _camera_meta(cam)
            base = 360.0 * ci                  # keep turning forward: pass B continues past 360
            for k in range(stops):
                ctx.check()
                nominal = k * step
                ctx.progress("photogrammetry", (ci * stops + k) / (2.0 * stops),
                             "Fotografiando con la cámara %s: %d de %d" % (cam_name, k + 1, stops),
                             camera=cam_name, index=k, total=stops)
                done = ctx.rotate_to(base + nominal, "photogrammetry")
                ctx.sleep(S.SETTLE_AFTER_ROTATE_S)
                name = photo_name(cam_name, nominal)
                shot = capture_still(cam, out / name, expect="normal")
                ctx.warnings.extend(shot["warnings"])
                poses[name] = {"platter_deg": round(float(done["deg"]) % 360.0, 4), "nominal_deg": nominal,
                               "camera": cam_name, "sharpness": shot["stats"]["sharpness"]}
        finally:
            cam.close()


def _copy_backgrounds(ctx: ScanContext, out) -> None:
    src = S.calib_dir() / "background"
    for cam in ("A", "B"):
        f = src / ("background_cam%s.jpg" % cam)
        info = src / ("background_cam%s.json" % cam)
        if not f.is_file():
            ctx.warn("No hay foto del plato vacío de la cámara %s (usa: python -m yq.box.cli capture-background)."
                     % cam)
            continue
        try:
            saved = json.loads(info.read_text()) if info.is_file() else {}
        except ValueError:
            saved = {}
        now = ctx.cameras_meta.get(cam, {})
        if saved and any(saved.get(k) != now.get(k) for k in ("focus", "exposure_us", "gain", "wb_k")):
            ctx.warn("La foto del plato vacío de la cámara %s se tomó con otros ajustes." % cam)
        shutil.copyfile(f, out / f.name)


def stage_rti(ctx: ScanContext) -> dict:
    box = ctx.box
    out = ctx.folder / "rti"
    out.mkdir(parents=True, exist_ok=True)
    ctx.require_doors("rti")
    box.all_off()
    cam = open_camera(S.RTI_CAMERA, ctx.scene_fn)
    try:
        cam.lock(exposure_us=S.RTI_EXPOSURE_US)
        cam.warm_up(2)
        ctx.cameras_meta.setdefault(S.RTI_CAMERA, _camera_meta(cam))
        ctx.progress("rti", 0.0, "Foto sin luz (ambiente)")
        shot = capture_still(cam, out / "ambient.jpg", expect="dark")
        ctx.warnings.extend(shot["warnings"])
        for i in range(1, 9):
            ch = "rake%d" % i
            ctx.progress("rti", i / 9.0, "Luz rasante %d de 8" % i, led=i)
            ctx.light(ch, 1.0, max_ms=4000, stage="rti")
            try:
                ctx.sleep(S.LIGHT_SETTLE_S)
                shot = capture_still(cam, out / ("led%d.jpg" % i), expect="rti")
                ctx.warnings.extend(shot["warnings"])
            finally:
                box.light(ch, 0)
    finally:
        cam.close()
    calibrated = S.calib_dir() / "rti_lights.json"
    lights = json.loads(calibrated.read_text()) if calibrated.is_file() else S.light_geometry()
    _write_json(out / "lights.json", lights)
    return {"camera": S.RTI_CAMERA, "exposure_us": S.RTI_EXPOSURE_US, "calibrated_lights": calibrated.is_file()}


def stage_uv(ctx: ScanContext) -> dict:
    box = ctx.box
    out = ctx.folder / "uv"
    out.mkdir(parents=True, exist_ok=True)
    ctx.require_doors("uv")
    box.all_off()
    cam = open_camera(S.UV_CAMERA, ctx.scene_fn)
    exposure = {"camera": S.UV_CAMERA, "uv_led_nm": 365, "filter": S.UV_FILTER}
    try:
        cam.lock(exposure_us=S.UV_EXPOSURE_US, gain=S.UV_GAIN)
        cam.warm_up(2)
        ctx.progress("uv", 0.1, "Foto a oscuras")
        capture_still(cam, out / "dark.jpg", expect="dark")
        exposure["dark"] = {"exposure_us": cam.controls["exposure_us"], "gain": cam.controls["gain"]}
        ctx.progress("uv", 0.4, "Luz ultravioleta: buscando restauraciones y pegamentos")
        ctx.light("uv", 1.0, max_ms=8000, stage="uv")
        try:
            ctx.sleep(S.LIGHT_SETTLE_S)
            shot = capture_still(cam, out / "uv.jpg", expect="uv")
            ctx.warnings.extend(shot["warnings"])
        finally:
            box.light("uv", 0)
        exposure["uv"] = {"exposure_us": cam.controls["exposure_us"], "gain": cam.controls["gain"]}
        base = S.CAMERA_CONTROLS[S.UV_CAMERA]
        cam.lock(exposure_us=S.VISIBLE_EXPOSURE_US, gain=base["gain"])
        cam.warm_up(2)
        ctx.progress("uv", 0.8, "Foto con luz blanca")
        ctx.light("cob", 1.0, max_ms=5000, stage="uv")
        try:
            ctx.sleep(S.LIGHT_SETTLE_S)
            capture_still(cam, out / "visible.jpg", expect="normal", min_sharpness=0)
        finally:
            box.light("cob", 0)
        exposure["visible"] = {"exposure_us": cam.controls["exposure_us"], "gain": cam.controls["gain"]}
    finally:
        cam.close()
    _write_json(out / "exposure.json", exposure)
    return exposure


def save_thermal(ctx: ScanContext, rec: dict, heat_on: float, heat_off: float) -> dict:
    out = ctx.folder / "thermal"
    out.mkdir(parents=True, exist_ok=True)
    frames, times = rec["frames_c"].astype(np.float32), rec["times"].astype(np.float64)
    np.save(out / "sequence.npy", frames)
    np.save(out / "times.npy", times)
    base = frames[times < heat_on] if heat_on else frames[:1]
    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else S.THERMAL_FPS
    frozen = np.where(rec["frozen"])[0].tolist()
    meta = {"heat_on_s": round(heat_on, 3), "heat_off_s": round(heat_off, 3), "halogen": "MR16 35W",
            "ambient_c": round(float(np.median(base)), 2) if base.size else None, "fps": round(float(fps), 3),
            "units": "C", "frozen_frames": frozen, "frames": int(len(times)), "camera": "PureThermal 3 + Lepton 3.5",
            "ambient_method": "median of the baseline frames (whole image)"}
    if frozen:
        ctx.warn("La cámara térmica se congeló %d cuadros (calibración FFC); están marcados." % len(frozen))
    _write_json(out / "meta.json", meta)
    return meta
