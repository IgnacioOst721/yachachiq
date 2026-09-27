"""Calibration captures (run from the CLI, stored in CALIB_DIR/box):

* background/          empty platter, cameras A and B, same locked settings as a scan
* calib_set/<stamp>/   a ChArUco/checkerboard target on the platter seen by A and B at
                       several platter angles (BOX-ANALYSIS computes intrinsics/extrinsics)
* rti_sphere/<stamp>/  a glossy black sphere lit by each raking LED (BOX-ANALYSIS
                       computes rti_lights.json from the highlights)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import settings as S
from .cameras import capture_still, open_camera


def _scene_fn(box):
    return box.sim.scene if box.sim is not None else None


def _check_empty(box, force: bool) -> None:
    if force:
        return
    try:
        w = box.weigh(n=10, timeout_ms=4000)
    except Exception:
        return   # scale not ready: trust the operator
    if w["grams"] > 2.0:
        raise RuntimeError("hay %.0f g sobre el plato: debe estar VACÍO (usa --force si es a propósito)" % w["grams"])


def capture_background(box, force: bool = False) -> list:
    _check_empty(box, force)
    out = S.calib_dir() / "background"
    out.mkdir(parents=True, exist_ok=True)
    box.all_off()
    box.light("cob", 1.0, max_ms=60000)
    files = []
    try:
        time.sleep(S.LIGHT_SETTLE_S)
        for cam_name in ("A", "B"):
            cam = open_camera(cam_name, _scene_fn(box))
            try:
                shot = capture_still(cam, out / ("background_cam%s.jpg" % cam_name), expect="normal", min_sharpness=0)
                c = cam.controls
                info = {"focus": c.get("focus"), "exposure_us": c.get("exposure_us"), "gain": c.get("gain"),
                        "wb_k": c.get("wb_k"), "resolution": list(cam.resolution), "device": cam.device,
                        "captured": time.time(), "stats": shot["stats"]}
                (out / ("background_cam%s.json" % cam_name)).write_text(json.dumps(info, indent=1))
                files.append(str(out / shot["file"]))
            finally:
                cam.close()
    finally:
        box.light("cob", 0)
    return files


def capture_calib_set(box, stops: int = 12) -> Path:
    out = S.calib_dir() / "calib_set" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    box.all_off()
    box.zero()
    poses = {}
    box.light("cob", 1.0, max_ms=600000)
    try:
        for ci, cam_name in enumerate(("A", "B")):
            cam = open_camera(cam_name, _scene_fn(box))
            try:
                for k in range(stops):
                    deg = k * 360.0 / stops
                    done = box.rotate_to(360.0 * ci + deg)
                    time.sleep(S.SETTLE_AFTER_ROTATE_S)
                    name = "cam%s_%03d.jpg" % (cam_name, round(deg))
                    capture_still(cam, out / name, expect="normal", min_sharpness=0)
                    poses[name] = {"platter_deg": round(float(done["deg"]) % 360.0, 4), "camera": cam_name}
            finally:
                cam.close()
    finally:
        box.light("cob", 0)
    box.rotate_to(720.0)
    box.zero()
    (out / "poses.json").write_text(json.dumps(poses, indent=1))
    (out / "info.json").write_text(json.dumps({"target": "ChArUco/checkerboard on the platter", "stops": stops,
                                               "cameras": S.CAMERA_CONTROLS, "geometry": S.CAMERA_GEOMETRY},
                                              indent=1, default=list))
    return out


def capture_rti_sphere(box) -> Path:
    out = S.calib_dir() / "rti_sphere" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    box.all_off()
    cam = open_camera(S.RTI_CAMERA, _scene_fn(box))
    try:
        cam.lock(exposure_us=S.RTI_EXPOSURE_US)
        cam.warm_up(2)
        capture_still(cam, out / "ambient.jpg", expect="dark")
        for i in range(1, 9):
            ch = "rake%d" % i
            box.light_wait(ch, 1.0, max_ms=4000)
            try:
                time.sleep(S.LIGHT_SETTLE_S)
                capture_still(cam, out / ("led%d.jpg" % i), expect="rti", min_sharpness=0)
            finally:
                box.light(ch, 0)
    finally:
        cam.close()
    (out / "info.json").write_text(json.dumps({"camera": S.RTI_CAMERA, "exposure_us": S.RTI_EXPOSURE_US,
                                               "nominal_lights": S.light_geometry()}, indent=1))
    return out
