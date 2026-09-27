"""Box scan: preflight() and run_scan() (CONTRACTS.md §6), writing the §4 folder.

Order: weight -> UV -> RTI -> photogrammetry -> thermal (heat last), then the
Mac analysis (or the local light analyses). Cancel = every light off and the
motor stopped (the doors are not touched). Progress messages are Spanish.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from yq.common import config
from yq.common.contracts import ScanRequest, ScanResult, to_dict

from . import scan_stages as st
from . import settings as S
from .device import get_box
from .protocol import BoxError
from .scanctx import ScanAborted, ScanCancelled, ScanContext

log = logging.getLogger("yq.box.scan")
SPANS = {"weight": (0.0, 0.05), "uv": (0.05, 0.12), "rti": (0.12, 0.2), "photogrammetry": (0.2, 0.62),
         "thermal": (0.62, 0.85), "analysis": (0.85, 1.0)}


def preflight(box=None) -> dict:
    """Is everything ready? {"ok", "problems_es", "weight_g"} (weight None when unknown)."""
    problems, weight = [], None
    try:
        box = box or get_box()
        stt = box.status()
    except Exception as e:
        return {"ok": False, "problems_es": ["No me puedo comunicar con la caja (%s)." % e], "weight_g": None}
    if not stt.get("doors_closed"):
        problems.append("Cierra las dos puertas de la caja.")
    if "tmc_uart" in stt.get("faults", []):
        problems.append("El driver del motor no responde (revisa el cable UART del TMC2209).")
    sc = stt.get("scale", {})
    if not sc.get("hx711"):
        problems.append("La balanza (HX711) no responde.")
    elif not (sc.get("tared") and sc.get("calibrated")):
        problems.append("La balanza no está tarada/calibrada.")
    else:
        try:
            w = box.weigh(n=10, timeout_ms=4000)
            weight = round(float(w["grams"]), 1)
            if weight < 2.0:
                problems.append("No hay ningún objeto sobre el plato.")
            elif weight > S.MAX_OBJECT_G:
                problems.append("El objeto pesa %.0f g: el máximo es %.0f g." % (weight, S.MAX_OBJECT_G))
        except BoxError as e:
            problems.append(e.message_es)
    problems += _device_problems()
    free = shutil.disk_usage(Path(config.SCANS_DIR).expanduser()).free / 1e9 if Path(config.SCANS_DIR).exists() else 99
    if free < 2.0:
        problems.append("Queda poco espacio en disco (%.1f GB)." % free)
    return {"ok": not problems, "problems_es": problems, "weight_g": weight}


def _device_problems() -> list:
    out = []
    if not config.mock("cameras"):
        try:
            from .cameras import find_box_cameras
            find_box_cameras()
        except Exception as e:
            out.append("Cámaras: %s" % e)
    if not config.mock("thermal"):
        try:
            from .thermal import find_thermal_device
            find_thermal_device()
        except Exception as e:
            out.append("Cámara térmica: %s" % e)
    return out


def _calibration_refs() -> dict:
    d = S.calib_dir()
    refs = {}
    for key, name in (("intrinsics", "intrinsics.json"), ("extrinsics", "extrinsics.json"),
                      ("rti_lights", "rti_lights.json"), ("thermal_reg", "thermal_reg.json")):
        refs[key] = str(d / name) if (d / name).is_file() else "nominal:CAD-R1"
    return refs


def _write_meta(ctx: ScanContext, req: ScanRequest, started: float, finished: float, extra: dict) -> None:
    box = ctx.box
    meta = {"scan_id": req.scan_id, "profile": req.profile, "analyses": list(req.analyses), "started": started,
            "finished": finished,
            "box": {"firmware": "%s %s" % (box.info_cache.get("fw", "?"), box.info_cache.get("version", "?")),
                    "platter_deg": round(box.platter_deg, 4), "steps_per_rev": box.info_cache.get("steps_per_rev"),
                    "simulated": box.sim is not None},
            "cameras": ctx.cameras_meta, "calibration": _calibration_refs(),
            "door_closed": extra.get("door_closed", True), "warnings": list(ctx.warnings),
            "captured": extra.get("captured", []), "coordinates": "object frame: platter centre, Z up, mm"}
    (ctx.folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def stage_thermal(ctx: ScanContext, profile: str) -> dict:
    from .thermal import open_thermal, record
    box = ctx.box
    heat_s = min(S.THERMAL_HEAT_S, S.FIRMWARE_DEFAULTS["max_on_ms"]["halogen"] / 1000.0 - 3.0)
    base_s, cool_s = S.THERMAL_BASELINE_S, S.PROFILES[profile]["thermal_cool_s"]
    total = base_s + heat_s + cool_s
    ctx.require_doors("thermal")
    box.all_off()
    ctx.rotate_to(0.0, "thermal")
    cam = open_thermal(ctx.scene_fn)
    times = {"on": None, "off": None}

    def on_frame(t, _frame):
        if times["on"] is None and t >= base_s:
            ctx.light("halogen", 1.0, max_ms=int((heat_s + 3.0) * 1000), stage="thermal")
            times["on"] = t
        elif times["on"] is not None and times["off"] is None and t >= times["on"] + heat_s:
            box.light("halogen", 0)
            times["off"] = t
        phase = "Midiendo la temperatura inicial" if t < base_s else (
            "Calentando suavemente con la lámpara" if times["off"] is None else "Enfriando: %d s" % (t - times["off"]))
        ctx.progress("thermal", t / total, phase, t=round(t, 1))

    try:
        rec = record(cam, total, time_scale=ctx.time_scale, on_frame=on_frame, cancel_event=ctx.cancel_event)
    finally:
        cam.close()
        try:
            box.light("halogen", 0)
        except BoxError:
            pass
    ctx.check()
    if times["on"] is None or times["off"] is None:
        raise ScanAborted("La lámpara no pudo calentar el objeto.")
    return st.save_thermal(ctx, rec, times["on"], times["off"])


def run_scan(req: ScanRequest, on_progress: Optional[Callable] = None,
             cancel_event: Optional[threading.Event] = None) -> ScanResult:
    from . import package
    box = get_box()
    folder = Path(config.SCANS_DIR) / req.scan_id
    folder.mkdir(parents=True, exist_ok=True)
    ctx = ScanContext(box, folder, on_progress, cancel_event)
    started, captured, ok = time.time(), [], True
    resets0 = box.resets
    wanted = [a for a in req.analyses]
    try:
        box.all_off()
        box.stop()
        box.zero()
        box.speed(S.FIRMWARE_DEFAULTS["dps"], S.FIRMWARE_DEFAULTS["accel"])
        _write_meta(ctx, req, started, 0.0, {"captured": captured})
        steps = [("weight", lambda: st.stage_weight(ctx)), ("uv", lambda: st.stage_uv(ctx)),
                 ("rti", lambda: st.stage_rti(ctx)),
                 ("photogrammetry", lambda: st.stage_photogrammetry(ctx, req.profile)),
                 ("thermal", lambda: stage_thermal(ctx, req.profile))]
        for name, fn in steps:
            if name not in wanted:
                continue
            ctx.span(*SPANS[name])
            ctx.check()
            fn()
            captured.append(name)
            if box.resets != resets0:
                raise ScanAborted("La caja se reinició durante el escaneo.")
    except ScanCancelled:
        ok = False
        ctx.warn("Escaneo cancelado.")
    except (ScanAborted, BoxError) as e:
        ok = False
        ctx.warn(getattr(e, "message_es", None) or "Error de la caja: %s" % e)
        log.exception("scan %s failed", req.scan_id)
    except Exception as e:   # camera, disk... the visitor still gets a result.json
        ok = False
        ctx.warn("Error durante la captura: %s" % str(e)[:200])
        log.exception("scan %s failed", req.scan_id)
    finally:
        box.safe_off()
        if "thermal" in captured:
            try:
                box.light("fan", 1.0, max_ms=int(S.FAN_AFTER_SCAN_S * 1000))
            except BoxError:
                pass
    finished = time.time()
    _write_meta(ctx, req, started, finished, {"captured": captured})
    base = None
    if ok and captured:
        ctx.span(*SPANS["analysis"])
        base = _analyze(ctx, req, package)
    result = package.finalize_result(folder, req.scan_id, req.profile, started, time.time(), base,
                                     ctx.warnings, ok)
    (folder / "result.json").write_text(json.dumps(to_dict(result), ensure_ascii=False, indent=1))
    ctx.progress("done", 1.0, "Listo" if result.ok else "El escaneo terminó con problemas")
    return result


def _analyze(ctx: ScanContext, req: ScanRequest, package) -> Optional[dict]:
    from yq.common.macclient import MacJobError, MacUnavailable
    ctx.progress("analysis", 0.0, "Analizando en la computadora de IA…")
    try:
        return package.analyze_on_mac(ctx.folder, req.profile, list(req.analyses),
                                      lambda f, m: ctx.progress("analysis", f, m or "Analizando…"))
    except (MacUnavailable, MacJobError) as e:
        ctx.warn("La computadora de IA no está disponible (%s): análisis local reducido." % str(e)[:120])
    def forward(*args, **kw):   # accepts Progress objects or (fraction, message)
        if len(args) == 1 and hasattr(args[0], "fraction"):
            p = args[0]
            ctx.progress("analysis", p.fraction, p.message_es or "Analizando…", **(p.detail or {}))
        elif args:
            ctx.progress("analysis", float(args[0]), str(args[1]) if len(args) > 1 else "Analizando…")

    try:
        res = package.analyze_locally(ctx.folder, forward)
        if res is None:
            ctx.warn("Sin análisis: solo se informa el peso.")
        return res
    except Exception as e:
        log.exception("local analysis failed")
        ctx.warn("El análisis local falló (%s): solo se informa el peso." % str(e)[:120])
        return None
