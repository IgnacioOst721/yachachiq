"""Active thermography stage with closed-loop, conservative heating.

baseline (THERMAL_BASELINE_S, halogen off) -> halogen on, watched every frame by
HeatGuard -> off at the first of: planned time, ΔT limit, absolute limit, camera
frozen -> cooling for the profile's cool time counted from the actual switch-off.
The firmware cap (45 s, 3x cool-down) stays as the independent outer limit.
"""
from __future__ import annotations

from . import scan_stages as st
from . import settings as S
from .heatguard import HeatGuard
from .protocol import BoxError
from .scanctx import ScanAborted, ScanContext

REASON_ES = {
    "time": "tiempo previsto",
    "delta_t": "el objeto subió %.1f °C (límite de seguridad)",
    "abs_limit": "el objeto llegó a %.1f °C (límite de seguridad)",
    "camera_frozen": "la cámara térmica dejó de ver (seguridad)",
}


def stage_thermal(ctx: ScanContext, profile: str) -> dict:
    from .thermal import open_thermal, record
    box = ctx.box
    heat_s = min(S.THERMAL_HEAT_S, S.FIRMWARE_DEFAULTS["max_on_ms"]["halogen"] / 1000.0 - 3.0)
    base_s, cool_s = S.THERMAL_BASELINE_S, S.PROFILES[profile]["thermal_cool_s"]
    guard = HeatGuard()
    state = {"on": None, "off": None, "reason": None, "end": base_s + heat_s + cool_s}
    ctx.require_doors("thermal")
    box.all_off()
    ctx.rotate_to(0.0, "thermal")
    cam = open_thermal(ctx.scene_fn)

    def halogen_off(t, reason):
        box.light("halogen", 0)
        state["off"], state["reason"], state["end"] = t, reason, t + cool_s
        return state["end"]

    def on_frame(t, frame):
        new_end = None
        if state["on"] is None:
            if t < base_s:
                guard.add_baseline(frame)
            else:
                guard.start()
                ctx.light("halogen", 1.0, max_ms=int((heat_s + 3.0) * 1000), stage="thermal")
                state["on"] = t
        elif state["off"] is None:
            reason = guard.update(t, frame)
            if reason is None and t >= state["on"] + heat_s:
                reason = "time"
            if reason:
                new_end = halogen_off(t, reason)
        else:
            guard.observe(t, frame)
        if state["on"] is None:
            phase = "Midiendo la temperatura inicial"
        elif state["off"] is None:
            phase = "Calentando suavemente con la lámpara (+%.1f °C)" % guard.last.get("dt_c", 0.0)
        else:
            phase = "Enfriando: %d s" % (t - state["off"])
        ctx.progress("thermal", min(1.0, t / state["end"]), phase, t=round(t, 1),
                     dt_c=round(guard.last.get("dt_c", 0.0), 2))
        return new_end

    try:
        rec = record(cam, state["end"], time_scale=ctx.time_scale, on_frame=on_frame,
                     cancel_event=ctx.cancel_event)
    finally:
        cam.close()
        try:
            box.light("halogen", 0)
        except BoxError:
            pass
    ctx.check()
    if state["on"] is None or state["off"] is None:
        raise ScanAborted("La lámpara no pudo calentar el objeto.")
    if state["reason"] != "time":
        detail = REASON_ES[state["reason"]]
        if "%" in detail:
            detail %= guard.max_dt_heating if state["reason"] == "delta_t" else guard.last.get("temp_c", 0.0)
        ctx.warn("Calentamiento detenido antes: %s." % detail)
    extra = {"stop_reason": state["reason"], "heat_s": round(state["off"] - state["on"], 3),
             "heat_planned_s": heat_s, "heating_enabled": True, **guard.summary()}
    return st.save_thermal(ctx, rec, state["on"], state["off"], extra)
