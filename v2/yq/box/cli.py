"""Command line for testing the box piece by piece (docs/box.md, in Spanish).

    python -m yq.box.cli ping | info | status
    python -m yq.box.cli weigh | tare | calibrate-scale 500
    python -m yq.box.cli rotate 90            (relative; --to for an absolute angle)
    python -m yq.box.cli light rake3 1.0 2000 | all-off
    python -m yq.box.cli config [key=value ...] [--save]
    python -m yq.box.cli capture-background | capture-calib-set | capture-rti-sphere
    python -m yq.box.cli focus-sweep A | thermal-test
    python -m yq.box.cli scan --profile standard
Add --mock (or YQ_MOCK=1) to use the simulator and the synthetic cameras.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _box():
    from .device import get_box
    return get_box()


def cmd_ping(a):
    b = _box()
    t0 = time.time()
    r = b.ping()
    print("Caja OK en %s (%.1f ms)%s" % (b.port, (time.time() - t0) * 1000, " [SIMULADOR]" if b.sim else ""))
    return r


def cmd_status(a):
    st = _box().status()
    m, sc = st["motor"], st["scale"]
    print("Puertas: frente=%s obturador=%s  -> %s" % (st["doors"]["front"], st["doors"]["shutter"],
                                                     "cerradas" if st["doors_closed"] else "ABIERTA(S)"))
    print("Motor: %s, %.2f°, TMC2209 %s" % ("activo" if m["enabled"] else "apagado", m["deg"],
                                           "OK" if m["tmc_ok"] else "NO RESPONDE"))
    print("Balanza: HX711 %s, tarada=%s, calibrada=%s" % ("OK" if sc["hx711"] else "NO", sc["tared"], sc["calibrated"]))
    on = {k: v for k, v in st["lights"].items() if v["level"] > 0}
    print("Luces encendidas: %s" % (on or "ninguna"))
    if st["faults"]:
        print("FALLAS: %s" % st["faults"])
    return st


def cmd_weigh(a):
    w = _box().weigh()
    print("Peso: %.2f g  (sigma %.2f g, %s)" % (w["grams"], w["sigma_g"], "estable" if w["stable"] else "NO estable"))
    return w


def cmd_tare(a):
    print("Tarando: el plato debe estar VACÍO y las puertas cerradas...")
    r = _box().tare()
    print("Tara guardada (offset %s)." % r["offset"])
    return r


def cmd_calibrate(a):
    print("Pon el peso conocido de %.1f g en el centro del plato..." % a.grams)
    r = _box().calibrate_scale(a.grams)
    print("Factor guardado en el ESP32: %.3f cuentas/g" % r["factor"])
    return r


def cmd_rotate(a):
    b = _box()
    r = b.rotate_to(a.deg) if a.to else b.rotate_by(a.deg)
    print("Plato en %.3f°" % r["deg"])
    return r


def cmd_light(a):
    r = _box().light(a.ch, a.level, a.ms)
    print("%s -> %.2f (se apaga solo en %d ms)" % (r["ch"], r["level"], r["off_in_ms"]))
    if a.hold:   # the ESP32 turns everything off 3 s after this program exits (heartbeat)
        time.sleep(a.hold)
    return r


def cmd_all_off(a):
    return _box().all_off()


def cmd_config(a):
    vals = {}
    for kv in a.values:
        k, v = kv.split("=", 1)
        vals[k] = json.loads(v) if v[:1] in "{[0123456789-tf" else v
    cfg = _box().config_set(save=a.save, **vals) if vals else _box().config_get()
    _print(cfg)
    return None


def cmd_capture_background(a):
    from .calibcapture import capture_background
    files = capture_background(_box(), force=a.force)
    print("Fondos guardados: %s" % files)
    return files


def cmd_capture_calib_set(a):
    from .calibcapture import capture_calib_set
    out = capture_calib_set(_box(), stops=a.stops)
    print("Fotos de calibración en %s" % out)
    return str(out)


def cmd_capture_rti_sphere(a):
    from .calibcapture import capture_rti_sphere
    out = capture_rti_sphere(_box())
    print("Fotos de la esfera en %s" % out)
    return str(out)


def cmd_focus_sweep(a):
    """Sharpness for several focus values: pick the best one for YQ_BOX_FOCUS_<cam>."""
    from .cameras import image_stats, open_camera
    b = _box()
    b.light("cob", 1.0, max_ms=120000)
    cam = open_camera(a.cam, b.sim.scene if b.sim else None)
    rows = []
    try:
        for f in range(a.start, a.stop + 1, a.step):
            cam.lock(focus=f)
            cam.warm_up(3)
            s = image_stats(cam.read())["sharpness"]
            rows.append((f, s))
            print("foco %4d  nitidez %8.1f %s" % (f, s, "#" * int(min(60, 12 * math.log10(1 + s)))))
    finally:
        cam.close()
        b.light("cob", 0)
    best = max(rows, key=lambda r: r[1])
    print("Mejor foco para la cámara %s: %d  -> export YQ_BOX_FOCUS_%s=%d" % (a.cam, best[0], a.cam, best[0]))
    return {"best": best[0], "sweep": rows}


def cmd_thermal_test(a):
    import numpy as np
    from .thermal import open_thermal, record
    b = _box()
    cam = open_thermal(b.sim.scene if b.sim else None)
    try:
        rec = record(cam, a.seconds, time_scale=getattr(b.sim, "time_scale", 1.0) if b.sim else 1.0)
    finally:
        cam.close()
    f = rec["frames_c"]
    print("%d cuadros, %.1f fps, min %.2f °C, mediana %.2f °C, max %.2f °C, congelados %d" % (
        len(f), (len(f) - 1) / max(1e-6, rec["times"][-1] - rec["times"][0]), f.min(), float(np.median(f)), f.max(),
        int(rec["frozen"].sum())))
    return {"frames": len(f)}


def cmd_scan(a):
    from yq.common.contracts import ANALYSES, ScanRequest, new_id
    from .scan import preflight, run_scan
    pf = preflight()
    for p in pf["problems_es"]:
        print("PROBLEMA: " + p)
    if not pf["ok"] and not a.force:
        return pf
    req = ScanRequest(scan_id=new_id("scan"), profile=a.profile,
                      analyses=a.analyses.split(",") if a.analyses else list(ANALYSES))
    res = run_scan(req, lambda p: print("[%3d%%] %s" % (round(p.fraction * 100), p.message_es)))
    print("Carpeta: %s" % res.folder)
    for w in res.warnings:
        print("AVISO: " + w)
    return {"ok": res.ok, "folder": res.folder}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m yq.box.cli", description="Caja de análisis de Yachachiq")
    p.add_argument("--mock", action="store_true", help="simulador + cámaras sintéticas")
    p.add_argument("--json", action="store_true", help="imprime la respuesta completa en JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ping", cmd_ping), ("info", lambda a: _box().info()), ("status", cmd_status),
                     ("weigh", cmd_weigh), ("tare", cmd_tare), ("all-off", cmd_all_off),
                     ("capture-rti-sphere", cmd_capture_rti_sphere)):
        sub.add_parser(name).set_defaults(fn=fn)
    s = sub.add_parser("calibrate-scale")
    s.add_argument("grams", type=float)
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("rotate")
    s.add_argument("deg", type=float)
    s.add_argument("--to", action="store_true", help="ángulo absoluto en vez de relativo")
    s.set_defaults(fn=cmd_rotate)
    s = sub.add_parser("light")
    s.add_argument("ch")
    s.add_argument("level", type=float)
    s.add_argument("ms", type=int, nargs="?", default=None)
    s.add_argument("--hold", type=float, default=0.0, help="segundos que el programa sigue vivo (latido)")
    s.set_defaults(fn=cmd_light)
    s = sub.add_parser("config")
    s.add_argument("values", nargs="*")
    s.add_argument("--save", action="store_true")
    s.set_defaults(fn=cmd_config)
    s = sub.add_parser("capture-background")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_capture_background)
    s = sub.add_parser("capture-calib-set")
    s.add_argument("--stops", type=int, default=12)
    s.set_defaults(fn=cmd_capture_calib_set)
    s = sub.add_parser("focus-sweep")
    s.add_argument("cam", choices=["A", "B"])
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=1000)
    s.add_argument("--step", type=int, default=50)
    s.set_defaults(fn=cmd_focus_sweep)
    s = sub.add_parser("thermal-test")
    s.add_argument("--seconds", type=float, default=5.0)
    s.set_defaults(fn=cmd_thermal_test)
    s = sub.add_parser("scan")
    s.add_argument("--profile", default="standard", choices=["quick", "standard", "detailed"])
    s.add_argument("--analyses", default="", help="p. ej. weight,rti (por defecto: todos)")
    s.add_argument("--force", action="store_true", help="escanear aunque el preflight falle")
    s.set_defaults(fn=cmd_scan)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if a.mock:
        os.environ["YQ_MOCK"] = "1"
        from yq.common import config
        config.MOCK = True
    from .protocol import BoxError
    try:
        out = a.fn(a)
        if a.json and out is not None:
            _print(out)
        return 0
    except BoxError as e:
        print("ERROR: %s (%s)" % (e.message_es, e), file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    finally:
        from . import device
        if device._box is not None:
            device._box.close()


if __name__ == "__main__":
    sys.exit(main())
