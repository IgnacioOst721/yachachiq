#!/usr/bin/env python
"""Calibrate the analysis box (results go to YQ_CALIB_DIR, default ~/yq-data/calibration).

    python tools/box_analysis_calibrate.py target --out ~/Desktop/targets          # print these
    python tools/box_analysis_calibrate.py intrinsics --camera A --images "fotos/intrA/*.jpg"
    python tools/box_analysis_calibrate.py turntable --folder fotos/plato             # camA_000.jpg, camB_030.jpg ...
    python tools/box_analysis_calibrate.py rti --folder fotos/esfera --radius 12.7   # pos1/outline.jpg, pos1/led1.jpg ...
    python tools/box_analysis_calibrate.py thermal --thermal frame.npy --visible camA_000.jpg --camera A
    python tools/box_analysis_calibrate.py show
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _img(p):
    from yq.box.analysis.scanio import load_image
    im = load_image(Path(p))
    if im is None:
        raise SystemExit("no se pudo leer %s" % p)
    return im


def cmd_target(a):
    from yq.box.analysis.calib_target import make_targets
    for k, v in make_targets(Path(a.out).expanduser()).items():
        print(k, *v, sep="\n  ")


def cmd_intrinsics(a):
    from yq.box.analysis import calib
    from yq.box.analysis.calib_charuco import calibrate_intrinsics
    files = sorted(glob.glob(str(Path(a.images).expanduser())))
    r = calibrate_intrinsics([_img(f) for f in files])
    print("cámara %s: %d fotos útiles de %d, error RMS %.3f px (bueno si < 0.5)" % (a.camera, r["n_images"], len(files), r["rms_px"]))
    print("guardado en", calib.save_intrinsics({a.camera: r}))


def cmd_turntable(a):
    from yq.box.analysis import calib
    from yq.box.analysis.calib_charuco import calibrate_turntable
    cal = calib.BoxCalibration.load()
    views = []
    for f in sorted(Path(a.folder).expanduser().glob("cam*_*.*")):
        m = re.match(r"cam([A-Za-z])_(\d{3})", f.stem)
        if m:
            views.append((m.group(1).upper(), float(m.group(2)), _img(f)))
    names = sorted({v[0] for v in views})
    intr = {n: cal.camera(n) for n in names}
    r = calibrate_turntable(views, intr, thickness_mm=a.thickness)
    print("dirección del plato %+d, error RMS %.3f px, inclinación del eje %.2f°, %d vistas"
          % (r["direction"], r["rms_px"], r["axis_tilt_deg"], r["n_views"]))
    cams = {n: {k: v for k, v in c.items() if k in ("R", "t", "rms_px", "n_views")} for n, c in r["cameras"].items()}
    extra = {k: r[k] for k in ("rms_px", "axis_tilt_deg", "angle_scale", "board_xy_mm", "n_views", "source", "thickness_mm")}
    print("guardado en", calib.save_extrinsics(cams, r["direction"], extra))


def cmd_rti(a):
    from yq.box.analysis import calib
    from yq.box.analysis.calib_rti import calibrate_lights
    cal = calib.BoxCalibration.load()
    shots = []
    for d in sorted(p for p in Path(a.folder).expanduser().iterdir() if p.is_dir()):
        leds = {int(m.group(1)): _img(f) for f in d.glob("led*.*") for m in [re.match(r"led(\d+)", f.stem)] if m}
        outline = next(iter(d.glob("outline.*")), None)
        if outline and leds:
            shots.append({"outline": _img(outline), "leds": leds})
    first = next(iter(shots[0]["leds"].values())) if shots else None
    if first is None:
        raise SystemExit("no hay carpetas con outline.jpg + led*.jpg")
    cam = cal.view(a.camera, 0.0, first.shape[1], first.shape[0])
    lights = calibrate_lights(shots, cam, a.radius / 1.0)
    for l in lights:
        print("LED %d: %s mm (%s, %d rayos, residuo %.1f mm)" % (l["index"], l["position_mm"], l["source"], l["rays"], l["residual_mm"]))
    print("guardado en", calib.save_rti_lights(lights, a.camera))


def cmd_thermal(a):
    import numpy as np
    from yq.box.analysis import calib
    from yq.box.analysis.calib_thermal import register
    cal = calib.BoxCalibration.load()
    fr = np.load(a.thermal)
    fr = fr[-1] if fr.ndim == 3 else fr
    vis = _img(a.visible)
    r = register(fr, vis, cal.camera(a.camera), a.platter_deg, cal.direction, to_camera=a.camera)
    print("homografía RMS %.2f px, pose térmica RMS %.2f px" % (r["homography"]["rms_px"], r["rms_px"]))
    print("guardado en", calib.save_thermal(r))


def cmd_show(a):
    from yq.box.analysis import calib
    c = calib.BoxCalibration.load()
    print(json.dumps({"sources": c.sources, "reproj_px": c.reproj_px, "direction": c.direction,
                      "lights": [l.get("source") for l in c.lights], "thermal": bool(c.thermal), "warnings": c.warnings},
                     indent=1, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("target").add_argument("--out", default="calib_targets")
    i = s.add_parser("intrinsics")
    i.add_argument("--camera", required=True)
    i.add_argument("--images", required=True)
    t = s.add_parser("turntable")
    t.add_argument("--folder", required=True)
    t.add_argument("--thickness", type=float, default=0.3, help="grosor del papel/placa del tablero (mm)")
    r = s.add_parser("rti")
    r.add_argument("--folder", required=True)
    r.add_argument("--radius", type=float, required=True, help="radio de la esfera cromada (mm)")
    r.add_argument("--camera", default="B")
    th = s.add_parser("thermal")
    th.add_argument("--thermal", required=True)
    th.add_argument("--visible", required=True)
    th.add_argument("--camera", default="A")
    th.add_argument("--platter-deg", type=float, default=0.0)
    s.add_parser("show")
    a = p.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
