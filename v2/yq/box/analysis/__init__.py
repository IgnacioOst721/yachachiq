"""BOX-ANALYSIS: turn a scan folder (CONTRACTS.md §4) into measurements, a 3D model, images,
findings and an identification.

    from yq.box.analysis import analyze_scan
    result = analyze_scan("~/yq-data/scans/scan-20261105-101500-ab12")      # ScanResult

Runs on the Mac (everything) and on the Jetson (light analyses: weight, silhouettes/volume, RTI,
UV, thermography, 3D model; identification only when the catalog and the model are local).
Heavy imports are lazy: importing this package needs only numpy.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from yq.common import config
from yq.common.contracts import Progress, ScanResult, to_dict

STAGES = [("load", 0.02), ("weight", 0.03), ("photogrammetry", 0.45), ("model3d", 0.10), ("rti", 0.10), ("uv", 0.06),
          ("thermal", 0.08), ("identify", 0.16)]
MESSAGES_ES = {"load": "Leyendo el escaneo...", "weight": "Revisando el peso...", "photogrammetry": "Midiendo la forma en 3D...",
               "model3d": "Armando el modelo 3D...", "rti": "Buscando marcas finas con luz rasante...",
               "uv": "Revisando la luz ultravioleta...", "thermal": "Buscando lo que hay bajo la superficie...",
               "identify": "Comparando con museos del mundo...", "done": "¡Análisis terminado!"}


class _Progress:
    def __init__(self, cb: Optional[Callable]):
        self.cb = cb
        self.start = {}
        acc = 0.0
        for name, w in STAGES:
            self.start[name] = (acc, w)
            acc += w

    def __call__(self, stage: str, frac: float = 0.0, detail: Optional[dict] = None) -> None:
        if not self.cb:
            return
        a, w = self.start.get(stage, (1.0, 0.0))
        try:
            self.cb(Progress(stage, min(1.0, a + w * max(0.0, min(1.0, frac))), MESSAGES_ES.get(stage, ""), detail or {}))
        except Exception:
            pass


def _is_mac() -> bool:
    return config.ROLE in ("mac", "dev")


def analyze_scan(folder, on_progress: Optional[Callable] = None, identify: bool = True, reconstruct: bool = True,
                 lang: str = "spa_Latn", out_dir: Optional[Path] = None, rel_to: Optional[Path] = None,
                 analyses: Optional[list] = None, refine: Optional[bool] = None) -> ScanResult:
    """Analyse a scan folder. Artifacts go to out_dir (default <folder>/analysis); their paths in the
    result are relative to rel_to (default: the scan folder)."""
    from . import settings
    from .calib import BoxCalibration
    from .scanio import load_scan
    folder = Path(folder).expanduser()
    out_dir = Path(out_dir) if out_dir else folder / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_to = Path(rel_to) if rel_to else folder
    prog = _Progress(on_progress)
    prog("load")
    scan = load_scan(folder)
    meta = scan.meta
    wanted = set(analyses or meta.get("analyses") or ["weight", "photogrammetry", "rti", "uv", "thermal", "identify"])
    res = ScanResult(scan_id=scan.scan_id, folder=str(folder), profile=meta.get("profile", "standard"), started=time.time())
    cal = BoxCalibration.load(meta)
    st = {"geometry": None, "views": []}

    def rel(name: str) -> str:
        p = (out_dir / name).resolve()
        try:
            return str(p.relative_to(rel_to.resolve()))
        except ValueError:
            return str(p)

    def run(stage: str, fn) -> None:
        prog(stage, 0.0)
        try:
            fn()
        except Exception as e:
            res.warnings.append("No se pudo completar %s: %s" % (MESSAGES_ES.get(stage, stage).rstrip(". "), e))
            (out_dir / ("error_%s.txt" % stage)).write_text(traceback.format_exc())
        prog(stage, 1.0)

    run("weight", lambda: _weight(scan, res))
    if "photogrammetry" in wanted and scan.photos:
        run("photogrammetry", lambda: _photogrammetry(scan, cal, res, st, prog, settings.REFINE_SURFACE if refine is None else refine))
        if reconstruct and st["geometry"] is not None:
            run("model3d", lambda: _model3d(st, res, out_dir, rel, settings.RECON_TARGET_FACES))
    elif "photogrammetry" in wanted:
        res.warnings.append("El escaneo no tiene fotos de fotogrametría.")
    mesh = (st["geometry"]["verts"], st["geometry"]["faces"]) if st["geometry"] else (None, None)
    if "rti" in wanted and scan.rti:
        run("rti", lambda: _merge(res, __import__("yq.box.analysis.rti_run", fromlist=["x"]).analyze_rti(scan, cal, out_dir, *mesh), rel))
    if "uv" in wanted and scan.uv:
        run("uv", lambda: _merge(res, __import__("yq.box.analysis.uv", fromlist=["x"]).analyze_uv(scan, cal, out_dir, *mesh), rel))
    if "thermal" in wanted and scan.thermal:
        run("thermal", lambda: _merge(res, __import__("yq.box.analysis.thermo", fromlist=["x"]).analyze_thermal(scan, cal, out_dir), rel))
    if identify and "identify" in wanted:
        run("identify", lambda: _identify(scan, res, st, out_dir, rel, meta.get("context")))
    res.warnings = list(dict.fromkeys(scan.warnings + cal.warnings + res.warnings))
    res.finished = time.time()
    res.ok = bool(res.measurements or res.findings or res.identification)
    (out_dir / "analysis.json").write_text(json.dumps(to_dict(res), indent=1, ensure_ascii=False))
    prog("identify", 1.0)
    if on_progress:
        on_progress(Progress("done", 1.0, MESSAGES_ES["done"], {}))
    return res


def _merge(res: ScanResult, part: dict, rel) -> None:
    for k, v in (part.get("artifacts") or {}).items():
        res.artifacts[k] = rel(v)
    for f in part.get("findings") or []:
        f = {k: v for k, v in f.items() if not k.startswith("_")}
        if f.get("image"):
            f["image"] = rel(f["image"])
        res.findings.append(f)


def _weight(scan, res: ScanResult) -> None:
    from .measure import mass_measurement
    m = mass_measurement(scan.weight, res.warnings)
    if m:
        res.measurements.append(to_dict(m))


def _photogrammetry(scan, cal, res: ScanResult, st: dict, prog, refine: bool) -> None:
    from .measure import density, density_finding, geometry, silhouettes
    views = silhouettes(scan, cal, on_progress=lambda f: prog("photogrammetry", 0.3 * f))
    if len(views) < 4:
        raise ValueError("solo %d fotos útiles (se necesitan al menos 4)" % len(views))
    g = geometry(views, cal, refine=refine, on_progress=lambda f: prog("photogrammetry", 0.3 + 0.7 * f))
    st["geometry"], st["views"] = g, views
    res.measurements += [to_dict(m) for m in g["measurements"]]
    from yq.common.contracts import Measurement, from_dict
    mass = next((from_dict(Measurement, m) for m in res.measurements if m["name"] == "mass"), None)
    vol = next((m for m in g["measurements"] if m.name == "volume_envelope"), None)
    rho = density(mass, vol)
    if rho:
        res.measurements.append(to_dict(rho))
        f = density_finding(rho)
        if f:
            res.findings.append(f)


def _model3d(st: dict, res: ScanResult, out_dir: Path, rel, faces: int) -> None:
    from .recon import reconstruct
    from .scanio import load_image
    import cv2
    photos = []
    for ph, cam, _m, _i in st["views"]:
        img = load_image(ph.path)
        if img is not None and (img.shape[1], img.shape[0]) != (cam.width, cam.height):
            img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        photos.append((cam, img))
    g = st["geometry"]
    r = reconstruct(g["verts"], g["faces"], photos, out_dir, target_faces=faces)
    for k, v in r["artifacts"].items():
        res.artifacts[k] = rel(v)


def _best_views(st: dict, n_a: int = 4) -> tuple:
    """Camera A photos closest to 0/90/180/270 deg plus camera B at ~0 deg, with their masks."""
    from .scanio import load_image
    views = st["views"]
    picks = []
    for target in [360.0 * k / n_a for k in range(n_a)]:
        cands = [v for v in views if v[0].camera == "A"]
        if cands:
            picks.append(min(cands, key=lambda v: abs(((v[0].platter_deg - target) + 180) % 360 - 180)))
    b = [v for v in views if v[0].camera == "B"]
    if b:
        picks.append(min(b, key=lambda v: abs(((v[0].platter_deg) + 180) % 360 - 180)))
    imgs, masks = [], []
    import cv2
    unique = list({id(v): v for v in picks}.values())
    for ph, _cam, m, _i in unique:
        img = load_image(ph.path)
        if img is None:
            continue
        if img.shape[:2] != m.shape:
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        imgs.append(img)
        masks.append(m)
    return imgs, masks


def _identify(scan, res: ScanResult, st: dict, out_dir: Path, rel, context) -> None:
    from .identify import identify as run_identify
    imgs, masks = _best_views(st) if st["views"] else ([], [])
    if not imgs:
        raise ValueError("no hay fotos del objeto para identificarlo")
    ident = run_identify(imgs, res.measurements, res.findings, "", context, out_dir, masks)
    for s in ident.get("similar") or []:
        if s.get("image"):
            s["image"] = rel(s["image"])
    res.identification = ident

