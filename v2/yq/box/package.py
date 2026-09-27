"""Hand the scan to BOX-ANALYSIS: zip it (photogrammetry downscaled), run the Mac
job `scan_analyze`, download its artifacts into analysis/; fall back to the
local light analyses, else report the weight only."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from yq.common.contracts import Measurement, ScanResult, from_dict, to_dict

from . import settings as S

log = logging.getLogger("yq.box.package")
SKIP_TOP = {"analysis", "result.json", "scan.zip"}


def package_scan(folder, zip_path=None, long_side: int = None) -> Path:
    """scan.zip with the §4 layout; photogrammetry JPEGs resized to `long_side` px."""
    import cv2
    folder = Path(folder)
    zip_path = Path(zip_path or folder / "scan.zip")
    long_side = long_side or S.PACKAGE_LONG_SIDE_PX
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(folder.rglob("*")):
            rel = p.relative_to(folder)
            if not p.is_file() or rel.parts[0] in SKIP_TOP:
                continue
            if rel.parts[0] == "photogrammetry" and p.suffix.lower() == ".jpg":
                img = cv2.imread(str(p))
                if img is not None and max(img.shape[:2]) > long_side:
                    s = long_side / float(max(img.shape[:2]))
                    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                                     interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, S.PACKAGE_JPEG_QUALITY])
                    if ok:
                        zf.writestr(str(rel), buf.tobytes(), compress_type=zipfile.ZIP_STORED)
                        continue
            zf.write(p, str(rel), compress_type=zipfile.ZIP_STORED if p.suffix.lower() == ".jpg" else None)
    return zip_path


def mass_measurement(folder) -> list:
    import json
    w = Path(folder) / "weight.json"
    if not w.is_file():
        return []
    d = json.loads(w.read_text())
    return [to_dict(Measurement(name="mass", value=d["grams"], unit="g", uncertainty=d.get("sigma_g", 0.0),
                                method="HX711 + TAL220B 5 kg, ventana estable n=%s" % d.get("n"),
                                note="" if d.get("stable") else "no estable"))]


def analyze_on_mac(folder, profile: str, analyses: list, on_progress=None) -> dict:
    """Run scan_analyze on the Mac; returns the ScanResult dict with artifact
    paths rewritten to analysis/<name>. Raises MacUnavailable / MacJobError."""
    from yq.common.macclient import MacUnavailable, client
    c = client()
    if not c.available():
        raise MacUnavailable("Mac worker not available")
    folder = Path(folder)
    zip_path = package_scan(folder)
    try:
        job = c.submit_job("scan_analyze", {"profile": profile, "analyses": analyses, "lang": "spa_Latn"},
                           files=[zip_path])
        result = c.wait_job(job, on_progress=on_progress)
        out_dir = folder / "analysis"
        artifacts = {}
        for name, rel in (result.get("artifacts") or {}).items():
            dest = out_dir / rel
            c.download(job, rel, dest)
            artifacts[name] = str(dest.relative_to(folder))
        for f in (result.get("findings") or []):
            if f.get("image"):
                f["image"] = "analysis/" + f["image"]
        result["artifacts"] = artifacts
        result["engine"] = "mac"
        return result
    finally:
        zip_path.unlink(missing_ok=True)


def analyze_locally(folder, on_progress=None):
    """BOX-ANALYSIS's local light analyses if they are installed, else None."""
    try:
        from yq.box.analysis import analyze_scan
    except ImportError:
        return None
    res = analyze_scan(folder, on_progress, identify=False, reconstruct=False)
    return to_dict(res) if not isinstance(res, dict) else res


def finalize_result(folder, scan_id: str, profile: str, started: float, finished: float, base, warnings: list,
                    ok: bool) -> ScanResult:
    """Merge an analysis result (dict or None) with what the capture knows."""
    res = from_dict(ScanResult, base or {}) if base else ScanResult(scan_id=scan_id, folder=str(folder),
                                                                   profile=profile, started=started)
    res.scan_id, res.folder, res.profile, res.started, res.finished = scan_id, str(folder), profile, started, finished
    if not any((m.get("name") if isinstance(m, dict) else m.name) == "mass" for m in res.measurements):
        res.measurements = mass_measurement(folder) + list(res.measurements)
    res.warnings = list(warnings) + [w for w in (res.warnings or []) if w not in warnings]
    res.ok = bool(ok and (base or {}).get("ok", True))
    return res
