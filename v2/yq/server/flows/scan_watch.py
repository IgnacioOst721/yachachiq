"""Watches a scan folder while BOX-CAPTURE fills it and tells the screen about new files.

Works with any box implementation because it only relies on the folder layout of
CONTRACTS.md §4: photogrammetry/camA_010.jpg (camera, angle), rti/ledN.jpg, uv/*.jpg and
thermal/sequence.npy (rendered here to a false-colour preview for the screen).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("yq.server.scan_watch")
PHOTO = re.compile(r"^cam([A-Za-z])_(\d{3})\.jpe?g$")


def normalize_detail(stage: str, det: dict) -> dict:
    """Fill the keys the live screen uses from whatever the box sends."""
    d = dict(det or {})
    if d.get("led") is not None and not d.get("light"):
        d["light"] = "led%s" % d["led"]
    if stage == "photogrammetry":
        if d.get("platter_deg") is None and d.get("index") is not None and d.get("total"):
            d["platter_deg"] = float(d["index"]) * 360.0 / float(d["total"])
        d.setdefault("light", "cob")
    if stage == "thermal" and not d.get("light") and d.get("phase"):
        d["light"] = "halogen" if d["phase"] == "heating" else "off"
    return d


def render_thermal(npy: Path, out_png: Path) -> bool:
    import numpy as np
    from PIL import Image
    from yq.server.mock_assets import iron
    try:
        seq = np.load(str(npy), mmap_mode="r")
        if seq.ndim != 3 or not len(seq):
            return False
        frame = np.asarray(seq[int(np.argmax(seq.reshape(len(seq), -1).mean(1)))], dtype="f4")
        lo, hi = float(np.percentile(frame, 1)), float(np.percentile(frame, 99.5))
        rgb = iron((frame - lo) / max(1e-3, hi - lo))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.kron(rgb, np.ones((3, 3, 1), "u1"))).save(out_png)
        return True
    except Exception as e:
        log.warning("thermal preview failed: %s", e)
        return False


class ScanWatcher(threading.Thread):
    def __init__(self, folder: Path, url: Callable[[str], str], emit: Callable, cache_dir: Path,
                 cache_url: Callable[[str], str], poll_s: float = 0.4):
        super().__init__(name="scan-watch", daemon=True)
        self.folder, self.url, self.emit, self.poll_s = Path(folder), url, emit, poll_s
        self.cache_dir, self.cache_url = Path(cache_dir), cache_url
        self.seen: set = set()
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.wait(self.poll_s):
            try:
                self.check()
            except Exception:
                log.exception("scan watch")
        try:
            self.check(final=True)                  # the scan ended: every file is complete
        except Exception:
            log.exception("scan watch (final)")

    def _settled(self, p: Path) -> bool:
        try:
            return time.time() - p.stat().st_mtime > 0.25 and p.stat().st_size > 0
        except OSError:
            return False

    def check(self, final: bool = False) -> None:
        for sub, kind in (("photogrammetry", "photogrammetry"), ("rti", "rti"), ("uv", "uv")):
            d = self.folder / sub
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                rel = "%s/%s" % (sub, p.name)
                if rel in self.seen or p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not (final or self._settled(p)):
                    continue
                if p.name.startswith(("background", "preview")):
                    continue
                self.seen.add(rel)
                m = PHOTO.match(p.name)
                self.emit("scan_photo", url=self.url(rel), kind=kind, name=p.name,
                          camera=m.group(1).upper() if m else None, angle=float(m.group(2)) if m else None)
        npy = self.folder / "thermal" / "sequence.npy"
        if "thermal" not in self.seen and npy.exists() and (final or self._settled(npy)):
            self.seen.add("thermal")
            out = self.cache_dir / "thermal_preview.png"
            if render_thermal(npy, out):
                self.emit("scan_thermal", url=self.cache_url("thermal_preview.png") + "?t=%d" % int(time.time()))
