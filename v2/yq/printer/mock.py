"""Printer simulator: shared by the mock HTTP server and the in-process mock client.

A job goes queued -> drawing_front -> flipping -> drawing_back -> done, with
durations in seconds (scaled by `speed`, smaller = faster).
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Optional

from yq.printer.client import FINAL, _bytes

PHASES = (("queued", 1.0), ("drawing_front", 20.0), ("flipping", 4.0), ("drawing_back", 10.0))


class PrinterSim:
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.jobs: dict = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, job: dict, files: dict) -> str:
        with self._lock:
            jid = "p%04d" % next(self._ids)
            self.jobs[jid] = {"id": jid, "job": job, "files": {k: len(v) for k, v in files.items()},
                              "t0": time.time(), "cancelled": False}
        return jid

    def status(self, jid: str) -> Optional[dict]:
        j = self.jobs.get(jid)
        if not j:
            return None
        if j["cancelled"]:
            return {"id": jid, "status": "cancelled", "progress": None}
        el = (time.time() - j["t0"]) / max(1e-6, self.speed)
        total = sum(d for _, d in PHASES)
        acc = 0.0
        for name, dur in PHASES:
            if el < acc + dur:
                return {"id": jid, "status": name, "progress": round(el / total, 3),
                        "message": {"queued": "En cola", "drawing_front": "Dibujando el frente",
                                    "flipping": "Dando vuelta la hoja",
                                    "drawing_back": "Escribiendo la historia y el QR"}[name]}
            acc += dur
        return {"id": jid, "status": "done", "progress": 1.0, "message": "Listo"}

    def cancel(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if not j:
            return False
        j["cancelled"] = True
        return True


class MockPrinterClient:
    """Same interface as PrinterClient, no network (kiosk in YQ_MOCK mode)."""
    source = "mock:ui"

    def __init__(self, speed: Optional[float] = None):
        from yq.server import settings
        self.sim = PrinterSim(speed if speed is not None else float(settings.MOCK_SPEED))
        self.url = "mock://printer"
        self.online = True             # tests flip it to simulate an unplugged printer

    def available(self, max_age_s: float = 5.0) -> bool:
        return self.online

    def submit(self, story_id, front_svg, back_svg, front_gcode=None, back_gcode=None, qr_url="",
               paper=None, pens=None) -> dict:
        from yq.printer.client import PrinterError
        if not self.online:
            raise PrinterError("impresora no disponible (mock apagado)")
        files = {"front.svg": _bytes(front_svg), "back.svg": _bytes(back_svg)}
        jid = self.sim.add({"story_id": story_id, "qr_url": qr_url, "flip": "long-edge"}, files)
        return {"id": jid, "status": "sent", "raw": {"id": jid}}

    def submit_drawing(self, drawing, qr_url: str = "") -> dict:
        d = drawing if isinstance(drawing, dict) else drawing.__dict__
        return self.submit(d["story_id"], d["front_svg"], d["back_svg"], qr_url=qr_url or d.get("qr_url", ""))

    def status(self, job_id: str) -> dict:
        from yq.printer.client import normalize
        raw = self.sim.status(job_id)
        return normalize(raw, job_id) if raw else {"id": job_id, "status": "sent", "progress": None,
                                                   "message": "", "raw": {}}

    def cancel(self, job_id: str) -> bool:
        return self.sim.cancel(job_id)

    def wait(self, job_id, on_status=None, timeout=3600.0, poll_s=0.5, cancel_event=None) -> dict:
        deadline = time.time() + timeout
        st = self.status(job_id)
        while time.time() < deadline:
            st = self.status(job_id)
            if on_status:
                on_status(st)
            if st["status"] in FINAL:
                break
            if cancel_event is not None and cancel_event.wait(poll_s):
                break
            if cancel_event is None:
                time.sleep(poll_s)
        return st
