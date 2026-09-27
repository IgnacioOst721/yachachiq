"""Client for Joaquín's pen printer (CONTRACTS.md §8, proposal until he confirms).

    POST {PRINTER_URL}/jobs   multipart, one part per file, field name = file name:
         front.svg, back.svg, [front.gcode, back.gcode], job.json
         job.json = {"story_id","paper":{"w_mm","h_mm"},"pens":[...],"flip":"long-edge","qr_url"}
         -> {"id": "<job id>", ...}
    GET  {PRINTER_URL}/jobs/<id>          -> {"status": queued|drawing_front|flipping|drawing_back|done|error|cancelled,
                                               "progress": 0..1, "message"?}      (optional, tolerated if missing)
    POST {PRINTER_URL}/jobs/<id>/cancel   (optional)
    GET  {PRINTER_URL}/health             (optional; any HTTP answer = reachable)

The client is tolerant: unknown fields are ignored, other status words are passed
through, a printer without a status endpoint just reports "sent".
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from yq.common import config

STAGES = ("queued", "drawing_front", "flipping", "drawing_back", "done", "error", "cancelled")
_ALIASES = {"pending": "queued", "waiting": "queued", "front": "drawing_front", "printing_front": "drawing_front",
            "flip": "flipping", "waiting_flip": "flipping", "turning": "flipping", "back": "drawing_back",
            "printing_back": "drawing_back", "finished": "done", "complete": "done", "completed": "done",
            "ok": "done", "failed": "error", "canceled": "cancelled"}
FINAL = ("done", "error", "cancelled")


class PrinterError(RuntimeError):
    pass


def _bytes(x) -> bytes:
    if x is None:
        return b""
    if isinstance(x, bytes):
        return x
    p = Path(str(x))
    if len(str(x)) < 4096 and p.exists():
        return p.read_bytes()
    return str(x).encode("utf-8")          # already SVG / G-code text


def normalize(raw: dict, job_id: str = "") -> dict:
    st = str(raw.get("status") or raw.get("state") or "unknown").lower()
    st = _ALIASES.get(st, st)
    prog = raw.get("progress")
    try:
        prog = float(prog) if prog is not None else None
        if prog is not None and prog > 1.0:
            prog = prog / 100.0
    except (TypeError, ValueError):
        prog = None
    return {"id": raw.get("id") or job_id, "status": st, "progress": prog,
            "message": raw.get("message_es") or raw.get("message") or "", "raw": raw}


class PrinterClient:
    source = "http"

    def __init__(self, url: Optional[str] = None, timeout: Optional[float] = None):
        self.url = (url or config.PRINTER_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT
        self._probe = (0.0, False)

    def _req(self, method: str, path: str, timeout: Optional[float] = None, **kw):
        import requests
        try:
            return requests.request(method, self.url + path, timeout=timeout or self.timeout, **kw)
        except requests.RequestException as e:
            raise PrinterError("impresora no disponible en %s (%s)" % (self.url, e.__class__.__name__))

    def available(self, max_age_s: float = 5.0) -> bool:
        t, ok = self._probe
        if time.time() - t < max_age_s:
            return ok
        try:
            ok = self._req("GET", "/health", timeout=2.0).status_code < 500
        except PrinterError:
            ok = False
        self._probe = (time.time(), ok)
        return ok

    def submit(self, story_id: str, front_svg, back_svg, front_gcode=None, back_gcode=None,
               qr_url: str = "", paper: Optional[dict] = None, pens: Optional[list] = None) -> dict:
        from yq.server import settings as ui
        job = {"story_id": story_id,
               "paper": paper or {"w_mm": float(ui.PAPER_W_MM), "h_mm": float(ui.PAPER_H_MM)},
               "pens": list(pens or ui.PENS), "flip": "long-edge", "qr_url": qr_url}
        parts = [("front.svg", ("front.svg", _bytes(front_svg), "image/svg+xml")),
                 ("back.svg", ("back.svg", _bytes(back_svg), "image/svg+xml"))]
        for name, g in (("front.gcode", front_gcode), ("back.gcode", back_gcode)):
            if g:
                parts.append((name, (name, _bytes(g), "text/plain")))
        parts.append(("job.json", ("job.json", json.dumps(job).encode(), "application/json")))
        r = self._req("POST", "/jobs", files=parts, timeout=max(self.timeout, 30.0))
        if r.status_code >= 400:
            raise PrinterError("la impresora rechazó el trabajo (HTTP %d): %s" % (r.status_code, r.text[:200]))
        try:
            data = r.json()
        except ValueError:
            data = {}
        return {"id": str(data.get("id") or data.get("job_id") or story_id), "status": "sent", "raw": data}

    def submit_drawing(self, drawing, qr_url: str = "") -> dict:
        d = drawing if isinstance(drawing, dict) else drawing.__dict__
        return self.submit(d["story_id"], d["front_svg"], d["back_svg"], d.get("front_gcode") or None,
                           d.get("back_gcode") or None, qr_url or d.get("qr_url", ""))

    def status(self, job_id: str) -> dict:
        try:
            r = self._req("GET", "/jobs/%s" % job_id, timeout=4.0)
        except PrinterError:
            return {"id": job_id, "status": "unreachable", "progress": None, "message": "", "raw": {}}
        if r.status_code == 404 or r.status_code == 405:
            return {"id": job_id, "status": "sent", "progress": None, "message": "", "raw": {}}
        try:
            return normalize(r.json(), job_id)
        except ValueError:
            return {"id": job_id, "status": "sent", "progress": None, "message": "", "raw": {}}

    def cancel(self, job_id: str) -> bool:
        try:
            return self._req("POST", "/jobs/%s/cancel" % job_id, timeout=4.0).status_code < 400
        except PrinterError:
            return False

    def wait(self, job_id: str, on_status: Optional[Callable[[dict], None]] = None, timeout: float = 3600.0,
             poll_s: float = 2.0, cancel_event: Optional[threading.Event] = None) -> dict:
        deadline = time.time() + timeout
        st = {"id": job_id, "status": "sent"}
        while time.time() < deadline:
            st = self.status(job_id)
            if on_status:
                on_status(st)
            if st["status"] in FINAL or st["status"] == "sent":
                return st
            if cancel_event is not None and cancel_event.wait(poll_s):
                return st
            elif cancel_event is None:
                time.sleep(poll_s)
        return st
