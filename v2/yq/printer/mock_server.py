"""Mock of Joaquín's printer HTTP server (CONTRACTS.md §8), for tests and demos.

    python -m yq.printer.mock_server --port 8900 [--speed 0.2]

It checks what a real printer would need: front.svg and back.svg present, SVG sizes
in millimetres, a valid job.json. Jobs advance through the printer stages over time.
"""
from __future__ import annotations

import argparse
import json
import re

from fastapi import FastAPI, HTTPException, Request

from yq.printer.mock import PrinterSim


def _svg_mm(data: bytes) -> bool:
    head = data[:2000].decode("utf-8", "ignore")
    return bool(re.search(r'<svg[^>]*\bwidth="[\d.]+mm"', head)) and bool(re.search(r'\bheight="[\d.]+mm"', head))


def create_app(sim: PrinterSim = None) -> FastAPI:
    app = FastAPI(title="Yachachiq printer (mock)")
    app.state.sim = sim or PrinterSim()

    @app.get("/health")
    def health():
        return {"ok": True, "kind": "printer-mock", "jobs": len(app.state.sim.jobs)}

    @app.post("/jobs")
    async def submit(request: Request):
        form = await request.form()
        files = {}
        for key, val in form.multi_items():
            if hasattr(val, "read"):
                files[getattr(val, "filename", None) or key] = await val.read()
        for need in ("front.svg", "back.svg", "job.json"):
            if need not in files:
                raise HTTPException(400, "missing %s" % need)
        for svg in ("front.svg", "back.svg"):
            if not _svg_mm(files[svg]):
                raise HTTPException(400, "%s must have width/height in mm" % svg)
        try:
            job = json.loads(files["job.json"])
        except ValueError:
            raise HTTPException(400, "job.json is not JSON")
        if not job.get("story_id"):
            raise HTTPException(400, "job.json needs story_id")
        jid = app.state.sim.add(job, files)
        return {"id": jid, "status": "queued"}

    @app.get("/jobs/{jid}")
    def status(jid: str):
        st = app.state.sim.status(jid)
        if st is None:
            raise HTTPException(404, "no such job")
        return st

    @app.post("/jobs/{jid}/cancel")
    def cancel(jid: str):
        if not app.state.sim.cancel(jid):
            raise HTTPException(404, "no such job")
        return {"ok": True}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--speed", type=float, default=1.0, help="smaller = faster jobs")
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(PrinterSim(a.speed)), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
