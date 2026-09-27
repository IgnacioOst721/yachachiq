"""Mock of Joaquín's hologram HTTP server (CONTRACTS.md §8), for tests and demos.

    python -m yq.hologram.mock_server --port 8950
"""
from __future__ import annotations

import argparse
import json

from fastapi import FastAPI, HTTPException, Request

from yq.hologram.client import HologramError
from yq.hologram.mock import HologramSim


def create_app(sim: HologramSim = None) -> FastAPI:
    app = FastAPI(title="Yachachiq hologram (mock)")
    app.state.sim = sim or HologramSim()

    @app.get("/health")
    def health():
        return {"ok": True, "kind": "hologram-mock", "stories": len(app.state.sim.stories)}

    @app.get("/status")
    def status():
        return app.state.sim.status()

    @app.post("/stories")
    async def stories(request: Request):
        form = await request.form()
        files = {}
        for key, val in form.multi_items():
            if hasattr(val, "read"):
                files[getattr(val, "filename", None) or key] = await val.read()
        if "story.json" not in files:
            raise HTTPException(400, "missing story.json")
        try:
            story = json.loads(files.pop("story.json").decode("utf-8"))
        except ValueError:
            raise HTTPException(400, "story.json is not JSON")
        for key in ("story_id", "title", "lang", "text", "scenes"):
            if key not in story:
                raise HTTPException(400, "story.json needs %s" % key)
        try:
            return app.state.sim.add(story, files)
        except HologramError as e:
            raise HTTPException(400, str(e))

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(HologramSim(a.speed)), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
