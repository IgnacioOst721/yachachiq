"""MacBook AI worker: one FastAPI server for every heavy model (CONTRACTS.md §3).

Run on the Mac:
    cd ~/yachachiq/v2 && .venv/bin/python -m yq.macworker.app

Domain modules named yq/macworker/routes_<domain>.py are discovered
automatically. Each may define:
    router = APIRouter()                       # synchronous endpoints
    def setup(jobs, models): ...               # register job kinds and models
A module that fails to import (missing optional dependency) is skipped and
reported in /health, so one broken domain never takes the others down.
"""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import platform
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from yq.common import config
from yq.macworker.jobs import jobs
from yq.macworker.modelmgr import models

log = logging.getLogger("yq.macworker")
STARTED = time.time()
DOMAINS: dict[str, str] = {}   # module -> "ok" | error text


def create_app() -> FastAPI:
    app = FastAPI(title="Yachachiq Mac worker", version="2.0")

    import yq.macworker as pkg
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("routes_"):
            continue
        name = "yq.macworker." + info.name
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "setup"):
                mod.setup(jobs, models)
            if hasattr(mod, "router"):
                app.include_router(mod.router)
            DOMAINS[info.name] = "ok"
        except Exception as e:  # keep serving the other domains
            DOMAINS[info.name] = "%s: %s" % (type(e).__name__, e)
            log.exception("domain %s failed to load", name)

    @app.get("/health")
    def health():
        return {"ok": True, "role": "mac", "host": platform.node(), "uptime_s": round(time.time() - STARTED),
                "domains": DOMAINS, "job_kinds": jobs.kinds(), "models": models.stats()}

    @app.post("/jobs")
    async def submit(kind: str = Form(...), params: str = Form("{}"), files: list[UploadFile] = File(default=[])):
        try:
            p = json.loads(params or "{}")
        except ValueError:
            raise HTTPException(400, "params must be JSON")
        uploads = [(f.filename, await f.read()) for f in files]
        try:
            job = jobs.submit(kind, p, uploads)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"id": job.id}

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "no such job")
        return {"ok": True}

    @app.get("/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        out = (job.folder / "out").resolve()
        path = (out / name).resolve()
        if out not in path.parents or not path.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    @app.post("/models/unload")
    def unload_all():
        models.unload_all()
        return models.stats()

    return app


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config.ensure_dirs()
    uvicorn.run(create_app(), host="0.0.0.0", port=config.MAC_PORT, log_level="info")


if __name__ == "__main__":
    main()
