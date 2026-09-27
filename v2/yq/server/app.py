"""Yachachiq v2 kiosk server (Jetson): SPA + WebSocket events + REST actions.

    cd ~/yachachiq/v2 && .venvs/ui/bin/python -m yq.server.app            # real robot
    YQ_MOCK=1 .venvs/ui/bin/python -m yq.server.app                       # any computer, no hardware
    -> http://localhost:8877

The server owns the state: flows run in worker threads and tell every screen what to
show with "screen" events, so a reloaded or second browser shows the same thing.
Events and actions are documented in docs/ui.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from yq.common import config
from yq.server import settings
from yq.server.bus import dumps
from yq.server.kiosk import Kiosk

log = logging.getLogger("yq.server")


def close_kiosk_browser() -> bool:
    """Emergency exit: kill the full-screen Chromium so the desktop shows (like v1)."""
    r = subprocess.run(["pkill", "-f", settings.KIOSK_KILL_PATTERN], capture_output=True)
    log.warning("kiosk exit requested from the screen (pkill rc=%s)", r.returncode)
    return r.returncode == 0


def create_app(kiosk: Optional[Kiosk] = None) -> FastAPI:
    k = kiosk or Kiosk()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.ensure_dirs()
        k.start_background()
        k.show_home()
        yield
        await run_in_threadpool(k.shutdown)

    app = FastAPI(title="Yachachiq kiosk", version="2.0", lifespan=lifespan)
    app.state.kiosk = k
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

    @app.middleware("http")
    async def no_stale_ui(request: Request, call_next):
        resp = await call_next(request)
        p = request.url.path
        if p.startswith("/static/") and not p.startswith("/static/vendor/"):
            resp.headers["Cache-Control"] = "no-cache"          # revalidate: a deploy is seen at once
        return resp

    @app.get("/")
    def index():
        return FileResponse(settings.STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/status")
    def status():
        return k.status()

    @app.get("/api/state")
    def state():
        return k.snapshot()

    @app.get("/api/languages")
    def languages(feature: str = "asr"):
        return {"languages": k.sub.languages.ui_list(feature or "any")}

    @app.post("/api/start")
    async def start(req: Request):
        body = await _json(req)
        opts = {}
        if body.get("flow") == "story" and body.get("method"):
            opts["method"] = body["method"]
        if body.get("flow") == "scan" and body.get("profile"):
            opts["profile"] = body["profile"]
        ok, err = k.start(str(body.get("flow", "")), **opts)
        return JSONResponse({"ok": ok, "error": err}, status_code=200 if ok else 409)

    @app.post("/api/act/{action}")
    async def act(action: str, req: Request):
        body = await _json(req)
        ok = await run_in_threadpool(k.act, action, body)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 409)

    @app.post("/api/cancel")
    async def cancel():
        k.touch()
        return {"ok": await run_in_threadpool(k.cancel, "home")}

    @app.post("/api/touch")
    def touch():
        k.touch()
        return {"ok": True}

    @app.post("/api/kiosk/exit")
    async def kiosk_exit(req: Request):
        body = await _json(req)
        if str(body.get("password", "")) != str(config.KIOSK_EXIT_PASSWORD):
            return JSONResponse({"ok": False, "error": "clave incorrecta"}, status_code=403)
        closed = await run_in_threadpool(close_kiosk_browser)
        return {"ok": True, "closed": closed}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        sub = k.bus.subscribe(asyncio.get_running_loop())
        try:
            await ws.send_text(dumps(k.snapshot()))

            async def sender():
                while True:
                    msg = await sub.queue.get()
                    await ws.send_text(dumps(msg))

            async def receiver():
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if msg.get("type") == "activity":
                        k.touch()
                    elif msg.get("type") == "act":
                        await run_in_threadpool(k.act, str(msg.get("action", "")), msg.get("payload") or {})

            tasks = [asyncio.ensure_future(sender()), asyncio.ensure_future(receiver())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    log.debug("ws task ended: %r", exc)
        except WebSocketDisconnect:
            pass
        finally:
            k.bus.unsubscribe(sub)

    from yq.server.routes_media import register
    register(app, k)
    return app


async def _json(req: Request) -> dict:
    try:
        body = await req.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Yachachiq v2 kiosk server")
    ap.add_argument("--host", default=settings.HOST)
    ap.add_argument("--port", type=int, default=int(config.JETSON_PORT))
    ap.add_argument("--mock", action="store_true", help="simulate every subsystem (same as YQ_MOCK=1)")
    a = ap.parse_args(argv)
    if a.mock:
        config.MOCK = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
