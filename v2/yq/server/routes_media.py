"""Media and operator routes: story/scan files, live camera previews, gear-menu admin."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from yq.common import config

MIME = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".svg": "image/svg+xml", ".wav": "audio/wav"}


def _safe_file(base: Path, rel: str) -> Path:
    base = Path(base).resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    return p


def _mjpeg(get: Callable[[], Optional[bytes]], max_frames: int = 0, fps: float = 12.0):
    async def gen():
        last, sent, t_last = None, 0, 0.0
        while True:
            frame = get()
            if frame and (frame is not last or time.time() - t_last > 1.0):
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame)) \
                    + frame + b"\r\n"
                last, t_last, sent = frame, time.time(), sent + 1
                if max_frames and sent >= max_frames:
                    return
            await asyncio.sleep(1.0 / fps)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


def register(app: FastAPI, k) -> None:
    def sign_frame() -> Optional[bytes]:
        eng = getattr(k.flow, "engine", None) if k.flow else None
        try:
            return eng.latest_jpeg() if eng is not None else None
        except Exception:
            return None

    @app.get("/sign/preview.mjpeg")
    def sign_preview(max_frames: int = 0):
        return _mjpeg(sign_frame, max_frames)

    @app.get("/sign/preview.jpg")
    def sign_preview_jpg():
        f = sign_frame()
        return Response(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"}) if f \
            else Response(status_code=204)

    @app.get("/camera/preview.mjpeg")
    def camera_preview(max_frames: int = 0):
        return _mjpeg(k.pump.latest, max_frames, fps=10)

    @app.get("/camera/preview.jpg")
    def camera_preview_jpg():
        f = k.pump.latest()
        return Response(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"}) if f \
            else Response(status_code=204)

    @app.get("/files/{kind}/{rel:path}")
    def files(kind: str, rel: str):
        base = {"stories": config.STORIES_DIR, "scans": config.SCANS_DIR,
                "uicache": Path(config.DATA_DIR) / "ui_cache"}.get(kind)
        if base is None:
            raise HTTPException(404, "no such folder")
        p = _safe_file(Path(base), rel)
        return FileResponse(p, media_type=MIME.get(p.suffix.lower()), headers={"Cache-Control": "no-cache"})

    # -- gear menu (operator); every call needs the kiosk password ------------------------------
    async def _auth(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:
            body = {}
        if str((body or {}).get("password", "")) != str(config.KIOSK_EXIT_PASSWORD):
            raise HTTPException(403, "clave incorrecta")
        return body

    @app.post("/api/admin/check")
    async def admin_check(req: Request):
        await _auth(req)
        return {"ok": True, "status": k.status(), "errors": list(k.bus.find("toast"))[-10:]}

    @app.post("/api/admin/stories")
    async def admin_stories(req: Request):
        await _auth(req)
        out = []
        sd = Path(config.STORIES_DIR)
        for d in sorted(sd.iterdir(), reverse=True)[:30] if sd.is_dir() else []:
            try:
                meta = json.loads((d / "story.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            state = "published" if (d / ".published").exists() else "private" if (d / ".private").exists() \
                else "pending" if (d / ".ready").exists() else "unfinished"
            out.append({"id": d.name, "title": meta.get("title"), "state": state,
                        "printer": (meta.get("printer") or {}).get("status"), "has_drawing": bool(meta.get("drawing"))})
        return {"stories": out}

    @app.post("/api/admin/reprint")
    async def admin_reprint(req: Request):
        body = await _auth(req)
        sid = str(body.get("story_id", ""))
        try:
            meta = json.loads((_safe_file(Path(config.STORIES_DIR), sid + "/story.json")).read_text(encoding="utf-8"))
        except HTTPException:
            return JSONResponse({"ok": False, "error": "no existe esa historia"}, status_code=404)
        if not meta.get("drawing"):
            return JSONResponse({"ok": False, "error": "esa historia no tiene dibujo"}, status_code=409)
        try:
            job = await run_in_threadpool(k.sub.printer.submit_drawing, meta["drawing"], meta.get("qr_url", ""))
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=502)
        return {"ok": True, "job": job.get("id")}

    @app.post("/api/admin/publish")
    async def admin_publish(req: Request):
        await _auth(req)
        from yq.publish import sync as publish
        return await run_in_threadpool(publish.sync)

    @app.post("/api/admin/refresh")
    async def admin_refresh(req: Request):
        await _auth(req)
        await run_in_threadpool(k.refresh_status)
        return k.status()

    @app.post("/api/admin/mock_camera")
    async def admin_mock_camera(req: Request):
        """Demo helper: simulate a covered visitor camera (only with the mock camera)."""
        body = await _auth(req)
        if not hasattr(k.sub.camera, "covered") or getattr(k.sub.camera, "mode", "") != "mock:ui":
            return JSONResponse({"ok": False, "error": "la cámara es real"}, status_code=409)
        k.sub.camera.covered = bool(body.get("covered"))
        return {"ok": True, "covered": k.sub.camera.covered}
