"""Yachachiq web server: kiosk page, WebSocket events, REST controls, TCP text input.

    python server.py                 -> http://localhost:8877
    YACHACHIQ_MOCK=1 python server.py  (no hardware, no models)

Inputs that reach the pipeline:
    the big button on the screen (microphone)
    POST /api/story {"text": ...}           typed text
    TCP port config.TEXT_PORT, one line     the LSP (sign language) camera app
Signs preview: lsp_app.py POSTs frames to /api/lsp/frame and the current
letters to /api/lsp/text so the kiosk shows what the signer is spelling.
"""
import asyncio
import json
import logging
import os
import socket
import threading
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import photo
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

app = FastAPI(title="Yachachiq")
STATIC = os.path.join(str(config.BASE_DIR), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

pipe: Pipeline = None
loop: asyncio.AbstractEventLoop = None
clients = set()
lsp = {"frame": b"", "t": 0.0, "text": "", "letter": "", "mode": "letter"}


async def _send_all(msg):
    dead = []
    for ws in list(clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def broadcast(event, data):
    if loop is None:
        return
    msg = json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str)
    asyncio.run_coroutine_threadsafe(_send_all(msg), loop)


def _tcp_text_server():
    """One line of text per connection = one story (from the LSP Pi or any device)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((config.HOST, int(config.TEXT_PORT)))
    srv.listen(4)
    log.info("text input on tcp port %s", config.TEXT_PORT)
    while True:
        conn, addr = srv.accept()
        try:
            conn.settimeout(5)
            buf = b""
            while b"\n" not in buf and len(buf) < 20000:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            text = buf.decode("utf-8", errors="ignore").split("\n", 1)[0].strip()
            ok = bool(text) and pipe.submit_text(text)
            log.info("tcp story from %s (%d chars): %s", addr[0], len(text), "accepted" if ok else "rejected")
            conn.sendall(b"ok\n" if ok else b"busy\n")
        except Exception as e:
            log.warning("tcp text error: %s", e)
        finally:
            conn.close()


@app.on_event("startup")
async def _startup():
    global pipe, loop
    loop = asyncio.get_event_loop()
    pipe = Pipeline(on_event=broadcast)
    threading.Thread(target=_tcp_text_server, daemon=True).start()
    log.info("modes: %s", pipe.modes())


# --- pages ---------------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps({"event": "snapshot", "data": pipe.snapshot()}, ensure_ascii=False, default=str))
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except Exception:
                continue
            _command(cmd.get("cmd"), cmd)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def _command(cmd, args):
    if cmd == "toggle":
        return pipe.toggle_listening()
    if cmd == "start":
        return pipe.start_listening()
    if cmd == "stop":
        return pipe.stop_listening()
    if cmd == "cancel":
        return pipe.cancel()
    if cmd == "story":
        return pipe.submit_text(args.get("text", ""))
    if cmd == "redraw":
        return pipe.redraw()
    return False


# --- REST -------------------------------------------------------------------------------------------------
@app.get("/api/state")
async def api_state():
    return pipe.snapshot()


@app.post("/api/story")
async def api_story(req: Request):
    body = await req.json()
    ok = pipe.submit_text(body.get("text", ""))
    return {"ok": ok, "state": pipe.state}


@app.post("/api/listen/start")
async def api_listen_start():
    return {"ok": pipe.start_listening(), "state": pipe.state}


@app.post("/api/listen/stop")
async def api_listen_stop():
    return {"ok": pipe.stop_listening(), "state": pipe.state}


@app.post("/api/listen/toggle")
async def api_listen_toggle():
    return {"ok": pipe.toggle_listening(), "state": pipe.state}


@app.post("/api/cancel")
async def api_cancel():
    return {"ok": pipe.cancel(), "state": pipe.state}


@app.post("/api/redraw")
async def api_redraw():
    return {"ok": pipe.redraw(), "state": pipe.state}


# manual plotter control (gear menu). Only when the pipeline is not drawing.
def _free():
    return not pipe.busy()


@app.post("/api/jog")
async def api_jog(req: Request):
    b = await req.json()
    if not _free():
        return JSONResponse({"ok": False, "error": "ocupado"}, status_code=409)
    pipe.plotter.jog(float(b.get("dx", 0)), float(b.get("dy", 0)), float(b.get("dz", 0)))
    return {"ok": True}


@app.post("/api/pen")
async def api_pen(req: Request):
    b = await req.json()
    if not _free():
        return JSONResponse({"ok": False}, status_code=409)
    (pipe.plotter.pen_up if b.get("up", True) else pipe.plotter.pen_down)()
    return {"ok": True}


@app.post("/api/origin")
async def api_origin():
    if not _free():
        return JSONResponse({"ok": False}, status_code=409)
    pipe.plotter.set_origin()
    return {"ok": True}


@app.post("/api/home")
async def api_home():
    if not _free():
        return JSONResponse({"ok": False}, status_code=409)
    pipe.plotter.home_xy()
    return {"ok": True}


@app.post("/api/unlock")
async def api_unlock():
    pipe.plotter.unlock()
    return {"ok": True}


@app.post("/api/plotter/settings")
async def api_settings():
    if not _free():
        return JSONResponse({"ok": False}, status_code=409)
    pipe.plotter.load_settings(os.path.join(str(config.BASE_DIR), "arduino", "grbl_settings.txt"))
    return {"ok": True}


@app.post("/api/plotter/test")
async def api_test_drawing():
    if not _free():
        return JSONResponse({"ok": False}, status_code=409)
    path = os.path.join(str(config.BASE_DIR), "gcode", "test_drawing.gcode")
    threading.Thread(target=pipe._draw_file, args=(path,), daemon=True).start()
    return {"ok": True}


@app.get("/api/cameras")
async def api_cameras():
    return {"cameras": photo.list_cameras(), "photo": config.PHOTO_CAMERA, "lsp": config.LSP_CAMERA}


@app.get("/api/photo/test")
async def api_photo_test():
    out = os.path.join(str(config.OUTPUT_DIR), "photo_test.png")
    ok = photo.capture(out)
    return FileResponse(out) if ok else JSONResponse({"ok": False, "error": "sin camara"}, status_code=500)


# --- sign language preview -------------------------------------------------------------------------------------
@app.post("/api/lsp/frame")
async def api_lsp_frame(req: Request):
    lsp["frame"] = await req.body()
    lsp["t"] = time.time()
    return {"ok": True}


@app.get("/api/lsp/frame.jpg")
async def api_lsp_frame_get():
    if not lsp["frame"] or time.time() - lsp["t"] > 3:
        return Response(status_code=204)
    return Response(content=lsp["frame"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/lsp/text")
async def api_lsp_text(req: Request):
    b = await req.json()
    lsp.update({"text": b.get("text", ""), "letter": b.get("letter", ""), "mode": b.get("mode", "letter")})
    broadcast("lsp", {"text": lsp["text"], "letter": lsp["letter"], "mode": lsp["mode"], "live": True})
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=int(config.PORT), log_level="warning")
