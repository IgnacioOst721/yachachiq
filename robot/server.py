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
consent = {"vote": None}          # set by the two buttons on the consent screen


# --- consent with the portrait camera -------------------------------------------------------------------
_face_cascade = None
_smile_cascade = None


def _cascades():
    global _face_cascade, _smile_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return _face_cascade, _smile_cascade


def _look_at(jpeg):
    """jpeg bytes -> dict(bright, covered, face, smile). Cheap enough to run twice a second."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    bright = float(gray.mean())
    out = {"bright": bright, "covered": bright < float(config.COVERED_BRIGHTNESS) or float(gray.std()) < 12,
           "face": False, "smile": False}
    if not out["covered"]:
        try:
            face_c, smile_c = _cascades()
            faces = face_c.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces):
                out["face"] = True
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi = gray[y + h // 2:y + h, x:x + w]
                out["smile"] = len(smile_c.detectMultiScale(roi, 1.7, 22, minSize=(25, 25))) > 0
        except Exception:
            pass
    return out


def _grab_portrait_frame():
    """Latest frame from the sign camera (lsp_app streams it); else open the camera ourselves."""
    if lsp["frame"] and time.time() - lsp["t"] < 2.5:
        return lsp["frame"]
    try:
        import cv2
        cam = config.LSP_CAMERA or 0
        cap = cv2.VideoCapture(cam) if not str(cam).startswith("/dev/") else cv2.VideoCapture(cam, cv2.CAP_V4L2)
        try:
            ok, frame = cap.read()
            if ok:
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return jpg.tobytes() if ok else b""
        finally:
            cap.release()
    except Exception:
        pass
    return b""


def _decide(looks, vote):
    """The contract shown on screen: covering the camera AT ANY MOMENT means no.
    (two consecutive covered readings ~1 s, so a hand passing by does not count)"""
    if vote is not None:
        return bool(vote), ("botón: sí" if vote else "botón: no")
    if not looks:
        return False, "sin cámara para preguntar"
    for a, b in zip(looks, looks[1:]):
        if a["covered"] and b["covered"]:
            return False, "cámara tapada"
    if len(looks) == 1 and looks[0]["covered"]:
        return False, "cámara tapada"
    return True, "cámara abierta" + (" · sonrisa" if any(l["smile"] for l in looks[-6:]) else "")


def consent_provider(seconds):
    """Runs inside the pipeline thread. Shows the consent screen, watches the camera for
    `seconds`, and decides: covered camera or 'No' button -> private; otherwise -> publish."""
    consent["vote"] = None
    broadcast("consent_start", {"seconds": seconds})
    t0 = time.time()
    looks, best, best_score = [], b"", -1.0
    while time.time() - t0 < seconds and not pipe._cancel.is_set():
        if consent["vote"] is not None:
            break
        frame = _grab_portrait_frame()
        look = _look_at(frame) if frame else None
        if look:
            looks.append(look)
            score = look["bright"] + (200 if look["face"] else 0) + (100 if look["smile"] else 0)
            if not look["covered"] and score > best_score:
                best, best_score = frame, score
        broadcast("consent_tick", {"remaining": max(0, int(seconds - (time.time() - t0))),
                                   "camera": bool(frame), **(look or {})})
        time.sleep(0.45)
    publish, reason = _decide(looks, consent["vote"])
    recent = looks[-6:]
    result = {"publish": publish, "reason": reason, "portrait": best if publish else b"",
              "face": any(l["face"] for l in recent), "smile": any(l["smile"] for l in recent)}
    broadcast("consent_result", {k: v for k, v in result.items() if k != "portrait"})
    return result


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
    pipe.consent_provider = consent_provider
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


@app.post("/api/consent")
async def api_consent(req: Request):
    b = await req.json()
    consent["vote"] = bool(b.get("publish"))
    return {"ok": True, "vote": consent["vote"]}


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
