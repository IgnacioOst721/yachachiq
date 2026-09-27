"""Fixtures for the UI tests: fast mocks, a real app in a TestClient, a WebSocket event reader."""
from __future__ import annotations

import contextlib
import json
import queue
import socket
import threading
import time

import pytest


@pytest.fixture
def fast(monkeypatch):
    """Force the UI mocks and make them quick."""
    from yq.common import config
    from yq.server import mocks_art_box, settings
    monkeypatch.setattr(settings, "UI_MOCKS", ["all"])
    monkeypatch.setattr(settings, "MOCK_SPEED", 0.02)
    monkeypatch.setattr(settings, "PAUSE_SCALE", 0.01)
    monkeypatch.setattr(config, "CONSENT_SECONDS", 1)
    monkeypatch.setattr(mocks_art_box.MockBox, "problems", [])
    monkeypatch.setattr(mocks_art_box.MockBox, "fail_at", "")
    monkeypatch.setattr(mocks_art_box.MockArt, "fail", False)
    yield settings


@pytest.fixture
def kiosk(fast):
    from yq.server.kiosk import Kiosk
    k = Kiosk(background=False)
    yield k
    k.shutdown()


@pytest.fixture
def client(kiosk):
    from fastapi.testclient import TestClient
    from yq.server.app import create_app
    with TestClient(create_app(kiosk)) as c:
        c.kiosk = kiosk
        yield c


class Events:
    """Reads a TestClient WebSocket in a thread so tests can wait with a timeout."""

    def __init__(self, ws):
        self.ws = ws
        self.q: queue.Queue = queue.Queue()
        self.all: list = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                msg = self.ws.receive_json()
            except Exception:
                return
            self.all.append(msg)
            self.q.put(msg)

    def until(self, pred, timeout: float = 15.0, what: str = ""):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if pred(msg):
                return msg
        seen = [(m.get("type"), m.get("screen")) for m in self.all[-25:]]
        raise AssertionError("timeout waiting for %s; last events: %s" % (what or pred, seen))

    def screen(self, name: str, timeout: float = 15.0):
        return self.until(lambda m: m.get("type") in ("screen", "hello") and m.get("screen") == name,
                          timeout, "screen " + name)

    def act(self, action: str, **payload):
        self.ws.send_text(json.dumps({"type": "act", "action": action, "payload": payload}))


@pytest.fixture
def events(client):
    with client.websocket_connect("/ws") as ws:
        ev = Events(ws)
        hello = ev.until(lambda m: m.get("type") == "hello", 5, "hello")
        ev.hello = hello
        yield ev


@contextlib.contextmanager
def serve(app):
    """Run an ASGI app on a real local port (for the HTTP clients)."""
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 10:
        time.sleep(0.02)
    try:
        yield "http://127.0.0.1:%d" % port
    finally:
        server.should_exit = True
        th.join(5)


def wait_for(pred, timeout: float = 10.0, step: float = 0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not met in %.1f s" % timeout)
