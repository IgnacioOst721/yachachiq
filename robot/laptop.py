"""Find the laptop (ComfyUI, ai_server) on whatever network the robot happens to be on.

The Mac is normally reached by its mDNS name (avahi on the Pi, Bonjour on the Mac). On a phone
hotspot mDNS sometimes does not cross, and the old fixed fallback (192.168.7.1) only exists on the
direct cable, so a story could silently drop to the offline motifs. Order of attempts: the last
address that worked, the mDNS name, the fixed cable address, and finally a quick scan of our own
/24 for a machine answering on the ComfyUI port. Results are cached for a minute, and a closed
port is reported closed in milliseconds instead of after a 3 s timeout, so backends that are not
running on the laptop (ai_server, Ollama) are skipped almost for free.
"""
import json
import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config

log = logging.getLogger("laptop")
CACHE = os.path.join(str(config.OUTPUT_DIR), "laptop.json")
PROBE_PORT = 8188                 # ComfyUI: the one service that is always up on the Mac
_lock = threading.Lock()
_found = {"host": None, "t": 0.0, "how": "todavía no buscada"}
_ports = {}                       # (host, port) -> (open, checked_at)


def _connect(host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def port_open(host, port, timeout=0.8, ttl=20.0):
    key, now = (host, port), time.time()
    hit = _ports.get(key)
    if hit and now - hit[1] < ttl:
        return hit[0]
    ok = _connect(host, port, timeout)
    _ports[key] = (ok, now)
    return ok


def _resolve(name, timeout=2.5):
    """gethostbyname with a hard timeout: an mDNS lookup of an absent host hangs for seconds."""
    out = {}

    def go():
        try:
            out["ip"] = socket.gethostbyname(name)
        except OSError:
            pass
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    return out.get("ip")


def _my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packet is sent: just picks the default interface
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _scan(port, timeout=0.4):
    """First host on our /24 that answers on `port` (about 2 s for the whole subnet)."""
    ip = _my_ip()
    if not ip or ip.startswith("127."):
        return None
    base = ip.rsplit(".", 1)[0]
    hosts = [f"{base}.{i}" for i in range(1, 255) if f"{base}.{i}" != ip]
    with ThreadPoolExecutor(max_workers=64) as ex:
        for h, ok in zip(hosts, ex.map(lambda h: _connect(h, port, timeout), hosts)):
            if ok:
                return h
    return None


def _load():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f).get("host")
    except Exception:
        return None


def _save(host):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump({"host": host, "t": time.time()}, f)
    except Exception:
        pass


def _remember(h, how):
    if h != _found["host"]:
        log.info("laptop en %s (%s)", h, how)
    _found.update({"host": h, "t": time.time(), "how": how})
    _save(h)
    return h


def find(force=False, ttl=60.0):
    """IP (or name) of the laptop, i.e. whatever answers on PROBE_PORT; None if nothing does."""
    with _lock:
        now = time.time()
        if not force and _found["host"] and now - _found["t"] < ttl:
            return _found["host"]
        candidates = []
        if _found["host"]:
            candidates.append((_found["host"], "la última que funcionó"))
        saved = _load()
        if saved:
            candidates.append((saved, "guardada de otro día"))
        if config.LAPTOP_HOST:
            ip = _resolve(config.LAPTOP_HOST)
            if ip:
                candidates.append((ip, f"nombre {config.LAPTOP_HOST}"))
        if config.LAPTOP_FALLBACK:
            candidates.append((config.LAPTOP_FALLBACK, "dirección fija del cable"))
        seen = set()
        for h, how in candidates:
            if h in seen:
                continue
            seen.add(h)
            if _connect(h, PROBE_PORT, 0.8):
                return _remember(h, how)
        h = _scan(PROBE_PORT)
        if h:
            return _remember(h, "escaneo de la red")
        _found.update({"host": None, "t": now, "how": "no encontrada en la red"})
        return None


def url(port, force=False):
    """http://host:port if the laptop is up AND that port answers; None (fast) otherwise."""
    h = find(force=force)
    if not h:
        return None
    if port != PROBE_PORT and not port_open(h, port):
        return None
    return f"http://{h}:{port}"


def status():
    h = find()
    return {"host": h, "how": _found["how"], "comfyui": bool(h),
            "ai_server": bool(h and port_open(h, 8600)), "checked": _found["t"]}
