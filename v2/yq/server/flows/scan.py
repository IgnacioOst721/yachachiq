"""Object analysis flow: intro -> preflight -> scanning (live) -> results.

The box is shared hardware: only one run_scan at a time (Box.lock). Cancelling sets
the box cancel event and waits (bounded) for run_scan to return, so the lights and
the heater are off before the screen goes back home.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from yq.common import config
from yq.common.contracts import SCAN_PROFILES, ScanRequest, new_id
from yq.server import settings
from yq.server.flows.base import Cancelled, Flow, FlowError
from yq.server.flows.scan_context import ScanContextMixin
from yq.server.flows.scan_watch import ScanWatcher, normalize_detail

log = logging.getLogger("yq.server.scan")
PROFILE_MINUTES = {"quick": 3, "standard": 8, "detailed": 15}


class ScanFlow(ScanContextMixin, Flow):
    kind = "scan"
    first = "intro"
    retry_map = {"context": "context", "preflight": "preflight", "scanning": "preflight", "results": "intro"}

    def __init__(self, kiosk, profile: str = "standard"):
        super().__init__(kiosk)
        self.profile = profile if profile in SCAN_PROFILES else "standard"
        self.scan_id = ""
        self.weight_g = None
        self.box_cancel = threading.Event()
        self.result: dict = {}
        self._scan_thread_done = threading.Event()
        self._scan_thread_done.set()
        self.immediate = {"stop_scan": lambda p: self.cancel("stopped")}
        self.init_context()

    def on_cancel(self) -> None:
        self.box_cancel.set()

    def cleanup(self) -> None:
        self.box_cancel.set()
        if not self._scan_thread_done.wait(20):             # hardware must be safe before we leave
            log.error("run_scan did not stop within 20 s after cancel")

    def url(self, rel: str) -> str:
        return "/files/scans/%s/%s" % (self.scan_id, str(rel).lstrip("/"))

    def _rel(self, path):
        if not path:
            return None
        p = Path(str(path))
        if p.is_absolute():
            try:
                p = p.resolve().relative_to((Path(config.SCANS_DIR) / self.scan_id).resolve())
            except ValueError:
                return None
        return self.url(p.as_posix())

    # -- states ----------------------------------------------------------------------------
    def s_intro(self):
        self.show("intro", profiles=[{"id": p, "minutes": PROFILE_MINUTES.get(p, 5)} for p in SCAN_PROFILES],
                  selected=self.profile)
        while True:
            name, p = self.wait("choose_profile", "start")
            if name == "choose_profile" and p.get("profile") in SCAN_PROFILES:
                self.profile = p["profile"]
                self.update(selected=self.profile)
            elif name == "start":
                return "context"

    def s_preflight(self):
        self.show("preflight")
        t0 = time.time()
        while self.sub.box.lock.locked():                   # a previous scan is still stopping
            if time.time() - t0 > 30:
                raise FlowError("La caja todavía está terminando el escaneo anterior. Espera un momento.",
                                "err_box_busy", retry="preflight")
            self.update(waiting_box=True)
            if self.cancel_event.wait(0.3):
                raise Cancelled()
        r = self.call(self.sub.box.preflight, timeout=60) or {}
        if not r.get("ok"):
            self.show("preflight_fail", problems=list(r.get("problems_es") or ["La caja no está lista."]))
            self.wait("retry")
            return "preflight"
        self.weight_g = r.get("weight_g")
        return "scanning"

    def s_scanning(self):
        self.scan_id = new_id("scan")
        self.box_cancel.clear()
        req = ScanRequest(scan_id=self.scan_id, profile=self.profile, context=dict(self.context))
        self.show("scanning", scan_id=self.scan_id, profile=self.profile, weight_g=self.weight_g,
                  stage="starting", fraction=0.0, message_es="Preparando la caja…")
        last = {"stage": ""}

        def progress(p):
            if not isinstance(p, dict):
                from yq.common.contracts import to_dict
                p = to_dict(p)
            det = normalize_detail(p.get("stage") or "", p.get("detail") or {})
            for key in ("photo", "thermal_preview", "image"):
                if det.get(key):
                    det[key + "_url"] = self._rel(det[key])
            ev = {"stage": p.get("stage"), "fraction": p.get("fraction"), "message_es": p.get("message_es"),
                  "detail": det}
            self.bus.emit("scan_live", **ev)
            if p.get("stage") != last["stage"]:
                last["stage"] = p.get("stage")
                self.data.update(stage=p.get("stage"))
            self.data.update(fraction=p.get("fraction"), message_es=p.get("message_es"))

        def run():
            try:
                with self.sub.box.lock:
                    if self.box_cancel.is_set():
                        return None
                    return self.sub.box.run_scan(req, on_progress=progress, cancel_event=self.box_cancel)
            finally:
                self._scan_thread_done.set()

        self._scan_thread_done.clear()
        cache = Path(config.DATA_DIR) / "ui_cache" / self.scan_id
        watcher = ScanWatcher(Path(config.SCANS_DIR) / self.scan_id, self.url, self.bus.emit, cache,
                              lambda n: "/files/uicache/%s/%s" % (self.scan_id, n))
        watcher.start()
        try:
            res = self.call(run, timeout=float(settings.SCAN_TIMEOUT), on_cancel=self.box_cancel.set)
        except Cancelled:
            self.show("stopping")
            self._scan_thread_done.wait(20)
            raise
        except TimeoutError:
            self.box_cancel.set()
            raise
        finally:
            watcher.stop()
            watcher.join(5)                              # its last photos reach the screen before the results
        if not res or not res.get("ok", True):
            warn = "; ".join((res or {}).get("warnings") or [])[:200]
            raise FlowError("El escaneo no terminó bien. Revisa la caja y probemos otra vez.", "err_scan",
                            retry="preflight", detail=warn)
        self.result = res
        return "results"

    def s_results(self):
        self.show("results", result=self.ui_result(self.result))
        name, _ = self.wait("finish", "new_scan")
        if name == "new_scan":
            self.next_flow = "scan"
        return None

    # -- results for the screen ------------------------------------------------------------------
    def ui_result(self, r: dict) -> dict:
        arts = {k: self._rel(v) for k, v in (r.get("artifacts") or {}).items() if v}

        def pick(*words, ext=()):
            for k, v in arts.items():
                if v and any(w in k.lower() for w in words) and (not ext or v.lower().endswith(ext)):
                    return v
            return None

        ident = r.get("identification") or None
        findings = []
        for f in r.get("findings") or []:
            f = dict(f)
            f["image_url"] = self._rel(f.get("image")) if f.get("image") else None
            findings.append(f)
        mock = bool(ident and str(ident.get("engine", "")).startswith("mock")) or \
            any("SIMULADO" in str(w).upper() for w in r.get("warnings") or [])
        return {
            "scan_id": r.get("scan_id") or self.scan_id, "profile": r.get("profile"), "ok": r.get("ok", True),
            "duration_s": round((r.get("finished") or time.time()) - (r.get("started") or time.time())),
            "measurements": r.get("measurements") or [], "identification": ident, "findings": findings,
            "warnings": r.get("warnings") or [], "artifacts": arts, "mock": mock, "context": dict(self.context),
            "viewers": {
                "model": pick("glb", "model", "mesh", ext=(".glb", ".gltf")),
                "rti": pick("rti", "ptm", ext=(".json",)),
                "uv": {"uv": pick("uv_image", "uv_raw") or pick("uv", ext=(".jpg", ".png")),
                       "visible": pick("visible"), "overlay": pick("uv_overlay", "uv_find")},
                "thermal": {"max": pick("thermal_max", "thermal_peak"),
                            "anomaly": pick("thermal_anomaly", "thermal_find", "thermal_diff")},
                "photo": pick("photo_front", "thumbnail", "photo"),
            },
        }
