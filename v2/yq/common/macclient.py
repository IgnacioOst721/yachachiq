"""Jetson-side client for the MacBook worker (yq.macworker, CONTRACTS.md §3).

Tries the Mac by mDNS name first and by fixed IP second, remembers which one
answered, and raises MacUnavailable quickly when neither does so callers can
fall back to local/offline behaviour instead of hanging the kiosk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from . import config


class MacUnavailable(RuntimeError):
    pass


class MacJobError(RuntimeError):
    pass


class MacClient:
    def __init__(self, urls: Optional[list] = None, timeout: float = None):
        self.urls = list(urls or config.mac_urls())
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT
        self._good: Optional[str] = None
        self._last_probe = 0.0
        self._last_ok = False

    # -- plumbing ---------------------------------------------------------------
    def _session(self):
        import requests
        return requests

    def _order(self) -> list:
        if self._good:
            return [self._good] + [u for u in self.urls if u != self._good]
        return list(self.urls)

    def _request(self, method: str, path: str, timeout: float = None, **kw):
        if config.mock("mac"):
            raise MacUnavailable("mac mocked (YQ_MOCK_MAC)")
        req = self._session()
        last = None
        for base in self._order():
            try:
                r = req.request(method, base + path, timeout=timeout or self.timeout, **kw)
                self._good = base
                return r
            except Exception as e:  # connection refused, DNS, timeout
                last = e
        self._good = None
        raise MacUnavailable("Mac worker unreachable at %s (%s)" % (self.urls, last))

    # -- health ----------------------------------------------------------------------
    def available(self, max_age_s: float = 5.0) -> bool:
        """Cheap cached health check."""
        now = time.time()
        if now - self._last_probe < max_age_s:
            return self._last_ok
        self._last_probe = now
        try:
            self._last_ok = self._request("GET", "/health", timeout=2.0).ok
        except MacUnavailable:
            self._last_ok = False
        return self._last_ok

    def health(self) -> dict:
        r = self._request("GET", "/health", timeout=3.0)
        r.raise_for_status()
        return r.json()

    # -- synchronous calls ------------------------------------------------------------
    def post_json(self, path: str, payload: dict, timeout: float = None) -> dict:
        r = self._request("POST", path, json=payload, timeout=timeout)
        if not r.ok:
            raise MacJobError("%s -> HTTP %d: %s" % (path, r.status_code, r.text[:500]))
        return r.json()

    def post_files(self, path: str, files: dict, data: Optional[dict] = None, timeout: float = None) -> dict:
        """files: {"field": (filename, bytes_or_fileobj, content_type)}"""
        r = self._request("POST", path, files=files, data=data or {}, timeout=timeout)
        if not r.ok:
            raise MacJobError("%s -> HTTP %d: %s" % (path, r.status_code, r.text[:500]))
        return r.json()

    def get_bytes(self, path: str, timeout: float = None) -> bytes:
        r = self._request("GET", path, timeout=timeout)
        if not r.ok:
            raise MacJobError("%s -> HTTP %d" % (path, r.status_code))
        return r.content

    # -- jobs (long work: images, identification, reconstruction) -----------------------
    def submit_job(self, kind: str, params: dict, files: Optional[list] = None) -> str:
        """files: list of local paths uploaded into the job's input folder (names kept)."""
        handles = []
        try:
            multipart = [("kind", (None, kind)), ("params", (None, json.dumps(params)))]
            for p in files or []:
                fh = open(p, "rb")
                handles.append(fh)
                multipart.append(("files", (Path(p).name, fh, "application/octet-stream")))
            r = self._request("POST", "/jobs", files=multipart, timeout=max(self.timeout, 120.0))
        finally:
            for fh in handles:
                fh.close()
        if not r.ok:
            raise MacJobError("submit %s -> HTTP %d: %s" % (kind, r.status_code, r.text[:500]))
        return r.json()["id"]

    def job(self, job_id: str) -> dict:
        r = self._request("GET", "/jobs/%s" % job_id, timeout=5.0)
        r.raise_for_status()
        return r.json()

    def wait_job(self, job_id: str, on_progress: Optional[Callable[[float, str], None]] = None,
                 timeout: float = None, poll_s: float = 0.5) -> dict:
        """Block until the job finishes; returns its `result` dict or raises MacJobError."""
        deadline = time.time() + (timeout or config.JOB_TIMEOUT)
        while time.time() < deadline:
            st = self.job(job_id)
            if on_progress:
                on_progress(float(st.get("progress") or 0.0), st.get("message") or "")
            if st["status"] == "done":
                return st.get("result") or {}
            if st["status"] == "error":
                raise MacJobError(st.get("error") or "job failed")
            time.sleep(poll_s)
        raise MacJobError("job %s timed out" % job_id)

    def download(self, job_id: str, name: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get_bytes("/jobs/%s/files/%s" % (job_id, name), timeout=120.0))
        return dest


_client: Optional[MacClient] = None


def client() -> MacClient:
    global _client
    if _client is None:
        _client = MacClient()
    return _client
