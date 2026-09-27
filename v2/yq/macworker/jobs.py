"""Background jobs on the MacBook worker (CONTRACTS.md §3.2).

A domain registers a handler per job kind:

    from yq.macworker.jobs import jobs
    def generate(ctx):                 # ctx: JobContext
        ctx.progress(0.1, "Imaginando...")
        (ctx.out_dir / "image.png").write_bytes(...)
        return {"image": "image.png"}  # JSON result; file names are relative to out_dir
    jobs.register("image", generate, heavy=True)

Heavy jobs (big models, reconstruction) run one at a time on a single worker so
two of them never fight for the 16 GB of unified memory; light jobs run on a
small pool. Every job gets JOBS_DIR/<id>/in (uploaded files) and /out.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from yq.common import config
from yq.common.contracts import new_id

log = logging.getLogger("yq.jobs")


@dataclass
class JobContext:
    id: str
    kind: str
    params: dict
    in_dir: Path
    out_dir: Path
    _job: "Job" = None

    def progress(self, fraction: float, message: str = "") -> None:
        self._job.progress = max(0.0, min(1.0, float(fraction)))
        if message:
            self._job.message = message
        self._job.updated = time.time()

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    cancel_requested: bool = False
    folder: Optional[Path] = None

    def public(self) -> dict:
        files = []
        out = self.folder / "out" if self.folder else None
        if out and out.is_dir():
            files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "progress": round(self.progress, 4), "message": self.message,
                "result": self.result, "error": self.error, "files": files,
                "created": self.created, "updated": self.updated}


class JobManager:
    def __init__(self, root: Path = None, light_workers: int = 3, keep: int = 200):
        self.root = Path(root or config.JOBS_DIR)
        self._handlers: dict[str, tuple[Callable[[JobContext], dict], bool]] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._heavy = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job-heavy")
        self._light = ThreadPoolExecutor(max_workers=light_workers, thread_name_prefix="job-light")
        self.keep = keep

    def register(self, kind: str, handler: Callable[[JobContext], dict], heavy: bool = True) -> None:
        self._handlers[kind] = (handler, heavy)

    def kinds(self) -> list[str]:
        return sorted(self._handlers)

    def submit(self, kind: str, params: dict, uploads: Optional[list] = None) -> Job:
        """uploads: list of (filename, bytes) placed in the job's in/ folder."""
        if kind not in self._handlers:
            raise KeyError("unknown job kind %r (known: %s)" % (kind, self.kinds()))
        job = Job(id=new_id(kind), kind=kind, params=params or {})
        job.folder = self.root / job.id
        (job.folder / "in").mkdir(parents=True, exist_ok=True)
        (job.folder / "out").mkdir(parents=True, exist_ok=True)
        for name, data in uploads or []:
            safe = Path(name).name
            (job.folder / "in" / safe).write_bytes(data)
        (job.folder / "params.json").write_text(json.dumps(params or {}, ensure_ascii=False, indent=1))
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        handler, heavy = self._handlers[kind]
        (self._heavy if heavy else self._light).submit(self._run, job, handler)
        return job

    def run_inline(self, kind: str, params: dict, uploads: Optional[list] = None) -> Job:
        """Run synchronously (tests, CLI). Same folders and semantics as submit()."""
        job = Job(id=new_id(kind), kind=kind, params=params or {})
        job.folder = self.root / job.id
        (job.folder / "in").mkdir(parents=True, exist_ok=True)
        (job.folder / "out").mkdir(parents=True, exist_ok=True)
        for name, data in uploads or []:
            (job.folder / "in" / Path(name).name).write_bytes(data)
        with self._lock:
            self._jobs[job.id] = job
        self._run(job, self._handlers[kind][0])
        return job

    def _run(self, job: Job, handler) -> None:
        if job.cancel_requested:
            job.status = "cancelled"
            return
        job.status = "running"
        job.updated = time.time()
        ctx = JobContext(job.id, job.kind, job.params, job.folder / "in", job.folder / "out", job)
        t0 = time.time()
        try:
            result = handler(ctx) or {}
            job.result = result
            job.status = "cancelled" if job.cancel_requested else "done"
            job.progress = 1.0
        except Exception as e:
            job.status = "error"
            job.error = "%s: %s" % (type(e).__name__, e)
            log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        job.updated = time.time()
        log.info("job %s %s in %.1f s", job.id, job.status, time.time() - t0)
        try:
            (job.folder / "status.json").write_text(json.dumps(job.public(), ensure_ascii=False, indent=1))
        except Exception:
            pass

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.cancel_requested = True
        return True

    def _prune(self) -> None:
        if len(self._jobs) <= self.keep:
            return
        finished = sorted((j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")),
                          key=lambda j: j.updated)
        for j in finished[: len(self._jobs) - self.keep]:
            self._jobs.pop(j.id, None)
            if j.folder:
                shutil.rmtree(j.folder, ignore_errors=True)


jobs = JobManager()
