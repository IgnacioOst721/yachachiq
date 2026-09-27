import time

from fastapi.testclient import TestClient

from yq.common import config
from yq.common.contracts import Transcript, from_dict, to_dict
from yq.macworker.jobs import JobManager
from yq.macworker.modelmgr import ModelManager


def test_contract_roundtrip():
    t = Transcript(text="hola", lang="spa_Latn", engine="mock", confidence=0.9)
    d = to_dict(t)
    d["unknown_future_field"] = 1
    assert from_dict(Transcript, d) == t


def test_model_manager_evicts_lru():
    mm = ModelManager(budget_gb=5)
    freed = []
    for n in ("a", "b", "c"):
        mm.register(n, loader=lambda n=n: n.upper(), size_gb=2, unloader=lambda o: freed.append(o))
    assert mm.get("a") == "A"
    mm.get("b")
    mm.get("a")          # a is now most recent
    mm.get("c")          # needs room: evicts b
    assert sorted(mm.loaded()) == ["a", "c"] and freed == ["B"]


def test_jobs_run_and_files(tmp_path):
    jm = JobManager(root=tmp_path)

    def handler(ctx):
        ctx.progress(0.5, "half")
        (ctx.out_dir / "x.txt").write_text(ctx.params["msg"] + (ctx.in_dir / "in.txt").read_text())
        return {"file": "x.txt"}

    jm.register("echo", handler, heavy=False)
    job = jm.submit("echo", {"msg": "hi "}, [("in.txt", b"there")])
    for _ in range(100):
        if jm.get(job.id).status == "done":
            break
        time.sleep(0.02)
    pub = jm.get(job.id).public()
    assert pub["status"] == "done" and pub["files"] == ["x.txt"]
    assert (job.folder / "out" / "x.txt").read_text() == "hi there"


def test_jobs_error_is_reported(tmp_path):
    jm = JobManager(root=tmp_path)
    jm.register("boom", lambda ctx: 1 / 0)
    job = jm.run_inline("boom", {})
    assert job.status == "error" and "ZeroDivisionError" in job.error


def test_worker_health_and_job_http():
    from yq.macworker.app import create_app
    from yq.macworker.jobs import jobs
    jobs.root = config.JOBS_DIR
    jobs.register("upper", lambda ctx: {"text": ctx.params["text"].upper()}, heavy=False)
    c = TestClient(create_app())
    h = c.get("/health").json()
    assert h["ok"] and "upper" in h["job_kinds"]
    jid = c.post("/jobs", data={"kind": "upper", "params": '{"text": "llama"}'}).json()["id"]
    for _ in range(100):
        st = c.get("/jobs/" + jid).json()
        if st["status"] == "done":
            break
        time.sleep(0.02)
    assert st["result"] == {"text": "LLAMA"}
    assert c.get("/jobs/" + jid + "/files/../../etc/passwd").status_code == 404


def test_mac_client_mocked(monkeypatch):
    from yq.common.macclient import MacClient, MacUnavailable
    monkeypatch.setattr(config, "MOCK", True)
    c = MacClient(urls=["http://127.0.0.1:9"])
    assert c.available() is False
    try:
        c.health()
        assert False
    except MacUnavailable:
        pass


def test_models_load_and_unload_on_the_mlx_thread():
    """MLX objects must be created and freed on ONE thread (else the worker aborts)."""
    import threading
    from yq.macworker import mlx_thread
    seen = []
    mm = ModelManager(budget_gb=3)

    def loader(n):
        seen.append(("load", n, threading.current_thread().name))
        return n

    for n in ("a", "b"):
        mm.register(n, loader=lambda n=n: loader(n), size_gb=2,
                    unloader=lambda o: seen.append(("unload", o, threading.current_thread().name)))
    t = threading.Thread(target=lambda: (mm.get("a"), mm.get("b")))   # from a request-like thread
    t.start(); t.join()
    assert [s[:2] for s in seen] == [("load", "a"), ("unload", "a"), ("load", "b")]
    assert all(s[2].startswith("mlx") for s in seen), seen
    assert mlx_thread.run(lambda: mlx_thread.run(lambda: 7)) == 7                  # re-entrant, no deadlock
