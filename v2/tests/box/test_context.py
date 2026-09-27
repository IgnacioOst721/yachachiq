"""Visitor context (CONTRACTS.md §9): cleaned, stored in meta.json, sent to the Mac
job and to the local analysis when it accepts it; validator accepts but does not require it."""
import json

from yq.box import layout
from yq.common import config
from yq.common.contracts import ScanRequest, new_id

RAW = {"found_where": "  en una huaca cerca de Trujillo " + "x" * 400, "region_hint": "costa_norte",
       "notes": "lo trajo mi abuela", "lang": "spa_Latn", "gps": [-8.1, -79.0], "evil": {"a": 1}}


def test_clean_context_keeps_only_allowed_keys_and_limits():
    c = layout.clean_context(RAW)
    assert set(c) == {"found_where", "region_hint", "notes", "lang"}
    assert c["found_where"].startswith("en una huaca") and len(c["found_where"]) == 300
    assert layout.clean_context(None) == {} and layout.clean_context("Trujillo") == {}
    assert layout.clean_context({"notes": 5, "lang": "  "}) == {}


def test_validator_accepts_but_does_not_require_context(tmp_path):
    from yq.box.layout import context_problems
    assert context_problems({"found_where": "Lima", "lang": "spa_Latn"}) == []
    probs = context_problems({"found_where": "y" * 301, "gps": 1, "notes": 3})
    assert len(probs) == 3
    assert context_problems([1]) == ["meta.json: context debe ser un objeto"]


def test_context_reaches_meta_mac_job_and_local_analysis(mock_all, monkeypatch):
    from yq.box import package, scan
    sent = {}

    def fake_mac(folder, profile, analyses, on_progress=None, context=None):
        sent["mac"] = context
        return {"ok": True, "measurements": [], "artifacts": {}, "findings": [], "warnings": []}

    monkeypatch.setattr(package, "analyze_on_mac", fake_mac)
    req = ScanRequest(scan_id=new_id("scan"), profile="quick", analyses=["weight"], context=RAW)
    res = scan.run_scan(req)
    folder = config.SCANS_DIR / res.scan_id
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["context"] == layout.clean_context(RAW)
    assert sent["mac"] == layout.clean_context(RAW)
    assert layout.validate_scan_folder(folder) == []

    del meta["context"]                                   # still valid without it
    (folder / "meta.json").write_text(json.dumps(meta))
    assert layout.validate_scan_folder(folder) == []


def test_mac_job_params_include_context(monkeypatch, tmp_path):
    from yq.box import package
    calls = {}

    class FakeClient:
        def available(self):
            return True

        def submit_job(self, kind, params, files=None):
            calls["kind"], calls["params"] = kind, params
            return "job-1"

        def wait_job(self, job, on_progress=None):
            return {"ok": True, "artifacts": {}, "findings": []}

    import yq.common.macclient as mc
    monkeypatch.setattr(mc, "client", lambda: FakeClient())
    (tmp_path / "meta.json").write_text("{}")
    package.analyze_on_mac(tmp_path, "quick", ["weight"], context={"found_where": "Cusco"})
    assert calls["kind"] == "scan_analyze" and calls["params"]["context"] == {"found_where": "Cusco"}


def test_local_analysis_gets_context_only_if_it_accepts_it(monkeypatch):
    import sys
    import types
    from yq.box import package
    got = {}

    def with_ctx(folder, on_progress=None, identify=True, reconstruct=True, context=None):
        got["with"] = context
        return {"ok": True}

    def without_ctx(folder, on_progress=None, identify=True, reconstruct=True):
        got["without"] = True
        return {"ok": True}

    for fn in (with_ctx, without_ctx):
        mod = types.ModuleType("yq.box.analysis")
        mod.analyze_scan = fn
        monkeypatch.setitem(sys.modules, "yq.box.analysis", mod)
        package.analyze_locally("x", None, context={"region_hint": "sierra_sur"})
    assert got == {"with": {"region_hint": "sierra_sur"}, "without": True}
