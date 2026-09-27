"""BOX-ANALYSIS jobs on the Mac worker (CONTRACTS.md §3.3).

job scan_analyze: files = scan.zip (the scan folder, §4); params {"profile", "analyses", "lang", "context"?}
                  -> ScanResult dict; artifacts in the job's out/ (paths relative to out/).
job identify:     files = object photos; params {"measurements", "notes", "context"?} -> Identification dict.
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def _safe_extract(zpath: Path, dest: Path) -> Path:
    dest = dest.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError("ruta insegura en scan.zip: %s" % m.filename)
        z.extractall(dest)
    metas = sorted(dest.rglob("meta.json"), key=lambda p: len(p.parts))
    return metas[0].parent if metas else dest


def scan_analyze(ctx) -> dict:
    import json
    from yq.box.analysis import analyze_scan
    from yq.common.contracts import to_dict
    zips = sorted(ctx.in_dir.glob("*.zip"))
    if not zips:
        raise ValueError("falta scan.zip")
    folder = _safe_extract(zips[0], ctx.in_dir / "scan")
    p = ctx.params or {}
    if p.get("context"):
        mp = folder / "meta.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        meta.setdefault("context", p["context"])
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    def prog(pr):
        ctx.progress(pr.fraction, pr.message_es)
    res = analyze_scan(folder, on_progress=prog, identify="identify" in (p.get("analyses") or ["identify"]),
                       lang=p.get("lang", "spa_Latn"), out_dir=ctx.out_dir, rel_to=ctx.out_dir,
                       analyses=p.get("analyses") or None)
    return to_dict(res)


def identify_job(ctx) -> dict:
    from yq.box.analysis.identify import identify
    p = ctx.params or {}
    imgs = sorted(x for x in ctx.in_dir.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ctx.progress(0.1, "Comparando con museos del mundo...")
    return identify([str(x) for x in imgs], p.get("measurements") or [], p.get("findings") or [], p.get("notes", ""),
                    p.get("context"), ctx.out_dir)


def setup(jobs, models) -> None:
    from yq.macworker.models.box_embed import register
    register(models)
    jobs.register("scan_analyze", scan_analyze, heavy=True)
    jobs.register("identify", identify_job, heavy=True)
