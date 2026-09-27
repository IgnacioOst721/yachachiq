#!/usr/bin/env python
"""Build the offline museum catalog used to identify objects (run ONCE on the Mac, with internet).

    cd ~/yachachiq/v2
    # 1) metadata + thumbnails (hours; resumable: just run it again after a stop)
    .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py --log ~/yq-data/catalog/download.log download
    # 2) image embeddings (inside the memory lock, in chunks so others can use the Mac too)
    .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py embed
    # 3) numbers
    .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py status
    .venvs/box_analysis/bin/python tools/box_analysis_build_catalog.py eval --n 2000

Everything lands in YQ_CATALOG_DIR (default ~/yq-data/catalog).
"""
from __future__ import annotations

import argparse
import collections
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _logger(path):
    fh = open(path, "a", encoding="utf-8") if path else None

    def log(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")
            fh.flush()
    return log


def cmd_download(a) -> int:
    from yq.box.analysis.catalog import CatalogStore
    from yq.box.analysis.catalog_sources import Builder
    log = _logger(a.log)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    b = Builder(CatalogStore(a.root), log=log, retry_errors=a.retry_errors, stop=stop, workers=a.workers)
    log("download start: sources=%s limit=%s" % (a.sources, a.limit))
    counts = b.build(tuple(a.sources.split(",")), limit_per_source=a.limit)
    log("download done: %s; catalog has %d objects" % (json.dumps(counts), b.store.count()))
    return 0


def cmd_embed(a) -> int:
    from yq.box.analysis import embed
    log = _logger(a.log)
    n = embed.build_index(model_key=a.model, root=a.root, chunk=a.chunk, log=log, batch=a.batch)
    log("index ready: %d vectors" % n)
    return 0


def cmd_status(a) -> int:
    from yq.box.analysis.catalog import CatalogStore, CatalogIndex
    st = CatalogStore(a.root)
    items = st.items()
    size = sum(p.stat().st_size for p in st.root.rglob("*") if p.is_file()) if st.root.exists() else 0
    print("catalog: %s" % st.root)
    print("objects: %d   disk: %.2f GB" % (len(items), size / 1e9))
    for key, title in (("source", "by source"), ("region_norm", "by region"), ("material_cls", "by material class"),
                       ("type_cls", "by object type")):
        c = collections.Counter(it.get(key) or "?" for it in items)
        print("%s: %s" % (title, ", ".join("%s %d" % kv for kv in c.most_common(30))))
    c = collections.Counter(it.get("culture_norm") for it in items if it.get("culture_norm"))
    print("cultures (%d with a canonical culture): %s" % (sum(c.values()), ", ".join("%s %d" % kv for kv in c.most_common(80))))
    idx_root = st.root / "index"
    if idx_root.exists():
        for d in sorted(idx_root.iterdir()):
            if CatalogIndex.exists(d.name, a.root):
                meta = json.loads((d / "meta.json").read_text())
                print("index %s: %s vectors, dim %s, saved %s" % (d.name, meta.get("count"), meta.get("dim"), meta.get("saved")))
    return 0


def cmd_eval(a) -> int:
    from yq.box.analysis import evaluate
    log = _logger(a.log)
    res = evaluate.leave_one_out(model_key=a.model, root=a.root, n=a.n, seed=a.seed, log=log, vlm_n=a.vlm_n)
    out = Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res.get("summary", res), indent=1, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="catalog folder (default YQ_CATALOG_DIR)")
    p.add_argument("--log", default=None, help="also append log lines to this file")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--sources", default="met,aic,cma")
    d.add_argument("--limit", type=int, default=None, help="max new objects per source (tests)")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--retry-errors", action="store_true")
    e = sub.add_parser("embed")
    e.add_argument("--model", default=None, help="embedding model key (default: settings.EMBED_MODEL)")
    e.add_argument("--chunk", type=int, default=2000, help="images per memory-lock turn")
    e.add_argument("--batch", type=int, default=32)
    sub.add_parser("status")
    v = sub.add_parser("eval")
    v.add_argument("--model", default=None)
    v.add_argument("--n", type=int, default=2000, help="held-out objects (leave-one-object-out)")
    v.add_argument("--vlm-n", type=int, default=0, help="also run retrieval+VLM on this many (needs ART's VLM)")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default=None, help="write full results JSON here")
    a = p.parse_args(argv)
    return {"download": cmd_download, "embed": cmd_embed, "status": cmd_status, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
