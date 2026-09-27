"""Refresh the `prototypes` field of existing letter models (no retraining).

    .venvs/sign/bin/python training/sign/patch_prototypes.py prl ase
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from letters_data import SOURCES, load_v1_csv, medoid_prototypes  # noqa: E402
from yq.sign import modelstore  # noqa: E402

for lang in sys.argv[1:]:
    d = load_v1_csv(SOURCES[lang])
    for base in (modelstore.sign_dir(), modelstore.PACKAGE_MODELS):
        p = base / ("letters_%s.json" % lang)
        if p.exists():
            meta = json.loads(p.read_text())
            meta["prototypes"] = medoid_prototypes(d["q"], d["y"])
            p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            print("patched", p)
