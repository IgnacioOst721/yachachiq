"""End-to-end quality evaluation of the drawing pipeline on this Mac (real models).

    cd ~/yachachiq/v2
    .venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python tools/art_eval.py \
        --out /tmp/art_eval [--stories condor-apu kuntur-qu] [--backend z-image-turbo]

For each story in tools/art_stories.py: /story/plan -> image job (generate + VLM check +
retries) -> vectorize -> both pages, through the real Mac-worker HTTP app (in-process).
Writes <out>/<story>/..., <out>/eval.json, small previews into docs/art_samples/ and prints
a Markdown table for docs/art.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "tools"))


def _local_client(tc):
    from yq.common import macclient

    class Resp:
        def __init__(self, r):
            self.r, self.status_code, self.content, self.text = r, r.status_code, r.content, r.text
            self.ok = r.is_success

        def json(self):
            return self.r.json()

        def raise_for_status(self):
            self.r.raise_for_status()

    class LocalClient(macclient.MacClient):
        def _request(self, method, path, timeout=None, **kw):
            return Resp(tc.request(method, path, **kw))

    return LocalClient(urls=["http://in-process"])


def _small_png(src: Path, dst: Path, max_bytes: int = 300_000) -> None:
    from PIL import Image
    im = Image.open(src).convert("L")
    w = 620
    while True:
        im2 = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        im2.save(dst, optimize=True)
        if dst.stat().st_size <= max_bytes or w < 250:
            return
        w = int(w * 0.85)


def main() -> int:
    from art_stories import STORIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/art_eval")
    ap.add_argument("--stories", nargs="*", default=[])
    ap.add_argument("--backend", default="")
    ap.add_argument("--samples", default=str(V2 / "docs" / "art_samples"))
    a = ap.parse_args()
    import os
    if a.backend:
        os.environ["YQ_ART_IMAGE_BACKEND"] = a.backend
    from fastapi.testclient import TestClient
    from yq.common import config, macclient
    from yq.common.contracts import StoryInput
    from yq.art.drawing import make_drawing
    from yq.macworker.app import create_app
    from yq.macworker.jobs import jobs
    config.ensure_dirs()
    jobs.root = config.JOBS_DIR
    macclient._client = _local_client(TestClient(create_app()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = Path(a.samples)
    samples.mkdir(parents=True, exist_ok=True)
    rows = []
    for st in STORIES:
        if a.stories and st["id"] not in a.stories:
            continue
        s = StoryInput(story_id="eval-" + st["id"], source="text", text=st["text"], lang=st["lang"],
                       text_es=st.get("text_es", ""), confirmed=True)
        t0 = time.time()
        r = make_drawing(s, out / st["id"], use_mac=True)
        rep = json.loads((out / st["id"] / "drawing.json").read_text())
        plan = json.loads((out / st["id"] / "plan.json").read_text())
        row = {"id": st["id"], "lang": st["lang"], "source": rep["source"], "expect": st["expect"],
               "elements": plan["elements"], "verified": r.verified, "attempts": r.attempts,
               "score": rep.get("score"), "strokes_front": rep["front"]["strokes"],
               "pen_m": round(r.pen_mm / 1000.0, 1), "travel_front_m": round(rep["front"]["travel_mm"] / 1000, 2),
               "travel_front_naive_m": round(rep["front"]["travel_naive_mm"] / 1000, 2),
               "est_min": r.est_minutes, "front_min": round(rep["front"]["seconds"] / 60, 1),
               "back_min": round(rep["back"]["seconds"] / 60, 1), "times": rep["times"],
               "total_s": round(time.time() - t0, 1), "title": plan["title"], "notes": plan["cultural_notes"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        kind = "ai" if rep["source"].startswith("mac") else "fallback"     # never mix the two up
        for side, limit in (("front", 300_000), ("back", 300_000), ("image", 200_000)):
            _small_png(out / st["id"] / (side + ".png"), samples / ("%s_%s_%s.png" % (st["id"], kind, side)), limit)
    (out / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n| Historia | Idioma | Elementos verificados (VLM) | Intentos | Trazos | Tinta (m) | Viaje sin tinta (m) optimizado / ingenuo | Dibujo est. (min) frente+dorso | Tiempos (s) plan / imagen / verificación / trazado / total |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        v = r["verified"] or {}
        ok = "%d/%d" % (sum(1 for x in v.values() if x), len(v)) if v else "-"
        t = r["times"]
        print("| %s | %s | %s (%s) | %s | %d | %.1f | %.2f / %.2f | %.1f + %.1f | %s / %s / %s / %s / %.0f |" % (
            r["id"], r["lang"][:3], ok, ", ".join(k for k, x in v.items() if x) or "-", r["attempts"],
            r["strokes_front"], r["pen_m"], r["travel_front_m"], r["travel_front_naive_m"], r["front_min"],
            r["back_min"], t.get("plan_s", "-"), t.get("generate_s", "-"), t.get("verify_s", "-"),
            t.get("front_s", "-"), r["total_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
