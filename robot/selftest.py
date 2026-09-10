#!/usr/bin/env python3
"""Run the whole pipeline with no hardware, no models and no network, and check every stage.

    python selftest.py            # ~10 s
"""
import os
import sys
import time

os.environ["YACHACHIQ_MOCK"] = "1"
os.environ["YACHACHIQ_STORY_BACKENDS"] = "rules"
os.environ["YACHACHIQ_IMAGE_BACKENDS"] = "motifs"
os.environ["YACHACHIQ_MOCK_LINE_DELAY"] = "0"

import config                      # noqa: E402
import gcode                       # noqa: E402
import motifs                      # noqa: E402
import story                       # noqa: E402
import vectorize                   # noqa: E402
from language import detect_language, find_elements   # noqa: E402
from pipeline import Pipeline      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


print("1. language")
lang, words = detect_language("Ñuqa rimani runa simi, tayta inti killa pacha yaku")
check(lang == "quechua", f"quechua detection -> {lang} {words}")
els = find_elements("Un cóndor volaba sobre las montañas mientras una niña y su llama iban al río bajo el sol")
check(set(els) >= {"condor", "mountain", "person", "llama", "river", "sun"}, f"elements -> {els}")

print("2. motifs -> strokes -> gcode")
for key in motifs.all_keys():
    pls = motifs.compose([key], config.PAPER_W_MM, config.PAPER_H_MM)
    check(len(pls) > 3, f"motif '{key}' composes ({len(pls)} strokes)")
pls = motifs.compose(els, config.PAPER_W_MM, config.PAPER_H_MM)
pls = vectorize.clamp(vectorize.order_strokes(pls))
inside = all(0 <= x <= config.PAPER_W_MM and 0 <= y <= config.PAPER_H_MM for pl in pls for x, y in pl)
check(inside, "all strokes inside the paper")
lines = gcode.from_polylines(pls, "test")
check(lines[1] == "G21" and any(l.startswith("G1 X") for l in lines), f"gcode has {len(lines)} lines")
parked = any(l.startswith("G0 X0 Y0") for l in lines[-7:])          # ...then the optional home sweep
check(parked and (lines[-1] in ("G54", "M5") or lines[-1].startswith("G0 X0 Y0")), "gcode ends by parking at origin (+ home sweep)")
st = vectorize.stats(pls)
check(st["seconds"] < 900, f"estimated plot time {st['seconds']} s")
svg = vectorize.to_svg(pls)
check(svg.count("<polyline") == len(pls), "svg preview has one polyline per stroke")

print("3. image tracing (synthetic drawing)")
try:
    import cv2
    import numpy as np
    img = np.full((400, 500), 255, np.uint8)
    cv2.circle(img, (250, 200), 120, 0, 4)
    cv2.rectangle(img, (60, 60), (200, 160), 0, 4)
    cv2.line(img, (20, 380), (480, 300), 0, 5)
    p = os.path.join(str(config.OUTPUT_DIR), "selftest_shapes.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, img)
    t0 = time.time()
    traced, w, h = vectorize.trace_image(p)
    check(2 <= len(traced) <= 12, f"traced {len(traced)} strokes in {time.time() - t0:.1f}s (expect ~3)")
    mm = vectorize.fit_to_paper(traced, w, h)
    check(all(config.MARGIN_MM - 0.01 <= x <= config.PAPER_W_MM - config.MARGIN_MM + 0.01 for pl in mm for x, _ in pl), "fitted strokes respect margins")
except ImportError:
    print("  skip opencv not installed")

print("4. story rules backend")
a = story.analyze("Había una vez un puma que vivía cerca de una laguna con muchas flores.", "spanish", backends=["rules"])
check(a["backend"] == "rules" and "puma" in a["elements"] and "lake" in a["elements"], f"rules -> {a['title']!r} {a['elements']}")
check(len(a["narration"]) > 20, "narration text produced")

print("4b. comfyui workflow patching (offline)")
import imagegen                    # noqa: E402
import json                        # noqa: E402
wf = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptop", "comfyui_workflow.json")))
patched = imagegen.patch_workflow(wf, "POS", "NEG", size=640, seed=42, checkpoint="x.safetensors", steps=6)
check(patched["6"]["inputs"]["text"] == "POS" and patched["7"]["inputs"]["text"] == "NEG", "prompts patched via KSampler links")
check(patched["5"]["inputs"]["width"] == 640 and patched["3"]["inputs"]["seed"] == 42 and patched["3"]["inputs"]["steps"] == 6, "size, seed, steps patched")
check(patched["4"]["inputs"]["ckpt_name"] == "x.safetensors" and wf["4"]["inputs"]["ckpt_name"] != "x.safetensors", "checkpoint patched on a copy")

print("5. full pipeline (mock hardware)")
events = []
pipe = Pipeline(on_event=lambda e, d: events.append((e, d)))
check(pipe.modes()["plotter"] == "mock", f"modes {pipe.modes()}")
ok = pipe.submit_text("Un cóndor volaba sobre las montañas mientras una niña y su llama caminaban hacia el río.")
check(ok, "submit_text accepted")
t0 = time.time()
while pipe.state not in ("done", "error") and time.time() - t0 < 60:
    time.sleep(0.2)
check(pipe.state == "done", f"pipeline finished with state={pipe.state} in {time.time() - t0:.1f}s (error={pipe.last_error})")
seen = [e for e, _ in events]
for want in ("transcript", "analysis", "image", "progress", "narration"):
    check(want in seen, f"event '{want}' emitted")
check(len(pipe.plotter.sent) > 50, f"mock plotter received {len(pipe.plotter.sent)} lines")
check(os.path.exists(os.path.join(str(config.OUTPUT_DIR), "last.gcode")), "last.gcode written")

print("6. microphone start/stop (mock)")
pipe2 = Pipeline(on_event=lambda e, d: None)
check(pipe2.start_listening() and pipe2.state == "listening", "start_listening")
time.sleep(0.5)
check(pipe2.stop_listening(), "stop_listening runs the pipeline")
t0 = time.time()
while pipe2.state not in ("done", "error") and time.time() - t0 < 60:
    time.sleep(0.2)
check(pipe2.state == "done", f"audio path finished with state={pipe2.state}")

print()
if FAILS:
    print(f"{len(FAILS)} check(s) failed")
    sys.exit(1)
print("all checks passed")
