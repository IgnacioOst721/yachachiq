# Yachachiq v2: contracts between modules

This file is the single source of truth for how the parts of Yachachiq v2 talk to
each other. Code in one domain may only rely on another domain through what is
written here. If a contract must change, say so in your final report; do not
silently change another domain's files.

Yachachiq is a WRO 2026 Future Innovators robot ("Robots Meet Culture", team from
FDR, Lima, Peru). A visitor tells a story by **voice (any language)** or in
**sign language (ASL, Peruvian Sign Language LSP, International Sign)**; the
robot understands it, plans a picture, generates it, and a pen printer built by
teammate Joaquín draws it on paper (front: drawing; back: the story text and a QR
to the web gallery). A hologram (also Joaquín's) plays the story with sound.
Separately, visitors put an archaeological object in the **analysis box**: the
robot weighs it, photographs it all around, relights it (RTI), looks at it under
UV, heats it gently and films it with a thermal camera, reconstructs it in 3D
and identifies it (type, material, culture, period) fully offline.

## 0. Ground rules (every domain)

1. **Offline at runtime.** Nothing may need internet during the competition.
   Downloading models/datasets while developing is fine; document every download
   in your domain doc so it can be repeated.
2. **Open source only** (WRO FI rule 5.3: software must be the team's own or
   open source / freely available; no paid APIs). Record every model/dataset/library
   and its license in `v2/docs/licenses_<domain>.md`. Prefer Apache-2.0/MIT/BSD;
   CC-BY-NC weights are acceptable for the competition but must be flagged.
3. **Python 3.10 compatible** (Jetson = JetPack 6, Ubuntu 22.04, Python 3.10).
   `from __future__ import annotations` everywhere. No 3.11+ syntax.
4. **Heavy imports are lazy** (inside functions). Every module must import and its
   non-heavy tests must pass on a plain Mac with only `requirements-common.txt`.
5. **Every subsystem has a mock** selected by `yq.common.config.mock("<name>")`
   (env `YQ_MOCK=1` or `YQ_MOCK_<NAME>=1`) so the whole robot runs with no
   hardware and no models. Real code paths must still be exercised by tests
   (simulators, synthetic data), not only the mocks.
6. **Tests** live in `v2/tests/<domain>/`. Default `pytest` runs fast tests only.
   Tests that load real models are `@pytest.mark.heavy`, tests that need the real
   hardware `@pytest.mark.hardware`. Run heavy things inside the memory lock:
   `.venvs/<domain>/bin/python -m yq.common.heavylock <cmd...>` (the MacBook has
   16 GB shared by everyone working in parallel).
7. **Accuracy is the point.** Measure it and write the numbers (with how you
   measured) in your domain doc. Never claim accuracy you did not measure. When
   the model is unsure, the UI must be able to show alternatives and let the
   visitor confirm (humans in the loop beat silent mistakes).
8. **Settings**: shared ones are in `yq/common/config.py` (read only for you).
   Domain settings go in `yq/<domain>/settings.py` using `yq.common.config.env`
   (env var `YQ_<NAME>`).
9. **Git**: work on the files of your domain only. Do NOT run any git command that
   writes (commit, add, stash, checkout, reset, push, branch). The lead integrates
   and commits. Never touch `robot/`, `lsp/`, `docs/`, `pi/`, `mac/`, `arduino/`
   (v1, still in use); you may read them and copy/improve code from them.
10. **Python env**: your own venv at `v2/.venvs/<domain>` (uv, Python 3.10), created
    with `uv venv --python 3.10 .venvs/<domain>` and
    `uv pip install --python .venvs/<domain>/bin/python -r requirements-common.txt`.
    List your extra dependencies in `v2/requirements/<domain>-jetson.txt` and
    `v2/requirements/<domain>-mac.txt`.
11. **Docs for Ignacio** (a beginner in Python/electronics, reads Spanish): write
    `v2/docs/<domain>.md` in simple Spanish: what it does, how to install on the
    Jetson / Mac, how to run, how to test, how to calibrate, measured accuracy,
    known limits.
12. UI text shown to visitors is Spanish first (plus English and Quechua where the
    UI domain provides them). Code, identifiers and comments in English.

## 1. Machines and network

| Machine | Runs |
|---|---|
| **Jetson Orin Nano Super 8 GB** (Yahboom kit, official NVIDIA carrier; DisplayPort; JetPack 6) | Kiosk server + UI (Chromium kiosk on the 13.3" touch screen via DP-to-HDMI), microphone/speaker (KAYSUDA USB speakerphone), sign camera (Arducam OV9782 global shutter USB) + pose model on the GPU, box control (ESP32 over USB serial), box cameras (2x Arducam IMX519 16 MP USB, autofocus), thermal camera (PureThermal 3 + FLIR Lepton 3.5, UVC Y16 radiometric), printer and hologram clients, web publishing |
| **MacBook Air/Pro M4, 16 GB** | `yq.macworker` HTTP server on port 8700: big speech models, translation, LLM, VLM, image generator, 3D reconstruction, object identification, offline museum catalog |
| **Printer** (Joaquín) | Pen printer under our section; receives SVG/G-code jobs (§8) |
| **Hologram** (Joaquín) | Looking Glass Go + speakers; receives story packages (§8) |

All on the GL.iNet Beryl AX router (LAN 192.168.8.0/24, fixed leases: Jetson .10,
Mac .20, printer .30, hologram .40). Use `yq.common.macclient.client()` to reach
the Mac; it raises `MacUnavailable` fast so callers can degrade gracefully.

## 2. Package layout and ownership

```
v2/
  CONTRACTS.md  pytest.ini  requirements-common.txt  requirements/  docs/
  yq/common/        lead: config, contracts, macclient, heavylock; voice owns languages.py
  yq/macworker/     lead: app.py, jobs.py, modelmgr.py
                    each domain: routes_<domain>.py + models/<domain>_*.py
  yq/voice/         VOICE   (Jetson side of speech)
  yq/sign/          SIGN    (Jetson side of sign language) + v2/training/sign/
  yq/box/           BOX-CAPTURE: device.py, sim.py, cameras.py, thermal.py, scan.py, cli.py
  yq/box/analysis/  BOX-ANALYSIS: rti, uv, thermo, measure, recon, identify, catalog, calib
  yq/art/           ART     (drawing: plan, image, vectorize, text, QR, G-code/SVG)
  yq/server/        UI      (Jetson FastAPI app, kiosk SPA in static/, flows)
  yq/printer/ yq/hologram/ yq/publish/   UI
  firmware/box_esp32/   BOX-CAPTURE (PlatformIO, Arduino framework)
  tools/                any domain, prefixed with the domain name
  tests/<domain>/
```

## 3. Mac worker API (port 8700)

Generic (lead, already implemented and tested):
- `GET /health` -> `{"ok", "domains": {module: "ok"|error}, "job_kinds", "models": stats}`
- `POST /jobs` multipart: `kind`, `params` (JSON string), `files` (0..n) -> `{"id"}`
- `GET /jobs/{id}` -> `{"id","kind","status": queued|running|done|error|cancelled, "progress": 0..1, "message", "result", "error", "files": [...]}`
- `GET /jobs/{id}/files/{name}`; `POST /jobs/{id}/cancel`; `POST /models/unload`

A domain registers models with `yq.macworker.modelmgr.models.register(name, loader, size_gb, unloader)`
and gets them with `models.get(name)` (LRU eviction under a 10.5 GB budget).
A domain registers jobs with `jobs.register(kind, handler, heavy=True)`;
`handler(ctx) -> dict` gets `ctx.params`, `ctx.in_dir`, `ctx.out_dir`,
`ctx.progress(fraction, message_es)`, `ctx.cancelled`.

### 3.1 Voice (`routes_voice.py`)
- `POST /asr` multipart `audio` (WAV 16 kHz mono), `lang` (canonical code or `"auto"`), `prompt` (optional context) -> `Transcript`
- `POST /lid` multipart `audio` -> `{"candidates": [[code, prob], ...]}`
- `POST /translate` JSON `{"text","src","tgt"}` -> `{"text","engine"}`
- `POST /tts` JSON `{"text","lang","voice"?}` -> `audio/wav` bytes
- Python API on the Mac: `yq.macworker.models.voice_translate.translate(text, src, tgt) -> str`

### 3.2 Art (`routes_art.py`)
- `POST /story/clean` JSON `{"text","lang"}` -> `{"text"}` (spelling/punctuation, meaning unchanged)
- `POST /story/plan` JSON `{"text","lang","text_es"?,"text_en"?}` -> `{"plan": ScenePlan, "text_es", "text_en"}`
- job `image`: params `{"plan": ScenePlan, "width", "height", "seed"?, "max_attempts"?, "verify": true}` ->
  result `{"image": "image.png", "attempts", "verified": {element: bool}, "score"}`
- Python API on the Mac (owned by ART, used by BOX-ANALYSIS too):
  - `yq.macworker.models.llm.chat(messages: list[dict], max_tokens=800, json_mode=False, temperature=0.2) -> str`
  - `yq.macworker.models.vlm.ask(images: list, prompt: str, max_tokens=600) -> str` (images: paths or PIL images)

### 3.3 Box analysis (`routes_box.py`)
- job `scan_analyze`: files = the scan folder zipped as `scan.zip` (layout §4, images may be
  downscaled by the Jetson); params `{"profile","analyses":[...],"lang":"spa_Latn"}` ->
  result = `ScanResult` as dict (artifact paths relative to the job's out/, which also
  contains the artifacts; the Jetson downloads them into `scans/<id>/analysis/`).
  params may include `"context"` = `ScanRequest.context` (see §9).
- job `identify` (also usable alone): files = images; params `{"measurements": [...], "notes"}` -> `Identification`

## 4. Scan folder layout (BOX-CAPTURE writes, BOX-ANALYSIS reads)

```
scans/<scan_id>/
  meta.json            {"scan_id","profile","analyses","context","started","finished","box":{firmware, platter_deg},
                        "cameras":{"A":{id, resolution, focus, exposure_us, gain, wb_k},...},
                        "calibration":{"intrinsics": "...", "extrinsics": "...", "rti_lights": "...", "thermal_reg": "..."},
                        "door_closed": true, "warnings": []}
  weight.json          {"grams","sigma_g","samples":[...],"tare_g","stable":true,"calibration_factor"}
  photogrammetry/      camA_000.jpg camA_010.jpg ... camB_000.jpg ...  (angle in degrees, 3 digits)
                       background_camA.jpg background_camB.jpg (empty platter, same settings, when available)
                       poses.json {"camA_000.jpg": {"platter_deg": 0.0}, ...}
  rti/                 led1.jpg ... led8.jpg, ambient.jpg (all lights off), lights.json (index, position_mm, direction)
  uv/                  uv.jpg (UV LED only), visible.jpg (COB only), dark.jpg (all off), exposure.json
  thermal/             sequence.npy (T x 120 x 160 float32, degrees C), times.npy (s), meta.json
                       {"heat_on_s","heat_off_s","halogen":"MR16 35W","ambient_c","fps"}
  analysis/            (BOX-ANALYSIS outputs, copied back from the Mac)
  result.json          ScanResult as dict (final)
```

Coordinates: platter centre is the origin, Z up, millimetres; platter top surface at
Z=0 of the object frame. Camera/LED positions come from `CALIB_DIR` (measured with
the calibration tools), with the CAD R1 nominal values as defaults.

## 5. ESP32 serial protocol (BOX-CAPTURE defines precisely in `firmware/box_esp32/PROTOCOL.md`)

JSON lines at 115200 baud. Host -> ESP32 `{"id":n,"cmd":"...", ...}`; ESP32 -> host
replies `{"id":n,"ok":true|false,...}` and events `{"event":"door","closed":false}`.
The firmware is the last line of safety: all outputs off at boot; UV and halogen
refuse to turn on and are cut within 10 ms when a door interlock opens; every light
channel has a maximum on-time and cool-down; no host heartbeat for 3 s = all off.

## 6. Python interfaces on the Jetson (what the UI calls)

VOICE (`yq.voice`, `yq.common.languages`)
- `languages.all() -> list[Language]`, `languages.get(code)`, `languages.ui_list(feature="asr") -> list[dict]`
  where `Language` has `code, iso639_3, name_native, name_es, name_en, region, asr: list[str], tts: list[str], translate: bool, whisper_code, popular: int`
- `capture.Recorder().record(on_level=None, max_s=90.0, stop_event=None) -> numpy float32 16 kHz mono` (VAD endpointing)
- `asr.transcribe(audio, lang="auto") -> Transcript` (Mac best model -> local Jetson model -> mock)
- `tts.say(text, lang, wait=False) -> bool`, `tts.stop()`
- `translate.translate(text, src, tgt) -> str` (Mac; falls back to returning `text`)

SIGN (`yq.sign`)
- `yq.sign.available() -> list[{"code","name_es","letters":bool,"words":int}]`
- `SignEngine(sign_lang, camera=None, on_token=None, on_status=None)` with `start()`, `stop()`,
  `set_mode("letters"|"words")`, `latest_jpeg() -> bytes|None` (preview with skeleton), `state() -> dict`
  (`hands_visible, fps, buffer, candidates`), `accept(index=0)`, `backspace()`, `clear()`, `text() -> str`

ART (`yq.art`)
- `drawing.make_drawing(story: StoryInput, out_dir: Path, on_progress=None) -> DrawingResult`
  (Mac plan + image + verify when available; offline procedural fallback otherwise)

BOX (`yq.box`)
- `scan.preflight() -> {"ok": bool, "problems_es": [...], "weight_g": float|None}`
- `scan.run_scan(req: ScanRequest, on_progress=None, cancel_event=None) -> ScanResult`
  (captures, then analyzes on the Mac, else the light analyses locally)
- `device.get_box()` -> the `Box` driver (real or simulated)

`on_progress(Progress)` callbacks are called from worker threads; the UI forwards them over its WebSocket.

## 7. Kiosk (UI domain)

Jetson FastAPI on port 8877, SPA in `yq/server/static/` (vanilla ES modules, no CDN,
everything vendored for offline). WebSocket `/ws` pushes `{"type": ..., ...}` events
defined by the UI domain in `docs/ui.md`.

## 8. Printer and hologram packages

Printer job (to Joaquín's printer, `PRINTER_URL`): `POST /jobs` multipart with
`front.svg`, `back.svg`, optional `front.gcode`, `back.gcode`, and `job.json`
`{"story_id","paper":{"w_mm","h_mm"},"pens":[...],"flip":"long-edge","qr_url"}`.
SVG in millimetres (`width="210mm"`), one `<path>` per stroke, groups per pen.

Hologram package (`HOLOGRAM_URL`): `POST /stories` multipart with `story.json`
`{"story_id","title","lang","text","text_es","scenes":[{"image","caption_es","audio"}]}`
plus the referenced images and WAV narration files.

Both interfaces are proposals until Joaquín confirms; clients must be tolerant and
have mocks.

## 9. Visitor context for identification (added 2026-09-27, Ignacio's idea)

Before the scan the kiosk asks, optionally, "¿Dónde lo encontraron?": a short free
text (on-screen keyboard or voice dictation, any language) plus quick region chips
(Costa norte, Costa central, Costa sur, Sierra norte, Sierra central, Sierra sur,
Altiplano, Selva, Lima, Otro país, No sé). It travels as `ScanRequest.context`
`{"found_where", "region_hint", "notes", "lang"}` → `meta.json["context"]` → job
params `"context"`.

Identification uses it as a CLUE, never as proof:
1. Always identify from the object alone first (`Identification.image_only`).
2. Parse the context into structured hints (country, region, site, altitude zone)
   and apply a bounded soft prior to the retrieval voting (it may reorder close
   candidates, it may not create a culture the images do not support).
3. The VLM sees the context labelled "lugar reportado por el visitante; puede ser
   incorrecto".
4. Report `context_effect_es`: whether the place helped decide, changed nothing, or
   contradicts what is seen (then the image wins and the UI says so).
