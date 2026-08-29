# Yachachiq — Where Stories Come to Life

Yachachiq (/ya-cha-CHEEK/, Quechua for a keeper of knowledge) is a desktop CNC pen
plotter that turns a spoken or signed story into a physical, hand-drawn picture book.

Built by FDR students **Joaquin Cisneros, Ignacio Osterling, and Clara Verderas**
as our Coding & Robotics final project and WRO 2026 Future Innovators entry.

## How it works

```
 Voice (mic)  ──► whisper.cpp ──┐
                                ├──► scene extraction ──► ComfyUI (Stable Diffusion)
 ASL (camera) ──► ML model  ────┘            │                    │
                                             ▼                    ▼
                                       story.txt            line-art PNG
                                                                  │
                                             OpenCV contours ◄────┘
                                                  │
                                                  ▼
                                          G-code ──► Arduino UNO (GRBL) ──► plotter
```

Three machines talk to each other over a private network:

| Machine | Role | Code |
|---|---|---|
| **Raspberry Pi 5** | Conductor: records the story, transcribes it, converts images to G-code, streams to the plotter | [`pi/`](pi/) |
| **Mac (GPU)** | Image generation server: receives scene text, runs ComfyUI / Stable Diffusion, returns a PNG | [`mac/`](mac/) |
| **Arduino UNO** | Runs GRBL firmware, drives the NEMA 17 steppers and the pen servo | GRBL (see below) |

## Repository contents

### `pi/` — Raspberry Pi pipeline
- **`listen.py`** — main loop: records 30 s of audio, transcribes with whisper.cpp,
  extracts scenes, and pipelines image generation + plotting in parallel threads.
- **`image_to_gcode.py`** — OpenCV: threshold → contours → simplified paths → GRBL
  G-code (G0 travel / G1 draw), scaled to the plotter's drawing area.
- **`send_gcode.py`** — streams G-code to the Arduino over USB serial, translating
  pen up/down into servo commands (M3 S30 / M3 S90) and waiting for GRBL's `ok`.

### `mac/` — image generation server
- **`comfy_server.py`** — small HTTP server (port 5001): receives scene text from
  the Pi, simplifies the prompt, queues a ComfyUI workflow (DreamShaper checkpoint,
  coloring-book style), and sends the PNG back.
- **`comfyui_main.py`** — *not our code*: this is ComfyUI's own `main.py` entry
  point, included for reference because our dossier links to it. ComfyUI is
  open source: https://github.com/comfyanonymous/ComfyUI

### `asl/` — camera-based ASL recognition
- **`asl_app.py`** — live sign-to-text app: MediaPipe hand landmarks → trained
  classifier → stability filter → text, sent over a socket to the pipeline.
- **`collect_data.py`** / **`train_model.py`** — record landmark samples per letter
  and train the model (`model.pkl`, ~96% accuracy — not committed, too large).
- **`landmark_utils.py`** — shared feature extraction (hand shape, fingertip
  positions, motion features for I vs J).
- **`translate.py`**, **`receiver.py`**, **`hello_hands.py`** — earlier/demo tools.

The training data (CSV) and trained model (`model.pkl`) are not in the repo because
of their size; run `collect_data.py` + `train_model.py` to rebuild them.

## Firmware

The Arduino UNO runs **GRBL** — we used the standard GRBL upload sketch.
Earlier prototypes used [GRBL-28byj-48](https://github.com/TGit-Tech/GRBL-28byj-48);
the final machine uses NEMA 17 steppers on DRV8825 drivers with a CNC shield.

## Credits

Team Yachachiq — Joaquin Cisneros (team lead), Ignacio Osterling (lead engineer,
ASL system), Clara Verderas (lead programmer, pipeline). Thanks to Ms. Castro,
Mr. Kirsch, and Museo de Arte de Lima.
