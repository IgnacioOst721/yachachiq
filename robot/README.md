# Yachachiq

*"The one who teaches"* (Quechua). A visitor presses a button, tells a story in Spanish or
Quechua, and the robot draws it on paper with a pen plotter while narrating it back.
Built for WRO 2026 Future Innovators, "Robots Meet Culture". FDR, Lima.

**This folder is the whole robot.** It replaces the older `pi/` + `mac/` scripts. Stories can
also be told in **Peruvian Sign Language** (`../lsp/`), the finished drawing is **photographed
automatically**, and every story is **published to the web gallery** (`../docs/`, GitHub Pages)
whenever there is a connection.

```
 touchscreen / big button
        │
   Raspberry Pi 5 ── USB ── Arduino UNO + CNC Shield (grbl-servo) ── pen plotter
        │  mic in, speaker out
        │  Whisper (speech to text)          Piper (text to speech)
        │
   Ethernet cable (optional)
        │
   Laptop: ComfyUI (Stable Diffusion line art)   [optional: Ollama for analysis]
        │
   camera 1 (signs, ../lsp/lsp_app.py)  camera 2 (photo of the drawing)
```

Everything on the Pi works with no network at all. The laptop makes the drawings richer;
if it is missing or slow, the Pi falls back to an offline procedural Andean scene
built from the words in the story (condor, llama, mountains, sun, river, house...).
The demo never stalls because of a missing piece: each module degrades to a mock and
the UI says which parts are real.

## Files

| File | What it does |
|---|---|
| `config.py` | Every setting. Override any value with `YACHACHIQ_<NAME>=...` in the environment. |
| `server.py` | FastAPI web server: kiosk page, WebSocket events, REST controls, TCP text input on :5005. |
| `static/index.html` | The kiosk UI. One big button. Spanish. No external assets, works offline. |
| `pipeline.py` | The state machine: listening, transcribing, thinking, imagining, drawing, done. |
| `audio.py` | Microphone recording with live level meter and silence auto-stop. |
| `stt.py` | faster-whisper transcription (int8, CPU). |
| `language.py` | Quechua detection, story words to motif keys. |
| `story.py` | Story analysis with a fallback chain: laptop, Ollama, Claude, offline rules. |
| `imagegen.py` | Picture for the story: Stable Diffusion via the laptop, or offline motifs. |
| `motifs.py` | Procedural Andean motifs and scene composer (pen strokes, no image model). |
| `vectorize.py` | Image to strokes (threshold, thinning, tracing, simplify, order), SVG/PNG previews. |
| `gcode.py` | Strokes to GRBL G-code. |
| `plotter.py` | GRBL sender with progress and stop. Mock when no serial port. |
| `tts.py` | Piper narration (macOS `say` while developing). |
| `button.py` | Optional GPIO arcade button that toggles listening. |
| `textclean.py` | Tidies the story before it is drawn and published: sentence case, punctuation, safe accents, and an optional LLM pass. |
| `photo.py` | Photographs the finished drawing (USB webcam or Pi camera, fixed by-id path). |
| `publish.py` | Store-and-forward upload of stories to `../docs/` (GitHub Pages gallery). |
| `selftest.py` | Runs the whole pipeline with mocks and checks every stage. |
| `yachachiq-lsp.service` | Boot service for the sign-language camera (`../lsp/lsp_app.py robot`). |
| `setup_pi.sh`, `yachachiq.service`, `kiosk.sh` | One-shot Pi install, boot service, full-screen browser. |
| `laptop/` | `ai_server.py` (Ollama + Stable Diffusion over HTTP), its setup script. |
| `arduino/` | `grbl_settings.txt` for the plotter, `motor_test.ino` to test motors without GRBL. |
| `gcode/` | `test_drawing.gcode` (a house), `calibrate_square.gcode` (50 mm square). |

## Quick start on any computer (no hardware)

```bash
cd yachachiq
pip install fastapi "uvicorn[standard]" numpy opencv-contrib-python-headless
python selftest.py
YACHACHIQ_MOCK=1 python server.py
```

Open http://localhost:8877. Press the button, wait six seconds (the mock "hears" a story),
and watch it draw. The gear icon (bottom right) has a text box to submit any story by typing.

## Raspberry Pi install

1. Raspberry Pi OS 64-bit (Bookworm) with desktop, a USB microphone, a speaker on the
   3.5 mm jack or USB, the plotter's Arduino on USB, and the touchscreen.
2. Copy this folder to the Pi (`scp -r yachachiq pi@raspberrypi.local:~/`) and run:

```bash
cd ~/yachachiq && bash setup_pi.sh
```

That installs packages, the Python venv, the Whisper `base` model, Piper with a Spanish
voice, runs the self test, installs the `yachachiq` systemd service and a kiosk autostart.
Reboot and the screen shows the big button.

Useful commands:

```bash
journalctl -u yachachiq -f          # live logs
sudo systemctl restart yachachiq    # after editing config.py
arecord -l                          # is the microphone seen?
arecord -d 3 t.wav && aplay t.wav   # mic + speaker round trip
ls /dev/ttyACM* /dev/ttyUSB*        # the Arduino's port -> SERIAL_PORT in config.py
```

Overrides go in `/etc/systemd/system/yachachiq.service` as `Environment=YACHACHIQ_...` lines,
or in the shell when running by hand.

## Networking with the laptop (optional)

Competition rules: no WiFi. Use one Ethernet cable between the Pi and the laptop.

- Laptop (macOS): System Settings, Network, the USB/Thunderbolt Ethernet adapter,
  Configure IPv4 "Manually", IP `192.168.7.1`, mask `255.255.255.0`.
- Pi: `sudo nmcli con add type ethernet ifname eth0 con-name lan ip4 192.168.7.2/24` then
  `sudo nmcli con up lan`.
- Check: `ping 192.168.7.1` from the Pi, `curl http://192.168.7.1:8600/health`.

The defaults in `config.py` already point at `192.168.7.1`. If you use other addresses set
`YACHACHIQ_AI_SERVER_URL`. Without the laptop the Pi uses `rules` + `motifs` automatically.

Laptop side, once with internet:

```bash
cd yachachiq/laptop && bash setup_laptop.sh
source venv/bin/activate && python ai_server.py
```

Apple Silicon generates a 512 px `sd-turbo` picture in a few seconds; a CPU-only laptop takes
about a minute, which is still fine since the Pi shows the story text while waiting.

### Using ComfyUI instead

If ComfyUI already runs on the laptop, it is the first image backend by default:

1. Start ComfyUI so the Pi can reach it: `python main.py --listen 0.0.0.0 --port 8188`.
2. Put the checkpoint you want in `ComfyUI/models/checkpoints/` (sd-turbo works well: 4 steps,
   cfg 1). The bundled `laptop/comfyui_workflow.json` expects `sd_turbo.safetensors`; set
   `YACHACHIQ_COMFYUI_CHECKPOINT=<file>` to use another one without editing the file.
3. To use your own workflow: in ComfyUI enable dev mode (Settings, "Enable Dev mode Options"),
   build any text-to-image graph, click **Save (API Format)**, and copy the JSON to
   `laptop/comfyui_workflow.json` (or point `YACHACHIQ_COMFYUI_WORKFLOW` at it). The Pi finds
   the positive and negative `CLIPTextEncode` nodes through the KSampler's links and patches
   the story prompt, a random seed, and the size in; everything else (LoRAs, ControlNet,
   upscalers) stays as you built it. Keep a `SaveImage` node at the end.
4. Check from the Pi: `curl http://192.168.7.1:8188/system_stats`.

The Pi asks for black ink line art (`STYLE_PROMPT` in `config.py`), traces the result to
strokes, and adds the Andean border. If ComfyUI is off, it falls back to `remote`, then `motifs`.

With internet you can also use Claude for the analysis: `pip install anthropic`, export
`ANTHROPIC_API_KEY`, and set `YACHACHIQ_STORY_BACKENDS=anthropic,rules`.

## Plotter

Hardware: Arduino UNO + CNC Shield V3 + DRV8825 drivers + NEMA 17 on X, Y and Z (pen lift),
12 V 3 A supply, grbl-servo firmware. See the previous plotter package notes for driver Vref
and wiring; the settings live in `arduino/grbl_settings.txt`.

Load the settings once and test:

```bash
source venv/bin/activate
python - <<'EOF'
from plotter import Plotter
p = Plotter(); p.connect()
for line in open("arduino/grbl_settings.txt"): p.send(line)
p.run(open("gcode/test_drawing.gcode").readlines()); p.close()
EOF
```

Before every session: put a sheet down, use the gear menu on the screen to jog the pen to
the bottom-left corner of the paper, lower it until it touches, and press `0` (set origin).
Paper size and margins are `PAPER_W_MM`, `PAPER_H_MM`, `MARGIN_MM` in `config.py`.
`PEN_MODE = "servo"` switches pen lift to the SG90 on D11 (`M3 S<angle>`).

If a plot takes too long, lower `MAX_STROKES` or raise `MIN_STROKE_PX`; the UI shows the
time estimate before drawing starts.

## Sign language (LSP)

`../lsp/lsp_app.py robot` runs on the same Pi with the second camera. It recognises the
Peruvian manual alphabet (model trained by the team), streams its camera to the kiosk page so
visitors see the letters appear, and when the signer holds both open palms for 2 s it sends the
whole story to TCP port 5005 — from there it is exactly like a spoken story.

## Instalación en la Raspberry Pi (Debian 12 o 13)

```bash
git clone --branch dev https://github.com/IgnacioOst721/yachachiq.git ~/yachachiq
cd ~/yachachiq/robot && bash setup_pi.sh
```

`setup_pi.sh` resuelve dos cosas que cambian entre versiones de Debian:

- **Nombres de paquetes** (`libopenblas0` en vez de `libatlas-base-dev`, `chromium`, `libglib2.0-0t64`):
  instala el que exista.
- **Python.** El reconocimiento de señas usa MediaPipe **0.10** (`mp.solutions.hands`); la versión 1.x
  eliminó esa API y solo existe para Python 3.13. Como Debian 13 trae Python 3.13, el instalador
  descarga un Python 3.12 independiente en `~/.local/pythons` y crea el `venv` con él. Los archivos del
  modelo de manos son idénticos entre MediaPipe 0.10.9 (donde se entrenó `model.pkl`) y 0.10.18 (ARM),
  así que los 21 puntos y la precisión son los mismos.

## Corrección del texto

Antes de dibujar y publicar, la historia pasa por `textclean.py`. Importa sobre todo en señas,
donde el texto llega en crudo:

| Origen | Antes | Después |
|---|---|---|
| Señas | `MI ABUELA VIVIA EN LA SIERRA` | `Mi abuela vivía en la sierra.` |
| Voz | `habia una vez un condor que bajaba al rio` | `Había una vez un cóndor que bajaba al río.` |

Las reglas son conservadoras y funcionan sin red: mayúscula inicial, punto final, espacios y
tildes **solo** en palabras donde el español no deja duda (`vivia`→`vivía`, `condor`→`cóndor`).
Las ambiguas (`papa`/`papá`, `esta`/`está`, `el`/`él`) se dejan como están.

Si la laptop está conectada, además pasa por el LLM (`/correct` en `ai_server.py`, u Ollama /
Claude), que corrige tildes y comas de verdad. **Su respuesta solo se acepta si sigue siendo la
misma historia** (se compara el conjunto de palabras): así el modelo nunca puede reescribir,
resumir ni inventar lo que la persona contó.

`YACHACHIQ_CLEAN_TEXT=0` lo apaga; `YACHACHIQ_CLEAN_WITH_LLM=0` deja solo las reglas offline.
Probar cualquier texto: `curl -X POST localhost:8877/api/clean -H 'content-type: application/json' -d '{"text":"..."}'`

## Photo + web archive

After the plotter reports `Idle`, `photo.py` takes a picture with the camera over the bed and the
story folder (`output/stories/<timestamp>/`: `story.txt`, `scene_1.png`, `scene_1_photo.png`) is
copied to `../docs/stories/` and pushed. No internet → it just waits for the next time.
Two USB cameras: find their fixed ids with `ls /dev/v4l/by-id/` and set `YACHACHIQ_PHOTO_CAMERA`
(robot) and `ASL_CAMERA` (lsp service) so they never swap after a reboot.

## Cómo se usa (sin botones ni teclado)

El robot está pensado para una mesa de exhibición: **nadie tiene que tocar nada.**

1. **Elige cómo contar la historia.** La pantalla de bienvenida ofrece dos tarjetas animadas:
   🎙️ **Con mi voz** y 🤟 **En lengua de señas**. Se eligen con el dedo (pantalla táctil), con un
   mouse, o con Tab + Enter. Al elegir, la tarjeta se ilumina y la otra se apaga.
   - **Voz** → empieza a grabar de inmediato.
   - **Señas** → la cámara pasa al centro de la pantalla en grande, con la letra que se está
     señando y la frase construyéndose en vivo. El micrófono se apaga para que no interrumpa.
2. **O, si se activa `YACHACHIQ_AUTO_LISTEN=1`, simplemente habla.** Con `AUTO_LISTEN` (apagado por defecto) el micrófono queda escuchando
   en reposo y arranca la grabación cuando alguien habla de verdad (más fuerte que el ruido del
   ambiente, sostenido 0.35 s). El círculo de la pantalla late en amarillo cuando está listo.
3. **Deja de hablar.** Se detiene solo tras 3.5 s de silencio (`SILENCE_SECONDS`).
   En señas, dos palmas abiertas 2 s envían la historia.
4. **Al final decides** si se publica: tapar la cámara = no; sonreír = sí. También hay dos botones
   en pantalla por si la pantalla es táctil.

El círculo grande de la pantalla también funciona como botón si hay pantalla táctil o mouse,
y la barra espaciadora si hay teclado — pero **ninguno es necesario**.

Ajustes: `YACHACHIQ_AUTO_LISTEN=1` lo enciende (por defecto la voz se inicia tocando **Con mi voz**). Si arranca solo con el ruido
del ambiente, subir `YACHACHIQ_AUTO_LISTEN_RMS` (0.030 por defecto); si cuesta despertarlo, bajarlo.

## Inputs other than the microphone

- **Typed text**: gear menu on the screen, or `curl -X POST localhost:8877/api/story -H 'content-type: application/json' -d '{"text":"..."}'`.
- **Sign-language camera / any device**: send one line of text to TCP port 5005: `echo "Un cóndor volaba..." | nc 192.168.7.2 5005`.
- **Physical button**: any USB button that types Space or Enter, or an arcade button on
  GPIO17 to GND with `python button.py`.
- **Keyboard**: Space starts and stops, Escape cancels.

## Pipeline events (WebSocket `/ws`)

`snapshot`, `state` (idle, listening, transcribing, thinking, imagining, drawing, narrating,
done, error), `level` (mic RMS), `transcript`, `analysis`, `image` (SVG preview + stats),
`progress` (sent/total/pct), `narration`, `backend`. The last story is saved in
`output/last_story.json` and `output/last.gcode` (redraw it with "Dibujar de nuevo").
