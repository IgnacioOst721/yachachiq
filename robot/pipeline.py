"""The Yachachiq state machine.

idle -> listening -> transcribing -> thinking -> imagining -> drawing
     -> photographing -> publishing -> narrating -> done   (or error)

Text can also come in directly (typed, or signed from the LSP camera through
server.py's TCP port) with submit_text(); it joins at "thinking".
Every step emits events through on_event(name, data) for the screen.
"""
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime

import config
import gcode as gcode_mod
import imagegen
import photo
import publish
import story
import textclean
import vectorize
from audio import Recorder, VoiceTrigger, save_wav
from language import detect_language
from plotter import Plotter
from stt import STT
from tts import TTS

log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, on_event=None):
        self.on_event = on_event or (lambda e, d: None)
        self.state = "idle"
        self.last_error = None
        self.last_story = None
        self.last_image = None
        self.progress = (0, 0)
        self.level = 0.0
        self.story_dir = None
        self.input_mode = None              # "voz" | "senas" | None (visitor has not chosen)
        self.consent = None                 # last decision: {"publish", "reason", "portrait"...}
        self.consent_provider = None        # set by server.py: fn(seconds) -> decision dict
        self.recorder = Recorder()
        self.trigger = VoiceTrigger(on_speech=self._voice_woke_us,
                                    on_level=lambda v: self.emit("level", {"level": v, "idle": True}))
        self.stt = STT()
        self.tts = TTS()
        self.plotter = Plotter().connect()
        vectorize.MACHINE_LIMITS = self.plotter.limits()     # honest "≈ N min" on screen
        threading.Timer(10.0, self._watch_plotter).start()
        self._thread = None
        self._cancel = threading.Event()
        os.makedirs(str(config.OUTPUT_DIR), exist_ok=True)
        os.makedirs(str(config.STORIES_DIR), exist_ok=True)
        self._arm_trigger()

    # --- helpers ---------------------------------------------------------------------------------
    def emit(self, event, data=None):
        try:
            self.on_event(event, data or {})
        except Exception as e:      # never let the UI kill the robot
            log.warning("on_event failed: %s", e)

    def _set_state(self, s, **extra):
        self.state = s
        self.emit("state", {"state": s, **extra})
        if s in ("idle", "done", "error"):
            self.input_mode = None
            threading.Timer(float(config.AUTO_LISTEN_COOLDOWN), self._arm_trigger).start()
        if s in ("done", "error") and config.IDLE_RESET_SECONDS:
            self._schedule_reset()

    def _schedule_reset(self):
        """Clear the finished story after a while so the next visitor finds a clean screen."""
        t = getattr(self, "_reset_timer", None)
        if t:
            t.cancel()
        self._reset_timer = threading.Timer(float(config.IDLE_RESET_SECONDS), self.reset)
        self._reset_timer.daemon = True
        self._reset_timer.start()

    def reset(self):
        """Back to the welcome screen: forget the last story, drawing and error."""
        if self.busy():
            return False
        self.last_story = None
        self.last_image = None
        self.last_error = None
        self.progress = (0, 0)
        self.level = 0.0
        self.input_mode = None
        self.emit("reset", {})
        self._set_state("idle")
        return True

    def _watch_plotter(self):
        """Every 10 s, if the board was missing, see whether it has been plugged in."""
        try:
            if not self.busy() and getattr(self.plotter, "missing", False):
                if self.plotter.reconnect_if_needed():
                    log.info("plotter appeared: %s", self.plotter.port)
                    vectorize.MACHINE_LIMITS = self.plotter.limits()
                    self.emit("modes", self.modes())
        finally:
            t = threading.Timer(10.0, self._watch_plotter)
            t.daemon = True
            t.start()

    def _arm_trigger(self):
        """Listen for a voice again once the robot is free (never while signing).
        With no microphone yet, try again every 20 s so plugging one in later just works."""
        if self.state == "waiting_signs":
            return
        if not self.busy() and config.AUTO_LISTEN:
            if self.trigger.start():
                self.emit("auto_listen", {"armed": True})
            else:
                if getattr(self, "_rearm", None):
                    self._rearm.cancel()
                self._rearm = threading.Timer(20.0, self._arm_trigger)
                self._rearm.daemon = True
                self._rearm.start()

    def _voice_woke_us(self):
        """Somebody started talking while the robot was idle."""
        if self.busy():
            return
        self.trigger.stop()
        self.emit("auto_listen", {"armed": False, "woke": True})
        self.start_listening()

    def choose_mode(self, mode):
        """The visitor picks how to tell the story on the welcome screen."""
        if self.busy():
            return False
        self.input_mode = mode if mode in ("voz", "senas") else None
        self.emit("mode", {"mode": self.input_mode})
        if self.input_mode == "voz":
            return self.start_listening()
        if self.input_mode is None:
            # "elegir otra forma": back to the welcome screen. This used to fall through into
            # waiting_signs, so the back button in sign mode did nothing visible.
            self._set_state("idle")
            return True
        # sign language: the mic must not butt in while somebody is signing
        self.trigger.stop()
        self._set_state("waiting_signs")
        return True

    def busy(self):
        return self.state not in ("idle", "done", "error", "waiting_signs")

    def modes(self):
        return {"plotter": self.plotter.mode, "stt": self.stt.mode, "audio": self.recorder.mode,
                "tts": self.tts.mode, "image": ",".join(config.IMAGE_BACKENDS),
                "story": ",".join(config.STORY_BACKENDS), "photo": photo.mode(), "publish": publish.mode(),
                "input_mode": self.input_mode or "-",
                "auto_listen": ("on" if self.trigger.running else ("off" if not config.AUTO_LISTEN else self.trigger.mode)),
                "consent": ("off" if not config.CONSENT_REQUIRED else ("camera" if self.consent_provider else "none"))}

    def snapshot(self):
        return {"state": self.state, "modes": self.modes(), "input_mode": self.input_mode,
                "progress": self.progress, "level": self.level,
                "last_error": self.last_error, "story": self.last_story, "image": self.last_image}

    # --- inputs ------------------------------------------------------------------------------------
    def start_listening(self):
        if self.busy():
            return False
        if not self.recorder.available():
            log.warning("start_listening: no microphone connected")
            self.input_mode = None
            self.emit("mode", {"mode": None})
            self.emit("notice", {"text": "No hay micrófono conectado. Cuenta tu historia en lengua de señas, "
                                         "o conecta un micrófono."})
            return False
        self._cancel.clear()
        self.trigger.stop()
        self._set_state("listening")
        def on_level(v):
            self.level = v
            self.emit("level", {"level": v})
        self.recorder.start(on_level=on_level, on_auto_stop=self.stop_listening)
        return True

    def stop_listening(self):
        if self.state != "listening":
            return False
        audio = self.recorder.stop()
        if getattr(self.recorder, "no_speech", False):
            # nobody said anything: back to the welcome screen instead of transcribing silence
            log.info("nobody spoke in %.0f s -> back to idle", config.NO_SPEECH_SECONDS)
            self._set_state("idle")
            self.emit("notice", {"text": "No escuché nada. Elige cómo contar tu historia y empieza cuando quieras."})
            return True
        self._start(self._run_audio, audio)
        return True

    def toggle_listening(self):
        return self.stop_listening() if self.state == "listening" else self.start_listening()

    def submit_text(self, text):
        text = (text or "").strip()
        if self.busy() or not text:
            return False
        self._cancel.clear()
        self._start(self._run, text)
        return True

    def redraw(self):
        path = os.path.join(str(config.OUTPUT_DIR), "last.gcode")
        if self.busy() or not os.path.exists(path):
            return False
        self._cancel.clear()
        self._start(self._draw_file, path)
        return True

    def cancel(self):
        self._cancel.set()
        if self.state == "listening":
            self.recorder.stop()
        self.plotter.stop()
        self.tts.stop()
        self._set_state("idle")
        return True

    def _start(self, fn, *args):
        self._thread = threading.Thread(target=fn, args=args, daemon=True)
        self._thread.start()

    # --- stages -----------------------------------------------------------------------------------
    def _run_audio(self, audio):
        try:
            self._set_state("transcribing")
            try:
                save_wav(audio, os.path.join(str(config.OUTPUT_DIR), "last.wav"))
            except Exception:
                pass
            dur = len(audio) / float(config.SAMPLE_RATE)
            if self.recorder.mode != "mock" and dur < config.MIN_SPEECH_SECONDS:
                raise RuntimeError("No se escuchó nada. Intenta de nuevo.")
            text = self.stt.transcribe(audio)
            if not text.strip():
                raise RuntimeError("No entendí la historia. Intenta de nuevo, más cerca del micrófono.")
            self._run(text)
        except Exception as e:
            self._fail(e)

    def _run(self, text):
        try:
            crudo = text
            if config.CLEAN_TEXT:
                text, notas = textclean.clean(text)
                if notas:
                    log.info("texto corregido (%s)", ", ".join(notas))
                    self.emit("cleaned", {"before": crudo, "after": text, "notes": notas})
            self.emit("transcript", {"text": text, "raw": crudo})
            lang, words = detect_language(text)
            self.emit("language", {"lang": lang, "words": words})

            self._set_state("thinking")
            analysis = story.analyze(text, lang, on_status=lambda b: self.emit("backend", {"stage": "story", "backend": b}))
            self.emit("analysis", analysis)
            if self._cancel.is_set():
                return

            self._set_state("imagining")
            img = imagegen.generate(analysis, on_status=lambda b: self.emit("backend", {"stage": "image", "backend": b}))
            stats = img["stats"]
            self.last_image = {"svg": img["svg"], "stats": stats, "source": img["source"], "png_path": img.get("png_path")}
            self.emit("image", self.last_image)
            if self._cancel.is_set():
                return

            lines = gcode_mod.from_polylines(img["polylines_mm"], analysis.get("title", "historia"))
            gcode_path = os.path.join(str(config.OUTPUT_DIR), "last.gcode")
            gcode_mod.save(lines, gcode_path)
            self.story_dir = self._save_story(text, analysis, img, lines)
            self.last_story = {"title": analysis.get("title"), "text": text, "lang": lang, "elements": analysis.get("elements"),
                               "scene": analysis.get("scene"),
                               "narration": analysis.get("narration"), "backend": analysis.get("backend"),
                               "image_source": img["source"], "stats": stats, "dir": self.story_dir}
            with open(os.path.join(str(config.OUTPUT_DIR), "last_story.json"), "w", encoding="utf-8") as f:
                json.dump(self.last_story, f, ensure_ascii=False, indent=1)

            # the screen shows what the robot understood and is about to draw; the narration
            # text stays in the data for the speaker (tts) when there is one
            self.emit("narration", {"text": analysis.get("scene") or analysis.get("narration", "")})
            if config.SPEAK_WHILE_DRAWING:
                self.tts.speak(analysis.get("narration", ""))

            if not self._draw(lines):
                return

            self._after_drawing(analysis)
        except Exception as e:
            self._fail(e)

    def _draw(self, lines):
        self._set_state("drawing", total=len(lines))
        self.progress = (0, len(lines))
        def on_progress(sent, total):
            self.progress = (sent, total)
            self.emit("progress", {"sent": sent, "total": total, "pct": int(100 * sent / max(total, 1))})
        # the Arduino may have been plugged in after the robot started
        if self.plotter.reconnect_if_needed():
            self.emit("modes", self.modes())
        if self.plotter.mode != "mock":
            self.plotter.unlock()
        ok = self.plotter.run(lines, on_progress=on_progress)
        if not ok or self._cancel.is_set():
            self._set_state("idle")
            return False
        return True

    def _draw_file(self, path):
        try:
            with open(path) as f:
                lines = f.readlines()
            if self._draw(lines):
                self._set_state("done")
        except Exception as e:
            self._fail(e)

    def _after_drawing(self, analysis):
        if config.PHOTO_ENABLED and self.story_dir:
            self._set_state("photographing")
            out = os.path.join(self.story_dir, "scene_1_photo.png")
            ok = photo.capture(out, fallback_png=os.path.join(self.story_dir, "preview.png"))
            self.emit("photo", {"ok": ok, "path": out if ok else None})
        decision = self._ask_consent()
        if config.PUBLISH_ENABLED and decision["publish"]:
            self._set_state("publishing")
            res = publish.sync()
            self.emit("published", res)
        elif config.PUBLISH_ENABLED:
            self.emit("published", {"status": "private", "new": 0, "reason": decision["reason"]})
        if not config.SPEAK_WHILE_DRAWING:
            self._set_state("narrating")
            self.tts.speak(analysis.get("narration", ""), blocking=True)
        self._set_state("done")

    def _ask_consent(self):
        """Ask the storyteller (on screen, with the portrait camera) whether the story
        may go to the public archive. Camera covered = no. Never raises."""
        if not config.CONSENT_REQUIRED:
            d = {"publish": True, "reason": "sin consentimiento requerido", "portrait": None}
        elif self.consent_provider is None:
            # no screen/camera to ask: private by default (mock says yes so tests flow)
            d = {"publish": bool(config.MOCK), "reason": "sin pantalla ni cámara para preguntar", "portrait": None}
        else:
            self._set_state("consent", seconds=int(config.CONSENT_SECONDS))
            try:
                d = self.consent_provider(int(config.CONSENT_SECONDS))
            except Exception as e:
                log.warning("consent provider failed: %s", e)
                d = {"publish": False, "reason": f"error: {e}", "portrait": None}
        if self._cancel.is_set():
            d = {"publish": False, "reason": "cancelado", "portrait": None}
        if self.story_dir:
            if d.get("portrait") and config.PORTRAIT_ENABLED and d["publish"]:
                with open(os.path.join(self.story_dir, "storyteller_photo.jpg"), "wb") as f:
                    f.write(d["portrait"])
            if not d["publish"]:
                with open(os.path.join(self.story_dir, ".private"), "w", encoding="utf-8") as f:
                    f.write(d.get("reason", ""))
        self.consent = {k: v for k, v in d.items() if k != "portrait"}
        self.consent["portrait"] = bool(d.get("portrait"))
        self.emit("consent", self.consent)
        return d

    def _save_story(self, text, analysis, img, lines):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        d = os.path.join(str(config.STORIES_DIR), ts)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "story.txt"), "w", encoding="utf-8") as f:
            f.write("STORY\n" + "=" * 40 + "\n" + text + "\n\nSCENES\n" + "=" * 40 + "\n")
            f.write(f"1. {analysis.get('scene', '')}\n")
            f.write(f"\nTITLE: {analysis.get('title', '')}\nLANG: {analysis.get('lang', '')}\nIMAGE: {img['source']}\n")
        try:
            vectorize.to_png(img["polylines_mm"], os.path.join(d, "preview.png"))
        except Exception as e:
            log.warning("preview failed: %s", e)
        if img.get("png_path") and os.path.exists(img["png_path"]):
            shutil.copy(img["png_path"], os.path.join(d, "scene_1.png"))
        else:
            src = os.path.join(d, "preview.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, "scene_1.png"))
        gcode_mod.save(lines, os.path.join(d, "scene_1.gcode"))
        return d

    def _fail(self, e):
        log.exception("pipeline error")
        self.last_error = str(e)
        self._set_state("error", error=str(e))
        self.emit("error", {"error": str(e)})
