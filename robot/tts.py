"""Narration. Piper on the Pi, `say` on a Mac while developing, mock otherwise.
Without a speaker just leave it: the mock does nothing and the pipeline goes on."""
import logging
import shutil
import subprocess
import sys
import threading

import config

log = logging.getLogger("tts")


class TTS:
    def __init__(self):
        self.proc = None
        voice = str(config.PIPER_VOICE)
        if config.MOCK:
            self.mode = "mock"
        elif shutil.which(config.PIPER_BIN) and __import__("os").path.exists(voice):
            self.mode = "piper"
        elif sys.platform == "darwin" and shutil.which("say"):
            self.mode = "say"
        else:
            self.mode = "mock"

    def speak(self, text, blocking=False):
        text = (text or "").strip()
        if not text or self.mode == "mock":
            return
        def run():
            try:
                if self.mode == "say":
                    self.proc = subprocess.Popen(["say", "-v", "Paulina", text])
                    self.proc.wait()
                else:
                    p1 = subprocess.Popen([config.PIPER_BIN, "--model", str(config.PIPER_VOICE), "--output_raw"],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    self.proc = subprocess.Popen([config.AUDIO_PLAYER, "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                                                 stdin=p1.stdout)
                    p1.stdin.write(text.encode()); p1.stdin.close()
                    self.proc.wait()
            except Exception as e:
                log.warning("tts failed: %s", e)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        if blocking:
            t.join()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
