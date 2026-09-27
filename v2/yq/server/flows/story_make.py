"""Story flow, part 2: consent photo -> drawing -> printer + hologram + narration -> publish -> QR."""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

from yq.common import config
from yq.common.contracts import StoryInput, to_dict
from yq.publish import gallery
from yq.publish import sync as publish
from yq.server import camera as cam
from yq.server import settings
from yq.server.flows.base import Cancelled, FlowError

log = logging.getLogger("yq.server.story")
FINAL = ("done", "error", "cancelled")


def sentences(text: str, max_n: int = 24) -> list:
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", text or "") if p.strip()]
    out = []
    for p in parts:
        if out and len(out[-1]) < 25:
            out[-1] += " " + p
        else:
            out.append(p)
    return out[:max_n] or ([text.strip()] if (text or "").strip() else [])


class StoryMakeMixin:
    def init_make(self) -> None:
        self.vote = None
        self.consent: dict = {}
        self.drawing: dict = {}
        self.title = ""
        self.summary_es = ""
        self.qr_url = ""
        self.print_status: dict = {"status": "none"}
        self.holo_status: dict = {"status": "none"}
        self.published: dict = {}
        self.skip = threading.Event()

    def cleanup_make(self) -> None:
        self.k.pump.stop()

    def _vote(self, p: dict) -> None:
        self.vote = bool(p.get("publish"))

    def _skip_narration(self, p: dict) -> None:
        self.skip.set()
        self.sub.voice.stop_tts()

    # -- consent + portrait ------------------------------------------------------------------
    def s_consent(self):
        secs = int(config.CONSENT_SECONDS)
        self.vote = None
        self.k.pump.start(self.sub.camera)
        self.show("consent", seconds=secs, preview="/camera/preview.mjpeg")
        t_wait = time.time()
        while self.k.pump.latest() is None and time.time() - t_wait < 1.5:   # first frame before counting
            if self.cancel_event.wait(0.05):
                raise Cancelled()
        t0, looks, best, best_score = time.time(), [], None, -1.0
        while time.time() - t0 < secs and self.vote is None:
            jpg = self.k.pump.latest()
            look = cam.look_at(jpg)
            if look:
                looks.append(look)
                if not look["covered"] and cam.score(look) > best_score:
                    best, best_score = jpg, cam.score(look)
            self.bus.emit("consent_tick", remaining=max(0, int(round(secs - (time.time() - t0)))),
                          camera=bool(jpg), **{k: (look or {}).get(k) for k in ("covered", "face", "smile")})
            if self.cancel_event.wait(0.45):
                raise Cancelled()
        publish_ok, reason = cam.decide(looks, self.vote)
        t1 = time.time()
        while publish_ok and best is None and time.time() - t1 < 2.5:   # "Sí" tapped before any good frame
            jpg = self.k.pump.latest()
            look = cam.look_at(jpg)
            if look and not look["covered"]:
                best = jpg
            elif self.cancel_event.wait(0.1):
                raise Cancelled()
        self.k.pump.stop()
        self.consent = {"publish": publish_ok, "reason": reason, "portrait": bool(best and publish_ok)}
        self.dir.mkdir(parents=True, exist_ok=True)
        if best and publish_ok:
            (self.dir / "storyteller_photo.jpg").write_bytes(best)
        self.show("consent_result", publish=publish_ok, reason=reason)
        self.pause(3.0)
        return "making"

    # -- drawing -------------------------------------------------------------------------------
    def _file_url(self, path):
        if not path:
            return None
        p = Path(str(path))
        if not p.is_absolute():
            p = self.dir / "art" / p
        try:
            return self.url(str(p.resolve().relative_to(self.dir.resolve())))
        except ValueError:
            return None

    def s_making(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self._save_meta()
        self.show("making", stage="planning", fraction=0.0, message_es="Pensando qué dibujar…", image=None)
        story = StoryInput(story_id=self.story_id, source=self.source, text=self.text, lang=self.lang,
                           text_es=self.text_es, sign_lang=self.sign_lang, confirmed=True)

        def progress(p):
            p = to_dict(p) if not isinstance(p, dict) else p
            det = p.get("detail") or {}
            data = {"stage": p.get("stage"), "fraction": p.get("fraction"), "message_es": p.get("message_es")}
            url = self._file_url(det.get("image"))
            if url:
                data["image"] = url + "?t=%d" % int(time.time() * 1000)
            self.update(**data)

        public = bool(self.consent.get("publish"))
        d = self.call(self.sub.art.make_drawing, story, self.dir / "art", on_progress=progress, published=public,
                      timeout=float(settings.DRAWING_TIMEOUT))
        if not d or not d.get("front_svg"):
            raise FlowError("No pude terminar el dibujo. Probemos otra vez.", "err_drawing", retry="making")
        self.drawing = d
        if d.get("image") and Path(d["image"]).exists():
            shutil.copy2(d["image"], self.dir / "scene_1.png")
        self.qr_url = d.get("qr_url") or (gallery.story_url(self.story_id) if public else config.PUBLIC_BASE_URL)
        plan = self._read_plan()
        self.title = plan.get("title_es") or plan.get("title") or self._auto_title()
        self.summary_es = plan.get("summary_es", "")
        self._save_meta()
        return "showtime"

    def _read_plan(self) -> dict:
        for name in ("plan.json", "scene_plan.json"):
            p = self.dir / "art" / name
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    return data.get("plan", data)
                except ValueError:
                    pass
        return {}

    def _auto_title(self) -> str:
        words = (self.text_es or self.text).split()
        return " ".join(words[:6]).rstrip(".,;:!?") + ("…" if len(words) > 6 else "")

    # -- printer, hologram, narration ---------------------------------------------------------------
    def s_showtime(self):
        narr_lang, narr_text = (self.lang, self.text)
        if not self.sub.languages.has_tts(self.lang) and self.text_es:
            narr_lang, narr_text = "spa_Latn", self.text_es
        sents = sentences(narr_text)
        self.show("showtime", image=self.url("scene_1.png"), title=self.title, text=self.text, text_es=self.text_es,
                  lang_name=self.sub.languages.name_es(self.lang), printer={"status": "sending"},
                  hologram={"status": "sending"}, narration={"index": -1, "total": len(sents), "sentences": sents})
        self._print()
        self._hologram(narr_lang, narr_text)
        self.skip.clear()
        for i, s in enumerate(sents):
            if self.skip.is_set():
                break
            self.update(narration={"index": i, "total": len(sents), "sentences": sents})
            t0 = time.time()
            try:
                self.call(self.sub.voice.say, s, narr_lang, wait=True, timeout=float(settings.NARRATE_TIMEOUT),
                          on_cancel=self.sub.voice.stop_tts)
            except Cancelled:
                raise
            except Exception as e:                       # narration is nice to have, never blocking
                log.warning("narration failed: %s", e)
            # each subtitle stays at least its reading time (no voice installed, or a mock speaker)
            rest = max(2.2, 0.055 * len(s)) * float(settings.PAUSE_SCALE) - (time.time() - t0)
            if rest > 0 and self.skip.wait(rest) is False and self.cancel_event.is_set():
                raise Cancelled()
        self.update(narration={"index": len(sents), "total": len(sents), "sentences": sents, "done": True})
        self._finalize()
        return "done"

    def _print(self) -> None:
        pr = self.sub.printer
        try:
            if not pr.available():
                self.print_status = {"status": "offline"}
            else:
                job = self.call(pr.submit_drawing, self.drawing, self.qr_url, timeout=40)
                self.print_status = {"status": "sent", "id": job.get("id")}
                threading.Thread(target=self._poll_printer, args=(job.get("id"),), name="printer-poll",
                                 daemon=True).start()
        except Cancelled:
            raise
        except Exception as e:
            log.warning("printer failed: %s", e)
            self.print_status = {"status": "error", "message": str(e)[:160]}
        self.update(printer=self.print_status)

    def _poll_printer(self, job_id: str) -> None:
        pr, t0 = self.sub.printer, time.time()
        poll = max(0.05, 2.0 * min(1.0, float(settings.MOCK_SPEED)))
        while time.time() - t0 < 7200 and not self.k.stopping.is_set():
            st = pr.status(job_id)
            st = {k: st.get(k) for k in ("status", "progress", "message")}
            st["id"] = job_id
            self.print_status = st
            self.bus.emit("printer", story_id=self.story_id, **st)
            if st["status"] in FINAL or st["status"] in ("sent", "unreachable"):
                break
            time.sleep(poll)
        self._save_meta()

    def _hologram(self, lang: str, text: str) -> None:
        from yq.hologram.client import build_story, package_files
        holo = self.sub.hologram
        try:
            if not holo.available():
                self.holo_status = {"status": "offline"}
            else:
                wav = None
                try:
                    wav = self.call(self.sub.voice.synthesize, text, lang, timeout=90)
                except Cancelled:
                    raise
                except Exception as e:
                    log.warning("narration audio for the hologram failed: %s", e)
                if wav:
                    (self.dir / "narration_1.wav").write_bytes(wav)
                scenes = [{"image": str(self.dir / "scene_1.png"), "caption_es": self.text_es or self.text,
                           "audio": str(self.dir / "narration_1.wav") if wav else ""}]
                story = build_story(self.story_id, self.title, self.lang, self.text, self.text_es, scenes)
                ans = self.call(holo.send, story, package_files(scenes), timeout=90)
                self.holo_status = {"status": "playing", "duration_s": (ans or {}).get("duration_s")}
        except Cancelled:
            raise
        except Exception as e:
            log.warning("hologram failed: %s", e)
            self.holo_status = {"status": "error", "message": str(e)[:160]}
        self.update(hologram=self.holo_status)

    # -- save, publish, goodbye -------------------------------------------------------------------------
    def _save_meta(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"story": to_dict(StoryInput(story_id=self.story_id, source=self.source, text=self.text,
                                            lang=self.lang, text_es=self.text_es, sign_lang=self.sign_lang,
                                            confirmed=True)),
                "title": self.title, "summary_es": self.summary_es,
                "lang_name_es": self.sub.languages.name_es(self.lang), "qr_url": self.qr_url,
                "web_id": gallery.web_id(self.story_id), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "consent": self.consent, "drawing": self.drawing, "printer": self.print_status,
                "hologram": self.holo_status,
                "transcript": {k: self.transcript.get(k) for k in ("engine", "confidence", "lang", "lang_candidates")}}
        (self.dir / "story.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
        (self.dir / "story.txt").write_text(gallery.story_txt(
            self.text, self.text_es, self.title, self.summary_es, self.lang, meta["lang_name_es"],
            self.source, self.sign_lang), encoding="utf-8")

    def _finalize(self) -> None:
        self._save_meta()
        marker = ".ready" if self.consent.get("publish") else ".private"
        (self.dir / marker).write_text(self.consent.get("reason", ""), encoding="utf-8")
        if marker == ".ready":
            self.published = publish.sync_later(
                on_done=lambda r: self.bus.emit("published", story_id=self.story_id, **r))
        else:
            self.published = {"status": "private"}

    def s_done(self):
        public = bool(self.consent.get("publish"))
        self.show("done", public=public, qr_url=self.qr_url,
                  image=self.url("scene_1.png"), title=self.title, printer=self.print_status,
                  published={k: self.published.get(k) for k in ("status", "new")})
        name, _ = self.wait("finish", "new_story")
        if name == "new_story":
            self.next_flow = "story"
        return None
