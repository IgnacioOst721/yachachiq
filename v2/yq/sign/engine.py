"""SignEngine: camera -> pose -> features -> recognizers, in one worker thread (CONTRACTS.md §6).

    eng = SignEngine("prl", on_token=print)
    eng.start(); eng.set_mode("letters")
    eng.latest_jpeg()   # preview; the skeleton is drawn on the SAME frame the keypoints came from
    eng.state()         # {"hands_visible", "fps", "buffer", "candidates", ...}
    eng.accept(0); eng.backspace(); eng.clear(); eng.text()

Mock: with YQ_MOCK=1 / YQ_MOCK_SIGN_CAMERA=1 (or when the pose model is not downloaded) the
engine replays synthetic keypoints built from the letter model's own hand prototypes, so the
REAL classifier, segmenter and lexicon still run.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from yq.common import config
from yq.common.contracts import SignToken

from . import features as F
from . import lexicon as lx
from . import settings, sources
from .letters import LetterClassifier
from .spell import Speller


class SignEngine:
    def __init__(self, sign_lang: str, camera=None, on_token: Optional[Callable] = None,
                 on_status: Optional[Callable] = None, pose=None):
        self.sign_lang = sign_lang
        self.on_token, self.on_status = on_token, on_status
        self._camera_arg, self._pose = camera, pose
        self.mode = "letters"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._last_jpeg_t = 0.0
        self._fps = 0.0
        self._hands = False
        self._error = ""
        self._words: list = []             # confirmed story words (display strings)
        self._word_cands: list = []        # words mode: alternatives of the last sign
        self._shown_spelled: list = []     # letters mode: the completion chips the screen last got
        self._pending_letter = None
        self.speller: Optional[Speller] = None
        self.word_rec = None
        self.spotter = None
        self._load_models()

    # ---------------------------------------------------------------- models
    def _load_models(self) -> None:
        try:
            clf = LetterClassifier.load(self.sign_lang)
            self.speller = Speller(self.sign_lang, clf, lx.get(self.sign_lang),
                                   on_letter=self._letter_written, on_word_end=self._word_end)
        except FileNotFoundError as e:
            self._error = str(e)
        try:
            from .words import SignSpotter, WordRecognizer
            self.word_rec = WordRecognizer.load(self.sign_lang, backend="cpu")
            self.spotter = SignSpotter(rest_s=settings.SIGN_WORD_REST_S, max_s=settings.SIGN_WORD_MAX_S)
        except (FileNotFoundError, ImportError):
            self.word_rec = None

    def _open_source(self):
        cam = self._camera_arg
        if cam is not None:
            if getattr(cam, "is_keypoints", False) or isinstance(cam, sources.LatestFrameCamera):
                return cam
            return sources.LatestFrameCamera(cam)       # any cv2.VideoCapture-like object
        if config.mock("sign_camera") or self._pose_missing():
            shapes = {}
            if self.speller is not None:
                shapes = {k: np.asarray(v, np.float32) for k, v in self.speller.clf.prototypes.items()}
            return sources.SyntheticSource(words=("condor", "sol"), shapes=shapes)
        return sources.open_camera()

    @staticmethod
    def _pose_missing() -> bool:
        from . import modelstore
        return modelstore.pose_model_path(settings.SIGN_POSE_MODEL) is None

    # ---------------------------------------------------------------- public API
    def start(self) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True, name="sign-engine")
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=3.0)
        self._th = None

    def set_mode(self, mode: str) -> None:
        if mode not in ("letters", "words"):
            raise ValueError("mode must be 'letters' or 'words'")
        with self._lock:
            self.mode = mode
            if self.speller:
                self.speller.seg.reset()
            if self.spotter:
                self.spotter.reset()

    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def text(self) -> str:
        with self._lock:
            parts = list(self._words)
            if self.speller and self.speller.letters:
                parts.append(self.speller.buffer())
            return " ".join(parts)

    def state(self) -> dict:
        with self._lock:
            st = {"hands_visible": self._hands, "fps": round(self._fps, 1), "mode": self.mode,
                  "sign_lang": self.sign_lang, "text": self.text(), "error": self._error,
                  "buffer": "", "candidates": [], "letter": None}
            if self.mode == "letters" and self.speller:
                sp = self.speller
                st["buffer"] = sp.buffer()
                if sp.letters:
                    cands = sp.candidates()
                    self._shown_spelled = [w for w, _p in cands]
                    st["candidates"] = [{"text": w, "prob": round(p, 3), "kind": "word"} for w, p in cands]
                else:       # alternatives for the word just closed (accept(i) swaps it)
                    st["candidates"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in self._word_cands]
                st["letter"] = {"current": sp.current[0], "conf": round(sp.current[1], 3),
                                "progress": round(sp.progress, 2), "hand": sp.side}
            elif self.mode == "words":
                st["candidates"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in self._word_cands]
                st["words_available"] = self.word_rec is not None
            return st

    def accept(self, index: int = 0) -> Optional[str]:
        """Letters mode: take word completion `index` for the letters signed so far.
        Words mode: replace the last recognised sign with alternative `index`."""
        with self._lock:
            if self.mode == "letters" and self.speller and self.speller.letters:
                # index into the SAME list the visitor sees (state() ranks with final=False; re-ranking
                # here with final=True made tapping "help" commit "hell", 2026-09-27 review)
                words = self._shown_spelled or [w for w, _p in self.speller.candidates()]
                if not words:
                    return None
                word = words[min(index, len(words) - 1)]
                self._words.append(word)
                self.speller.clear()
                self._word_cands = []
                self._shown_spelled = []
                self._emit_word(word, word, 1.0, [])
                return word
            if self._word_cands:          # swap the last word for one of its alternatives
                c = self._word_cands[min(index, len(self._word_cands) - 1)]
                if self._words and self._word_cands[0].get("_appended"):
                    self._words[-1] = c["text"]
                else:
                    self._words.append(c["text"])
                self._word_cands = []
                return c["text"]
        return None

    def backspace(self) -> None:
        with self._lock:
            if self.mode == "letters" and self.speller and self.speller.backspace_letter():
                return
            if self._words:
                self._words.pop()
            self._word_cands = []

    def clear(self) -> None:
        with self._lock:
            self._words, self._word_cands, self._shown_spelled = [], [], []
            if self.speller:
                self.speller.clear()
            if self.spotter:
                self.spotter.reset()

    # ---------------------------------------------------------------- per-frame logic
    def process_keypoints(self, t: float, xy: np.ndarray, conf: np.ndarray) -> None:
        """One frame of canonical keypoints (used by the worker thread and by tests)."""
        with self._lock:
            self._hands = any(F.hand_usable(F.hand_block(xy, conf, s)[1]) for s in ("left", "right"))
            if self.mode == "letters" and self.speller:
                self.speller.step(t, xy, conf)
            elif self.mode == "words" and self.word_rec is not None and self.spotter is not None:
                seg = self.spotter.update(t, xy, conf)
                if seg is not None:
                    self._sign_spotted(*seg)

    def _sign_spotted(self, seq_xy, seq_cf, t0, t1) -> None:
        top = self.word_rec.predict(seq_xy, seq_cf, k=5)
        cands = [{"value": g, "text": self.word_rec.text.get(g, g.lower()), "prob": round(p, 3), "kind": "word"}
                 for g, p in top]
        appended = top[0][1] >= settings.SIGN_WORD_MIN_CONF
        if appended:
            self._words.append(cands[0]["text"])
        cands[0]["_appended"] = appended
        self._word_cands = cands
        self._emit(SignToken("word", top[0][0], self.sign_lang, top[0][1], [[g, p] for g, p in top[1:]],
                             t0, t1, cands[0]["text"]))

    def _letter_written(self, letter, c, alts, t0, t1) -> None:
        self._word_cands = []
        if letter == "SPACE":
            self._word_end(force=True)
        elif letter == "BACK":
            if self.speller and not self.speller.backspace_letter() and self._words:
                self._words.pop()
        self._emit(SignToken("letter", letter, self.sign_lang, float(c), [list(a) for a in alts], t0, t1,
                             letter if len(letter) == 1 else ""))

    def _word_end(self, force: bool = False) -> None:
        """Hand rested after spelling (or the SPACE sign): the word is closed with the best
        candidate, and the other candidates stay on screen so the visitor can swap it."""
        if not self.speller or not self.speller.letters:
            return
        cands = self.speller.candidates(final=True)
        if not cands:
            return
        word = cands[0][0]
        self._words.append(word)
        self.speller.clear()
        self._shown_spelled = []          # the chips of the next word come from the next state()
        self._word_cands = [{"text": w, "prob": round(p, 3), "kind": "word", "replace": True} for w, p in cands]
        self._word_cands[0]["_appended"] = True
        self._emit_word(word, word, cands[0][1], [[w, p] for w, p in cands[1:]])

    def _emit_word(self, value, text, conf, alts) -> None:
        self._emit(SignToken("word", value, self.sign_lang, float(conf), alts, time.time(), time.time(), text))

    def _emit(self, tok: SignToken) -> None:
        if self.on_token:
            try:
                self.on_token(tok)
            except Exception:
                pass

    # ---------------------------------------------------------------- worker thread
    def _run(self) -> None:
        try:
            src = self._open_source()
        except Exception as e:
            self._error = "camera: %s" % e
            return
        pose = self._pose
        is_kp = getattr(src, "is_keypoints", False)
        if not is_kp and pose is None:
            try:
                from .pose import PoseEstimator
                pose = PoseEstimator()
            except Exception as e:
                self._error = "pose: %s" % e
                src.release()
                return
        last_n, last_t = -1, None
        try:
            while not self._stop.is_set():
                frame = None
                if is_kp:
                    got = src.next()
                    if got is None:
                        break
                    t, xy, cf = got
                    wh = getattr(src, "image_wh", (1280, 800))
                else:
                    ok, frame, t, last_n = src.read(last_n)
                    if not ok:
                        continue
                    if pose is not None and self.speller is not None:
                        pose.refine_side = self.speller.side if (self.mode == "letters" and
                                                                  settings.SIGN_HAND_REFINE != "off") else None
                    pf = pose.process(frame, t)
                    xy, cf, wh = pf.canon_xy, pf.canon_conf, pf.image_wh
                self.process_keypoints(t, xy, cf)
                if last_t is not None and t > last_t:
                    inst = 1.0 / (t - last_t)
                    self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
                last_t = t
                self._maybe_preview(frame, xy, cf, wh)
        except Exception as e:  # keep the kiosk alive; report
            self._error = "engine: %r" % e
        finally:
            src.release()

    def _maybe_preview(self, frame, xy, cf, wh) -> None:
        now = time.monotonic()
        if now - self._last_jpeg_t < 1.0 / max(settings.SIGN_PREVIEW_FPS, 1e-3):
            return
        self._last_jpeg_t = now
        import cv2
        img = frame.copy() if frame is not None else sources.blank_canvas(wh)
        hl = self.speller.side if (self.speller and self.mode == "letters") else None
        sources.draw_skeleton(img, xy, cf, highlight=hl)      # same frame as the keypoints
        if settings.SIGN_MIRROR:
            img = cv2.flip(img, 1)                          # image AND dots flipped together
        if self.mode == "letters" and self.speller and self.speller.current[0]:
            letter, p = self.speller.current
            h = img.shape[0]
            cv2.putText(img, letter.upper(), (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
            cv2.rectangle(img, (20, h - 20), (20 + int(200 * self.speller.progress), h - 10), (0, 220, 0), -1)
        w = settings.SIGN_PREVIEW_WIDTH
        if img.shape[1] > w:
            img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self._jpeg = buf.tobytes()
        if self.on_status:
            try:
                self.on_status(self.state())
            except Exception:
                pass
