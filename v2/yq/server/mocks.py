"""UI-side mocks for every subsystem the kiosk calls (CONTRACTS.md §6).

Used when the real module is missing, fails to import, or YQ_UI_MOCKS names it.
They follow the contract signatures exactly so flows cannot tell the difference;
timings scale with YQ_MOCK_SPEED (tests use a small value). Art and box mocks
live in mocks_art_box.py and are re-exported here.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from yq.common.contracts import Transcript
from yq.server import mock_assets, settings
from yq.server.mocks_art_box import MockArt, MockBox  # noqa: F401  (re-export)
from yq.server.mocks_sign import SIGN_LANGS, MockSign, MockSignEngine  # noqa: F401  (re-export)


def msleep(seconds: float, stop: Optional[threading.Event] = None) -> bool:
    """Sleep seconds * MOCK_SPEED; returns True if `stop` was set meanwhile."""
    s = max(0.0, seconds * float(settings.MOCK_SPEED))
    if stop is not None:
        return stop.wait(s)
    time.sleep(s)
    return False


# --- languages (VOICE owns the real yq.common.languages) -------------------------------------
_L = [  # code, iso639_3, native, es, en, region, asr, tts, popular (display rank, 1 = first)
    ("spa_Latn", "spa", "Español", "Español", "Spanish", "PE", 1, 1, 1),
    ("quy_Latn", "quy", "Runasimi (Chanka)", "Quechua ayacuchano", "Ayacucho Quechua", "PE", 1, 1, 2),
    ("quz_Latn", "quz", "Runasimi (Qusqu)", "Quechua cusqueño", "Cusco Quechua", "PE", 1, 0, 3),
    ("ayr_Latn", "ayr", "Aymar aru", "Aimara", "Aymara", "PE", 1, 0, 4),
    ("cni_Latn", "cni", "Asháninka", "Asháninka", "Asháninka", "PE", 1, 0, 5),
    ("agr_Latn", "agr", "Awajún", "Awajún", "Awajún", "PE", 1, 0, 6),
    ("shp_Latn", "shp", "Shipibo-Konibo", "Shipibo-konibo", "Shipibo-Conibo", "PE", 1, 0, 7),
    ("eng_Latn", "eng", "English", "Inglés", "English", "", 1, 1, 8),
    ("por_Latn", "por", "Português", "Portugués", "Portuguese", "", 1, 1, 9),
    ("fra_Latn", "fra", "Français", "Francés", "French", "", 1, 1, 10),
    ("deu_Latn", "deu", "Deutsch", "Alemán", "German", "", 1, 1, 11),
    ("ita_Latn", "ita", "Italiano", "Italiano", "Italian", "", 1, 1, 12),
    ("cmn_Hans", "cmn", "中文", "Chino mandarín", "Mandarin Chinese", "", 1, 1, 13),
    ("jpn_Jpan", "jpn", "日本語", "Japonés", "Japanese", "", 1, 1, 14),
    ("kor_Hang", "kor", "한국어", "Coreano", "Korean", "", 1, 1, 15),
    ("arb_Arab", "arb", "العربية", "Árabe", "Arabic", "", 1, 1, 16),
    ("rus_Cyrl", "rus", "Русский", "Ruso", "Russian", "", 1, 1, 17),
    ("hin_Deva", "hin", "हिन्दी", "Hindi", "Hindi", "", 1, 1, 18),
    ("gug_Latn", "gug", "Avañe'ẽ", "Guaraní", "Guarani", "", 1, 0, 19),
    ("nld_Latn", "nld", "Nederlands", "Neerlandés", "Dutch", "", 1, 1, 20),
]


class MockLanguages:
    source = "mock:ui"

    def all(self) -> list:
        return [self._d(r) for r in _L]

    def get(self, code: str) -> Optional[dict]:
        return next((self._d(r) for r in _L if r[0] == code), None)

    def ui_list(self, feature: str = "asr") -> list:
        rows = [self._d(r) for r in _L]
        return [r for r in rows if not feature or r.get(feature)]

    @staticmethod
    def _d(r) -> dict:
        return {"code": r[0], "iso639_3": r[1], "name_native": r[2], "name_es": r[3], "name_en": r[4],
                "region": r[5], "asr": ["mock"] if r[6] else [], "tts": ["mock"] if r[7] else [],
                "translate": True, "whisper_code": r[1][:2], "popular": r[8]}


# --- voice ------------------------------------------------------------------------------------
STORIES = {  # mock content only
    "spa_Latn": "Había una vez un cóndor que vivía en lo alto de las montañas. Cada mañana volaba sobre "
                "el lago y saludaba a los pescadores. Un día encontró una llama perdida y la ayudó a volver "
                "con su familia.",
    "eng_Latn": "Once upon a time a condor lived high in the mountains. Every morning it flew over the lake "
                "and greeted the fishermen. One day it found a lost llama and helped it return to its family.",
    "quy_Latn": "Unay pachas huk kuntur hatun urqukunapi tiyarqa. Sapa paqarinmi qucha hawanta phawarqa. "
                "Huk punchawsi chinkasqa llamata tarispa ayllunman kutichirqa.",
    "por_Latn": "Era uma vez um condor que vivia no alto das montanhas. Todas as manhãs voava sobre o lago. "
                "Um dia encontrou uma lhama perdida e ajudou-a a voltar para a sua família.",
}


class MockVoice:
    source = "mock:ui"

    def __init__(self):
        self._tts_stop = threading.Event()
        self.said: list = []           # (text, lang) - tests read it

    def record(self, on_level=None, max_s: float = 90.0, stop_event: Optional[threading.Event] = None):
        import numpy as np
        t0 = time.time()
        speech_s = 5.0 * float(settings.MOCK_SPEED)
        while time.time() - t0 < min(max_s, speech_s):
            if stop_event is not None and stop_event.is_set():
                break
            if on_level:
                x = (time.time() - t0) * 7
                on_level(float(abs(np.sin(x)) * 0.7 + 0.1 * abs(np.sin(x * 3.1))))
            time.sleep(0.05)
        spoken = (time.time() - t0) / max(1e-3, float(settings.MOCK_SPEED))   # simulated seconds of speech
        return np.zeros(int(16000 * min(max_s, max(0.2, spoken))), dtype=np.float32)

    def transcribe(self, audio, lang: str = "auto") -> Transcript:
        msleep(1.5)
        code = "spa_Latn" if lang in ("auto", "", None) else lang
        text = STORIES.get(code, STORIES["spa_Latn"])
        cands = [["spa_Latn", 0.86], ["quy_Latn", 0.09], ["por_Latn", 0.05]] if lang in ("auto", "", None) else []
        return Transcript(text=text, lang=code, engine="mock", confidence=0.86 if cands else 0.93,
                          duration_s=round(len(audio) / 16000.0, 2), lang_candidates=cands)

    def translate(self, text: str, src: str, tgt: str) -> str:
        msleep(0.6)
        if src == tgt:
            return text
        for code, story in STORIES.items():
            if story == text and tgt in STORIES:
                return STORIES[tgt]
        return text

    def say(self, text: str, lang: str, wait: bool = False) -> bool:
        self.said.append((text, lang))
        self._tts_stop.clear()
        if wait:
            msleep(min(6.0, 0.045 * len(text)), self._tts_stop)
        return True

    def stop_tts(self) -> None:
        self._tts_stop.set()

    def synthesize(self, text: str, lang: str) -> Optional[bytes]:
        return mock_assets.wav_tone(min(8.0, 0.06 * len(text)))
