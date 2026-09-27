"""Data shapes shared by every Yachachiq v2 module (see v2/CONTRACTS.md).

Plain dataclasses so they serialize with `to_dict` / `from_dict` and travel as
JSON between the Jetson, the MacBook, the kiosk UI, the printer and the hologram.
Language codes are canonical FLORES-200 style: ISO 639-3 + "_" + ISO 15924
script ("spa_Latn", "quy_Latn", "eng_Latn", "cmn_Hans"). Sign languages use
their ISO 639-3 code: "ase" (ASL), "prl" (Peruvian Sign Language, LSP),
"ils" (International Sign).
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def new_id(prefix: str) -> str:
    """Sortable, readable id: prefix-YYYYmmdd-HHMMSS-xxxx."""
    return "%s-%s-%s" % (prefix, time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:4])


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def from_dict(cls, data: dict):
    """Build dataclass `cls` from a dict, ignoring unknown keys (forward compatible)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


# --- Voice ----------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class Transcript:
    text: str
    lang: str                      # canonical code actually used/detected
    engine: str                    # e.g. "whisper-large-v3@mac", "omniasr-ctc-1b@mac", "mock"
    confidence: float = 1.0        # 0..1
    duration_s: float = 0.0
    segments: list = field(default_factory=list)          # list[Segment as dict]
    lang_candidates: list = field(default_factory=list)   # [[code, prob], ...] when auto-detected


# --- Sign language ----------------------------------------------------------------
@dataclass
class SignToken:
    kind: str                      # "letter" | "word"
    value: str                     # "a", "ñ", "CONDOR" (gloss) ...
    sign_lang: str                 # "ase" | "prl" | "ils"
    confidence: float
    alternatives: list = field(default_factory=list)      # [[value, prob], ...] best first, excluding value
    t_start: float = 0.0
    t_end: float = 0.0
    text: str = ""                 # spoken-language rendering, e.g. gloss CONDOR -> "cóndor"


# --- Story ----------------------------------------------------------------------
@dataclass
class StoryInput:
    story_id: str
    source: str                    # "voice" | "sign" | "text"
    text: str                      # as told (original language), after the visitor confirmed it
    lang: str                      # canonical code of `text` (sign stories: the spoken language of the glosses)
    text_es: str = ""              # Spanish version (printed on the back, shown on screen)
    text_en: str = ""              # English version (prompting)
    sign_lang: str = ""            # when source == "sign"
    confirmed: bool = False


@dataclass
class ScenePlan:
    title: str                     # short title in the story language
    title_es: str
    summary_es: str
    subject: str                   # main subject, English
    elements: list = field(default_factory=list)   # things that MUST be visible, English, used by verification
    setting: str = ""
    mood: str = ""
    cultural_notes: list = field(default_factory=list)
    prompt: str = ""               # final positive prompt, English
    negative: str = ""


@dataclass
class DrawingResult:
    story_id: str
    image: str                     # path of the chosen generated image (PNG)
    front_svg: str                 # plotter paths, millimetres
    back_svg: str                  # story text + QR, millimetres
    front_gcode: str = ""
    back_gcode: str = ""
    verified: dict = field(default_factory=dict)   # element -> bool, from the VLM check
    attempts: int = 1
    strokes: int = 0
    pen_mm: float = 0.0
    est_minutes: float = 0.0
    qr_url: str = ""


# --- Box scan --------------------------------------------------------------------
SCAN_PROFILES = ("quick", "standard", "detailed")
ANALYSES = ("weight", "photogrammetry", "rti", "uv", "thermal", "identify")


@dataclass
class ScanRequest:
    scan_id: str
    profile: str = "standard"      # one of SCAN_PROFILES
    analyses: list = field(default_factory=lambda: list(ANALYSES))
    # What the visitor says about the object before the scan (optional, short, any language):
    # {"found_where": "en una huaca cerca de Trujillo", "region_hint": "costa_norte", "notes": "", "lang": "spa_Latn"}
    # It is a CLUE for identification, never proof: see Identification.context_effect_es.
    context: dict = field(default_factory=dict)


@dataclass
class Measurement:
    name: str                      # "mass", "height", "width", "depth", "volume_envelope", "density_apparent"...
    value: float
    unit: str                      # "g", "mm", "cm3", "g/cm3"
    uncertainty: float = 0.0       # same unit, ~1 sigma
    method: str = ""
    note: str = ""


@dataclass
class Identification:
    object_type: str               # e.g. "stirrup-spout vessel"
    object_type_es: str
    material: str
    material_es: str
    culture: str                   # best candidate, e.g. "Moche"
    period: str                    # e.g. "100-800 CE"
    region: str
    confidence: float              # 0..1, calibrated as well as we can
    alternatives: list = field(default_factory=list)   # [{"culture","period","confidence","why"}]
    evidence: list = field(default_factory=list)       # human-readable reasons (Spanish)
    similar: list = field(default_factory=list)        # [{"title","culture","date","museum","url","image","score"}]
    description_es: str = ""
    engine: str = ""
    # How the visitor's context (ScanRequest.context) changed the answer, in Spanish, e.g.
    # "El lugar ayudó a decidir entre Moche y Chimú" or "El lugar no coincide con lo que se ve; se priorizó la imagen".
    context_effect_es: str = ""
    image_only: Optional[dict] = None   # {"culture","period","material","confidence"} identified WITHOUT the context


@dataclass
class ScanResult:
    scan_id: str
    folder: str
    profile: str
    started: float
    finished: float = 0.0
    measurements: list = field(default_factory=list)   # list[Measurement as dict]
    artifacts: dict = field(default_factory=dict)      # name -> relative path inside the scan folder
    findings: list = field(default_factory=list)       # [{"analysis","title_es","detail_es","severity","image"}]
    identification: Optional[dict] = None              # Identification as dict
    warnings: list = field(default_factory=list)
    ok: bool = True


# --- Progress (UI) -------------------------------------------------------------------
@dataclass
class Progress:
    stage: str                     # machine name, e.g. "rotating", "rti", "generating"
    fraction: float                # 0..1 inside the whole task
    message_es: str = ""
    detail: dict = field(default_factory=dict)
