"""Story folders and the web gallery format (GitHub Pages: docs/index.html + docs/stories.json).

The gallery page (v1, still live) reads docs/stories.json entries shaped like
    {"id": "2026-06-11_14-22-10", "images": ["scene_1.png"], "photos": [...],
     "portrait": "storyteller_photo.jpg" | null, "story": "<text of story.txt>"}
and parses story.txt ("STORY\\n===\\n<text>\\n\\nSCENES ... TITLE: ...\\nLANG: ...").
v2 keeps all of that and only ADDS fields (from docs/stories/<id>/info.json):
    "v": 2, "title", "lang", "lang_name", "source", "sign_lang", "sign_lang_name",
    "text_es", "qr_url"
so the existing page keeps working and a future page can use the new data.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from yq.common import config

WEB_FILES = ("story.txt", "scene_1.png", "storyteller_photo.jpg")   # what goes to the web, nothing else
SIGN_NAMES = {"ase": "Lengua de señas americana (ASL)", "prl": "Lengua de Señas Peruana (LSP)",
              "ils": "Señas Internacionales"}


def web_id(story_id: str) -> str:
    """story-20260926-201500-ab12 -> 2026-09-26_20-15-00_ab12 (sorts with v1 ids, the page shows the date)."""
    m = re.match(r"^[a-z]+-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-(\w+)$", story_id or "")
    if m:
        y, mo, d, h, mi, s, tail = m.groups()
        return "%s-%s-%s_%s-%s-%s_%s" % (y, mo, d, h, mi, s, tail)
    return re.sub(r"[^A-Za-z0-9_-]", "_", story_id or "historia")


def story_url(story_id: str, base: Optional[str] = None) -> str:
    """Where the QR points: the gallery page, anchored at this story."""
    base = (base or config.PUBLIC_BASE_URL).rstrip("/") + "/"
    return base + "#" + web_id(story_id)


def story_txt(text: str, text_es: str = "", title: str = "", summary_es: str = "",
              lang: str = "", lang_name_es: str = "", source: str = "", sign_lang: str = "") -> str:
    """story.txt in the v1 format the gallery parses (text before SCENES is shown)."""
    body = (text or "").strip()
    if text_es and text_es.strip() and text_es.strip() != body:
        body += "\n\n— En español —\n" + text_es.strip()
    lang_tag = ("%s (%s)" % (lang_name_es.lower(), lang)) if lang_name_es else lang
    lines = ["STORY", "=" * 40, body, "", "SCENES", "=" * 40, "1. " + (summary_es or "").strip(), "",
             "TITLE: " + (title or "").strip(), "LANG: " + lang_tag, "SOURCE: " + (source or "")]
    if sign_lang:
        lines.append("SIGN: %s (%s)" % (SIGN_NAMES.get(sign_lang, sign_lang), sign_lang))
    return "\n".join(lines) + "\n"


def info_json(meta: dict) -> dict:
    """The extra v2 fields added to a gallery entry (subset of story.json, never private data)."""
    st = meta.get("story") or {}
    return {"v": 2, "title": meta.get("title") or "", "lang": st.get("lang") or "",
            "lang_name": meta.get("lang_name_es") or "", "source": st.get("source") or "",
            "sign_lang": st.get("sign_lang") or "", "sign_lang_name": SIGN_NAMES.get(st.get("sign_lang") or "", ""),
            "text_es": st.get("text_es") or "", "qr_url": meta.get("qr_url") or ""}


def index(web_dir: Path) -> list:
    """Rebuild the stories.json list from docs/stories/ (same rules as v1, plus info.json fields)."""
    out = []
    web_dir = Path(web_dir)
    if not web_dir.is_dir():
        return out
    for name in sorted(os.listdir(web_dir), reverse=True):
        d = web_dir / name
        if not d.is_dir():
            continue
        e = {"id": name, "images": [], "photos": [], "portrait": None, "story": ""}
        for f in sorted(os.listdir(d)):
            if f == "story.txt":
                e["story"] = (d / f).read_text(encoding="utf-8", errors="ignore")
            elif f.startswith("storyteller_photo"):
                e["portrait"] = f
            elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                (e["photos"] if "photo" in f else e["images"]).append(f)
        info = d / "info.json"
        if info.exists():
            try:
                extra = json.loads(info.read_text(encoding="utf-8"))
                for k, v in extra.items():
                    if k not in e:                      # never override the v1 fields
                        e[k] = v
            except ValueError:
                pass
        out.append(e)
    return out


def redirect_html(story_id: str) -> str:
    """docs/<story_id>/index.html: ART prints PUBLIC_BASE_URL + story_id on the paper QR (a path);
    this tiny page sends that URL to the gallery, anchored at the story."""
    target = "../#" + web_id(story_id)
    return ('<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0; url=%s"><title>Yachachiq</title>'
            '<link rel="canonical" href="%s"></head><body><a href="%s">Ver la historia en el archivo de Yachachiq</a>'
            '</body></html>\n' % (target, target, target))
