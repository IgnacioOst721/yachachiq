"""Store-and-forward publishing of stories to the web gallery (GitHub Pages).

Every story lives in config.STORIES_DIR/<timestamp>/ (story.txt, scene_1.png,
scene_1_photo.png). sync() copies the new ones into <repo>/docs/stories/,
rebuilds docs/stories.json and pushes. No internet -> nothing happens, the
stories wait on the SD card and go up next time. Never raises.
"""
import json
import logging
import os
import shutil
import subprocess

import config

log = logging.getLogger("publish")


def mode():
    if not config.PUBLISH_ENABLED:
        return "off"
    if config.MOCK:
        return "mock"
    return "git"


def have_internet(timeout=4):
    try:
        subprocess.run(["git", "ls-remote", "--exit-code", "-h", "origin"], cwd=str(config.PUBLISH_REPO_DIR),
                       capture_output=True, timeout=timeout, check=True)
        return True
    except Exception:
        return False


def _index(web_dir):
    out = []
    for name in sorted(os.listdir(web_dir), reverse=True):
        d = os.path.join(web_dir, name)
        if not os.path.isdir(d):
            continue
        e = {"id": name, "images": [], "photos": [], "portrait": None, "story": ""}
        for f in sorted(os.listdir(d)):
            if f == "story.txt":
                with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
                    e["story"] = fh.read()
            elif f.startswith("storyteller_photo"):
                e["portrait"] = f
            elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                (e["photos"] if "photo" in f else e["images"]).append(f)
        out.append(e)
    return out


def sync(stories_dir=None):
    """Returns dict(status, new). status: off | mock | offline | nothing | published | error."""
    if mode() == "off":
        return {"status": "off", "new": 0}
    stories_dir = str(stories_dir or config.STORIES_DIR)
    if mode() == "mock":
        return {"status": "mock", "new": len(os.listdir(stories_dir)) if os.path.isdir(stories_dir) else 0}
    repo = str(config.PUBLISH_REPO_DIR)
    web = os.path.join(repo, "docs", "stories")
    try:
        if not os.path.isdir(stories_dir) or not os.path.isdir(os.path.join(repo, ".git")):
            return {"status": "error", "new": 0, "error": "no stories dir or repo"}
        if not have_internet():
            return {"status": "offline", "new": 0}
        os.makedirs(web, exist_ok=True)
        subprocess.run(["git", "pull", "--rebase", "-q"], cwd=repo, capture_output=True, timeout=60)
        new = 0
        for name in sorted(os.listdir(stories_dir)):
            src, dst = os.path.join(stories_dir, name), os.path.join(web, name)
            if os.path.exists(os.path.join(src, ".private")):
                continue                                   # the storyteller said no: stays on the robot
            if os.path.isdir(src) and not os.path.exists(dst) and os.path.exists(os.path.join(src, "story.txt")):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.gcode", "*.wav", "*.json", ".private"))
                new += 1
        with open(os.path.join(repo, "docs", "stories.json"), "w", encoding="utf-8") as f:
            json.dump(_index(web), f, ensure_ascii=False, indent=1)
        subprocess.run(["git", "add", "docs"], cwd=repo, check=True, capture_output=True)
        r = subprocess.run(["git", "commit", "-q", "-m", f"Publicar {new} historia(s)"], cwd=repo, capture_output=True, text=True)
        if "nothing to commit" in (r.stdout + r.stderr):
            return {"status": "nothing", "new": 0}
        subprocess.run(["git", "push", "-q"], cwd=repo, check=True, capture_output=True, timeout=120)
        log.info("published %d new stories", new)
        return {"status": "published", "new": new}
    except Exception as e:
        log.warning("publish failed: %s", e)
        return {"status": "error", "new": 0, "error": str(e)}
