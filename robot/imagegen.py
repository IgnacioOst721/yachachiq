"""Picture for the story, then pen strokes in millimetres.

Backends (config.IMAGE_BACKENDS, tried in order):
  comfyui  ComfyUI on the laptop: submit laptop/comfyui_workflow.json with the story
           prompt patched in, wait for the image, download it
  remote   laptop/ai_server.py /generate (diffusers Stable Diffusion)
  motifs   offline procedural Andean scene from the story's elements, cannot fail

generate(analysis) -> dict(source, png_path|None, polylines_mm, svg, stats)
"""
import copy
import json
import logging
import random
import time
import uuid
from pathlib import Path

import config
import motifs
import vectorize

log = logging.getLogger("imagegen")


def _out_dir():
    d = Path(config.OUTPUT_DIR) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prompt(analysis):
    # the SUBJECT goes first: Stable Diffusion weighs the first tokens most, and with the style
    # words in front it drew "Andean folk art" and forgot the story
    subject = analysis.get("image_prompt") or analysis.get("scene") or "an Andean landscape"
    return f"a simple line drawing of {subject}, {config.STYLE_PROMPT}"


def _trace_with_border(png):
    """Trace an image file to strokes and frame it with the Andean border."""
    pls_px, w, h = vectorize.trace_image(png)
    if len(pls_px) < 5:
        raise RuntimeError("image traced to almost nothing")
    pls_mm = vectorize.fit_to_paper(pls_px, w, h)
    if not config.AI_BORDER:
        # no frame: on the plotter the stepped border was the longest, thickest thing on the
        # page and the actual drawing ended up squeezed inside it
        return pls_mm
    inset = min(config.PAPER_W_MM, config.PAPER_H_MM) * 0.075
    border = motifs.stepped_border(config.PAPER_W_MM, config.PAPER_H_MM, inset)
    return border + pls_mm


# --- ComfyUI ---------------------------------------------------------------------------------

def patch_workflow(workflow, positive, negative, size=None, seed=None, checkpoint=None, steps=None, cfg=None):
    """Return a copy of an API-format ComfyUI workflow with our prompt, size, seed, checkpoint.

    Positive/negative CLIPTextEncode nodes are found by following the KSampler's
    'positive' and 'negative' links, so any ordinary txt2img workflow works unchanged.
    """
    wf = copy.deepcopy(workflow)
    samplers = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type", "").startswith("KSampler")]
    if not samplers:
        raise ValueError("workflow has no KSampler node; export it in API format (Save (API Format))")
    for s in samplers:
        pos_id = s["inputs"].get("positive", [None])[0]
        neg_id = s["inputs"].get("negative", [None])[0]
        if pos_id in wf and "text" in wf[pos_id]["inputs"]:
            wf[pos_id]["inputs"]["text"] = positive
        if neg_id in wf and "text" in wf[neg_id]["inputs"]:
            wf[neg_id]["inputs"]["text"] = negative
        if "seed" in s["inputs"]:
            s["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**31 - 1)
        if "noise_seed" in s["inputs"]:
            s["inputs"]["noise_seed"] = seed if seed is not None else random.randint(0, 2**31 - 1)
        if steps:
            s["inputs"]["steps"] = int(steps)
        if cfg and "cfg" in s["inputs"]:
            s["inputs"]["cfg"] = float(cfg)
    for n in wf.values():
        if not isinstance(n, dict):
            continue
        if n.get("class_type") == "EmptyLatentImage" and size:
            w, h = size if isinstance(size, (tuple, list)) else (int(size), int(size))
            n["inputs"]["width"], n["inputs"]["height"] = int(w), int(h)
        if n.get("class_type") == "CheckpointLoaderSimple" and checkpoint:
            n["inputs"]["ckpt_name"] = checkpoint
    return wf


def comfyui_generate(positive, negative, out_path, url=None, workflow_path=None, size=None, timeout=None):
    """Run the workflow on ComfyUI and save the first output image to out_path."""
    import requests
    url = (url or config.COMFYUI_URL).rstrip("/")
    timeout = timeout or config.COMFYUI_TIMEOUT
    with open(workflow_path or config.COMFYUI_WORKFLOW, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    if "nodes" in workflow and "links" in workflow:
        raise ValueError("this is a UI-format workflow; in ComfyUI use 'Save (API Format)' (enable dev mode in settings)")
    wf = patch_workflow(workflow, positive, negative, size=size or config.image_wh(),
                        checkpoint=config.COMFYUI_CHECKPOINT or None, steps=config.COMFYUI_STEPS or None,
                        cfg=getattr(config, "COMFYUI_CFG", 0) or None)

    client_id = uuid.uuid4().hex
    r = requests.post(f"{url}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=(3, 30))
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:300]}")
    prompt_id = r.json()["prompt_id"]
    log.info("comfyui job %s queued", prompt_id[:8])

    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{url}/history/{prompt_id}", timeout=(3, 30)).json()
        entry = h.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI workflow failed: " + json.dumps(status.get("messages", []))[:300])
            for node_out in entry.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    v = requests.get(f"{url}/view", params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                                            "type": img.get("type", "output")}, timeout=(3, 60))
                    v.raise_for_status()
                    Path(out_path).write_bytes(v.content)
                    log.info("comfyui image in %.1fs", time.time() - t0)
                    return out_path
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI took longer than {timeout:.0f}s")


def from_comfyui(analysis):
    png = _out_dir() / f"comfy_{int(time.time())}.png"
    urls = [config.COMFYUI_URL] + [u for u in config.laptop_urls(8188) if u != config.COMFYUI_URL]
    last = None
    for i, u in enumerate(urls):
        try:
            comfyui_generate(_prompt(analysis), config.NEGATIVE_PROMPT, png, url=u)
            if i:
                log.info("comfyui reached at %s (the first address did not answer)", u)
            break
        except Exception as e:
            last = e
            log.info("comfyui at %s: %s", u, str(e)[:90])
    else:
        raise last
    return {"source": "comfyui", "png_path": str(png), "polylines_mm": _trace_with_border(png)}


# --- diffusers server ------------------------------------------------------------------------

def from_remote(analysis):
    import requests
    r = requests.post(f"{config.AI_SERVER_URL.rstrip('/')}/generate",
                      json={"prompt": _prompt(analysis), "negative_prompt": config.NEGATIVE_PROMPT, "size": list(config.image_wh())},
                      timeout=(3, config.AI_SERVER_TIMEOUT))
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError(r.text[:200])
    png = _out_dir() / f"sd_{int(time.time())}.png"
    png.write_bytes(r.content)
    return {"source": "stable-diffusion", "png_path": str(png), "polylines_mm": _trace_with_border(png)}


# --- offline motifs ------------------------------------------------------------------------------

def from_motifs(analysis):
    elements = analysis.get("elements") or []
    pls = motifs.compose(elements, config.PAPER_W_MM, config.PAPER_H_MM, seed=hash(analysis.get("title", "")) & 0xFFFF)
    png = _out_dir() / f"motifs_{int(time.time())}.png"
    try:
        vectorize.to_png(pls, png)
        png_path = str(png)
    except Exception:
        png_path = None
    return {"source": "motifs", "png_path": png_path, "polylines_mm": pls}


BACKENDS = {"comfyui": from_comfyui, "remote": from_remote, "motifs": from_motifs}


def _finish(out, errors):
    out["polylines_mm"] = vectorize.clamp(vectorize.order_strokes(out["polylines_mm"]))
    out["svg"] = vectorize.to_svg(out["polylines_mm"])
    out["stats"] = vectorize.stats(out["polylines_mm"])
    out["errors"] = errors
    return out


def generate(analysis, backends=None, on_status=None):
    errors = []
    for name in (backends or config.IMAGE_BACKENDS):
        fn = BACKENDS.get(name)
        if fn is None:
            continue
        try:
            if on_status:
                on_status(name)
            return _finish(fn(analysis), errors)
        except Exception as e:
            log.warning("image backend '%s' failed: %s", name, e)
            errors.append(f"{name}: {e}")
    return _finish(from_motifs(analysis), errors)
