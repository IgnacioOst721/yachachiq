"""Image generators for the front drawing (Mac).

    generate(prompt, width, height, seed, backend=None, negative="") -> PIL.Image (RGB)

Backends (YQ_ART_IMAGE_BACKEND; all Apache-2.0 weights, pre-quantized 4-bit for MLX via mflux):
  z-image-turbo   Tongyi-MAI Z-Image-Turbo 6B, mflux-community q4 (~5.9 GB), 9 steps
  flux2-klein-4b  BFL FLUX.2 [klein] 4B, mflux-community q4 (~4.6 GB), 4 steps
  schnell         BFL FLUX.1-schnell 12B, mflux-community q4 (~9.6 GB), 4 steps
  comfyui         v1 fallback: DreamShaper 8 (SD1.5) in the ComfyUI server at ~/ComfyUI (port 8188)
Mock (YQ_MOCK=1 / YQ_MOCK_IMAGEGEN=1): procedural Andean motifs from the words in the prompt.
"""
from __future__ import annotations

import copy
import json
import logging
import random
import time
import uuid

from yq.common import config
from yq.common.config import env
from yq.macworker.modelmgr import models

from .art_runtime import local_repo, run

log = logging.getLogger("yq.art.image")

BACKENDS = {
    "z-image-turbo": {"repo": "mflux-community/z-image-turbo-mflux-q4", "size_gb": 6.6, "steps": 9, "family": "zimage"},
    "flux2-klein-4b": {"repo": "mflux-community/flux2-klein-4b-mflux-q4", "size_gb": 5.4, "steps": 4, "family": "flux2"},
    "schnell": {"repo": "mflux-community/flux-1-schnell-mflux-q4", "size_gb": 10.2, "steps": 4, "family": "flux1"},
    "comfyui": {"size_gb": 0.0, "steps": 22},
}
BACKEND = env("ART_IMAGE_BACKEND", "flux2-klein-4b")
COMFYUI_URL = env("ART_COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_CHECKPOINT = env("ART_COMFYUI_CHECKPOINT", "dreamshaper_8.safetensors")
COMFYUI_FREE_AFTER = env("ART_COMFYUI_FREE_AFTER", True)


def _load(name: str):
    from mflux.models.common.config import ModelConfig
    b = BACKENDS[name]
    path = local_repo(b["repo"])
    if b["family"] == "zimage":
        from mflux.models.z_image import ZImage
        return ZImage(model_config=ModelConfig.z_image_turbo(), model_path=path)
    if b["family"] == "flux2":
        from mflux.models.flux2.variants import Flux2Klein
        return Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), model_path=path)
    from mflux.models.flux.variants.txt2img.flux import Flux1
    return Flux1(model_config=ModelConfig.schnell(), model_path=path)


def register(name: str) -> str:
    mname = "img:" + name
    if mname not in models.registered():
        models.register(mname, loader=lambda: _load(name), size_gb=BACKENDS[name]["size_gb"],
                        repo=BACKENDS[name].get("repo", ""), kind="image", domain="art")
    return mname


CACHE_LIMIT_GB = env("ART_MLX_CACHE_GB", 1.0)


class Cancelled(RuntimeError):
    pass


class _StepHook:
    """mflux in-loop callback: reports denoising progress and stops early when cancelled."""
    def __init__(self):
        self.progress = None
        self.cancelled = None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps=None, **kw):
        if self.cancelled and self.cancelled():
            raise Cancelled("image generation cancelled")
        if self.progress:
            n = max(1, int(getattr(config, "num_inference_steps", 1) or 1))
            self.progress(min(1.0, (int(t) + 1) / float(n)))


def _mflux_generate(name: str, prompt: str, width: int, height: int, seed: int, steps: int,
                    progress=None, cancelled=None):
    import mlx.core as mx
    m = models.get(register(name))
    hook = getattr(m, "_yq_hook", None)
    if hook is None:
        hook = _StepHook()
        m.callbacks.register(hook)
        m._yq_hook = hook
    hook.progress, hook.cancelled = progress, cancelled
    # MLX keeps freed buffers in a cache that grows every denoising step; on the shared 16 GB
    # Mac that pushed Z-Image to 11.5 GB peak and into swap. Cap it (mflux --low-ram does the same).
    mx.set_cache_limit(int(CACHE_LIMIT_GB * 1e9))
    kw = dict(seed=int(seed), prompt=prompt, num_inference_steps=int(steps), width=int(width), height=int(height))
    if BACKENDS[name]["family"] == "flux2":
        kw["guidance"] = 1.0
    # Keep the weights resident while generating (mlx-lm does the same for text): without a wired
    # limit macOS compressed 4.7 of the 5.2 GB of weights on the shared Mac and a 4-step image
    # took 20 minutes instead of 50 s (measured).
    old_limit = mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    try:
        out = m.generate_image(**kw)
    finally:
        hook.progress = hook.cancelled = None
        mx.synchronize()
        mx.set_wired_limit(old_limit)
        mx.clear_cache()
    img = getattr(out, "image", out)
    return img.convert("RGB")


def generate(prompt: str, width: int = 768, height: int = 1088, seed: int = None, backend: str = None,
             negative: str = "", steps: int = None, progress=None, cancelled=None):
    """progress(fraction 0..1) is called after every denoising step; cancelled() -> True stops."""
    backend = backend or BACKEND
    seed = random.randint(0, 2 ** 31 - 1) if seed is None else int(seed)
    if config.mock("imagegen"):
        return mock_image(prompt, width, height, seed)
    if backend == "comfyui":
        return comfyui_generate(prompt, negative, width, height, seed, steps)
    if backend not in BACKENDS:
        raise KeyError("unknown image backend %r (known: %s)" % (backend, sorted(BACKENDS)))
    w, h = (int(width) // 16) * 16, (int(height) // 16) * 16
    return run(_mflux_generate, backend, prompt, w, h, seed, steps or BACKENDS[backend]["steps"],
               progress, cancelled)


# --- ComfyUI (v1 robot/imagegen.py, adapted) ---------------------------------------------------

def _workflow(positive: str, negative: str, w: int, h: int, seed: int, steps: int, cfg: float) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": COMFYUI_CHECKPOINT}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                                                    "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                                    "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0],
                                                    "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "yachachiq_v2", "images": ["8", 0]}},
    }


def comfyui_generate(positive: str, negative: str, width: int, height: int, seed: int, steps: int = None,
                     timeout: float = 240.0):
    """SD1.5 works best near 512 px: generate at ~512 x 728 and let the vectorizer upsample."""
    import io
    import requests
    from PIL import Image
    s = 728.0 / max(width, height)
    w, h = max(256, int(width * s) // 8 * 8), max(256, int(height * s) // 8 * 8)
    wf = _workflow(positive, negative, w, h, seed, steps or 22, 7.0)
    url = COMFYUI_URL.rstrip("/")
    r = requests.post(url + "/prompt", json={"prompt": copy.deepcopy(wf), "client_id": uuid.uuid4().hex},
                      timeout=(3, 30))
    if r.status_code != 200:
        raise RuntimeError("ComfyUI rejected the workflow: %s" % r.text[:300])
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        entry = requests.get(url + "/history/" + pid, timeout=(3, 30)).json().get(pid)
        if entry:
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError("ComfyUI failed: " + json.dumps(st.get("messages", []))[:300])
            for node in entry.get("outputs", {}).values():
                for im in node.get("images", []):
                    v = requests.get(url + "/view", params={"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                            "type": im.get("type", "output")}, timeout=(3, 60))
                    v.raise_for_status()
                    img = Image.open(io.BytesIO(v.content)).convert("RGB")
                    if COMFYUI_FREE_AFTER:            # ComfyUI keeps up to ~9 GB otherwise (measured)
                        try:
                            requests.post(url + "/free", json={"unload_models": True, "free_memory": True},
                                          timeout=5)
                        except Exception:
                            pass
                    return img
        time.sleep(0.4)
    raise TimeoutError("ComfyUI took longer than %.0f s" % timeout)


def mock_image(prompt: str, width: int, height: int, seed: int):
    """Procedural motif scene as a line image: exercises the whole pipeline with no model."""
    import numpy as np
    from PIL import Image
    from yq.art import motifs, offline
    from yq.art.geom import render
    els = offline.find_elements(prompt) or ["mountain", "sun", "llama"]
    strokes = motifs.compose_page(els, (width * 0.06, height * 0.06, width * 0.94, height * 0.94), seed=seed)
    img = render(strokes, int(width), int(height), width_px=max(2, width // 256))
    return Image.fromarray(np.stack([img] * 3, axis=2))
