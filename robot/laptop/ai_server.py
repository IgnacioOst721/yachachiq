"""Optional laptop AI server (the "remote" backends): Ollama story analysis + diffusers Stable Diffusion.

You normally do NOT need this: the robot talks to ComfyUI directly (config.IMAGE_BACKENDS
starts with "comfyui"). Run this only if you want Ollama analysis or have no ComfyUI.

    python ai_server.py      -> http://0.0.0.0:8600   (/health, /analyze, /generate)
"""
import io
import json
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Yachachiq laptop AI")
OLLAMA = "http://127.0.0.1:11434"
MODEL = "llama3.2"
_pipe = None


@app.get("/health")
async def health():
    return {"ok": True, "ollama": MODEL, "sd": _pipe is not None}


@app.post("/analyze")
async def analyze(req: Request):
    import requests
    b = await req.json()
    keys = ("person condor llama puma cat dog bird fish bear fox snake mountain river lake sea sun moon star "
            "cloud rain tree flower corn potato field house bridge path temple boat fire music")
    prompt = ("Reply ONLY with JSON keys title (Spanish), elements (list from: %s), scene (Spanish sentence), "
              "image_prompt (English, black ink line drawing, no text), narration (2-3 warm Spanish sentences).\n\nStory:\n%s"
              % (keys, b.get("text", "")))
    r = requests.post(f"{OLLAMA}/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json"}, timeout=120)
    r.raise_for_status()
    m = re.search(r"\{.*\}", r.json().get("response", ""), re.S)
    return JSONResponse(json.loads(m.group(0)) if m else {})


@app.post("/generate")
async def generate(req: Request):
    global _pipe
    b = await req.json()
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev)
    size = int(b.get("size", 512))
    img = _pipe(prompt=b.get("prompt", ""), negative_prompt=b.get("negative_prompt", ""), num_inference_steps=4,
                guidance_scale=0.0, width=size, height=size).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600, log_level="warning")
