"""BOX-ANALYSIS settings (env YQ_<NAME>, see yq.common.config.env)."""
from __future__ import annotations

from yq.common.config import env

# Image embedding model for the museum catalog (keys in embed.MODELS).
EMBED_MODEL = env("BOX_EMBED_MODEL", "siglip2-base-384")   # the 14k catalog (and its evaluation) uses base
# Allow Hugging Face downloads when a model is missing (development only; the robot runs offline).
ALLOW_DOWNLOAD = env("BOX_ALLOW_DOWNLOAD", True)
# Retrieval: neighbours used for voting and how many references are shown to the visitor.
KNN = env("BOX_KNN", 40)
SHOW_SIMILAR = env("BOX_SHOW_SIMILAR", 6)
# Use ART's VLM (yq.macworker.models.vlm.ask) when it is available.
USE_VLM = env("BOX_USE_VLM", True)
VLM_MAX_TOKENS = env("BOX_VLM_MAX_TOKENS", 700)
# Visitor context prior: maximum multiplicative boost of a culture score (see context.py).
CONTEXT_MAX_BOOST = env("BOX_CONTEXT_MAX_BOOST", 1.5)
# Heavy parts of analyze_scan.
RECON_TARGET_FACES = env("BOX_RECON_FACES", 150000)
REFINE_SURFACE = env("BOX_REFINE_SURFACE", True)
