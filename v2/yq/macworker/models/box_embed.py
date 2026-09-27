"""BOX-ANALYSIS models on the Mac worker: SigLIP 2 image embedder for the museum catalog."""
from __future__ import annotations


def register(models) -> list:
    """Register every embedding model of yq.box.analysis.embed.MODELS as "box-<key>" (fp16 vision tower;
    size_gb measured on the M4 including activations for batch 32)."""
    from yq.box.analysis.embed import MODELS, Embedder
    names = []
    for key, spec in MODELS.items():
        name = "box-" + key
        models.register(name, (lambda k=key: Embedder(k)), size_gb=spec["size_gb"],
                        unloader=lambda obj: obj.close(), domain="box_analysis", repo=spec["repo"])
        names.append(name)
    return names
