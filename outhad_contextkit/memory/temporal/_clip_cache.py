"""Module-level CLIP / SigLIP model + processor cache.

Vision-language weights are 500 MB-2 GB; loading takes 2-10 seconds on
CPU. Without a cache, both ``CLIPTextEmbedder`` and ``CLIPImageEmbedder``
(in ``text_embedders.py`` and ``image_embedders.py``) call
``CLIPModel.from_pretrained(...)`` independently → 2× the load time
+ 2× the RAM footprint.

 adds opt-in support for larger / better-aligned vision-language
backbones — CLIP-large and SigLIP — through the same cache. The
processor + model API surface (``get_text_features`` /
``get_image_features``) is identical across backbones, so the embedder
classes don't need to special-case them. Detection is by model-id
prefix:

* ``openai/clip-…``   → ``CLIPModel`` + ``CLIPProcessor``
* ``google/siglip-…`` → ``AutoModel`` + ``AutoProcessor`` (transformers ≥ 4.37)

Loads are guarded by an ``RLock`` so concurrent first-touchers share a
single load, not race two loads.

Usage::

    from outhad_contextkit.memory.temporal._clip_cache import get_clip
    model, processor = get_clip("openai/clip-vit-base-patch32")          # default
    model, processor = get_clip("openai/clip-vit-large-patch14")         # +5% recall
    model, processor = get_clip("google/siglip-base-patch16-224")        # better text-image alignment
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# quantisation dtype.
# - "fp32" (default): full precision; matches pre-C6 behaviour.
# - "fp16": half precision via ``model.half()``. ~50% RAM cut, 1.5-2×
#           faster inference on CUDA. CPU fp16 is sometimes slower
#           than fp32 so the default stays fp32.
# - "int8": dynamic int8 quantisation via torch.ao.quantization. CPU-
#           only; ~75% RAM cut on linear layers. Inference quality
#           drop is small (1-3% recall on CLIP-base) but measurable;
#           opt in for cost-bound deploys.
SUPPORTED_DTYPES = ("fp32", "fp16", "int8")

# Recognised model-id prefixes → loader strategy. Order-sensitive: the
# first prefix that matches wins.
_BACKBONE_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("openai/clip-", "clip"),
    ("google/siglip-", "siglip"),
    ("clip-", "clip"),  # bare HF ids like "clip-vit-base-patch32"
    ("siglip-", "siglip"),
)

_CACHE: Dict[str, Tuple[Any, Any]] = {}
_LOCK = threading.RLock()


def _detect_backbone(model_name: str) -> str:
    """Return the loader strategy id for ``model_name``."""
    lowered = model_name.lower()
    for prefix, kind in _BACKBONE_PREFIXES:
        if lowered.startswith(prefix):
            return kind
    # Default: assume CLIP-compatible. Caller can pin via the prefix
    # registry above when adding a new family.
    return "clip"


def _load(model_name: str) -> Tuple[Any, Any]:
    """Resolve loader class for ``model_name`` and instantiate."""
    kind = _detect_backbone(model_name)
    if kind == "siglip":
        # SigLIP ships through the generic Auto* loader — its model
        # exposes ``get_text_features`` / ``get_image_features`` so the
        # downstream embedder code path is identical to CLIP.
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading SigLIP %s (first-time)", model_name)
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    # Default — CLIP family.
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP %s (first-time)", model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def _quantise(model: Any, dtype: str) -> Any:
    """apply opt-in quantisation to a freshly-loaded model.

    ``fp32`` returns the model untouched. ``fp16`` casts via
    ``model.half()``. ``int8`` runs dynamic quantisation on linear
    layers via ``torch.ao.quantization.quantize_dynamic``. Failures
    fall back to the original fp32 model with a warning so a misconfig
    can never break the embedder path.
    """
    if dtype == "fp32":
        return model
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype!r}. Choose from {SUPPORTED_DTYPES}."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard runtime dep here
        logger.warning("torch unavailable — keeping fp32 weights (%s)", exc)
        return model

    if dtype == "fp16":
        try:
            return model.half()
        except Exception as exc:  # pragma: no cover - rare hardware quirks
            logger.warning("fp16 cast failed — keeping fp32 weights: %s", exc)
            return model

    if dtype == "int8":
        try:
            from torch.ao.quantization import quantize_dynamic

            return quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:  # pragma: no cover - layer compat
            logger.warning(
                "int8 quantisation failed — keeping fp32 weights: %s", exc
            )
            return model

    return model


def get_clip(
    model_name: str = DEFAULT_CLIP_MODEL,
    *,
    dtype: str = "fp32",
) -> Tuple[Any, Any]:
    """Return ``(model, processor)`` for ``(model_name, dtype)``.

    First call loads + caches. Subsequent calls return the cached
    instance. Lazy ``from transformers import ...`` so packages that
    don't enable TCMGM never pay the import cost.

    Supports CLIP and SigLIP; selection is by model-id prefix.

    ``dtype`` selects opt-in quantisation:
    * ``"fp32"`` (default): full precision; matches pre-C6 behaviour.
    * ``"fp16"``: ~50% RAM cut, 1.5-2× faster on CUDA.
    * ``"int8"``: dynamic int8 on linear layers; ~75% RAM cut on CPU.
    Each ``(model_name, dtype)`` pair caches independently so a single
    process can serve fp32 and fp16 builds side by side without
    re-downloading weights.
    """
    cache_key = f"{model_name}::{dtype}"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        model, processor = _load(model_name)
        model = _quantise(model, dtype)
        _CACHE[cache_key] = (model, processor)
        return model, processor


def clear_clip_cache() -> int:
    """Test helper — drop every cached pair. Returns number of entries removed."""
    with _LOCK:
        n = len(_CACHE)
        _CACHE.clear()
        return n


__all__ = [
    "get_clip",
    "clear_clip_cache",
    "DEFAULT_CLIP_MODEL",
    "SUPPORTED_DTYPES",
]
