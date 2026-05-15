"""Phase C6 — opt-in fp16 / int8 quantisation for the CLIP / SigLIP cache.

Tests cover the dtype kwarg flowing from the factory → embedder →
``_clip_cache.get_clip``, the ``_quantise`` dispatch (fp32 untouched,
fp16 calls ``model.half()``, int8 routes through
``torch.ao.quantization.quantize_dynamic``), and that
``(model_name, dtype)`` pairs are cached independently so a process
can serve fp32 and fp16 builds side by side.

No real weights are downloaded — every from_pretrained call is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _wipe_cache():
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    yield
    clear_clip_cache()


# ---------------------------------------------------------------------------
# _quantise dispatch
# ---------------------------------------------------------------------------

def test_quantise_fp32_returns_model_unchanged():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import _quantise

    sentinel = MagicMock(name="model")
    out = _quantise(sentinel, "fp32")
    assert out is sentinel
    sentinel.half.assert_not_called()


def test_quantise_fp16_calls_model_half():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import _quantise

    model = MagicMock(name="model")
    model.half.return_value = "fp16-model"
    out = _quantise(model, "fp16")
    assert out == "fp16-model"
    model.half.assert_called_once_with()


def test_quantise_fp16_falls_back_on_failure():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import _quantise

    model = MagicMock(name="model")
    model.half.side_effect = RuntimeError("no fp16 hw")
    out = _quantise(model, "fp16")
    assert out is model  # fallback to fp32


def test_quantise_unknown_dtype_raises():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import _quantise

    with pytest.raises(ValueError):
        _quantise(MagicMock(), "fp64")


def test_quantise_int8_routes_through_torch_dynamic():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from outhad_contextkit.memory.temporal._clip_cache import _quantise

    model = MagicMock(name="model")
    with patch(
        "torch.ao.quantization.quantize_dynamic", return_value="int8-model"
    ) as m_quant:
        out = _quantise(model, "int8")
    assert out == "int8-model"
    m_quant.assert_called_once()


# ---------------------------------------------------------------------------
# get_clip dtype plumbing + per-dtype cache isolation
# ---------------------------------------------------------------------------

def test_get_clip_default_dtype_is_fp32():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    fake_model = MagicMock(name="model")
    with patch(
        "transformers.CLIPModel.from_pretrained", return_value=fake_model
    ), patch("transformers.CLIPProcessor.from_pretrained", return_value="proc"):
        out_model, _ = get_clip("openai/clip-vit-base-patch32")

    fake_model.half.assert_not_called()
    assert out_model is fake_model


def test_get_clip_fp16_calls_half_once():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    fake_model = MagicMock(name="model")
    fake_model.half.return_value = "fp16-model"
    with patch(
        "transformers.CLIPModel.from_pretrained", return_value=fake_model
    ), patch("transformers.CLIPProcessor.from_pretrained", return_value="proc"):
        out_model, _ = get_clip(
            "openai/clip-vit-base-patch32", dtype="fp16"
        )

    fake_model.half.assert_called_once_with()
    assert out_model == "fp16-model"


def test_get_clip_caches_fp32_and_fp16_independently():
    """Same model_name with different dtype must NOT collide in the cache."""
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()

    def fresh_model(*args, **kwargs):
        m = MagicMock()
        m.half.return_value = m  # treat half() as identity for the test
        return m

    with patch(
        "transformers.CLIPModel.from_pretrained", side_effect=fresh_model
    ) as m_load, patch(
        "transformers.CLIPProcessor.from_pretrained", return_value="proc"
    ):
        get_clip("openai/clip-vit-base-patch32", dtype="fp32")
        get_clip("openai/clip-vit-base-patch32", dtype="fp32")  # cache hit
        get_clip("openai/clip-vit-base-patch32", dtype="fp16")  # different key
        get_clip("openai/clip-vit-base-patch32", dtype="fp16")  # cache hit

    assert m_load.call_count == 2  # one per dtype


def test_get_clip_unknown_dtype_raises():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch(
        "transformers.CLIPModel.from_pretrained", return_value=MagicMock()
    ), patch("transformers.CLIPProcessor.from_pretrained", return_value="proc"):
        with pytest.raises(ValueError):
            get_clip("openai/clip-vit-base-patch32", dtype="bogus")


# ---------------------------------------------------------------------------
# Embedder + factory plumbing
# ---------------------------------------------------------------------------

def test_clip_text_embedder_passes_dtype_to_get_clip():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.text_embedders import (
            CLIPTextEmbedder,
        )

        e = CLIPTextEmbedder(model_name="openai/clip-vit-base-patch32", dtype="fp16")
        assert e.dtype == "fp16"
        m_get_clip.assert_called_once_with(
            "openai/clip-vit-base-patch32", dtype="fp16"
        )
    clear_clip_cache()


def test_clip_image_embedder_passes_dtype_to_get_clip():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.image_embedders import (
            CLIPImageEmbedder,
        )

        e = CLIPImageEmbedder(model_name="openai/clip-vit-large-patch14", dtype="int8")
        assert e.dtype == "int8"
        m_get_clip.assert_called_once_with(
            "openai/clip-vit-large-patch14", dtype="int8"
        )
    clear_clip_cache()


def test_factory_threads_clip_dtype_through_both_embedders():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.embedder_factory import (
            create_embedder,
        )

        emb = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="whisper",
            openai_api_key="sk-test",
            clip_dtype="fp16",
        )

        assert emb.text_embedder.dtype == "fp16"
        assert emb.image_embedder.dtype == "fp16"
        assert all(
            call.kwargs.get("dtype") == "fp16"
            for call in m_get_clip.call_args_list
        )
    clear_clip_cache()


def test_create_fp16_embedder_alias():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.embedder_factory import (
            create_fp16_embedder,
        )

        emb = create_fp16_embedder(openai_api_key="sk-test")
        assert emb.text_embedder.dtype == "fp16"
        assert emb.image_embedder.dtype == "fp16"
    clear_clip_cache()


def test_config_to_dict_exposes_clip_dtype():
    from outhad_contextkit.memory.temporal.embedder_factory import EmbedderConfig

    cfg = EmbedderConfig(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="whisper",
        clip_dtype="fp16",
        openai_api_key="sk-test",
    )
    d = cfg.to_dict()
    assert d["clip_dtype"] == "fp16"
    assert d["openai_api_key"] == "***"
