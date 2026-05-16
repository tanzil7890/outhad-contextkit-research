""" opt-in larger CLIP / SigLIP backbone.

These tests verify the *plumbing* (model_name flows through factory →
embedder → ``_clip_cache.get_clip``) and the *backbone detection* (CLIP
vs SigLIP loaders are picked correctly). They do **not** download any
weights — ``transformers.CLIPModel.from_pretrained`` /
``transformers.AutoModel.from_pretrained`` are patched to return
sentinels so the test runs in milliseconds and works offline.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _clip_cache backbone detection + dispatch
# ---------------------------------------------------------------------------

def test_clip_cache_default_dispatches_to_clipmodel():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch("transformers.CLIPModel.from_pretrained") as m_model, patch(
        "transformers.CLIPProcessor.from_pretrained"
    ) as m_proc:
        m_model.return_value = "clip-model"
        m_proc.return_value = "clip-proc"
        model, proc = get_clip("openai/clip-vit-base-patch32")

    assert (model, proc) == ("clip-model", "clip-proc")
    m_model.assert_called_once_with("openai/clip-vit-base-patch32")
    m_proc.assert_called_once_with("openai/clip-vit-base-patch32")
    clear_clip_cache()


def test_clip_cache_large_dispatches_to_clipmodel():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch("transformers.CLIPModel.from_pretrained") as m_model, patch(
        "transformers.CLIPProcessor.from_pretrained"
    ) as m_proc:
        m_model.return_value = "clip-large-model"
        m_proc.return_value = "clip-large-proc"
        model, proc = get_clip("openai/clip-vit-large-patch14")

    assert model == "clip-large-model"
    m_model.assert_called_once_with("openai/clip-vit-large-patch14")
    clear_clip_cache()


def test_clip_cache_siglip_dispatches_to_automodel():
    """SigLIP must go through ``AutoModel`` / ``AutoProcessor``."""
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch("transformers.AutoModel.from_pretrained") as m_model, patch(
        "transformers.AutoProcessor.from_pretrained"
    ) as m_proc, patch("transformers.CLIPModel.from_pretrained") as m_clip:
        m_model.return_value = "siglip-model"
        m_proc.return_value = "siglip-proc"
        model, proc = get_clip("google/siglip-base-patch16-224")

    assert (model, proc) == ("siglip-model", "siglip-proc")
    m_model.assert_called_once_with("google/siglip-base-patch16-224")
    m_proc.assert_called_once_with("google/siglip-base-patch16-224")
    # CLIP loader must not have been invoked for a siglip id.
    m_clip.assert_not_called()
    clear_clip_cache()


def test_clip_cache_unknown_prefix_falls_back_to_clip():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch("transformers.CLIPModel.from_pretrained") as m_model, patch(
        "transformers.CLIPProcessor.from_pretrained"
    ):
        m_model.return_value = "unknown-model"
        get_clip("custom/some-other-vlm")

    m_model.assert_called_once()
    clear_clip_cache()


def test_clip_cache_caches_per_model_name():
    """Different model_name keys load separately; same key hits the cache."""
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    with patch("transformers.CLIPModel.from_pretrained") as m_model, patch(
        "transformers.CLIPProcessor.from_pretrained"
    ):
        m_model.side_effect = ["m1", "m2"]
        get_clip("openai/clip-vit-base-patch32")
        get_clip("openai/clip-vit-base-patch32")  # second call hits cache
        get_clip("openai/clip-vit-large-patch14")  # different key → fresh load

    assert m_model.call_count == 2
    clear_clip_cache()


# ---------------------------------------------------------------------------
# Embedder + factory plumbing
# ---------------------------------------------------------------------------

def test_clip_text_embedder_propagates_model_name():
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

        e = CLIPTextEmbedder(model_name="google/siglip-base-patch16-224")
        assert e.model_name == "google/siglip-base-patch16-224"
        m_get_clip.assert_called_once_with(
            "google/siglip-base-patch16-224", dtype="fp32"
        )
        caps = e.get_capabilities()
        assert caps["model"] == "google/siglip-base-patch16-224"
    clear_clip_cache()


def test_clip_image_embedder_propagates_model_name():
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

        e = CLIPImageEmbedder(model_name="openai/clip-vit-large-patch14")
        assert e.model_name == "openai/clip-vit-large-patch14"
        m_get_clip.assert_called_once_with(
            "openai/clip-vit-large-patch14", dtype="fp32"
        )
        caps = e.get_capabilities()
        assert caps["model"] == "openai/clip-vit-large-patch14"
    clear_clip_cache()


def test_factory_threads_clip_model_name_into_embedders():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.embedder_factory import (
            EmbedderConfig,
            FlexibleMultimodalEmbedder,
        )

        cfg = EmbedderConfig(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="whisper",
            clip_model_name="google/siglip-base-patch16-224",
            openai_api_key="sk-test",
        )
        emb = FlexibleMultimodalEmbedder(cfg)

        assert emb.text_embedder.model_name == "google/siglip-base-patch16-224"
        assert emb.image_embedder.model_name == "google/siglip-base-patch16-224"
        # Same backbone → text + image share one cached load (a single
        # ``get_clip`` call is enough; subsequent calls hit the cache).
        assert m_get_clip.call_args_list[0].args == (
            "google/siglip-base-patch16-224",
        )
    clear_clip_cache()


def test_factory_default_keeps_clip_base():
    """Default config must still use clip-vit-base-patch32 for back-compat."""
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
        )

        assert emb.text_embedder.model_name == "openai/clip-vit-base-patch32"
        assert emb.image_embedder.model_name == "openai/clip-vit-base-patch32"
    clear_clip_cache()


def test_create_large_clip_embedder_alias():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.embedder_factory import (
            create_large_clip_embedder,
        )

        emb = create_large_clip_embedder(openai_api_key="sk-test")
        assert emb.text_embedder.model_name == "openai/clip-vit-large-patch14"
    clear_clip_cache()


def test_create_siglip_embedder_alias():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import clear_clip_cache

    clear_clip_cache()
    with patch(
        "outhad_contextkit.memory.temporal._clip_cache.get_clip"
    ) as m_get_clip:
        m_get_clip.return_value = ("model", "proc")
        from outhad_contextkit.memory.temporal.embedder_factory import (
            create_siglip_embedder,
        )

        emb = create_siglip_embedder(openai_api_key="sk-test")
        assert emb.text_embedder.model_name == "google/siglip-base-patch16-224"
        assert emb.image_embedder.model_name == "google/siglip-base-patch16-224"
    clear_clip_cache()


def test_config_to_dict_contains_clip_model_name():
    from outhad_contextkit.memory.temporal.embedder_factory import EmbedderConfig

    cfg = EmbedderConfig(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="whisper",
        clip_model_name="openai/clip-vit-large-patch14",
        openai_api_key="sk-test",
    )
    d = cfg.to_dict()
    assert d["clip_model_name"] == "openai/clip-vit-large-patch14"
    # Secret never leaks into the dict.
    assert d["openai_api_key"] == "***"
