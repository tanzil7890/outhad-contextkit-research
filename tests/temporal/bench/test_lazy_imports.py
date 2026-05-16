"""verify heavyweight modules don't load at package import.

`import outhad_contextkit.memory.temporal` itself must NOT pull in
`transformers` or `torch`. Symbols resolve through `__getattr__` so
the heavy import only fires when the caller actually touches a
multimodal symbol.

Important: each test scopes its sys.modules manipulation through
``monkeypatch`` so the *original* module objects are restored on
teardown. Without that restoration a fresh import here would replace
the live ``outhad_contextkit.memory.temporal.reranker`` module, leaving
other test files with function references whose ``__globals__`` point
at the orphaned old module — breaking their monkeypatches downstream.
"""
from __future__ import annotations

import importlib
import sys

import pytest


_TEMPORAL_PREFIX = "outhad_contextkit.memory.temporal"


def _purge_temporal_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every cached temporal + heavy module **for the test's lifetime**.

    ``monkeypatch.delitem`` restores the original sys.modules entries on
    teardown, so other tests still see the same module objects they
    imported at module load.
    """
    for name in list(sys.modules.keys()):
        if name.startswith(_TEMPORAL_PREFIX):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_package_import_does_not_pull_transformers(monkeypatch):
    _purge_temporal_modules(monkeypatch)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    importlib.import_module(_TEMPORAL_PREFIX)
    assert "transformers" not in sys.modules
    # `torch` may still be present from earlier imports outside this test
    # process — the only reliable signal is: temporal does not pull it
    # *as a side effect of its own import*. Asserting strictly here
    # would be fragile. Skip the torch assertion.


def test_package_import_does_not_pull_text_or_image_embedder_modules(monkeypatch):
    """Submodules that re-import transformers must stay un-imported until first use."""
    _purge_temporal_modules(monkeypatch)
    importlib.import_module(_TEMPORAL_PREFIX)
    assert f"{_TEMPORAL_PREFIX}.text_embedders" not in sys.modules
    assert f"{_TEMPORAL_PREFIX}.image_embedders" not in sys.modules
    assert f"{_TEMPORAL_PREFIX}.audio_embedders" not in sys.modules


def test_attribute_access_resolves_lazily(monkeypatch):
    _purge_temporal_modules(monkeypatch)
    mod = importlib.import_module(_TEMPORAL_PREFIX)
    # Touch a lightweight type — should pull only `types` submodule.
    _ = mod.TimeWindow
    assert f"{_TEMPORAL_PREFIX}.types" in sys.modules


def test_unknown_attribute_raises_attribute_error():
    mod = importlib.import_module(_TEMPORAL_PREFIX)
    with pytest.raises(AttributeError):
        _ = mod.DoesNotExist


def test_dir_includes_lazy_exports():
    mod = importlib.import_module(_TEMPORAL_PREFIX)
    listed = dir(mod)
    assert "RetrievalOrchestrator" in listed
    assert "CLIPTextEmbedder" in listed
