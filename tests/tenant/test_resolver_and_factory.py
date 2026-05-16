""" TenantResolver + VectorStoreFactory collection routing.

Covers:
* Resolver disabled → always default tenant, base collection.
* Resolver mode='filter' → base collection, history/cgl filters populated.
* Resolver mode='collection' → derived collection name; deterministic.
* Slug collision-resistance via SHA-256 suffix.
* Default tenant id on enabled resolver still routes to base.
* VectorStoreFactory.create accepts collection_override; original
  config dict is not mutated.
* Memory._init_tenant builds resolver + registry when feature is on.
* Memory._vector_store_for caches per-tenant clients.
"""
from __future__ import annotations

import pytest

from outhad_contextkit.memory.tenant.config import TenantConfig, IsolationConfig
from outhad_contextkit.memory.tenant.resolver import (
    ResolvedTenant,
    TenantResolver,
)
from outhad_contextkit.memory.tenant.types import TenantContext
from outhad_contextkit.utils.factory import VectorStoreFactory


# ---------------------------------------------------------------------------
# TenantResolver
# ---------------------------------------------------------------------------

def _resolver(*, enabled=True, mode="filter") -> TenantResolver:
    cfg = TenantConfig(enabled=enabled, isolation=IsolationConfig(mode=mode))
    return TenantResolver(cfg=cfg, base_collection="ck")


def test_disabled_resolver_returns_default_tenant():
    r = TenantResolver(
        cfg=TenantConfig(enabled=False), base_collection="ck"
    )
    out = r.resolve(TenantContext(tenant_id="acme"))
    assert out.is_default is True
    assert out.collection_name == "ck"
    assert out.history_filter == {}
    assert out.cgl_filter == {}


def test_filter_mode_uses_base_collection_and_populates_filters():
    r = _resolver(mode="filter")
    out = r.resolve(TenantContext(tenant_id="acme", sub_tenant_id="finance"))
    assert out.is_default is False
    assert out.collection_name == "ck"
    assert out.history_filter == {"tenant_id": "acme", "sub_tenant_id": "finance"}
    assert out.cgl_filter == {"tenant_id": "acme", "sub_tenant_id": "finance"}


def test_collection_mode_derives_per_tenant_collection():
    r = _resolver(mode="collection")
    out = r.resolve(TenantContext(tenant_id="acme"))
    assert out.collection_name == "ck__acme"


def test_collection_mode_slug_with_special_chars_uses_hash_suffix():
    r = _resolver(mode="collection")
    a = r.resolve(TenantContext(tenant_id="Acme Corp")).collection_name
    b = r.resolve(TenantContext(tenant_id="Acme Corp")).collection_name
    assert a == b  # deterministic
    # Two distinct ids that slug to the same prefix must end up at
    # different collections via the hash suffix.
    c = r.resolve(TenantContext(tenant_id="acme corp")).collection_name
    d = r.resolve(TenantContext(tenant_id="acme-corp")).collection_name
    assert c != d  # SHA-256 differs


def test_default_tenant_routes_to_base_even_in_collection_mode():
    r = _resolver(mode="collection")
    out = r.resolve(TenantContext(tenant_id="__default__"))
    assert out.is_default is True
    assert out.collection_name == "ck"


def test_resolve_none_context_returns_default():
    r = _resolver(mode="filter")
    out = r.resolve(None)
    assert out.is_default is True
    assert out.collection_name == "ck"


def test_collection_for_alias_works():
    r = _resolver(mode="collection")
    assert r.collection_for("acme") == "ck__acme"
    assert r.collection_for("__default__") == "ck"


# ---------------------------------------------------------------------------
# VectorStoreFactory collection_override
# ---------------------------------------------------------------------------

class _FakeVectorStore:
    def __init__(self, **kwargs):
        self.config = kwargs


def test_factory_collection_override_does_not_mutate_input(monkeypatch):
    monkeypatch.setitem(
        VectorStoreFactory.provider_to_class,
        "fake",
        "tests.tenant.test_resolver_and_factory._FakeVectorStore",
    )
    cfg_dict = {"collection_name": "default", "embedding_model_dims": 8}
    instance = VectorStoreFactory.create(
        "fake", cfg_dict, collection_override="acme_t"
    )
    # Override applied on instance.
    assert instance.config["collection_name"] == "acme_t"
    # Caller's dict untouched.
    assert cfg_dict["collection_name"] == "default"


def test_factory_no_override_preserves_collection_name(monkeypatch):
    monkeypatch.setitem(
        VectorStoreFactory.provider_to_class,
        "fake",
        "tests.tenant.test_resolver_and_factory._FakeVectorStore",
    )
    cfg_dict = {"collection_name": "default", "embedding_model_dims": 8}
    instance = VectorStoreFactory.create("fake", cfg_dict)
    assert instance.config["collection_name"] == "default"


def test_factory_baidu_uses_table_name_field(monkeypatch):
    monkeypatch.setitem(
        VectorStoreFactory.provider_to_class,
        "baidu",
        "tests.tenant.test_resolver_and_factory._FakeVectorStore",
    )
    cfg_dict = {"table_name": "default", "endpoint": "x"}
    instance = VectorStoreFactory.create(
        "baidu", cfg_dict, collection_override="acme_t"
    )
    assert instance.config["table_name"] == "acme_t"
    # collection_name field is NOT created since baidu uses table_name.
    assert "collection_name" not in instance.config


# ---------------------------------------------------------------------------
# Memory._init_tenant + _vector_store_for
# ---------------------------------------------------------------------------

def test_memory_init_tenant_disabled_leaves_attributes_none():
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.collection_name = "ck"
    Memory._init_tenant(obj)
    assert obj._tenant_registry is None
    assert obj._tenant_resolver is None
    assert obj._tenant_vector_stores == {}


def test_memory_init_tenant_enabled_builds_resolver(tmp_path):
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory
    from outhad_contextkit.memory.tenant.resolver import TenantResolver

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.config.tenant.enabled = True
    obj.config.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    obj.collection_name = "ck"
    Memory._init_tenant(obj)
    assert obj._tenant_registry is not None
    assert isinstance(obj._tenant_resolver, TenantResolver)


def test_resolve_tenant_returns_resolved(tmp_path):
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.config.tenant.enabled = True
    obj.config.tenant.isolation.mode = "collection"
    obj.config.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    obj.collection_name = "ck"
    Memory._init_tenant(obj)
    resolved = Memory._resolve_tenant(obj, tenant_id="acme")
    # Compare by class name rather than ``isinstance`` because the
    # T1 disabled-import test pops the resolver module from sys.modules,
    # which can give us a fresh class identity on later test runs.
    assert resolved is not None
    assert type(resolved).__name__ == "ResolvedTenant"
    assert resolved.collection_name == "ck__acme"


def test_resolve_tenant_returns_none_when_disabled():
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.collection_name = "ck"
    Memory._init_tenant(obj)
    assert Memory._resolve_tenant(obj, tenant_id="acme") is None


def test_vector_store_for_default_returns_existing_client(tmp_path):
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.config.tenant.enabled = True
    obj.config.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    obj.collection_name = "ck"
    obj.vector_store = object()
    Memory._init_tenant(obj)
    # Default tenant → existing client; no per-tenant cache populated.
    assert Memory._vector_store_for(obj, None) is obj.vector_store
    resolved = Memory._resolve_tenant(obj)  # default
    assert Memory._vector_store_for(obj, resolved) is obj.vector_store
    assert obj._tenant_vector_stores == {}


def test_vector_store_for_collection_mode_caches_per_tenant_client(
    monkeypatch, tmp_path
):
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    monkeypatch.setitem(
        VectorStoreFactory.provider_to_class,
        "fake_provider",
        "tests.tenant.test_resolver_and_factory._FakeVectorStore",
    )

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.config.tenant.enabled = True
    obj.config.tenant.isolation.mode = "collection"
    obj.config.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    obj.config.vector_store.provider = "fake_provider"
    obj.collection_name = "ck"
    obj.vector_store = object()
    Memory._init_tenant(obj)

    resolved = Memory._resolve_tenant(obj, tenant_id="acme")
    a = Memory._vector_store_for(obj, resolved)
    b = Memory._vector_store_for(obj, resolved)
    assert a is b  # cached on second call
    assert isinstance(a, _FakeVectorStore)
    assert a.config["collection_name"] == "ck__acme"
