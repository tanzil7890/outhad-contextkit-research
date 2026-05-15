import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import pytz
from pydantic import ValidationError

from outhad_contextkit.configs.base import MemoryConfig, MemoryItem
from outhad_contextkit.configs.enums import MemoryType
from outhad_contextkit.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from outhad_contextkit.memory.base import MemoryBase
from outhad_contextkit.memory.privacy import (
    EncryptionManager,
    MemorySanitizer,
    TelemetrySanitizer,
    redact_text,
)
from outhad_contextkit.memory.privacy.adversarial import AdversarialTestRunner
from outhad_contextkit.memory.privacy.enums import PrivacyLevel
from outhad_contextkit.memory.setup import outhad_contextkit_dir, setup_config
from outhad_contextkit.memory.history_store import build_history_store
from outhad_contextkit.memory.storage import SQLiteManager  # noqa: F401  — kept for back-compat imports
from outhad_contextkit.memory.telemetry import capture_event
from outhad_contextkit.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from outhad_contextkit.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided.")

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


def _sanitize_filters_for_vector_store(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize filters for vector store search.
    
    Qdrant's MatchValue only accepts bool, int, or str values.
    This function removes or converts unsupported types like floats.
    
    Args:
        filters: Original filters dictionary
        
    Returns:
        Sanitized filters dictionary safe for vector store search
    """
    if not filters:
        return filters
    
    sanitized = {}
    skip_keys = {"confidence", "timestamp", "modality", "image_hash", "audio_hash"}
    
    for key, value in filters.items():
        # Skip TCMGM metadata fields that aren't useful for filtering
        if key in skip_keys:
            continue
        
        # Keep only types supported by Qdrant MatchValue
        if isinstance(value, (bool, int, str)):
            sanitized[key] = value
        elif isinstance(value, float):
            # Convert float to string for Qdrant compatibility
            sanitized[key] = str(value)
        else:
            # Skip other unsupported types
            logger.debug(f"Skipping filter key '{key}' with unsupported type: {type(value)}")
    
    return sanitized


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = build_history_store(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            if self.config.graph_store.provider == "memgraph":
                from outhad_contextkit.memory.memgraph_memory import MemoryGraph
            elif self.config.graph_store.provider == "neptune":
                from outhad_contextkit.graphs.neptune.main import MemoryGraph
            else:
                from outhad_contextkit.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True
        else:
            self.graph = None
        self.config.vector_store.config.collection_name = "outhad_contextkitmigrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            self.config.vector_store.config.path = os.path.join(outhad_contextkit_dir, provider_path)
            os.makedirs(self.config.vector_store.config.path, exist_ok=True)
        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        
        # Initialize PPMF (Privacy-Preserving Memory Firewall) if enabled
        self.ppmf_enabled = self.config.ppmf.enabled
        if self.ppmf_enabled:
            self._init_ppmf()
        else:
            self._memory_sanitizer = None
            self._encryption_manager = None
            self._telemetry_sanitizer = None
            self._adversarial_runner = None
        
        # Initialize TCMGM (Temporal-Causal Multimodal Graph Memory) if graph is enabled
        if self.enable_graph:
            self._init_tcmgm()
        else:
            self._tcmgm_enabled = False
            self._timeline_builder = None
            self._retrieval_orchestrator = None

        # Initialize chunker if enabled
        if self.config.chunking.enabled:
            from outhad_contextkit.memory.chunking import ChunkerFactory
            self._chunker = ChunkerFactory.create(self.config.chunking)
            logger.info(f"Chunking enabled: {self._chunker}")
        else:
            self._chunker = None

        # Initialize Context-Graph Layer (CGL) if enabled
        self._init_context_graph()
        # Initialize Multi-Stage Personalized Retrieval (MSPR) if enabled
        self._init_mspr()
        # Initialize tenant subsystem (T1+T2+T3) if enabled
        self._init_tenant()
        # Initialize lifecycle subsystem (D1-D8) if enabled
        self._init_lifecycle()

        capture_event("outhad_contextkit.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
    
    def _init_ppmf(self):
        """Initialize Privacy-Preserving Memory Firewall components."""
        logger.info("Initializing PPMF (Privacy-Preserving Memory Firewall)")
        
        # Initialize classifier
        self._memory_sanitizer = MemorySanitizer(
            config=self.config.ppmf,
            llm=self.llm if self.config.ppmf.use_llm_classifier else None
        )
        
        # Initialize encryption manager
        self._encryption_manager = EncryptionManager(config=self.config.ppmf)
        
        # Initialize telemetry sanitizer
        self._telemetry_sanitizer = TelemetrySanitizer(
            config=self.config.ppmf,
            classifier=self._memory_sanitizer
        )
        
        # Initialize adversarial test runner
        if self.config.ppmf.enable_adversarial_testing:
            self._adversarial_runner = AdversarialTestRunner(
                memory_instance=self,
                interval=self.config.ppmf.adversarial_test_interval
            )
        else:
            self._adversarial_runner = None
        
        logger.info("PPMF initialized successfully")
    
    def _init_tcmgm(self):
        """Initialize TCMGM (Temporal-Causal Multimodal Graph Memory) components."""
        from outhad_contextkit.memory.timeline import TimelineBuilder
        from outhad_contextkit.memory.temporal.orchestrator import RetrievalOrchestrator
        
        logger.info("Initializing TCMGM (Temporal-Causal Multimodal Graph Memory)")
        
        try:
            # Initialize timeline builder
            self._timeline_builder = TimelineBuilder(self.llm, self.graph)
            
            # Initialize retrieval orchestrator with embedding model
            self._retrieval_orchestrator = RetrievalOrchestrator(
                vector_store=self.vector_store,
                graph_store=self.graph,
                timeline_builder=self._timeline_builder,
                embedding_model=self.embedding_model
            )
            
            self._tcmgm_enabled = True
            logger.info("TCMGM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TCMGM: {e}")
            self._tcmgm_enabled = False
            self._timeline_builder = None
            self._retrieval_orchestrator = None

    def _init_context_graph(self):
        """Initialize the Context-Graph Layer if enabled.

        When ``config.context_graph.enabled`` is ``False`` (default) this method
        installs no-op placeholders so the hook sites below remain cheap.
        """
        self._context_graph = None
        self._context_graph_builder = None
        cgl_config = getattr(self.config, "context_graph", None)
        if cgl_config is None or not cgl_config.enabled:
            return

        try:
            from outhad_contextkit.memory.context_graph import build_context_graph
            from outhad_contextkit.memory.context_graph.builder import (
                IncrementalGraphBuilder,
            )

            driver = None
            if cgl_config.backend == "neo4j":
                driver = getattr(self.graph, "graph", None) if self.enable_graph else None

            self._context_graph = build_context_graph(cgl_config, neo4j_driver=driver)
            if self._context_graph is not None:
                # Phase C — give the builder access to the extraction LLM so
                # it can classify semantic edges. Opt-in via
                # ``cgl_config.edges.use_llm_inference`` / ``edges.llm.enabled``.
                llm_for_edges = None
                edges_cfg = cgl_config.edges
                if edges_cfg.use_llm_inference or edges_cfg.llm.enabled:
                    llm_for_edges = getattr(self, "llm", None)
                self._context_graph_builder = IncrementalGraphBuilder(
                    self._context_graph,
                    cgl_config.edges,
                    llm=llm_for_edges,
                )
                logger.info("Context-Graph Layer initialized (backend=%s)", cgl_config.backend)
        except Exception as exc:
            logger.error("Failed to initialize Context-Graph Layer: %s", exc)
            self._context_graph = None
            self._context_graph_builder = None

    def _init_mspr(self) -> None:
        """Initialise the MSPR subsystem (feedback store, etc.).

        All imports stay local so an operator who leaves ``mspr.enabled``
        False never pays the import cost for SQLite wrappers / providers.
        Idempotent: safe to call twice (the second call is a no-op).
        """
        self._feedback_store = None
        self._success_store = None  # F7
        self._intent_router = None  # F5
        self._role_policy = None  # F6
        self._mspr_pipeline = None  # F8
        mspr_cfg = getattr(self.config, "mspr", None)
        if mspr_cfg is None or not mspr_cfg.enabled:
            return

        if mspr_cfg.feedback.enabled:
            from outhad_contextkit.memory.personalized.feedback_store import (
                FeedbackStore,
            )

            fb_path = mspr_cfg.feedback.sqlite_path or os.path.join(
                outhad_contextkit_dir, "feedback_events.db"
            )
            try:
                self._feedback_store = FeedbackStore(fb_path)
                logger.info("MSPR FeedbackStore initialised at %s", fb_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to initialise FeedbackStore: %s", exc)
                self._feedback_store = None

        if mspr_cfg.intent.enabled:
            from outhad_contextkit.memory.personalized.intent import IntentRouter

            try:
                self._intent_router = IntentRouter(
                    cfg=mspr_cfg.intent,
                    llm=getattr(self, "llm", None),
                )
                logger.info(
                    "MSPR IntentRouter initialised (strategy=%s)",
                    mspr_cfg.intent.strategy,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to initialise IntentRouter: %s", exc)
                self._intent_router = None

        # Phase F7 — QuerySuccessStore. Lazily constructed; default path
        # mirrors feedback_events.db so operators can ship one volume.
        if mspr_cfg.success.enabled:
            from outhad_contextkit.memory.personalized.success_store import (
                QuerySuccessStore,
            )

            qs_path = mspr_cfg.success.sqlite_path or os.path.join(
                outhad_contextkit_dir, "query_success.db"
            )
            try:
                self._success_store = QuerySuccessStore(qs_path)
                logger.info("MSPR QuerySuccessStore initialised at %s", qs_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to initialise QuerySuccessStore: %s", exc)
                self._success_store = None

        # Phase F8 — PersonalizedRetrievalPipeline (orchestrator). The
        # pipeline is a thin wrapper over _graph_first_rerank used by
        # external callers and tests. It always exists once the master
        # switch is on so apps can call it directly even with all
        # sub-features off (it falls through to a passthrough).
        try:
            from outhad_contextkit.memory.personalized.pipeline import (
                PersonalizedRetrievalPipeline,
            )

            self._mspr_pipeline = PersonalizedRetrievalPipeline(self)
            logger.info("MSPR pipeline initialised")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise MSPR pipeline: %s", exc)
            self._mspr_pipeline = None

    # ------------------------------------------------------------------
    #  Tenant subsystem (Phases T1–T3) — initialised here so
    #  _vector_store_for / resolver are available to every public method.
    # ------------------------------------------------------------------
    def _init_tenant(self) -> None:
        """Initialise the tenant routing subsystem.

        All imports stay local so an operator who leaves
        ``tenant.enabled=False`` never pays the import cost for the
        SQLite registry / resolver. Idempotent.
        """
        self._tenant_registry = None
        self._tenant_resolver = None
        self._tenant_vector_stores: Dict[str, Any] = {}
        self.tenant = None  # public TenantAdmin facade (T6)
        tenant_cfg = getattr(self.config, "tenant", None)
        if tenant_cfg is None or not tenant_cfg.enabled:
            return

        from outhad_contextkit.memory.tenant.registry import TenantRegistry
        from outhad_contextkit.memory.tenant.resolver import TenantResolver

        registry_path = tenant_cfg.registry.sqlite_path or os.path.join(
            outhad_contextkit_dir, "tenant_registry.db"
        )
        try:
            self._tenant_registry = TenantRegistry(
                registry_path,
                cache_size=tenant_cfg.registry.cache_size,
            )
            logger.info("Tenant registry initialised at %s", registry_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise TenantRegistry: %s", exc)
            self._tenant_registry = None

        try:
            base_collection = self.collection_name
            self._tenant_resolver = TenantResolver(
                cfg=tenant_cfg,
                base_collection=base_collection,
                registry=self._tenant_registry,
            )
            logger.info(
                "Tenant resolver initialised (mode=%s)",
                tenant_cfg.isolation.mode,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise TenantResolver: %s", exc)
            self._tenant_resolver = None

        # Phase T6 — public admin facade. Always built once registry +
        # resolver are in place so callers see a consistent API.
        try:
            from outhad_contextkit.memory.tenant.admin import TenantAdmin

            self.tenant = TenantAdmin(self)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise TenantAdmin: %s", exc)
            self.tenant = None

    def _resolve_tenant(self, tenant_id=None, sub_tenant_id=None,
                        user_id=None, agent_id=None, run_id=None,
                        role=None):
        """Build a :class:`ResolvedTenant` for the supplied context.

        Returns ``None`` when the tenant subsystem is disabled —
        callers that respect ``None`` keep their pre-tenant behaviour.
        """
        if self._tenant_resolver is None:
            return None
        from outhad_contextkit.memory.tenant.types import TenantContext

        ctx = TenantContext(
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            role=role,
        )
        return self._tenant_resolver.resolve(ctx)

    def _vector_store_for(self, resolved=None):
        """Return the vector-store client routing for ``resolved``.

        - Tenant disabled / default tenant / mode='filter' →
          ``self.vector_store`` (existing client; byte-identical).
        - mode='collection' on a non-default tenant → cached per-tenant
          client. Built on first access; collection creation is
          delegated to the underlying vector-store driver.
        """
        if (
            resolved is None
            or self._tenant_resolver is None
            or resolved.is_default
            or resolved.collection_name == self.collection_name
        ):
            return self.vector_store
        cache_key = resolved.collection_name
        if cache_key in self._tenant_vector_stores:
            return self._tenant_vector_stores[cache_key]
        try:
            client = VectorStoreFactory.create(
                self.config.vector_store.provider,
                self.config.vector_store.config,
                collection_override=cache_key,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Per-tenant vector store init failed for %s: %s",
                cache_key,
                exc,
            )
            return self.vector_store
        self._tenant_vector_stores[cache_key] = client
        return client

    # ------------------------------------------------------------------
    # Lifecycle subsystem (Phases D1–D8) — decay + versioning + cold storage.
    # ------------------------------------------------------------------
    def _init_lifecycle(self) -> None:
        """Initialise decay-v2 + versioning + scheduler + cold storage.

        Master switch is ``MemoryConfig.decay_v2.enabled``. When off,
        every public lifecycle method either short-circuits or raises
        ``RuntimeError`` so silent no-ops can never mask typos.
        """
        self._lifecycle_scheduler = None
        self._lifecycle_versioning = None
        self._lifecycle_references = None
        self._lifecycle_cold_storage = None
        cfg = getattr(self.config, "decay_v2", None)
        if cfg is None or not cfg.enabled:
            return

        # Reference tracker — always built (zero cost when track_references off).
        try:
            from outhad_contextkit.memory.lifecycle.references import (
                ReferenceTracker,
            )

            self._lifecycle_references = ReferenceTracker(self)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("ReferenceTracker init failed: %s", exc)

        # Versioning store — built when mode='immutable' for cheap reuse.
        if cfg.versioning.mode == "immutable":
            try:
                from outhad_contextkit.memory.lifecycle.versioning import (
                    VersionedMemoryStore,
                )

                self._lifecycle_versioning = VersionedMemoryStore(self)
                logger.info(
                    "VersionedMemoryStore initialised (immutable mode)"
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("VersionedMemoryStore init failed: %s", exc)

        # Cold storage adapter.
        cold_cfg = cfg.cold_storage
        if cold_cfg.backend == "local":
            from outhad_contextkit.memory.lifecycle.cold_storage import (
                LocalDiskAdapter,
            )

            root = cold_cfg.local_root or os.path.join(
                outhad_contextkit_dir, "cold_storage"
            )
            try:
                self._lifecycle_cold_storage = LocalDiskAdapter(root)
                logger.info(
                    "LocalDiskAdapter initialised at %s", root
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("LocalDiskAdapter init failed: %s", exc)
        elif cold_cfg.backend == "s3":
            try:
                from outhad_contextkit.memory.lifecycle.cold_storage import (
                    S3Adapter,
                )

                self._lifecycle_cold_storage = S3Adapter(
                    bucket=cold_cfg.s3_bucket or "",
                    prefix=cold_cfg.s3_prefix,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("S3Adapter init failed: %s", exc)

        # Scheduler — built but not auto-started unless schedule.enabled.
        if cfg.schedule.enabled:
            try:
                from outhad_contextkit.memory.lifecycle.scheduler import (
                    DecayScheduler,
                )

                self._lifecycle_scheduler = DecayScheduler(
                    self, cfg=cfg.schedule
                )
                self._lifecycle_scheduler.start()
                logger.info("DecayScheduler started (sync)")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("DecayScheduler init failed: %s", exc)

    def decay_score(self, memory_id: str) -> Optional[float]:
        """Phase D2 — return the computed decay score for one memory.

        Returns ``None`` when decay-v2 is disabled or the memory is
        unknown.
        """
        cfg = getattr(self.config, "decay_v2", None)
        if cfg is None or not cfg.enabled or self._context_graph is None:
            return None
        node = self._context_graph.backend.get_node(memory_id)
        if node is None:
            return None
        from outhad_contextkit.memory.lifecycle.scoring import (
            cached_peak_access,
            compute_decay_score,
        )

        return compute_decay_score(
            node,
            cfg=cfg,
            peak_access=cached_peak_access(self._context_graph),
            half_life_days=self.config.context_graph.decay.half_life_days,
        )

    def archive_low(
        self,
        *,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Phase D2 — archive every node whose decay_score is below ``threshold``.

        ``threshold`` defaults to ``decay_v2.archive_threshold``.
        ``dry_run=True`` returns the candidate ids without mutation so
        operators can audit before committing.
        """
        cfg = getattr(self.config, "decay_v2", None)
        if cfg is None or not cfg.enabled or self._context_graph is None:
            return {"archived": 0, "candidates": []}
        thr = float(cfg.archive_threshold if threshold is None else threshold)
        from outhad_contextkit.memory.lifecycle.scoring import (
            cached_peak_access,
            compute_decay_score,
        )

        backend = self._context_graph.backend
        peak = cached_peak_access(self._context_graph)
        half_life = self.config.context_graph.decay.half_life_days
        candidates: List[str] = []
        try:
            for node in backend.iter_nodes(include_archived=False):
                score = compute_decay_score(
                    node, cfg=cfg, peak_access=peak, half_life_days=half_life
                )
                if score < thr:
                    candidates.append(node.id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("archive_low iter failed: %s", exc)
        if dry_run:
            return {"archived": 0, "candidates": candidates, "threshold": thr}
        archived = 0
        for mid in candidates:
            try:
                self._context_graph.archive_memory_node(mid)
                archived += 1
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("archive_low archive failed for %s: %s", mid, exc)
        try:
            from outhad_contextkit.memory.telemetry import capture_event

            capture_event(
                "outhad_contextkit.lifecycle.archive",
                self,
                {"archived": archived, "threshold": thr, "sync_type": "sync"},
            )
        except Exception:  # pragma: no cover - telemetry never fatal
            pass
        return {"archived": archived, "candidates": candidates, "threshold": thr}

    def start_decay_scheduler(self) -> None:
        """Phase D3 — start the periodic decay scheduler (idempotent)."""
        if self._lifecycle_scheduler is None:
            cfg = getattr(self.config, "decay_v2", None)
            if cfg is None or not cfg.enabled:
                raise RuntimeError(
                    "start_decay_scheduler requires decay_v2.enabled=True"
                )
            from outhad_contextkit.memory.lifecycle.scheduler import (
                DecayScheduler,
            )

            self._lifecycle_scheduler = DecayScheduler(self, cfg=cfg.schedule)
        self._lifecycle_scheduler.start()

    def stop_decay_scheduler(self, *, timeout: float = 5.0) -> None:
        """Phase D3 — stop the scheduler thread (idempotent)."""
        if self._lifecycle_scheduler is not None:
            self._lifecycle_scheduler.stop(timeout=timeout)

    def close(self) -> None:
        """Idempotent cleanup. Stops the decay scheduler if running."""
        try:
            self.stop_decay_scheduler()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("close() scheduler stop failed: %s", exc)

    def record_reference(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        strength: float = 1.0,
    ) -> Optional[int]:
        """Phase D6 — bump access_count + last_accessed_at for a memory
        the agent actually used downstream.

        Returns the new access count, or ``None`` when the lifecycle
        subsystem is disabled.
        """
        if (
            self._lifecycle_references is None
            or not getattr(self.config.decay_v2, "track_references", False)
        ):
            return None
        return self._lifecycle_references.record_reference(
            memory_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            strength=strength,
        )

    def versions(self, memory_id: str) -> List[Any]:
        """Phase D5 — return the version chain (oldest → newest)."""
        if self._lifecycle_versioning is None:
            return []
        return self._lifecycle_versioning.list_versions(memory_id)

    def latest_version(self, memory_id: str) -> Optional[Any]:
        """Phase D5 — return the newest non-superseded version, or None."""
        if self._lifecycle_versioning is None:
            return None
        return self._lifecycle_versioning.latest(memory_id)

    def diff_versions(
        self,
        old_id: str,
        new_id: str,
        *,
        mode: Optional[str] = None,
    ) -> str:
        """Phase D5 — diff two memory versions.

        ``mode`` defaults to ``decay_v2.versioning.diff_provider`` —
        ``"unified"`` (text), ``"json"`` (JSON-Patch), or ``"none"``.
        """
        if self.vector_store is None:
            return ""
        try:
            old_row = self.vector_store.get(vector_id=old_id)
            new_row = self.vector_store.get(vector_id=new_id)
        except Exception:  # pragma: no cover - defensive
            return ""
        old_text = ""
        new_text = ""
        if old_row is not None:
            old_text = str((getattr(old_row, "payload", {}) or {}).get("data") or "")
        if new_row is not None:
            new_text = str((getattr(new_row, "payload", {}) or {}).get("data") or "")
        cfg = getattr(self.config, "decay_v2", None)
        chosen = mode or (cfg.versioning.diff_provider if cfg else "unified")
        if chosen == "none":
            return f"--- old\n{old_text}\n+++ new\n{new_text}"
        if chosen == "json":
            import json as _json

            try:
                old_obj = _json.loads(old_text)
                new_obj = _json.loads(new_text)
                return _json.dumps(
                    {"removed": old_obj, "added": new_obj}, indent=2
                )
            except Exception:
                # Fall through to unified diff if not JSON.
                pass
        import difflib

        return "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=old_id,
                tofile=new_id,
                lineterm="",
            )
        )

    def rollback(
        self,
        memory_id: str,
        *,
        to_version: int,
        actor_id: Optional[str] = None,
    ) -> Optional[str]:
        """Phase D5 — append a new version whose payload equals
        ``version=to_version``'s payload.

        Returns the new memory_id or ``None`` when versioning is off /
        the requested version cannot be found.
        """
        if self._lifecycle_versioning is None:
            return None
        chain = self._lifecycle_versioning.list_versions(memory_id)
        if not chain:
            return None
        target = next((v for v in chain if v.version == int(to_version)), None)
        if target is None:
            return None
        try:
            target_row = self.vector_store.get(vector_id=target.memory_id)
        except Exception:  # pragma: no cover - defensive
            return None
        if target_row is None:
            return None
        target_payload = dict(getattr(target_row, "payload", {}) or {})
        latest = chain[-1]
        new_record = self._lifecycle_versioning.insert_version(
            latest.memory_id,
            str(target_payload.get("data") or ""),
            target_payload,
            actor_id=actor_id,
        )
        try:
            from outhad_contextkit.memory.telemetry import capture_event

            capture_event(
                "outhad_contextkit.lifecycle.rollback",
                self,
                {
                    "memory_id": memory_id,
                    "to_version": int(to_version),
                    "new_id": new_record.memory_id,
                },
            )
        except Exception:  # pragma: no cover - telemetry never fatal
            pass
        return new_record.memory_id

    def demote_to_cold(self, memory_id: str) -> bool:
        """Phase D7 — move ``memory_id`` to cold storage; remove hot copies."""
        if self._lifecycle_cold_storage is None:
            return False
        try:
            row = self.vector_store.get(vector_id=memory_id)
        except Exception:  # pragma: no cover - defensive
            row = None
        if row is None:
            return False
        payload = dict(getattr(row, "payload", {}) or {})
        try:
            self._lifecycle_cold_storage.put(memory_id, payload)
        except Exception as exc:
            logger.error("demote_to_cold put failed: %s", exc)
            return False
        try:
            self.vector_store.delete(vector_id=memory_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("demote_to_cold vector delete failed: %s", exc)
        if self._context_graph is not None:
            try:
                self._context_graph.archive_memory_node(memory_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("demote_to_cold archive failed: %s", exc)
        return True

    def promote_from_cold(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Phase D7 — pull payload back from cold storage and re-insert."""
        if self._lifecycle_cold_storage is None:
            return None
        payload = self._lifecycle_cold_storage.get(memory_id)
        if payload is None:
            return None
        # Re-insert into vector store; embed if possible.
        embed = None
        text = str(payload.get("data") or "")
        if text and getattr(self, "embedding_model", None) is not None:
            try:
                embed = self.embedding_model.embed(text, "add")
            except Exception:  # pragma: no cover - embed is best-effort
                embed = None
        try:
            self.vector_store.insert(
                vectors=[embed] if embed is not None else None,
                ids=[memory_id],
                payloads=[payload],
            )
        except Exception as exc:
            logger.error("promote_from_cold insert failed: %s", exc)
            return None
        # Drop from cold storage so the layer stays consistent.
        try:
            self._lifecycle_cold_storage.delete(memory_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("promote_from_cold delete failed: %s", exc)
        # Un-archive in CGL if present.
        if self._context_graph is not None:
            try:
                node = self._context_graph.backend.get_node(memory_id)
                if node is not None:
                    node.archived = False
                    self._context_graph.backend.upsert_node(node)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("promote_from_cold un-archive failed: %s", exc)
        return payload

    def _prune_version_chains(self) -> int:
        """Phase D4 helper — invoked by the scheduler each tick."""
        if self._lifecycle_versioning is None:
            return 0
        cfg = getattr(self.config, "decay_v2", None)
        if cfg is None:
            return 0
        keep = max(1, int(cfg.versioning.keep_versions))
        if self._context_graph is None:
            return 0
        backend = self._context_graph.backend
        seen_roots: set = set()
        removed = 0
        try:
            for node in backend.iter_nodes(include_archived=True):
                root = (
                    node.metadata.get("version_root")
                    if node.metadata
                    else None
                )
                root = root or node.id
                if root in seen_roots:
                    continue
                seen_roots.add(root)
                removed += int(
                    self._lifecycle_versioning.prune(node.id, keep=keep) or 0
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("_prune_version_chains failed: %s", exc)
        return removed

    def backfill_decay_v2(self) -> Dict[str, int]:
        """Phase D8 — recompute decay scores + run archive_low once.

        Idempotent. Useful right after enabling decay_v2 on an existing
        deployment to bring archived state in line with the new
        thresholds.
        """
        cfg = getattr(self.config, "decay_v2", None)
        if cfg is None or not cfg.enabled:
            raise RuntimeError(
                "backfill_decay_v2 requires decay_v2.enabled=True"
            )
        result = self.archive_low()
        return {
            "archived": int(result.get("archived", 0) or 0),
            "candidates": len(result.get("candidates") or []),
        }

    # ------------------------------------------------------------------
    # Tenant lifecycle (Phase T7) — sync helpers used by TenantAdmin.
    # ------------------------------------------------------------------
    def _tenant_soft_delete(self, tenant_id: str) -> Dict[str, int]:
        """Archive every CGL node carrying ``tenant_id``.

        Vector-store rows + history rows stay in place so the tenant
        can be reactivated. Returns counts of archived nodes only;
        history/vector counts are 0 by design.
        """
        archived = 0
        if self._context_graph is not None:
            backend = self._context_graph.backend
            try:
                for node in list(backend.iter_nodes(include_archived=False)):
                    if getattr(node, "tenant_id", None) == tenant_id:
                        try:
                            self._context_graph.archive_memory_node(node.id)
                            archived += 1
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug(
                                "soft delete archive failed for %s: %s",
                                node.id,
                                exc,
                            )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("soft delete iter failed: %s", exc)
        return {"history": 0, "vectors": 0, "nodes": archived, "edges": 0}

    def _tenant_hard_delete(self, tenant_id: str) -> Dict[str, int]:
        """Cascade through every storage layer for a hard delete.

        Returns counts dict with keys ``vectors``, ``history``,
        ``nodes``, ``edges``. Vector-store deletion is best-effort:
        when the per-tenant collection cannot be dropped (e.g. driver
        lacks ``delete_col``), the count stays 0 but other layers
        proceed so the tenant becomes unusable regardless.
        """
        counts = {"vectors": 0, "history": 0, "nodes": 0, "edges": 0}

        # 1) Vector store — drop the per-tenant collection when
        #    isolation mode='collection'. mode='filter' falls through
        #    because data lives in the shared collection (best-effort
        #    delete by metadata is left to T8 backfill helpers).
        tenant_cfg = getattr(self.config, "tenant", None)
        if (
            tenant_cfg is not None
            and tenant_cfg.isolation.mode == "collection"
            and tenant_id != tenant_cfg.default_tenant_id
        ):
            client = self._tenant_vector_stores.pop(
                self._tenant_resolver.collection_for(tenant_id), None
            )
            if client is not None:
                try:
                    client.delete_col()
                    counts["vectors"] = 1
                except Exception as exc:  # pragma: no cover - driver-dependent
                    logger.debug("vector delete_col failed: %s", exc)

        # 2) History — DELETE rows tagged with tenant_id.
        try:
            with self.db._lock:
                cur = self.db.connection.execute(
                    "DELETE FROM history WHERE tenant_id = ?",
                    (tenant_id,),
                )
                counts["history"] = int(cur.rowcount or 0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("history hard delete failed: %s", exc)

        # 3) CGL — drop every node tagged with tenant_id (cascades edges
        #    via backend.delete_node).
        if self._context_graph is not None:
            backend = self._context_graph.backend
            try:
                for node in list(backend.iter_nodes(include_archived=True)):
                    if getattr(node, "tenant_id", None) != tenant_id:
                        continue
                    try:
                        edges_before = len(
                            backend.neighbours(node.id, depth=1, min_weight=0.0)
                        )
                    except Exception:
                        edges_before = 0
                    try:
                        self._context_graph.delete_memory_node(node.id)
                        counts["nodes"] += 1
                        counts["edges"] += edges_before
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug(
                            "hard delete node %s failed: %s", node.id, exc
                        )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("CGL hard delete iter failed: %s", exc)

        return counts

    def record_feedback(
        self,
        memory_id: str,
        *,
        helpful: bool,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record user feedback on a retrieved memory.

        Returns ``{'memory_id', 'delta_applied', 'new_relevance', 'event_id',
        'helpful'}`` on success. Returns ``None`` when MSPR / CGL /
        feedback is disabled or the memory is unknown.

        Side effects:
        * Persists a ``FeedbackEvent`` to ``feedback_events.db``.
        * Bumps ``MemoryNode.relevance`` via
          :meth:`ContextGraph.bump_relevance` (floor pinned on helpful
          when ``FeedbackConfig.pin_floor_on_boost=True``).
        * Clamps the per-memory cumulative boost at
          ``FeedbackConfig.max_boost_per_memory`` so repeated helpful
          verdicts cannot snowball past the cap.
        * Increments ``helpful_count`` / ``unhelpful_count`` + writes
          ``last_feedback_at`` on the node.
        * Emits a ``memory_feedback_recorded`` change-bus event.
        """
        mspr_cfg = getattr(self.config, "mspr", None)
        if (
            mspr_cfg is None
            or not mspr_cfg.enabled
            or not mspr_cfg.feedback.enabled
            or self._context_graph is None
            or self._feedback_store is None
        ):
            return None

        node = self._context_graph.backend.get_node(memory_id)
        if node is None:
            return None

        fb_cfg = mspr_cfg.feedback
        # Base delta from the config + signed by verdict.
        raw_delta = (
            float(fb_cfg.boost_delta) if helpful else -float(fb_cfg.penalty_delta)
        )

        # Clamp the POSITIVE cumulative boost contribution against
        # max_boost_per_memory. Penalties always apply at full magnitude.
        clamped_delta = raw_delta
        if helpful and fb_cfg.max_boost_per_memory > 0:
            cumulative = self._feedback_store.aggregate(
                memory_id, user_id=user_id
            )["net"]
            remaining = float(fb_cfg.max_boost_per_memory) - max(0.0, cumulative)
            if remaining <= 0.0:
                clamped_delta = 0.0
            else:
                clamped_delta = min(raw_delta, remaining)

        from datetime import datetime, timezone

        from outhad_contextkit.memory.personalized.feedback_store import (
            hash_query,
        )
        from outhad_contextkit.memory.personalized.types import FeedbackEvent

        now = datetime.now(timezone.utc)
        q_hash = hash_query(query)

        # Mutate node counters + last_feedback_at via the backend so both
        # in-proc and Neo4j stores stay in sync. Relevance + floor go
        # through bump_relevance so we get the atomic Cypher path on Neo4j.
        new_relevance: Optional[float] = None
        try:
            if abs(clamped_delta) > 0:
                set_floor = bool(helpful and fb_cfg.pin_floor_on_boost)
                new_relevance = self._context_graph.bump_relevance(
                    memory_id,
                    clamped_delta,
                    set_floor=set_floor,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_feedback bump_relevance failed: %s", exc)

        try:
            # Read-modify-write counters on a live snapshot of the node.
            node = self._context_graph.backend.get_node(memory_id) or node
            if helpful:
                node.helpful_count = int(node.helpful_count or 0) + 1
            else:
                node.unhelpful_count = int(node.unhelpful_count or 0) + 1
            node.last_feedback_at = now
            self._context_graph.backend.upsert_node(node)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_feedback counter update failed: %s", exc)

        event = FeedbackEvent(
            memory_id=memory_id,
            helpful=bool(helpful),
            delta_applied=float(clamped_delta),
            timestamp=now,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            query=query,
            query_hash=q_hash,
            metadata=dict(metadata or {}),
        )
        event_id = self._feedback_store.append(event)

        # Phase F7 — also record (query_hash, memory_id, helpful) into the
        # success store so future searches with the same query hash can
        # bias their ranking toward this memory.
        success_row_id: Optional[int] = None
        if (
            mspr_cfg.success.enabled
            and self._success_store is not None
            and q_hash is not None
        ):
            try:
                success_row_id = self._success_store.record(
                    query_hash=q_hash,
                    memory_id=memory_id,
                    helpful=bool(helpful),
                    user_id=user_id,
                    timestamp=now,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("record_feedback success_store.record failed: %s", exc)

        # Emit a change-bus event so Phase-E subscribers (webhook/kafka/sse)
        # pick up feedback without having to tail the SQLite file themselves.
        try:
            self._context_graph._emit(
                "memory_feedback_recorded",
                memory_id,
                user_id,
                {
                    "helpful": bool(helpful),
                    "delta_applied": float(clamped_delta),
                    "new_relevance": (
                        None if new_relevance is None else float(new_relevance)
                    ),
                    "event_id": int(event_id),
                    "query_hash": q_hash,
                    "success_row_id": success_row_id,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_feedback emit failed: %s", exc)

        return {
            "memory_id": memory_id,
            "delta_applied": float(clamped_delta),
            "new_relevance": new_relevance,
            "event_id": int(event_id),
            "helpful": bool(helpful),
        }

    def reset_feedback(self, *, user_id: Optional[str] = None) -> int:
        """Delete feedback rows (optionally scoped to one user).

        Returns the number of rows deleted. No-op if MSPR is disabled.
        """
        removed = 0
        if self._feedback_store is not None:
            removed = int(self._feedback_store.reset(user_id=user_id))
        # Phase F7 — keep the two stores in sync when the operator wipes
        # feedback. Without this, hit_rate() would still serve stale rows.
        if self._success_store is not None:
            try:
                self._success_store.reset(user_id=user_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("reset_feedback success_store.reset failed: %s", exc)
        return removed

    def set_role_policy(self, policy: Optional[Any]) -> None:
        """Install a :class:`RolePolicy` for role/tenant-aware retrieval.

        Pass ``None`` to disable role-based filtering. The policy is
        consulted by ``_graph_first_rerank`` when ``mspr.role.enabled``
        is True. Pydantic config stays JSON-serialisable because the
        callable lives off-config on the Memory instance.
        """
        if policy is not None:
            from outhad_contextkit.memory.personalized.role import RolePolicy

            if not isinstance(policy, RolePolicy):
                raise TypeError(
                    "policy must be a RolePolicy instance or None; "
                    f"got {type(policy).__name__}"
                )
        self._role_policy = policy

    def get_role_policy(self) -> Optional[Any]:
        """Return the currently installed RolePolicy or ``None``."""
        return getattr(self, "_role_policy", None)

    def backfill_mspr(self) -> Dict[str, int]:
        """Phase F8 — idempotent backfill of MSPR-derived tables.

        Today this rebuilds ``query_success`` from ``feedback_events``
        when an operator upgrades from a pre-F7 build. Future phases
        can hang additional backfills off the same entrypoint.

        Returns a counts dict ``{'success_rows': N, 'feedback_rows': M}``
        for logging / progress display. Raises ``RuntimeError`` when
        MSPR is disabled — by design, so silent no-ops on a misconfigured
        Memory don't mask a typo.
        """
        mspr_cfg = getattr(self.config, "mspr", None)
        if mspr_cfg is None or not mspr_cfg.enabled:
            raise RuntimeError(
                "Memory.backfill_mspr() called but mspr.enabled=False"
            )

        success_rows = 0
        feedback_rows = 0
        if (
            self._feedback_store is not None
            and self._success_store is not None
        ):
            try:
                events = self._feedback_store.list_events(limit=1_000_000)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("backfill_mspr feedback list failed: %s", exc)
                events = []
            feedback_rows = len(events)
            for ev in events:
                if not ev.query_hash:
                    continue
                try:
                    rid = self._success_store.record(
                        query_hash=ev.query_hash,
                        memory_id=ev.memory_id,
                        helpful=bool(ev.helpful),
                        user_id=ev.user_id,
                        timestamp=ev.timestamp,
                    )
                    if rid > 0:
                        success_rows += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("backfill_mspr record failed: %s", exc)
        logger.info(
            "backfill_mspr: %d success rows reconstructed from %d feedback events",
            success_rows,
            feedback_rows,
        )
        return {
            "success_rows": int(success_rows),
            "feedback_rows": int(feedback_rows),
        }

    # ------------------------------------------------------------------
    # Tenant migration / lifecycle (T7 + T8) — public facade.
    # ------------------------------------------------------------------
    def backfill_tenant(
        self,
        *,
        default_tenant_id: Optional[str] = None,
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """Phase T8 — stamp legacy memory + CGL nodes with the default tenant.

        Idempotent. Returns ``{"history": N, "nodes": M, "edges": K}``.
        Raises ``RuntimeError`` when ``tenant.enabled=False``.
        """
        from outhad_contextkit.memory.tenant.migration import backfill_tenant

        tid = default_tenant_id or self.config.tenant.default_tenant_id
        return backfill_tenant(
            self, default_tenant_id=tid, batch_size=batch_size
        )

    def export_tenant(self, tenant_id: str, dest_dir: str) -> Dict[str, int]:
        """Phase T7 — write every storage row for ``tenant_id`` to disk."""
        from outhad_contextkit.memory.tenant.migration import export_tenant

        return export_tenant(self, tenant_id, dest_dir)

    def import_tenant(self, src_dir: str) -> Dict[str, int]:
        """Phase T7 — reverse of :meth:`export_tenant`."""
        from outhad_contextkit.memory.tenant.migration import import_tenant

        return import_tenant(self, src_dir)

    def migrate_tenant(self, src_id: str, dst_id: str) -> Dict[str, int]:
        """Phase T7 — rewrite every storage row from src_id to dst_id."""
        from outhad_contextkit.memory.tenant.migration import migrate_tenant

        return migrate_tenant(self, src_id, dst_id)

    def _cgl_on_created(
        self,
        memory_id: str,
        data: str,
        embedding,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self._context_graph_builder is None:
            return
        try:
            self._context_graph_builder.on_memory_created(
                memory_id,
                data,
                user_id=(metadata or {}).get("user_id"),
                agent_id=(metadata or {}).get("agent_id"),
                run_id=(metadata or {}).get("run_id"),
                metadata=metadata,
                embedding=list(embedding) if embedding is not None else None,
            )
        except Exception as exc:  # pragma: no cover - hook must never raise
            logger.debug("CGL on_created hook failed for %s: %s", memory_id, exc)

    def _cgl_on_updated(
        self,
        memory_id: str,
        data: str,
        embedding,
        metadata: Optional[Dict[str, Any]],
        prev_value: Optional[str] = None,
    ) -> None:
        if self._context_graph_builder is None:
            return
        try:
            prev_version_id = memory_id if prev_value and prev_value != data else None
            self._context_graph_builder.on_memory_updated(
                memory_id,
                data,
                prev_version_id=prev_version_id,
                user_id=(metadata or {}).get("user_id"),
                agent_id=(metadata or {}).get("agent_id"),
                run_id=(metadata or {}).get("run_id"),
                metadata=metadata,
                embedding=list(embedding) if embedding is not None else None,
            )
        except Exception as exc:  # pragma: no cover - hook must never raise
            logger.debug("CGL on_updated hook failed for %s: %s", memory_id, exc)

    def _cgl_on_deleted(self, memory_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._context_graph_builder is None:
            return
        try:
            self._context_graph_builder.on_memory_deleted(
                memory_id,
                user_id=(metadata or {}).get("user_id"),
                hard=False,
            )
        except Exception as exc:  # pragma: no cover - hook must never raise
            logger.debug("CGL on_deleted hook failed for %s: %s", memory_id, exc)

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                msg_content = message_dict["content"]
                actor_name = message_dict.get("name")

                # NEW: Adaptive chunking logic
                # Only chunk if:
                # 1. Chunking is enabled
                # 2. Document is large enough (>= min_document_size)
                # 3. Document is larger than chunk_size
                should_chunk = (
                    self._chunker and
                    len(msg_content) >= self.config.chunking.min_document_size and
                    len(msg_content) > self.config.chunking.chunk_size
                )

                if should_chunk:
                    # Chunk the content
                    chunks = self._chunker.split_text(
                        msg_content,
                        metadata={
                            "role": message_dict["role"],
                            "actor_id": actor_name,
                            **metadata
                        }
                    )

                    logger.info(
                        f"Chunked message into {len(chunks)} chunks "
                        f"(original: {len(msg_content)} chars)"
                    )

                    # Store each chunk separately
                    for chunk in chunks:
                        chunk_metadata = {
                            **chunk.metadata,
                            **metadata,
                            "role": message_dict["role"],
                        }
                        if actor_name:
                            chunk_metadata["actor_id"] = actor_name

                        chunk_embeddings = self.embedding_model.embed(chunk.content, "add")
                        mem_id = self._create_memory(
                            chunk.content,
                            chunk_embeddings,
                            chunk_metadata
                        )

                        returned_memories.append({
                            "id": mem_id,
                            "memory": chunk.content,
                            "event": "ADD",
                            "actor_id": actor_name,
                            "role": message_dict["role"],
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "document_id": chunk.document_id
                        })
                else:
                    # Existing non-chunked path
                    per_msg_meta = deepcopy(metadata)
                    per_msg_meta["role"] = message_dict["role"]

                    if actor_name:
                        per_msg_meta["actor_id"] = actor_name

                    msg_embeddings = self.embedding_model.embed(msg_content, "add")
                    mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                    returned_memories.append({
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name,
                        "role": message_dict["role"],
                    })

            return returned_memories

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            # Sanitize filters for vector store (remove TCMGM metadata like confidence floats)
            search_filters = _sanitize_filters_for_vector_store(filters)
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                response = remove_code_blocks(response)
                new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")
                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "outhad_contextkit.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("outhad_contextkit.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "outhad_contextkit.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'` (which returns a dict with a 'results' key). "
                "The current format (direct list for v1.0) will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories_result
        else:
            return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)
        actual_memories = (
            memories_result[0]
            if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0
            else memories_result
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        use_tcmgm: bool = False,
        time_window: Optional[Dict] = None,
        include_causal: bool = True,
        merge_chunks: Optional[bool] = None,
        chunk_context_window: Optional[int] = None,
        use_context_graph: Optional[bool] = None,
        include_archived: bool = False,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None..
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            use_tcmgm (bool, optional): Use TCMGM fused retrieval (combines vector, graph, and timeline). Defaults to False.
            time_window (dict, optional): Time window for TCMGM temporal filtering. Dict with 'start' and 'end' datetime strings (ISO format). Defaults to None.
            include_causal (bool, optional): Include causal chain exploration when using TCMGM. Defaults to True.
            merge_chunks (bool, optional): Merge chunks from same document (default: config.chunking.merge_chunks_on_retrieval). Defaults to None.
            chunk_context_window (int, optional): Number of neighboring chunks to include (default: config.chunking.chunk_context_window). Defaults to None.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  When use_tcmgm=True, also includes "timeline_results", "causal_chains", and "fused_ranking".
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")
        
        # If TCMGM enabled and requested, use fused retrieval
        if use_tcmgm and self._tcmgm_enabled:
            from outhad_contextkit.memory.temporal.types import TimeWindow
            from datetime import datetime
            
            logger.info(f"Using TCMGM fused retrieval for query: '{query}'")
            
            # Parse time window
            tw = None
            if time_window:
                tw = TimeWindow(
                    start=datetime.fromisoformat(time_window.get('start')) if time_window.get('start') else None,
                    end=datetime.fromisoformat(time_window.get('end')) if time_window.get('end') else None
                )
            
            # Use primary user_id for TCMGM
            tcmgm_user_id = user_id or agent_id or run_id or ""
            
            # Use fused retrieval
            results = self._retrieval_orchestrator.fused_search(
                query=query,
                user_id=tcmgm_user_id,
                time_window=tw,
                include_causal=include_causal,
                include_multimodal=True,
                top_k=limit
            )
            
            logger.info("TCMGM fused search completed")
            return results

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "outhad_contextkit.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
            },
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # NEW: Set defaults from config for chunk merging
        if merge_chunks is None:
            merge_chunks = self.config.chunking.merge_chunks_on_retrieval
        if chunk_context_window is None:
            chunk_context_window = self.config.chunking.chunk_context_window

        # Build results dict
        if self.enable_graph:
            results = {"results": original_memories, "relations": graph_entities}
        elif self.api_version == "v1.0":
            warnings.warn(
                "The current search API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            results = {"results": original_memories}
        else:
            results = {"results": original_memories}

        # Graph-first re-ranking (Context-Graph Layer) — fully opt-in.
        should_use_cgl = self._should_use_context_graph(use_context_graph)
        if should_use_cgl:
            # Phase F6 — build a RoleContext for the policy layer. tenant_id
            # / role come from the filters dict the caller supplied (e.g.
            # ``filters={"tenant_id": "acme"}``). Falls through harmlessly
            # when those keys are absent.
            role_ctx = self._build_role_ctx(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                filters=effective_filters,
            )
            expanded = self._graph_first_rerank(
                query=query,
                seed_results=results.get("results", []),
                limit=limit,
                include_archived=include_archived,
                role_ctx=role_ctx,
            )
            if expanded is not None:
                results["results"] = expanded["results"]
                if expanded.get("subgraph"):
                    results["subgraph"] = expanded["subgraph"]

        # NEW: Apply chunk merging if enabled
        if self.config.chunking.enabled and merge_chunks:
            results = self._merge_chunk_results(
                results,
                chunk_context_window=chunk_context_window
            )

        return results

    def _emit_mspr_telemetry(
        self,
        expanded: Dict[str, Any],
        *,
        seed_count: int,
    ) -> None:
        """Phase F8 — fire ``outhad_contextkit.mspr.search`` once per search.

        Aggregates per-candidate context_graph debug payloads into a
        single counts payload so dashboards can chart adoption of each
        sub-feature without per-result fan-out.
        """
        cfg = self.config.mspr
        results = expanded.get("results") or []
        ctx_payloads = [r.get("context_graph") or {} for r in results]
        feedback_hits = sum(
            1 for c in ctx_payloads if abs(float(c.get("personal", 0.0))) > 0
        )
        success_hits = sum(
            1 for c in ctx_payloads if float(c.get("success", 0.0)) > 0
        )
        frequency_hits = sum(
            1 for c in ctx_payloads if float(c.get("frequency", 0.0)) > 0
        )
        capture_event(
            "outhad_contextkit.mspr.search",
            self,
            {
                "intent_enabled": bool(cfg.intent.enabled),
                "feedback_enabled": bool(cfg.feedback.enabled),
                "frequency_enabled": bool(cfg.frequency.enabled),
                "role_enabled": bool(cfg.role.enabled),
                "success_enabled": bool(cfg.success.enabled),
                "seed_count": int(seed_count),
                "result_count": len(results),
                "feedback_hits": int(feedback_hits),
                "success_hits": int(success_hits),
                "frequency_hits": int(frequency_hits),
                "sync_type": "sync",
            },
        )

    def _build_role_ctx(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Phase F6 — assemble a RoleContext for the policy layer.

        ``tenant_id`` and ``role`` are read from the supplied filters
        dict (or its nested ``metadata`` dict, mirroring the vector-store
        payload shape). When neither MSPR nor a role policy is engaged
        this method still returns a ``RoleContext`` so call-sites stay
        cheap; the returned object is simply ignored downstream.
        """
        from outhad_contextkit.memory.personalized.types import RoleContext

        tenant_id = None
        role = None
        if isinstance(filters, dict):
            tenant_id = filters.get("tenant_id")
            role = filters.get("role")
            if tenant_id is None or role is None:
                nested = filters.get("metadata")
                if isinstance(nested, dict):
                    tenant_id = tenant_id or nested.get("tenant_id")
                    role = role or nested.get("role")
        return RoleContext(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            tenant_id=tenant_id,
            role=role,
        )

    def _should_use_context_graph(self, override: Optional[bool]) -> bool:
        if override is False:
            return False
        if self._context_graph is None:
            return False
        if override is True:
            return True
        retrieval_cfg = getattr(self.config.context_graph, "retrieval", None)
        return bool(retrieval_cfg and retrieval_cfg.enabled)

    def _graph_first_rerank(
        self,
        *,
        query: str,
        seed_results: list,
        limit: int,
        include_archived: bool,
        role_ctx: Optional[Any] = None,
    ):
        if self._context_graph is None:
            return None
        try:
            retrieval_cfg = self.config.context_graph.retrieval
            self._context_graph.maybe_tick_decay()

            # Phase F6 — apply RolePolicy.allow as a hard pre-rerank filter
            # when configured. Soft (reweight) mode still runs but defers
            # the score adjustment to the post-rerank pass below.
            mspr_cfg_root = getattr(self.config, "mspr", None)
            role_enabled = bool(
                mspr_cfg_root
                and getattr(mspr_cfg_root, "enabled", False)
                and getattr(getattr(mspr_cfg_root, "role", None), "enabled", False)
                and getattr(self, "_role_policy", None) is not None
            )
            if role_enabled and role_ctx is not None:
                from outhad_contextkit.memory.personalized.role import (
                    apply_role_policy,
                )

                hard_filter = bool(
                    getattr(mspr_cfg_root.role, "hard_filter", True)
                )
                if hard_filter:
                    seed_results = apply_role_policy(
                        list(seed_results),
                        policy=self._role_policy,
                        role_ctx=role_ctx,
                        hard_filter=True,
                    )

            # Phase F5 — apply intent-based weight override before building
            # the retriever so the per-intent α/β/γ/δ deltas take effect.
            # Returns the original config unchanged when intent is disabled.
            mspr_cfg = getattr(self.config, "mspr", None)
            if (
                mspr_cfg is not None
                and getattr(mspr_cfg, "enabled", False)
                and getattr(getattr(mspr_cfg, "intent", None), "enabled", False)
                and getattr(self, "_intent_router", None) is not None
            ):
                try:
                    from outhad_contextkit.memory.personalized.intent import (
                        apply_weight_override,
                    )

                    intent = self._intent_router.classify(query)
                    retrieval_cfg = apply_weight_override(retrieval_cfg, intent.override)
                    logger.debug(
                        "Intent classified as %s (conf=%.2f); weights adjusted",
                        intent.label,
                        intent.confidence,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Intent override failed: %s", exc)

            algorithm = getattr(retrieval_cfg, "algorithm", "bfs")
            if algorithm == "ppr":
                from outhad_contextkit.memory.context_graph.ppr_retriever import (
                    PersonalisedPageRankRetriever,
                )

                retriever = PersonalisedPageRankRetriever(
                    self._context_graph, retrieval_cfg
                )
            else:
                from outhad_contextkit.memory.context_graph.retriever import (
                    GraphFirstRetriever,
                )

                retriever = GraphFirstRetriever(
                    self._context_graph, retrieval_cfg
                )
            q_embedding = None
            try:
                q_embedding = self.embedding_model.embed(query, "search")
            except Exception as exc:  # pragma: no cover - embed is best-effort
                logger.debug("CGL query embed failed: %s", exc)

            # Phase F4 — build a personal-boost provider only when MSPR
            # feedback is enabled AND δ·personal is tuned > 0. When the
            # user has not opted in the provider stays None and the
            # retriever's score formula collapses to pre-MSPR maths.
            mspr_cfg = getattr(self.config, "mspr", None)
            personal_provider = None
            if (
                mspr_cfg is not None
                and getattr(mspr_cfg, "enabled", False)
                and getattr(
                    getattr(mspr_cfg, "feedback", None), "enabled", False
                )
                and getattr(retrieval_cfg, "delta_personal", 0.0) > 0
                and getattr(self, "_feedback_store", None) is not None
            ):
                try:
                    from outhad_contextkit.memory.personalized.providers import (
                        PersonalBoostProvider,
                    )

                    personal_provider = PersonalBoostProvider(
                        self._feedback_store,
                        user_id=None,
                        agent_id=None,
                        run_id=None,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "PersonalBoostProvider init failed: %s", exc
                    )
                    personal_provider = None

            # Phase F7 — build the SuccessProvider only when the success
            # store is enabled AND ζ·success > 0. The query hash is the
            # same SHA-256[:16] used by record_feedback so the lookup key
            # space stays consistent across writes and reads.
            success_provider = None
            if (
                mspr_cfg is not None
                and getattr(mspr_cfg, "enabled", False)
                and getattr(getattr(mspr_cfg, "success", None), "enabled", False)
                and getattr(retrieval_cfg, "zeta_success", 0.0) > 0
                and getattr(self, "_success_store", None) is not None
            ):
                try:
                    from outhad_contextkit.memory.personalized.feedback_store import (
                        hash_query,
                    )
                    from outhad_contextkit.memory.personalized.success_store import (
                        SuccessProvider,
                    )

                    q_hash = hash_query(query)
                    success_provider = SuccessProvider(
                        self._success_store,
                        query_hash=q_hash,
                        user_id=(role_ctx.user_id if role_ctx is not None else None),
                        min_samples=int(
                            getattr(
                                mspr_cfg.success, "min_sample_size", 3
                            )
                        ),
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("SuccessProvider init failed: %s", exc)
                    success_provider = None

            expanded = retriever.retrieve(
                query,
                seed_results,
                limit=limit,
                payload_resolver=self._cgl_resolve_payloads,
                query_embedding=q_embedding,
                include_archived=include_archived,
                personal_boost=personal_provider,
                success_provider=success_provider,
            )

            # Phase F6 — soft-filter pass: when role is enabled but
            # ``hard_filter=False`` we keep all candidates and multiply
            # their score via ``RolePolicy.reweight``. Re-sort to honour
            # the new scores.
            if (
                role_enabled
                and role_ctx is not None
                and not getattr(mspr_cfg_root.role, "hard_filter", True)
                and expanded
                and expanded.get("results")
            ):
                try:
                    from outhad_contextkit.memory.personalized.role import (
                        apply_role_policy,
                    )

                    expanded["results"] = apply_role_policy(
                        expanded["results"],
                        policy=self._role_policy,
                        role_ctx=role_ctx,
                        hard_filter=False,
                    )
                    expanded["results"].sort(
                        key=lambda r: float(r.get("score") or 0.0),
                        reverse=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Role soft-filter failed: %s", exc)
            # Phase F2 — only record_access on the top-K we are about to
            # hand to the caller. Skipping BFS-expanded but unranked nodes
            # keeps the access_count signal proportional to what the user
            # actually consumes.
            mspr_cfg = getattr(self.config, "mspr", None)
            frequency_enabled = bool(
                mspr_cfg
                and getattr(mspr_cfg, "enabled", False)
                and getattr(getattr(mspr_cfg, "frequency", None), "enabled", False)
            )
            if frequency_enabled and expanded and expanded.get("results"):
                for entry in expanded["results"][:limit]:
                    mem_id = entry.get("id")
                    if not mem_id:
                        continue
                    try:
                        self._context_graph.record_access(mem_id)
                    except Exception as exc:  # pragma: no cover - logging-only
                        logger.debug(
                            "record_access failed for %s: %s", mem_id, exc
                        )

            # Phase F8 — telemetry. Fires once per MSPR-enabled search;
            # no-op when the master switch is off so existing CGL-only
            # users see no new events. Best-effort; never raises.
            if (
                mspr_cfg is not None
                and getattr(mspr_cfg, "enabled", False)
                and expanded is not None
            ):
                try:
                    self._emit_mspr_telemetry(expanded, seed_count=len(seed_results))
                except Exception as exc:  # pragma: no cover - telemetry never fatal
                    logger.debug("MSPR telemetry emit failed: %s", exc)
            return expanded
        except Exception as exc:  # pragma: no cover - hook must never raise
            logger.debug("Graph-first rerank failed: %s", exc)
            return None

    def _cgl_resolve_payloads(self, ids):
        """Best-effort payload lookup for BFS-expanded candidates."""
        resolved: Dict[str, Dict[str, Any]] = {}
        for cid in ids:
            try:
                row = self.vector_store.get(vector_id=cid)
            except Exception:
                row = None
            if not row:
                continue
            payload = getattr(row, "payload", {}) or {}
            resolved[cid] = {
                "id": cid,
                "memory": payload.get("data", ""),
                "hash": payload.get("hash"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "score": 0.0,
                "user_id": payload.get("user_id"),
                "agent_id": payload.get("agent_id"),
                "run_id": payload.get("run_id"),
            }
        return resolved

    @property
    def context_graph(self):
        """Public handle to the Context-Graph Layer facade.

        Returns ``None`` when ``MemoryConfig.context_graph.enabled`` is False.
        Examples and tests should prefer this public attribute over the
        private ``_context_graph`` field.
        """
        return self._context_graph

    def tick_decay(self) -> Dict[str, int]:
        """Apply one round of CGL decay. No-op when the layer is disabled."""
        if self._context_graph is None:
            return {"decayed": 0, "pruned": 0, "archived": 0}
        return self._context_graph.tick_decay()

    def reset_context_graph(self) -> None:
        """Drop every CGL node / edge while leaving vector and entity stores alone.

        Emits a single synthetic ``reset`` changelog event with ``target_id="*"``
        so subscribers can detect full resets.
        """
        if self._context_graph is None:
            return
        try:
            for node in list(self._context_graph.backend.iter_nodes()):
                self._context_graph.backend.delete_node(node.id)
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.debug("reset_context_graph failed: %s", exc)
        # Emit synthetic reset event for change-bus subscribers
        if (
            self._context_graph.changelog is not None
            and self._context_graph.config.log_changes
        ):
            try:
                from datetime import timezone
                from outhad_contextkit.memory.context_graph.types import ChangeEvent
                self._context_graph.changelog.append(
                    ChangeEvent(
                        event_type="reset",
                        target_id="*",
                        timestamp=__import__("datetime").datetime.now(timezone.utc),
                        user_id=None,
                        payload={},
                    )
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("reset_context_graph: failed to emit reset event: %s", exc)

    def subscribe_context_graph_changes(
        self,
        callback: Callable[[Any], None],
        *,
        from_event_id: Optional[int] = None,
        event_types: Optional[Iterable[str]] = None,
    ) -> Optional[int]:
        """Register *callback* on the CGL change bus and return a subscription token.

        The callback is invoked in a background daemon thread; it **must not raise**.
        Returns ``None`` when the Context-Graph Layer or its changelog is disabled.

        Args:
            callback: ``(ChangeEvent) -> None`` — called once per matching event.
            from_event_id: Replay all events with id > this value before streaming
                live events.  Pass ``0`` to replay the entire history.
            event_types: Optional whitelist of event-type strings (e.g.
                ``["node_added", "edge_added"]``).  ``None`` accepts all types.
        """
        if self._context_graph is None or self._context_graph.changelog is None:
            return None
        return self._context_graph.changelog.subscribe(
            callback,
            from_event_id=from_event_id,
            event_types=event_types,
        )

    def unsubscribe_context_graph_changes(self, token: int) -> bool:
        """Deregister a change-bus subscription by *token*.

        Returns True if the token was found and removed.  Returns False when the
        CGL is disabled or the token is unknown.
        """
        if self._context_graph is None or self._context_graph.changelog is None:
            return False
        return self._context_graph.changelog.unsubscribe(token)

    def mark_relevant(self, memory_id: str, delta: float = 0.25) -> Optional[float]:
        """Boost a memory's CGL relevance and pin a floor against decay.

        Returns the new relevance in ``[0, 1]``, or ``None`` when the
        Context-Graph Layer is disabled or the memory is unknown.
        """
        if self._context_graph is None:
            return None
        return self._context_graph.bump_relevance(memory_id, float(delta), set_floor=True)

    def snapshot_context_graph(self, path: str) -> str:
        """Write a portable JSON-Lines snapshot of the Context-Graph Layer.

        Works for both the networkx and Neo4j backends. Returns the path.
        Raises ``RuntimeError`` if the CGL is disabled.
        """
        if self._context_graph is None:
            raise RuntimeError(
                "Context-Graph Layer is disabled. Enable it via "
                "MemoryConfig.context_graph.enabled=True before snapshotting."
            )
        return self._context_graph.export_jsonl(path)

    def replay_context_graph(
        self,
        *,
        until: Optional[datetime] = None,
        target_backend: Optional[Any] = None,
    ) -> Any:
        """Rebuild a fresh :class:`ContextGraph` from the changelog.

        Returns a *new* :class:`ContextGraph` facade (the production graph
        on ``self`` is left untouched). When ``target_backend`` is omitted
        an in-process :class:`NetworkXBackend` is used, which is a sensible
        default for audit / diff workflows regardless of the live backend.
        ``until`` caps the replay at a point-in-time (inclusive).
        """
        if self._context_graph is None:
            raise RuntimeError(
                "Context-Graph Layer is disabled. Enable it via "
                "MemoryConfig.context_graph.enabled=True before replay."
            )
        from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
            NetworkXBackend,
        )
        from outhad_contextkit.memory.context_graph.facade import ContextGraph
        from outhad_contextkit.memory.context_graph.replay import (
            replay_from_changelog,
        )

        backend = target_backend or NetworkXBackend()
        changelog = self._context_graph.changelog
        if changelog is None:
            raise RuntimeError(
                "Context-Graph changelog is disabled. Enable "
                "MemoryConfig.context_graph.log_changes=True before replay."
            )
        replay_from_changelog(changelog, backend, until=until)
        return ContextGraph(
            config=self._context_graph.config,
            backend=backend,
            changelog=None,
        )

    def mark_irrelevant(self, memory_id: str, delta: float = 0.25) -> Optional[float]:
        """Attenuate a memory's CGL relevance without pinning a floor."""
        if self._context_graph is None:
            return None
        return self._context_graph.bump_relevance(
            memory_id, -abs(float(delta)), set_floor=False
        )

    def backfill_context_graph(
        self,
        *,
        batch_size: int = 500,
        embed: bool = True,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Back-fill the Context-Graph Layer from existing vector-store memories.

        Idempotent: nodes are upserted, not duplicated.  Safe to run multiple
        times.  Use after enabling CGL on a Memory instance that already has
        historical data.

        Args:
            batch_size: Memories processed per embedding batch (controls
                concurrency / rate-limiting with the embedding provider).
            embed: Re-compute embeddings so TOPIC_SIMILAR edges can be
                synthesised.  Set to False for a fast structural-only backfill.
            user_id / agent_id / run_id: Scope the backfill to a subset of
                memories.  When all three are None every memory in the store
                is backfilled.

        Returns:
            ``{"nodes": N, "edges": E}`` where *N* is the number of nodes
            upserted and *E* is the net new edges created.

        Raises:
            RuntimeError: If the Context-Graph Layer is not enabled.
        """
        if self._context_graph is None or self._context_graph_builder is None:
            raise RuntimeError(
                "Context-Graph Layer is not enabled. "
                "Set MemoryConfig.context_graph.enabled=True before backfilling."
            )

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        # Load all matching memories.  The vector_store.list() API has no
        # cursor/offset support, so we fetch one large page.  Backfill is a
        # one-off operation; holding all payloads in RAM is acceptable.
        raw = self.vector_store.list(filters=filters or None, limit=100_000)
        # Different backends return either [list] or list directly.
        if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], list):
            memories = raw[0]
        else:
            memories = raw if isinstance(raw, list) else []

        edges_before = self._context_graph.backend.edge_count()

        # Reset the builder session cache so backfill does not inherit stale
        # REPLY_TO / TEMPORAL_NEXT pointers from a prior live session.
        self._context_graph_builder.reset_session_cache()

        nodes_upserted = 0
        for i in range(0, max(1, len(memories)), batch_size):
            batch = memories[i : i + batch_size]
            for mem in batch:
                text: str = mem.payload.get("data", "")
                promoted = {"user_id", "agent_id", "run_id", "data", "hash", "created_at", "updated_at"}
                metadata = {k: v for k, v in mem.payload.items() if k not in promoted}

                embedding = None
                if embed and text:
                    try:
                        embedding = self.embedding_model.embed(text, "add")
                    except Exception as exc:  # pragma: no cover - provider dependent
                        logger.debug("backfill embed failed for %s: %s", mem.id, exc)

                raw_ts = mem.payload.get("created_at")
                created_at: Optional[datetime] = None
                if isinstance(raw_ts, datetime):
                    created_at = raw_ts
                elif isinstance(raw_ts, str):
                    try:
                        created_at = datetime.fromisoformat(raw_ts)
                    except ValueError:
                        pass

                self._context_graph_builder.on_memory_created(
                    mem.id,
                    text,
                    user_id=mem.payload.get("user_id"),
                    agent_id=mem.payload.get("agent_id"),
                    run_id=mem.payload.get("run_id"),
                    metadata=metadata,
                    embedding=embedding,
                    created_at=created_at,
                )
                nodes_upserted += 1

            logger.debug(
                "backfill_context_graph: processed %d / %d",
                min(i + batch_size, len(memories)),
                len(memories),
            )

        edges_after = self._context_graph.backend.edge_count()
        result: Dict[str, int] = {
            "nodes": nodes_upserted,
            "edges": max(0, edges_after - edges_before),
        }
        logger.info("backfill_context_graph complete: %s", result)
        return result

    def _merge_chunk_results(
        self,
        results: Dict,
        chunk_context_window: int = 1
    ) -> Dict:
        """
        Merge chunks from the same document in search results.

        Args:
            results: Search results dict with 'results' key
            chunk_context_window: Number of neighboring chunks to include (0 = merge only, >0 = include neighbors)

        Returns:
            Updated results dict with merged chunks
        """
        from outhad_contextkit.memory.chunking import ChunkMerger

        result_list = results.get("results", [])
        if not result_list:
            return results

        # Merge chunks from same document
        merged_list = ChunkMerger.merge_by_document(
            result_list,
            include_neighbors=chunk_context_window > 0,
            neighbor_window=chunk_context_window
        )

        results["results"] = merged_list
        logger.info(
            f"Merged {len(result_list)} chunk results into {len(merged_list)} documents "
            f"(context_window={chunk_context_window})"
        )
        return results

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        # Sanitize filters for vector store (remove TCMGM metadata like confidence floats)
        search_filters = _sanitize_filters_for_vector_store(filters)
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=search_filters)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("outhad_contextkit.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("outhad_contextkit.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("outhad_contextkit.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(
        self,
        memory_id,
        *,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.
            tenant_id (Optional[str]): Phase T4 — when supplied, scopes
                the read to rows tagged with this tenant (or legacy
                NULL rows). When omitted and ``MemoryConfig.tenant.enabled``
                is True, the default tenant id is used.
            sub_tenant_id (Optional[str]): Phase T4 — sub-tenant scope
                filter. Same NULL-tolerance as ``tenant_id``.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("outhad_contextkit.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        # Phase T4 — when tenant subsystem is enabled, scope by the
        # resolved tenant. Legacy callers that don't pass tenant_id
        # still get the default-tenant view, which (because the
        # resolver returns is_default=True for that case) maps to
        # "no scope filter" — preserving pre-T4 behaviour.
        eff_tenant = tenant_id
        eff_sub = sub_tenant_id
        resolved = self._resolve_tenant(
            tenant_id=tenant_id, sub_tenant_id=sub_tenant_id
        )
        if resolved is not None and not resolved.is_default:
            eff_tenant = resolved.tenant_id
            eff_sub = resolved.sub_tenant_id
        elif resolved is not None and resolved.is_default:
            eff_tenant = None
            eff_sub = None
        return self.db.get_history(
            memory_id, tenant_id=eff_tenant, sub_tenant_id=eff_sub
        )

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        self._cgl_on_created(memory_id, data, embeddings, metadata)
        capture_event("outhad_contextkit._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("outhad_contextkit._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        self._cgl_on_updated(memory_id, data, embeddings, new_metadata, prev_value=prev_value)
        capture_event("outhad_contextkit._update_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        self._cgl_on_deleted(memory_id, existing_memory.payload)
        capture_event("outhad_contextkit._delete_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = build_history_store(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        capture_event("outhad_contextkit.reset", self, {"sync_type": "sync"})

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = build_history_store(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            from outhad_contextkit.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True
        else:
            self.graph = None

        # Context-Graph Layer (CGL) — shares implementation with sync Memory.
        Memory._init_context_graph(self)
        # Multi-Stage Personalized Retrieval (MSPR) — shared init.
        Memory._init_mspr(self)
        # Tenant subsystem (T1–T3) — shared init.
        Memory._init_tenant(self)
        # Lifecycle subsystem (D1-D8) — shared init.
        Memory._init_lifecycle(self)

        capture_event("outhad_contextkit.init", self, {"sync_type": "async"})

    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(vector_store_task, graph_task)

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
    ):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                msg_content = message_dict["content"]
                actor_name = message_dict.get("name")

                # NEW: Adaptive chunking logic
                # Only chunk if:
                # 1. Chunking is enabled
                # 2. Document is large enough (>= min_document_size)
                # 3. Document is larger than chunk_size
                should_chunk = (
                    self._chunker and
                    len(msg_content) >= self.config.chunking.min_document_size and
                    len(msg_content) > self.config.chunking.chunk_size
                )

                if should_chunk:
                    # Chunk the content
                    chunks = await asyncio.to_thread(
                        self._chunker.split_text,
                        msg_content,
                        metadata={
                            "role": message_dict["role"],
                            "actor_id": actor_name,
                            **metadata
                        }
                    )

                    logger.info(
                        f"Chunked message into {len(chunks)} chunks "
                        f"(original: {len(msg_content)} chars)"
                    )

                    # Store each chunk separately
                    for chunk in chunks:
                        chunk_metadata = {
                            **chunk.metadata,
                            **metadata,
                            "role": message_dict["role"],
                        }
                        if actor_name:
                            chunk_metadata["actor_id"] = actor_name

                        chunk_embeddings = await asyncio.to_thread(self.embedding_model.embed, chunk.content, "add")
                        mem_id = await self._create_memory(
                            chunk.content,
                            chunk_embeddings,
                            chunk_metadata
                        )

                        returned_memories.append({
                            "id": mem_id,
                            "memory": chunk.content,
                            "event": "ADD",
                            "actor_id": actor_name,
                            "role": message_dict["role"],
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "document_id": chunk.document_id
                        })
                else:
                    # Existing non-chunked path
                    per_msg_meta = deepcopy(metadata)
                    per_msg_meta["role"] = message_dict["role"]

                    if actor_name:
                        per_msg_meta["actor_id"] = actor_name

                    msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                    mem_id = await self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                    returned_memories.append({
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name,
                        "role": message_dict["role"],
                    })

            return returned_memories

        parsed_messages = parse_messages(messages)
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                limit=5,
                filters=effective_filters,  # 'filters' is query_filters_for_inference
            )
            return [{"id": mem.id, "text": mem.payload["data"]} for mem in existing_mems]

        search_tasks = [process_fact_for_search(fact) for fact in new_retrieved_facts]
        search_results_list = await asyncio.gather(*search_tasks)
        for result_group in search_results_list:
            retrieved_old_memory.extend(result_group)

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )
            try:
                response = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""
            try:
                response = remove_code_blocks(response)
                new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}

        returned_memories = []
        try:
            memory_tasks = []
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        continue
                    event_type = resp.get("event")

                    if event_type == "ADD":
                        task = asyncio.create_task(
                            self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "ADD", None))
                    elif event_type == "UPDATE":
                        task = asyncio.create_task(
                            self._update_memory(
                                memory_id=temp_uuid_mapping[resp["id"]],
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
                    elif event_type == "DELETE":
                        task = asyncio.create_task(self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")]))
                        memory_tasks.append((task, resp, "DELETE", temp_uuid_mapping[resp.get("id")]))
                    elif event_type == "NONE":
                        logger.info("NOOP for Memory (async).")
                except Exception as e:
                    logger.error(f"Error processing memory action (async): {resp}, Error: {e}")

            for task, resp, event_type, mem_id in memory_tasks:
                try:
                    result_id = await task
                    if event_type == "ADD":
                        returned_memories.append({"id": result_id, "memory": resp.get("text"), "event": event_type})
                    elif event_type == "UPDATE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        returned_memories.append({"id": mem_id, "memory": resp.get("text"), "event": event_type})
                except Exception as e:
                    logger.error(f"Error awaiting memory task (async): {e}")
        except Exception as e:
            logger.error(f"Error in memory processing loop (async): {e}")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "outhad_contextkit.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("outhad_contextkit.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

         Args:
             user_id (str, optional): user id
             agent_id (str, optional): agent id
             run_id (str, optional): run id
             filters (dict, optional): Additional custom key-value filters to apply to the search.
                 These are merged with the ID-based scoping filters. For example,
                 `filters={"actor_id": "some_user"}`.
             limit (int, optional): The maximum number of memories to return. Defaults to 100.

         Returns:
             dict: A dictionary containing a list of memories under the "results" key,
                   and potentially "relations" if graph store is enabled. For API v1.0,
                   it might return a direct list (see deprecation warning).
                   Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "When 'conversation_id' is not provided (classic mode), "
                "at least one of 'user_id', 'agent_id', or 'run_id' must be specified for get_all."
            )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "outhad_contextkit.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'` (which returns a dict with a 'results' key). "
                "The current format (direct list for v1.0) will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories_result
        else:
            return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)
        actual_memories = (
            memories_result[0]
            if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0
            else memories_result
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        merge_chunks: Optional[bool] = None,
        chunk_context_window: Optional[int] = None,
        use_context_graph: Optional[bool] = None,
        include_archived: bool = False,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            merge_chunks (bool, optional): Whether to merge chunks from same document.
                Defaults to config.chunking.merge_chunks_on_retrieval.
            chunk_context_window (int, optional): Number of neighboring chunks to include (±N).
                Defaults to config.chunking.chunk_context_window.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("at least one of 'user_id', 'agent_id', or 'run_id' must be specified ")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "outhad_contextkit.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
            },
        )

        vector_store_task = asyncio.create_task(self._search_vector_store(query, effective_filters, limit, threshold))

        graph_task = None
        if self.enable_graph:
            if hasattr(self.graph.search, "__await__"):  # Check if graph search is async
                graph_task = asyncio.create_task(self.graph.search(query, effective_filters, limit))
            else:
                graph_task = asyncio.create_task(asyncio.to_thread(self.graph.search, query, effective_filters, limit))

        if graph_task:
            original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            original_memories = await vector_store_task
            graph_entities = None

        # NEW: Set defaults from config for chunk merging
        if merge_chunks is None:
            merge_chunks = self.config.chunking.merge_chunks_on_retrieval
        if chunk_context_window is None:
            chunk_context_window = self.config.chunking.chunk_context_window

        # Build results dict
        if self.enable_graph:
            results = {"results": original_memories, "relations": graph_entities}
        elif self.api_version == "v1.0":
            warnings.warn(
                "The current search API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in outhad_contextkitai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            results = {"results": original_memories}
        else:
            results = {"results": original_memories}

        # Graph-first re-ranking (Context-Graph Layer) — fully opt-in.
        should_use_cgl = self._should_use_context_graph(use_context_graph)
        if should_use_cgl:
            expanded = await asyncio.to_thread(
                self._graph_first_rerank,
                query=query,
                seed_results=results.get("results", []),
                limit=limit,
                include_archived=include_archived,
            )
            if expanded is not None:
                results["results"] = expanded["results"]
                if expanded.get("subgraph"):
                    results["subgraph"] = expanded["subgraph"]

        # NEW: Apply chunk merging if enabled
        if self.config.chunking.enabled and merge_chunks:
            results = await asyncio.to_thread(
                self._merge_chunk_results,
                results,
                chunk_context_window=chunk_context_window
            )

        return results

    async def tick_decay(self) -> Dict[str, int]:
        """Async wrapper around the CGL decay primitive."""
        if self._context_graph is None:
            return {"decayed": 0, "pruned": 0, "archived": 0}
        return await asyncio.to_thread(self._context_graph.tick_decay)

    async def reset_context_graph(self) -> None:
        if self._context_graph is None:
            return
        await asyncio.to_thread(Memory.reset_context_graph, self)

    async def mark_relevant(
        self, memory_id: str, delta: float = 0.25
    ) -> Optional[float]:
        """Async twin of :meth:`Memory.mark_relevant`."""
        if self._context_graph is None:
            return None
        return await asyncio.to_thread(
            Memory.mark_relevant, self, memory_id, delta
        )

    async def mark_irrelevant(
        self, memory_id: str, delta: float = 0.25
    ) -> Optional[float]:
        """Async twin of :meth:`Memory.mark_irrelevant`."""
        if self._context_graph is None:
            return None
        return await asyncio.to_thread(
            Memory.mark_irrelevant, self, memory_id, delta
        )

    async def record_feedback(
        self,
        memory_id: str,
        *,
        helpful: bool,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async twin of :meth:`Memory.record_feedback`."""
        return await asyncio.to_thread(
            Memory.record_feedback,
            self,
            memory_id,
            helpful=helpful,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            query=query,
            metadata=metadata,
        )

    async def reset_feedback(
        self, *, user_id: Optional[str] = None
    ) -> int:
        """Async twin of :meth:`Memory.reset_feedback`."""
        return await asyncio.to_thread(
            Memory.reset_feedback, self, user_id=user_id
        )

    def set_role_policy(self, policy: Optional[Any]) -> None:
        """Async twin of :meth:`Memory.set_role_policy` (sync — stores a ref)."""
        Memory.set_role_policy(self, policy)

    def get_role_policy(self) -> Optional[Any]:
        """Async twin of :meth:`Memory.get_role_policy`."""
        return Memory.get_role_policy(self)

    async def backfill_mspr(self) -> Dict[str, int]:
        """Async twin of :meth:`Memory.backfill_mspr`."""
        return await asyncio.to_thread(Memory.backfill_mspr, self)

    async def backfill_tenant(
        self,
        *,
        default_tenant_id: Optional[str] = None,
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """Async twin of :meth:`Memory.backfill_tenant`."""
        return await asyncio.to_thread(
            Memory.backfill_tenant,
            self,
            default_tenant_id=default_tenant_id,
            batch_size=batch_size,
        )

    async def export_tenant(
        self, tenant_id: str, dest_dir: str
    ) -> Dict[str, int]:
        """Async twin of :meth:`Memory.export_tenant`."""
        return await asyncio.to_thread(
            Memory.export_tenant, self, tenant_id, dest_dir
        )

    async def import_tenant(self, src_dir: str) -> Dict[str, int]:
        """Async twin of :meth:`Memory.import_tenant`."""
        return await asyncio.to_thread(Memory.import_tenant, self, src_dir)

    async def migrate_tenant(
        self, src_id: str, dst_id: str
    ) -> Dict[str, int]:
        """Async twin of :meth:`Memory.migrate_tenant`."""
        return await asyncio.to_thread(
            Memory.migrate_tenant, self, src_id, dst_id
        )

    # ------------------------------------------------------------------
    # Lifecycle (Part D) async twins.
    # ------------------------------------------------------------------
    async def decay_score(self, memory_id: str) -> Optional[float]:
        return await asyncio.to_thread(Memory.decay_score, self, memory_id)

    async def archive_low(
        self, *, threshold: Optional[float] = None, dry_run: bool = False
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            Memory.archive_low, self, threshold=threshold, dry_run=dry_run
        )

    async def start_decay_scheduler(self) -> None:
        return await asyncio.to_thread(Memory.start_decay_scheduler, self)

    async def stop_decay_scheduler(self, *, timeout: float = 5.0) -> None:
        return await asyncio.to_thread(
            Memory.stop_decay_scheduler, self, timeout=timeout
        )

    async def close(self) -> None:
        return await asyncio.to_thread(Memory.close, self)

    async def record_reference(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        strength: float = 1.0,
    ) -> Optional[int]:
        return await asyncio.to_thread(
            Memory.record_reference,
            self,
            memory_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            strength=strength,
        )

    async def versions(self, memory_id: str) -> List[Any]:
        return await asyncio.to_thread(Memory.versions, self, memory_id)

    async def latest_version(self, memory_id: str) -> Optional[Any]:
        return await asyncio.to_thread(
            Memory.latest_version, self, memory_id
        )

    async def diff_versions(
        self, old_id: str, new_id: str, *, mode: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(
            Memory.diff_versions, self, old_id, new_id, mode=mode
        )

    async def rollback(
        self,
        memory_id: str,
        *,
        to_version: int,
        actor_id: Optional[str] = None,
    ) -> Optional[str]:
        return await asyncio.to_thread(
            Memory.rollback,
            self,
            memory_id,
            to_version=to_version,
            actor_id=actor_id,
        )

    async def demote_to_cold(self, memory_id: str) -> bool:
        return await asyncio.to_thread(
            Memory.demote_to_cold, self, memory_id
        )

    async def promote_from_cold(
        self, memory_id: str
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            Memory.promote_from_cold, self, memory_id
        )

    async def backfill_decay_v2(self) -> Dict[str, int]:
        return await asyncio.to_thread(Memory.backfill_decay_v2, self)

    async def snapshot_context_graph(self, path: str) -> str:
        """Async twin of :meth:`Memory.snapshot_context_graph`."""
        return await asyncio.to_thread(Memory.snapshot_context_graph, self, path)

    def subscribe_context_graph_changes(
        self,
        callback: Callable[[Any], None],
        *,
        from_event_id: Optional[int] = None,
        event_types: Optional[Iterable[str]] = None,
    ) -> Optional[int]:
        """Async-safe twin of :meth:`Memory.subscribe_context_graph_changes`.

        Subscribe is synchronous (thread registration); the callback fires in a
        background daemon thread regardless of whether the caller is sync or async.
        """
        return Memory.subscribe_context_graph_changes(
            self,
            callback,
            from_event_id=from_event_id,
            event_types=event_types,
        )

    def unsubscribe_context_graph_changes(self, token: int) -> bool:
        """Async-safe twin of :meth:`Memory.unsubscribe_context_graph_changes`."""
        return Memory.unsubscribe_context_graph_changes(self, token)

    async def replay_context_graph(
        self,
        *,
        until: Optional[datetime] = None,
        target_backend: Optional[Any] = None,
    ) -> Any:
        """Async twin of :meth:`Memory.replay_context_graph`."""
        return await asyncio.to_thread(
            Memory.replay_context_graph,
            self,
            until=until,
            target_backend=target_backend,
        )

    async def backfill_context_graph(
        self,
        *,
        batch_size: int = 500,
        embed: bool = True,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Async wrapper around :meth:`Memory.backfill_context_graph`."""
        return await asyncio.to_thread(
            Memory.backfill_context_graph,
            self,
            batch_size=batch_size,
            embed=embed,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

    _should_use_context_graph = Memory._should_use_context_graph
    _graph_first_rerank = Memory._graph_first_rerank
    _cgl_resolve_payloads = Memory._cgl_resolve_payloads

    async def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    async def update(self, memory_id, data):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("outhad_contextkit.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("outhad_contextkit.delete", self, {"memory_id": memory_id, "sync_type": "async"})
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("outhad_contextkit.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(
        self,
        memory_id,
        *,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.
            tenant_id (Optional[str]): Phase T4 — tenant scope filter.
            sub_tenant_id (Optional[str]): Phase T4 — sub-tenant scope filter.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("outhad_contextkit.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(
            Memory.history, self, memory_id,
            tenant_id=tenant_id, sub_tenant_id=sub_tenant_id,
        )

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )

        await asyncio.to_thread(Memory._cgl_on_created, self, memory_id, data, embeddings, metadata)

        capture_event("outhad_contextkit._create_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("outhad_contextkit._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        await asyncio.to_thread(
            Memory._cgl_on_updated, self, memory_id, data, embeddings, new_metadata, prev_value
        )
        capture_event("outhad_contextkit._update_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload["data"]

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        await asyncio.to_thread(Memory._cgl_on_deleted, self, memory_id, existing_memory.payload)

        capture_event("outhad_contextkit._delete_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = build_history_store(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("outhad_contextkit.reset", self, {"sync_type": "async"})

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
