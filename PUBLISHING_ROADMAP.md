# Publishing Roadmap

This roadmap turns `outhad-contextkit` from a broad engineering library into a focused, publishable research artifact.

## Target Thesis

> A temporal-personalized context graph memory architecture improves long-horizon agent retrieval over vector-only and graph-only baselines while preserving privacy and tenant isolation.

The paper should not present the project as "a library with many features." It should present a validated memory architecture with clear evidence.

## Phase 1: Narrow The Contribution

Focus the research claim around four components:

1. **Context Graph Layer**
   - Memories are graph nodes.
   - Edges represent reply, temporal, topic, document, causal, update, and version relations.
   - Retrieval expands through the graph using BFS or Personalized PageRank.

2. **Temporal-Causal Memory**
   - Supports before, after, and time-window queries.
   - Supports causal-chain retrieval.

3. **Personalized Retrieval**
   - Feedback, frequency, intent routing, and query-success signals improve ranking.

4. **Privacy-Aware Memory**
   - Sensitive memories are classified, encrypted, filtered, or redacted.
   - Privacy is measured as a retrieval-safety tradeoff.

Backend integrations such as Pinecone, Qdrant, Neo4j, OpenAI wrappers, and embedding providers should be treated as implementation support, not as the core research novelty.

## Phase 2: Define Research Questions

Recommended research questions:

1. **RQ1:** Does context-graph retrieval improve long-horizon memory recall compared with vector-only and graph-only baselines?
2. **RQ2:** Does temporal-causal retrieval improve performance on time-sensitive and causal questions?
3. **RQ3:** Do personalization signals improve retrieval quality for user-specific memory queries?
4. **RQ4:** What is the cost of the full system in latency, storage, and index-update overhead?
5. **RQ5:** Can privacy filtering reduce sensitive-memory leakage without severely reducing answer quality?
6. **RQ6:** Which components matter most according to ablation experiments?

## Phase 3: Clean The Research Artifact

Before running serious experiments, make the repository reproducible.

Required items:

- Add `pyproject.toml`.
- Add dependency groups: `core`, `dev`, `eval`, `neo4j`, `privacy`, and `all`.
- Add `requirements-lock.txt` or `uv.lock`.
- Add `Dockerfile`.
- Add `docker-compose.yml` for Neo4j and local services.
- Add `LICENSE`.
- Add `CITATION.cff`.
- Add an `evaluation/` directory with dataset loaders, baseline runners, ablation runners, metric scripts, and result aggregation.
- Fix test collection so optional dependency tests skip cleanly.
- Prevent tests from downloading models during collection.
- Prevent imports from failing when optional backends are missing.

This is not just polish. Reviewers often reject strong ideas if the artifact is hard to reproduce.

## Phase 4: Build The Evaluation Suite

Suggested structure:

```text
evaluation/
  datasets/
    locomo_loader.py
    long_context_qa_loader.py
    temporal_qa_loader.py
    personalization_loader.py
    privacy_leakage_loader.py

  baselines/
    vector_only.py
    graph_only.py
    dense_bm25.py
    graphrag_style.py
    existing_memory_frameworks.py

  systems/
    full_contextkit.py
    no_graph.py
    no_decay.py
    no_feedback.py
    no_intent.py
    no_causal.py
    no_ppr.py
    no_privacy.py

  metrics/
    retrieval.py
    answer_quality.py
    privacy.py
    latency.py
    storage.py
    statistics.py

  scripts/
    run_all.py
    run_locomo.py
    run_ablations.py
    aggregate_results.py
```

## Phase 5: Choose Benchmarks

Use multiple dataset and task types.

Minimum benchmark suite:

| Benchmark | Purpose |
|---|---|
| LoCoMo full split | Long conversation memory, multi-session QA, temporal questions |
| Long-context QA | Tests whether graph memory beats plain vector retrieval as context grows |
| Temporal QA | Tests before, after, time-window, and state-change questions |
| Personalization benchmark | Tests same query across different user histories |
| Privacy leakage benchmark | Tests whether sensitive facts leak under adversarial queries |
| Synthetic controlled benchmark | Tests exact ground-truth edges, temporal order, causal chains, and privacy labels |

## Phase 6: Implement Baselines

Use serious baselines, not only previous internal versions.

Recommended baselines:

| Baseline | Description |
|---|---|
| Vector-only RAG | Dense embedding retrieval only |
| BM25-only | Lexical retrieval only |
| Dense + BM25 hybrid | Standard hybrid retrieval baseline |
| Graph-only memory | Entity or memory graph retrieval without dense vector search |
| Vector + graph simple fusion | No temporal module and no personalization |
| GraphRAG-style baseline | Entity extraction, graph traversal, and retrieval |
| Existing memory framework | Compare with an available open-source agent memory framework where feasible |
| Oracle upper bound | Give the system gold relevant memories to estimate maximum answer quality |

## Phase 7: Run Ablations

Ablations prove that each module contributes measurable value.

| Variant | Purpose |
|---|---|
| Full system | Main proposed method |
| No graph | Measures graph contribution |
| No decay | Measures time-aware forgetting |
| No feedback | Measures personalization feedback |
| No frequency | Measures access-count signal |
| No intent routing | Measures query-aware weighting |
| No causal search | Measures causal-chain contribution |
| No PPR | Compares BFS against Personalized PageRank graph expansion |
| No privacy filter | Measures privacy and utility tradeoff |
| No temporal filtering | Measures timeline contribution |
| No RRF fusion | Measures fusion strategy |

For every ablation, report retrieval quality, answer quality, latency, storage overhead, and privacy leakage where relevant.

## Phase 8: Define Metrics

### Retrieval Metrics

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`
- `nDCG@10`

### Answer Metrics

- `Exact Match`
- `Token F1`
- LLM-as-judge as a secondary metric only
- Human evaluation if possible

### Temporal Metrics

- Temporal answer accuracy
- Before/after ordering accuracy
- Time-window hit rate

### Privacy Metrics

- Sensitive leakage rate
- False positive rate
- False negative rate
- Privacy utility drop

### Performance Metrics

- p50, p95, and p99 latency
- Ingest throughput
- Query throughput
- Graph update time
- Storage per memory
- Index size growth

### Statistical Metrics

- Confidence intervals
- Paired bootstrap
- Wilcoxon signed-rank or paired t-test
- Multiple-seed variance

## Phase 9: Run Experiments Properly

Experiment requirements:

1. Run every system variant on the same dataset splits.
2. Use at least three random seeds where randomness exists.
3. Evaluate across many users and sessions, not one conversation.
4. Save every raw result.
5. Save config files for every run.
6. Track model names, embedding models, prompts, temperatures, and run dates.
7. Freeze generated synthetic datasets.

Example output structure:

```text
results/
  2026-06-01_locomo_full/
    config.yaml
    vector_only.jsonl
    dense_bm25.jsonl
    full_system.jsonl
    no_graph.jsonl
    no_decay.jsonl
    summary.csv
    significance_tests.json
```

## Phase 10: Strengthen The Method Section

The paper needs formal clarity.

Document:

- Memory representation: node schema, edge schema, metadata, temporal attributes, and privacy labels.
- Graph construction: edge creation, edge weights, update/version edges, and causal edges.
- Retrieval: seed retrieval, graph expansion, BFS/PPR, score fusion, and personalization terms.
- Decay: exponential decay formula, relevance update, and archive behavior.
- Privacy: classifier, encryption, filtering, redaction behavior, and threat model.
- Complexity: ingest cost, query cost, graph traversal cost, and storage cost.

## Phase 11: Decide Venue Strategy

Possible publication directions:

| Direction | Focus |
|---|---|
| Information retrieval / RAG systems | Retrieval quality and long-horizon memory |
| AI agents / LLM memory | Persistent agent memory |
| Knowledge graph + retrieval | Memory-as-graph architecture |
| Privacy-aware AI memory | Privacy-preserving retrieval |

Most realistic path:

1. Publish an arXiv technical report or workshop paper.
2. Improve evaluation and artifact quality.
3. Submit to a stronger conference or journal.

Do not start with the highest-tier journal until the experiments strongly support the thesis.

## Phase 12: Paper Outline

Recommended paper structure:

1. **Abstract**
   - Problem, method, results, privacy/latency tradeoff.

2. **Introduction**
   - Long-horizon agent memory is hard.
   - Vector retrieval loses temporal, causal, and personalized context.
   - The proposed solution is a temporal-personalized context graph memory.

3. **Related Work**
   - RAG
   - GraphRAG
   - Agent memory
   - Temporal retrieval
   - Personalization
   - Privacy-preserving retrieval

4. **Method**
   - Architecture
   - Graph layer
   - Temporal-causal module
   - Personalized scoring
   - Privacy filter

5. **Experimental Setup**
   - Datasets
   - Baselines
   - Ablations
   - Metrics

6. **Results**
   - Main comparison table
   - Per-task results
   - Ablations
   - Latency and storage
   - Privacy results

7. **Analysis**
   - Where graph retrieval helps
   - Where temporal search helps
   - Failure cases
   - Cost tradeoffs

8. **Limitations**
   - Dependency on LLM extraction
   - Graph growth
   - Privacy classifier errors
   - Dataset limitations

9. **Conclusion**

## Phase 13: Concrete 12-Week Plan

| Week | Goal |
|---|---|
| 1 | Freeze thesis, define research questions, decide datasets and baselines |
| 2 | Add `pyproject.toml`, lockfile, Docker Compose, and fix test collection |
| 3 | Build evaluation harness and result format |
| 4 | Implement vector-only, BM25, dense+BM25, and graph-only baselines |
| 5 | Implement full LoCoMo evaluation |
| 6 | Add temporal QA and personalization datasets |
| 7 | Add privacy leakage benchmark |
| 8 | Implement all ablation configs |
| 9 | Run full experiments across seeds and splits |
| 10 | Aggregate results, statistical tests, and plots |
| 11 | Write paper draft |
| 12 | Clean artifact, verify reproducibility, and prepare submission package |

## Immediate Next Steps

Start with these tasks:

1. Add a proper package file: `pyproject.toml`.
2. Fix `pytest --collect-only` so the suite collects without optional dependency failures.
3. Create `evaluation/` with a real LoCoMo runner.
4. Implement `vector_only`, `dense_bm25`, `graph_only`, and `full_system` configs.
5. Produce the first results table with `Recall@5`, `MRR`, `Answer F1`, and p95 latency on LoCoMo.

Once that first table exists, the project will have evidence for whether the thesis is actually supported.
