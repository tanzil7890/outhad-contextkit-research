# LoCoMo Evaluation - Usage Guide

**Quick Start Guide for Running TCMGM Benchmarks**

---

## 🚀 Quick Start

### Run Full Evaluation (Baseline + TCMGM)

```bash
python evaluation/run_locomo_eval.py evaluation/locomo_dataset_sample.json
```

**Output**:
- `evaluation/results_tcmgm_False.json` - Baseline results
- `evaluation/results_tcmgm_True.json` - TCMGM results
- Console comparison report

---

## 📋 Prerequisites

### Environment Variables (.env)

```bash
# Required for both baseline and TCMGM
OPENAI_API_KEY=your_openai_key_here

# Required only for TCMGM evaluation
NEO4J_URI=bolt://localhost:7689
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### Install Dependencies

```bash
pip install outhad_contextkit
```

---

## 💻 Programmatic Usage

### 1. Basic Adapter Usage

```python
from evaluation.locomo_adapter import LoCoMoAdapter
from outhad_contextkit import Memory

# Initialize memory (baseline mode)
memory = Memory()

# Create adapter
adapter = LoCoMoAdapter(memory)

# Ingest persona
persona = {
    "events": [
        {"content": "Alice graduated from MIT in 2020", "type": "education"},
        {"content": "Alice works at Tech Corp", "type": "employment"}
    ]
}
adapter.ingest_persona(persona, user_id="alice")

# Run QA evaluation
questions = [
    {"question": "Where did Alice graduate from?", "answer": "MIT"}
]
results = adapter.evaluate_qa(questions, user_id="alice", use_tcmgm=False)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct_answers']}/{results['total_questions']}")
```

### 2. With TCMGM Enabled

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig
from evaluation.locomo_adapter import LoCoMoAdapter

# Initialize memory with Neo4j (TCMGM mode)
config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
    )
)
memory = Memory(config=config)

# Create adapter (automatically detects TCMGM)
adapter = LoCoMoAdapter(memory)
print(f"TCMGM available: {adapter._tcmgm_available}")

# Run evaluation with TCMGM features
results = adapter.evaluate_qa(questions, user_id="alice", use_tcmgm=True)
```

### 3. Conversation Ingestion

```python
conversation = [
    {"role": "user", "content": "I finished the project"},
    {"role": "assistant", "content": "Great!"},
    {"role": "user", "content": "Now I'll present it tomorrow"}
]

# Ingest with timeline building (if TCMGM enabled)
from datetime import datetime
result = adapter.ingest_conversation(
    conversation,
    user_id="alice",
    session_start=datetime.utcnow()
)

print(f"Events extracted: {len(result['events'])}")
print(f"Causal links found: {len(result['causal_links'])}")
```

### 4. Summarization Evaluation

```python
summaries = [
    {
        "time_window": {
            "start": "2024-01-15T10:00:00Z",
            "end": "2024-01-15T11:00:00Z"
        },
        "reference_summary": "Alice completed a project and planned to present it."
    }
]

results = adapter.evaluate_summarization(
    summaries,
    user_id="alice",
    use_tcmgm=True  # Use timeline-based summarization
)

print(f"Avg FactScore: {results['avg_fact_score']:.3f}")
```

---

## 📊 Dataset Format

### LoCoMo JSON Structure

```json
{
  "test_user_id": "user_001",
  "personas": [
    {
      "user_id": "user_001",
      "name": "Alice",
      "events": [
        {
          "content": "Alice graduated from MIT in 2020",
          "type": "education",
          "metadata": {"year": 2020, "institution": "MIT"}
        }
      ]
    }
  ],
  "conversations": [
    {
      "user_id": "user_001",
      "start_time": "2024-01-15T10:00:00Z",
      "messages": [
        {"role": "user", "content": "I finished the project"},
        {"role": "assistant", "content": "Great!"}
      ]
    }
  ],
  "qa_questions": [
    {
      "question": "Where did Alice graduate from?",
      "answer": "MIT",
      "time_window": null
    },
    {
      "question": "What happened after the project?",
      "answer": "Presentation",
      "time_window": {
        "start": "2024-01-15T09:00:00Z",
        "end": "2024-01-15T12:00:00Z"
      }
    }
  ],
  "summarization_tasks": [
    {
      "time_window": {
        "start": "2024-01-15T10:00:00Z",
        "end": "2024-01-15T11:00:00Z"
      },
      "reference_summary": "Alice completed a project."
    }
  ]
}
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# Unit tests (mocked)
pytest tests/memory/temporal/test_locomo_evaluation.py -v

# Real integration tests (requires OpenAI + Neo4j)
python tests/memory/temporal/test_real_integration.py
```

### Expected Output

```
================================================================================
🎉 ALL  TESTS PASSED! 🎉
================================================================================

Test Summary:
  ✅ Memory initialization: Working
  ✅ LoCoMo adapter: Working
  ✅ Persona ingestion: Working
  ✅ Conversation ingestion: Working
  ✅ QA evaluation: 100% accuracy
  ✅ Summarization: Working
  ✅ Helper methods: All validated
```

---

## 📈 Results Format

### JSON Output Structure

```json
{
  "tcmgm_enabled": true,
  "qa_accuracy": 0.75,
  "qa_correct": 6,
  "qa_total": 8,
  "avg_fact_score": 0.456,
  "summarization_total": 3,
  "timestamp": "2025-11-14T22:51:23.881196",
  "dataset_path": "evaluation/locomo_dataset_sample.json",
  "detailed_qa": [
    {
      "question": "Where did Alice work?",
      "predicted_answer": "Tech Corp",
      "expected_answer": "Tech Corp",
      "correct": true,
      "context_used": "Alice started working at Tech Corp..."
    }
  ],
  "detailed_summarization": [
    {
      "generated_summary": "Alice completed a project...",
      "reference_summary": "Alice completed a project...",
      "fact_score": 0.85
    }
  ]
}
```

---

## 🔍 Interpreting Results

### QA Accuracy

- **Formula**: `correct_answers / total_questions`
- **Good**: > 80%
- **Acceptable**: 60-80%
- **Needs improvement**: < 60%

### FactScore

- **Formula**: F1 score of sentence overlap
- **Range**: 0.0 to 1.0
- **Good**: > 0.7
- **Acceptable**: 0.5-0.7
- **Needs improvement**: < 0.5

### Comparison Delta

```
QA Accuracy:
  Baseline:  75.00% (6/8)
  TCMGM:     87.50% (7/8)
  Delta:     +12.5% ✅ Improvement

FactScore:
  Baseline:  0.456
  TCMGM:     0.523
  Delta:     +0.067 ✅ Improvement
```

---

## 🛠️ Troubleshooting

### Issue: "No relevant context found"

**Cause**: Conversations not ingested or timeline not built  
**Fix**:
```python
# Ensure conversations are ingested
for conv in dataset["conversations"]:
    adapter.ingest_conversation(conv["messages"], user_id=conv["user_id"])
```

### Issue: TCMGM not available

**Cause**: Neo4j not configured  
**Fix**:
```bash
# Add to .env
NEO4J_URI=bolt://localhost:7689
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Issue: Low QA accuracy

**Possible causes**:
1. Dataset too small (< 10 memories)
2. Questions not aligned with ingested data
3. Time windows too restrictive

**Fixes**:
- Increase dataset size
- Verify question-answer alignment
- Check time window ranges

### Issue: Zero FactScore

**Cause**: Summary format mismatch  
**Fix**:
- Ensure summaries use complete sentences
- Adjust FactScore computation method
- Use more overlap in content

---


## 📚 Advanced Usage

### Custom Evaluation Metrics

```python
# Extend LoCoMoAdapter
class CustomAdapter(LoCoMoAdapter):
    def _evaluate_answer(self, predicted, expected):
        # Use semantic similarity instead of substring matching
        from sentence_transformers import util
        score = util.cos_sim(
            self.embedding_model.embed(predicted),
            self.embedding_model.embed(expected)
        )
        return score > 0.8
```

### Batch Processing

```python
# Evaluate multiple datasets
datasets = ["dataset1.json", "dataset2.json", "dataset3.json"]

results = []
for dataset_path in datasets:
    result = run_evaluation(use_tcmgm=True, dataset_path=dataset_path)
    results.append(result)

# Aggregate results
avg_accuracy = sum(r['qa_accuracy'] for r in results) / len(results)
print(f"Average accuracy: {avg_accuracy:.2%}")
```

### Custom Dataset Creation

```python
import json
from datetime import datetime

dataset = {
    "test_user_id": "custom_user",
    "personas": [{
        "user_id": "custom_user",
        "events": [
            {"content": "Custom event 1", "type": "custom"},
            {"content": "Custom event 2", "type": "custom"}
        ]
    }],
    "conversations": [{
        "user_id": "custom_user",
        "start_time": datetime.utcnow().isoformat(),
        "messages": [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
    }],
    "qa_questions": [
        {"question": "Custom question?", "answer": "Custom answer"}
    ],
    "summarization_tasks": []
}

with open("custom_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

---
