# BOG-TS / GOAT-TS — Thinking System

**A single-file, graph-based cognition engine where reasoning emerges from graph dynamics.**

BOG-TS (also referred to as GOAT-TS — *Thinking System*) is a research cognitive system implemented in one Python file. It builds a **concept graph** from text, runs **spreading activation**, **memory state transitions**, **physics-based layout**, and **tension-driven hypothesis generation**. The TS graph is the primary reasoning system; optional local LLMs (Ollama) are used only for concept extraction and explanation.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Subsystems](#subsystems)
- [Optional Ollama Integration](#optional-ollama-integration)
- [Performance & Constraints](#performance--constraints)
- [Output Files](#output-files)
- [License](#license)

---

## Features

- **Single-file implementation** — Entire system in `goat_ts_complete.py`; no external modules beyond NumPy, Matplotlib, and optional Streamlit.
- **Semantic concept graph** — Text is converted to nodes (concepts) and edges (co-occurrence, sentence flow, and **semantic similarity** via cached embeddings).
- **Spreading activation** — Activation propagates over the graph with decay, **mass influence**, and **reinforcement** from repeated activation.
- **Memory states** — Nodes transition between **ACTIVE**, **DORMANT**, and **DEEP** based on activation and time.
- **Graph physics** — Fruchterman–Reingold layout plus **gravitational attraction** (mass × mass / distance²) so conceptual clusters form naturally.
- **Tension detection** — Geometric tension (ideal vs actual edge length), **activation–connectivity mismatch** (high activation but weak links), and **semantic gaps** (similar concepts without edges).
- **Hypothesis engine** — Structured hypotheses (source, target, similarity, tension, confidence, kind) from layout tension, activation-weak pairs, and semantic gaps.
- **Unified reasoning pipeline** — `ts_reason(prompt)` runs: concept extraction → graph update → activation cycles → tension → hypotheses → explanation.
- **Interactive CLI** — `Ask TS >` mode: type prompts and see top nodes, tension pairs, hypotheses, and explanations.
- **Optional Ollama** — Use a local Ollama server only for concept extraction and natural-language explanation; **reasoning stays TS-driven**.

---

## Requirements

- **Python 3.10+** (uses `list[X]`, `X | None` style type hints)
- **NumPy** — graph math, adjacency, activation, layout
- **Matplotlib** — concept map plotting (backend `Agg` for headless use)
- **Streamlit** (optional) — web GUI when running as Streamlit app

No GPU or large models required. Designed to run on **low-end laptops**.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BoggersTheFish/BOG-TS.git
   cd BOG-TS
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or: source venv/bin/activate   # Linux/macOS
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
   Optional (for web UI):
   ```bash
   pip install streamlit
   ```

4. (Optional) Install [Ollama](https://ollama.ai) and pull a model (e.g. `llama3.2`, `nomic-embed-text`) if you want concept extraction and explanation via LLM.

---

## Quick Start

**Batch mode** — run cognition on a string and print tension trend:
```bash
python goat_ts_complete.py --text "Artificial intelligence will change everything. Humans must adapt or be left behind." --seed intelligence --ticks 20 --cycles 3
```

**Interactive mode** — prompt the system in a loop:
```bash
python goat_ts_complete.py --interactive
```
Then type at `Ask TS >` and press Enter. Type `quit` or `exit` to stop.

**With Ollama** (concept extraction + explanation):
```bash
python goat_ts_complete.py --interactive --ollama-concepts --ollama-explain
```

---

## Usage

### Command-line options

| Option | Description |
|--------|-------------|
| `--text TEXT` | Input text for cognition (required unless `--interactive`) |
| `--seed WORD` | Seed concept for activation (default: `intelligence`) |
| `--ticks N` | Ticks per cycle (default: 20) |
| `--cycles N` | Number of cognition cycles (default: 3) |
| `--no-physics` | Disable graph layout physics |
| `--interactive` | Run interactive `Ask TS >` mode |
| `--ollama-concepts` | Use Ollama for concept extraction from prompt |
| `--ollama-explain` | Use Ollama for natural-language explanation |

### Programmatic use

```python
from goat_ts_complete import ts_reason, run_cognition, text_to_graph

# Full reasoning pipeline (returns nodes, edges, tension_trend, hypotheses, explanation)
nodes, edges, tension_trend, hypotheses, explanation = ts_reason(
    "Your prompt here",
    seed_label="intelligence",
    ticks=20,
    cycles=3,
    use_ollama_concepts=False,
    use_ollama_explain=False,
    enable_physics=True,
)

# Legacy batch API (returns nodes, edges, tension_trend)
nodes, edges, trend = run_cognition(
    "Input text",
    seed_label="intelligence",
    ticks=20,
    cycles=3,
    enable_physics=True,
)

# Build graph from text only (nodes + edges, with semantic edges)
nodes, edges = text_to_graph("Some sentences here.", add_semantic=True)
```

### Streamlit GUI

If Streamlit is installed and you run the script in a Streamlit context (e.g. `streamlit run goat_ts_complete.py`), a web UI is shown with text area, seed, sliders, and results.

---

## Architecture

The system is organized around:

1. **Graph representation** — Lists of `Node` and `Edge`; adjacency matrix built on demand.
2. **Text → graph** — Sentences → words (filtered) → unique nodes; edges from within-sentence distance, sentence boundaries, and semantic similarity (cached embeddings).
3. **Activation** — Matrix spread with decay, threshold, seed reinforcement, mass scaling, and reinforcement from previous activation.
4. **Memory** — Each node has state (ACTIVE / DORMANT / DEEP) and `dormant_ticks`; transitions are threshold- and time-based.
5. **Layout** — Fruchterman–Reingold repulsion/attraction plus gravity; node mass influences clustering.
6. **Tension** — Geometric (ideal vs actual distance), activation–weak-link, and semantic-gap signals.
7. **Hypotheses** — Candidates from tension pairs, activation-weak pairs, and semantic gaps; each has source, target, similarity, tension, confidence, and kind.
8. **Pipeline** — `ts_reason` wires: concept extraction → graph merge/init → activation cycles → tension → hypotheses → explanation (template or Ollama).

---

## Subsystems

### Node representation

- **Node**: `node_id`, `label`, `mass`, `activation`, `state` (ACTIVE/DORMANT/DEEP), `position` [x, y, z], `last_activation`, `dormant_ticks`.
- Mass defaults to 1.0; used in layout (gravity) and in activation scaling.

### Edge representation

- **Edge**: `src_id`, `dst_id`, `weight`, `directed`.
- Edges come from: co-occurrence (distance 1–5 in sentence), sentence boundaries, hardcoded concept pairs, and **semantic similarity** (embedding cosine ≥ threshold).

### Semantic embeddings

- **Local fallback**: 64-dimensional vector from character trigrams (deterministic, no external API).
- **Optional Ollama**: `/api/embeddings` (e.g. `nomic-embed-text`) when available; results cached.
- **Cosine similarity** between node labels drives automatic **semantic edges** and **semantic-gap** tension.

### Activation dynamics

- **Spreading**: `act_new = adj @ act * (1 - decay)` plus noise, threshold, seed clamp, mass influence, and reinforcement from `last_activation`.
- **Memory tick**: Activation decays; state becomes ACTIVE above threshold, DORMANT then DEEP after several ticks below threshold.

### Graph physics

- **Fruchterman–Reingold**: Repulsion and edge-based attraction; cooling over iterations.
- **Gravity**: Force ∝ mass₁ × mass₂ / distance² so high-mass nodes pull together into clusters.

### Tension detection

- **Geometric**: For each edge, ideal length ∝ 1/weight; tension from (actual − ideal)².
- **Activation-weak**: Pairs where both nodes have high activation but edge weight is low or missing.
- **Semantic gaps**: Pairs with high embedding similarity but no edge.

### Hypothesis generation

- **Hypothesis** dataclass: `source_node`, `target_node`, `similarity_score`, `tension_score`, `confidence`, `kind` (e.g. `semantic_gap`, `tension_layout`, `activation_weak`, `cluster_conflict`).
- Engine collects candidates from the three tension sources, deduplicates, scores confidence, returns top N.

---

## Optional Ollama Integration

- **Default**: No Ollama; concept extraction is regex-based from text; embeddings are local (trigram); explanation is a short template.
- **With `--ollama-concepts`**: Concept extraction uses Ollama generate (e.g. “list main concepts”) with fallback to text extraction.
- **With `--ollama-explain`**: Explanation is generated by Ollama from top concepts, tension, and hypotheses.
- **Embeddings**: If you call the embedding path with Ollama enabled, the code can use Ollama’s embedding API for node labels (and cache them); otherwise it uses the local trigram embedding.

Ollama is used **only** for:

- Concept extraction (optional)
- Explanation generation (optional)
- Optional embedding of labels (when wired)

**Reasoning (activation, tension, hypotheses) is always performed by the TS graph engine.**

---

## Performance & Constraints

- **MAX_NODES** (default 400): Prevents graph explosion.
- **Pruning**: Nodes below an activation threshold can be pruned between cycles.
- **Embedding cache**: Label → vector cached in memory to avoid recomputation.
- **Single file**: No stub or placeholder functions; all logic is implemented.
- Targeted to run on **low-end laptops** without GPU.

---

## Output Files

- **goat_ts_graph.json** — Current graph (nodes with id, label, mass, activation, state, position; edges with src, dst, weight). Written after each run; can be loaded on next run to continue reasoning.
- **goat_ts_layout.png** — Concept map plot (nodes by position, colored by activation, edges drawn). Produced when using the legacy `run_cognition` path (which calls `save_layout_plot`).

---

## License

This project is provided as-is for research and experimentation. See repository for any license file added later.

---

## Repository

**GitHub**: [github.com/BoggersTheFish/BOG-TS](https://github.com/BoggersTheFish/BOG-TS)
