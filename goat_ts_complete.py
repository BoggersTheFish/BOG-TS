#!/usr/bin/env python3
"""
GOAT-TS (Thinking System) — Architecture Upgrade
Single-file graph-based cognition engine. Reasoning emerges from graph dynamics.

PHASE 1 — SYSTEM ANALYSIS (subsystems and interactions):
  • Node representation: Node(id, label, mass, activation, state, position, last_activation, dormant_ticks)
  • Edge representation: Edge(src_id, dst_id, weight, directed)
  • Graph: (nodes list, edges list); build_adjacency(nodes, edges) → n×n matrix
  • text_to_graph: sentences → words → unique nodes; edges from co-occurrence, sentence flow, semantic similarity
  • spreading_activation: adj @ act with decay, threshold, seed reinforcement; mass and edge weight influence
  • memory_tick: activation decay; state transitions ACTIVE → DORMANT → DEEP by thresholds and dormant_ticks
  • layout: Fruchterman–Reingold + gravitational attraction (mass); positions updated
  • tension: geometric (ideal vs actual distance), activation–connectivity mismatch, semantic gaps
  • hypothesis engine: high-act nodes, semantic-without-edge, tension pairs → Hypothesis(source, target, similarity, tension, confidence)
  • pipeline: ts_reason(prompt) → concept extraction → graph update → activation → tension → hypotheses → explanation

PHASE 2–3 — AGENT SYNTHESIS:
  Graph: semantic edges, mass in layout/activation, pruning and node cap.
  Cognitive: reinforcement, explicit ACTIVE→DORMANT→DEEP, tension from semantics and activation.
  Engineering: embedding cache, limit graph size, avoid redundant recomputation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import streamlit as st
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants and limits (performance / low-end)
# ---------------------------------------------------------------------------
MAX_NODES = 400
EMBEDDING_DIM = 64
SEMANTIC_EDGE_THRESHOLD = 0.62
WEAK_NODE_ACTIVATION_THRESHOLD = 0.008
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 8

# ---------------------------------------------------------------------------
# Node / Edge / Hypothesis
# ---------------------------------------------------------------------------
class MemoryState(StrEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    DEEP = "deep"


@dataclass(slots=True)
class Node:
    node_id: str
    label: str
    mass: float = 1.0
    activation: float = 0.0
    state: MemoryState = MemoryState.DORMANT
    position: list[float] = field(default_factory=lambda: [np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0.0])
    last_activation: float = 0.0
    dormant_ticks: int = 0


@dataclass(slots=True)
class Edge:
    src_id: str
    dst_id: str
    weight: float = 1.0
    directed: bool = True


@dataclass
class Hypothesis:
    source_node: Node
    target_node: Node
    similarity_score: float
    tension_score: float
    confidence: float
    kind: str  # "semantic_gap" | "tension_layout" | "activation_weak" | "cluster_conflict"


# ---------------------------------------------------------------------------
# Embedding cache and local semantic vectors
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, np.ndarray] = {}


def _char_ngram_vector(label: str, n: int = 3, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic local embedding from character n-grams. No external model."""
    label = label.lower().strip()
    if not label:
        return np.zeros(dim, dtype=np.float64)
    counts: dict[int, float] = defaultdict(float)
    for i in range(len(label) - n + 1):
        trig = label[i : i + n]
        h = int(hashlib.sha256(trig.encode()).hexdigest()[:8], 16) % dim
        counts[h] += 1.0
    vec = np.zeros(dim, dtype=np.float64)
    for h, c in counts.items():
        vec[h] = c
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


def get_embedding(label: str, use_ollama: bool = False) -> np.ndarray:
    """Return cached or computed embedding for a concept label."""
    if label in _embedding_cache:
        return _embedding_cache[label]
    if use_ollama:
        emb = _ollama_embedding(label)
        if emb is not None:
            _embedding_cache[label] = emb
            return emb
    vec = _char_ngram_vector(label)
    _embedding_cache[label] = vec
    return vec


def _ollama_embedding(label: str) -> np.ndarray | None:
    """Optional: get embedding via Ollama /embed endpoint if available."""
    try:
        req = Request(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            data=json.dumps({"model": "nomic-embed-text", "prompt": label}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            vec = np.array(data.get("embedding", []), dtype=np.float64)
            if len(vec) >= EMBEDDING_DIM:
                vec = vec[:EMBEDDING_DIM]
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    vec = vec / norm
                return vec
    except (URLError, OSError, ValueError, KeyError):
        pass
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for unit vectors; 0 if invalid."""
    if a.size != b.size or a.size == 0:
        return 0.0
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n < 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / n)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def add_semantic_edges(nodes: list[Node], edges: list[Edge], threshold: float = SEMANTIC_EDGE_THRESHOLD, use_ollama: bool = False) -> list[Edge]:
    """Add edges between nodes whose label embeddings are above similarity threshold."""
    existing = set()
    for e in edges:
        key = (min(e.src_id, e.dst_id), max(e.src_id, e.dst_id))
        existing.add(key)
    id_to_node = {n.node_id: n for n in nodes}
    embeddings = {n.node_id: get_embedding(n.label, use_ollama) for n in nodes}
    added: list[Edge] = list(edges)
    for i, a in enumerate(nodes):
        va = embeddings[a.node_id]
        for j, b in enumerate(nodes):
            if i >= j:
                continue
            key = (min(a.node_id, b.node_id), max(a.node_id, b.node_id))
            if key in existing:
                continue
            sim = cosine_similarity(va, embeddings[b.node_id])
            if sim >= threshold:
                w = 0.3 + 0.5 * sim
                added.append(Edge(src_id=a.node_id, dst_id=b.node_id, weight=min(2.0, w)))
                existing.add(key)
    return added


# ---------------------------------------------------------------------------
# Concept extraction (regex or optional Ollama)
# ---------------------------------------------------------------------------
STOP_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being"}


def extract_concepts_text(text: str) -> list[str]:
    """Extract concept tokens from raw text (no LLM)."""
    sentences = re.split(r"[.!?]+", text)
    words: list[str] = []
    for s in sentences:
        s = s.strip()
        for w in re.findall(r"[a-zA-Z]+", s.lower()):
            w = w.strip(".,!?()[]")
            if len(w) > 2 and w not in STOP_WORDS:
                words.append(w)
    return list(dict.fromkeys(words))


def extract_concepts_ollama(prompt: str) -> list[str]:
    """Optional: extract concepts using Ollama. Falls back to text extraction on failure."""
    try:
        req = Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps({
                "model": "llama3.2",
                "prompt": f"List the main concepts or keywords in this text, one per line. Only output the words, nothing else.\n\nText: {prompt[:800]}",
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            reply = data.get("response", "")
        concepts = [w.strip().lower() for line in reply.strip().splitlines() for w in line.split() if len(w.strip()) > 2 and w.strip().lower() not in STOP_WORDS]
        return list(dict.fromkeys(concepts))[:80] if concepts else extract_concepts_text(prompt)
    except (URLError, OSError, ValueError, KeyError):
        return extract_concepts_text(prompt)


# ---------------------------------------------------------------------------
# Text → graph (core) then semantic enrichment
# ---------------------------------------------------------------------------
def text_to_graph(text: str, add_semantic: bool = True, use_ollama_embed: bool = False) -> tuple[list[Node], list[Edge]]:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    all_words: list[str] = []
    for sent in sentences:
        words = [w.strip(".,!?()[]") for w in sent.lower().split() if len(w.strip(".,!?()[]")) > 2 and w.lower() not in STOP_WORDS]
        all_words.extend(words)
    unique = list(dict.fromkeys(all_words))
    if len(unique) > MAX_NODES:
        unique = unique[:MAX_NODES]
    nodes = [Node(node_id=str(uuid.uuid4()), label=w) for w in unique]
    id_map = {node.label: node.node_id for node in nodes}
    edges: list[Edge] = []
    for sent in sentences:
        words = [w.strip(".,!?()[]") for w in sent.lower().split() if len(w.strip(".,!?()[]")) > 2 and w.lower() not in STOP_WORDS]
        for dist in range(1, 6):
            for i in range(len(words) - dist):
                if words[i] not in id_map or words[i + dist] not in id_map:
                    continue
                src = id_map[words[i]]
                dst = id_map[words[i + dist]]
                weight = 1.0 / (dist ** 1.10)
                edges.append(Edge(src, dst, weight))
    for i in range(len(sentences) - 1):
        if sentences[i] and sentences[i + 1]:
            last = [w.strip(".,!?()[]") for w in sentences[i].lower().split() if len(w.strip(".,!?()[]")) > 2 and w.lower() not in STOP_WORDS]
            first = [w.strip(".,!?()[]") for w in sentences[i + 1].lower().split() if len(w.strip(".,!?()[]")) > 2 and w.lower() not in STOP_WORDS]
            if last and first and last[-1] in id_map and first[0] in id_map:
                edges.append(Edge(id_map[last[-1]], id_map[first[0]], weight=0.42))
    for src_txt, dst_txt, w in [("left", "behind", 1.65), ("must", "adapt", 1.65), ("artificial", "intelligence", 1.65), ("will", "change", 1.4)]:
        if src_txt in id_map and dst_txt in id_map:
            edges.append(Edge(id_map[src_txt], id_map[dst_txt], w))
    for e in edges:
        e.weight = min(2.0, e.weight * 1.12)
    if add_semantic and len(nodes) >= 2:
        edges = add_semantic_edges(nodes, edges, SEMANTIC_EDGE_THRESHOLD, use_ollama_embed)
    return nodes, edges


# ---------------------------------------------------------------------------
# Graph update: merge new concepts from prompt into existing graph
# ---------------------------------------------------------------------------
def graph_merge_concepts(nodes: list[Node], edges: list[Edge], concepts: list[str], add_semantic: bool = True) -> tuple[list[Node], list[Edge]]:
    """Add new concept nodes and link them; cap total nodes."""
    id_map = {n.label.lower(): n for n in nodes}
    node_list = list(nodes)
    edge_list = list(edges)
    for c in concepts:
        if len(node_list) >= MAX_NODES:
            break
        if c.lower() in id_map:
            continue
        new_node = Node(node_id=str(uuid.uuid4()), label=c)
        node_list.append(new_node)
        id_map[c.lower()] = new_node
    if add_semantic and len(node_list) >= 2:
        edge_list = add_semantic_edges(node_list, edge_list, SEMANTIC_EDGE_THRESHOLD, False)
    return node_list, edge_list


# ---------------------------------------------------------------------------
# Adjacency and activation (mass- and weight-aware)
# ---------------------------------------------------------------------------
def build_adjacency(nodes: list[Node], edges: list[Edge]) -> np.ndarray:
    n = len(nodes)
    id2idx = {n.node_id: i for i, n in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        i = id2idx.get(e.src_id)
        j = id2idx.get(e.dst_id)
        if i is not None and j is not None:
            adj[i, j] = max(adj[i, j], e.weight)
            adj[j, i] = max(adj[j, i], e.weight * 0.22)
    return adj


def spreading_activation(
    nodes: list[Node],
    edges: list[Edge],
    seed_ids: list[str],
    ticks_inner: int = 6,
    decay: float = 0.068,
    threshold: float = 0.018,
    mass_influence: float = 0.12,
    reinforcement: float = 0.15,
) -> list[Node]:
    """Spreading activation with decay, edge weight, mass influence, and reinforcement from repeated activation."""
    adj = build_adjacency(nodes, edges)
    n = len(nodes)
    id2idx = {n.node_id: i for i, n in enumerate(nodes)}
    act = np.zeros(n, dtype=np.float64)
    masses = np.array([nd.mass for nd in nodes], dtype=np.float64)
    for sid in seed_ids:
        if sid in id2idx:
            act[id2idx[sid]] = 1.0
    for _ in range(ticks_inner):
        act_new = adj @ act * (1.0 - decay)
        act_new += 0.007 * np.random.uniform(0.85, 1.15, n)
        if mass_influence != 0:
            act_new *= (1.0 + mass_influence * (masses - 1.0))
        act_new[act_new < threshold] = 0.0
        for sid in seed_ids:
            if sid in id2idx:
                act_new[id2idx[sid]] = max(act_new[id2idx[sid]], 0.72)
        for i, nd in enumerate(nodes):
            if nd.last_activation > 0 and act_new[i] > 0:
                act_new[i] = act_new[i] + reinforcement * nd.last_activation
        act = np.clip(act_new, 0.0, 3.2)
    out = []
    for i, node in enumerate(nodes):
        out.append(replace(node, activation=float(act[i]), last_activation=float(act[i])))
    return out


def memory_tick(
    nodes: list[Node],
    decay_rate: float = 0.954,
    active_th: float = 0.165,
    dormant_th: float = 0.032,
    ticks_to_deep: int = 5,
) -> list[Node]:
    """Memory state transitions: ACTIVE → DORMANT → DEEP (after ticks_to_deep below dormant_th)."""
    updated = []
    for n in nodes:
        act = max(0.0, n.activation * decay_rate)
        state = n.state
        dormant_ticks = n.dormant_ticks
        if act >= active_th:
            state = MemoryState.ACTIVE
            dormant_ticks = 0
        elif act < dormant_th:
            if state == MemoryState.ACTIVE:
                state = MemoryState.DORMANT
            dormant_ticks = n.dormant_ticks + 1
            if dormant_ticks >= ticks_to_deep:
                state = MemoryState.DEEP
        else:
            dormant_ticks = 0
        updated.append(replace(n, activation=act, state=state, dormant_ticks=dormant_ticks))
    return updated


# ---------------------------------------------------------------------------
# Layout: Fruchterman–Reingold + gravitational clustering (mass)
# ---------------------------------------------------------------------------
def fruchterman_reingold(nodes: list[Node], edges: list[Edge], iterations: int = 120, k_scale: float = 4.6, gravity_G: float = 0.04) -> list[Node]:
    n = len(nodes)
    if n < 2:
        return nodes
    pos = np.array([nd.position[:2] for nd in nodes], dtype=np.float64)
    mass = np.array([nd.mass for nd in nodes], dtype=np.float64)
    k = np.sqrt(1.0 / max(n, 1)) * k_scale
    id2idx = {nd.node_id: i for i, nd in enumerate(nodes)}
    for it in range(iterations):
        disp = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist_sq = np.dot(delta, delta) + 1e-10
                dist = math.sqrt(dist_sq)
                repulse = k * k / dist
                fvec = delta / dist * repulse
                disp[i] += fvec
                disp[j] -= fvec
                grav = gravity_G * mass[i] * mass[j] / dist_sq
                disp[i] -= delta / dist * grav
                disp[j] += delta / dist * grav
        for e in edges:
            i = id2idx.get(e.src_id)
            j = id2idx.get(e.dst_id)
            if i is None or j is None or i == j:
                continue
            delta = pos[j] - pos[i]
            dist_sq = np.dot(delta, delta) + 1e-10
            dist = math.sqrt(dist_sq)
            force = 1.38 * (dist * dist / k) - (k * k / dist)
            fvec = delta / dist * force * e.weight
            disp[i] += fvec
            disp[j] -= fvec * 0.58
        max_disp = 0.28 * k
        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp_norm[disp_norm == 0] = 1
        disp = np.where(disp_norm > max_disp, disp / disp_norm * max_disp, disp)
        cooling = 0.082 * (1.0 - 0.78 * it / max(iterations, 1))
        pos += disp * cooling
        pos = np.clip(pos, -22, 22)
        if np.max(np.linalg.norm(disp, axis=1)) < 2.8e-4:
            break
    if np.any(~np.isfinite(pos)):
        pos = np.array([nd.position[:2] for nd in nodes], dtype=np.float64)
    return [replace(nd, position=[float(pos[i, 0]), float(pos[i, 1]), 0.0]) for i, nd in enumerate(nodes)]


# ---------------------------------------------------------------------------
# Tension: geometric + activation–connectivity + semantic gaps
# ---------------------------------------------------------------------------
def compute_tension(nodes: list[Node], edges: list[Edge]) -> tuple[float, list[tuple[Node, Node, float, float, float]]]:
    pos_dict = {n.node_id: np.array(n.position[:2]) for n in nodes if np.all(np.isfinite(n.position[:2]))}
    total = 0.0
    pairs: list[tuple[Node, Node, float, float, float]] = []
    n_edges = max(1, len(edges))
    id_to_node = {n.node_id: n for n in nodes}
    for e in edges:
        if e.src_id not in pos_dict or e.dst_id not in pos_dict:
            continue
        a = pos_dict[e.src_id]
        b = pos_dict[e.dst_id]
        actual = float(np.linalg.norm(a - b))
        if not np.isfinite(actual):
            continue
        ideal = 1.35 / max(e.weight, 0.20)
        delta_sq = (actual - ideal) ** 2
        total += delta_sq
        src = id_to_node.get(e.src_id)
        dst = id_to_node.get(e.dst_id)
        if src and dst:
            pairs.append((src, dst, delta_sq, actual, ideal))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return total / n_edges, pairs[:20]


def tension_activation_weak(nodes: list[Node], edges: list[Edge], act_threshold: float = 0.25) -> list[tuple[Node, Node, float]]:
    """Pairs where at least one node is highly activated but edge weight is low or missing."""
    id2idx = {n.node_id: i for i, n in enumerate(nodes)}
    adj = build_adjacency(nodes, edges)
    n = len(nodes)
    weak_pairs: list[tuple[Node, Node, float]] = []
    for i in range(n):
        if nodes[i].activation < act_threshold:
            continue
        for j in range(n):
            if i == j:
                continue
            if nodes[j].activation < 0.05:
                continue
            w = max(adj[i, j], adj[j, i])
            if w < 0.2:
                weak_pairs.append((nodes[i], nodes[j], nodes[i].activation + nodes[j].activation))
    weak_pairs.sort(key=lambda x: -x[2])
    return weak_pairs[:15]


def tension_semantic_gaps(nodes: list[Node], edges: list[Edge], threshold: float = SEMANTIC_EDGE_THRESHOLD) -> list[tuple[Node, Node, float]]:
    """Pairs with high semantic similarity but no or weak edge."""
    edge_set = set()
    for e in edges:
        key = (min(e.src_id, e.dst_id), max(e.src_id, e.dst_id))
        edge_set.add(key)
    id_to_node = {n.node_id: n for n in nodes}
    embeddings = {n.node_id: get_embedding(n.label, False) for n in nodes}
    gaps: list[tuple[Node, Node, float]] = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j:
                continue
            key = (min(a.node_id, b.node_id), max(a.node_id, b.node_id))
            if key in edge_set:
                continue
            sim = cosine_similarity(embeddings[a.node_id], embeddings[b.node_id])
            if sim >= threshold:
                gaps.append((a, b, sim))
    gaps.sort(key=lambda x: -x[2])
    return gaps[:15]


# ---------------------------------------------------------------------------
# Hypothesis generation engine (structured)
# ---------------------------------------------------------------------------
def generate_hypotheses_engine(
    nodes: list[Node],
    edges: list[Edge],
    tension_pairs: list[tuple[Node, Node, float, float, float]],
    activation_weak: list[tuple[Node, Node, float]],
    semantic_gaps: list[tuple[Node, Node, float]],
) -> list[Hypothesis]:
    """Produce candidate hypotheses with source, target, similarity, tension, confidence."""
    hypotheses: list[Hypothesis] = []
    seen = set()
    def key(a: Node, b: Node) -> tuple[str, str]:
        return (min(a.node_id, b.node_id), max(a.node_id, b.node_id))
    for a, b, tension_sq, actual, ideal in tension_pairs:
        k = key(a, b)
        if k in seen:
            continue
        seen.add(k)
        diff = actual - ideal
        tension_score = math.sqrt(tension_sq)
        sim = cosine_similarity(get_embedding(a.label), get_embedding(b.label))
        confidence = 0.4 + 0.3 * min(1.0, tension_score / 3.0) + 0.2 * sim
        confidence = min(1.0, confidence)
        kind = "tension_layout"
        if abs(diff) > 1.35:
            kind = "cluster_conflict"
        hypotheses.append(Hypothesis(source_node=a, target_node=b, similarity_score=sim, tension_score=tension_score, confidence=confidence, kind=kind))
    for a, b, act_sum in activation_weak:
        k = key(a, b)
        if k in seen:
            continue
        seen.add(k)
        sim = cosine_similarity(get_embedding(a.label), get_embedding(b.label))
        tension_score = 0.5
        confidence = 0.3 + 0.2 * (act_sum / 2.0) + 0.2 * sim
        confidence = min(1.0, confidence)
        hypotheses.append(Hypothesis(source_node=a, target_node=b, similarity_score=sim, tension_score=tension_score, confidence=confidence, kind="activation_weak"))
    for a, b, sim in semantic_gaps:
        k = key(a, b)
        if k in seen:
            continue
        seen.add(k)
        tension_score = 0.4
        confidence = 0.35 + 0.4 * sim
        confidence = min(1.0, confidence)
        hypotheses.append(Hypothesis(source_node=a, target_node=b, similarity_score=sim, tension_score=tension_score, confidence=confidence, kind="semantic_gap"))
    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses[:25]


def generate_hypotheses_legacy(tension_pairs: list[tuple[Node, Node, float, float, float]]) -> list[str]:
    """Legacy string hypotheses for backward compatibility."""
    hyps = []
    for a, b, tension, actual, ideal in tension_pairs:
        if not np.isfinite(tension):
            continue
        diff = actual - ideal
        if abs(diff) > 1.35:
            dir_str = "CLOSER to" if diff > 0 else "FARTHER from"
            hyps.append(f"High tension ({tension:5.2f}): '{a.label}' should be {dir_str} '{b.label}' (actual {actual:5.2f}, ideal ~{ideal:4.1f})")
        else:
            hyps.append(f"Balanced ({tension:5.2f}): '{a.label}' – '{b.label}'")
    return hyps or ["No strong tension signals."]


# ---------------------------------------------------------------------------
# Prune weak nodes (performance)
# ---------------------------------------------------------------------------
def prune_weak_nodes(nodes: list[Node], edges: list[Edge], act_threshold: float = WEAK_NODE_ACTIVATION_THRESHOLD, min_degree: int = 0) -> tuple[list[Node], list[Edge]]:
    """Remove nodes with activation below threshold and optionally low degree; keep at least 2 nodes."""
    if len(nodes) <= 2:
        return nodes, edges
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e.src_id] += 1
        degree[e.dst_id] += 1
    keep_ids = set()
    for n in nodes:
        if n.activation >= act_threshold or degree[n.node_id] >= max(2, min_degree):
            keep_ids.add(n.node_id)
    if len(keep_ids) < 2:
        keep_ids = {nodes[0].node_id, nodes[1].node_id}
    new_nodes = [n for n in nodes if n.node_id in keep_ids]
    new_edges = [e for e in edges if e.src_id in keep_ids and e.dst_id in keep_ids]
    return new_nodes, new_edges


# ---------------------------------------------------------------------------
# TS Reasoning pipeline
# ---------------------------------------------------------------------------
def ts_reason(
    prompt: str,
    seed_label: str | None = None,
    ticks: int = 20,
    cycles: int = 3,
    use_ollama_concepts: bool = False,
    use_ollama_explain: bool = False,
    enable_physics: bool = True,
) -> tuple[list[Node], list[Edge], list[float], list[Hypothesis], str]:
    """
    Full TS reasoning: concept extraction → graph update → activation → tension → hypotheses → explanation.
    Returns (nodes, edges, tension_trend, hypotheses, explanation_string).
    """
    concepts = extract_concepts_ollama(prompt) if use_ollama_concepts else extract_concepts_text(prompt)
    loaded_nodes, loaded_edges = load_graph_json()
    if loaded_nodes is not None and len(loaded_nodes) > 0:
        nodes, edges = graph_merge_concepts(loaded_nodes, loaded_edges, concepts, add_semantic=True)
    else:
        nodes, edges = text_to_graph(prompt, add_semantic=True, use_ollama_embed=False)
    if len(nodes) > MAX_NODES:
        nodes, edges = prune_weak_nodes(nodes, edges, WEAK_NODE_ACTIVATION_THRESHOLD)
        nodes, edges = nodes[:MAX_NODES], [e for e in edges if e.src_id in {n.node_id for n in nodes} and e.dst_id in {n.node_id for n in nodes}]
    seed_ids = [n.node_id for n in nodes if seed_label and seed_label.lower() in n.label.lower()]
    if not seed_ids and nodes:
        seed_ids = [nodes[0].node_id]
    tension_trend: list[float] = []
    for cycle in range(1, cycles + 1):
        for _ in range(ticks):
            nodes = spreading_activation(nodes, edges, seed_ids, mass_influence=0.12, reinforcement=0.15)
            nodes = memory_tick(nodes, ticks_to_deep=5)
            if enable_physics and len(nodes) >= 2:
                nodes = fruchterman_reingold(nodes, edges, gravity_G=0.04)
            score, top_pairs = compute_tension(nodes, edges)
            tension_trend.append(score)
        nodes, edges = prune_weak_nodes(nodes, edges, WEAK_NODE_ACTIVATION_THRESHOLD)
    _, tension_pairs = compute_tension(nodes, edges)
    activation_weak = tension_activation_weak(nodes, edges, act_threshold=0.25)
    semantic_gaps = tension_semantic_gaps(nodes, edges, threshold=SEMANTIC_EDGE_THRESHOLD)
    hypotheses = generate_hypotheses_engine(nodes, edges, tension_pairs, activation_weak, semantic_gaps)
    explanation = generate_explanation(nodes, hypotheses, tension_trend, use_ollama=use_ollama_explain)
    save_graph_json(nodes, edges)
    return nodes, edges, tension_trend, hypotheses, explanation


def generate_explanation(nodes: list[Node], hypotheses: list[Hypothesis], tension_trend: list[float], use_ollama: bool = False) -> str:
    """Produce natural-language explanation; optionally use Ollama."""
    top = sorted([n for n in nodes if n.activation > 0.05], key=lambda x: -x.activation)[:8]
    top_labels = ", ".join(n.label for n in top)
    hyp_summary = "; ".join(f"{h.source_node.label}–{h.target_node.label}({h.kind})" for h in hypotheses[:5])
    trend_avg = sum(tension_trend) / len(tension_trend) if tension_trend else 0.0
    base = f"Top concepts: {top_labels}. Tension trend avg: {trend_avg:.2f}. Hypotheses: {hyp_summary}."
    if use_ollama:
        ollama_expl = _ollama_explain(top_labels, hyp_summary, trend_avg)
        if ollama_expl:
            return ollama_expl
    return base


def _ollama_explain(top_labels: str, hyp_summary: str, trend_avg: float) -> str | None:
    try:
        req = Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps({
                "model": "llama3.2",
                "prompt": f"In 2-3 sentences, explain what the thinking system inferred. Key concepts: {top_labels}. Tension: {trend_avg:.2f}. Relation hypotheses: {hyp_summary}.",
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            return (data.get("response") or "").strip() or None
    except (URLError, OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Persistence and plotting
# ---------------------------------------------------------------------------
def save_layout_plot(nodes: list[Node], edges: list[Edge], filename: str = "goat_ts_layout.png") -> None:
    valid_nodes = [n for n in nodes if np.all(np.isfinite(n.position[:2]))]
    if len(valid_nodes) < 2:
        return
    pos = np.array([n.position[:2] for n in valid_nodes])
    plt.figure(figsize=(13, 9))
    id2idx = {n.node_id: i for i, n in enumerate(valid_nodes)}
    for e in edges:
        i, j = id2idx.get(e.src_id), id2idx.get(e.dst_id)
        if i is None or j is None:
            continue
        alpha = 0.4 + 0.4 * min(e.weight, 1.0)
        plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="gray", alpha=alpha, lw=0.9 + e.weight * 2.1, zorder=1)
    acts = np.array([n.activation for n in valid_nodes])
    plt.scatter(pos[:, 0], pos[:, 1], c=acts, cmap="inferno", s=520 + acts * 1300, edgecolor="black", linewidth=1.4, zorder=2)
    for n, (x, y) in zip(valid_nodes, pos):
        plt.text(x, y, n.label, fontsize=11, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2.5))
    plt.title("GOAT-TS Concept Map + Activation")
    plt.colorbar(label="Activation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def save_graph_json(nodes: list[Node], edges: list[Edge], filename: str = "goat_ts_graph.json") -> None:
    graph_data = {
        "nodes": [
            {"id": n.node_id, "label": n.label, "mass": n.mass, "activation": n.activation, "state": n.state.value, "position": n.position}
            for n in nodes
        ],
        "edges": [{"src": e.src_id, "dst": e.dst_id, "weight": e.weight} for e in edges],
    }
    with open(filename, "w") as f:
        json.dump(graph_data, f, indent=2)


def load_graph_json(filename: str = "goat_ts_graph.json") -> tuple[list[Node] | None, list[Edge] | None]:
    if not os.path.exists(filename):
        return None, None
    with open(filename, "r") as f:
        data = json.load(f)
    nodes = []
    for nd in data["nodes"]:
        raw_pos = nd.get("position")
        if raw_pos and len(raw_pos) >= 2:
            pos = [float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]) if len(raw_pos) > 2 else 0.0]
        else:
            pos = [np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0.0]
        node = Node(
            node_id=nd["id"],
            label=nd["label"],
            mass=float(nd.get("mass", 1.0)),
            activation=float(nd.get("activation", 0.0)),
            state=MemoryState(nd.get("state", "dormant")),
            position=pos,
        )
        nodes.append(node)
    edges = [Edge(src_id=ed["src"], dst_id=ed["dst"], weight=float(ed.get("weight", 1.0))) for ed in data["edges"]]
    return nodes, edges


# ---------------------------------------------------------------------------
# Legacy run_cognition (preserves original API)
# ---------------------------------------------------------------------------
def run_cognition(text: str, seed_label: str, ticks: int = 20, enable_physics: bool = True, cycles: int = 3) -> tuple[list[Node], list[Edge], list[float]]:
    loaded_nodes, loaded_edges = load_graph_json()
    if loaded_nodes is not None:
        nodes, edges = loaded_nodes, loaded_edges
        concepts = extract_concepts_text(text)
        nodes, edges = graph_merge_concepts(nodes, edges, concepts, add_semantic=True)
    else:
        nodes, edges = text_to_graph(text, add_semantic=True, use_ollama_embed=False)
    if len(nodes) > MAX_NODES:
        nodes, edges = prune_weak_nodes(nodes, edges, WEAK_NODE_ACTIVATION_THRESHOLD)
        nodes = nodes[:MAX_NODES]
        edges = [e for e in edges if e.src_id in {n.node_id for n in nodes} and e.dst_id in {n.node_id for n in nodes}]
    seed_ids = [n.node_id for n in nodes if seed_label.lower() in n.label.lower()]
    if not seed_ids and nodes:
        seed_ids = [nodes[0].node_id]
    tension_trend = []
    for cycle in range(1, cycles + 1):
        for _ in range(ticks):
            nodes = spreading_activation(nodes, edges, seed_ids, mass_influence=0.12, reinforcement=0.15)
            nodes = memory_tick(nodes, ticks_to_deep=5)
            if enable_physics and len(nodes) >= 2:
                nodes = fruchterman_reingold(nodes, edges, gravity_G=0.04)
            score, top_pairs = compute_tension(nodes, edges)
            tension_trend.append(score)
        for a, b, tension_sq, actual, ideal in top_pairs[:14]:
            diff = actual - ideal
            for e in edges:
                if (e.src_id == a.node_id and e.dst_id == b.node_id) or (e.src_id == b.node_id and e.dst_id == a.node_id):
                    if diff < -1.35:
                        e.weight = max(0.08, e.weight - 0.18)
                    elif diff > 1.35:
                        e.weight = min(2.8, e.weight + 0.4)
                    break
        dormant_low = [n for n in nodes if n.activation < 0.12]
        for idx, n in enumerate(dormant_low[:4]):
            i = nodes.index(n)
            nodes[i] = replace(nodes[i], activation=max(nodes[i].activation, 0.48))
        if not any(n.label == "minimize_tension" for n in nodes):
            nodes.append(Node(node_id=str(uuid.uuid4()), label="minimize_tension", activation=1.8))
    save_layout_plot(nodes, edges)
    save_graph_json(nodes, edges)
    return nodes, edges, tension_trend


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
def interactive_cli(use_ollama_concepts: bool = False, use_ollama_explain: bool = False) -> None:
    """Ask TS > prompt runs full cognition pipeline; display top nodes, tension, hypotheses, explanation."""
    print("GOAT-TS Interactive Mode. Type a prompt and press Enter. 'quit' or 'exit' to stop.\n")
    while True:
        try:
            prompt = input("Ask TS > ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        print("Running TS reasoning...")
        nodes, edges, tension_trend, hypotheses, explanation = ts_reason(
            prompt, seed_label=None, ticks=18, cycles=2, use_ollama_concepts=use_ollama_concepts, use_ollama_explain=use_ollama_explain, enable_physics=True
        )
        print("\n--- Top activated nodes ---")
        for n in sorted([n for n in nodes if n.activation > 0.05], key=lambda x: -x.activation)[:12]:
            print(f"  {n.label}: act={n.activation:.3f} state={n.state}")
        print("\n--- High-tension pairs (layout) ---")
        _, tension_pairs = compute_tension(nodes, edges)
        for a, b, tsq, actual, ideal in tension_pairs[:6]:
            print(f"  {a.label} - {b.label} (tension^2={tsq:.2f})")
        print("\n--- Generated hypotheses ---")
        for h in hypotheses[:8]:
            print(f"  [{h.kind}] {h.source_node.label} -> {h.target_node.label} sim={h.similarity_score:.2f} tension={h.tension_score:.2f} conf={h.confidence:.2f}")
        print("\n--- Explanation ---")
        print("  ", explanation)
        print()


# ---------------------------------------------------------------------------
# CLI entrypoints
# ---------------------------------------------------------------------------
def cli_mode() -> None:
    parser = argparse.ArgumentParser(description="GOAT-TS Thinking System")
    parser.add_argument("--text", type=str, required=False, help="Input text for cognition")
    parser.add_argument("--seed", type=str, default="intelligence")
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Run interactive Ask TS > mode")
    parser.add_argument("--ollama-concepts", action="store_true", help="Use Ollama for concept extraction")
    parser.add_argument("--ollama-explain", action="store_true", help="Use Ollama for explanation")
    args = parser.parse_args()
    if args.interactive:
        interactive_cli(use_ollama_concepts=args.ollama_concepts, use_ollama_explain=args.ollama_explain)
        return
    if not args.text:
        print("Either provide --text or use --interactive.", file=sys.stderr)
        sys.exit(1)
    _, _, trend = run_cognition(args.text, args.seed, args.ticks, not args.no_physics, args.cycles)
    print("\n=== Tension trend ===")
    for i, t in enumerate(trend[-30:], 1):
        print(f"  Step {i:3d} -> Tension: {t:6.2f}")


def gui_mode() -> None:
    st.title("GOAT-TS Thinking System")
    text = st.text_area("Input text", height=150, value="Artificial intelligence will change everything. Humans must adapt or be left behind.")
    seed = st.text_input("Seed keyword", "intelligence")
    ticks = st.slider("Ticks per cycle", 10, 100, 20)
    cycles = st.slider("Number of cycles", 1, 10, 3)
    if st.button("Run Cognition"):
        with st.spinner("Running cognition cycles..."):
            nodes, edges, trend = run_cognition(text, seed, ticks, True, cycles)
        st.success("Done.")
        st.subheader("Tension Trend")
        st.line_chart(trend)
        st.subheader("Top Activated Concepts")
        for n in sorted([n for n in nodes if n.activation > 0.05], key=lambda x: -x.activation)[:12]:
            st.write(f"• {n.label} (act={n.activation:.3f}, {n.state})")


if __name__ == "__main__":
    if GUI_AVAILABLE and ("streamlit" in str(sys.argv) or os.getenv("PYTHON_SCRIPT", "").lower() == "streamlit"):
        gui_mode()
    else:
        cli_mode()
