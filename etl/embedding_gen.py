"""Embedding generator for FailureMode and Anomaly nodes.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) when available.
Falls back to deterministic mock embeddings so the pipeline can run
without GPU or the sentence-transformers package installed.

Creates HNSW vector indexes via the Samyama SDK and inserts per-node
embedding vectors.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class _MockEmbedder:
    """Deterministic pseudo-embeddings derived from text hash.

    Uses byte-level expansion (not struct.unpack) to avoid NaN/Inf floats
    that crash the HNSW index. Each dimension is derived from a hash byte
    combined with positional mixing, producing values in [0, 1].
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = []
            for i in range(self.dimensions):
                byte_idx = i % len(h)
                vec.append(((h[byte_idx] + i * 7) % 256) / 255.0)
            norm = max(sum(v * v for v in vec) ** 0.5, 1e-10)
            vec = [v / norm for v in vec]
            results.append(vec)
        return results


def _get_embedder(model_name: str) -> tuple[object, int, bool]:
    """Return (embedder, dimensions, is_real).

    Tries sentence-transformers first; falls back to mock.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        logger.info("Using sentence-transformers model '%s' (%d-dim)", model_name, dim)
        return model, dim, True
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; using mock embeddings. "
            "Install with: pip install sentence-transformers"
        )
        dim = 384
        return _MockEmbedder(dim), dim, False
    except Exception as e:
        logger.warning("Failed to load model '%s': %s — using mock embeddings", model_name, e)
        dim = 384
        return _MockEmbedder(dim), dim, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embeddings(
    client: SamyamaClient,
    graph: str = "industrial",
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, int]:
    """Generate and store vector embeddings for FailureMode and Anomaly nodes.

    1. Creates HNSW vector indexes for FailureMode.embedding and Anomaly.embedding.
    2. Fetches all FailureMode (and Anomaly) nodes from the graph.
    3. Encodes their description text into embedding vectors.
    4. Stores each vector via client.add_vector().

    Returns dict: {embedded_nodes, dimensions, model, is_real_model}.
    """
    embedder, dimensions, is_real = _get_embedder(model_name)

    # ── Create vector indexes ────────────────────────────────────
    try:
        client.create_vector_index("FailureMode", "embedding", dimensions, "cosine")
        logger.info("Created vector index FailureMode.embedding (%d-dim)", dimensions)
    except Exception as e:
        logger.debug("Vector index FailureMode.embedding may already exist: %s", e)

    try:
        client.create_vector_index("Anomaly", "embedding", dimensions, "cosine")
        logger.info("Created vector index Anomaly.embedding (%d-dim)", dimensions)
    except Exception as e:
        logger.debug("Vector index Anomaly.embedding may already exist: %s", e)

    embedded_count = 0

    # ── Embed FailureMode descriptions ───────────────────────────
    embedded_count += _embed_label(
        client, graph, embedder, "FailureMode", "embedding", "description"
    )

    # ── Embed Anomaly descriptions (if any exist) ────────────────
    embedded_count += _embed_label(
        client, graph, embedder, "Anomaly", "embedding", "description"
    )

    return {
        "embedded_nodes": embedded_count,
        "dimensions": dimensions,
        "model": model_name if is_real else f"mock-{dimensions}d",
        "is_real_model": is_real,
    }


def _embed_label(
    client: SamyamaClient,
    graph: str,
    embedder: object,
    label: str,
    vector_property: str,
    text_property: str,
) -> int:
    """Embed all nodes of a given label on their text_property.

    Returns count of nodes embedded.
    """
    cypher = f"MATCH (n:{label}) RETURN id(n), n.{text_property}"
    try:
        result = client.query_readonly(cypher, graph)
    except Exception as e:
        logger.warning("Could not query %s nodes: %s", label, e)
        return 0

    if not result.records:
        logger.info("No %s nodes found; skipping embedding", label)
        return 0

    # Collect (node_id, text) pairs
    pairs: list[tuple[int, str]] = []
    for record in result.records:
        node_id = record[0]
        text = record[1]
        if node_id is not None and text is not None:
            pairs.append((int(node_id), str(text)))

    if not pairs:
        return 0

    # Encode all texts in one batch
    texts = [text for _, text in pairs]
    vectors = embedder.encode(texts)

    # Store each vector
    count = 0
    for (node_id, _text), vec in zip(pairs, vectors):
        try:
            # Convert to list of float32
            vec_f32 = [float(v) for v in vec]
            client.add_vector(label, vector_property, node_id, vec_f32)
            count += 1
        except Exception as e:
            logger.debug("Failed to add vector for %s node %d: %s", label, node_id, e)

    logger.info("Embedded %d %s nodes", count, label)
    return count
