"""Tests use a tiny deterministic bag-of-words embedder so they run instantly, offline."""
import hashlib

import numpy as np

from src.chunking import chunk_documents
from src.vectorstore import VectorStore

DIM = 128


def fake_embed(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            word = word.strip(".,?!()")
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def test_chunks_respect_size_and_keep_metadata():
    record = {"source": "a.txt", "page": 1, "text": ("word " * 400).strip()}
    chunks = chunk_documents([record], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 200 for c in chunks)
    assert all(c["source"] == "a.txt" and c["page"] == 1 for c in chunks)


def test_search_returns_most_relevant_chunk():
    chunks = [
        {"chunk_id": 0, "text": "FAISS is a library for fast vector similarity search", "source": "x", "page": 1},
        {"chunk_id": 1, "text": "Pandas provides dataframes for tabular data analysis", "source": "x", "page": 1},
        {"chunk_id": 2, "text": "The monsoon season brings heavy rain to Kolkata", "source": "x", "page": 1},
    ]
    store = VectorStore(fake_embed)
    store.add(chunks)
    hits = store.search("vector similarity search library", k=2)
    assert hits[0]["chunk_id"] == 0
    assert hits[0]["score"] >= hits[1]["score"]


def test_save_and_load_roundtrip(tmp_path):
    chunks = [{"chunk_id": 0, "text": "hello vector world", "source": "x", "page": 1}]
    store = VectorStore(fake_embed)
    store.add(chunks)
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, fake_embed)
    assert loaded.search("hello vector world", k=1)[0]["chunk_id"] == 0
