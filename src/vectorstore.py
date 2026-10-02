"""A tiny FAISS-backed vector store (exact cosine-similarity search)."""
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn  # any callable: list[str] -> (n, dim) float32 array
        self.index = None
        self.chunks = []

    def add(self, chunks):
        vectors = np.asarray(self.embed_fn([c["text"] for c in chunks]), dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query, k=4):
        if self.index is None or not self.chunks:
            return []
        query_vec = np.asarray(self.embed_fn([query]), dtype="float32")
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder, embed_fn):
        folder = Path(folder)
        store = cls(embed_fn)
        store.index = faiss.read_index(str(folder / "index.faiss"))
        store.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return store
