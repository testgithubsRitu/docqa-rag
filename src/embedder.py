"""Turn text into embedding vectors with a sentence-transformers model."""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, small and fast on CPU


class Embedder:
    """Callable: list[str] -> float32 array of shape (n, dim), L2-normalised.

    Because vectors are normalised, inner product == cosine similarity, which is what
    the FAISS IndexFlatIP index in vectorstore.py relies on.
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # imported lazily (slow import)

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")
