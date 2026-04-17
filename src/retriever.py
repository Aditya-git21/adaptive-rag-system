import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

MODEL_NAME = "all-MiniLM-L6-v2"

class HybridRetriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)

        print("Building vector index...")
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        self.embeddings = np.array(embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 16)
        self.index.hnsw.efSearch = 50
        self.index.add(self.embeddings)

        print("Building BM25 index...")
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Retriever ready — {len(chunks)} chunks indexed")

    def retrieve_vector(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Returns list of (chunk_index, score)"""
        qvec = np.array(
            self.model.encode([query], normalize_embeddings=True)
        ).astype("float32")
        scores, indices = self.index.search(qvec, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def retrieve_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Returns list of (chunk_index, score)"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_k = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]

    def retrieve_hybrid(self, query: str, k: int, alpha: float = 0.5) -> List[str]:
        """
        Combine vector + BM25 scores.
        alpha = 1.0 → pure vector
        alpha = 0.0 → pure BM25
        alpha = 0.5 → equal weight (default)
        """
        # Get vector scores for all chunks
        qvec = np.array(
            self.model.encode([query], normalize_embeddings=True)
        ).astype("float32")
        vec_scores, vec_indices = self.index.search(qvec, len(self.chunks))
        vec_map = {int(idx): float(score) 
                   for idx, score in zip(vec_indices[0], vec_scores[0])}

        # Get BM25 scores for all chunks
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # Combine scores
        combined = {}
        for i in range(len(self.chunks)):
            v = vec_map.get(i, 0.0)
            b = float(bm25_scores[i]) / bm25_max
            combined[i] = alpha * v + (1 - alpha) * b

        top_k = sorted(combined, key=combined.get, reverse=True)[:k]
        return [self.chunks[i] for i in top_k]
