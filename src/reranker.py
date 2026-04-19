from sentence_transformers import CrossEncoder
from typing import List

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: List[str], top_n: int = None) -> List[str]:
    if not chunks:
        return chunks

    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs)

    scored = sorted(zip(scores, chunks), reverse=True)

    if top_n:
        scored = scored[:top_n]

    return [chunk for score, chunk in scored]
