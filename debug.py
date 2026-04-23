from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever

chunks = load_and_chunk("data/met_syllabus.pdf")
retriever = HybridRetriever(chunks)

questions = [
    "What is the test duration for MET 2026?",
    "What is the maximum marks in MET 2026?",
    "How many total questions are in MET 2026?",
]

for q in questions:
    print(f"\nQ: {q}")
    scores = retriever.bm25.get_scores(q.lower().split())
    import numpy as np
    top4 = np.argsort(scores)[::-1][:4]
    print("BM25 top chunks:")
    for idx in top4:
        print(f"  [{idx}] score={scores[idx]:.2f}: {chunks[idx][:200]}")
