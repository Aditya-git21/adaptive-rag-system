from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever
from src.adaptive import analyze_query, select_k, select_alpha

if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.txt")
    retriever = HybridRetriever(chunks)

    test_queries = [
        "What is BM25?",
        "How does caching work in RAG systems?",
        "Compare vector search and BM25 and explain the tradeoffs between them",
        "P95",
        "Explain the difference between HNSW and IVF and when to use each one",
    ]

    print("\n" + "="*55)
    print("ADAPTIVE DECISION LAYER TEST")
    print("="*55)

    for query in test_queries:
        complexity = analyze_query(query)
        k = select_k(complexity, latency_ms=0)
        alpha = select_alpha(query)

        results = retriever.retrieve_hybrid(query, k=k, alpha=alpha)

        print(f"\nQuery : '{query}'")
        print(f"  Label  : {complexity['label']} "
              f"(score={complexity['complexity_score']}, "
              f"words={complexity['word_count']})")
        print(f"  K      : {k}")
        print(f"  Alpha  : {alpha} "
              f"({'vector-leaning' if alpha > 0.5 else 'BM25-leaning' if alpha < 0.5 else 'balanced'})")
        print(f"  Got {len(results)} chunks")
