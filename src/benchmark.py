import time
import numpy as np
from ingestion import load_and_chunk
from retriever import HybridRetriever
from adaptive import analyze_query, select_k, select_alpha
from reranker import rerank

def run_benchmark(retriever, queries, use_adaptive=True):
    latencies = []
    retrieval_times = []
    rerank_times = []

    for query in queries:
        if use_adaptive:
            complexity = analyze_query(query)
            k = select_k(complexity)
            alpha = select_alpha(query)
        else:
            k = 3
            alpha = 0.5

        t0 = time.perf_counter()
        retrieved = retriever.retrieve_hybrid(query, k=k, alpha=alpha)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        rerank(query, retrieved, top_n=min(3, len(retrieved)))
        rerank_ms = (time.perf_counter() - t1) * 1000

        total = retrieval_ms + rerank_ms
        latencies.append(total)
        retrieval_times.append(retrieval_ms)
        rerank_times.append(rerank_ms)

    return {
        "p50": round(np.percentile(latencies, 50), 2),
        "p95": round(np.percentile(latencies, 95), 2),
        "avg_retrieval": round(np.mean(retrieval_times), 2),
        "avg_rerank": round(np.mean(rerank_times), 2),
        "avg_total": round(np.mean(latencies), 2),
    }

if __name__ == "__main__":
    chunks = load_and_chunk("data/oops.pdf")
    retriever = HybridRetriever(chunks)

    queries = [
        "What is inheritance?",
        "What is polymorphism?",
        "What is a constructor?",
        "What is the difference between abstraction and encapsulation?",
        "What is the difference between overloading and overriding?",
        "What is inheritance in OOP?",
        "How does polymorphism work?",
        "What is encapsulation?",
        "What is abstraction?",
        "What is method overriding?",
    ]

    print("\nRunning fixed K=3 benchmark...")
    fixed = run_benchmark(retriever, queries, use_adaptive=False)

    print("Running adaptive K benchmark...")
    adaptive = run_benchmark(retriever, queries, use_adaptive=True)

    print("\n" + "="*55)
    print("BENCHMARK — Fixed K=3 vs Adaptive K")
    print("="*55)
    print(f"{'Metric':<20} {'Fixed K=3':>12} {'Adaptive K':>12} {'Diff':>10}")
    print("-"*55)

    metrics = [
        ("P50 latency ms", fixed["p50"], adaptive["p50"]),
        ("P95 latency ms", fixed["p95"], adaptive["p95"]),
        ("Avg retrieval ms", fixed["avg_retrieval"], adaptive["avg_retrieval"]),
        ("Avg rerank ms", fixed["avg_rerank"], adaptive["avg_rerank"]),
        ("Avg total ms", fixed["avg_total"], adaptive["avg_total"]),
    ]

    for name, f, a in metrics:
        diff = round(a - f, 2)
        sign = "+" if diff > 0 else ""
        print(f"{name:<20} {f:>12} {a:>12} {sign+str(diff):>10}")

    print("\nNote: negative diff = adaptive is faster")
    print("      positive diff = fixed is faster for that metric")
