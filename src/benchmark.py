import time
import sys
import numpy as np

sys.path.insert(0, '.')
from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever
from src.adaptive import analyze_query, select_k, select_alpha
from src.reranker import rerank
from src.cache import QueryCache


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
        "p99": round(np.percentile(latencies, 99), 2),
        "avg_retrieval": round(np.mean(retrieval_times), 2),
        "avg_rerank": round(np.mean(rerank_times), 2),
        "avg_total": round(np.mean(latencies), 2),
    }


def run_cache_benchmark(retriever, queries):
    cache = QueryCache()
    cold_times = []
    warm_times = []

    # cold pass — no cache
    for query in queries:
        t0 = time.perf_counter()
        retrieved = retriever.retrieve_hybrid(query, k=4, alpha=0.5)
        rerank(query, retrieved, top_n=3)
        total = (time.perf_counter() - t0) * 1000
        cold_times.append(total)
        cache.set(query, "cached")

    # warm pass — all hits
    for query in queries:
        t0 = time.perf_counter()
        cache.get(query)
        total = (time.perf_counter() - t0) * 1000
        warm_times.append(total)

    return {
        "cold_avg": round(np.mean(cold_times), 2),
        "cold_p95": round(np.percentile(cold_times, 95), 2),
        "warm_avg": round(np.mean(warm_times), 2),
        "warm_p95": round(np.percentile(warm_times, 95), 2),
        "speedup": round(np.mean(cold_times) / max(np.mean(warm_times), 0.01), 1),
    }


if __name__ == "__main__":
    print("Loading HDFC credit card PDF...")
    chunks = load_and_chunk("data/hdfc_credit_card.pdf")
    retriever = HybridRetriever(chunks)

    queries = [
        "What is the annual fee for HDFC credit card?",
        "What is the rate of interest or finance charge percentage per month?",
        "What is the minimum amount due calculation for HDFC credit card?",
        "What is the cash withdrawal fee or transaction fee?",
        "What is the interest free grace period on HDFC credit card?",
        "What is the late payment charge for HDFC credit card?",
        "What is the over limit charge for HDFC credit card?",
        "What is the foreign currency markup fee?",
        "How are reward points calculated?",
        "What happens if minimum amount due is not paid?",
    ]

    print("Running fixed K=3 benchmark...")
    fixed = run_benchmark(retriever, queries, use_adaptive=False)

    print("Running adaptive K benchmark...")
    adaptive = run_benchmark(retriever, queries, use_adaptive=True)

    print("Running cache benchmark...")
    cache_stats = run_cache_benchmark(retriever, queries)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — HDFC MITC Dataset (10 queries)")
    print("=" * 60)

    print(f"\n{'Metric':<22} {'Fixed K=3':>12} {'Adaptive K':>12} {'Diff':>10}")
    print("-" * 60)

    metrics = [
        ("P50 latency ms", fixed["p50"], adaptive["p50"]),
        ("P95 latency ms", fixed["p95"], adaptive["p95"]),
        ("P99 latency ms", fixed["p99"], adaptive["p99"]),
        ("Avg retrieval ms", fixed["avg_retrieval"], adaptive["avg_retrieval"]),
        ("Avg rerank ms", fixed["avg_rerank"], adaptive["avg_rerank"]),
        ("Avg total ms", fixed["avg_total"], adaptive["avg_total"]),
    ]

    for name, f, a in metrics:
        diff = round(a - f, 2)
        sign = "+" if diff > 0 else ""
        print(f"{name:<22} {f:>12} {a:>12} {sign+str(diff):>10}")

    print("\nnegative diff = adaptive is faster")

    print(f"\n{'Cache Benchmark':}")
    print("-" * 60)
    print(f"  Cold (no cache) avg : {cache_stats['cold_avg']} ms")
    print(f"  Cold (no cache) p95 : {cache_stats['cold_p95']} ms")
    print(f"  Warm (cache hit) avg: {cache_stats['warm_avg']} ms")
    print(f"  Warm (cache hit) p95: {cache_stats['warm_p95']} ms")
    print(f"  Speedup             : {cache_stats['speedup']}x faster with cache")
