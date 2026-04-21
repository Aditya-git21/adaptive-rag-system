import time
import numpy as np
import requests
from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever
from src.adaptive import analyze_query, select_k, select_alpha
from src.feedback import FeedbackTracker
from src.reranker import rerank
from src.cache import QueryCache
from src.decompose import needs_decomposition, decompose

def ask_llm(context_chunks, question):
    context = "\n".join(f"- {c[:300]}" for c in context_chunks)
    prompt = f"""Use the context below to answer the question. Be direct and concise.

Context:
{context}

Question: {question}

Answer:"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=60
        )
        return r.json()["response"].strip()
    except Exception as e:
        return f"LLM error: {e}"

def run_query(query, retriever, tracker, cache):
    cached = cache.get(query)
    if cached:
        print(f"  CACHE HIT — 0ms")
        return cached

    complexity = analyze_query(query)
    k = select_k(complexity, latency_ms=tracker.get_stats().get("p95_ms", 0))
    alpha = select_alpha(query)

    t0 = time.perf_counter()
    retrieved = retriever.retrieve_hybrid(query, k=k, alpha=alpha)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    reranked = rerank(query, retrieved, top_n=min(3, len(retrieved)))
    rerank_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    answer = ask_llm(reranked, query)
    llm_ms = (time.perf_counter() - t2) * 1000

    total_ms = retrieval_ms + rerank_ms + llm_ms
    tracker.record(total_ms, answer, query)
    cache.set(query, answer)

    print(f"  k={k} alpha={alpha} | "
          f"retrieval={retrieval_ms:.0f}ms "
          f"rerank={rerank_ms:.0f}ms "
          f"llm={llm_ms:.0f}ms")
    return answer

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

        latencies.append(retrieval_ms + rerank_ms)
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
    tracker = FeedbackTracker(window_size=20)
    cache = QueryCache()

    # normal queries
    normal_queries = [
        "What is inheritance?",
        "What is polymorphism?",
        "What is a constructor?",
    ]

    # multi-part queries that need decomposition
    multi_queries = [
        "What is inheritance and what is polymorphism?",
        "What is abstraction and also explain encapsulation?",
        "What is a constructor and how is it different from a method?",
    ]

    print("\n" + "="*55)
    print("NORMAL QUERIES")
    print("="*55)

    for query in normal_queries:
        print(f"\nQ: {query}")
        answer = run_query(query, retriever, tracker, cache)
        print(f"  A: {answer[:200]}")

    print("\n" + "="*55)
    print("MULTI-PART QUERIES WITH DECOMPOSITION")
    print("="*55)

    for query in multi_queries:
        print(f"\nOriginal Q: {query}")

        if needs_decomposition(query):
            sub_queries = decompose(query)
            print(f"  Split into {len(sub_queries)} sub-queries: {sub_queries}")

            answers = []
            for sq in sub_queries:
                print(f"\n  Sub-query: '{sq}'")
                ans = run_query(sq, retriever, tracker, cache)
                answers.append(ans)

            merged = "\n\n".join(
                f"[{sq}]\n{ans}" 
                for sq, ans in zip(sub_queries, answers)
            )
            print(f"\n  MERGED ANSWER:")
            print(f"  {merged[:400]}")
        else:
            answer = run_query(query, retriever, tracker, cache)
            print(f"  A: {answer[:200]}")

    print("\n" + "="*55)
    print("BENCHMARK — Fixed K=3 vs Adaptive K")
    print("="*55)

    bench_queries = [
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

    fixed = run_benchmark(retriever, bench_queries, use_adaptive=False)
    adaptive = run_benchmark(retriever, bench_queries, use_adaptive=True)

    print(f"{'Metric':<20} {'Fixed K=3':>12} {'Adaptive K':>12} {'Diff':>10}")
    print("-"*55)

    metrics = [
        ("P50 ms", fixed["p50"], adaptive["p50"]),
        ("P95 ms", fixed["p95"], adaptive["p95"]),
        ("Avg retrieval ms", fixed["avg_retrieval"], adaptive["avg_retrieval"]),
        ("Avg rerank ms", fixed["avg_rerank"], adaptive["avg_rerank"]),
        ("Avg total ms", fixed["avg_total"], adaptive["avg_total"]),
    ]

    for name, f, a in metrics:
        diff = round(a - f, 2)
        sign = "+" if diff > 0 else ""
        print(f"{name:<20} {f:>12} {a:>12} {sign+str(diff):>10}")

    tracker.report()
    cache.stats()
