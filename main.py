import time
import requests
from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever
from src.adaptive import analyze_query, select_k, select_alpha
from src.feedback import FeedbackTracker

def ask_llm(context_chunks, question):
    context = "\n".join(f"- {c[:300]}" for c in context_chunks)
    prompt = f"""Use the context below to answer the question. Be direct and concise.

Context:
{context}

Question: {question}

Answer:"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=60
        )
        return response.json()["response"].strip()
    except Exception as e:
        return f"LLM error: {e}"

if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.txt")
    retriever = HybridRetriever(chunks)
    tracker = FeedbackTracker(window_size=20)

    queries = [
        "What is BM25?",
        "How does caching work?",
        "What is P95 latency?",
        "Compare vector search and BM25 and explain the tradeoffs",
        "What is FAISS?",
        "How does the feedback loop adjust K?",
    ]

    print("\n" + "="*55)
    print("FULL ADAPTIVE RAG PIPELINE")
    print("="*55)

    for query in queries:
        complexity = analyze_query(query)
        k = select_k(complexity, latency_ms=tracker.get_stats().get("p95_ms", 0))
        alpha = select_alpha(query)

        t0 = time.perf_counter()
        chunks_retrieved = retriever.retrieve_hybrid(query, k=k, alpha=alpha)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        answer = ask_llm(chunks_retrieved, query)
        llm_ms = (time.perf_counter() - t1) * 1000

        total_ms = retrieval_ms + llm_ms
        tracker.record(total_ms, answer, query)

        print(f"\nQ: {query}")
        print(f"  K={k} alpha={alpha} | retrieval={retrieval_ms:.1f}ms LLM={llm_ms:.0f}ms")
        print(f"  A: {answer[:200]}")

    tracker.report()
