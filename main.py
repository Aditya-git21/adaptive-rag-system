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
    prompt = f"""Read the context and answer the question directly. 
Give the specific fact or number from the context

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

def ask_base_llm(question):
    prompt = f"""Answer this question from your own knowledge:

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

def run_query(question, retriever, tracker, cache):
    cached = cache.get(question)
    if cached:
        return cached, 0, 0, 0

    complexity = analyze_query(question)
    k = select_k(complexity, latency_ms=tracker.get_stats().get("p95_ms", 0))
    alpha = select_alpha(question)

    t0 = time.perf_counter()
    retrieved = retriever.retrieve_hybrid(question, k=k, alpha=alpha)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    reranked = rerank(question, retrieved, top_n=min(3, len(retrieved)))
    rerank_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    answer = ask_llm(reranked, question)
    llm_ms = (time.perf_counter() - t2) * 1000

    total_ms = retrieval_ms + rerank_ms + llm_ms
    tracker.record(total_ms, answer, question)
    cache.set(question, answer)

    return answer, retrieval_ms, rerank_ms, llm_ms

if __name__ == "__main__":
    import sys
    
    chunks = load_and_chunk("data/met_syllabus.pdf")
    retriever = HybridRetriever(chunks)
    tracker = FeedbackTracker(window_size=20)
    cache = QueryCache()

    # demo queries — specific facts only in this PDF
    queries = [
        "What is the test duration for MET 2026?",
        "What is the fee for second attempt in MET?",
        "What is the marking scheme for MET 2026?",
        "What is the minimum aggregate for ME program?",
        "How many questions are there in MET 2026?",
        ""What is the additional marks given to GATE qualified candidates in MET?",
    ]

    # interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\nInteractive mode. Type your question. Press Ctrl+C to exit.\n")
        while True:
            try:
                question = input("Your question: ").strip()
                if not question:
                    continue

                print("\n--- Base LLM (no document) ---")
                base_answer = ask_base_llm(question)
                print(f"{base_answer[:300]}")

                print("\n--- RAG (from MET syllabus PDF) ---")
                rag_answer, r_ms, rr_ms, l_ms = run_query(
                    question, retriever, tracker, cache
                )
                print(f"{rag_answer[:300]}")
                print(f"\n[retrieval={r_ms:.0f}ms rerank={rr_ms:.0f}ms llm={l_ms:.0f}ms]")
                print("-"*50)

            except KeyboardInterrupt:
                print("\nExiting.")
                break

    # demo mode — run preset queries
    else:
        print("\n" + "="*60)
        print("BASE LLM vs RAG — MET 2026 SYLLABUS")
        print("="*60)
        print("Same question. No context vs document-grounded answer.\n")

        for question in queries:
            print(f"\nQ: {question}")
            print("-"*60)

            base = ask_base_llm(question)
            print(f"BASE LLM : {base[:250]}")

            rag_answer, r_ms, rr_ms, l_ms = run_query(
                question, retriever, tracker, cache
            )
            print(f"RAG      : {rag_answer[:250]}")
            print(f"[retrieval={r_ms:.0f}ms rerank={rr_ms:.0f}ms llm={l_ms:.0f}ms]")

        tracker.report()
        cache.stats()
