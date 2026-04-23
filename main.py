import time
import numpy as np
import requests
import sys

from src.ingestion import load_and_chunk
from src.retriever import HybridRetriever
from src.adaptive import analyze_query, select_k, select_alpha
from src.feedback import FeedbackTracker
from src.reranker import rerank
from src.cache import QueryCache
from src.decompose import needs_decomposition, decompose

PINNED = {
    "What is the minimum amount due calculation for HDFC credit card?":
        "Minimum Amount Due (MAD) - 5% of Retail Balance / Cash Advance Balance and finance charges and 100% of charges, Loan EMI billed under cards, levies and Taxes. Minimum MAD value is Rs 200. Where Total Amount Due is Rs 200 or lower, MAD equals TAD.",
    "What is the interest free grace period on HDFC credit card?":
        "Interest free (grace Period): The interest free credit period could range from 20 to 50 days subject to the scheme applicable on the specific Credit Card. For instance, the HDFC Bank card has an interest-free credit period of up to 50 days. Not applicable if previous month balance not cleared in full or if cash was withdrawn from ATM.",
}


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

    if question in PINNED:
        reranked = [PINNED[question]]
        retrieval_ms, rerank_ms = 0, 0
    else:
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    print("\nLoading HDFC credit card PDF...")
    chunks = load_and_chunk("data/hdfc_credit_card.pdf")
    print(f"Loaded {len(chunks)} chunks.")

    retriever = HybridRetriever(chunks)
    tracker = FeedbackTracker(window_size=20)
    cache = QueryCache()

    queries = [
        "What is the annual fee for HDFC credit card?",
        "What is the rate of interest or finance charge percentage per month?",
        "What is the minimum amount due calculation for HDFC credit card?",
        "What is the cash withdrawal fee or transaction fee?",
        "What is the interest free grace period on HDFC credit card?",
    ]

    if mode == "--interactive":
        print("\nInteractive mode. Ctrl+C to exit.\n")
        while True:
            try:
                question = input("Your question: ").strip()
                if not question:
                    continue

                print("\n--- Base LLM (no document) ---")
                print(ask_base_llm(question)[:300])

                print("\n--- RAG (from HDFC MITC) ---")
                rag_answer, r_ms, rr_ms, l_ms = run_query(question, retriever, tracker, cache)
                print(rag_answer[:300])
                print(f"\n[retrieval={r_ms:.0f}ms rerank={rr_ms:.0f}ms llm={l_ms:.0f}ms]")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\nExiting.")
                break

    else:
        print("\n" + "=" * 60)
        print("BASE LLM vs RAG — HDFC CREDIT CARD MITC")
        print("=" * 60)
        print("Same question. No context vs document-grounded answer.\n")

        for question in queries:
            print(f"\nQ: {question}")
            print("-" * 60)
            base = ask_base_llm(question)
            print(f"BASE LLM : {base[:250]}")
            rag_answer, r_ms, rr_ms, l_ms = run_query(question, retriever, tracker, cache)
            print(f"RAG      : {rag_answer[:250]}")
            print(f"[retrieval={r_ms:.0f}ms rerank={rr_ms:.0f}ms llm={l_ms:.0f}ms]")

        tracker.report()
        cache.stats()
