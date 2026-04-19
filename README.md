# Adaptive RAG Inference System

Local RAG pipeline that optimizes itself at runtime.
No API keys. Runs on your machine using Ollama.

![Architecture](architecture.png)

---

## Stack
Python 3.11, FAISS, BM25, Ollama (llama3.2:1b), sentence-transformers

---

## What it does
Takes a question, figures out how complex it is, searches a 
document using both meaning-based and keyword search, reranks 
the results, and generates an answer. Tracks latency after 
every query and adjusts search depth automatically.

---

## Progress

### Day 1
Built basic pipeline — chunking, hybrid retrieval, adaptive K, feedback tracker.

| P50 | P95 | Quality |
|-----|-----|---------|
| 957ms | 2126ms | 0.532 |

Problem: only 3 chunks, retrieval order was approximate.
Fix: add reranker, use real PDF.

### Day 2
Added cross-encoder reranker. Switched to real RAG paper PDF (62 chunks).

| P50 | P95 | Quality |
|-----|-----|---------|
| 1579ms | 2705ms | 0.631 |

Quality improved. LLM is 85% of total time — retrieval is not the bottleneck.
Fix: add cache, build before/after benchmark.

---

## How to run
```bash
source venv311/bin/activate
ollama serve  # separate terminal
python3 main.py
```
