from typing import List

COMPLEX_KEYWORDS = [
    "compare", "difference", "versus", "vs", "explain",
    "how does", "why does", "what are all", "list",
    "summarize", "pros and cons", "tradeoff", "tradeoffs",
    "relationship", "between", "multiple", "steps"
]

def analyze_query(query: str) -> dict:
    q = query.lower().strip()
    word_count = len(q.split())
    is_complex = any(kw in q for kw in COMPLEX_KEYWORDS)
    has_multiple_parts = " and " in q or " also " in q or q.count("?") > 1

    if word_count <= 4 and not is_complex:
        score = 1
        label = "simple"
    elif word_count <= 10 and not is_complex and not has_multiple_parts:
        score = 2
        label = "medium"
    else:
        score = 3
        label = "complex"

    return {
        "query": query,
        "word_count": word_count,
        "complexity_score": score,
        "label": label,
        "is_complex_keyword": is_complex,
        "has_multiple_parts": has_multiple_parts,
    }

def select_k(complexity: dict, latency_ms: float = 0.0) -> int:
    score = complexity["complexity_score"]
    if latency_ms > 500:
        return 2
    if score == 1:
        return 2
    elif score == 2:
        return 4
    else:
        return 6

def select_alpha(query: str) -> float:
    q = query.lower()
    keyword_signals = ["define", "what is", "who is", "when", "where"]
    semantic_signals = ["explain", "how", "why", "compare", "difference"]
    keyword_score = sum(1 for s in keyword_signals if s in q)
    semantic_score = sum(1 for s in semantic_signals if s in q)
    if keyword_score > semantic_score:
        return 0.3
    elif semantic_score > keyword_score:
        return 0.7
    else:
        return 0.5
