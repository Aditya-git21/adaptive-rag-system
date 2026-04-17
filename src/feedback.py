import time
from collections import deque
from typing import Optional

class FeedbackTracker:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.latency_window = deque(maxlen=window_size)
        self.quality_window = deque(maxlen=window_size)
        self.current_k = 4
        self.current_alpha = 0.5
        self.history = []

    def record(self, latency_ms: float, answer: str, query: str):
        self.latency_window.append(latency_ms)
        quality = self._score_quality(answer, query)
        self.quality_window.append(quality)
        self.history.append({
            "latency_ms": round(latency_ms, 2),
            "quality": round(quality, 2),
            "k": self.current_k,
            "alpha": self.current_alpha,
        })
        self._adjust()

    def _score_quality(self, answer: str, query: str) -> float:
        if not answer or len(answer.strip()) < 10:
            return 0.0
        word_count = len(answer.split())
        if word_count < 5:
            return 0.2
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        length_score = min(word_count / 50, 1.0)
        return round((overlap + length_score) / 2, 3)

    def _adjust(self):
        if len(self.latency_window) < 5:
            return

        avg_latency = sum(self.latency_window) / len(self.latency_window)
        avg_quality = sum(self.quality_window) / len(self.quality_window)

        # Latency too high → reduce K
        if avg_latency > 800 and self.current_k > 2:
            self.current_k -= 1
            print(f"  [feedback] High latency ({avg_latency:.0f}ms) → K reduced to {self.current_k}")

        # Quality too low → increase K
        elif avg_quality < 0.3 and self.current_k < 8:
            self.current_k += 1
            print(f"  [feedback] Low quality ({avg_quality:.2f}) → K increased to {self.current_k}")

        # Quality low + latency ok → shift alpha toward vector
        if avg_quality < 0.3 and avg_latency < 500:
            self.current_alpha = min(self.current_alpha + 0.1, 0.9)
            print(f"  [feedback] Low quality → alpha increased to {self.current_alpha:.1f}")

    def get_stats(self) -> dict:
        if not self.latency_window:
            return {}
        lats = sorted(self.latency_window)
        n = len(lats)
        return {
            "queries_tracked": len(self.history),
            "avg_latency_ms": round(sum(lats) / n, 2),
            "p50_ms": round(lats[int(n * 0.50)], 2),
            "p95_ms": round(lats[min(int(n * 0.95), n-1)], 2),
            "avg_quality": round(sum(self.quality_window) / n, 3),
            "current_k": self.current_k,
            "current_alpha": self.current_alpha,
        }

    def report(self):
        stats = self.get_stats()
        print("\n--- Feedback Tracker Report ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
