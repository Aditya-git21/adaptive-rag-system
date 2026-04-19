import time

class QueryCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def get(self, query):
        key = query.lower().strip()
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def set(self, query, answer):
        key = query.lower().strip()
        self.store[key] = answer

    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total > 0 else 0
        print(f"\n--- Cache Stats ---")
        print(f"  hits: {self.hits}")
        print(f"  misses: {self.misses}")
        print(f"  hit rate: {rate:.0f}%")
