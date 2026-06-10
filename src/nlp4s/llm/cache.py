"""On-disk response cache for LLM calls.

LLM inference at the experiment scale we run (3 models x 2 conditions x
several few-shot configs x ~thousands of MHC eval rows) costs real money and
takes hours. Re-running after a prompt tweak should not re-charge for cells
that didn't change, so every ``LLMClient.complete`` call is keyed by
``(model_id, prompt, temperature, max_tokens)`` and the response is persisted
in a SQLite file.

Wrap any ``LLMClient`` in ``CachedLLMClient`` to opt in.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock

from nlp4s.llm.client import LLMClient


def _key(model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {"m": model_id, "p": prompt, "t": temperature, "n": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Tiny SQLite-backed key-value store for LLM completions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "  key TEXT PRIMARY KEY,"
            "  model TEXT NOT NULL,"
            "  response TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def put(self, key: str, model_id: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, response) "
                "VALUES (?, ?, ?)",
                (key, model_id, response),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedLLMClient(LLMClient):
    """Decorator that consults a ResponseCache before calling the inner client."""

    def __init__(self, inner: LLMClient, cache: ResponseCache, model_id: str) -> None:
        self._inner = inner
        self._cache = cache
        self.model_id = model_id
        self.hits = 0
        self.misses = 0

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        key = _key(self.model_id, prompt, temperature, max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        response = self._inner.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        self._cache.put(key, self.model_id, response)
        self.misses += 1
        return response
