"""
session_store.py
================
Server-side conversation memory for multi-turn detection.

When a request carries a `session_id`, the gate remembers the prompts seen in
that session and feeds them to the classifier as context on the next turn. This
is what catches a gradual (crescendo) jailbreak whose individual turns each look
harmless: the caller need only send the current turn.

Design / privacy:
  * IN-MEMORY and process-local by default -- nothing is written to disk.
  * BOUNDED: at most `max_turns` turns per session, `max_sessions` sessions
    (LRU eviction), and each session expires after `ttl_s` of inactivity.
  * It stores prompt TEXT because that is what the classifier needs as context.
    This is deliberate content retention, kept separate from the audit log
    (which stores only a hash). It is cleared with `forget()`/`clear()` and is
    lost on process restart.
  * Stored turns are DATA fed to our own classifier only -- never executed,
    never passed to the rules engine as instructions.
Disable entirely with PIGATE_SESSION_MEMORY=0 (the gate then uses only the
history a caller passes explicitly).
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict


class SessionStore:
    def __init__(self, max_turns: int = 10, ttl_s: int = 1800, max_sessions: int = 1000,
                 enabled: bool | None = None):
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        if enabled is None:
            enabled = os.environ.get("PIGATE_SESSION_MEMORY", "1").lower() not in ("0", "false", "no")
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, dict]" = OrderedDict()   # sid -> {turns: [str], ts: float}

    def _expired(self, entry, now) -> bool:
        return now - entry["ts"] > self.ttl_s

    def _evict(self, now):
        # drop expired, then LRU-trim to capacity
        for sid in [s for s, e in self._data.items() if self._expired(e, now)]:
            self._data.pop(sid, None)
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def history(self, session_id: str | None) -> list[str]:
        """Prior turns for this session, oldest first. Empty if memory is off/unknown."""
        if not (self.enabled and session_id):
            return []
        now = time.time()
        with self._lock:
            self._evict(now)
            entry = self._data.get(session_id)
            if entry is None or self._expired(entry, now):
                self._data.pop(session_id, None)
                return []
            self._data.move_to_end(session_id)          # mark recently used
            return list(entry["turns"])

    def append(self, session_id: str | None, text: str) -> None:
        """Record the current turn so the next turn in this session sees it."""
        if not (self.enabled and session_id):
            return
        now = time.time()
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None or self._expired(entry, now):
                entry = {"turns": [], "ts": now}
            entry["turns"].append(text)
            entry["turns"] = entry["turns"][-self.max_turns:]   # keep the last N
            entry["ts"] = now
            self._data[session_id] = entry
            self._data.move_to_end(session_id)
            self._evict(now)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
