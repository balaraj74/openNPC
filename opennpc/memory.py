"""Memory backends for OpenNPC agents."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from opennpc.types import MemoryEvent


class MemoryStore(Protocol):
    def add(self, event: MemoryEvent) -> None:
        ...

    def recent(self, agent_id: str, limit: int = 8) -> list[MemoryEvent]:
        ...

    def important(self, agent_id: str, minimum_importance: float = 0.7, limit: int = 8) -> list[MemoryEvent]:
        ...

    def summarize(self, agent_id: str, limit: int = 8) -> str:
        ...

    def clear(self, agent_id: str | None = None) -> None:
        ...


class InMemoryMemoryStore:
    """Short-term plus promoted long-term memory stored in process."""

    def __init__(self, short_term_limit: int = 32) -> None:
        self.short_term_limit = short_term_limit
        self._events: dict[str, list[MemoryEvent]] = defaultdict(list)

    def add(self, event: MemoryEvent) -> None:
        events = self._events[event.agent_id]
        events.append(event)
        low_value = [item for item in events if item.importance < 0.7]
        if len(low_value) > self.short_term_limit:
            remove_count = len(low_value) - self.short_term_limit
            removable_ids = {id(item) for item in low_value[:remove_count]}
            self._events[event.agent_id] = [item for item in events if id(item) not in removable_ids]

    def recent(self, agent_id: str, limit: int = 8) -> list[MemoryEvent]:
        return sorted(self._events.get(agent_id, []), key=lambda item: item.timestamp, reverse=True)[:limit]

    def important(self, agent_id: str, minimum_importance: float = 0.7, limit: int = 8) -> list[MemoryEvent]:
        events = [event for event in self._events.get(agent_id, []) if event.importance >= minimum_importance]
        return sorted(events, key=lambda item: (item.importance, item.timestamp), reverse=True)[:limit]

    def summarize(self, agent_id: str, limit: int = 8) -> str:
        recent = self.recent(agent_id, limit=limit)
        important = self.important(agent_id, limit=limit)
        seen: set[str] = set()
        fragments: list[str] = []
        for event in important + recent:
            if event.text in seen:
                continue
            seen.add(event.text)
            label = "important" if event.importance >= 0.7 else "recent"
            fragments.append(f"{label}: {event.text}")
        return " | ".join(fragments)

    def clear(self, agent_id: str | None = None) -> None:
        if agent_id is None:
            self._events.clear()
        else:
            self._events.pop(agent_id, None)


class SQLiteMemoryStore:
    """Persistent memory store suitable for demos and local game tooling."""

    def __init__(self, path: str | Path = "opennpc_memory.sqlite3") -> None:
        self.path = Path(path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    tags TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent_time ON memory_events(agent_id, timestamp)")

    def add(self, event: MemoryEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_events(agent_id, text, importance, tags, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.agent_id,
                    event.text,
                    event.importance,
                    json.dumps(event.tags),
                    event.timestamp,
                    json.dumps(event.metadata),
                ),
            )

    def _rows_to_events(self, rows: list[tuple]) -> list[MemoryEvent]:
        return [
            MemoryEvent(
                agent_id=row[0],
                text=row[1],
                importance=row[2],
                tags=json.loads(row[3]),
                timestamp=row[4],
                metadata=json.loads(row[5]),
            )
            for row in rows
        ]

    def recent(self, agent_id: str, limit: int = 8) -> list[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_id, text, importance, tags, timestamp, metadata
                FROM memory_events
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (agent_id, limit),
            ).fetchall()
        return self._rows_to_events(rows)

    def important(self, agent_id: str, minimum_importance: float = 0.7, limit: int = 8) -> list[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_id, text, importance, tags, timestamp, metadata
                FROM memory_events
                WHERE agent_id = ? AND importance >= ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (agent_id, minimum_importance, limit),
            ).fetchall()
        return self._rows_to_events(rows)

    def summarize(self, agent_id: str, limit: int = 8) -> str:
        recent = self.recent(agent_id, limit=limit)
        important = self.important(agent_id, limit=limit)
        seen: set[str] = set()
        fragments: list[str] = []
        for event in important + recent:
            if event.text in seen:
                continue
            seen.add(event.text)
            label = "important" if event.importance >= 0.7 else "recent"
            fragments.append(f"{label}: {event.text}")
        return " | ".join(fragments)

    def clear(self, agent_id: str | None = None) -> None:
        with self._connect() as conn:
            if agent_id is None:
                conn.execute("DELETE FROM memory_events")
            else:
                conn.execute("DELETE FROM memory_events WHERE agent_id = ?", (agent_id,))
