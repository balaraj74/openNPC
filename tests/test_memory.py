from opennpc.memory import InMemoryMemoryStore, SQLiteMemoryStore
from opennpc.types import MemoryEvent


def test_in_memory_summary_prioritizes_important_events() -> None:
    store = InMemoryMemoryStore()
    store.add(MemoryEvent("npc_1", "trading here is profitable", importance=0.3))
    store.add(MemoryEvent("npc_1", "player is hostile", importance=0.9))

    summary = store.summarize("npc_1")

    assert "important: player is hostile" in summary
    assert "recent: trading here is profitable" in summary


def test_sqlite_memory_persists_events(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(path)
    store.add(MemoryEvent("npc_1", "route is dangerous", importance=0.8, tags=["route"]))

    reopened = SQLiteMemoryStore(path)
    important = reopened.important("npc_1")

    assert len(important) == 1
    assert important[0].text == "route is dangerous"
