"""OpenNPC backends package.

Provides plug-in memory store backends beyond the default in-process store.

Available backends
------------------
``SQLiteMemoryStore``
    Persistent, file-backed store. Zero extra dependencies (uses stdlib
    ``sqlite3``).  Already available in ``opennpc.memory`` for backwards
    compat; re-exported here for discoverability.

``RedisMemoryStore``
    Shared, network-accessible store.  Requires ``redis-py``::

        pip install 'opennpc[backends]'

Usage
-----
    from opennpc.backends import RedisMemoryStore
    store = RedisMemoryStore(host="localhost", port=6379, db=0)
    engine = DecisionEngine(memory_store=store)
"""

from opennpc.memory import SQLiteMemoryStore

try:
    from opennpc.backends.redis_memory import RedisMemoryStore
except ImportError:
    RedisMemoryStore = None  # type: ignore[assignment, misc]

__all__ = ["SQLiteMemoryStore", "RedisMemoryStore"]
