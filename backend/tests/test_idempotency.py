"""The idempotency cache behind ``/ask`` (``src/core/idempotency.py``)."""

import threading

from src.core import idempotency
from src.core.idempotency import IdempotencyStore, build_key


def test_an_explicit_key_is_scoped_to_the_user_and_ignores_the_body():
    assert build_key("alice", "k1", {"query": "a"}) == "alice:k1"
    assert build_key("alice", "k1", {"query": "b"}) == "alice:k1"
    assert build_key("bob", "k1", {"query": "a"}) != build_key("alice", "k1", {"query": "a"})


def test_without_a_key_the_body_is_fingerprinted_regardless_of_field_order():
    left = build_key("alice", None, {"query": "a", "thread_id": None})
    right = build_key("alice", "", {"thread_id": None, "query": "a"})

    assert left == right
    assert left.startswith("alice:auto:")
    assert build_key("alice", None, {"query": "b"}) != left


def test_a_stored_response_is_returned_with_its_status_code():
    store = IdempotencyStore()

    assert store.get("k") is None

    store.put("k", {"status": "answered"}, 200)
    cached = store.get("k")

    assert cached.payload == {"status": "answered"}
    assert cached.status_code == 200


def test_entries_expire_after_the_ttl(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(idempotency.time, "time", lambda: clock["now"])

    store = IdempotencyStore(ttl_seconds=10)
    store.put("k", {"x": 1}, 202)

    clock["now"] += 9
    assert store.get("k") is not None

    clock["now"] += 2
    assert store.get("k") is None


def test_a_later_put_overwrites_the_earlier_one():
    store = IdempotencyStore()
    store.put("k", {"v": 1}, 200)
    store.put("k", {"v": 2}, 202)

    assert store.get("k").payload == {"v": 2}
    assert store.get("k").status_code == 202


def test_the_store_survives_concurrent_writers():
    store = IdempotencyStore()

    def hammer(prefix):
        for n in range(200):
            store.put(f"{prefix}-{n}", {"n": n}, 200)
            store.get(f"{prefix}-{n // 2}")

    threads = [threading.Thread(target=hammer, args=(t,)) for t in range(8)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert store.get("3-199").payload == {"n": 199}
    assert len(store._entries) == 8 * 200
