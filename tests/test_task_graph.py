"""Tests for the parallel task graph (TaskGraph, SharedResponses, dependency resolution).

Validates that:
- Independent tasks run concurrently
- Dependent tasks wait for prerequisites
- Failed dependencies propagate errors
- SharedResponses is thread-safe
- Task graph handles edge cases (empty, single task, diamond dependencies)
"""

from __future__ import annotations

import threading
import time

import pytest

from src.parallel_runner import Task, TaskGraph, SharedResponses


# ---------------------------------------------------------------------------
# TaskGraph basics
# ---------------------------------------------------------------------------

class TestTaskGraphBasics:
    """Basic TaskGraph functionality."""

    def test_empty_graph(self) -> None:
        graph = TaskGraph(max_workers=2)
        result = graph.run()
        assert result == {}

    def test_single_task(self) -> None:
        graph = TaskGraph(max_workers=1)
        graph.add(Task(id="t1", fn=lambda: 42))
        result = graph.run()
        assert "t1" in result
        assert result["t1"].result == 42
        assert result["t1"].error is None

    def test_independent_tasks_run(self) -> None:
        """Multiple independent tasks all complete."""
        graph = TaskGraph(max_workers=4)
        for i in range(5):
            graph.add(Task(id=f"t{i}", fn=lambda x=i: x * 10))
        result = graph.run()
        assert len(result) == 5
        for i in range(5):
            assert result[f"t{i}"].result == i * 10

    def test_independent_tasks_run_concurrently(self) -> None:
        """Independent tasks actually execute in parallel, not sequentially."""
        barrier = threading.Barrier(3, timeout=5)
        results = []

        def concurrent_fn(task_id):
            # All 3 tasks must arrive at the barrier simultaneously
            barrier.wait()
            results.append(task_id)
            return task_id

        graph = TaskGraph(max_workers=3)
        graph.add(Task(id="a", fn=lambda: concurrent_fn("a")))
        graph.add(Task(id="b", fn=lambda: concurrent_fn("b")))
        graph.add(Task(id="c", fn=lambda: concurrent_fn("c")))
        graph.run()
        assert set(results) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    """Dependency ordering and propagation."""

    def test_linear_chain(self) -> None:
        """A → B → C: each task waits for its predecessor."""
        order = []

        def make_fn(label):
            def fn():
                order.append(label)
                return label
            return fn

        graph = TaskGraph(max_workers=4)
        graph.add(Task(id="a", fn=make_fn("a")))
        graph.add(Task(id="b", fn=make_fn("b"), depends_on=["a"]))
        graph.add(Task(id="c", fn=make_fn("c"), depends_on=["b"]))
        graph.run()

        assert order == ["a", "b", "c"]

    def test_diamond_dependency(self) -> None:
        """Diamond: A → B, A → C, B+C → D."""
        order = []
        lock = threading.Lock()

        def make_fn(label):
            def fn():
                with lock:
                    order.append(label)
                return label
            return fn

        graph = TaskGraph(max_workers=4)
        graph.add(Task(id="a", fn=make_fn("a")))
        graph.add(Task(id="b", fn=make_fn("b"), depends_on=["a"]))
        graph.add(Task(id="c", fn=make_fn("c"), depends_on=["a"]))
        graph.add(Task(id="d", fn=make_fn("d"), depends_on=["b", "c"]))
        graph.run()

        # A must be first, D must be last
        assert order[0] == "a"
        assert order[-1] == "d"
        # B and C can be in any order
        assert set(order[1:3]) == {"b", "c"}

    def test_failed_dependency_propagates(self) -> None:
        """If A fails, B (which depends on A) should also fail."""
        def fail():
            raise ValueError("boom")

        graph = TaskGraph(max_workers=2)
        graph.add(Task(id="a", fn=fail))
        graph.add(Task(id="b", fn=lambda: "ok", depends_on=["a"]))
        result = graph.run()

        assert result["a"].error is not None
        assert "boom" in str(result["a"].error)
        assert result["b"].error is not None
        assert "Dependency a failed" in str(result["b"].error)

    def test_partial_failure_doesnt_block_independent(self) -> None:
        """If A fails, C (independent of A) should still run."""
        def fail():
            raise ValueError("boom")

        graph = TaskGraph(max_workers=4)
        graph.add(Task(id="a", fn=fail))
        graph.add(Task(id="b", fn=lambda: "ok_b", depends_on=["a"]))
        graph.add(Task(id="c", fn=lambda: "ok_c"))  # independent
        result = graph.run()

        assert result["a"].error is not None
        assert result["b"].error is not None
        assert result["c"].result == "ok_c"
        assert result["c"].error is None

    def test_fan_out_fan_in(self) -> None:
        """One root fans out to N tasks, then N tasks fan into one."""
        n = 5
        results_collector = []
        lock = threading.Lock()

        def root_fn():
            return "root"

        def worker_fn(i):
            def fn():
                with lock:
                    results_collector.append(i)
                return i
            return fn

        def collector_fn():
            return sorted(results_collector)

        graph = TaskGraph(max_workers=8)
        graph.add(Task(id="root", fn=root_fn))
        for i in range(n):
            graph.add(Task(id=f"w{i}", fn=worker_fn(i), depends_on=["root"]))
        graph.add(Task(
            id="collect",
            fn=collector_fn,
            depends_on=[f"w{i}" for i in range(n)],
        ))
        result = graph.run()

        assert result["collect"].result == list(range(n))


# ---------------------------------------------------------------------------
# SharedResponses
# ---------------------------------------------------------------------------

class TestSharedResponses:
    """Thread-safe shared response store."""

    def test_set_and_get(self) -> None:
        shared = SharedResponses()
        shared.set("key1", "value1")
        assert shared.get("key1") == "value1"

    def test_get_missing_key(self) -> None:
        shared = SharedResponses()
        assert shared.get("nonexistent") == ""

    def test_overwrite(self) -> None:
        shared = SharedResponses()
        shared.set("k", "v1")
        shared.set("k", "v2")
        assert shared.get("k") == "v2"

    def test_concurrent_writes(self) -> None:
        """Multiple threads writing different keys concurrently."""
        shared = SharedResponses()
        n = 50
        barrier = threading.Barrier(n, timeout=5)

        def writer(i):
            barrier.wait()
            shared.set(f"key_{i}", f"value_{i}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for i in range(n):
            assert shared.get(f"key_{i}") == f"value_{i}"

    def test_concurrent_read_write(self) -> None:
        """Reading while another thread writes should not crash."""
        shared = SharedResponses()
        shared.set("k", "initial")
        errors = []

        def writer():
            for i in range(100):
                shared.set("k", f"v{i}")

        def reader():
            for _ in range(100):
                val = shared.get("k")
                if not val.startswith(("initial", "v")):
                    errors.append(f"Unexpected value: {val}")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Psych QA helpers
# ---------------------------------------------------------------------------

class TestPsychQAHelpers:
    """Test the psych Q&A storage helpers used in the task graph."""

    def test_store_and_retrieve(self) -> None:
        from src.parallel_runner import _store_psych_qa, _get_psych_prior_qa

        shared = SharedResponses()
        key = "test_psych"

        _store_psych_qa(shared, key, "Q1?", "A1")
        _store_psych_qa(shared, key, "Q2?", "A2")

        result = _get_psych_prior_qa(shared, key)
        assert len(result) == 2
        assert result[0] == ("Q1?", "A1")
        assert result[1] == ("Q2?", "A2")

    def test_empty_retrieval(self) -> None:
        from src.parallel_runner import _get_psych_prior_qa

        shared = SharedResponses()
        result = _get_psych_prior_qa(shared, "nonexistent")
        assert result == []
