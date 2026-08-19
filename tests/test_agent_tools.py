"""Isolated JSON-IR tool-session tests."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cppl.agents.tools import (
    IRToolSession,
    ToolSessionLimits,
    apply_json_patch,
    candidate_hash,
    query_pointer,
)
from cppl.ir.errors import PatternMismatch


def make_session(
    candidate=None,
    *,
    has_patterns=False,
    pattern_attempts=2,
    pattern_results=None,
    validator=None,
):
    calls = []

    def validate(body, run_patterns):
        calls.append((json.loads(json.dumps(body)), run_patterns))
        if validator is not None:
            return validator(body, run_patterns)
        return {"body": body}, 3 if run_patterns else 0

    session = IRToolSession(
        initial_candidate=candidate,
        task_context="module M requires output out",
        validate_candidate=validate,
        has_patterns=has_patterns,
        remaining_pattern_attempts=pattern_attempts,
        pattern_results=pattern_results,
        limits=ToolSessionLimits(timeout_seconds=0.5, max_output_chars=4096),
    )
    return session, calls


def test_json_pointer_and_patch_subset_are_transactional():
    original = [{"id": "sum", "op": "add", "args": ["a", "b"]}]
    patched = apply_json_patch(
        original,
        [
            {"op": "replace", "path": "/0/id", "value": "result"},
            {"op": "add", "path": "/-", "value": {"op": "output", "args": {"out": "result"}}},
        ],
    )

    assert original[0]["id"] == "sum"
    assert query_pointer(patched, "/0/id") == "result"
    assert query_pointer(patched, "/1/args/out") == "result"

    with pytest.raises(ValueError, match="unsupported"):
        apply_json_patch(original, [{"op": "move", "path": "/0"}])
    assert original == [{"id": "sum", "op": "add", "args": ["a", "b"]}]


def test_tool_session_edits_only_its_candidate_and_reports_diff():
    initial = [{"op": "output", "args": {"out": "a"}}]
    session, _ = make_session(initial)
    root = Path(session.root)
    try:
        result = session.execute(
            "patch_ir",
            {
                "operations": [
                    {"op": "replace", "path": "/0/args/out", "value": "b"}
                ]
            },
        )
        diff = session.execute("diff_ir", {})

        assert result["ok"]
        assert session.candidate[0]["args"]["out"] == "b"
        assert initial[0]["args"]["out"] == "a"
        assert diff["ok"]
        assert '-      "out": "a"' in diff["diff"]
        assert '+      "out": "b"' in diff["diff"]
        assert root.exists()
    finally:
        session.close()
    assert not root.exists()


def test_failed_patch_does_not_mutate_candidate():
    initial = [{"op": "output", "args": {"out": "a"}}]
    session, _ = make_session(initial)
    try:
        before = candidate_hash(session.candidate)
        result = session.execute(
            "patch_ir",
            {"operations": [{"op": "remove", "path": "/9"}]},
        )

        assert not result["ok"]
        assert candidate_hash(session.candidate) == before
        assert session.stats.failures == 1
    finally:
        session.close()


def test_search_and_diff_have_python_fallbacks():
    session, _ = make_session([{"op": "output", "args": {"out": "a"}}])
    try:
        session.executables = {"jq": None, "rg": None, "diff": None}
        search = session.execute("search_context", {"pattern": "requires", "target": "task"})
        diff = session.execute("diff_ir", {})

        assert search["ok"] and search["executor"] == "python"
        assert "requires" in search["matches"]
        assert diff["ok"] and diff["executor"] == "python"
        assert "transform_ir_jq" not in session.capabilities
    finally:
        session.close()


def test_pattern_attempts_are_cached_by_candidate_hash():
    session, calls = make_session(
        [{"op": "output", "args": {"out": "a"}}],
        has_patterns=True,
        pattern_attempts=1,
    )
    try:
        first = session.execute("run_patterns", {})
        second = session.execute("run_patterns", {})
        submit = session.execute("submit_ir", {})

        assert first["ok"] and not first["cached"]
        assert second["ok"] and second["cached"]
        assert submit["ok"] and submit["submitted"]
        assert len(session.stats.pattern_attempt_hashes) == 1
        assert sum(1 for _, patterns in calls if patterns) == 1
    finally:
        session.close()


def test_pattern_results_are_reused_across_tool_sessions():
    candidate = [{"op": "output", "args": {"out": "a"}}]
    first, first_calls = make_session(
        candidate,
        has_patterns=True,
        pattern_attempts=1,
    )
    try:
        assert first.execute("run_patterns", {})["ok"]
        saved_results = first.pattern_results
    finally:
        first.close()

    second, second_calls = make_session(
        candidate,
        has_patterns=True,
        pattern_attempts=0,
        pattern_results=saved_results,
    )
    try:
        reused = second.execute("run_patterns", {})
        assert reused["ok"] and reused["cached"]
        assert second.stats.pattern_attempt_hashes == set()
        assert not second_calls
        assert sum(1 for _, patterns in first_calls if patterns) == 1
    finally:
        second.close()


def test_pattern_budget_blocks_a_second_distinct_candidate():
    session, _ = make_session(
        [{"op": "output", "args": {"out": "a"}}],
        has_patterns=True,
        pattern_attempts=1,
    )
    try:
        assert session.execute("run_patterns", {})["ok"]
        assert session.execute(
            "patch_ir",
            {"operations": [{"op": "replace", "path": "/0/args/out", "value": "b"}]},
        )["ok"]
        denied = session.execute("run_patterns", {})

        assert not denied["ok"]
        assert "budget" in denied["message"]
    finally:
        session.close()


def test_static_failure_before_patterns_does_not_consume_simulation_attempt():
    def validator(body, run_patterns):
        if run_patterns:
            raise ValueError("static width failure")
        return {"body": body}, 0

    session, _ = make_session(
        [{"op": "output", "args": {"out": "a"}}],
        has_patterns=True,
        pattern_attempts=1,
        validator=validator,
    )
    try:
        result = session.execute("run_patterns", {})
        assert not result["ok"]
        assert session.stats.pattern_attempt_hashes == set()
    finally:
        session.close()


def test_pattern_mismatch_consumes_one_simulation_attempt():
    def validator(body, run_patterns):
        if run_patterns:
            raise PatternMismatch("behavior mismatch")
        return {"body": body}, 0

    session, _ = make_session(
        [{"op": "output", "args": {"out": "a"}}],
        has_patterns=True,
        pattern_attempts=1,
        validator=validator,
    )
    try:
        result = session.execute("run_patterns", {})
        assert not result["ok"]
        assert len(session.stats.pattern_attempt_hashes) == 1
    finally:
        session.close()


def test_tool_output_is_bounded():
    session, _ = make_session(
        [{"id": "x" * 500, "op": "constant", "value": 0, "width": 1}]
    )
    try:
        session.limits = ToolSessionLimits(timeout_seconds=0.5, max_output_chars=120)
        result = session.execute("query_ir", {"pointer": ""})

        assert result["truncated"] is True
        assert len(json.dumps(result)) < 300
    finally:
        session.close()


def test_invalid_arguments_and_expired_deadline_are_structured_failures():
    session, _ = make_session([])
    try:
        invalid = session.execute("query_ir", ["not", "an", "object"])
        assert not invalid["ok"]
        assert "JSON object" in invalid["message"]

        session.limits = ToolSessionLimits(deadline_monotonic=time.monotonic() - 1)
        expired = session.execute("diff_ir", {})
        assert not expired["ok"]
        assert "deadline" in expired["message"]
    finally:
        session.close()


def test_parallel_sessions_have_independent_temporary_candidates():
    def run(value):
        session, _ = make_session([])
        try:
            result = session.execute(
                "replace_ir",
                {"body": [{"op": "output", "args": {"out": value}}]},
            )
            return str(session.root), result, session.candidate
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(run, ("a", "b"))

    assert first[0] != second[0]
    assert first[1]["ok"] and second[1]["ok"]
    assert first[2][0]["args"]["out"] == "a"
    assert second[2][0]["args"]["out"] == "b"
    assert not Path(first[0]).exists()
    assert not Path(second[0]).exists()
