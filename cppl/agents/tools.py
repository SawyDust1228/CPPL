"""Isolated tools for incrementally constructing and validating JSON-IR."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..ir.errors import PatternMismatch


TOOL_PROTOCOL_VERSION = "cppl-ir-tools-v1"
MAX_CANDIDATE_BYTES = 256 * 1024


def detect_unix_capabilities() -> tuple[str, ...]:
    """Return optional Unix executors available to the isolated tool layer."""
    return tuple(name for name in ("jq", "rg", "diff") if shutil.which(name))


class IRToolError(ValueError):
    """A tool request is invalid or cannot be completed safely."""


class ToolSessionError(RuntimeError):
    """A model tool session ended without producing a usable candidate."""


@dataclass(frozen=True)
class ToolSessionLimits:
    timeout_seconds: float = 2.0
    max_output_chars: int = 8192
    max_candidate_bytes: int = MAX_CANDIDATE_BYTES
    deadline_monotonic: float | None = None
    remaining_model_tokens: int | None = None


@dataclass
class ToolSessionStats:
    calls: int = 0
    failures: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    pattern_attempt_hashes: set[str] = field(default_factory=set)


ValidationCallback = Callable[[list[dict[str, Any]], bool], tuple[dict, int]]
EventCallback = Callable[[str, str, float, bool, Mapping[str, Any]], None]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def candidate_hash(candidate: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_bytes(candidate)).hexdigest()


def _decode_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise IRToolError("JSON Pointer must be empty or begin with '/'.")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _array_index(token: str, length: int, *, allow_end: bool = False) -> int:
    if token == "-" and allow_end:
        return length
    if not token or (token.startswith("0") and token != "0") or not token.isdigit():
        raise IRToolError(f"Invalid JSON array index {token!r}.")
    index = int(token)
    upper = length if allow_end else length - 1
    if index < 0 or index > upper:
        raise IRToolError(f"JSON array index {index} is out of range.")
    return index


def _resolve_parent(document: Any, pointer: str) -> tuple[Any, str]:
    tokens = _decode_pointer(pointer)
    if not tokens:
        return None, ""
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[_array_index(token, len(current))]
        elif isinstance(current, dict):
            if token not in current:
                raise IRToolError(f"JSON Pointer component {token!r} does not exist.")
            current = current[token]
        else:
            raise IRToolError("JSON Pointer traverses a scalar value.")
    return current, tokens[-1]


def query_pointer(document: Any, pointer: str) -> Any:
    current = document
    for token in _decode_pointer(pointer):
        if isinstance(current, list):
            current = current[_array_index(token, len(current))]
        elif isinstance(current, dict):
            if token not in current:
                raise IRToolError(f"JSON Pointer component {token!r} does not exist.")
            current = current[token]
        else:
            raise IRToolError("JSON Pointer traverses a scalar value.")
    return copy.deepcopy(current)


def apply_json_patch(document: list, operations: list[dict[str, Any]]) -> list:
    """Apply the safe add/remove/replace subset of RFC 6902 transactionally."""
    result: Any = copy.deepcopy(document)
    if not isinstance(operations, list) or not operations:
        raise IRToolError("operations must be a non-empty JSON array.")
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise IRToolError(f"Patch operation {index} must be an object.")
        kind = operation.get("op")
        path = operation.get("path")
        if kind not in {"add", "remove", "replace"}:
            raise IRToolError(
                f"Patch operation {index} uses unsupported op {kind!r}; "
                "use add, remove, or replace."
            )
        if not isinstance(path, str):
            raise IRToolError(f"Patch operation {index} path must be a string.")
        if path == "":
            if kind == "remove":
                raise IRToolError("The candidate root cannot be removed.")
            if "value" not in operation:
                raise IRToolError(f"Patch operation {index} requires value.")
            result = copy.deepcopy(operation["value"])
            continue

        parent, token = _resolve_parent(result, path)
        if isinstance(parent, list):
            if kind == "add":
                if "value" not in operation:
                    raise IRToolError(f"Patch operation {index} requires value.")
                position = _array_index(token, len(parent), allow_end=True)
                parent.insert(position, copy.deepcopy(operation["value"]))
            else:
                position = _array_index(token, len(parent))
                if kind == "remove":
                    parent.pop(position)
                else:
                    if "value" not in operation:
                        raise IRToolError(f"Patch operation {index} requires value.")
                    parent[position] = copy.deepcopy(operation["value"])
        elif isinstance(parent, dict):
            if kind == "add":
                if "value" not in operation:
                    raise IRToolError(f"Patch operation {index} requires value.")
                parent[token] = copy.deepcopy(operation["value"])
            else:
                if token not in parent:
                    raise IRToolError(f"Patch path {path!r} does not exist.")
                if kind == "remove":
                    del parent[token]
                else:
                    if "value" not in operation:
                        raise IRToolError(f"Patch operation {index} requires value.")
                    parent[token] = copy.deepcopy(operation["value"])
        else:
            raise IRToolError(f"Patch path {path!r} has no container parent.")
    if not isinstance(result, list):
        raise IRToolError("The JSON-IR candidate root must remain an array.")
    return result


class IRToolSession:
    """One module-local, auditable JSON-IR editing session."""

    def __init__(
        self,
        *,
        initial_candidate: list | None,
        task_context: str,
        validate_candidate: ValidationCallback,
        has_patterns: bool,
        remaining_pattern_attempts: int,
        pattern_results: Mapping[str, tuple[bool, int, str]] | None = None,
        limits: ToolSessionLimits,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.limits = limits
        self._validate_candidate = validate_candidate
        self.has_patterns = has_patterns
        self.remaining_pattern_attempts = max(0, remaining_pattern_attempts)
        self.event_callback = event_callback
        self.stats = ToolSessionStats()
        self.submitted = False
        self.submitted_patterns_checked = 0
        self._pattern_cache: dict[str, tuple[bool, int, str]] = dict(
            pattern_results or {}
        )
        self._temp = tempfile.TemporaryDirectory(prefix="cppl-ir-tools-")
        self.root = Path(self._temp.name)
        self.baseline: list = copy.deepcopy(initial_candidate or [])
        self.candidate: list = copy.deepcopy(self.baseline)
        self._commit_candidate(self.candidate, initialize=True)
        self._write_json(self.root / "baseline.json", self.baseline)
        (self.root / "task_context.txt").write_text(task_context, encoding="utf-8")
        self.executables = {name: shutil.which(name) for name in ("jq", "rg", "diff")}

    def close(self) -> None:
        self._temp.cleanup()

    def __enter__(self) -> "IRToolSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def capabilities(self) -> tuple[str, ...]:
        result = [
            "replace_ir",
            "patch_ir",
            "query_ir",
            "search_context",
            "diff_ir",
            "validate_ir",
            "run_patterns",
            "submit_ir",
        ]
        if self.executables["jq"]:
            result.insert(3, "transform_ir_jq")
        return tuple(result)

    @property
    def pattern_results(self) -> dict[str, tuple[bool, int, str]]:
        """Return pattern outcomes so later repair sessions can reuse them."""
        return dict(self._pattern_cache)

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        definitions = [
            self._definition(
                "replace_ir",
                "Replace the entire JSON-IR body candidate with the supplied array.",
                {
                    "type": "object",
                    "properties": {"body": {"type": "array", "items": {"type": "object"}}},
                    "required": ["body"],
                    "additionalProperties": False,
                },
            ),
            self._definition(
                "patch_ir",
                "Atomically apply add/remove/replace JSON Patch operations to the candidate.",
                {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "op": {"enum": ["add", "remove", "replace"]},
                                    "path": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["op", "path"],
                            },
                        }
                    },
                    "required": ["operations"],
                    "additionalProperties": False,
                },
            ),
            self._definition(
                "query_ir",
                "Read one value from the candidate using an RFC 6901 JSON Pointer.",
                {
                    "type": "object",
                    "properties": {"pointer": {"type": "string"}},
                    "required": ["pointer"],
                    "additionalProperties": False,
                },
            ),
            self._definition(
                "search_context",
                "Search the current candidate or task context for a regular expression.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "target": {"enum": ["candidate", "task"]},
                    },
                    "required": ["pattern", "target"],
                    "additionalProperties": False,
                },
            ),
            self._definition(
                "diff_ir",
                "Show a unified diff between the initial and current candidates.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._definition(
                "validate_ir",
                "Run JSON schema, SSA, interface, cycle, and width validation without patterns.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._definition(
                "run_patterns",
                "Run executable behavior patterns for the current candidate.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            self._definition(
                "submit_ir",
                "Validate and submit the current candidate as the final module body.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
        ]
        if self.executables["jq"]:
            definitions.insert(
                3,
                self._definition(
                    "transform_ir_jq",
                    "Transform the candidate with a jq filter; output must remain a JSON array.",
                    {
                        "type": "object",
                        "properties": {"filter": {"type": "string"}},
                        "required": ["filter"],
                        "additionalProperties": False,
                    },
                ),
            )
        return definitions

    @staticmethod
    def _definition(name: str, description: str, parameters: dict) -> dict:
        return {"name": name, "description": description, "parameters": parameters}

    def execute(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        started = time.monotonic()
        self.stats.calls += 1
        self.stats.counts[name] = self.stats.counts.get(name, 0) + 1
        if self.event_callback:
            self.event_callback("tool_start", name, 0.0, True, {})
        try:
            self.check_runtime_budget()
            if arguments is not None and not isinstance(arguments, Mapping):
                raise IRToolError("Tool arguments must be a JSON object.")
            arguments = dict(arguments or {})
            if name not in self.capabilities:
                raise IRToolError(f"Unknown or unavailable tool {name!r}.")
            handler = getattr(self, f"_tool_{name}")
            payload = handler(**arguments)
            result = {"ok": True, **payload}
            ok = True
        except Exception as exc:
            self.stats.failures += 1
            result = {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc)[:4000],
            }
            ok = False
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = round(elapsed, 6)
        result = self._limit_payload(result)
        if self.event_callback:
            self.event_callback(
                "tool_finish" if ok else "tool_failed",
                name,
                elapsed,
                ok,
                {"error": result.get("error", "")},
            )
        return result

    def check_runtime_budget(self, model_tokens: int = 0) -> None:
        if (
            self.limits.deadline_monotonic is not None
            and time.monotonic() >= self.limits.deadline_monotonic
        ):
            raise ToolSessionError("The module deadline was exhausted during tool use.")
        if (
            self.limits.remaining_model_tokens is not None
            and model_tokens >= self.limits.remaining_model_tokens
        ):
            raise ToolSessionError("The module token budget was exhausted during tool use.")

    def _tool_replace_ir(self, body: list) -> dict:
        if not isinstance(body, list):
            raise IRToolError("body must be a JSON array.")
        self._commit_candidate(copy.deepcopy(body))
        return self._candidate_summary()

    def _tool_patch_ir(self, operations: list[dict[str, Any]]) -> dict:
        self._commit_candidate(apply_json_patch(self.candidate, operations))
        return self._candidate_summary()

    def _tool_query_ir(self, pointer: str) -> dict:
        return {"value": query_pointer(self.candidate, pointer)}

    def _tool_transform_ir_jq(self, filter: str) -> dict:
        executable = self.executables.get("jq")
        if not executable:
            raise IRToolError("jq is not available in this environment.")
        completed = self._run_command(
            [executable, filter], input_text=json.dumps(self.candidate, ensure_ascii=False)
        )
        if completed.returncode != 0:
            raise IRToolError(f"jq failed: {completed.stderr.strip()[:2000]}")
        try:
            transformed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise IRToolError(f"jq returned invalid JSON: {exc}") from exc
        if not isinstance(transformed, list):
            raise IRToolError("jq output must be a JSON array.")
        self._commit_candidate(transformed)
        return {**self._candidate_summary(), "executor": "jq"}

    def _tool_search_context(self, pattern: str, target: str) -> dict:
        if not isinstance(pattern, str) or not pattern:
            raise IRToolError("pattern must be a non-empty string.")
        if target not in {"candidate", "task"}:
            raise IRToolError("target must be 'candidate' or 'task'.")
        path = self.root / ("candidate.json" if target == "candidate" else "task_context.txt")
        executable = self.executables.get("rg")
        if executable:
            completed = self._run_command(
                [
                    executable,
                    "-n",
                    "--color",
                    "never",
                    "--max-count",
                    "100",
                    "--",
                    pattern,
                    str(path),
                ]
            )
            if completed.returncode not in (0, 1):
                raise IRToolError(f"rg failed: {completed.stderr.strip()[:2000]}")
            return {"matches": completed.stdout, "executor": "rg"}
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise IRToolError(f"Invalid regular expression: {exc}") from exc
        matches = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if expression.search(line):
                matches.append(f"{number}:{line}")
                if len(matches) >= 100:
                    break
        return {"matches": "\n".join(matches), "executor": "python"}

    def _tool_diff_ir(self) -> dict:
        executable = self.executables.get("diff")
        if executable:
            completed = self._run_command(
                [
                    executable,
                    "-u",
                    str(self.root / "baseline.json"),
                    str(self.root / "candidate.json"),
                ]
            )
            if completed.returncode not in (0, 1):
                raise IRToolError(f"diff failed: {completed.stderr.strip()[:2000]}")
            return {"diff": completed.stdout, "executor": "diff"}
        before = (self.root / "baseline.json").read_text(encoding="utf-8").splitlines(True)
        after = (self.root / "candidate.json").read_text(encoding="utf-8").splitlines(True)
        rendered = "".join(
            difflib.unified_diff(before, after, fromfile="baseline.json", tofile="candidate.json")
        )
        return {"diff": rendered, "executor": "python"}

    def _tool_validate_ir(self) -> dict:
        _, _ = self._validate_candidate(self._typed_candidate(), False)
        return {"valid": True, **self._candidate_summary()}

    def _tool_run_patterns(self) -> dict:
        return self._run_patterns_cached()

    def _tool_submit_ir(self) -> dict:
        self._validate_candidate(self._typed_candidate(), False)
        pattern_result = (
            self._run_patterns_cached()
            if self.has_patterns
            else {"patterns_checked": 0}
        )
        self.submitted = True
        self.submitted_patterns_checked = int(pattern_result.get("patterns_checked", 0))
        return {
            "submitted": True,
            "patterns_checked": self.submitted_patterns_checked,
            **self._candidate_summary(),
        }

    def _run_patterns_cached(self) -> dict:
        if not self.has_patterns:
            return {"valid": True, "patterns_checked": 0, "cached": True}
        digest = candidate_hash(self._typed_candidate())
        cached = self._pattern_cache.get(digest)
        if cached is not None:
            ok, checked, message = cached
            if not ok:
                raise IRToolError(
                    "This unchanged candidate already failed patterns. Edit it "
                    "with patch_ir or replace_ir before running patterns again. "
                    f"Cached failure: {message}"
                )
            return {"valid": True, "patterns_checked": checked, "cached": True}
        if len(self.stats.pattern_attempt_hashes) >= self.remaining_pattern_attempts:
            raise IRToolError("The module pattern-attempt budget is exhausted.")
        try:
            _, checked = self._validate_candidate(self._typed_candidate(), True)
        except PatternMismatch as exc:
            self.stats.pattern_attempt_hashes.add(digest)
            self._pattern_cache[digest] = (False, 0, str(exc)[:4000])
            raise
        except Exception as exc:
            raise
        self.stats.pattern_attempt_hashes.add(digest)
        self._pattern_cache[digest] = (True, checked, "")
        return {"valid": True, "patterns_checked": checked, "cached": False}

    def _typed_candidate(self) -> list[dict[str, Any]]:
        if any(not isinstance(operation, dict) for operation in self.candidate):
            raise IRToolError("Every JSON-IR body entry must be an object.")
        return copy.deepcopy(self.candidate)

    def _commit_candidate(self, candidate: list, *, initialize: bool = False) -> None:
        if not isinstance(candidate, list):
            raise IRToolError("The JSON-IR candidate root must be an array.")
        encoded = _json_bytes(candidate)
        if len(encoded) > self.limits.max_candidate_bytes:
            raise IRToolError(
                f"Candidate is {len(encoded)} bytes; limit is {self.limits.max_candidate_bytes}."
            )
        self.candidate = copy.deepcopy(candidate)
        self._write_json(self.root / "candidate.json", self.candidate)
        if not initialize:
            self.submitted = False

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _candidate_summary(self) -> dict[str, Any]:
        return {
            "operation_count": len(self.candidate),
            "candidate_hash": candidate_hash(self._typed_candidate())[:16]
            if all(isinstance(item, dict) for item in self.candidate)
            else hashlib.sha256(_json_bytes(self.candidate)).hexdigest()[:16],
        }

    def _run_command(
        self, argv: list[str], *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        timeout = self.limits.timeout_seconds
        if self.limits.deadline_monotonic is not None:
            timeout = min(
                timeout,
                max(0.001, self.limits.deadline_monotonic - time.monotonic()),
            )
        try:
            return subprocess.run(
                argv,
                cwd=self.root,
                env={"PATH": os.defpath, "LC_ALL": "C"},
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise IRToolError(
                f"Tool exceeded {timeout:.3g}s timeout."
            ) from exc

    def _limit_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        if len(encoded) <= self.limits.max_output_chars:
            return payload
        budget = max(0, self.limits.max_output_chars - 200)
        return {
            "ok": payload.get("ok", False),
            "truncated": True,
            "output": encoded[:budget],
            "message": "Tool output was truncated.",
        }
