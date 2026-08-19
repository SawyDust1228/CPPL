"""Configuration and reporting models for the CPPL agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..env import resolve_env_value, resolve_llm_generation_kwargs
from ..ir.errors import CircuitPPLError


class AgentRuntimeError(CircuitPPLError):
    """Base error raised by the agent runtime."""


class ContextBudgetError(AgentRuntimeError):
    """A module cannot fit in the configured model context window."""


class GenerationBudgetError(AgentRuntimeError):
    """A module exhausted its total generation time or token budget."""


def env_int(name: str, default: int) -> int:
    value = resolve_env_value(name)
    return int(value) if value is not None else default


def env_bool(name: str, default: bool) -> bool:
    value = resolve_env_value(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


@dataclass(frozen=True)
class AgentConfig:
    """Optional agent settings; omitted fields resolve from environment/defaults."""

    max_parallelism: Optional[int] = None
    context_window_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    safety_margin_tokens: Optional[int] = None
    cache_enabled: Optional[bool] = None
    cache_dir: Optional[str | Path] = None
    fail_fast: Optional[bool] = None
    log_enabled: Optional[bool] = None
    transport_retries: Optional[int] = None
    request_timeout: Optional[float] = None
    generation_kwargs: Optional[dict[str, Any]] = None
    module_deadline_seconds: Optional[float] = None
    max_module_tokens: Optional[int] = None
    checkpoint_enabled: Optional[bool] = None
    checkpoint_path: Optional[str | Path] = None

    def resolve(self) -> "ResolvedAgentConfig":
        """Merge Python overrides with CPPL_AGENT_* and existing LLM defaults."""
        llm_kwargs = resolve_llm_generation_kwargs()
        generation_kwargs = dict(llm_kwargs)
        if self.generation_kwargs:
            generation_kwargs.update(self.generation_kwargs)

        legacy_output_tokens = int(generation_kwargs.pop("max_tokens", 12000))
        legacy_timeout = float(generation_kwargs.pop("timeout", 90))

        env_output = resolve_env_value("CPPL_AGENT_OUTPUT_TOKENS")
        output_tokens = self.output_tokens
        if output_tokens is None:
            output_tokens = (
                int(env_output) if env_output is not None else legacy_output_tokens
            )

        env_timeout = resolve_env_value("CPPL_AGENT_REQUEST_TIMEOUT")
        request_timeout = self.request_timeout
        if request_timeout is None:
            request_timeout = (
                float(env_timeout) if env_timeout is not None else legacy_timeout
            )

        cache_dir = self.cache_dir
        if cache_dir is None:
            cache_dir = resolve_env_value("CPPL_AGENT_CACHE_DIR") or ".cppl/cache"

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = (
                resolve_env_value("CPPL_AGENT_CHECKPOINT_PATH")
                or ".cppl/checkpoints.sqlite3"
            )

        resolved = ResolvedAgentConfig(
            max_parallelism=(
                self.max_parallelism
                if self.max_parallelism is not None
                else env_int("CPPL_AGENT_MAX_PARALLELISM", 4)
            ),
            context_window_tokens=(
                self.context_window_tokens
                if self.context_window_tokens is not None
                else env_int("CPPL_AGENT_CONTEXT_WINDOW", 32768)
            ),
            output_tokens=output_tokens,
            safety_margin_tokens=(
                self.safety_margin_tokens
                if self.safety_margin_tokens is not None
                else env_int("CPPL_AGENT_SAFETY_MARGIN", 2048)
            ),
            cache_enabled=(
                self.cache_enabled
                if self.cache_enabled is not None
                else env_bool("CPPL_AGENT_CACHE_ENABLED", True)
            ),
            cache_dir=Path(cache_dir),
            fail_fast=(
                self.fail_fast
                if self.fail_fast is not None
                else env_bool("CPPL_AGENT_FAIL_FAST", True)
            ),
            log_enabled=(
                self.log_enabled
                if self.log_enabled is not None
                else env_bool("CPPL_AGENT_LOG_ENABLED", True)
            ),
            transport_retries=(
                self.transport_retries
                if self.transport_retries is not None
                else env_int("CPPL_AGENT_TRANSPORT_RETRIES", 2)
            ),
            request_timeout=request_timeout,
            generation_kwargs=generation_kwargs,
            module_deadline_seconds=(
                self.module_deadline_seconds
                if self.module_deadline_seconds is not None
                else float(resolve_env_value("CPPL_AGENT_MODULE_DEADLINE") or 600)
            ),
            max_module_tokens=(
                self.max_module_tokens
                if self.max_module_tokens is not None
                else int(resolve_env_value("CPPL_AGENT_MAX_MODULE_TOKENS") or 60000)
            ),
            checkpoint_enabled=(
                self.checkpoint_enabled
                if self.checkpoint_enabled is not None
                else env_bool("CPPL_AGENT_CHECKPOINT_ENABLED", True)
            ),
            checkpoint_path=Path(checkpoint_path),
        )
        resolved.validate()
        return resolved


@dataclass(frozen=True)
class ResolvedAgentConfig:
    max_parallelism: int
    context_window_tokens: int
    output_tokens: int
    safety_margin_tokens: int
    cache_enabled: bool
    cache_dir: Path
    fail_fast: bool
    log_enabled: bool
    transport_retries: int
    request_timeout: float
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    module_deadline_seconds: float = 600.0
    max_module_tokens: int = 60000
    checkpoint_enabled: bool = True
    checkpoint_path: Path = Path(".cppl/checkpoints.sqlite3")

    def validate(self) -> None:
        if self.max_parallelism <= 0:
            raise ValueError("max_parallelism must be positive")
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be non-negative")
        if self.transport_retries < 0:
            raise ValueError("transport_retries must be non-negative")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.module_deadline_seconds <= 0:
            raise ValueError("module_deadline_seconds must be positive")
        if self.max_module_tokens <= 0:
            raise ValueError("max_module_tokens must be positive")
        if self.output_tokens + self.safety_margin_tokens >= self.context_window_tokens:
            raise ValueError(
                "output_tokens plus safety_margin_tokens must be smaller than "
                "context_window_tokens"
            )

    @property
    def input_budget_tokens(self) -> int:
        return (
            self.context_window_tokens - self.output_tokens - self.safety_margin_tokens
        )

    def cache_identity(self) -> dict[str, Any]:
        return {
            "output_tokens": self.output_tokens,
            "generation_kwargs": self.generation_kwargs,
        }


@dataclass(frozen=True)
class CompileOptions:
    """Per-run controls for resumable compilation.

    ``max_simulation_attempts`` limits candidates that reach executable pattern
    checking. ``max_semantic_attempts`` is retained as a compatibility alias.
    Static JSON/IR repair is bounded only by the module time/token budgets.
    """

    run_id: Optional[str] = None
    resume: bool = True
    max_simulation_attempts: Optional[int] = None
    max_semantic_attempts: Optional[int] = None
    module_deadline_seconds: Optional[float] = None


@dataclass(frozen=True)
class DiagnosticPacket:
    """Small, non-sensitive validation feedback sent to a Repair Agent."""

    category: str
    message: str
    location: str = ""
    related_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "location": self.location,
            "related_ids": list(self.related_ids),
            "details": self.details,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class CompiledModuleArtifact:
    """Validated output published by one atomic user-defined module task."""

    module_name: str
    semantic_hash: str
    interface_hash: str
    dependency_hashes: tuple[tuple[str, str], ...]
    patterns_checked: int
    module_dict: dict = field(repr=False)


@dataclass
class ModuleCompileReport:
    module_name: str
    # Total LLM generation calls, including syntax/IR repair calls.
    status: str = "queued"
    attempts: int = 0
    # Candidates that reached executable pattern simulation. max_retries limits this.
    simulation_attempts: int = 0
    # JSON/schema/SSA/width/interface repairs do not consume simulation attempts.
    static_repairs: int = 0
    compression_calls: int = 0
    transport_retries: int = 0
    duration_seconds: float = 0.0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    cache_hit: bool = False
    patterns_checked: int = 0
    cache_committed: bool = False
    resumed: bool = False
    failure_fingerprints: list[str] = field(default_factory=list)
    budget_seconds: float = 0.0
    budget_tokens: int = 0
    hierarchy_depth: int = 0
    direct_dependencies: list[str] = field(default_factory=list)
    dependency_hashes: dict[str, str] = field(default_factory=dict)
    artifact_hash: str = ""
    interface_hash: str = ""
    validation_stage: str = ""
    failure_origin: str = ""
    # Compatibility fields retained for one release. Modules are always atomic.
    plan_strategy: str = "hierarchy_node"
    planned_blocks: list[str] = field(default_factory=list)
    compression_events: list[str] = field(default_factory=list)
    error_category: Optional[str] = None
    error: Optional[str] = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    module_dict: Optional[dict] = field(default=None, repr=False)
    cache_key: Optional[str] = field(default=None, repr=False)

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass
class CompilationReport:
    modules: list[dict]
    module_reports: dict[str, ModuleCompileReport]
    duration_seconds: float
    llm_calls: int
    cache_hits: int
    design_error: Optional[str] = None
    run_id: Optional[str] = None
    resumed_modules: int = 0

    @property
    def success(self) -> bool:
        return self.design_error is None and all(
            report.success for report in self.module_reports.values()
        )

    @property
    def failures(self) -> dict[str, ModuleCompileReport]:
        return {
            name: report
            for name, report in self.module_reports.items()
            if not report.success
        }
