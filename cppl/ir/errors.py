"""Exception hierarchy and structured diagnostics for CircuitPPL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DiagnosticIssue:
    """One precise, actionable compiler problem."""

    message: str
    location: str = ""
    code: str = ""
    expected: Any = None
    actual: Any = None
    hint: str = ""

    def format(self) -> str:
        text = f"{self.location}: {self.message}" if self.location else self.message
        details: list[str] = []
        if self.expected is not None:
            details.append(f"expected={self.expected!r}")
        if self.actual is not None:
            details.append(f"actual={self.actual!r}")
        if self.hint:
            details.append(f"fix={self.hint}")
        return text + ("; " + "; ".join(details) if details else "")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"message": self.message}
        for key, value in (
            ("location", self.location),
            ("code", self.code),
            ("expected", self.expected),
            ("actual", self.actual),
            ("hint", self.hint),
        ):
            if value not in (None, ""):
                result[key] = value
        return result


class CircuitPPLError(Exception):
    """Base exception for all CircuitPPL errors."""

    def __init__(
        self,
        message: str | Sequence[DiagnosticIssue],
        *,
        summary: str | None = None,
    ) -> None:
        if isinstance(message, str):
            self.issues: tuple[DiagnosticIssue, ...] = ()
            rendered = message
        else:
            self.issues = tuple(message)
            rendered = summary or f"{len(self.issues)} error(s) detected"
            if self.issues:
                rendered += "\n" + "\n".join(
                    f"  {index}. {issue.format()}"
                    for index, issue in enumerate(self.issues, 1)
                )
        super().__init__(rendered)


def issues_from_errors(errors: Iterable[BaseException]) -> list[DiagnosticIssue]:
    """Flatten structured errors and preserve legacy exception messages."""
    issues: list[DiagnosticIssue] = []
    for error in errors:
        nested = getattr(error, "issues", ())
        if nested:
            issues.extend(nested)
        else:
            issues.append(DiagnosticIssue(message=str(error)))
    return issues


class ParseError(CircuitPPLError):
    """JSON structure or field errors during parsing."""


class ValidationError(CircuitPPLError):
    """Semantic validation errors."""


class SSAError(ValidationError):
    """SSA violations: forward references or redefinitions."""


class WidthError(ValidationError):
    """Bit-width mismatches or inference failures."""


class CycleError(ValidationError):
    """Combinational cycle detected in the design."""


class CodegenError(CircuitPPLError):
    """Errors during MLIR/CIRCT code generation."""


class SimulationError(CircuitPPLError):
    """Errors while interpreting or simulating JSON-IR."""


class PatternMismatch(CircuitPPLError):
    """An interpreted module did not satisfy an executable pattern."""


class PatternDependencyMissing(CircuitPPLError):
    """Pattern validation cannot run because a real dependency IR is missing."""
