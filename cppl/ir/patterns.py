"""Execute frontend patterns against parsed CPPL IR modules."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..frontend.patterns import Case, Pattern, Sequence
from .errors import DiagnosticIssue, PatternMismatch, SimulationError
from .infer import ValueInfo
from .interpreter import Interpreter
from .models import Module


def format_mismatch_message(
    module_name: str,
    pattern_name: str,
    step_index: int,
    inputs: dict[str, int],
    expected: dict[str, int],
    actual: dict[str, int],
) -> str:
    mismatches = {
        port: {"expected": value, "actual": actual.get(port)}
        for port, value in expected.items()
        if actual.get(port) != value
    }
    return (
        f"Module '{module_name}' pattern '{pattern_name}' step {step_index} "
        f"failed: inputs={inputs}, expected={expected}, actual="
        f"{{{', '.join(f'{port!r}: {actual.get(port)!r}' for port in expected)}}}, "
        f"mismatches={mismatches}"
    )


def run_patterns(
    modules: Sequence[Module],
    top: str,
    patterns: Sequence[Pattern],
    *,
    widths: dict[str, dict[str, ValueInfo]] | None = None,
    base_dir: str | Path | None = None,
) -> int:
    """Run all patterns and return the number of successfully checked patterns."""
    if not patterns:
        return 0

    checked_patterns = 0
    issues: list[DiagnosticIssue] = []
    for pattern in patterns:
        assert pattern.name is not None
        fixtures = {fixture.path: fixture.words for fixture in pattern.fixtures}
        try:
            simulator = Interpreter(
                modules,
                top=top,
                widths=widths,
                base_dir=base_dir,
                memory_fixtures=fixtures,
            )
        except SimulationError as exc:
            issues.append(
                DiagnosticIssue(
                    code="pattern.initialization",
                    location=f"Module '{top}' pattern '{pattern.name}'",
                    message=f"could not initialize simulation: {exc}",
                    hint="Fix the referenced module, memory fixture, or simulator-visible state.",
                )
            )
            continue
        if isinstance(pattern, Case):
            steps = ((dict(pattern.inputs), dict(pattern.outputs), dict(pattern.probes)),)
        elif isinstance(pattern, Sequence):
            steps = tuple(
                (dict(step.inputs), dict(step.outputs), dict(step.probes))
                for step in pattern.steps
            )
        else:  # pragma: no cover - decorator normalization prevents this
            raise TypeError(f"Unsupported pattern type {type(pattern).__name__}")

        for step_index, (inputs, expected, probes) in enumerate(steps):
            try:
                actual = simulator.evaluate(inputs)
            except SimulationError as exc:
                issues.append(
                    DiagnosticIssue(
                        code="pattern.simulation",
                        location=f"Module '{top}' pattern '{pattern.name}' step {step_index}",
                        message=f"could not simulate with inputs={inputs}: {exc}",
                        actual=inputs,
                        hint="Fix the operation or reference reported by the simulator.",
                    )
                )
                break
            mismatches = {
                port: {"expected": value, "actual": actual.get(port)}
                for port, value in expected.items()
                if actual.get(port) != value
            }
            if mismatches:
                issues.append(
                    DiagnosticIssue(
                        code="pattern.output_mismatch",
                        location=f"Module '{top}' pattern '{pattern.name}' step {step_index}",
                        message=format_mismatch_message(
                            top, pattern.name, step_index, inputs, expected, actual
                        ),
                        expected=expected,
                        actual={port: actual.get(port) for port in expected},
                        hint="Trace the listed outputs back through their SSA producers and correct the logic.",
                    )
                )
            actual_probes: dict[str, int] = {}
            for path, expected_value in probes.items():
                try:
                    actual_probes[path] = simulator.peek_probe(path)
                except SimulationError as exc:
                    issues.append(
                        DiagnosticIssue(
                            code="pattern.probe_unreadable",
                            location=(
                                f"Module '{top}' pattern '{pattern.name}' "
                                f"step {step_index} probe '{path}'"
                            ),
                            message=f"could not read probe: {exc}",
                            actual=path,
                            hint="Use a valid hierarchical probe path to an instance, SSA value, or memory word.",
                        )
                    )
                    continue
                if actual_probes[path] != expected_value:
                    issues.append(
                        DiagnosticIssue(
                            code="pattern.probe_mismatch",
                            location=(
                                f"Module '{top}' pattern '{pattern.name}' "
                                f"step {step_index} probe '{path}'"
                            ),
                            message="probe value did not match",
                            expected=expected_value,
                            actual=actual_probes[path],
                            hint="Correct the internal state transition or combinational value feeding this probe.",
                        )
                    )
        checked_patterns += 1
    if issues:
        raise PatternMismatch(
            issues,
            summary=f"Module '{top}' pattern validation found {len(issues)} error(s)",
        )
    return checked_patterns


__all__ = ["run_patterns"]
