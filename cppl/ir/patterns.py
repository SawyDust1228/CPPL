"""Execute frontend patterns against parsed CPPL IR modules."""

from __future__ import annotations

from typing import Sequence

from ..frontend.patterns import Case, Pattern, Sequence
from .errors import PatternMismatch, SimulationError
from .infer import ValueInfo
from .interpreter import Interpreter
from .models import Module


def _mismatch_message(
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
) -> int:
    """Run all patterns and return the number of successfully checked patterns."""
    if not patterns:
        return 0

    try:
        simulator = Interpreter(modules, top=top, widths=widths)
    except SimulationError as exc:
        raise PatternMismatch(
            f"Module '{top}' pattern validation could not initialize simulation: {exc}"
        ) from exc
    checked_patterns = 0
    for pattern in patterns:
        assert pattern.name is not None
        simulator.reset_state()
        if isinstance(pattern, Case):
            steps = ((dict(pattern.inputs), dict(pattern.outputs)),)
        elif isinstance(pattern, Sequence):
            steps = tuple(
                (dict(step.inputs), dict(step.outputs)) for step in pattern.steps
            )
        else:  # pragma: no cover - decorator normalization prevents this
            raise TypeError(f"Unsupported pattern type {type(pattern).__name__}")

        for step_index, (inputs, expected) in enumerate(steps):
            try:
                actual = simulator.evaluate(inputs)
            except SimulationError as exc:
                raise PatternMismatch(
                    f"Module '{top}' pattern '{pattern.name}' step {step_index} "
                    f"could not simulate with inputs={inputs}: {exc}"
                ) from exc
            if not expected:
                continue
            if any(actual.get(port) != value for port, value in expected.items()):
                raise PatternMismatch(
                    _mismatch_message(
                        top, pattern.name, step_index, inputs, expected, actual
                    )
                )
        checked_patterns += 1
    return checked_patterns


__all__ = ["run_patterns"]
