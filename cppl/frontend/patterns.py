"""Executable input/output patterns for ``@module`` specifications."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Union


def _freeze_values(values: Mapping[str, int], field_name: str) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping of port names to integers")
    result: dict[str, int] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"{field_name} port names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name}['{name}'] must be an integer")
        result[name] = value
    return MappingProxyType(result)


def _validate_name(name: str | None) -> None:
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise TypeError("Pattern name must be a non-empty string or None")


@dataclass(frozen=True)
class Step:
    """One input update and optional output checkpoint in a sequence."""

    inputs: Mapping[str, int] = field(default_factory=dict)
    outputs: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _freeze_values(self.inputs, "Step.inputs"))
        object.__setattr__(self, "outputs", _freeze_values(self.outputs, "Step.outputs"))


@dataclass(frozen=True)
class Case:
    """A single input/output example evaluated from reset state."""

    inputs: Mapping[str, int]
    outputs: Mapping[str, int]
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name)
        object.__setattr__(self, "inputs", _freeze_values(self.inputs, "Case.inputs"))
        object.__setattr__(self, "outputs", _freeze_values(self.outputs, "Case.outputs"))


@dataclass(frozen=True)
class Sequence:
    """A stateful series of input updates and output checkpoints."""

    steps: SequenceABC[Step]
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, SequenceABC):
            raise TypeError("Sequence steps must be a sequence of Step objects")
        steps = tuple(self.steps)
        if any(not isinstance(step, Step) for step in steps):
            raise TypeError("Sequence steps must contain only Step objects")
        object.__setattr__(self, "steps", steps)


Pattern = Union[Case, Sequence]


def normalize_patterns(patterns: object, ports: SequenceABC[object]) -> tuple[Pattern, ...]:
    """Validate patterns against module ports and return immutable values."""
    if patterns is None:
        return ()
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, SequenceABC):
        raise TypeError("module patterns must be a sequence of Case/Sequence objects")

    input_widths = {
        port.name: port.width for port in ports if port.direction == "input"
    }
    output_widths = {
        port.name: port.width for port in ports if port.direction == "output"
    }
    normalized: list[Pattern] = []
    names: set[str] = set()

    def normalize_values(
        values: Mapping[str, int],
        valid_ports: Mapping[str, int],
        location: str,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for port_name, value in values.items():
            width = valid_ports.get(port_name)
            if width is None:
                raise ValueError(f"{location} references unknown port '{port_name}'")
            result[port_name] = value & ((1 << width) - 1)
        return result

    for index, pattern in enumerate(patterns):
        if isinstance(pattern, Case):
            name = pattern.name.strip() if pattern.name else f"case[{index}]"
            if not pattern.outputs:
                raise ValueError(f"Pattern '{name}' must check at least one output")
            item: Pattern = Case(
                inputs=normalize_values(
                    pattern.inputs, input_widths, f"Pattern '{name}' inputs"
                ),
                outputs=normalize_values(
                    pattern.outputs, output_widths, f"Pattern '{name}' outputs"
                ),
                name=name,
            )
        elif isinstance(pattern, Sequence):
            name = pattern.name.strip() if pattern.name else f"sequence[{index}]"
            if not pattern.steps:
                raise ValueError(f"Pattern '{name}' must contain at least one step")
            steps = tuple(
                Step(
                    inputs=normalize_values(
                        step.inputs,
                        input_widths,
                        f"Pattern '{name}' step {step_index} inputs",
                    ),
                    outputs=normalize_values(
                        step.outputs,
                        output_widths,
                        f"Pattern '{name}' step {step_index} outputs",
                    ),
                )
                for step_index, step in enumerate(pattern.steps)
            )
            if not any(step.outputs for step in steps):
                raise ValueError(f"Pattern '{name}' must check at least one output")
            item = Sequence(steps=steps, name=name)
        else:
            raise TypeError(
                "module patterns must contain only Case or Sequence objects"
            )

        assert item.name is not None
        if item.name in names:
            raise ValueError(f"Duplicate pattern name '{item.name}'")
        names.add(item.name)
        normalized.append(item)

    return tuple(normalized)


def pattern_as_dict(pattern: Pattern) -> dict:
    """Return a stable JSON-compatible representation for prompts and caches."""
    if isinstance(pattern, Case):
        return {
            "kind": "case",
            "name": pattern.name,
            "inputs": dict(pattern.inputs),
            "outputs": dict(pattern.outputs),
        }
    return {
        "kind": "sequence",
        "name": pattern.name,
        "steps": [
            {"inputs": dict(step.inputs), "outputs": dict(step.outputs)}
            for step in pattern.steps
        ],
    }


__all__ = ["Case", "Sequence", "Step"]
