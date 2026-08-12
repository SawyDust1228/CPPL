"""Pure-Python interpreter for CPPL JSON-IR.

The interpreter is intentionally a deterministic two-state simulator.  Every
value is represented as a width-limited integer, registers and uninitialized
memories start at zero, and sequential state changes only on rising clock
edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .errors import SimulationError
from .infer import ValueInfo, infer_widths
from .models import (
    COMPARE_OPS,
    BinaryOp,
    CastOp,
    ConstantOp,
    ExtractOp,
    InstanceOp,
    MemOp,
    Module,
    MuxOp,
    OutputOp,
    PortDir,
    RegOp,
    UnaryOp,
    VariadicOp,
)
from .parser import parse_design
from .validator import validate_design


_MAX_DELTA_CYCLES = 100


def bit_mask(width: int) -> int:
    return (1 << width) - 1


def truncate_value(value: int, width: int) -> int:
    return value & bit_mask(width)


def as_signed(value: int, width: int) -> int:
    value = truncate_value(value, width)
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def signed_divide(lhs: int, rhs: int) -> int:
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) != (rhs < 0) else quotient


def parse_integer_literal(value: Union[int, str]) -> int:
    return int(value, 0) if isinstance(value, str) else value


@dataclass
class _MemoryState:
    op: MemOp
    values: List[int]
    public_name: str


@dataclass
class _InstanceState:
    module: Module
    path: str
    inputs: Dict[str, int]
    registers: Dict[int, int] = field(default_factory=dict)
    memories: Dict[int, _MemoryState] = field(default_factory=dict)
    children: Dict[int, "_InstanceState"] = field(default_factory=dict)
    child_names: Dict[str, "_InstanceState"] = field(default_factory=dict)
    previous_clocks: Dict[int, int] = field(default_factory=dict)
    env: Dict[str, int] = field(default_factory=dict)
    outputs: Dict[str, int] = field(default_factory=dict)


class Interpreter:
    """Execute a validated CPPL IR design without lowering it to RTL.

    Args:
        modules: Parsed :class:`Module` objects comprising the design.
        top: Top module name.  If omitted, the unique uninstantiated module is
            selected.
        widths: Optional precomputed result of :func:`infer_widths`.
        base_dir: Directory used to resolve memory ``initFile`` paths.
    """

    def __init__(
        self,
        modules: Sequence[Module],
        top: Optional[str] = None,
        widths: Optional[Dict[str, Dict[str, ValueInfo]]] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.modules = list(modules)
        validate_design(self.modules)
        self.widths = widths if widths is not None else infer_widths(self.modules)
        self._modules_by_name = {module.name: module for module in self.modules}
        self.check_recursive_instances()
        self.top = self.select_top(top)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self._initializing = False
        self._root = self.build_instance_state(
            self._modules_by_name[self.top], self.top
        )
        self.reset_state()

    @classmethod
    def from_json(
        cls,
        raw: object,
        top: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> "Interpreter":
        """Parse, validate, and create an interpreter from JSON-compatible IR."""
        return cls(parse_design(raw), top=top, base_dir=base_dir)

    def evaluate(self, inputs: Optional[Mapping[str, int]] = None) -> Dict[str, int]:
        """Apply top-level inputs, process clock edges, and return outputs.

        ``inputs`` may be partial; omitted ports retain their previous values.
        A sequential operation updates only when its clock changes from 0 to 1.
        """
        if inputs is not None:
            self.apply_inputs(inputs)

        for _ in range(_MAX_DELTA_CYCLES):
            self.evaluate_instance(self._root, self._root.inputs)
            edge_updates = self.collect_edge_updates(self._root)
            self.record_clock_levels(self._root)
            if not edge_updates:
                return dict(self._root.outputs)
            for update in edge_updates:
                update()

        raise SimulationError(
            f"Simulation did not settle after {_MAX_DELTA_CYCLES} delta cycles"
        )

    def peek_outputs(self) -> Dict[str, int]:
        """Return the currently settled top-level outputs without changing state."""
        return dict(self._root.outputs)

    def peek(self, path: str) -> int:
        """Read a port, SSA value, or register using a dotted hierarchy path."""
        instance, value_name = self.resolve_value_path(path)
        if value_name not in instance.env:
            raise SimulationError(f"Unknown value path '{path}'")
        return instance.env[value_name]

    def peek_memory(self, path: str) -> tuple[int, ...]:
        """Return an immutable snapshot of a memory at a dotted hierarchy path."""
        instance, memory_name = self.resolve_value_path(path)
        matches = [
            memory
            for memory in instance.memories.values()
            if memory.public_name == memory_name
        ]
        if not matches:
            raise SimulationError(f"Unknown memory path '{path}'")
        if len(matches) > 1:
            raise SimulationError(f"Ambiguous memory path '{path}'")
        return tuple(matches[0].values)

    def reset_state(self) -> None:
        """Restore inputs and sequential state to their initial values."""
        self.reset_instance(self._root)
        self._initializing = True
        try:
            self.evaluate_instance(self._root, self._root.inputs)
            self.record_clock_levels(self._root)
        finally:
            self._initializing = False

    def select_top(self, requested: Optional[str]) -> str:
        if requested is not None:
            if requested not in self._modules_by_name:
                raise SimulationError(f"Top module '{requested}' not found in design")
            return requested

        instantiated = {
            op.module
            for module in self.modules
            for op in module.body
            if isinstance(op, InstanceOp)
        }
        roots = [
            module.name for module in self.modules if module.name not in instantiated
        ]
        if len(roots) == 1:
            return roots[0]
        if not roots:
            raise SimulationError("Design has no root module; specify 'top' explicitly")
        raise SimulationError(
            f"Design has multiple root modules {roots}; specify 'top' explicitly"
        )

    def check_recursive_instances(self) -> None:
        visiting: List[str] = []
        visited: set[str] = set()

        def visit(module_name: str) -> None:
            if module_name in visiting:
                start = visiting.index(module_name)
                cycle = visiting[start:] + [module_name]
                raise SimulationError(
                    "Recursive instance hierarchy: " + " -> ".join(cycle)
                )
            if module_name in visited:
                return
            visiting.append(module_name)
            module = self._modules_by_name[module_name]
            for op in module.body:
                if isinstance(op, InstanceOp):
                    visit(op.module)
            visiting.pop()
            visited.add(module_name)

        for module in self.modules:
            visit(module.name)

    def build_instance_state(self, module: Module, path: str) -> _InstanceState:
        inputs = {
            name: 0 for name, port in module.ports.items() if port.dir == PortDir.INPUT
        }
        instance = _InstanceState(module=module, path=path, inputs=inputs)
        unnamed_index = 0

        for index, op in enumerate(module.body):
            if isinstance(op, RegOp):
                instance.registers[index] = 0
                instance.previous_clocks[index] = 0
            elif isinstance(op, MemOp):
                public_name = op.name or (op.id[0] if op.id else "mem")
                values = self.load_memory(op, path)
                instance.memories[index] = _MemoryState(op, values, public_name)
                instance.previous_clocks[index] = 0
            elif isinstance(op, InstanceOp):
                child_name = op.name
                if not child_name:
                    child_name = f"{op.module.lower()}_{unnamed_index}"
                    unnamed_index += 1
                if child_name in instance.child_names:
                    raise SimulationError(
                        f"Instance path '{path}' has duplicate child name '{child_name}'"
                    )
                child = self.build_instance_state(
                    self._modules_by_name[op.module], f"{path}.{child_name}"
                )
                instance.children[index] = child
                instance.child_names[child_name] = child
        return instance

    def load_memory(self, op: MemOp, instance_path: str) -> List[int]:
        values = [0] * op.depth
        if not op.initFile:
            return values

        init_path = Path(op.initFile)
        if not init_path.is_absolute():
            init_path = self.base_dir / init_path
        try:
            text = init_path.read_text()
        except OSError as exc:
            raise SimulationError(
                f"Memory '{instance_path}.{op.name or (op.id[0] if op.id else 'mem')}' "
                f"cannot read initFile '{init_path}': {exc}"
            ) from exc

        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        tokens: List[str] = []
        for line in text.splitlines():
            line = line.split("//", 1)[0].split("#", 1)[0]
            tokens.extend(line.split())

        address = 0
        base = 2 if op.initFormat == "bin" else 16
        for token in tokens:
            if token.startswith("@"):
                try:
                    address = int(token[1:].replace("_", ""), 16)
                except ValueError as exc:
                    raise SimulationError(
                        f"Invalid memory address token '{token}' in '{init_path}'"
                    ) from exc
                continue
            if address < 0 or address >= op.depth:
                raise SimulationError(
                    f"Memory initializer '{init_path}' writes address {address}, "
                    f"outside depth {op.depth}"
                )
            try:
                value = int(token.replace("_", ""), base)
            except ValueError as exc:
                raise SimulationError(
                    f"Invalid {op.initFormat} memory value '{token}' in '{init_path}'"
                ) from exc
            values[address] = truncate_value(value, op.width)
            address += 1
        return values

    def reset_instance(self, instance: _InstanceState) -> None:
        for name in instance.inputs:
            instance.inputs[name] = 0
        for index in instance.registers:
            instance.registers[index] = 0
            instance.previous_clocks[index] = 0
        for index, memory in instance.memories.items():
            memory.values[:] = self.load_memory(memory.op, instance.path)
            instance.previous_clocks[index] = 0
        instance.env.clear()
        instance.outputs.clear()
        for child in instance.children.values():
            self.reset_instance(child)

    def apply_inputs(self, inputs: Mapping[str, int]) -> None:
        normalized: Dict[str, int] = {}
        for name, value in inputs.items():
            port = self._root.module.ports.get(name)
            if port is None or port.dir != PortDir.INPUT:
                raise SimulationError(
                    f"Module '{self.top}' has no input port named '{name}'"
                )
            if not isinstance(value, int):
                raise SimulationError(f"Input '{name}' must be an integer")
            normalized[name] = truncate_value(value, port.width)
        self._root.inputs.update(normalized)

    def evaluate_instance(
        self,
        instance: _InstanceState,
        input_values: Mapping[str, int],
        *,
        allow_partial: bool = False,
    ) -> Dict[str, int]:
        module = instance.module
        widths = self.widths[module.name]
        env = {
            name: truncate_value(input_values[name], port.width)
            for name, port in module.ports.items()
            if port.dir == PortDir.INPUT and name in input_values
        }

        pending: List[tuple[int, object]] = list(enumerate(module.body[:-1]))
        while pending:
            next_pending: List[tuple[int, object]] = []
            progress = False
            for index, op in pending:
                if isinstance(op, ConstantOp):
                    env[op.id] = truncate_value(
                        parse_integer_literal(op.value), op.width
                    )
                elif isinstance(op, RegOp):
                    env[op.id] = instance.registers[index]
                elif isinstance(op, UnaryOp) and self.arguments_ready(op.args, env):
                    env[op.id] = self.evaluate_unary(op, env, widths)
                elif isinstance(op, BinaryOp) and self.arguments_ready(op.args, env):
                    env[op.id] = self.evaluate_binary(op, env, widths, instance.path)
                elif isinstance(op, VariadicOp) and self.arguments_ready(op.args, env):
                    env[op.id] = self.evaluate_concat(op, env, widths)
                elif isinstance(op, MuxOp) and self.arguments_ready(op.args, env):
                    env[op.id] = env[op.args[1]] if env[op.args[0]] else env[op.args[2]]
                elif isinstance(op, CastOp) and self.arguments_ready(op.args, env):
                    env[op.id] = self.evaluate_cast(op, env, widths)
                elif isinstance(op, ExtractOp) and self.arguments_ready(op.args, env):
                    env[op.id] = truncate_value(env[op.args[0]] >> op.lowBit, op.width)
                elif isinstance(op, MemOp) and self.memory_read_arguments_ready(
                    op, env
                ):
                    memory = instance.memories[index]
                    for output_id, (addr_ref, enable_ref) in zip(op.id, op.reads):
                        if env[enable_ref]:
                            address = env[addr_ref]
                            self.check_address(address, op, instance.path, "read")
                            env[output_id] = memory.values[address]
                        else:
                            env[output_id] = 0
                elif isinstance(op, InstanceOp):
                    child = instance.children[index]
                    child_inputs = {
                        name: env[ref] for name, ref in op.args.items() if ref in env
                    }
                    child.inputs.update(child_inputs)
                    child_outputs = self.evaluate_instance(
                        child, child_inputs, allow_partial=True
                    )
                    output_names = [
                        name
                        for name, port in child.module.ports.items()
                        if port.dir == PortDir.OUTPUT
                    ]
                    produced = False
                    for output_id, output_name in zip(op.id, output_names):
                        if output_name in child_outputs and output_id not in env:
                            env[output_id] = child_outputs[output_name]
                            produced = True
                    if not all(output_id in env for output_id in op.id):
                        next_pending.append((index, op))
                    if not produced:
                        continue
                else:
                    next_pending.append((index, op))
                    continue
                progress = True

            if not progress:
                if allow_partial:
                    break
                unresolved = ", ".join(
                    (
                        getattr(op, "id", getattr(op, "op", "operation"))
                        if isinstance(getattr(op, "id", ""), str)
                        else getattr(op, "op", "operation")
                    )
                    for _, op in next_pending
                )
                raise SimulationError(
                    f"Cannot resolve combinational values in '{instance.path}': {unresolved}"
                )
            pending = next_pending

        output_op = module.body[-1]
        assert isinstance(output_op, OutputOp)
        outputs = {name: env[ref] for name, ref in output_op.args.items() if ref in env}
        instance.env = env
        instance.outputs = outputs
        return outputs

    @staticmethod
    def arguments_ready(args: object, env: Mapping[str, int]) -> bool:
        return all(arg in env for arg in args)

    @staticmethod
    def memory_read_arguments_ready(op: MemOp, env: Mapping[str, int]) -> bool:
        return all(addr in env and enable in env for addr, enable in op.reads)

    def evaluate_unary(
        self, op: UnaryOp, env: Mapping[str, int], widths: Mapping[str, ValueInfo]
    ) -> int:
        value = env[op.args[0]]
        source_width = widths[op.args[0]].width
        result_width = widths[op.id].width
        if op.op == "not":
            result = ~value
        elif op.op == "neg":
            result = -value
        elif op.op == "reverse":
            result = 0
            for bit in range(source_width):
                result = (result << 1) | ((value >> bit) & 1)
        elif op.op == "or_reduce":
            result = int(value != 0)
        elif op.op == "and_reduce":
            result = int(value == bit_mask(source_width))
        elif op.op == "xor_reduce":
            result = value.bit_count() & 1
        else:  # pragma: no cover - parser prevents this
            raise SimulationError(f"Unsupported unary operation '{op.op}'")
        return truncate_value(result, result_width)

    def evaluate_binary(
        self,
        op: BinaryOp,
        env: Mapping[str, int],
        widths: Mapping[str, ValueInfo],
        instance_path: str,
    ) -> int:
        lhs = env[op.args[0]]
        rhs = env[op.args[1]]
        operand_width = widths[op.args[0]].width
        result_width = widths[op.id].width
        lhs_signed = as_signed(lhs, operand_width)
        rhs_signed = as_signed(rhs, operand_width)

        if op.op == "add":
            result = lhs + rhs
        elif op.op == "sub":
            result = lhs - rhs
        elif op.op == "mul":
            result = lhs * rhs
        elif op.op == "div":
            result = lhs // rhs if self.check_divisor(rhs, op.op, instance_path) else 0
        elif op.op == "div_s":
            result = (
                signed_divide(lhs_signed, rhs_signed)
                if self.check_divisor(rhs_signed, op.op, instance_path)
                else 0
            )
        elif op.op == "mod_u":
            result = lhs % rhs if self.check_divisor(rhs, op.op, instance_path) else 0
        elif op.op == "mod_s":
            if self.check_divisor(rhs_signed, op.op, instance_path):
                quotient = signed_divide(lhs_signed, rhs_signed)
                result = lhs_signed - quotient * rhs_signed
            else:
                result = 0
        elif op.op == "and":
            result = lhs & rhs
        elif op.op == "or":
            result = lhs | rhs
        elif op.op == "xor":
            result = lhs ^ rhs
        elif op.op == "shl":
            result = 0 if rhs >= operand_width else lhs << rhs
        elif op.op == "shr_u":
            result = 0 if rhs >= operand_width else lhs >> rhs
        elif op.op == "shr_s":
            result = (
                (-1 if lhs_signed < 0 else 0)
                if rhs >= operand_width
                else lhs_signed >> rhs
            )
        elif op.op in COMPARE_OPS:
            comparisons = {
                "eq": lhs == rhs,
                "ne": lhs != rhs,
                "lt_s": lhs_signed < rhs_signed,
                "le_s": lhs_signed <= rhs_signed,
                "gt_s": lhs_signed > rhs_signed,
                "ge_s": lhs_signed >= rhs_signed,
                "lt_u": lhs < rhs,
                "le_u": lhs <= rhs,
                "gt_u": lhs > rhs,
                "ge_u": lhs >= rhs,
            }
            result = int(comparisons[op.op])
        else:  # pragma: no cover - parser prevents this
            raise SimulationError(f"Unsupported binary operation '{op.op}'")
        return truncate_value(result, result_width)

    def check_divisor(self, value: int, op_name: str, instance_path: str) -> bool:
        if value == 0:
            if self._initializing:
                return False
            raise SimulationError(
                f"Division by zero in '{instance_path}' operation '{op_name}'"
            )
        return True

    @staticmethod
    def evaluate_concat(
        op: VariadicOp,
        env: Mapping[str, int],
        widths: Mapping[str, ValueInfo],
    ) -> int:
        result = 0
        for arg in op.args:
            result = (result << widths[arg].width) | env[arg]
        return truncate_value(result, widths[op.id].width)

    @staticmethod
    def evaluate_cast(
        op: CastOp,
        env: Mapping[str, int],
        widths: Mapping[str, ValueInfo],
    ) -> int:
        value = env[op.args[0]]
        if op.op == "sext":
            value = as_signed(value, widths[op.args[0]].width)
        return truncate_value(value, op.width)

    def collect_edge_updates(self, instance: _InstanceState) -> List[object]:
        updates: List[object] = []
        env = instance.env
        for index, op in enumerate(instance.module.body[:-1]):
            if not isinstance(op, (RegOp, MemOp)):
                continue
            current_clock = env[op.clock] & 1
            if instance.previous_clocks[index] or not current_clock:
                continue
            if isinstance(op, RegOp):
                if op.reset and env[op.reset]:
                    next_value = truncate_value(
                        parse_integer_literal(op.resetValue),
                        self.widths[instance.module.name][op.id].width,
                    )
                elif not op.enable or env[op.enable]:
                    next_value = env[op.args[0]]
                else:
                    next_value = instance.registers[index]

                def update_reg(
                    target: _InstanceState = instance,
                    target_index: int = index,
                    value: int = next_value,
                ) -> None:
                    target.registers[target_index] = value

                updates.append(update_reg)
            else:
                memory = instance.memories[index]
                if op.reset and env[op.reset]:

                    def reset_memory(target: _MemoryState = memory) -> None:
                        target.values[:] = [0] * target.op.depth

                    updates.append(reset_memory)
                    continue

                writes: Dict[int, int] = {}
                for addr_ref, data_ref, enable_ref in op.writes:
                    if not env[enable_ref]:
                        continue
                    address = env[addr_ref]
                    self.check_address(address, op, instance.path, "write")
                    if address in writes:
                        raise SimulationError(
                            f"Conflicting writes to address {address} in memory "
                            f"'{instance.path}.{memory.public_name}'"
                        )
                    writes[address] = truncate_value(env[data_ref], op.width)
                if writes:

                    def update_memory(
                        target: _MemoryState = memory,
                        values: Dict[int, int] = writes,
                    ) -> None:
                        for address, value in values.items():
                            target.values[address] = value

                    updates.append(update_memory)

        for child in instance.children.values():
            updates.extend(self.collect_edge_updates(child))
        return updates

    def record_clock_levels(self, instance: _InstanceState) -> None:
        for index, op in enumerate(instance.module.body[:-1]):
            if isinstance(op, (RegOp, MemOp)):
                instance.previous_clocks[index] = instance.env[op.clock] & 1
        for child in instance.children.values():
            self.record_clock_levels(child)

    @staticmethod
    def check_address(address: int, op: MemOp, path: str, action: str) -> None:
        if address < 0 or address >= op.depth:
            memory_name = op.name or (op.id[0] if op.id else "mem")
            raise SimulationError(
                f"Memory {action} address {address} is outside depth {op.depth} "
                f"for '{path}.{memory_name}'"
            )

    def resolve_value_path(self, path: str) -> tuple[_InstanceState, str]:
        if not isinstance(path, str) or not path:
            raise SimulationError("Path must be a non-empty string")
        parts = path.split(".")
        if parts[0] == self.top:
            parts = parts[1:]
        if not parts:
            raise SimulationError(f"Path '{path}' does not name a value")
        instance = self._root
        for child_name in parts[:-1]:
            child = instance.child_names.get(child_name)
            if child is None:
                raise SimulationError(f"Unknown instance path '{path}'")
            instance = child
        return instance, parts[-1]


__all__ = ["Interpreter", "SimulationError"]
