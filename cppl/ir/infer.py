"""Bit-width inference and type checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .errors import WidthError
from .models import (
    BinaryOp,
    CastOp,
    COMPARE_OPS,
    ConstantOp,
    ExtractOp,
    InstanceOp,
    MemOp,
    Module,
    MuxOp,
    Operation,
    OutputOp,
    PortDir,
    REDUCE_OPS,
    RegOp,
    UnaryOp,
    VariadicOp,
)


@dataclass
class ValueInfo:
    width: int
    source: str  # human-readable origin description


def infer_widths(modules: List[Module]) -> Dict[str, Dict[str, ValueInfo]]:
    """Infer widths for all values in all modules.

    Returns {module_name: {value_id: ValueInfo}}.
    Raises WidthError on mismatches.
    """
    module_map: Dict[str, Module] = {m.name: m for m in modules}
    result: Dict[str, Dict[str, ValueInfo]] = {}
    for mod in modules:
        result[mod.name] = _infer_module(mod, module_map)
    return result


def _infer_module(
    mod: Module, module_map: Dict[str, Module]
) -> Dict[str, ValueInfo]:
    ctx = f"Module '{mod.name}'"
    env: Dict[str, ValueInfo] = {}

    # Seed with input ports
    for pname, pdef in mod.ports.items():
        if pdef.dir == PortDir.INPUT:
            env[pname] = ValueInfo(width=pdef.width, source=f"input port '{pname}'")

    # Pre-register RegOps that have an explicit width annotation.
    # This breaks inference cycles (e.g. pc_reg -> npc -> pc_reg).
    for i, op in enumerate(mod.body):
        if isinstance(op, RegOp) and op.width > 0:
            env[op.id] = ValueInfo(
                width=op.width,
                source=f"{ctx} body[{i}] reg (explicit width)",
            )

    # Pre-register InstanceOp outputs.  Output widths are determined
    # entirely by the target module's output port definitions, not by
    # the instance's input values.  Registering them early allows other
    # ops (and other instances) to reference them immediately.
    for i, op in enumerate(mod.body):
        if isinstance(op, InstanceOp):
            target = module_map.get(op.module)
            if target is not None:
                target_outputs = [
                    (name, pdef) for name, pdef in target.ports.items()
                    if pdef.dir == PortDir.OUTPUT
                ]
                loc = f"{ctx} body[{i}]"
                for id_name, (port_name, pdef) in zip(op.id, target_outputs):
                    env[id_name] = ValueInfo(
                        width=pdef.width,
                        source=f"{loc} instance '{op.module}'.{port_name}",
                    )

    # Pre-register MemOp outputs.  Each read port output has width =
    # op.width (element width), known statically.
    for i, op in enumerate(mod.body):
        if isinstance(op, MemOp):
            loc = f"{ctx} body[{i}]"
            for id_name in op.id:
                env[id_name] = ValueInfo(
                    width=op.width,
                    source=f"{loc} mem read port",
                )

    # Iterative fixpoint: keep processing ops whose dependencies are
    # resolved until all ops are inferred.  This supports forward
    # references (use before definition) which is natural for hardware
    # where all values in a module exist simultaneously.
    pending = [(i, op) for i, op in enumerate(mod.body)]

    while pending:
        still_pending = []
        progress = False
        for i, op in pending:
            loc = f"{ctx} body[{i}]"
            if _can_infer(op, env):
                _infer_op(op, env, loc, mod, module_map)
                progress = True
            else:
                still_pending.append((i, op))
        if not progress and still_pending:
            # No progress — force-infer the first pending op to surface
            # a proper error message about the missing dependency.
            i, op = still_pending[0]
            loc = f"{ctx} body[{i}]"
            _infer_op(op, env, loc, mod, module_map)
        pending = still_pending

    return env


def _can_infer(op: Operation, env: Dict[str, ValueInfo]) -> bool:
    """Check whether all dependencies of *op* are already resolved in *env*."""
    if isinstance(op, ConstantOp):
        return True  # no dependencies
    elif isinstance(op, OutputOp):
        return all(ref in env for ref in op.args.values())
    elif isinstance(op, InstanceOp):
        return all(ref in env for ref in op.args.values())
    elif isinstance(op, RegOp):
        if not all(arg in env for arg in op.args):
            return False
        if op.clock not in env:
            return False
        if op.reset and op.reset not in env:
            return False
        if op.enable and op.enable not in env:
            return False
        return True
    elif isinstance(op, MemOp):
        if op.clock not in env or op.reset not in env:
            return False
        for addr, enable in op.reads:
            if addr not in env or enable not in env:
                return False
        for addr, data, enable in op.writes:
            if addr not in env or data not in env or enable not in env:
                return False
        return True
    else:
        # UnaryOp, BinaryOp, MuxOp, CastOp, VariadicOp, ExtractOp
        return all(arg in env for arg in op.args)


def _width_of(name: str, env: Dict[str, ValueInfo], loc: str) -> int:
    info = env.get(name)
    if info is None:
        raise WidthError(f"{loc}: unknown value '{name}' during width inference")
    return info.width


def _infer_op(
    op: Operation,
    env: Dict[str, ValueInfo],
    loc: str,
    mod: Module,
    module_map: Dict[str, Module],
) -> None:
    if isinstance(op, ConstantOp):
        env[op.id] = ValueInfo(width=op.width, source=f"{loc} constant")

    elif isinstance(op, UnaryOp):
        w = _width_of(op.args[0], env, loc)
        if op.op in REDUCE_OPS:
            env[op.id] = ValueInfo(width=1, source=f"{loc} {op.op}")
        else:
            env[op.id] = ValueInfo(width=w, source=f"{loc} {op.op}")

    elif isinstance(op, BinaryOp):
        w0 = _width_of(op.args[0], env, loc)
        w1 = _width_of(op.args[1], env, loc)
        if w0 != w1:
            raise WidthError(
                f"{loc}: operands of '{op.op}' have different widths "
                f"({w0} vs {w1})"
            )
        if op.op in COMPARE_OPS:
            env[op.id] = ValueInfo(width=1, source=f"{loc} {op.op}")
        else:
            env[op.id] = ValueInfo(width=w0, source=f"{loc} {op.op}")

    elif isinstance(op, MuxOp):
        sel_w = _width_of(op.args[0], env, loc)
        if sel_w != 1:
            raise WidthError(
                f"{loc}: mux selector '{op.args[0]}' must be 1-bit, got {sel_w}"
            )
        w_true = _width_of(op.args[1], env, loc)
        w_false = _width_of(op.args[2], env, loc)
        if w_true != w_false:
            raise WidthError(
                f"{loc}: mux true/false operands have different widths "
                f"({w_true} vs {w_false})"
            )
        env[op.id] = ValueInfo(width=w_true, source=f"{loc} mux")

    elif isinstance(op, CastOp):
        src_w = _width_of(op.args[0], env, loc)
        if op.width < src_w:
            raise WidthError(
                f"{loc}: {op.op} target width {op.width} is less than "
                f"source width {src_w}"
            )
        env[op.id] = ValueInfo(width=op.width, source=f"{loc} {op.op}")

    elif isinstance(op, VariadicOp):
        total = 0
        for arg in op.args:
            total += _width_of(arg, env, loc)
        env[op.id] = ValueInfo(width=total, source=f"{loc} concat")

    elif isinstance(op, ExtractOp):
        src_w = _width_of(op.args[0], env, loc)
        if op.lowBit + op.width > src_w:
            raise WidthError(
                f"{loc}: extract lowBit({op.lowBit}) + width({op.width}) = "
                f"{op.lowBit + op.width} exceeds source width {src_w}"
            )
        env[op.id] = ValueInfo(width=op.width, source=f"{loc} extract")

    elif isinstance(op, RegOp):
        data_w = _width_of(op.args[0], env, loc)
        clk_w = _width_of(op.clock, env, loc)
        if clk_w != 1:
            raise WidthError(
                f"{loc}: reg clock '{op.clock}' must be 1-bit, got {clk_w}"
            )
        if op.reset:
            rst_w = _width_of(op.reset, env, loc)
            if rst_w != 1:
                raise WidthError(
                    f"{loc}: reg reset '{op.reset}' must be 1-bit, got {rst_w}"
                )
        if op.enable:
            en_w = _width_of(op.enable, env, loc)
            if en_w != 1:
                raise WidthError(
                    f"{loc}: reg enable '{op.enable}' must be 1-bit, got {en_w}"
                )
        if op.width > 0 and data_w != op.width:
            raise WidthError(
                f"{loc}: reg explicit width {op.width} doesn't match "
                f"data input width {data_w}"
            )
        env[op.id] = ValueInfo(width=data_w, source=f"{loc} reg")

    elif isinstance(op, MemOp):
        clk_w = _width_of(op.clock, env, loc)
        if clk_w != 1:
            raise WidthError(
                f"{loc}: mem clock '{op.clock}' must be 1-bit, got {clk_w}"
            )
        rst_w = _width_of(op.reset, env, loc)
        if rst_w != 1:
            raise WidthError(
                f"{loc}: mem reset '{op.reset}' must be 1-bit, got {rst_w}"
            )
        for i, (addr, enable) in enumerate(op.reads):
            en_w = _width_of(enable, env, loc)
            if en_w != 1:
                raise WidthError(
                    f"{loc}: mem reads[{i}].enable '{enable}' must be 1-bit, got {en_w}"
                )
        for i, (addr, data, enable) in enumerate(op.writes):
            en_w = _width_of(enable, env, loc)
            if en_w != 1:
                raise WidthError(
                    f"{loc}: mem writes[{i}].enable '{enable}' must be 1-bit, got {en_w}"
                )
            data_w = _width_of(data, env, loc)
            if data_w != op.width:
                raise WidthError(
                    f"{loc}: mem writes[{i}].data '{data}' has width {data_w} "
                    f"but element width is {op.width}"
                )
        for id_name in op.id:
            env[id_name] = ValueInfo(
                width=op.width,
                source=f"{loc} mem read port",
            )

    elif isinstance(op, InstanceOp):
        target = module_map.get(op.module)
        if target is None:
            raise WidthError(f"{loc}: unknown module '{op.module}'")

        # Check input widths match target ports
        for port_name, ref in op.args.items():
            ref_w = _width_of(ref, env, loc)
            target_port = target.ports.get(port_name)
            if target_port is None:
                continue  # validator catches this
            if ref_w != target_port.width:
                raise WidthError(
                    f"{loc}: instance input '{port_name}' has width {ref_w} "
                    f"but module '{op.module}' expects {target_port.width}"
                )

        # Register output widths
        target_outputs = [
            (name, pdef) for name, pdef in target.ports.items()
            if pdef.dir == PortDir.OUTPUT
        ]
        for id_name, (port_name, pdef) in zip(op.id, target_outputs):
            env[id_name] = ValueInfo(
                width=pdef.width,
                source=f"{loc} instance '{op.module}'.{port_name}",
            )

    elif isinstance(op, OutputOp):
        for port_name, ref in op.args.items():
            ref_w = _width_of(ref, env, loc)
            port_def = mod.ports.get(port_name)
            if port_def is None:
                continue  # validator catches this
            if ref_w != port_def.width:
                raise WidthError(
                    f"{loc}: output port '{port_name}' expects width "
                    f"{port_def.width} but got {ref_w}"
                )
