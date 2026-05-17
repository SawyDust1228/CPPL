"""MLIR/CIRCT code generation via pycde (circt Python bindings).

Uses pycde.circt.ir and pycde.circt.dialects.{hw, comb} to construct
a proper MLIR Module in-memory, then serializes it to text via str().
Optionally exports Verilog via circt.export_verilog().
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

import pycde.circt as circt
from pycde.circt.ir import (
    ArrayAttr,
    Context,
    FlatSymbolRefAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module as MLIRModule,
    Operation as MLIROperation,
    StringAttr,
    Type,
    Value,
)
from pycde.circt.dialects import comb, hw, seq

from ..ir.errors import CodegenError
from ..ir.models import (
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
from ..ir.infer import ValueInfo


def _iN(width: int) -> IntegerType:
    return IntegerType.get_signless(width)


class _ModuleEmitter:
    """Builds one hw.module from a JSON-IR Module using the pycde API."""

    def __init__(
        self,
        mod: Module,
        widths: Dict[str, ValueInfo],
        all_modules: Dict[str, Module],
    ) -> None:
        self.mod = mod
        self.widths = widths
        self._all_modules = all_modules
        self._env: Dict[str, Value] = {}  # JSON-IR id -> MLIR Value
        self._instance_counter = 0

    def _width(self, id_: str) -> int:
        info = self.widths.get(id_)
        if info is None:
            raise CodegenError(f"No inferred width for '{id_}'")
        return info.width

    def _val(self, id_: str) -> Value:
        v = self._env.get(id_)
        if v is None:
            raise CodegenError(f"No MLIR value for '{id_}'")
        return v

    def _fresh_instance_name(self, module_name: str) -> str:
        name = f"{module_name}_{self._instance_counter}"
        self._instance_counter += 1
        return name

    def _args_ready(self, op: Operation) -> bool:
        """Check whether all operand references are available in _env."""
        if isinstance(op, ConstantOp):
            return True
        elif isinstance(op, OutputOp):
            return all(ref in self._env for ref in op.args.values())
        elif isinstance(op, InstanceOp):
            return all(ref in self._env for ref in op.args.values())
        elif isinstance(op, RegOp):
            if op.args[0] not in self._env:
                return False
            if op.clock not in self._env:
                return False
            if op.reset and op.reset not in self._env:
                return False
            if op.enable and op.enable not in self._env:
                return False
            return True
        elif isinstance(op, MemOp):
            if op.clock not in self._env or op.reset not in self._env:
                return False
            for addr, enable in op.reads:
                if addr not in self._env or enable not in self._env:
                    return False
            for addr, data, enable in op.writes:
                if addr not in self._env or data not in self._env or enable not in self._env:
                    return False
            return True
        else:
            # UnaryOp, BinaryOp, MuxOp, CastOp, VariadicOp, ExtractOp
            return all(arg in self._env for arg in op.args)

    def emit(self) -> None:
        """Create the hw.module op at the current insertion point."""
        in_ports = [
            (name, _iN(pdef.width))
            for name, pdef in self.mod.ports.items()
            if pdef.dir == PortDir.INPUT
        ]
        out_ports = [
            (name, _iN(pdef.width))
            for name, pdef in self.mod.ports.items()
            if pdef.dir == PortDir.OUTPUT
        ]

        emitter = self  # capture for closure

        def body_builder(module_proxy):
            for name, _ in in_ports:
                emitter._env[name] = getattr(module_proxy, name)

            # Some IR producers reference register and instance results before
            # their defining ops.  Seed placeholders and replace them later.
            reg_placeholders: dict = {}  # op.id -> placeholder op
            for op in emitter.mod.body:
                if isinstance(op, RegOp):
                    w = emitter._width(op.id)
                    placeholder = MLIROperation.create(
                        "builtin.unrealized_conversion_cast",
                        results=[_iN(w)],
                    )
                    emitter._env[op.id] = placeholder.result
                    reg_placeholders[op.id] = placeholder

            inst_placeholders: dict = {}  # id -> placeholder op
            for op in emitter.mod.body:
                if isinstance(op, (InstanceOp, MemOp)):
                    for id_ in op.id:
                        w = emitter._width(id_)
                        placeholder = MLIROperation.create(
                            "builtin.unrealized_conversion_cast",
                            results=[_iN(w)],
                        )
                        emitter._env[id_] = placeholder.result
                        inst_placeholders[id_] = placeholder

            pending = [
                op for op in emitter.mod.body
                if not isinstance(op, (RegOp, OutputOp))
            ]
            while pending:
                still_pending = []
                progress = False
                for op in pending:
                    if emitter._args_ready(op):
                        emitter._emit_op(op)
                        if isinstance(op, (InstanceOp, MemOp)):
                            for id_ in op.id:
                                if id_ in inst_placeholders:
                                    old = inst_placeholders.pop(id_)
                                    actual = emitter._env[id_]
                                    old.result.replace_all_uses_with(actual)
                                    old.erase()
                        progress = True
                    else:
                        still_pending.append(op)
                if not progress and still_pending:
                    emitter._emit_op(still_pending[0])
                pending = still_pending

            for op in emitter.mod.body:
                if not isinstance(op, RegOp):
                    continue
                old_placeholder = reg_placeholders[op.id]
                emitter._emit_reg(op)
                actual_val = emitter._env[op.id]
                old_placeholder.result.replace_all_uses_with(actual_val)
                old_placeholder.erase()

            last = emitter.mod.body[-1]
            assert isinstance(last, OutputOp)
            return {
                pname: emitter._val(ref)
                for pname, ref in last.args.items()
            }

        hw.HWModuleOp(
            name=self.mod.name,
            input_ports=in_ports,
            output_ports=out_ports,
            body_builder=body_builder,
        )

    def _emit_op(self, op: Operation) -> None:
        if isinstance(op, ConstantOp):
            self._emit_constant(op)
        elif isinstance(op, UnaryOp):
            self._emit_unary(op)
        elif isinstance(op, BinaryOp):
            self._emit_binary(op)
        elif isinstance(op, VariadicOp):
            self._emit_variadic(op)
        elif isinstance(op, ExtractOp):
            self._emit_extract(op)
        elif isinstance(op, MuxOp):
            self._emit_mux(op)
        elif isinstance(op, CastOp):
            self._emit_cast(op)
        elif isinstance(op, RegOp):
            self._emit_reg(op)
        elif isinstance(op, MemOp):
            self._emit_mem(op)
        elif isinstance(op, InstanceOp):
            self._emit_instance(op)
        elif isinstance(op, OutputOp):
            pass  # handled by body_builder return

    def _emit_constant(self, op: ConstantOp) -> None:
        if isinstance(op.value, str):
            val = int(op.value, 0)
        else:
            val = op.value
        attr = IntegerAttr.get(_iN(op.width), val)
        result = hw.ConstantOp(attr).result
        self._env[op.id] = result

    def _emit_unary(self, op: UnaryOp) -> None:
        src = self._val(op.args[0])
        w = self._width(op.args[0])

        if op.op == "not":
            # ~x == x ^ all_ones
            allones = hw.ConstantOp(IntegerAttr.get(_iN(w), -1)).result
            self._env[op.id] = comb.xor([src, allones])
        elif op.op == "neg":
            # -x == ~x + 1 == (x ^ all_ones) + 1
            allones = hw.ConstantOp(IntegerAttr.get(_iN(w), -1)).result
            xored = comb.xor([src, allones])
            one = hw.ConstantOp(IntegerAttr.get(_iN(w), 1)).result
            self._env[op.id] = comb.add([xored, one])
        elif op.op == "or_reduce":
            # or_reduce(x) == x != 0
            zero = hw.ConstantOp(IntegerAttr.get(_iN(w), 0)).result
            pred = IntegerAttr.get(IntegerType.get_signless(64), 1)  # ne
            self._env[op.id] = comb.ICmpOp(pred, src, zero).result
        elif op.op == "and_reduce":
            # and_reduce(x) == x == all_ones
            allones = hw.ConstantOp(IntegerAttr.get(_iN(w), -1)).result
            pred = IntegerAttr.get(IntegerType.get_signless(64), 0)  # eq
            self._env[op.id] = comb.ICmpOp(pred, src, allones).result
        elif op.op == "xor_reduce":
            self._env[op.id] = comb.parity(src)
        elif op.op == "reverse":
            self._env[op.id] = comb.reverse(src)

    def _emit_binary(self, op: BinaryOp) -> None:
        lhs = self._val(op.args[0])
        rhs = self._val(op.args[1])

        if op.op in COMPARE_OPS:
            _CMP_PRED = {
                "eq": 0,
                "ne": 1,
                "lt_s": 2,
                "le_s": 3,
                "gt_s": 4,
                "ge_s": 5,
                "lt_u": 6,
                "le_u": 7,
                "gt_u": 8,
                "ge_u": 9,
            }
            pred_val = _CMP_PRED[op.op]
            pred = IntegerAttr.get(IntegerType.get_signless(64), pred_val)
            self._env[op.id] = comb.ICmpOp(pred, lhs, rhs).result
            return

        # Dispatch to the correct comb helper
        _BINARY_DISPATCH = {
            "add": lambda: comb.add([lhs, rhs]),
            "sub": lambda: comb.sub(lhs, rhs),
            "mul": lambda: comb.mul([lhs, rhs]),
            "div": lambda: comb.divu(lhs, rhs),
            "div_s": lambda: comb.divs(lhs, rhs),
            "mod_u": lambda: comb.modu(lhs, rhs),
            "mod_s": lambda: comb.mods(lhs, rhs),
            "and": lambda: comb.and_([lhs, rhs]),
            "or": lambda: comb.or_([lhs, rhs]),
            "xor": lambda: comb.xor([lhs, rhs]),
            "shl": lambda: comb.shl(lhs, rhs),
            "shr_u": lambda: comb.shru(lhs, rhs),
            "shr_s": lambda: comb.shrs(lhs, rhs),
        }
        fn = _BINARY_DISPATCH.get(op.op)
        if fn is None:
            raise CodegenError(f"No CIRCT mapping for binary op '{op.op}'")
        self._env[op.id] = fn()

    def _emit_variadic(self, op: VariadicOp) -> None:
        args = [self._val(a) for a in op.args]
        self._env[op.id] = comb.concat(args)

    def _emit_extract(self, op: ExtractOp) -> None:
        src = self._val(op.args[0])
        result_type = _iN(op.width)
        self._env[op.id] = comb.extract(result_type, src, low_bit=op.lowBit)

    def _emit_mux(self, op: MuxOp) -> None:
        sel = self._val(op.args[0])
        true_val = self._val(op.args[1])
        false_val = self._val(op.args[2])
        self._env[op.id] = comb.MuxOp(sel, true_val, false_val).result

    def _emit_cast(self, op: CastOp) -> None:
        src = self._val(op.args[0])
        src_w = self._width(op.args[0])
        target_w = op.width
        extend_bits = target_w - src_w

        if extend_bits == 0:
            self._env[op.id] = src
            return

        if op.op == "sext":
            # Extract sign bit, replicate it, then concat
            sign_bit = comb.extract(_iN(1), src, low_bit=src_w - 1)
            extension = comb.replicate(_iN(extend_bits), sign_bit)
            self._env[op.id] = comb.concat([extension, src])
        elif op.op == "zext":
            # Prepend zeros
            zero = hw.ConstantOp(IntegerAttr.get(_iN(extend_bits), 0)).result
            self._env[op.id] = comb.concat([zero, src])

    def _emit_reg(self, op: RegOp) -> None:
        data = self._val(op.args[0])
        clk = seq.to_clock(self._val(op.clock))
        data_w = self._width(op.args[0])

        kwargs: dict = {"name": op.id}

        if op.reset:
            kwargs["reset"] = self._val(op.reset)
            if isinstance(op.resetValue, str):
                rv = int(op.resetValue, 0)
            else:
                rv = op.resetValue
            kwargs["reset_value"] = hw.ConstantOp(
                IntegerAttr.get(_iN(data_w), rv)
            ).result

        if op.enable:
            self._env[op.id] = seq.compreg_ce(
                data, clk, self._val(op.enable), **kwargs
            )
        else:
            self._env[op.id] = seq.compreg(data, clk, **kwargs)

    def _emit_mem(self, op: MemOp) -> None:
        element_type = _iN(op.width)
        hlmem_type = Type.parse(f"!seq.hlmem<{op.depth}x{element_type}>")

        clk = seq.to_clock(self._val(op.clock))
        rst = self._val(op.reset)

        # Use the first output id as a reasonable name for the memory
        mem_name = op.id[0] if op.id else "mem"
        mem = seq.HLMemOp(
            handle=hlmem_type, clk=clk, rst=rst, name=mem_name,
        ).result

        # Create read ports (combinational, latency=0)
        for i, (addr_ref, enable_ref) in enumerate(op.reads):
            addr = self._val(addr_ref)
            enable = self._val(enable_ref)
            rdata = seq.ReadPortOp(
                readData=element_type, memory=mem,
                addresses=[addr], latency=0, rdEn=enable,
            ).result
            self._env[op.id[i]] = rdata

        # Create write ports (synchronous, latency=1)
        for addr_ref, data_ref, enable_ref in op.writes:
            addr = self._val(addr_ref)
            data = self._val(data_ref)
            enable = self._val(enable_ref)
            seq.WritePortOp(
                memory=mem, addresses=[addr],
                inData=data, wrEn=enable, latency=1,
            )

    def _emit_instance(self, op: InstanceOp) -> None:
        inst_name = self._fresh_instance_name(op.module)

        # Collect input values in the order they appear in op.args
        input_values = [self._val(ref) for ref in op.args.values()]

        # Build result types from widths of output ids
        result_types = [_iN(self._width(id_)) for id_ in op.id]

        # Build name arrays
        arg_names = ArrayAttr.get([StringAttr.get(k) for k in op.args.keys()])

        # resultNames must match the child module's output port names
        child_mod = self._all_modules[op.module]
        child_out_names = [
            name for name, pdef in child_mod.ports.items()
            if pdef.dir == PortDir.OUTPUT
        ]
        result_names = ArrayAttr.get(
            [StringAttr.get(n) for n in child_out_names]
        )

        inst = hw.InstanceOp(
            result_types,
            inst_name,
            FlatSymbolRefAttr.get(op.module),
            input_values,
            argNames=arg_names,
            resultNames=result_names,
            parameters=ArrayAttr.get([]),
        )

        # Map each output id to its corresponding result
        for i, id_ in enumerate(op.id):
            self._env[id_] = inst.results[i]


def _resolve_top_modules(
    modules: List[Module],
    top: str,
) -> List[Module]:
    """Return *top* and its transitive dependencies in dependency-first order."""
    by_name = {m.name: m for m in modules}
    if top not in by_name:
        raise CodegenError(f"Top module '{top}' not found in design")

    ordered: List[Module] = []
    visited: set[str] = set()

    def _visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        mod = by_name[name]
        for op in mod.body:
            if isinstance(op, InstanceOp):
                _visit(op.module)
        ordered.append(mod)

    _visit(top)
    return ordered


def _build_mlir_module(
    modules: List[Module],
    all_widths: Dict[str, Dict[str, ValueInfo]],
    ctx: Context,
) -> MLIRModule:
    """Build an in-memory MLIR Module from parsed design modules.

    Must be called inside an active ``Context`` / ``Location`` scope.
    Returns the ``MLIRModule`` so callers can serialize or further process it.
    """
    mlir_module = MLIRModule.create()
    all_modules = {m.name: m for m in modules}
    with InsertionPoint(mlir_module.body):
        for mod in modules:
            widths = all_widths[mod.name]
            emitter = _ModuleEmitter(mod, widths, all_modules)
            emitter.emit()
    return mlir_module


def generate_mlir(
    modules: List[Module],
    all_widths: Dict[str, Dict[str, ValueInfo]],
    top: Optional[str] = None,
) -> str:
    """Generate complete MLIR text for a design.

    Creates a proper MLIR Module using pycde/circt Python bindings,
    then returns its textual representation.

    If *top* is given, only that module and its transitive dependencies
    are included.
    """
    if top is not None:
        modules = _resolve_top_modules(modules, top)

    with Context() as ctx, Location.unknown():
        circt.register_dialects(ctx)
        mlir_module = _build_mlir_module(modules, all_widths, ctx)

        buf = io.StringIO()
        mlir_module.operation.print(file=buf, assume_verified=True)
        return buf.getvalue()


def generate_verilog(
    modules: List[Module],
    all_widths: Dict[str, Dict[str, ValueInfo]],
    top: Optional[str] = None,
    optimize: bool = True,
) -> str:
    """Generate Verilog text for a design.

    Builds an MLIR Module, lowers ``seq`` dialect ops to SystemVerilog
    constructs, optionally runs cleanup/optimization passes, then exports
    Verilog using ``circt.export_verilog()``.

    If *top* is given, only that module and its transitive dependencies
    are included.  If *optimize* is true, run CIRCT/MLIR canonicalization
    and Verilog prettification passes before export.
    """
    from pycde.circt import passmanager

    if top is not None:
        modules = _resolve_top_modules(modules, top)

    with Context() as ctx, Location.unknown():
        circt.register_dialects(ctx)
        mlir_module = _build_mlir_module(modules, all_widths, ctx)

        pipeline = ["hw.module(lower-seq-hlmem)", "lower-seq-to-sv"]
        if optimize:
            pipeline.extend([
                "canonicalize",
                "cse",
                "hw.module(prettify-verilog)",
                "hw.module(hw-cleanup)",
            ])

        pm = passmanager.PassManager.parse(
            "builtin.module(" + ",".join(pipeline) + ")"
        )
        try:
            pm.run(mlir_module.operation)
        except Exception as exc:
            raise CodegenError(str(exc)) from exc

        buf = io.StringIO()
        circt.export_verilog(mlir_module, buf)
        return buf.getvalue()
