"""JSON parsing and structural validation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from .errors import ParseError
from .models import (
    BINARY_OPS,
    CAST_OPS,
    COMPARE_OPS,
    REDUCE_OPS,
    UNARY_OPS,
    VARIADIC_OPS,
    BinaryOp,
    CastOp,
    ConstantOp,
    ExtractOp,
    InstanceOp,
    MemOp,
    Module,
    MuxOp,
    Operation,
    OutputOp,
    PortDef,
    PortDir,
    RegOp,
    UnaryOp,
    VariadicOp,
)


def parse_design(raw: Union[str, Any]) -> List[Module]:
    """Parse a JSON string or already-decoded object into a list of Modules.

    Accepts either a JSON array of modules or a single module dict.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e
    else:
        data = raw

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ParseError("Design root must be a JSON array or object")

    modules: List[Module] = []
    for i, mod_raw in enumerate(data):
        if not isinstance(mod_raw, dict):
            raise ParseError(f"Module at index {i} must be a JSON object")
        modules.append(_parse_module(mod_raw, i))
    return modules


def _parse_module(raw: Dict[str, Any], index: int) -> Module:
    ctx = f"module[{index}]"

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError(f"{ctx}: 'name' must be a non-empty string")

    ctx = f"module '{name}'"

    ports_raw = raw.get("ports")
    if not isinstance(ports_raw, dict):
        raise ParseError(f"{ctx}: 'ports' must be an object")

    ports: Dict[str, PortDef] = {}
    for pname, pdef in ports_raw.items():
        if not isinstance(pdef, dict):
            raise ParseError(f"{ctx}: port '{pname}' must be an object")
        d = pdef.get("dir")
        if d not in ("input", "output"):
            raise ParseError(f"{ctx}: port '{pname}' dir must be 'input' or 'output'")
        w = pdef.get("width")
        if not isinstance(w, int) or w <= 0:
            raise ParseError(f"{ctx}: port '{pname}' width must be a positive integer")
        ports[pname] = PortDef(dir=PortDir(d), width=w)

    body_raw = raw.get("body")
    if not isinstance(body_raw, list):
        raise ParseError(f"{ctx}: 'body' must be an array")

    body: List[Operation] = []
    for j, op_raw in enumerate(body_raw):
        if isinstance(op_raw, dict) and "_comment" in op_raw and "op" not in op_raw:
            continue  # skip comment-only entries
        body.append(_parse_operation(op_raw, ctx, j))

    return Module(name=name, ports=ports, body=body)


def _parse_operation(raw: Dict[str, Any], ctx: str, index: int) -> Operation:
    if not isinstance(raw, dict):
        raise ParseError(f"{ctx} body[{index}]: operation must be an object")

    op = raw.get("op")
    if not isinstance(op, str):
        raise ParseError(f"{ctx} body[{index}]: 'op' must be a string")

    loc = f"{ctx} body[{index}] (op='{op}')"

    if op == "constant":
        return _parse_constant(raw, loc)
    elif op == "mux":
        return _parse_mux(raw, loc)
    elif op in UNARY_OPS or op in REDUCE_OPS:
        return _parse_unary(raw, loc, op)
    elif op in BINARY_OPS or op in COMPARE_OPS:
        return _parse_binary(raw, loc, op)
    elif op in VARIADIC_OPS:
        return _parse_variadic(raw, loc, op)
    elif op == "extract":
        return _parse_extract(raw, loc)
    elif op in CAST_OPS:
        return _parse_cast(raw, loc, op)
    elif op == "reg":
        return _parse_reg(raw, loc)
    elif op == "mem":
        return _parse_mem(raw, loc)
    elif op == "instance":
        return _parse_instance(raw, loc)
    elif op == "output":
        return _parse_output(raw, loc)
    else:
        raise ParseError(f"{loc}: unknown op '{op}'")


def _require_id(raw: Dict[str, Any], loc: str) -> str:
    id_ = raw.get("id")
    if not isinstance(id_, str) or not id_:
        raise ParseError(f"{loc}: 'id' must be a non-empty string")
    return id_


def _require_str_list(raw: Dict[str, Any], key: str, loc: str, *, exact: int = None) -> List[str]:
    val = raw.get(key)
    if not isinstance(val, list):
        raise ParseError(f"{loc}: '{key}' must be an array")
    for i, v in enumerate(val):
        if not isinstance(v, str):
            raise ParseError(f"{loc}: '{key}[{i}]' must be a string")
    if exact is not None and len(val) != exact:
        raise ParseError(f"{loc}: '{key}' must have exactly {exact} element(s), got {len(val)}")
    return val


def _require_str_dict(raw: Dict[str, Any], key: str, loc: str) -> Dict[str, str]:
    val = raw.get(key)
    if not isinstance(val, dict):
        raise ParseError(f"{loc}: '{key}' must be an object")
    for k, v in val.items():
        if not isinstance(v, str):
            raise ParseError(f"{loc}: '{key}.{k}' must be a string")
    return val


def _parse_constant(raw: Dict[str, Any], loc: str) -> ConstantOp:
    id_ = _require_id(raw, loc)
    value = raw.get("value")
    if not isinstance(value, (int, str)):
        raise ParseError(f"{loc}: 'value' must be an integer or string")
    width = raw.get("width")
    if not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return ConstantOp(id=id_, op="constant", value=value, width=width)


def _parse_unary(raw: Dict[str, Any], loc: str, op: str) -> UnaryOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=1)
    return UnaryOp(id=id_, op=op, args=args)


def _parse_binary(raw: Dict[str, Any], loc: str, op: str) -> BinaryOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=2)
    return BinaryOp(id=id_, op=op, args=args)


def _parse_variadic(raw: Dict[str, Any], loc: str, op: str) -> VariadicOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc)
    if len(args) < 1:
        raise ParseError(f"{loc}: 'args' must have at least 1 element")
    return VariadicOp(id=id_, op=op, args=args)


def _parse_extract(raw: Dict[str, Any], loc: str) -> ExtractOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=1)
    low = raw.get("lowBit")
    if not isinstance(low, int) or low < 0:
        raise ParseError(f"{loc}: 'lowBit' must be a non-negative integer")
    width = raw.get("width")
    if not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return ExtractOp(id=id_, op="extract", args=args, lowBit=low, width=width)


def _parse_mux(raw: Dict[str, Any], loc: str) -> MuxOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=3)
    return MuxOp(id=id_, op="mux", args=args)


def _parse_cast(raw: Dict[str, Any], loc: str, op: str) -> CastOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=1)
    width = raw.get("width")
    if not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return CastOp(id=id_, op=op, args=args, width=width)


def _parse_reg(raw: Dict[str, Any], loc: str) -> RegOp:
    id_ = _require_id(raw, loc)
    args = _require_str_list(raw, "args", loc, exact=1)

    clock = raw.get("clock")
    if not isinstance(clock, str) or not clock:
        raise ParseError(f"{loc}: 'clock' must be a non-empty string")

    reset = raw.get("reset", "")
    enable = raw.get("enable", "")
    reset_value = raw.get("resetValue", 0)

    if reset:
        if not isinstance(reset, str):
            raise ParseError(f"{loc}: 'reset' must be a string")
        if "resetValue" not in raw:
            raise ParseError(f"{loc}: 'resetValue' is required when 'reset' is provided")
        if not isinstance(reset_value, (int, str)):
            raise ParseError(f"{loc}: 'resetValue' must be an integer or string")
    else:
        if "reset" in raw and not isinstance(reset, str):
            raise ParseError(f"{loc}: 'reset' must be a string")

    if enable and not isinstance(enable, str):
        raise ParseError(f"{loc}: 'enable' must be a string")

    reg_width = raw.get("width", 0)
    if not isinstance(reg_width, int) or reg_width < 0:
        raise ParseError(f"{loc}: 'width' must be a non-negative integer")

    return RegOp(
        id=id_, op="reg", args=args, clock=clock,
        reset=reset, resetValue=reset_value, enable=enable,
        width=reg_width,
    )


def _parse_mem(raw: Dict[str, Any], loc: str) -> MemOp:
    id_ = raw.get("id")
    if not isinstance(id_, list):
        raise ParseError(f"{loc}: 'id' must be an array of strings")
    for i, v in enumerate(id_):
        if not isinstance(v, str) or not v:
            raise ParseError(f"{loc}: 'id[{i}]' must be a non-empty string")

    width = raw.get("width")
    if not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")

    depth = raw.get("depth")
    if not isinstance(depth, int) or depth <= 0:
        raise ParseError(f"{loc}: 'depth' must be a positive integer")

    clock = raw.get("clock")
    if not isinstance(clock, str) or not clock:
        raise ParseError(f"{loc}: 'clock' must be a non-empty string")

    reset = raw.get("reset")
    if not isinstance(reset, str) or not reset:
        raise ParseError(f"{loc}: 'reset' must be a non-empty string")

    reads_raw = raw.get("reads")
    if not isinstance(reads_raw, list):
        raise ParseError(f"{loc}: 'reads' must be an array")
    reads = []
    for i, r in enumerate(reads_raw):
        if not isinstance(r, dict):
            raise ParseError(f"{loc}: 'reads[{i}]' must be an object")
        addr = r.get("addr")
        enable = r.get("enable")
        if not isinstance(addr, str) or not addr:
            raise ParseError(f"{loc}: 'reads[{i}].addr' must be a non-empty string")
        if not isinstance(enable, str) or not enable:
            raise ParseError(f"{loc}: 'reads[{i}].enable' must be a non-empty string")
        reads.append((addr, enable))

    writes_raw = raw.get("writes")
    if not isinstance(writes_raw, list):
        raise ParseError(f"{loc}: 'writes' must be an array")
    writes = []
    for i, w in enumerate(writes_raw):
        if not isinstance(w, dict):
            raise ParseError(f"{loc}: 'writes[{i}]' must be an object")
        addr = w.get("addr")
        data = w.get("data")
        enable = w.get("enable")
        if not isinstance(addr, str) or not addr:
            raise ParseError(f"{loc}: 'writes[{i}].addr' must be a non-empty string")
        if not isinstance(data, str) or not data:
            raise ParseError(f"{loc}: 'writes[{i}].data' must be a non-empty string")
        if not isinstance(enable, str) or not enable:
            raise ParseError(f"{loc}: 'writes[{i}].enable' must be a non-empty string")
        writes.append((addr, data, enable))

    if len(id_) != len(reads):
        raise ParseError(
            f"{loc}: 'id' has {len(id_)} element(s) but 'reads' has {len(reads)} — "
            "must have one output id per read port"
        )

    return MemOp(
        id=id_, op="mem", width=width, depth=depth,
        clock=clock, reset=reset,
        reads=tuple(reads), writes=tuple(writes),
    )


def _parse_instance(raw: Dict[str, Any], loc: str) -> InstanceOp:
    id_ = raw.get("id")
    if not isinstance(id_, list):
        raise ParseError(f"{loc}: 'id' must be an array of strings")
    for i, v in enumerate(id_):
        if not isinstance(v, str) or not v:
            raise ParseError(f"{loc}: 'id[{i}]' must be a non-empty string")
    module = raw.get("module")
    if not isinstance(module, str) or not module:
        raise ParseError(f"{loc}: 'module' must be a non-empty string")
    args = _require_str_dict(raw, "args", loc)
    return InstanceOp(id=id_, op="instance", module=module, args=args)


def _parse_output(raw: Dict[str, Any], loc: str) -> OutputOp:
    args = _require_str_dict(raw, "args", loc)
    return OutputOp(op="output", args=args)
