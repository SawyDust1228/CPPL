"""JSON parsing and structural validation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from .errors import DiagnosticIssue, ParseError, issues_from_errors
from .identifiers import is_valid_ssa_id
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
    errors: list[ParseError] = []
    for i, mod_raw in enumerate(data):
        if not isinstance(mod_raw, dict):
            errors.append(
                ParseError(
                    [
                        DiagnosticIssue(
                            code="parse.module_type",
                            location=f"module[{i}]",
                            message="module must be a JSON object",
                            expected="object",
                            actual=type(mod_raw).__name__,
                            hint="Wrap the module name, ports, and body in one JSON object.",
                        )
                    ]
                )
            )
            continue
        try:
            modules.append(parse_module(mod_raw, i))
        except ParseError as exc:
            errors.append(exc)
    if errors:
        issues = issues_from_errors(errors)
        raise ParseError(
            issues,
            summary=f"JSON-IR parsing found {len(issues)} error(s)",
        )
    return modules


def parse_module(raw: Dict[str, Any], index: int) -> Module:
    ctx = f"module[{index}]"

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError(f"{ctx}: 'name' must be a non-empty string")

    ctx = f"module '{name}'"

    ports_raw = raw.get("ports")
    if not isinstance(ports_raw, dict):
        raise ParseError(f"{ctx}: 'ports' must be an object")

    ports: Dict[str, PortDef] = {}
    errors: list[ParseError] = []
    for pname, pdef in ports_raw.items():
        try:
            ports[pname] = parse_port_definition(ctx, pname, pdef)
        except ParseError as exc:
            errors.append(exc)

    body_raw = raw.get("body")
    if not isinstance(body_raw, list):
        errors.append(
            ParseError(
                [
                    DiagnosticIssue(
                        code="parse.body_type",
                        location=f"{ctx} body",
                        message="'body' must be an array",
                        expected="array",
                        actual=type(body_raw).__name__,
                        hint="Return a JSON array of operations.",
                    )
                ]
            )
        )
        body_raw = []

    body: List[Operation] = []
    for j, op_raw in enumerate(body_raw):
        if isinstance(op_raw, dict) and "_comment" in op_raw and "op" not in op_raw:
            continue  # skip comment-only entries
        try:
            body.append(parse_operation(op_raw, ctx, j))
        except ParseError as exc:
            errors.append(exc)

    if errors:
        issues = issues_from_errors(errors)
        raise ParseError(
            issues,
            summary=f"{ctx} parsing found {len(issues)} error(s)",
        )

    return Module(name=name, ports=ports, body=body)


def parse_port_definition(ctx: str, pname: object, pdef: Any) -> PortDef:
    """Parse one port with a detailed, independently collectable error."""
    loc = f"{ctx} port '{pname}'"
    if not is_valid_ssa_id(pname):
        raise ParseError(
            [
                DiagnosticIssue(
                    code="parse.port_identifier",
                    location=loc,
                    message="port name must be a valid SSA identifier",
                    expected="a meaningful string beginning with a letter or underscore",
                    actual=pname,
                    hint="Use letters, digits, '_', or '$'; numeric-only names are forbidden.",
                )
            ]
        )
    if not isinstance(pdef, dict):
        raise ParseError(
            f"{loc}: port definition must be an object; expected={{'dir': 'input|output', 'width': <positive integer>}}; actual={pdef!r}"
        )
    d = pdef.get("dir")
    if d not in ("input", "output"):
        raise ParseError(
            f"{loc}: dir must be 'input' or 'output'; actual={d!r}; fix=use the exact interface direction"
        )
    w = pdef.get("width")
    if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
        raise ParseError(
            f"{loc}: width must be a positive integer; actual={w!r}; fix=set the intended bit width"
        )
    port_type = pdef.get("type", "bits")
    if port_type not in ("bits", "clock"):
        raise ParseError(
            f"{loc}: type must be 'bits' or 'clock'; actual={port_type!r}"
        )
    if port_type == "clock" and (d != "input" or w != 1):
        raise ParseError(
            f"{loc}: clock ports must be 1-bit inputs; actual direction={d!r}, width={w!r}"
        )
    return PortDef(dir=PortDir(d), width=w, type=port_type)


def parse_operation(raw: Dict[str, Any], ctx: str, index: int) -> Operation:
    if not isinstance(raw, dict):
        raise ParseError(f"{ctx} body[{index}]: operation must be an object")

    op = raw.get("op")
    if not isinstance(op, str):
        raise ParseError(f"{ctx} body[{index}]: 'op' must be a string")

    loc = f"{ctx} body[{index}] (op='{op}')"

    if op == "constant":
        return parse_constant_operation(raw, loc)
    elif op == "mux":
        return parse_mux_operation(raw, loc)
    elif op in UNARY_OPS or op in REDUCE_OPS:
        return parse_unary_operation(raw, loc, op)
    elif op in BINARY_OPS or op in COMPARE_OPS:
        return parse_binary_operation(raw, loc, op)
    elif op in VARIADIC_OPS:
        return parse_variadic_operation(raw, loc, op)
    elif op == "extract":
        return parse_extract_operation(raw, loc)
    elif op in CAST_OPS:
        return parse_cast_operation(raw, loc, op)
    elif op == "reg":
        return parse_register_operation(raw, loc)
    elif op == "mem":
        return parse_memory_operation(raw, loc)
    elif op == "instance":
        return parse_instance_operation(raw, loc)
    elif op == "output":
        return parse_output_operation(raw, loc)
    else:
        raise ParseError(f"{loc}: unknown op '{op}'")


def require_id(raw: Dict[str, Any], loc: str) -> str:
    id_ = raw.get("id")
    if not is_valid_ssa_id(id_):
        raise ParseError(
            f"{loc}: 'id' must be an SSA identifier beginning with a letter or "
            "underscore and containing only letters, digits, underscores, or '$'"
        )
    return id_


def require_str_list(
    raw: Dict[str, Any], key: str, loc: str, *, exact: int = None
) -> List[str]:
    val = raw.get(key)
    if not isinstance(val, list):
        raise ParseError(f"{loc}: '{key}' must be an array")
    for i, v in enumerate(val):
        if not isinstance(v, str):
            raise ParseError(f"{loc}: '{key}[{i}]' must be a string")
    if exact is not None and len(val) != exact:
        raise ParseError(
            f"{loc}: '{key}' must have exactly {exact} element(s), got {len(val)}"
        )
    return val


def require_str_dict(raw: Dict[str, Any], key: str, loc: str) -> Dict[str, str]:
    val = raw.get(key)
    if not isinstance(val, dict):
        raise ParseError(f"{loc}: '{key}' must be an object")
    for k, v in val.items():
        if not isinstance(v, str):
            raise ParseError(f"{loc}: '{key}.{k}' must be a string")
    return val


def require_integer_literal(value: Any, field: str, loc: str) -> None:
    """Ensure a value can be lowered by codegen as an integer literal."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ParseError(f"{loc}: '{field}' must be an integer or integer string")
    if isinstance(value, str):
        try:
            int(value, 0)
        except ValueError as exc:
            raise ParseError(
                f"{loc}: '{field}' must be a valid integer string, got {value!r}"
            ) from exc


def parse_constant_operation(raw: Dict[str, Any], loc: str) -> ConstantOp:
    id_ = require_id(raw, loc)
    value = raw.get("value")
    require_integer_literal(value, "value", loc)
    width = raw.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return ConstantOp(id=id_, op="constant", value=value, width=width)


def parse_unary_operation(raw: Dict[str, Any], loc: str, op: str) -> UnaryOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=1)
    return UnaryOp(id=id_, op=op, args=args)


def parse_binary_operation(raw: Dict[str, Any], loc: str, op: str) -> BinaryOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=2)
    return BinaryOp(id=id_, op=op, args=args)


def parse_variadic_operation(raw: Dict[str, Any], loc: str, op: str) -> VariadicOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc)
    if len(args) < 1:
        raise ParseError(f"{loc}: 'args' must have at least 1 element")
    return VariadicOp(id=id_, op=op, args=args)


def parse_extract_operation(raw: Dict[str, Any], loc: str) -> ExtractOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=1)
    low = raw.get("lowBit")
    if isinstance(low, bool) or not isinstance(low, int) or low < 0:
        raise ParseError(f"{loc}: 'lowBit' must be a non-negative integer")
    width = raw.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return ExtractOp(id=id_, op="extract", args=args, lowBit=low, width=width)


def parse_mux_operation(raw: Dict[str, Any], loc: str) -> MuxOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=3)
    return MuxOp(id=id_, op="mux", args=args)


def parse_cast_operation(raw: Dict[str, Any], loc: str, op: str) -> CastOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=1)
    width = raw.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")
    return CastOp(id=id_, op=op, args=args, width=width)


def parse_register_operation(raw: Dict[str, Any], loc: str) -> RegOp:
    id_ = require_id(raw, loc)
    args = require_str_list(raw, "args", loc, exact=1)

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
            raise ParseError(
                f"{loc}: 'resetValue' is required when 'reset' is provided"
            )
        require_integer_literal(reset_value, "resetValue", loc)
    else:
        if "reset" in raw and not isinstance(reset, str):
            raise ParseError(f"{loc}: 'reset' must be a string")

    if enable and not isinstance(enable, str):
        raise ParseError(f"{loc}: 'enable' must be a string")

    reg_width = raw.get("width", 0)
    if isinstance(reg_width, bool) or not isinstance(reg_width, int) or reg_width < 0:
        raise ParseError(f"{loc}: 'width' must be a non-negative integer")

    return RegOp(
        id=id_,
        op="reg",
        args=args,
        clock=clock,
        reset=reset,
        resetValue=reset_value,
        enable=enable,
        width=reg_width,
    )


def parse_memory_operation(raw: Dict[str, Any], loc: str) -> MemOp:
    id_ = raw.get("id")
    if not isinstance(id_, list):
        raise ParseError(f"{loc}: 'id' must be an array of strings")
    for i, v in enumerate(id_):
        if not is_valid_ssa_id(v):
            raise ParseError(
                f"{loc}: 'id[{i}]' must be an SSA identifier beginning with a "
                "letter or underscore and containing only letters, digits, "
                "underscores, or '$'"
            )

    width = raw.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ParseError(f"{loc}: 'width' must be a positive integer")

    depth = raw.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ParseError(f"{loc}: 'depth' must be a positive integer")

    clock = raw.get("clock")
    if not isinstance(clock, str) or not clock:
        raise ParseError(f"{loc}: 'clock' must be a non-empty string")

    reset = raw.get("reset", "")
    if not isinstance(reset, str):
        raise ParseError(f"{loc}: 'reset' must be a string")

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
        mask = w.get("mask", "")
        if not isinstance(addr, str) or not addr:
            raise ParseError(f"{loc}: 'writes[{i}].addr' must be a non-empty string")
        if not isinstance(data, str) or not data:
            raise ParseError(f"{loc}: 'writes[{i}].data' must be a non-empty string")
        if not isinstance(enable, str) or not enable:
            raise ParseError(f"{loc}: 'writes[{i}].enable' must be a non-empty string")
        if not isinstance(mask, str):
            raise ParseError(f"{loc}: 'writes[{i}].mask' must be a string")
        writes.append((addr, data, enable, mask))

    if len(id_) != len(reads):
        raise ParseError(
            f"{loc}: 'id' has {len(id_)} element(s) but 'reads' has {len(reads)} — "
            "must have one output id per read port"
        )

    name = raw.get("name", "")
    if not isinstance(name, str):
        raise ParseError(f"{loc}: 'name' must be a string")

    init_file = raw.get("initFile", "")
    if not isinstance(init_file, str):
        raise ParseError(f"{loc}: 'initFile' must be a string")

    init_format = raw.get("initFormat", "hex")
    if init_format not in ("hex", "bin"):
        raise ParseError(f"{loc}: 'initFormat' must be 'hex' or 'bin'")

    return MemOp(
        id=id_,
        op="mem",
        width=width,
        depth=depth,
        clock=clock,
        reset=reset,
        reads=tuple(reads),
        writes=tuple(writes),
        name=name,
        initFile=init_file,
        initFormat=init_format,
    )


def parse_instance_operation(raw: Dict[str, Any], loc: str) -> InstanceOp:
    id_ = raw.get("id")
    if not isinstance(id_, list):
        raise ParseError(f"{loc}: 'id' must be an array of strings")
    for i, v in enumerate(id_):
        if not is_valid_ssa_id(v):
            raise ParseError(
                f"{loc}: 'id[{i}]' must be an SSA identifier beginning with a "
                "letter or underscore and containing only letters, digits, "
                "underscores, or '$'"
            )
    module = raw.get("module")
    if not isinstance(module, str) or not module:
        raise ParseError(f"{loc}: 'module' must be a non-empty string")
    args = require_str_dict(raw, "args", loc)
    name = raw.get("name", "")
    if not isinstance(name, str):
        raise ParseError(f"{loc}: 'name' must be a string")
    return InstanceOp(id=id_, op="instance", module=module, args=args, name=name)


def parse_output_operation(raw: Dict[str, Any], loc: str) -> OutputOp:
    args = require_str_dict(raw, "args", loc)
    return OutputOp(op="output", args=args)
