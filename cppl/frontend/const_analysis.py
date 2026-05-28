"""Constant analysis for LLM-produced JSON-IR bodies."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

from .module import InstanceCall


def _literal_int(ref: object) -> Optional[int]:
    if isinstance(ref, int):
        return ref
    if not isinstance(ref, str):
        return None

    text = ref.strip()
    if not text:
        return None
    if not text.isdecimal():
        return None

    try:
        return int(text, 10)
    except ValueError:
        return None


def _collect_defined_names(ports: dict, body: list) -> set[str]:
    names = set(ports)
    for op in body:
        if not isinstance(op, dict):
            continue
        id_ = op.get("id")
        if isinstance(id_, str):
            names.add(id_)
        elif isinstance(id_, list):
            names.update(v for v in id_ if isinstance(v, str))
    return names


def normalize_constants(
    body: list,
    ports: dict,
    instances: list[InstanceCall],
) -> list:
    """Promote untyped decimal constants into explicit constant ops.

    The LLM may emit compact JSON numbers such as ``0`` or ``1`` in operand
    positions.  The rest of the compiler expects SSA value IDs, so this pass
    inserts appropriately-wide ``constant`` ops and rewrites those literal
    references to the generated IDs.
    """
    out: list = []
    used_names = _collect_defined_names(ports, body)
    value_widths = {
        name: pdef["width"]
        for name, pdef in ports.items()
        if pdef.get("dir") == "input"
    }
    instance_inputs = {
        inst.target_name: {
            p.name: p.width for p in inst.target_ports
            if p.direction == "input"
        }
        for inst in instances
    }
    const_ids: dict[tuple[int, int], str] = {}

    def fresh_const(value: int, width: int) -> str:
        width = max(1, width)
        key = (value, width)
        if key in const_ids:
            return const_ids[key]

        base = f"_const_{value}_{width}"
        name = base
        i = 0
        while name in used_names:
            i += 1
            name = f"{base}_{i}"

        used_names.add(name)
        const_ids[key] = name
        out.append({
            "id": name,
            "op": "constant",
            "value": value,
            "width": width,
        })
        value_widths[name] = width
        return name

    def normalize_ref(ref: object, width: int) -> object:
        value = _literal_int(ref)
        if value is None:
            return ref
        return fresh_const(value, width)

    def width_of(ref: object, default: int = 1) -> int:
        if isinstance(ref, str) and ref in value_widths:
            return value_widths[ref]
        return default

    def record_result_width(op: dict) -> None:
        id_ = op.get("id")
        opname = op.get("op")
        if not isinstance(id_, str):
            if opname == "mem" and isinstance(id_, list):
                for name in id_:
                    if isinstance(name, str):
                        value_widths[name] = int(op.get("width", 1))
            return

        if opname == "constant":
            value_widths[id_] = int(op.get("width", 1))
        elif opname in {
            "eq", "ne", "lt_s", "lt_u", "ge_s", "ge_u",
            "gt_s", "gt_u", "le_s", "le_u",
            "or_reduce", "and_reduce", "xor_reduce",
        }:
            value_widths[id_] = 1
        elif opname in {"sext", "zext", "extract"}:
            value_widths[id_] = int(op.get("width", 1))
        elif opname == "concat":
            value_widths[id_] = sum(width_of(arg) for arg in op.get("args", []))
        elif opname == "mux":
            args = op.get("args", [])
            if len(args) == 3:
                value_widths[id_] = max(width_of(args[1]), width_of(args[2]))
        elif opname in {"not", "neg", "reverse"}:
            args = op.get("args", [])
            if args:
                value_widths[id_] = width_of(args[0])
        elif opname in {
            "add", "sub", "mul", "div", "div_s", "mod_u", "mod_s",
            "and", "or", "xor", "shl", "shr_u", "shr_s",
        }:
            args = op.get("args", [])
            if args:
                value_widths[id_] = max(width_of(arg) for arg in args)
        elif opname == "reg":
            args = op.get("args", [])
            explicit = int(op.get("width", 0) or 0)
            if explicit:
                value_widths[id_] = explicit
            elif args:
                value_widths[id_] = width_of(args[0])

    for raw_op in body:
        if not isinstance(raw_op, dict):
            out.append(raw_op)
            continue

        op = deepcopy(raw_op)
        opname = op.get("op")

        if opname in {
            "not", "neg", "reverse", "or_reduce", "and_reduce", "xor_reduce",
        }:
            args = op.get("args", [])
            if len(args) == 1:
                args[0] = normalize_ref(args[0], width_of(args[0]))
        elif opname in {
            "add", "sub", "mul", "div", "div_s", "mod_u", "mod_s",
            "and", "or", "xor", "shl", "shr_u", "shr_s",
            "eq", "ne", "lt_s", "lt_u", "ge_s", "ge_u",
            "gt_s", "gt_u", "le_s", "le_u",
        }:
            args = op.get("args", [])
            if len(args) == 2:
                width = max(width_of(args[0]), width_of(args[1]))
                args[0] = normalize_ref(args[0], width)
                args[1] = normalize_ref(args[1], width)
        elif opname == "mux":
            args = op.get("args", [])
            if len(args) == 3:
                args[0] = normalize_ref(args[0], 1)
                data_width = max(width_of(args[1]), width_of(args[2]))
                args[1] = normalize_ref(args[1], data_width)
                args[2] = normalize_ref(args[2], data_width)
        elif opname in {"sext", "zext"}:
            args = op.get("args", [])
            if len(args) == 1:
                args[0] = normalize_ref(args[0], 1)
        elif opname == "extract":
            args = op.get("args", [])
            if len(args) == 1:
                min_width = int(op.get("lowBit", 0)) + int(op.get("width", 1))
                args[0] = normalize_ref(args[0], min_width)
        elif opname == "concat":
            args = op.get("args", [])
            for i, arg in enumerate(args):
                args[i] = normalize_ref(arg, width_of(arg))
        elif opname == "reg":
            args = op.get("args", [])
            if len(args) == 1:
                data_width = int(op.get("width", 0) or 0) or width_of(args[0])
                args[0] = normalize_ref(args[0], data_width)
            if "clock" in op:
                op["clock"] = normalize_ref(op["clock"], 1)
            if op.get("reset"):
                op["reset"] = normalize_ref(op["reset"], 1)
            if op.get("enable"):
                op["enable"] = normalize_ref(op["enable"], 1)
        elif opname == "mem":
            depth = int(op.get("depth", 1))
            addr_width = max(1, math.ceil(math.log2(depth)))
            elem_width = int(op.get("width", 1))
            if "clock" in op:
                op["clock"] = normalize_ref(op["clock"], 1)
            if op.get("reset"):
                op["reset"] = normalize_ref(op["reset"], 1)
            for read in op.get("reads", []):
                read["addr"] = normalize_ref(read.get("addr"), addr_width)
                read["enable"] = normalize_ref(read.get("enable"), 1)
            for write in op.get("writes", []):
                write["addr"] = normalize_ref(write.get("addr"), addr_width)
                write["data"] = normalize_ref(write.get("data"), elem_width)
                write["enable"] = normalize_ref(write.get("enable"), 1)
        elif opname == "instance":
            widths = instance_inputs.get(op.get("module"), {})
            for port, ref in op.get("args", {}).items():
                op["args"][port] = normalize_ref(ref, widths.get(port, 1))
        elif opname == "output":
            for port, ref in op.get("args", {}).items():
                port_def = ports.get(port, {})
                op["args"][port] = normalize_ref(ref, port_def.get("width", 1))

        out.append(op)
        record_result_width(op)

    return out
