"""Data models for JSON-IR (frozen dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union


class PortDir(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class PortDef:
    dir: PortDir
    width: int
    type: str = "bits"


@dataclass(frozen=True)
class ConstantOp:
    id: str
    op: str  # "constant"
    value: Union[int, str]
    width: int


@dataclass(frozen=True)
class UnaryOp:
    id: str
    op: str  # "not" | "neg" | "reverse"
    args: List[str]  # exactly 1 element


@dataclass(frozen=True)
class BinaryOp:
    id: str
    op: str  # "add"|"sub"|"mul"|"div"|"div_s"|"mod_u"|"mod_s"|"and"|"or"|"xor"|"shl"|"shr_u"|"shr_s"
    args: List[str]  # exactly 2 elements


@dataclass(frozen=True)
class VariadicOp:
    id: str
    op: str  # "concat"
    args: List[str]  # N elements


@dataclass(frozen=True)
class MuxOp:
    id: str
    op: str  # "mux"
    args: List[str]  # [sel, true_val, false_val]


@dataclass(frozen=True)
class CastOp:
    id: str
    op: str  # "sext" | "zext"
    args: List[str]  # exactly 1 element
    width: int  # target width


@dataclass(frozen=True)
class ExtractOp:
    id: str
    op: str  # "extract"
    args: List[str]  # exactly 1 element
    lowBit: int
    width: int


@dataclass(frozen=True)
class RegOp:
    id: str
    op: str  # "reg"
    args: List[str]  # exactly 1 element (data input)
    clock: str  # 1-bit clock signal reference
    reset: str = ""  # optional: 1-bit synchronous reset signal
    resetValue: Union[int, str] = 0  # required when reset is provided
    enable: str = ""  # optional: 1-bit clock enable signal
    width: int = 0  # optional explicit width (breaks inference cycles)


@dataclass(frozen=True)
class MemOp:
    id: List[str]           # one output per read port
    op: str                 # "mem"
    width: int              # element bit width
    depth: int              # number of entries
    clock: str              # clock signal ref
    reset: str              # optional reset signal ref
    reads: tuple            # ((addr_ref, enable_ref), ...)
    writes: tuple           # ((addr_ref, data_ref, enable_ref), ...)
    name: str = ""
    initFile: str = ""
    initFormat: str = "hex"


@dataclass(frozen=True)
class InstanceOp:
    id: List[str]  # one per output port of instantiated module
    op: str  # "instance"
    module: str
    args: Dict[str, str]  # child_input_port -> value_id
    name: str = ""


@dataclass(frozen=True)
class OutputOp:
    op: str  # "output"
    args: Dict[str, str]  # module_output_port -> value_id


Operation = Union[
    ConstantOp, UnaryOp, BinaryOp, VariadicOp,
    ExtractOp, MuxOp, CastOp, RegOp, MemOp, InstanceOp, OutputOp,
]

UNARY_OPS = frozenset({"not", "neg", "reverse"})
BINARY_OPS = frozenset({
    "add", "sub", "mul", "div", "div_s",
    "mod_u", "mod_s",
    "and", "or", "xor",
    "shl", "shr_u", "shr_s",
})
VARIADIC_OPS = frozenset({"concat"})
COMPARE_OPS = frozenset({
    "eq", "ne", "lt_s", "lt_u", "ge_s", "ge_u",
    "gt_s", "gt_u", "le_s", "le_u",
})
REDUCE_OPS = frozenset({"or_reduce", "and_reduce", "xor_reduce"})
CAST_OPS = frozenset({"sext", "zext"})


@dataclass(frozen=True)
class Module:
    name: str
    ports: Dict[str, PortDef]
    body: List[Operation] = field(default_factory=list)
