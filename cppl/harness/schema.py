"""Strict model-facing schema for CPPL JSON-IR module bodies."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    ValidationError,
)

from ..ir.errors import DiagnosticIssue
from ..ir.identifiers import SSA_ID_PATTERN


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Operand = Union[str, int]
SSAIdentifier = Annotated[str, StringConstraints(pattern=SSA_ID_PATTERN)]


class ConstantSchema(StrictModel):
    op: Literal["constant"]
    id: SSAIdentifier
    value: Union[int, str]
    width: int = Field(gt=0)


class UnarySchema(StrictModel):
    op: Literal["not", "neg", "reverse", "or_reduce", "and_reduce", "xor_reduce"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=1, max_length=1)


class BinarySchema(StrictModel):
    op: Literal[
        "add", "sub", "mul", "div", "div_s", "mod_u", "mod_s", "and", "or",
        "xor", "shl", "shr_u", "shr_s", "eq", "ne", "lt_s", "lt_u", "le_s",
        "le_u", "gt_s", "gt_u", "ge_s", "ge_u",
    ]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=2, max_length=2)


class ConcatSchema(StrictModel):
    op: Literal["concat"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=1)


class ExtractSchema(StrictModel):
    op: Literal["extract"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=1, max_length=1)
    lowBit: int = Field(ge=0)
    width: int = Field(gt=0)


class MuxSchema(StrictModel):
    op: Literal["mux"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=3, max_length=3)


class CastSchema(StrictModel):
    op: Literal["sext", "zext"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=1, max_length=1)
    width: int = Field(gt=0)


class RegSchema(StrictModel):
    op: Literal["reg"]
    id: SSAIdentifier
    args: list[Operand] = Field(min_length=1, max_length=1)
    clock: Operand
    reset: Operand | None = None
    resetValue: int | str | None = None
    enable: Operand | None = None
    width: int | None = Field(default=None, ge=0)


class ReadPortSchema(StrictModel):
    addr: Operand
    enable: Operand


class WritePortSchema(StrictModel):
    addr: Operand
    data: Operand
    enable: Operand


class MemSchema(StrictModel):
    op: Literal["mem"]
    id: list[SSAIdentifier]
    width: int = Field(gt=0)
    depth: int = Field(gt=0)
    clock: Operand
    reset: Operand | None = None
    name: str | None = None
    initFile: str | None = None
    initFormat: Literal["hex", "bin"] | None = None
    reads: list[ReadPortSchema]
    writes: list[WritePortSchema]


class InstanceSchema(StrictModel):
    op: Literal["instance"]
    id: list[SSAIdentifier]
    module: str = Field(min_length=1)
    name: str | None = None
    args: dict[str, Operand]


class OutputSchema(StrictModel):
    op: Literal["output"]
    args: dict[str, Operand]


OperationSchema = Annotated[
    ConstantSchema
    | UnarySchema
    | BinarySchema
    | ConcatSchema
    | ExtractSchema
    | MuxSchema
    | CastSchema
    | RegSchema
    | MemSchema
    | InstanceSchema
    | OutputSchema,
    Field(discriminator="op"),
]


class ModuleBodySchema(RootModel[list[OperationSchema]]):
    """One complete generated CPPL module body."""


class IRSchemaError(ValueError):
    """Model output does not satisfy the public JSON-IR schema."""

    def __init__(self, issues: list[DiagnosticIssue]) -> None:
        self.issues = tuple(issues)
        message = f"IR schema validation found {len(issues)} error(s)"
        if issues:
            message += "\n" + "\n".join(
                f"  {index}. {issue.format()}"
                for index, issue in enumerate(issues, 1)
            )
        super().__init__(message)


def validate_ir_body(value: object) -> list[dict]:
    """Validate and return a plain JSON-compatible operation list."""
    try:
        body = ModuleBodySchema.model_validate(value)
    except ValidationError as exc:
        issues = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(item) for item in error.get("loc", ()))
            issues.append(
                DiagnosticIssue(
                    code=f"schema.{error.get('type', 'invalid')}",
                    location=location or "body",
                    message=error["msg"],
                    actual=error.get("input"),
                    hint="Correct this field to match the selected operation schema.",
                )
            )
        raise IRSchemaError(issues) from exc
    return body.model_dump(mode="json", exclude_none=True)


__all__ = ["IRSchemaError", "ModuleBodySchema", "validate_ir_body"]
