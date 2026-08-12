"""Intermediate representation: parsing, validation, and inference."""

from .parser import parse_design
from .validator import validate_design
from .infer import infer_widths
from .interpreter import Interpreter
from .errors import (
    CircuitPPLError,
    ParseError,
    ValidationError,
    SSAError,
    CycleError,
    WidthError,
    CodegenError,
    SimulationError,
)

__all__ = [
    "parse_design",
    "validate_design",
    "infer_widths",
    "Interpreter",
    "CircuitPPLError",
    "ParseError",
    "ValidationError",
    "SSAError",
    "CycleError",
    "WidthError",
    "CodegenError",
    "SimulationError",
]
