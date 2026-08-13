"""Intermediate representation: parsing, validation, and inference."""

from .parser import parse_design
from .validator import validate_design
from .infer import infer_widths
from .interpreter import Interpreter
from .patterns import run_patterns
from .errors import (
    CircuitPPLError,
    DiagnosticIssue,
    ParseError,
    ValidationError,
    SSAError,
    CycleError,
    WidthError,
    CodegenError,
    SimulationError,
    PatternMismatch,
    PatternDependencyMissing,
)

__all__ = [
    "parse_design",
    "validate_design",
    "infer_widths",
    "Interpreter",
    "run_patterns",
    "CircuitPPLError",
    "DiagnosticIssue",
    "ParseError",
    "ValidationError",
    "SSAError",
    "CycleError",
    "WidthError",
    "CodegenError",
    "SimulationError",
    "PatternMismatch",
    "PatternDependencyMissing",
]
