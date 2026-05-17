"""Intermediate representation: parsing, validation, and inference."""

from .parser import parse_design
from .validator import validate_design
from .infer import infer_widths
from .errors import CircuitPPLError, ParseError, ValidationError, SSAError, CycleError, WidthError, CodegenError
