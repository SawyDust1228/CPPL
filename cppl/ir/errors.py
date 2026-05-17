"""Exception hierarchy for CircuitPPL JSON-IR compiler."""


class CircuitPPLError(Exception):
    """Base exception for all CircuitPPL errors."""


class ParseError(CircuitPPLError):
    """JSON structure or field errors during parsing."""


class ValidationError(CircuitPPLError):
    """Semantic validation errors."""


class SSAError(ValidationError):
    """SSA violations: forward references or redefinitions."""


class WidthError(ValidationError):
    """Bit-width mismatches or inference failures."""


class CycleError(ValidationError):
    """Combinational cycle detected in the design."""


class CodegenError(CircuitPPLError):
    """Errors during MLIR/CIRCT code generation."""
