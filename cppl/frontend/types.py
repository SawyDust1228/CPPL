"""Port type descriptors for CPPL DSL: In[N], Out[N], and Clock."""


class _PortType:
    """Runtime descriptor carrying width and direction for a port annotation."""

    def __init__(self, width: int, direction: str, kind: str = "bits"):
        self.width = width
        self.direction = direction  # "input" or "output"
        self.kind = kind

    def __repr__(self) -> str:
        if self.kind == "clock":
            return "Clock"
        tag = "In" if self.direction == "input" else "Out"
        return f"{tag}[{self.width}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PortType):
            return NotImplemented
        return (
            self.width == other.width
            and self.direction == other.direction
            and self.kind == other.kind
        )


class In:
    """N-bit input port type.  Usage: ``In[8]``."""

    @classmethod
    def __class_getitem__(cls, width: int) -> _PortType:
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"In width must be a positive integer, got {width!r}")
        return _PortType(width, "input")


class Out:
    """N-bit output port type.  Usage: ``Out[8]``."""

    @classmethod
    def __class_getitem__(cls, width: int) -> _PortType:
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"Out width must be a positive integer, got {width!r}")
        return _PortType(width, "output")


Clock = _PortType(1, "input", "clock")
