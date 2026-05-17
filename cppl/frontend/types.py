"""Port type descriptors for CPPL DSL: In[N] and Out[N]."""


class _PortType:
    """Runtime descriptor carrying width and direction for a port annotation."""

    def __init__(self, width: int, direction: str):
        self.width = width
        self.direction = direction  # "input" or "output"

    def __repr__(self) -> str:
        tag = "In" if self.direction == "input" else "Out"
        return f"{tag}[{self.width}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PortType):
            return NotImplemented
        return self.width == other.width and self.direction == other.direction


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
