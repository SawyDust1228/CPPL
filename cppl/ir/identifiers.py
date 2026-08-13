"""Shared identifier rules for CPPL JSON-IR."""

from __future__ import annotations

import re


SSA_ID_PATTERN = r"^[A-Za-z_][A-Za-z0-9_$]*$"
_SSA_ID_RE = re.compile(SSA_ID_PATTERN)


def is_valid_ssa_id(value: object) -> bool:
    """Return whether *value* is a legal, non-numeric SSA identifier."""
    return isinstance(value, str) and _SSA_ID_RE.fullmatch(value) is not None


__all__ = ["SSA_ID_PATTERN", "is_valid_ssa_id"]
