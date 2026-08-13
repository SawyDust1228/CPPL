"""Thread-safe terminal renderer for CPPL compilation events."""

from __future__ import annotations

import os
import sys
import threading
from typing import TextIO

from termcolor import colored

from .events import CompileEvent


_EVENT_LABELS = {
    "design_start": "START",
    "wave_start": "WAVE",
    "cache_check": "CACHE",
    "cache_hit": "HIT",
    "cache_miss": "MISS",
    "compress": "COMPRESS",
    "generate": "GENERATE",
    "repair": "REPAIR",
    "validate": "VALIDATE",
    "repair_scheduled": "RETRY",
    "module_success": "SUCCESS",
    "module_failed": "FAILED",
    "module_blocked": "BLOCKED",
    "module_cancelled": "CANCEL",
    "design_validate": "CHECK",
    "design_success": "DONE",
    "design_failed": "FAILED",
}

_EVENT_COLORS = {
    "design_start": "cyan",
    "wave_start": "cyan",
    "cache_hit": "green",
    "compress": "yellow",
    "generate": "blue",
    "repair": "yellow",
    "validate": "cyan",
    "repair_scheduled": "yellow",
    "module_success": "green",
    "module_failed": "red",
    "module_blocked": "red",
    "module_cancelled": "yellow",
    "design_validate": "cyan",
    "design_success": "green",
    "design_failed": "red",
}


class TerminalCompileUI:
    """Render stable one-line compile logs to stderr by default."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = (
            is_tty and "NO_COLOR" not in os.environ if color is None else color
        )
        self._lock = threading.Lock()

    def format_event(self, event: CompileEvent) -> str:
        label = _EVENT_LABELS.get(event.kind, event.kind.upper())
        raw_label = f"{label:<8}"
        display_label = colored(
            raw_label,
            _EVENT_COLORS.get(event.kind),
            attrs=("bold",),
            force_color=self.color,
            no_color=not self.color,
        )
        target = f" {event.module_name}" if event.module_name else ""
        message = f" — {event.message}" if event.message else ""
        return (
            f"[CPPL {event.elapsed_seconds:7.2f}s] "
            f"{display_label}{target}{message}"
        )

    def on_event(self, event: CompileEvent) -> None:
        line = self.format_event(event)
        with self._lock:
            print(line, file=self.stream, flush=True)
