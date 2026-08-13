"""Public compile-event interfaces for CPPL harnesses and UIs."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class CompileEvent:
    """One non-sensitive state transition emitted during compilation."""

    kind: str
    elapsed_seconds: float
    module_name: str | None = None
    stage: str | None = None
    attempt: int | None = None
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class CompileObserver(Protocol):
    """Consumer of real-time compilation events."""

    def on_event(self, event: CompileEvent) -> None: ...


class NullCompileObserver:
    """Observer used when compile logging is disabled."""

    def on_event(self, event: CompileEvent) -> None:
        return None


class CompositeCompileObserver:
    """Forward events to multiple observers in registration order."""

    def __init__(self, *observers: CompileObserver) -> None:
        self.observers = tuple(observers)

    def on_event(self, event: CompileEvent) -> None:
        for observer in self.observers:
            observer.on_event(event)


class CompileEventEmitter:
    """Timestamp and safely deliver events without affecting compilation."""

    def __init__(
        self,
        observer: CompileObserver,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.observer = observer
        self.clock = clock
        self.started = clock()
        self._lock = threading.Lock()

    def emit(
        self,
        kind: str,
        *,
        module_name: str | None = None,
        stage: str | None = None,
        attempt: int | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        event = CompileEvent(
            kind=kind,
            elapsed_seconds=max(0.0, self.clock() - self.started),
            module_name=module_name,
            stage=stage,
            attempt=attempt,
            message=message,
            metadata=dict(metadata or {}),
        )
        try:
            with self._lock:
                self.observer.on_event(event)
        except Exception:
            # Observability must never change compilation behavior.
            return None
