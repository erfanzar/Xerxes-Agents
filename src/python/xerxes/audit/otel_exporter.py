# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry exporter for Xerxes audit events.

Maps the typed event hierarchy from :mod:`xerxes.audit.events` into
spans + log records. The ``opentelemetry-api`` / ``opentelemetry-sdk``
packages are imported lazily; tests work without them by using the
``NoopTracer`` fallback.

The exporter is implemented as an :class:`AuditCollector` so it slots
into the existing audit fan-out via :class:`CompositeCollector`.

Usage::

    from xerxes.audit import AuditEmitter, CompositeCollector, InMemoryCollector
    from xerxes.audit.otel_exporter import OTelCollector

    inmem = InMemoryCollector()
    otel = OTelCollector(service_name="xerxes")
    emitter = AuditEmitter(collector=CompositeCollector([inmem, otel]))
"""

from __future__ import annotations

import logging
import typing as tp
from contextlib import contextmanager
from datetime import datetime

from .events import (
    AuditEvent,
    ErrorEvent,
    SkillUsedEvent,
    ToolCallAttemptEvent,
    ToolCallCompleteEvent,
    ToolCallFailureEvent,
    TurnEndEvent,
    TurnStartEvent,
)

logger = logging.getLogger(__name__)


def _try_import_otel() -> tuple[tp.Any, tp.Any] | None:
    """Lazily import OTel; returns ``(trace_module, tracer)`` or ``None``."""
    try:
        from opentelemetry import trace as _trace
    except ImportError:
        return None
    try:
        return _trace, _trace.get_tracer("xerxes")
    except Exception:
        return None


class OTelCollector:
    """An :class:`AuditCollector` that fans events out to OpenTelemetry.

    Spans are opened for ``TurnStartEvent`` and closed (with status) on
    the matching ``TurnEndEvent`` (correlated by ``turn_id``). Tool
    invocations are emitted as child spans. Other event types are
    recorded as span events on the active turn span when available, or
    as log records otherwise.

    When OpenTelemetry is not installed, the exporter degrades to a
    structured logger sink so audit fan-out still works (the
    :class:`AuditCollector` protocol is satisfied either way).

    Attributes:
        service_name: Logical service name used for the tracer.
    """

    def __init__(self, service_name: str = "xerxes") -> None:
        """Create the collector, lazily probing for the ``opentelemetry`` package."""
        self.service_name = service_name
        otel = _try_import_otel()
        if otel is None:
            self._trace = None
            self._tracer = None
            logger.info("OpenTelemetry not installed; OTelCollector will log instead")
        else:
            self._trace, self._tracer = otel
        self._open_turn_spans: dict[str, tp.Any] = {}
        self._noop_log: list[dict[str, tp.Any]] = []

    def emit(self, event: AuditEvent) -> None:
        """Convert *event* to spans / span events / logs."""
        try:
            if isinstance(event, TurnStartEvent):
                self._on_turn_start(event)
            elif isinstance(event, TurnEndEvent):
                self._on_turn_end(event)
            elif isinstance(event, ToolCallAttemptEvent):
                self._on_tool_attempt(event)
            elif isinstance(event, ToolCallCompleteEvent):
                self._on_tool_complete(event)
            elif isinstance(event, ToolCallFailureEvent):
                self._on_tool_failure(event)
            elif isinstance(event, SkillUsedEvent):
                self._on_skill_used(event)
            elif isinstance(event, ErrorEvent):
                self._on_error(event)
            else:
                self._record_event(event.event_type, event)
        except Exception:
            logger.warning("OTelCollector failed to handle %s", event.event_type, exc_info=True)

    def flush(self) -> None:
        """Force-close any orphan spans (best-effort)."""
        for span in list(self._open_turn_spans.values()):
            try:
                span.end()
            except Exception:
                pass
        self._open_turn_spans.clear()

    @property
    def has_otel(self) -> bool:
        """Whether a real OpenTelemetry tracer was successfully imported."""
        return self._tracer is not None

    @property
    def fallback_log(self) -> list[dict[str, tp.Any]]:
        """Read-only view of log records when OTel is not installed."""
        return list(self._noop_log)

    @contextmanager
    def _span(self, name: str, attributes: dict[str, tp.Any]):
        """Start a standalone span (or log a fallback record) and end it on exit."""
        if self._tracer is None:
            self._noop_log.append({"name": name, "attributes": dict(attributes)})
            yield None
            return
        span = self._tracer.start_span(name=name, attributes=_clean_attrs(attributes))
        try:
            yield span
        finally:
            try:
                span.end()
            except Exception:
                pass

    def _record_event(self, name: str, event: AuditEvent) -> None:
        """Attach *event* as a span event on the active turn span, else as a standalone span/log."""
        attrs = _clean_attrs(event.to_dict())
        turn_id = getattr(event, "turn_id", None)
        if turn_id and turn_id in self._open_turn_spans and self._tracer is not None:
            try:
                self._open_turn_spans[turn_id].add_event(name, attributes=attrs)
                return
            except Exception:
                pass
        if self._tracer is None:
            self._noop_log.append({"name": name, "attributes": attrs})
        else:
            with self._span(name, attrs):
                pass

    def _on_turn_start(self, event: TurnStartEvent) -> None:
        """Open a ``xerxes.turn`` span and index it by ``turn_id`` for later close."""
        if self._tracer is None or not event.turn_id:
            self._noop_log.append({"name": "turn", "attributes": _clean_attrs(event.to_dict())})
            return
        span = self._tracer.start_span(
            name="xerxes.turn",
            attributes=_clean_attrs(
                {
                    "xerxes.turn_id": event.turn_id,
                    "xerxes.agent_id": event.agent_id,
                    "xerxes.session_id": event.session_id,
                    "xerxes.prompt_preview": event.prompt_preview,
                    "service.name": self.service_name,
                }
            ),
        )
        self._open_turn_spans[event.turn_id] = span

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        """Set ``xerxes.function_calls_count`` and close the matching turn span."""
        if not event.turn_id:
            return
        span = self._open_turn_spans.pop(event.turn_id, None)
        if span is None:
            return
        try:
            span.set_attribute("xerxes.function_calls_count", event.function_calls_count)
            span.end()
        except Exception:
            pass

    def _on_tool_attempt(self, event: ToolCallAttemptEvent) -> None:
        """Emit a ``tool.attempt:<tool>`` span event on the active turn span."""
        self._record_event(f"tool.attempt:{event.tool_name}", event)

    def _on_tool_complete(self, event: ToolCallCompleteEvent) -> None:
        """Emit a ``tool.complete:<tool>`` span event carrying the tool's result payload."""
        self._record_event(f"tool.complete:{event.tool_name}", event)

    def _on_tool_failure(self, event: ToolCallFailureEvent) -> None:
        """Emit a ``tool.failure:<tool>`` span event with the error details."""
        self._record_event(f"tool.failure:{event.tool_name}", event)

    def _on_skill_used(self, event: SkillUsedEvent) -> None:
        """Emit a ``skill.used:<skill>`` span event marking skill invocation."""
        self._record_event(f"skill.used:{event.skill_name}", event)

    def _on_error(self, event: ErrorEvent) -> None:
        """Emit an ``error:<type>`` span event capturing an audited error."""
        self._record_event(f"error:{event.error_type}", event)


def _clean_attrs(d: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Filter ``None`` and coerce non-primitive values to strings."""
    out: dict[str, tp.Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, str | int | float | bool):
            out[k] = v
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = str(v)[:200]
    return out


__all__ = ["OTelCollector"]
