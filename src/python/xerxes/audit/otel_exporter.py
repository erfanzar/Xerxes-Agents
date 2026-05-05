# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenTelemetry trace exporter for audit events.

This module provides :class:`OTelCollector`, which converts Xerxes audit events
into OpenTelemetry spans and events when the ``opentelemetry`` package is
available. Falls back to an in-memory log when OpenTelemetry is not installed.
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
    """Attempt to import OpenTelemetry trace components.

    Returns:
        tuple[Any, Any] | None: OUT: ``(trace_module, tracer)`` if available,
            otherwise ``None``.
    """
    try:
        from opentelemetry import trace as _trace
    except ImportError:
        return None
    try:
        return _trace, _trace.get_tracer("xerxes")
    except Exception:
        return None


class OTelCollector:
    """Collector that exports audit events to OpenTelemetry as spans and events.

    Falls back to an in-memory noop log when OpenTelemetry is unavailable.
    """

    def __init__(self, service_name: str = "xerxes") -> None:
        """Initialize the OTel collector.

        Args:
            service_name (str): IN: Service name for span attributes. OUT: Stored
                and used in span attributes.
        """
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
        """Route an audit event to the appropriate OTel handler.

        Args:
            event (AuditEvent): IN: The event to export. OUT: Dispatched to a
                typed handler or generic recorder.
        """
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
        """End all open turn spans and clear the tracking dictionary."""
        for span in list(self._open_turn_spans.values()):
            try:
                span.end()
            except Exception:
                pass
        self._open_turn_spans.clear()

    @property
    def has_otel(self) -> bool:
        """Check whether OpenTelemetry is available.

        Returns:
            bool: OUT: ``True`` if a tracer was successfully created.
        """
        return self._tracer is not None

    @property
    def fallback_log(self) -> list[dict[str, tp.Any]]:
        """Return the fallback noop log.

        Returns:
            list[dict[str, Any]]: OUT: Copy of in-memory log entries.
        """
        return list(self._noop_log)

    @contextmanager
    def _span(self, name: str, attributes: dict[str, tp.Any]):
        """Context manager for creating an OTel span or noop fallback.

        Args:
            name (str): IN: Span name. OUT: Used for the OTel span or log entry.
            attributes (dict[str, Any]): IN: Span attributes. OUT: Cleaned and
                attached to the span or log.

        Yields:
            Any | None: OUT: The active span, or ``None`` in fallback mode.
        """
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
        """Record a generic event as an OTel span event or standalone span.

        Args:
            name (str): IN: Event name. OUT: Used as the span event name.
            event (AuditEvent): IN: The audit event. OUT: Serialized to attributes.
        """
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
        """Handle a TurnStartEvent by creating an OTel span.

        Args:
            event (TurnStartEvent): IN: Turn start event. OUT: Converted to a span.
        """
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
        """Handle a TurnEndEvent by ending the associated turn span.

        Args:
            event (TurnEndEvent): IN: Turn end event. OUT: Used to close the span.
        """
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
        """Handle a ToolCallAttemptEvent.

        Args:
            event (ToolCallAttemptEvent): IN: Tool attempt event. OUT: Recorded.
        """
        self._record_event(f"tool.attempt:{event.tool_name}", event)

    def _on_tool_complete(self, event: ToolCallCompleteEvent) -> None:
        """Handle a ToolCallCompleteEvent.

        Args:
            event (ToolCallCompleteEvent): IN: Tool complete event. OUT: Recorded.
        """
        self._record_event(f"tool.complete:{event.tool_name}", event)

    def _on_tool_failure(self, event: ToolCallFailureEvent) -> None:
        """Handle a ToolCallFailureEvent.

        Args:
            event (ToolCallFailureEvent): IN: Tool failure event. OUT: Recorded.
        """
        self._record_event(f"tool.failure:{event.tool_name}", event)

    def _on_skill_used(self, event: SkillUsedEvent) -> None:
        """Handle a SkillUsedEvent.

        Args:
            event (SkillUsedEvent): IN: Skill used event. OUT: Recorded.
        """
        self._record_event(f"skill.used:{event.skill_name}", event)

    def _on_error(self, event: ErrorEvent) -> None:
        """Handle an ErrorEvent.

        Args:
            event (ErrorEvent): IN: Error event. OUT: Recorded.
        """
        self._record_event(f"error:{event.error_type}", event)


def _clean_attrs(d: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Sanitize a dictionary for OpenTelemetry attribute values.

    Removes ``None`` values and converts unsupported types to strings.

    Args:
        d (dict[str, Any]): IN: Raw attribute dictionary. OUT: Cleaned and
            truncated where necessary.

    Returns:
        dict[str, Any]: OUT: OTel-compatible attribute dictionary.
    """
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
