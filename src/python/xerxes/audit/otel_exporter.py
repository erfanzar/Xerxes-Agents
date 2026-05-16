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
"""Translate :class:`AuditEvent` objects into OpenTelemetry spans.

Each turn becomes a parent span (``xerxes.turn``); tool attempts,
completions, failures, skill uses, and errors are attached to that
span as OpenTelemetry events when the turn is in flight, otherwise
emitted as short standalone spans. When ``opentelemetry`` is not
installed the collector falls back to an in-memory dict log accessible
via :attr:`OTelCollector.fallback_log` for tests.
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
    """Return ``(trace, tracer)`` if OpenTelemetry imports cleanly, else ``None``."""
    try:
        from opentelemetry import trace as _trace
    except ImportError:
        return None
    try:
        return _trace, _trace.get_tracer("xerxes")
    except Exception:
        return None


class OTelCollector:
    """Audit collector backed by OpenTelemetry tracing.

    Each ``TurnStartEvent`` opens a span; intermediate events attach as
    span events; ``TurnEndEvent`` closes the span. Without OTel the
    collector buffers entries in :attr:`fallback_log` instead.
    """

    def __init__(self, service_name: str = "xerxes") -> None:
        """Construct the collector and try to acquire a tracer.

        ``service_name`` is attached to every span as the ``service.name``
        attribute and used by downstream OTel collectors for grouping.
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
        """Dispatch ``event`` to its type-specific handler.

        Unknown event types are recorded with a generic name. Exceptions
        from the OTel SDK are caught and logged so audit failures never
        propagate to the caller.
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
        """End every still-open turn span (used at shutdown)."""
        for span in list(self._open_turn_spans.values()):
            try:
                span.end()
            except Exception:
                pass
        self._open_turn_spans.clear()

    @property
    def has_otel(self) -> bool:
        """Return ``True`` when an OpenTelemetry tracer is available."""
        return self._tracer is not None

    @property
    def fallback_log(self) -> list[dict[str, tp.Any]]:
        """Return a copy of the in-memory log used when OTel is missing."""
        return list(self._noop_log)

    @contextmanager
    def _span(self, name: str, attributes: dict[str, tp.Any]):
        """Yield a short-lived span or append a fallback log entry."""
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
        """Attach ``event`` to its turn span, or emit a standalone span."""
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
        """Open and remember a ``xerxes.turn`` span keyed by ``turn_id``."""
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
        """Close the turn span and record the final function-call count."""
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
        """Record a ``tool.attempt:<name>`` event."""
        self._record_event(f"tool.attempt:{event.tool_name}", event)

    def _on_tool_complete(self, event: ToolCallCompleteEvent) -> None:
        """Record a ``tool.complete:<name>`` event."""
        self._record_event(f"tool.complete:{event.tool_name}", event)

    def _on_tool_failure(self, event: ToolCallFailureEvent) -> None:
        """Record a ``tool.failure:<name>`` event."""
        self._record_event(f"tool.failure:{event.tool_name}", event)

    def _on_skill_used(self, event: SkillUsedEvent) -> None:
        """Record a ``skill.used:<name>`` event."""
        self._record_event(f"skill.used:{event.skill_name}", event)

    def _on_error(self, event: ErrorEvent) -> None:
        """Record an ``error:<type>`` event."""
        self._record_event(f"error:{event.error_type}", event)


def _clean_attrs(d: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Coerce ``d`` to OpenTelemetry-safe attribute values.

    Drops ``None``, stringifies non-primitives (truncated to 200 chars),
    and renders datetimes as ISO strings. The OTel SDK only accepts
    primitive scalar values; objects survive only as their ``str`` form.
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
