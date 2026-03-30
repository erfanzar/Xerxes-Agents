# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Structured logging configuration for Calute.

This module provides an advanced logging configuration system for the
Calute framework with support for:
- Structured logging via structlog (when available)
- OpenTelemetry distributed tracing integration
- Prometheus metrics for observability
- JSON logging format support
- Rotating file handlers for log persistence

The module gracefully degrades when optional dependencies are not installed,
providing fallback implementations where necessary.

Example:
    >>> from calute.logging.structured import get_logger, configure_logging
    >>> configure_logging(level="DEBUG", enable_json=True)
    >>> logger = get_logger()
    >>> logger.log_function_call("agent1", "search", {"query": "test"})
"""

import logging
import logging.handlers
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import structlog  # type:ignore

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type:ignore
    from opentelemetry.instrumentation.logging import LoggingInstrumentor  # type:ignore
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

    class DummyMetric:
        """No-op metric implementation for when prometheus_client is not installed.

        Provides the same interface as Prometheus ``Counter``, ``Gauge``, and
        ``Histogram`` metrics but performs no actual operations, allowing code
        to function without the ``prometheus_client`` dependency.

        All methods are intentionally no-ops and return ``self`` where
        appropriate to support fluent method chaining (e.g.,
        ``metric.labels(foo="bar").inc()``).

        Attributes:
            None. This class carries no state.
        """

        def labels(self, **kwargs):
            """Return self for method chaining compatibility.

            Mimics the Prometheus ``labels()`` method which returns a
            child metric filtered by the given label values.

            Args:
                **kwargs: Label key-value pairs (ignored).

            Returns:
                ``self``, allowing subsequent calls to ``inc``, ``dec``,
                or ``observe`` to succeed silently.
            """
            return self

        def inc(self, amount=1):
            """No-op increment operation.

            Mimics the Prometheus ``Counter.inc`` / ``Gauge.inc`` method.

            Args:
                amount: The amount to increment by (ignored). Defaults to 1.
            """
            pass

        def dec(self, amount=1):
            """No-op decrement operation.

            Mimics the Prometheus ``Gauge.dec`` method.

            Args:
                amount: The amount to decrement by (ignored). Defaults to 1.
            """
            pass

        def observe(self, value):
            """No-op observation operation.

            Mimics the Prometheus ``Histogram.observe`` method.

            Args:
                value: The observed value (ignored).
            """
            pass

    Counter = Histogram = Gauge = lambda *args, **kwargs: DummyMetric()

    def generate_latest():
        """Return empty bytes when prometheus_client is not installed.

        This is a fallback replacement for
        ``prometheus_client.generate_latest`` that returns an empty byte
        string when the Prometheus client library is unavailable.

        Returns:
            An empty ``bytes`` object (``b""``).
        """
        return b""


try:
    from pythonjsonlogger import jsonlogger  # type:ignore

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False
    jsonlogger = None


FUNCTION_CALLS = Counter(
    "calute_function_calls_total", "Total number of function calls", ["agent_id", "function_name", "status"]
)

FUNCTION_DURATION = Histogram(
    "calute_function_duration_seconds", "Duration of function calls in seconds", ["agent_id", "function_name"]
)

AGENT_SWITCHES = Counter("calute_agent_switches_total", "Total number of agent switches", ["from_agent", "to_agent"])

MEMORY_USAGE = Gauge("calute_memory_entries", "Number of memory entries", ["memory_type", "agent_id"])

LLM_REQUESTS = Counter("calute_llm_requests_total", "Total number of LLM requests", ["provider", "model", "status"])

LLM_TOKENS = Counter(
    "calute_llm_tokens_total",
    "Total number of tokens processed",
    ["provider", "model", "type"],
)

ERROR_COUNTER = Counter("calute_errors_total", "Total number of errors", ["error_type", "component"])


class CaluteLogger:
    """Enhanced logger with structured logging and tracing.

    Provides a comprehensive logging solution that integrates with
    structlog for structured logging, OpenTelemetry for distributed
    tracing, and Prometheus for metrics collection.

    Attributes:
        name: Logger name identifier.
        level: Numeric logging level.
        log_file: Optional path to log file for persistence.
        enable_json: Whether JSON output format is enabled.
        enable_tracing: Whether OpenTelemetry tracing is enabled.
        logger: The underlying logger instance (structlog or standard).

    Note:
        Falls back to standard Python logging if structlog is not installed.
    """

    def __init__(
        self,
        name: str = "calute",
        level: str = "INFO",
        log_file: Path | None = None,
        enable_json: bool = True,
        enable_tracing: bool = False,
        trace_endpoint: str | None = None,
    ):
        """Initialize the CaluteLogger with the specified configuration.

        Args:
            name: Logger name for identification in logs.
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            log_file: Optional path for file-based log persistence.
            enable_json: Whether to format logs as JSON.
            enable_tracing: Whether to enable OpenTelemetry tracing.
            trace_endpoint: OTLP endpoint for trace export.
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.enable_json = enable_json
        self.enable_tracing = enable_tracing

        self._setup_structlog()

        self._setup_standard_logging()

        if enable_tracing:
            self._setup_tracing(trace_endpoint)

        if HAS_STRUCTLOG:
            self.logger = structlog.get_logger(name)
            self._use_structlog = True
        else:
            self.logger = logging.getLogger(name)
            self._use_structlog = False

    def _setup_structlog(self):
        """Configure structlog for structured logging output.

        Sets up the structlog processor chain including log level filtering,
        timestamp formatting, stack info rendering, and output formatting
        (JSON or console-friendly).

        Note:
            This method is a no-op if structlog is not installed.
        """
        if not HAS_STRUCTLOG:
            return

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            self._add_context_processor,
        ]

        if self.enable_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _setup_standard_logging(self):
        """Configure standard Python logging with console and optional file handlers.

        Sets up the root logger with appropriate formatters based on
        configuration. Includes support for JSON formatting and rotating
        file handlers for log persistence.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        if self.enable_json and HAS_JSON_LOGGER:
            formatter = jsonlogger.JsonFormatter(
                "%(timestamp)s %(level)s %(name)s %(message)s",
                rename_fields={"timestamp": "@timestamp", "level": "log.level"},
            )
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def _setup_tracing(self, endpoint: str | None):
        """Configure OpenTelemetry distributed tracing.

        Sets up a TracerProvider with OTLP export capabilities for
        distributed tracing across service boundaries.

        Args:
            endpoint: OTLP endpoint URL for trace export. If None,
                tracing is set up without an exporter.

        Note:
            This method is a no-op if OpenTelemetry is not installed.
        """
        if not HAS_OTEL:
            return

        resource = Resource.create(
            {
                "service.name": self.name,
                "service.version": "0.0.18",
            }
        )

        provider = TracerProvider(resource=resource)

        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

        LoggingInstrumentor().instrument(set_logging_format=True)

    @staticmethod
    def _add_context_processor(logger, log_method, event_dict):
        """Add contextual information to log events.

        Structlog processor that enriches log events with timestamps
        and OpenTelemetry trace context for correlation.

        Args:
            logger: The structlog logger instance.
            log_method: The log method being called.
            event_dict: The event dictionary to enrich.

        Returns:
            The enriched event dictionary with timestamp and trace IDs.
        """
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.utcnow().isoformat()

        if HAS_OTEL and trace:
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                event_dict["trace_id"] = format(span_context.trace_id, "032x")
                event_dict["span_id"] = format(span_context.span_id, "016x")

        return event_dict

    def log_function_call(
        self,
        agent_id: str,
        function_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
        duration: float = 0.0,
    ):
        """Log a function call with metrics and structured data.

        Records function execution details including arguments, results,
        and timing. Updates Prometheus metrics for monitoring.

        Args:
            agent_id: Identifier of the agent executing the function.
            function_name: Name of the function being called.
            arguments: Dictionary of function arguments.
            result: Optional function result (truncated to 200 chars in logs).
            error: Optional exception if the function failed.
            duration: Execution duration in seconds.
        """
        status = "success" if error is None else "error"

        FUNCTION_CALLS.labels(agent_id=agent_id, function_name=function_name, status=status).inc()

        FUNCTION_DURATION.labels(agent_id=agent_id, function_name=function_name).observe(duration)

        log_data = {
            "event": "function_call",
            "agent_id": agent_id,
            "function_name": function_name,
            "arguments": arguments,
            "duration": duration,
            "status": status,
        }

        if result is not None:
            log_data["result"] = str(result)[:200]

        if error:
            log_data["error"] = str(error)
            ERROR_COUNTER.labels(error_type=type(error).__name__, component="function_executor").inc()
            if self._use_structlog:
                self.logger.error("Function call failed", **log_data)
            else:
                self.logger.error(f"Function call failed: {log_data}")
        else:
            if self._use_structlog:
                self.logger.info("Function call completed", **log_data)
            else:
                self.logger.info(f"Function call completed: {log_data['function_name']} in {duration:.2f}s")

    def log_agent_switch(
        self,
        from_agent: str,
        to_agent: str,
        reason: str | None = None,
    ):
        """Log an agent switch event with metrics.

        Records when control transfers from one agent to another,
        updating the agent switch counter for monitoring.

        Args:
            from_agent: Name of the agent relinquishing control.
            to_agent: Name of the agent receiving control.
            reason: Optional reason for the switch.
        """
        AGENT_SWITCHES.labels(from_agent=from_agent, to_agent=to_agent).inc()

        if self._use_structlog:
            self.logger.info(
                "Agent switch",
                event="agent_switch",
                from_agent=from_agent,
                to_agent=to_agent,
                reason=reason,
            )
        else:
            self.logger.info(f"Agent switch from {from_agent} to {to_agent}, reason: {reason}")

    def log_llm_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
        error: Exception | None = None,
    ):
        """Log an LLM API request with token and timing metrics.

        Records LLM request details for monitoring and cost tracking,
        updating Prometheus counters for requests and tokens.

        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic').
            model: Model identifier (e.g., 'gpt-4', 'claude-3').
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            duration: Request duration in seconds.
            error: Optional exception if the request failed.
        """
        status = "success" if error is None else "error"

        LLM_REQUESTS.labels(provider=provider, model=model, status=status).inc()

        LLM_TOKENS.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)

        LLM_TOKENS.labels(provider=provider, model=model, type="completion").inc(completion_tokens)

        log_data = {
            "event": "llm_request",
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration": duration,
            "status": status,
        }

        if error:
            log_data["error"] = str(error)
            ERROR_COUNTER.labels(error_type=type(error).__name__, component="llm_client").inc()
            if self._use_structlog:
                self.logger.error("LLM request failed", **log_data)
            else:
                self.logger.error(f"LLM request failed: {log_data}")
        else:
            if self._use_structlog:
                self.logger.info("LLM request completed", **log_data)
            else:
                self.logger.info(
                    f"LLM request completed: {provider} {model}, tokens: {prompt_tokens}+{completion_tokens}"
                )

    def log_memory_operation(
        self,
        operation: str,
        memory_type: str,
        agent_id: str,
        entry_count: int = 1,
        error: Exception | None = None,
    ):
        """Log a memory store operation with metrics.

        Records memory operations (add, remove) and updates the
        memory usage gauge for monitoring.

        Args:
            operation: Type of operation ('add' or 'remove').
            memory_type: Type of memory (e.g., 'short_term', 'long_term').
            agent_id: Identifier of the agent performing the operation.
            entry_count: Number of entries affected by the operation.
            error: Optional exception if the operation failed.
        """
        if operation == "add":
            MEMORY_USAGE.labels(memory_type=memory_type, agent_id=agent_id).inc(entry_count)
        elif operation == "remove":
            MEMORY_USAGE.labels(memory_type=memory_type, agent_id=agent_id).dec(entry_count)

        log_data = {
            "event": "memory_operation",
            "operation": operation,
            "memory_type": memory_type,
            "agent_id": agent_id,
            "entry_count": entry_count,
        }

        if error:
            log_data["error"] = str(error)
            ERROR_COUNTER.labels(error_type=type(error).__name__, component="memory_store").inc()
            if self._use_structlog:
                self.logger.error("Memory operation failed", **log_data)
            else:
                self.logger.error(f"Memory operation failed: {log_data}")
        else:
            if self._use_structlog:
                self.logger.debug("Memory operation completed", **log_data)
            else:
                self.logger.debug(f"Memory operation {operation} completed for {agent_id}")

    @contextmanager
    def span(self, name: str, **attributes):
        """Create a tracing span context manager.

        Creates an OpenTelemetry span for distributed tracing. The span
        automatically records exceptions and sets error status on failure.

        Args:
            name: Name for the span.
            **attributes: Additional attributes to attach to the span.

        Yields:
            The created span object, or None if tracing is disabled.

        Example:
            >>> with logger.span("process_request", user_id="123"):
            ...     process_data()
        """
        if self.enable_tracing and HAS_OTEL:
            tracer = trace.get_tracer(self.name)
            with tracer.start_as_current_span(name, attributes=attributes) as span:
                try:
                    yield span
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        else:
            yield None

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format.

        Returns:
            Prometheus metrics formatted for HTTP exposition.
        """
        return generate_latest()


_logger: CaluteLogger | None = None


def get_logger(name: str | None = None, **kwargs) -> CaluteLogger:
    """Get or create the global CaluteLogger instance.

    Lazily initializes the logger using configuration from the global
    Calute config. Subsequent calls return the same instance.

    Args:
        name: Optional logger name (used only on first initialization).
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        The global CaluteLogger instance.

    Example:
        >>> logger = get_logger()
        >>> logger.log_function_call("agent1", "search", {"q": "test"})
    """
    global _logger

    if _logger is None:
        from ..core.config import get_config

        config = get_config()

        _logger = CaluteLogger(
            name=name or "calute",
            level=config.logging.level.value,
            log_file=Path(config.logging.file_path) if config.logging.file_path else None,
            enable_json=config.logging.enable_json_format,
            enable_tracing=config.observability.enable_tracing,
            trace_endpoint=config.observability.trace_endpoint,
        )

    return _logger


def configure_logging(**kwargs):
    """Configure global logging settings with custom parameters.

    Creates a new CaluteLogger with the specified settings and sets
    it as the global logger instance.

    Args:
        **kwargs: Configuration options passed to CaluteLogger:
            - name: Logger name identifier.
            - level: Log level string.
            - log_file: Optional Path for log file.
            - enable_json: Whether to use JSON format.
            - enable_tracing: Whether to enable tracing.
            - trace_endpoint: OTLP endpoint for traces.

    Example:
        >>> configure_logging(level="DEBUG", enable_json=False)
    """
    global _logger
    _logger = CaluteLogger(**kwargs)
