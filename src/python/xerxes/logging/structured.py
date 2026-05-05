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
"""Structured module for Xerxes.

Exports:
    - FUNCTION_CALLS
    - FUNCTION_DURATION
    - AGENT_SWITCHES
    - MEMORY_USAGE
    - LLM_REQUESTS
    - LLM_TOKENS
    - ERROR_COUNTER
    - XerxesLogger
    - get_logger
    - configure_logging"""

import logging
import logging.handlers
import sys
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import structlog as _structlog_module

    structlog = _structlog_module
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


def _get_prometheus() -> tuple[bool, Any, Any, Any, Callable[..., bytes]]:
    """Internal helper to get prometheus.

    Returns:
        tuple[bool, Any, Any, Any, Callable[..., bytes]]: OUT: Result of the operation."""
    try:
        from prometheus_client import Counter, Gauge, Histogram, generate_latest

        return True, Counter, Gauge, Histogram, generate_latest
    except ImportError:

        class DummyMetric:
            """Dummy metric."""

            def labels(self, **kwargs):
                """Labels.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
                Returns:
                    Any: OUT: Result of the operation."""

                return self

            def inc(self, amount=1):
                """Inc.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    amount (Any, optional): IN: amount. Defaults to 1. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""

                pass

            def dec(self, amount=1):
                """Dec.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    amount (Any, optional): IN: amount. Defaults to 1. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""

                pass

            def observe(self, value):
                """Observe.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    value (Any): IN: value. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""

                pass

        def _generate_latest(*args, **kwargs) -> bytes:
            """Internal helper to generate latest.

            Args:
                *args: IN: Additional positional arguments. OUT: Passed through to downstream calls.
                **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
            Returns:
                bytes: OUT: Result of the operation."""

            return b""

        return (
            False,
            (lambda *a, **k: DummyMetric()),
            (lambda *a, **k: DummyMetric()),
            (lambda *a, **k: DummyMetric()),
            _generate_latest,
        )


HAS_PROMETHEUS, Counter, Gauge, Histogram, generate_latest = _get_prometheus()

jsonlogger: Any = None
try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

FUNCTION_CALLS = Counter(
    "xerxes_function_calls_total", "Total number of function calls", ["agent_id", "function_name", "status"]
)

FUNCTION_DURATION = Histogram(
    "xerxes_function_duration_seconds", "Duration of function calls in seconds", ["agent_id", "function_name"]
)

AGENT_SWITCHES = Counter("xerxes_switches_total", "Total number of agent switches", ["from_agent", "to_agent"])

MEMORY_USAGE = Gauge("xerxes_memory_entries", "Number of memory entries", ["memory_type", "agent_id"])

LLM_REQUESTS = Counter("xerxes_llm_requests_total", "Total number of LLM requests", ["provider", "model", "status"])

LLM_TOKENS = Counter(
    "xerxes_llm_tokens_total",
    "Total number of tokens processed",
    ["provider", "model", "type"],
)

ERROR_COUNTER = Counter("xerxes_errors_total", "Total number of errors", ["error_type", "component"])


class XerxesLogger:
    """Xerxes logger."""

    def __init__(
        self,
        name: str = "xerxes",
        level: str = "INFO",
        log_file: Path | None = None,
        enable_json: bool = True,
        enable_tracing: bool = False,
        trace_endpoint: str | None = None,
    ):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str, optional): IN: name. Defaults to 'xerxes'. OUT: Consumed during execution.
            level (str, optional): IN: level. Defaults to 'INFO'. OUT: Consumed during execution.
            log_file (Path | None, optional): IN: log file. Defaults to None. OUT: Consumed during execution.
            enable_json (bool, optional): IN: enable json. Defaults to True. OUT: Consumed during execution.
            enable_tracing (bool, optional): IN: enable tracing. Defaults to False. OUT: Consumed during execution.
            trace_endpoint (str | None, optional): IN: trace endpoint. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to setup structlog.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to setup standard logging.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to setup tracing.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            endpoint (str | None): IN: endpoint. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to add context processor.

        Args:
            logger (Any): IN: logger. OUT: Consumed during execution.
            log_method (Any): IN: log method. OUT: Consumed during execution.
            event_dict (Any): IN: event dict. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Log function call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
            function_name (str): IN: function name. OUT: Consumed during execution.
            arguments (dict[str, Any]): IN: arguments. OUT: Consumed during execution.
            result (Any, optional): IN: result. Defaults to None. OUT: Consumed during execution.
            error (Exception | None, optional): IN: error. Defaults to None. OUT: Consumed during execution.
            duration (float, optional): IN: duration. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Log agent switch.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            from_agent (str): IN: from agent. OUT: Consumed during execution.
            to_agent (str): IN: to agent. OUT: Consumed during execution.
            reason (str | None, optional): IN: reason. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Log llm request.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            provider (str): IN: provider. OUT: Consumed during execution.
            model (str): IN: model. OUT: Consumed during execution.
            prompt_tokens (int): IN: prompt tokens. OUT: Consumed during execution.
            completion_tokens (int): IN: completion tokens. OUT: Consumed during execution.
            duration (float): IN: duration. OUT: Consumed during execution.
            error (Exception | None, optional): IN: error. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Log memory operation.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            operation (str): IN: operation. OUT: Consumed during execution.
            memory_type (str): IN: memory type. OUT: Consumed during execution.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
            entry_count (int, optional): IN: entry count. Defaults to 1. OUT: Consumed during execution.
            error (Exception | None, optional): IN: error. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Span.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            **attributes: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Retrieve the metrics.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bytes: OUT: Result of the operation."""

        return generate_latest()


_logger: XerxesLogger | None = None


def get_logger(name: str | None = None, **kwargs) -> XerxesLogger:
    """Retrieve the logger.

    Args:
        name (str | None, optional): IN: name. Defaults to None. OUT: Consumed during execution.
        **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        XerxesLogger: OUT: Result of the operation."""

    global _logger

    if _logger is None:
        from ..core.config import get_config

        config = get_config()

        _logger = XerxesLogger(
            name=name or "xerxes",
            level=config.logging.level.value,
            log_file=Path(config.logging.file_path) if config.logging.file_path else None,
            enable_json=config.logging.enable_json_format,
            enable_tracing=config.observability.enable_tracing,
            trace_endpoint=config.observability.trace_endpoint,
        )

    return _logger


def configure_logging(**kwargs):
    """Configure logging.

    Args:
        **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    global _logger
    _logger = XerxesLogger(**kwargs)
