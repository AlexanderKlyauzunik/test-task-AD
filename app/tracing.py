"""Phoenix Observability Integration - Configures OpenTelemetry tracing."""
import os
from contextlib import contextmanager

from phoenix.otel import register

# Initialize Phoenix tracer on module import
collector_endpoint = os.getenv("COLLECTOR_ENDPOINT", "phoenix:4317")
tracer_provider = register(
    project_name="agentic-rag",
    endpoint=collector_endpoint,
    auto_instrument=True,
)
otel_tracer = tracer_provider.get_tracer(__name__)


@contextmanager
def active_trace_span(span_name: str, attributes: dict | None = None):
    """Context manager for creating a new, active OpenTelemetry span."""
    with otel_tracer.start_as_current_span(span_name) as current_span:
        if attributes and current_span:
            for key, value in attributes.items():
                if value is not None:
                    # Attributes must be strings for OpenTelemetry
                    current_span.set_attribute(key, str(value))
        yield current_span


def trace_function_execution(span_name: str):
    """Decorator for tracing function execution with a new span."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with active_trace_span(span_name, {"function": func.__name__}):
                return func(*args, **kwargs)
        return wrapper
    return decorator
