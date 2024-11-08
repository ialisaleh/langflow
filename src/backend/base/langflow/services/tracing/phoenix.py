from __future__ import annotations

import os
import traceback
import types
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from typing_extensions import override

from langflow.schema.data import Data
from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain.callbacks.base import BaseCallbackHandler

    from langflow.graph.vertex.base import Vertex
    from langflow.services.tracing.schema import Log


class PhoenixTracer(BaseTracer):
    flow_id: str

    def __init__(self, trace_name: str, trace_type: str, project_name: str, trace_id: UUID):
        """Initializes the PhoenixTracer instance and sets up a root span."""
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id
        self.flow_id = trace_name.split(" - ")[-1]

        try:
            self._ready = self.setup_phoenix()
            if not self._ready:
                return

            self.tracer = self.tracer_provider.get_tracer(__name__)
            self.propagator = TraceContextTextMapPropagator()
            self.carrier = {}

            with self.tracer.start_as_current_span(
                name=self.flow_id,
                start_time=self._get_current_timestamp(),
            ) as root_span:
                root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, self.trace_type)
                self.propagator.inject(carrier=self.carrier)

            self.child_spans = {}

        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug("Error setting up Phoenix tracer")
            self._ready = False

    @property
    def ready(self):
        """Indicates if the tracer is ready for usage."""
        return self._ready

    def setup_phoenix(self) -> bool:
        """Configures Phoenix-specific environment variables and registers the tracer provider."""
        api_key = os.getenv("PHOENIX_API_KEY", None)
        if api_key is None:
            return False

        try:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={api_key}"
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"

            from phoenix.otel import register
            self.tracer_provider = register(project_name=self.project_name, set_global_tracer_provider=False)

        except ImportError:
            logger.exception("Could not import phoenix. Please install it with `pip install arize-phoenix-otel`.")
            return False

        return True

    @override
    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        """Adds a trace span, attaching inputs and metadata as attributes."""
        if not self._ready:
            return

        span_name = trace_name.removesuffix(f" ({trace_id})")
        span_context = self.propagator.extract(carrier=self.carrier)
        child_span = self.tracer.start_span(
            name=span_name,
            context=span_context,
            start_time=self._get_current_timestamp()
        )
        child_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, trace_type)

        if inputs:
            processed_inputs = self._convert_to_phoenix_types(inputs)
            for key, value in processed_inputs.items():
                child_span.set_attribute(f"input.{key}", value)

        if metadata:
            processed_metadata = self._convert_to_phoenix_types(metadata)
            for key, value in processed_metadata.items():
                child_span.set_attribute(f"metadata.{key}", value)

        self.child_spans[trace_id] = child_span

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        """Ends a trace span, attaching outputs, errors, and logs as attributes."""
        if not self._ready or trace_id not in self.child_spans:
            return

        child_span = self.child_spans[trace_id]
        if outputs:
            processed_outputs = self._convert_to_phoenix_types(outputs)
            for key, value in processed_outputs.items():
                child_span.set_attribute(f"output.{key}", str(value))

        if error:
            child_span.set_status(Status(StatusCode.ERROR))
            child_span.set_attribute("error.message", self._error_to_string(error))

        if logs:
            logs_dicts = [log if isinstance(log, dict) else log.model_dump() for log in logs]
            processed_logs = self._convert_to_phoenix_types({"logs": {log.get("name"): log for log in logs_dicts}})
            for key, value in processed_logs.items():
                child_span.set_attribute(f"log.{key}", value)

        child_span.set_status(Status(StatusCode.OK))
        child_span.end(end_time=self._get_current_timestamp())
        self.child_spans.pop(trace_id)

    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Ends tracing with the specified inputs, outputs, errors, and metadata as attributes."""
        if not self._ready:
            return

    def _convert_to_phoenix_types(self, io_dict: dict[str, Any]):
        """Converts data types to Phoenix-compatible formats."""
        converted = {}
        for key, value in io_dict.items():
            converted[key] = self._convert_to_phoenix_type(value)
        return converted

    def _convert_to_phoenix_type(self, value):
        """Recursively converts a value to a Phoenix-compatible type."""
        from langflow.schema.message import Message

        if isinstance(value, dict):
            for key, _value in value.copy().items():
                _value = self._convert_to_phoenix_type(_value)
                value[key] = _value

        elif isinstance(value, list):
            value = [self._convert_to_phoenix_type(v) for v in value]

        elif isinstance(value, Message):
            if "prompt" in value:
                value = value.load_lc_prompt()
            elif value.sender:
                value = value.to_lc_message()
            else:
                value = value.to_lc_document()

        elif isinstance(value, Data):
            value = value.to_lc_document()

        elif isinstance(value, types.GeneratorType):
            value = str(value)

        return value

    def _error_to_string(self, error: Exception | None):
        """Converts an error to a string with traceback details."""
        error_message = None
        if error:
            string_stacktrace = traceback.format_exception(error)
            error_message = f"{error.__class__.__name__}: {error}\n\n{string_stacktrace}"
        return error_message

    def _get_current_timestamp(self) -> int:
        """Gets the current UTC timestamp in nanoseconds."""
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        """Returns the LangChain callback handler if applicable."""
        return None
