from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from opentelemetry.trace import get_tracer, SpanKind
from typing_extensions import override

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
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id
        self.flow_id = trace_name.split(" - ")[-1]

        try:
            self._ready = self.setup_phoenix()
            if not self._ready:
                return

            self.tracer = get_tracer(__name__)
            self._span = self.tracer.start_span(
                name=self.trace_name,
                kind=SpanKind.SERVER,
                start_time=self._get_current_timestamp(),
            )
            self._span.set_attribute("project.name", self.project_name)
            self._span.set_attribute("trace.type", self.trace_type)
            self._children = {}

        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug("Error setting up Phoenix tracer")
            self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup_phoenix(self) -> bool:
        api_key = os.getenv("PHOENIX_API_KEY", None)
        if api_key is None:
            return False

        try:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={api_key}"
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"

            from phoenix.otel import register
            self.tracer_provider = register(project_name=self.project_name)

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
        if not self._ready:
            return

        context = self._span.get_span_context() if self._span else None
        child_span = self.tracer.start_span(
            name=trace_name,
            kind=SpanKind.INTERNAL,
            context=context,
            start_time=self._get_current_timestamp()
        )
        child_span.set_attribute("trace.type", trace_type)

        for key, value in inputs.items():
            child_span.set_attribute(f"input.{key}", str(value))

        if metadata:
            for key, value in metadata.items():
                child_span.set_attribute(f"metadata.{key}", str(value))

        self._children[trace_name] = child_span

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        if not self._ready or trace_name not in self._children:
            return

        child_span = self._children[trace_name]
        if outputs:
            for key, value in outputs.items():
                child_span.set_attribute(f"output.{key}", str(value))

        if error:
            child_span.set_status("Error")
            child_span.set_attribute("error.message", self._error_to_string(error))

        for log in logs:
            log_data = log if isinstance(log, dict) else log.model_dump()
            for key, value in log_data.items():
                child_span.set_attribute(f"log.{key}", str(value))

        child_span.end(end_time=self._get_current_timestamp())
        self._children.pop(trace_name)

    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ready:
            return

        for key, value in inputs.items():
            self._span.set_attribute(f"input.{key}", str(value))

        for key, value in outputs.items():
            self._span.set_attribute(f"output.{key}", str(value))

        if error:
            self._span.set_status("Error")
            self._span.set_attribute("error.message", self._error_to_string(error))

        if metadata:
            for key, value in metadata.items():
                self._span.set_attribute(f"metadata.{key}", str(value))

        self._span.end(end_time=self._get_current_timestamp())

    def _error_to_string(self, error: Exception | None):
        error_message = None
        if error:
            string_stacktrace = traceback.format_exception(error)
            error_message = f"{error.__class__.__name__}: {error}\n\n{string_stacktrace}"
        return error_message

    def _get_current_timestamp() -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        return None
