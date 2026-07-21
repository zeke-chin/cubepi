"""Backend-specific span attribute adapters.

Adapters run synchronously while Cubepi spans are still writable. They derive
backend attributes from Cubepi's standard GenAI semantic-convention attributes;
exporters remain ordinary OpenTelemetry exporters.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from cubepi.tracing.schema import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    OP_INVOKE_AGENT,
)


class SpanAdapter(Protocol):
    """Synchronous hooks for deriving additional writable span attributes."""

    def on_span_start(self, span: Any) -> None:
        """Called immediately after Cubepi creates a span."""

    def on_run_context(
        self,
        span: Any,
        *,
        session_id: str | None,
        user_id: str | None,
        tags: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> None:
        """Called after task-local run context is applied to a root span."""

    def on_content(self, span: Any, *, key: str, value: Any) -> None:
        """Called after a redacted content attribute is recorded."""


class LangfuseSpanAdapter:
    """Map Cubepi GenAI attributes to Langfuse's OTLP attribute contract.

    The adapter does not configure transport or authentication. Pair it with an
    ordinary OTLP/HTTP exporter pointed at Langfuse.
    """

    def on_span_start(self, span: Any) -> None:
        if _is_root_agent_span(span):
            span.set_attribute("langfuse.trace.name", span.name)

    def on_run_context(
        self,
        span: Any,
        *,
        session_id: str | None,
        user_id: str | None,
        tags: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        if session_id:
            span.set_attribute("session.id", session_id)
        if user_id:
            span.set_attribute("user.id", user_id)
        if tags:
            span.set_attribute("langfuse.trace.tags", tags)

    def on_content(self, span: Any, *, key: str, value: Any) -> None:
        del value
        if key in (GEN_AI_SYSTEM_INSTRUCTIONS, GEN_AI_INPUT_MESSAGES):
            envelope = _input_envelope(span)
            if envelope is not None:
                encoded = _json(envelope)
                span.set_attribute("langfuse.observation.input", encoded)
                if _is_root_agent_span(span):
                    span.set_attribute("langfuse.trace.input", encoded)
        elif key == GEN_AI_OUTPUT_MESSAGES:
            messages = _attribute_json(span, GEN_AI_OUTPUT_MESSAGES)
            flattened = _flatten_messages(messages)
            if flattened:
                encoded = _json({"messages": flattened})
                span.set_attribute("langfuse.observation.output", encoded)
                if _is_root_agent_span(span):
                    span.set_attribute("langfuse.trace.output", encoded)


def _is_root_agent_span(span: Any) -> bool:
    attributes = getattr(span, "attributes", None) or {}
    return attributes.get(GEN_AI_OPERATION_NAME) == OP_INVOKE_AGENT


def _input_envelope(span: Any) -> dict[str, Any] | None:
    system = _flatten_messages(_attribute_json(span, GEN_AI_SYSTEM_INSTRUCTIONS))
    inputs = _flatten_messages(_attribute_json(span, GEN_AI_INPUT_MESSAGES))
    if system and any(message.get("role") == "system" for message in inputs):
        system = []
    messages = [*system, *inputs]
    return {"messages": messages} if messages else None


def _attribute_json(span: Any, key: str) -> Any:
    attributes = getattr(span, "attributes", None) or {}
    value = attributes.get(key)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _flatten_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, list):
        return []

    flattened: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            content = message.get("content")
            flattened.append({"role": role, "content": content})
            continue

        texts: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_result: dict[str, Any] | None = None
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("content"), str):
                texts.append(part["content"])
            elif part_type == "reasoning" and isinstance(part.get("content"), str):
                reasoning.append(part["content"])
            elif part_type == "tool_call":
                tool_calls.append(
                    {
                        "id": part.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": part.get("name") or "",
                            "arguments": part.get("arguments") or {},
                        },
                    }
                )
            elif part_type == "tool_call_response":
                tool_result = {
                    "role": "tool",
                    "tool_call_id": part.get("id") or "",
                    "content": part.get("result") or "",
                }

        if tool_result is not None:
            flattened.append(tool_result)
            continue
        item: dict[str, Any] = {
            "role": role,
            "content": "".join(texts) if texts else None,
        }
        if reasoning:
            item["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            item["tool_calls"] = tool_calls
        flattened.append(item)
    return flattened


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
