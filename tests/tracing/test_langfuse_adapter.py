from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cubepi import Agent
from cubepi.providers.base import Model, TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer, tracing_context
from cubepi.tracing.adapters import LangfuseSpanAdapter


MODEL = Model(id="faux-1", provider_id="faux")


class _Capture(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _json_attr(span: ReadableSpan, key: str) -> Any:
    return json.loads(_attrs(span)[key])


async def _run(*, record_content: bool = True):
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("sunny")])
    agent = Agent(
        model=provider.model(MODEL.id),
        system_prompt="You are a weather assistant.",
    )
    exporter = _Capture()
    tracer = Tracer(
        service_name="test",
        agent_name="weather",
        exporters=[exporter],
        span_adapters=[LangfuseSpanAdapter()],
        record_content=record_content,
    )
    tracer.attach(agent)
    with tracing_context(
        session_id="session-123",
        user_id="user-456",
        tags=["qa", "weather"],
        metadata={"tenant": "demo"},
    ):
        await agent.prompt("Tokyo weather?")
        await agent.wait_for_idle()
    await tracer.shutdown()
    return exporter.spans


async def test_langfuse_root_identity_and_content_mapping():
    spans = await _run()
    root = next(span for span in spans if span.name == "invoke_agent")
    attrs = _attrs(root)

    assert attrs["session.id"] == "session-123"
    assert attrs["user.id"] == "user-456"
    assert attrs["langfuse.trace.tags"] == ("qa", "weather")
    assert attrs["langfuse.trace.name"] == "invoke_agent"
    assert attrs["cubepi.metadata.tenant"] == "demo"

    trace_input = _json_attr(root, "langfuse.trace.input")
    assert trace_input["messages"] == [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "Tokyo weather?"},
    ]
    trace_output = _json_attr(root, "langfuse.trace.output")
    assert trace_output["messages"][0] == {
        "role": "assistant",
        "content": "sunny",
    }


async def test_langfuse_chat_has_normalized_input_and_output():
    spans = await _run()
    chat = next(span for span in spans if span.name.startswith("chat "))

    chat_input = _json_attr(chat, "langfuse.observation.input")
    assert chat_input["messages"][0]["role"] == "system"
    assert isinstance(chat_input["messages"][0]["content"], str)
    assert chat_input["messages"][1] == {
        "role": "user",
        "content": "Tokyo weather?",
    }

    output_messages = _json_attr(chat, "gen_ai.output.messages")
    assert output_messages[0]["parts"][0]["content"] == "sunny"
    langfuse_output = _json_attr(chat, "langfuse.observation.output")
    assert langfuse_output["messages"][0]["content"] == "sunny"


async def test_record_content_false_keeps_langfuse_content_empty():
    spans = await _run(record_content=False)
    for span in spans:
        attrs = _attrs(span)
        assert "langfuse.observation.input" not in attrs
        assert "langfuse.observation.output" not in attrs
        assert "langfuse.trace.input" not in attrs
        assert "langfuse.trace.output" not in attrs


async def test_nested_session_and_user_context_override_then_reset():
    provider = FauxProvider(provider_id="faux")
    provider.append_responses(
        [faux_assistant_message("one"), faux_assistant_message("two")]
    )
    agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
    exporter = _Capture()
    tracer = Tracer(exporters=[exporter], span_adapters=[LangfuseSpanAdapter()])
    tracer.attach(agent)

    with tracing_context(session_id="outer", user_id="outer-user"):
        with tracing_context(session_id="inner"):
            await agent.prompt("one")
            await agent.wait_for_idle()
    await agent.prompt("two")
    await agent.wait_for_idle()
    await tracer.shutdown()

    roots = sorted(
        (span for span in exporter.spans if span.name == "invoke_agent"),
        key=lambda span: span.start_time or 0,
    )
    assert _attrs(roots[0])["session.id"] == "inner"
    assert _attrs(roots[0])["user.id"] == "outer-user"
    assert "session.id" not in _attrs(roots[1])
    assert "user.id" not in _attrs(roots[1])


async def test_oneshot_applies_langfuse_context_and_content():
    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("answer")])
    exporter = _Capture()
    tracer = Tracer(
        exporters=[exporter],
        span_adapters=[LangfuseSpanAdapter()],
        record_content=True,
    )

    with tracing_context(session_id="oneshot-session", user_id="oneshot-user"):
        async with tracer.oneshot(
            model=provider.model(MODEL.id), metadata={"source": "job"}
        ) as session:
            await session.generate(
                system="system",
                messages=[UserMessage(content=[TextContent(text="question")])],
                max_output_tokens=100,
            )
    await tracer.shutdown()

    root = next(span for span in exporter.spans if span.name == "invoke_agent")
    attrs = _attrs(root)
    assert attrs["session.id"] == "oneshot-session"
    assert attrs["user.id"] == "oneshot-user"
    assert attrs["cubepi.metadata.source"] == "job"
    assert _json_attr(root, "langfuse.trace.input")["messages"][0] == {
        "role": "system",
        "content": "system",
    }
    assert (
        _json_attr(root, "langfuse.trace.output")["messages"][0]["content"] == "answer"
    )


async def test_adapter_failure_does_not_break_agent_run():
    class BrokenAdapter:
        def on_span_start(self, span):  # noqa: ANN001
            raise RuntimeError("boom")

        def on_run_context(self, span, **kwargs):  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

        def on_content(self, span, **kwargs):  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

    provider = FauxProvider(provider_id="faux")
    provider.append_responses([faux_assistant_message("ok")])
    agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
    tracer = Tracer(span_adapters=[BrokenAdapter()], record_content=True)
    tracer.attach(agent)

    await agent.prompt("hello")
    await agent.wait_for_idle()
    await tracer.shutdown()

    assert agent.state.messages[-1].content[0].text == "ok"
