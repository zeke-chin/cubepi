import asyncio

from cubepi.providers.base import (
    AssistantMessage,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_text, faux_thinking, faux_tool_call


class TestFauxHelpers:
    def test_faux_text(self):
        block = faux_text("hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_faux_thinking(self):
        block = faux_thinking("step by step")
        assert block.type == "thinking"
        assert block.thinking == "step by step"

    def test_faux_tool_call(self):
        block = faux_tool_call("search", {"q": "test"}, id="tc-1")
        assert block.type == "tool_call"
        assert block.id == "tc-1"
        assert block.name == "search"

    def test_faux_tool_call_auto_id(self):
        block = faux_tool_call("search", {"q": "test"})
        assert block.id.startswith("tool:")

    def test_faux_assistant_message_string(self):
        msg = faux_assistant_message("hello")
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "hello"
        assert msg.stop_reason == "stop"

    def test_faux_assistant_message_blocks(self):
        msg = faux_assistant_message([faux_text("hi"), faux_tool_call("search", {"q": "x"}, id="t1")])
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "tool_call"

    def test_faux_assistant_message_tool_use_stop_reason(self):
        msg = faux_assistant_message(
            [faux_tool_call("search", {"q": "x"}, id="t1")],
            stop_reason="tool_use",
        )
        assert msg.stop_reason == "tool_use"


class TestFauxProvider:
    def _make_model(self):
        from cubepi.providers.base import Model
        return Model(id="faux-1", provider="faux")

    async def test_basic_text_response(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello world")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "hello world"
        assert result.stop_reason == "stop"
        assert any(e.type == "done" for e in events)

    async def test_responses_consumed_in_order(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])
        model = self._make_model()

        stream1 = await provider.stream(model, [])
        _ = [e async for e in stream1]
        r1 = await stream1.result()

        stream2 = await provider.stream(model, [])
        _ = [e async for e in stream2]
        r2 = await stream2.result()

        assert r1.content[0].text == "first"
        assert r2.content[0].text == "second"

    async def test_error_when_queue_exhausted(self):
        provider = FauxProvider()
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert "No more faux responses" in (result.error_message or "")

    async def test_set_responses_replaces_queue(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("old")])
        provider.set_responses([faux_assistant_message("new")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "new"

    async def test_append_responses(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("first")])
        provider.append_responses([faux_assistant_message("second")])
        model = self._make_model()

        assert provider.pending_response_count == 2

    async def test_async_response_factory(self):
        async def factory(context, model):
            return faux_assistant_message("dynamic response")

        provider = FauxProvider()
        provider.set_responses([factory])
        model = self._make_model()

        stream = await provider.stream(model, [])
        _ = [e async for e in stream]
        result = await stream.result()

        assert result.content[0].text == "dynamic response"

    async def test_streams_text_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message("AB")])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "start" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

    async def test_streams_thinking_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([faux_assistant_message([faux_thinking("think"), faux_text("ok")])])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "thinking_start" in event_types
        assert "thinking_delta" in event_types
        assert "thinking_end" in event_types

    async def test_streams_tool_call_deltas(self):
        provider = FauxProvider(token_size_min=1, token_size_max=1)
        provider.set_responses([
            faux_assistant_message(
                [faux_tool_call("search", {"q": "test"}, id="t1")],
                stop_reason="tool_use",
            ),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        event_types = [e.type for e in events]

        assert "toolcall_start" in event_types
        assert "toolcall_delta" in event_types
        assert "toolcall_end" in event_types

    async def test_abort_before_start(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("hello")])
        model = self._make_model()
        signal = asyncio.Event()
        signal.set()

        stream = await provider.stream(model, [], signal=signal)
        events = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "aborted"

    async def test_abort_mid_stream(self):
        provider = FauxProvider(tokens_per_second=20, token_size_min=2, token_size_max=2)
        provider.set_responses([
            faux_assistant_message("one two three four five six seven eight nine ten"),
        ])
        model = self._make_model()
        signal = asyncio.Event()

        stream = await provider.stream(model, [], signal=signal)

        events = []
        count = 0
        async for event in stream:
            events.append(event)
            count += 1
            if count >= 3:
                signal.set()

        result = await stream.result()
        assert result.stop_reason == "aborted"

    async def test_error_message_passthrough(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message("", stop_reason="error", error_message="API rate limit"),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        assert result.stop_reason == "error"
        assert any(e.type == "error" for e in events)

    async def test_multiple_tool_calls_in_one_message(self):
        provider = FauxProvider()
        provider.set_responses([
            faux_assistant_message(
                [
                    faux_tool_call("search", {"q": "a"}, id="t1"),
                    faux_tool_call("search", {"q": "b"}, id="t2"),
                ],
                stop_reason="tool_use",
            ),
        ])
        model = self._make_model()

        stream = await provider.stream(model, [])
        events = [e async for e in stream]
        result = await stream.result()

        toolcall_starts = [e for e in events if e.type == "toolcall_start"]
        assert len(toolcall_starts) == 2

    async def test_call_count_tracking(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("a"), faux_assistant_message("b")])
        model = self._make_model()

        assert provider.call_count == 0

        s1 = await provider.stream(model, [])
        _ = [e async for e in s1]
        assert provider.call_count == 1

        s2 = await provider.stream(model, [])
        _ = [e async for e in s2]
        assert provider.call_count == 2
