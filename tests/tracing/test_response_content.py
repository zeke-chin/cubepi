from __future__ import annotations

from cubepi.tracing.content import response_to_semconv_messages


def test_anthropic_response_to_semconv_messages():
    messages = response_to_semconv_messages(
        {
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "check"},
                {"type": "text", "text": "Working"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        }
    )
    assert messages[0]["parts"] == [
        {"type": "reasoning", "content": "check"},
        {"type": "text", "content": "Working"},
        {
            "type": "tool_call",
            "id": "tool-1",
            "name": "weather",
            "arguments": {"city": "Tokyo"},
        },
    ]


def test_openai_chat_response_to_semconv_messages():
    messages = response_to_semconv_messages(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    assert messages[0]["parts"][0]["arguments"] == {"city": "Tokyo"}


def test_openai_responses_output_to_semconv_messages():
    messages = response_to_semconv_messages(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Sunny"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "weather",
                    "arguments": '{"city":"Tokyo"}',
                },
            ]
        }
    )
    assert messages[0]["parts"] == [{"type": "text", "content": "Sunny"}]
    assert messages[1]["parts"][0]["arguments"] == {"city": "Tokyo"}
