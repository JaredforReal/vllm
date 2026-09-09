# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


def test_custom_tool_call_input_item_is_parsed() -> None:
    from openai.types.responses import ResponseCustomToolCall

    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    request = ResponsesRequest(
        input=[
            {"role": "user", "content": "run pwd"},
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "emit_command",
                "input": "pwd",
            },
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "/"},
        ],
        tools=[{"type": "custom", "name": "emit_command", "format": {"type": "text"}}],
        tool_choice={"type": "custom", "name": "emit_command"},
    )
    assert isinstance(request.input[1], ResponseCustomToolCall)
    assert request.tool_choice.type == "custom"
