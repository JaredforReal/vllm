# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterable
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_custom_tool_call_output_item import (
    ResponseCustomToolCallOutputItem,
)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool

from vllm import envs
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import FunctionCall, FunctionDefinition
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tool_parsers.utils import (
    CUSTOM_TOOL_INPUT_PARAM,
    build_responses_tool_call_name_map,
    custom_tool_input_from_arguments,
    custom_tool_names,
    flat_namespace_tool_name,
    iter_response_function_tool_dicts,
    resolve_responses_tool_call_name,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


def make_reasoning_item(
    text: str,
    *,
    item_id: str | None = None,
    status: str | None = None,
) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=item_id or f"rs_{random_uuid()}",
        summary=[],
        type="reasoning",
        content=[ResponseReasoningTextContent(text=text, type="reasoning_text")],
        status=status,  # type: ignore[arg-type]
    )


def make_output_message(
    text: str,
    *,
    item_id: str | None = None,
    logprobs: list[Logprob] | None = None,
) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id or f"msg_{random_uuid()}",
        content=[
            ResponseOutputText(
                text=text,
                annotations=[],
                type="output_text",
                logprobs=logprobs,
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )


def make_function_call_item(
    name: str,
    arguments: str,
    *,
    call_id: str | None = None,
    item_id: str | None = None,
    namespace: str | None = None,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=item_id or f"fc_{random_uuid()}",
        call_id=call_id or make_tool_call_id(),
        type="function_call",
        status="completed",
        name=name,
        namespace=namespace,
        arguments=arguments,
    )


def make_custom_tool_call_item(
    name: str,
    input: str,
    *,
    call_id: str | None = None,
    item_id: str | None = None,
    namespace: str | None = None,
) -> ResponseCustomToolCall:
    return ResponseCustomToolCall(
        id=item_id or f"ctc_{random_uuid()}",
        call_id=call_id or make_tool_call_id(),
        type="custom_tool_call",
        name=name,
        namespace=namespace,
        input=input,
    )


def build_response_output_items(
    reasoning: str | None,
    content: str | None,
    tool_calls: list[FunctionCall] | None,
    logprobs: list[Logprob] | None = None,
    tools: list[Tool] | None = None,
) -> list[ResponseOutputItem]:
    outputs: list[ResponseOutputItem] = []
    tool_call_name_map = build_responses_tool_call_name_map(tools)
    custom_tools = custom_tool_names(tools)

    if reasoning:
        outputs.append(make_reasoning_item(reasoning))

    if content:
        outputs.append(make_output_message(content, logprobs=logprobs))

    if tool_calls:
        for idx, tool_call in enumerate(tool_calls):
            call_id = tool_call.id or make_tool_call_id(
                func_name=tool_call.name, idx=idx
            )
            if tool_call.name in custom_tools:
                outputs.append(
                    make_custom_tool_call_item(
                        tool_call.name,
                        custom_tool_input_from_arguments(tool_call.arguments) or "",
                        call_id=call_id,
                    )
                )
                continue
            call_name = resolve_responses_tool_call_name(
                tool_call.name, tool_call_name_map=tool_call_name_map
            )
            outputs.append(
                make_function_call_item(
                    call_name.name,
                    tool_call.arguments,
                    call_id=call_id,
                    namespace=call_name.namespace,
                )
            )

    return outputs


def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool:
    """
    Determine if the last input message is a partial assistant message
    that should be continued rather than starting a new generation.

    This enables partial message completion similar to Anthropic's Messages API,
    where users can provide an incomplete assistant message and have the model
    continue from where it left off.

    A message is considered partial if:
    1. It's a ResponseOutputMessage or ResponseReasoningItem
    2. Its status is "in_progress" or "incomplete"

    Args:
        request_input: The input to the Responses API request

    Returns:
        True if the final message should be continued, False otherwise
    """
    if isinstance(request_input, str):
        # Simple string input is always a user message
        return False

    if not request_input:
        return False

    last_item = request_input[-1]

    # Check if the last item is a partial assistant message
    if isinstance(last_item, ResponseOutputMessage):
        return last_item.status in ("in_progress", "incomplete")

    # Check if the last item is a partial reasoning item
    if isinstance(last_item, ResponseReasoningItem):
        return last_item.status in ("in_progress", "incomplete")

    if isinstance(last_item, dict):
        # only support partial completion for messages for now
        if last_item.get("type", "message") not in ("message", "reasoning"):
            return False
        return last_item.get("status") in ("in_progress", "incomplete")

    return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
):
    messages: list[ChatCompletionMessageParam] = []
    if request_instructions:
        messages.append(
            {
                "role": "system",
                "content": request_instructions,
            }
        )

    # Prepend the conversation history.
    if prev_msg is not None:
        # Filter out system messages from previous conversation -- per the
        # OpenAI spec, instructions should NOT carry over across responses.
        # The current request's instructions (if any) were already added above.
        messages.extend(m for m in prev_msg if m.get("role") != "system")
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
    else:
        input_messages = construct_chat_messages_with_tool_call(request_input)
        messages.extend(input_messages)
    return messages


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """Build chat messages from response items.

    Some chat messages span multiple response items (e.g., reasoning + tool calls).
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        message = _construct_message_from_response_item(
            item, prev_msg=messages[-1] if messages else None
        )
        if message is not None:
            messages.append(message)

    return messages


def _construct_message_from_response_item(
    item: ResponseInputOutputItem,
    prev_msg: ChatCompletionMessageParam | None = None,
) -> ChatCompletionMessageParam | None:
    """
    Returns a new message or None. If `None`, `prev_msg` might be updated.
    If `prev_msg` is `None`, a new message is always returned.
    """
    prev_assistant_msg = (
        prev_msg if prev_msg and prev_msg.get("role") == "assistant" else None
    )

    if isinstance(item, (ResponseFunctionToolCall, ResponseCustomToolCall)):
        tool_name = item.name
        if item.namespace:
            tool_name = flat_namespace_tool_name(item.namespace, item.name)
        if isinstance(item, ResponseCustomToolCall):
            arguments = json.dumps(
                {CUSTOM_TOOL_INPUT_PARAM: item.input}, ensure_ascii=False
            )
        else:
            arguments = item.arguments
        tool_call = ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=tool_name,
                arguments=arguments,
            ),
            type="function",
        )
        if prev_assistant_msg:
            tool_calls = prev_assistant_msg.get("tool_calls")
            if tool_calls is None:
                prev_assistant_msg["tool_calls"] = [tool_call]
                return None
            if isinstance(tool_calls, list):
                tool_calls.append(tool_call)
                return None
            if isinstance(tool_calls, Iterable) and not isinstance(
                tool_calls, (dict, str)
            ):
                tool_calls = list(tool_calls)
                tool_calls.append(tool_call)
                prev_assistant_msg["tool_calls"] = tool_calls
                return None
            logger.warning(
                "Previous assistant message has unknown tool_calls format. "
                "Tool call merging is skipped and a new assistant message is created. "
                "Item %s",
                item.id,
            )
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[tool_call],
        )
    elif isinstance(item, ResponseReasoningItem):
        reasoning = ""
        if item.encrypted_content:
            raise VLLMValidationError(
                "Encrypted content is not supported.",
                parameter="input",
            )
        elif item.content and len(item.content) >= 1:
            reasoning = item.content[0].text
        elif len(item.summary) >= 1:
            reasoning = item.summary[0].text
            logger.warning(
                "Using summary text as reasoning content for item %s. "
                "Please use content instead of summary for "
                "reasoning items.",
                item.id,
            )

        if prev_assistant_msg:
            previous_reasoning = prev_assistant_msg.get("reasoning")
            if previous_reasoning is None:
                prev_assistant_msg["reasoning"] = reasoning
                return None
        return {
            "role": "assistant",
            "reasoning": reasoning,
        }
    elif isinstance(item, ResponseOutputMessage):
        output_text = item.content[0].text
        if prev_assistant_msg:
            previous_content = prev_assistant_msg.get("content")
            if previous_content is None:
                prev_assistant_msg["content"] = output_text
                return None
        return {
            "role": "assistant",
            "content": output_text,
        }
    elif isinstance(
        item, (ResponseFunctionToolCallOutputItem, ResponseCustomToolCallOutputItem)
    ):
        return ChatCompletionToolMessageParam(
            role="tool",
            content=_tool_output_text(item.output),
            tool_call_id=item.call_id,
        )
    elif isinstance(item, dict) and item.get("type") in (
        "function_call_output",
        "custom_tool_call_output",
    ):
        # Append the tool call output as a tool message.
        return ChatCompletionToolMessageParam(
            role="tool",
            content=_tool_output_text(item.get("output")),
            tool_call_id=item.get("call_id"),
        )
    elif isinstance(item, dict) and item.get("role") == "assistant":
        content = item.get("content")
        text: str | None = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and content:
            text = content[0].get("text")
        if text is not None:
            if prev_assistant_msg:
                previous_content = prev_assistant_msg.get("content")
                if previous_content is None:
                    prev_assistant_msg["content"] = text
                    return None
            return {"role": "assistant", "content": text}
    if isinstance(item, dict) and "role" in item:
        return item  # type: ignore[return-value]
    item_type = item.get("type") if isinstance(item, dict) else item.type
    raise VLLMValidationError(
        f"Unsupported input item type: {item_type}",
        parameter="input",
    )


def _tool_output_text(output: Any) -> Any:
    """Flatten a list-of-content tool output into plain text."""
    if not isinstance(output, list):
        return output
    texts = []
    for part in output:
        text = (
            part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        )
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def extract_function_tool_names(tools: list[Tool]) -> frozenset[str]:
    names = []
    for tool in tools:
        if tool.type == "function":
            names.append(tool.name)
        elif tool.type == "namespace":
            names.extend(
                flat_namespace_tool_name(tool.name, namespaced_tool.name)
                for namespaced_tool in tool.tools
                if namespaced_tool.type == "function"
            )
    return frozenset(names)


def extract_tool_types(tools: list[Tool]) -> set[str]:
    """
    Extracts the tool types from the given tools.
    """
    tool_types: set[str] = set()
    for tool in tools:
        if tool.type == "mcp":
            # Allow the MCP Tool type to enable built in tools if the
            # server_label is allowlisted in
            # envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS
            if tool.server_label in envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
                tool_types.add(tool.server_label)
        else:
            tool_types.add(tool.type)
    return tool_types


def convert_tool_responses_to_completions_format(
    tool: dict,
) -> ChatCompletionToolsParam:
    """
    Convert a flat Responses tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into a Chat Completions tool param for chat-template rendering.
    """
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition.model_validate(
            {k: v for k, v in tool.items() if k != "type"}
        ),
    )


def construct_tool_dicts(
    tools: list[Tool],
    tool_choice: ToolChoice,
    exclude_tools_when_tool_choice_none: bool = False,
) -> list[dict[str, Any]] | None:
    if not tools or (tool_choice == "none" and exclude_tools_when_tool_choice_none):
        return None
    return [
        convert_tool_responses_to_completions_format(tool).model_dump()
        for tool in iter_response_function_tool_dicts(tools)
    ]
