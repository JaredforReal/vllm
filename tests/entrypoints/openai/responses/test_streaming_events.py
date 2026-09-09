# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
    split_delta,
)


def _make_tool_call(
    index: int, name: str | None = None, arguments: str | None = None
) -> DeltaToolCall:
    fn = DeltaFunctionCall(name=name, arguments=arguments)
    return DeltaToolCall(index=index, function=fn)


class TestSplitDelta:
    def test_all_three_fields(self):
        tc = _make_tool_call(0, name="f")
        delta = DeltaMessage(reasoning="r", content="c", tool_calls=[tc])
        result = split_delta(delta)

        assert len(result) == 3
        assert result[0].reasoning == "r" and result[0].content is None
        assert result[1].content == "c" and result[1].reasoning is None
        assert len(result[2].tool_calls) == 1 and result[2].content is None

    def test_tool_calls_grouped_by_index(self):
        tc0 = _make_tool_call(0, name="f1")
        tc1 = _make_tool_call(1, name="f2")
        tc0b = _make_tool_call(0, arguments='{"a":1}')

        # Different indices → split
        result = split_delta(DeltaMessage(tool_calls=[tc0, tc1]))
        assert len(result) == 2
        assert result[0].tool_calls == [tc0]
        assert result[1].tool_calls == [tc1]

        # Same index → stays together
        delta = DeltaMessage(tool_calls=[tc0, tc0b])
        result = split_delta(delta)
        assert len(result) == 1
        assert result[0] is delta


def _run_through_processor(
    processor: SimpleStreamingEventProcessor,
    delta_message: DeltaMessage,
) -> list:
    """Simulate the streaming loop from serving.py for a single delta."""
    events = []
    for dm in split_delta(delta_message):
        target_state, tool_call = processor.resolve_target_state(dm)
        if target_state == _StateType.NONE:
            continue
        if processor.needs_transition(target_state, tool_call):
            events.extend(processor.close_current())
            events.extend(processor.open(target_state, tool_call))
        events.extend(processor.emit_delta(dm, None))
    return events


class TestProcessorCompoundDeltas:
    def test_all_three_states(self):
        tc = _make_tool_call(0, name="f", arguments="{}")
        delta = DeltaMessage(reasoning="r", content="c", tool_calls=[tc])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        types = [e.type for e in events]
        r_idx = types.index("response.reasoning_text.delta")
        c_idx = types.index("response.output_text.delta")
        fc_idx = types.index("response.function_call_arguments.delta")
        assert r_idx < c_idx < fc_idx

    def test_parallel_tool_calls(self):
        tc0 = _make_tool_call(0, name="f1", arguments='{"a":1}')
        tc1 = _make_tool_call(1, name="f2", arguments='{"b":2}')
        delta = DeltaMessage(tool_calls=[tc0, tc1])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        added = [e for e in events if e.type == "response.output_item.added"]
        deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(added) == 2
        assert len(deltas) == 2

    def test_split_name_and_args_same_index(self):
        """Regression: parsers like KimiK2 emit name and args as separate
        DeltaToolCalls at the same index within one DeltaMessage."""
        tc_name = _make_tool_call(0, name="get_weather")
        tc_args = _make_tool_call(0, arguments='{"city":"SF"}')
        delta = DeltaMessage(tool_calls=[tc_name, tc_args])

        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(processor, delta)

        deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(deltas) == 1
        assert deltas[0].delta == '{"city":"SF"}'

    def test_reasoning_to_content_transition(self):
        """Regression: the old special case in emit_delta handled this;
        now split_delta handles it generically."""
        processor = SimpleStreamingEventProcessor()
        _run_through_processor(processor, DeltaMessage(reasoning="think"))
        assert processor.state.current_state == _StateType.REASONING

        events = _run_through_processor(
            processor, DeltaMessage(reasoning="more", content="answer")
        )
        types = [e.type for e in events]
        assert "response.reasoning_text.delta" in types
        assert "response.output_text.delta" in types


def _done_items(events: list) -> list:
    return [e.item for e in events if e.type == "response.output_item.done"]


class TestProcessorFinalItems:
    """The final response reuses the streamed items, so every
    `output_item.done` item must be recorded verbatim in `state.output_items`."""

    def test_output_items_match_done_events(self):
        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(
            processor,
            DeltaMessage(
                reasoning="think",
                content="answer",
                tool_calls=[_make_tool_call(0, name="f", arguments="{}")],
            ),
        )
        events.extend(processor.close_current())

        done_items = _done_items(events)
        assert done_items == processor.state.output_items
        assert [item.type for item in done_items] == [
            "reasoning",
            "message",
            "function_call",
        ]
        assert done_items[0].id.startswith("rs_")
        assert done_items[1].id.startswith("msg_")
        assert done_items[2].id.startswith("fc_")

    def test_arguments_done_always_emitted(self):
        """OpenAI emits function_call_arguments.done even for empty arguments."""
        processor = SimpleStreamingEventProcessor()
        events = _run_through_processor(
            processor, DeltaMessage(tool_calls=[_make_tool_call(0, name="f")])
        )
        events.extend(processor.close_current())

        done = [e for e in events if e.type == "response.function_call_arguments.done"]
        assert len(done) == 1
        assert done[0].arguments == ""
        assert done[0].item_id == processor.state.output_items[0].id

    def test_encrypted_reasoning_round_trips(self):
        from vllm.entrypoints.openai.responses.encrypted_content import (
            decode_encrypted_content,
        )

        processor = SimpleStreamingEventProcessor(encrypt_reasoning=True)
        _run_through_processor(processor, DeltaMessage(reasoning="secret plan"))
        processor.close_current()

        (item,) = processor.state.output_items
        payload = decode_encrypted_content(
            item.encrypted_content, expected_type="reasoning"
        )
        assert payload["text"] == "secret plan"


class TestProcessorCustomTools:
    def _custom_tool(self):
        from openai.types.responses import CustomTool

        return CustomTool(type="custom", name="emit_command", format={"type": "text"})

    def test_custom_tool_input_streams_decoded_text(self):
        processor = SimpleStreamingEventProcessor(tools=[self._custom_tool()])
        events = []
        for chunk in ['{"inp', 'ut": "pw', "d \\n", 'ls"}']:
            events.extend(
                _run_through_processor(
                    processor,
                    DeltaMessage(
                        tool_calls=[
                            _make_tool_call(0, name="emit_command", arguments=chunk)
                        ]
                    ),
                )
            )
        events.extend(processor.close_current())

        types = [e.type for e in events]
        assert "response.function_call_arguments.delta" not in types
        deltas = [
            e.delta for e in events if e.type == "response.custom_tool_call_input.delta"
        ]
        assert "".join(deltas) == "pwd \nls"
        (done,) = [
            e for e in events if e.type == "response.custom_tool_call_input.done"
        ]
        assert done.input == "pwd \nls"

        (item,) = _done_items(events)
        assert item.type == "custom_tool_call"
        assert item.id.startswith("ctc_")
        assert item.name == "emit_command"
        assert item.input == "pwd \nls"
        assert item.call_id == done.item_id or item.call_id
        assert processor.state.output_items == [item]

    def test_function_tool_with_same_shape_is_not_custom(self):
        processor = SimpleStreamingEventProcessor(tools=[self._custom_tool()])
        events = _run_through_processor(
            processor,
            DeltaMessage(
                tool_calls=[_make_tool_call(0, name="other", arguments='{"input":"x"}')]
            ),
        )
        events.extend(processor.close_current())
        (item,) = _done_items(events)
        assert item.type == "function_call"
        assert item.arguments == '{"input":"x"}'


class TestHiddenReasoning:
    def test_encrypted_only_reasoning_item(self):
        """include_reasoning=false with reasoning.encrypted_content keeps an
        opaque reasoning item but streams no reasoning text."""
        processor = SimpleStreamingEventProcessor(
            encrypt_reasoning=True, include_reasoning_text=False
        )
        events = _run_through_processor(
            processor, DeltaMessage(reasoning="secret", content="visible")
        )
        events.extend(processor.close_current())

        types = [e.type for e in events]
        assert "response.reasoning_text.delta" not in types
        assert "response.reasoning_part.added" not in types
        reasoning, message = processor.state.output_items
        assert reasoning.type == "reasoning"
        assert reasoning.content is None
        assert reasoning.encrypted_content
        assert message.type == "message"
        assert _done_items(events) == [reasoning, message]
