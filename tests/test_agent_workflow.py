"""Test bounded agent execution with mocked model responses and MCP clients."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agent_workflow import MAX_TOOL_ROUNDS, StrategyDecision, graph, main, validate_input

TOOL = Tool(
    name="find_contracts_by_party",
    description="Find contracts by party.",
    input_schema={
        "type": "object", "properties": {"party": {"type": "string"}},
        "required": ["party"],
    },
    output_schema={
        "type": "object", "properties": {"result": {"type": "array"}},
        "required": ["result"],
    },
)
PAYLOAD = {"result": [{"title": "Synthetic agreement"}]}


def tool_request(call_id: str = "call-1", **overrides: object) -> AIMessage:
    call = {
        "name": TOOL.name, "args": {"party": "Harbor Labs"},
        "id": call_id, "type": "tool_call",
    }
    call.update(overrides)
    return AIMessage(content="", tool_calls=[call])


class AgentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict("os.environ", {
            "LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3:8b",
        }, clear=True))
        self.initialize = self.enterContext(patch("agent_workflow.init_chat_model"))
        self.strategy_model = self.initialize.return_value.with_structured_output.return_value
        self.strategy_model.ainvoke = AsyncMock(return_value=StrategyDecision(
            strategy="direct", reason="A straightforward request.",
        ))
        self.model = self.initialize.return_value.bind_tools.return_value
        self.model.ainvoke = AsyncMock(return_value=AIMessage(content="Final answer"))
        self.client_factory = self.enterContext(patch("agent_workflow.Client"))
        self.client = self.client_factory.return_value.__aenter__.return_value
        self.client.list_tools = AsyncMock(return_value=ListToolsResult(tools=[TOOL]))
        self.client.call_tool = AsyncMock(return_value=CallToolResult(
            content=[], structured_content=PAYLOAD,
        ))

    async def test_direct_strategy_trace_and_single_selection(self) -> None:
        self.model.ainvoke.side_effect = [tool_request(), AIMessage(content="Done")]
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, "Done\n")
        self.assertIn("Strategy: direct", trace)
        self.assertIn("Reason: A straightforward request.", trace)
        self.assertIn("Branch: call_llm (enter MCP agent loop)", trace)
        self.assertNotIn("Node: prepare_investigation", trace)
        self.assertNotIn("Node: prepare_clarification", trace)
        self.strategy_model.ainvoke.assert_awaited_once()

    async def test_investigate_preserves_conversation_and_dynamic_tool_loop(self) -> None:
        self.strategy_model.ainvoke.return_value = StrategyDecision(
            strategy="investigate", reason="Compare contract evidence.",
        )
        self.model.ainvoke.side_effect = [tool_request(), AIMessage(content="Comparison")]
        updates = [update async for update in graph.astream(
            {"prompt": "Compare contracts"}, stream_mode="updates"
        )]
        self.assertEqual([next(iter(update)) for update in updates], [
            "validate_input", "choose_strategy", "prepare_investigation", "call_llm",
            "execute_mcp_tools", "call_llm",
        ])
        original = updates[0]["validate_input"]["messages"]
        prepared = updates[2]["prepare_investigation"]["messages"]
        self.assertEqual(prepared[:-1], original)
        self.assertIsInstance(prepared[-1], SystemMessage)
        self.assertIn("Gather sufficient contract evidence", prepared[-1].text)
        self.assertEqual(self.model.ainvoke.call_args_list[0].args[0], prepared)
        self.client.call_tool.assert_awaited_once_with(TOOL.name, {"party": "Harbor Labs"})
        self.assertEqual(updates[-1]["call_llm"]["response"], "Comparison")
        self.strategy_model.ainvoke.assert_awaited_once()

    async def test_investigate_trace(self) -> None:
        self.strategy_model.ainvoke.return_value = StrategyDecision(
            strategy="investigate", reason="Detailed analysis needs evidence.",
        )
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, "Final answer\n")
        self.assertIn("Strategy: investigate", trace)
        self.assertIn("Node: prepare_investigation", trace)
        self.assertIn("prepare_investigation -> call_llm (enter MCP agent loop)", trace)
        self.assertEqual(trace.count("HumanMessage:"), 1)
        self.assertEqual(trace.count("SystemMessage:"), 2)

    async def test_clarify_trace_finishes_without_mcp_or_agent_invocation(self) -> None:
        question = "Which contract should I examine?"
        self.strategy_model.ainvoke.return_value = StrategyDecision(
            strategy="clarify", reason="The contract is unspecified.",
            clarification_question=question,
        )
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, question + "\n")
        self.assertIn("Strategy: clarify", trace)
        self.assertIn("prepare_clarification -> END (finish with clarification)", trace)
        self.assertNotIn("Node: call_llm", trace)
        self.client_factory.assert_not_called()
        self.model.ainvoke.assert_not_awaited()
        self.initialize.return_value.bind_tools.assert_not_called()
        self.strategy_model.ainvoke.assert_awaited_once()

    async def test_invalid_strategy_decisions_follow_error_path(self) -> None:
        for decision in (
            {"strategy": "execute_mcp_tools", "reason": "Bypass routing"},
            {"strategy": "clarify", "reason": "Ambiguous"},
            {"strategy": "clarify", "reason": "Ambiguous", "clarification_question": "  "},
            {"strategy": "direct", "reason": ""},
            {"strategy": "direct", "reason": "x" * 241},
            {"strategy": "direct", "reason": "Simple", "node": "END"},
            "not JSON",
        ):
            with self.subTest(decision=decision):
                self.strategy_model.ainvoke.return_value = decision
                updates = [update async for update in graph.astream(
                    {"prompt": "Find contracts"}, stream_mode="updates"
                )]
                self.assertEqual([next(iter(update)) for update in updates], [
                    "validate_input", "choose_strategy", "handle_error",
                ])
                self.assertIn("Strategy selection failed", updates[1]["choose_strategy"]["error"])
                self.client_factory.assert_not_called()
                self.model.ainvoke.assert_not_awaited()

    async def test_strategy_parsing_and_invocation_failures_are_fatal(self) -> None:
        for failure in (ValueError("Invalid structured JSON"), RuntimeError("Model unavailable")):
            with self.subTest(failure=failure):
                self.strategy_model.ainvoke.reset_mock()
                self.strategy_model.ainvoke.side_effect = failure
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertIn(str(failure), result["error"])
                self.assertNotIn("response", result)
                self.assertNotIn("strategy_decision", result)
                self.assertEqual(result["tool_rounds"], 0)
                self.client_factory.assert_not_called()
                self.strategy_model.ainvoke.assert_awaited_once()

    async def test_strategy_request_is_separate_from_agent_conversation(self) -> None:
        result = await graph.ainvoke({"prompt": "  Find contracts  "})
        request = self.strategy_model.ainvoke.call_args.args[0]
        self.assertEqual(len(request), 2)
        self.assertIsInstance(request[0], SystemMessage)
        self.assertEqual(request[1].text, "Find contracts")
        self.assertEqual(result["prompt"], "Find contracts")
        self.assertEqual(result["messages"][1].text, "Find contracts")
        self.assertNotIn("Choose an approach", result["messages"][0].text)

    async def test_direct_answer_and_discovered_schema_binding(self) -> None:
        answer = AIMessage(content="Hello")
        self.model.ainvoke.return_value = answer
        result = await graph.ainvoke({"prompt": "  Hello  "})
        self.assertEqual(result["response"], "Hello")
        self.assertEqual(result["prompt"], "Hello")
        self.assertIs(result["messages"][-1], answer)
        self.assertEqual(result["tool_rounds"], 0)
        self.assertIsInstance(result["messages"][1], HumanMessage)
        self.client.call_tool.assert_not_awaited()
        self.assertEqual(self.initialize.call_count, 2)
        self.initialize.assert_called_with(model="qwen3:8b", model_provider="ollama")
        self.initialize.return_value.with_structured_output.assert_called_once_with(StrategyDecision)
        self.assertEqual(result["strategy_decision"].strategy, "direct")
        self.initialize.return_value.bind_tools.assert_called_once_with([
            {"type": "function", "function": {
                "name": TOOL.name, "description": TOOL.description,
                "parameters": TOOL.input_schema,
            }},
        ])

    async def test_one_tool_then_final_answer(self) -> None:
        self.model.ainvoke.side_effect = [tool_request(), AIMessage(content="Summary")]
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertEqual(result["response"], "Summary")
        self.assertEqual(result["tool_rounds"], 1)
        self.client.call_tool.assert_awaited_once_with(TOOL.name, {"party": "Harbor Labs"})
        tool_message = result["messages"][-2]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertEqual(tool_message.tool_call_id, "call-1")
        self.assertEqual(json.loads(tool_message.content), PAYLOAD)
        self.assertIn(tool_message, self.model.ainvoke.call_args.args[0])
        self.client.list_tools.assert_awaited_once()

    async def test_sequential_tool_calls_and_graph_paths(self) -> None:
        self.model.ainvoke.side_effect = [
            tool_request("first"), tool_request("second"), AIMessage(content="Done"),
        ]
        updates = [update async for update in graph.astream(
            {"prompt": "Find contracts"}, stream_mode="updates"
        )]
        self.assertEqual([next(iter(update)) for update in updates], [
            "validate_input", "choose_strategy", "call_llm", "execute_mcp_tools", "call_llm",
            "execute_mcp_tools", "call_llm",
        ])
        self.assertEqual(self.client.call_tool.await_count, 2)
        self.assertEqual(updates[-2]["execute_mcp_tools"]["tool_rounds"], 2)

    async def test_multiple_calls_in_one_round(self) -> None:
        batch = AIMessage(content="", tool_calls=[
            *tool_request("first").tool_calls, *tool_request("second").tool_calls,
        ])
        self.model.ainvoke.side_effect = [batch, AIMessage(content="Done")]
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(self.client.call_tool.await_count, 2)
        self.assertEqual([
            message.tool_call_id for message in result["messages"]
            if isinstance(message, ToolMessage)
        ], ["first", "second"])

    async def test_fatal_batch_is_rejected_before_any_execution(self) -> None:
        bad_calls = [
            tool_request(name="arbitrary_python_function"),
            tool_request(call_id=""), tool_request(call_id=None),
            AIMessage(content="", tool_calls=[
                *tool_request().tool_calls, *tool_request().tool_calls,
            ]),
        ]
        for bad in bad_calls:
            with self.subTest(call=bad.tool_calls):
                self.model.ainvoke.return_value = bad
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertTrue(result["error"])
                self.assertNotIn("response", result)
                self.assertEqual(result["tool_rounds"], 0)
                self.client.call_tool.assert_not_awaited()
        self.model.ainvoke.return_value = AIMessage(content="", tool_calls=[
            *tool_request().tool_calls, *tool_request(name="unknown").tool_calls,
        ])
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertIn("Unknown MCP tool", result["error"])
        self.client.call_tool.assert_not_awaited()

    async def test_tool_failures_and_malformed_results(self) -> None:
        for response in (
            CallToolResult(content=[], structured_content={"wrong": []}),
            CallToolResult(content=[]), {"content": "malformed"},
        ):
            with self.subTest(response=response):
                self.model.ainvoke.return_value = tool_request()
                self.client.call_tool.return_value = response
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertIn("MCP tool execution failed", result["error"])
                self.assertNotIn("response", result)

    async def test_recovery_after_invalid_arguments(self) -> None:
        for arguments in ({}, {"party": 123}):
            with self.subTest(arguments=arguments):
                self.client.call_tool.reset_mock()
                self.model.ainvoke.side_effect = [
                    tool_request("bad", args=arguments), tool_request("fixed"),
                    AIMessage(content="Recovered answer"),
                ]
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertFalse(result.get("error"))
                self.assertEqual(result["response"], "Recovered answer")
                self.assertEqual(result["tool_rounds"], 2)
                self.client.call_tool.assert_awaited_once_with(
                    TOOL.name, {"party": "Harbor Labs"}
                )
                error_message = result["messages"][3]
                self.assertIsInstance(error_message, ToolMessage)
                self.assertEqual(error_message.status, "error")
                self.assertEqual(error_message.tool_call_id, "bad")
                self.assertEqual(error_message.name, TOOL.name)
                self.assertIn("Invalid arguments", error_message.text)
                self.assertIn(error_message, self.model.ainvoke.call_args_list[-2].args[0])

    async def test_recovery_after_mcp_tool_error_and_trace(self) -> None:
        self.model.ainvoke.side_effect = [
            tool_request("missing"), tool_request("retry"), AIMessage(content="Recovered"),
        ]
        self.client.call_tool.side_effect = [
            CallToolResult(is_error=True, content=[TextContent(type="text", text="Not found")]),
            CallToolResult(content=[], structured_content=PAYLOAD),
        ]
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, "Recovered\n")
        self.assertIn("status=error recoverable_error=Tool failed: Not found", trace)
        self.assertIn("status=success result=1 item", trace)
        self.assertIn("Tool rounds: 2/5", trace)
        self.assertNotIn("Node: handle_error", trace)
        error_message = self.model.ainvoke.call_args_list[1].args[0][-1]
        self.assertEqual(error_message.tool_call_id, "missing")
        self.assertEqual(error_message.status, "error")

    async def test_batch_with_recoverable_failure_resolves_every_call(self) -> None:
        for invalid_arguments in (True, False):
            with self.subTest(invalid_arguments=invalid_arguments):
                self.client.call_tool.reset_mock()
                batch = AIMessage(content="", tool_calls=[
                    *tool_request("first").tool_calls,
                    *tool_request("failed", args={} if invalid_arguments else {"party": "Missing"}).tool_calls,
                    *tool_request("last").tool_calls,
                ])
                success = CallToolResult(content=[], structured_content=PAYLOAD)
                failure = CallToolResult(is_error=True, content=[])
                self.client.call_tool.side_effect = (
                    [success, success] if invalid_arguments else [success, failure, success]
                )
                self.model.ainvoke.side_effect = [batch, AIMessage(content="Done")]
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertEqual(result["response"], "Done")
                self.assertEqual(result["tool_rounds"], 1)
                messages = self.model.ainvoke.call_args.args[0][-3:]
                self.assertEqual([m.tool_call_id for m in messages], ["first", "failed", "last"])
                self.assertEqual([m.status for m in messages], ["success", "error", "success"])
                self.assertEqual(self.client.call_tool.await_count, 2 if invalid_arguments else 3)

    async def test_repeated_recoverable_errors_are_bounded(self) -> None:
        for invalid_arguments in (True, False):
            with self.subTest(invalid_arguments=invalid_arguments):
                self.model.ainvoke.reset_mock()
                self.client.call_tool.reset_mock()
                self.model.ainvoke.return_value = tool_request(
                    args={} if invalid_arguments else {"party": "Missing"}
                )
                self.client.call_tool.return_value = CallToolResult(is_error=True, content=[])
                result = await graph.ainvoke({"prompt": "Keep retrying"})
                self.assertIn("Maximum of 5", result["error"])
                self.assertEqual(result["tool_rounds"], MAX_TOOL_ROUNDS)
                self.assertEqual(self.model.ainvoke.await_count, MAX_TOOL_ROUNDS + 1)
                self.assertEqual(self.client.call_tool.await_count, 0 if invalid_arguments else MAX_TOOL_ROUNDS)
                errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(len(errors), MAX_TOOL_ROUNDS)
                self.assertTrue(all(m.status == "error" for m in errors))
                self.assertNotIn("response", result)

    async def test_fatal_failure_after_partial_batch_stops_without_model_retry(self) -> None:
        for failure in (
            RuntimeError("Transport unavailable"),
            CallToolResult(content=[], structured_content={"wrong": []}),
        ):
            with self.subTest(failure=failure):
                self.model.ainvoke.reset_mock()
                self.client.call_tool.reset_mock()
                self.model.ainvoke.return_value = AIMessage(content="", tool_calls=[
                    *tool_request("first").tool_calls, *tool_request("second").tool_calls,
                    *tool_request("third").tool_calls,
                ])
                self.client.call_tool.side_effect = [
                    CallToolResult(content=[], structured_content=PAYLOAD), failure,
                ]
                result = await graph.ainvoke({"prompt": "Find contracts"})
                self.assertIn("MCP tool execution failed", result["error"])
                self.assertNotIn("response", result)
                self.assertEqual(self.model.ainvoke.await_count, 1)
                self.assertEqual(self.client.call_tool.await_count, 2)

    async def test_model_and_transport_failures(self) -> None:
        self.model.ainvoke.side_effect = RuntimeError("Model unavailable")
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertIn("Model unavailable", result["error"])
        self.model.ainvoke.side_effect = None
        self.client.list_tools.side_effect = RuntimeError("Discovery unavailable")
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertIn("Discovery unavailable", result["error"])
        self.client.list_tools.side_effect = None
        self.model.ainvoke.return_value = tool_request()
        self.client.call_tool.side_effect = RuntimeError("Transport unavailable")
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertIn("Transport unavailable", result["error"])

    async def test_configuration_and_binding_failures(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = await graph.ainvoke({"prompt": "Hello"})
        self.assertIn("Set both", result["error"])
        self.client_factory.assert_not_called()
        self.initialize.return_value.bind_tools.side_effect = RuntimeError("No tool support")
        result = await graph.ainvoke({"prompt": "Hello"})
        self.assertIn("No tool support", result["error"])

    async def test_maximum_round_limit_cannot_be_bypassed_by_batches(self) -> None:
        batch = AIMessage(content="", tool_calls=[
            *tool_request("first").tool_calls, *tool_request("second").tool_calls,
        ])
        self.model.ainvoke.return_value = batch
        result = await graph.ainvoke({"prompt": "Keep searching"})
        self.assertIn("Maximum of 5", result["error"])
        self.assertEqual(result["tool_rounds"], MAX_TOOL_ROUNDS)
        self.assertEqual(self.client.call_tool.await_count, 2 * MAX_TOOL_ROUNDS)
        self.assertEqual(self.model.ainvoke.await_count, MAX_TOOL_ROUNDS + 1)
        self.assertNotIn("response", result)

    async def test_final_answer_after_fifth_round_is_allowed(self) -> None:
        self.model.ainvoke.side_effect = [
            *(tool_request(f"round-{i}") for i in range(MAX_TOOL_ROUNDS)),
            AIMessage(content="Done"),
        ]
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertFalse(result.get("error"))
        self.assertEqual(result["tool_rounds"], MAX_TOOL_ROUNDS)
        self.assertEqual(result["response"], "Done")

    async def test_invalid_input_routes_to_error_without_external_calls(self) -> None:
        for request in ({}, {"prompt": None}, {"prompt": 1}, {"prompt": " \n "}):
            with self.subTest(request=request):
                updates = [update async for update in graph.astream(
                    request, stream_mode="updates"
                )]
                self.assertEqual([next(iter(update)) for update in updates], [
                    "validate_input", "handle_error",
                ])
                self.initialize.assert_not_called()
                self.client_factory.assert_not_called()
        state = {"prompt": "  Hello  "}
        self.assertEqual(validate_input(state)["prompt"], "Hello")
        self.assertEqual(state, {"prompt": "  Hello  "})

    async def test_malformed_model_tool_calls(self) -> None:
        self.model.ainvoke.return_value = AIMessage(content="", invalid_tool_calls=[{
            "name": TOOL.name, "args": "invalid json", "id": "bad", "error": "Invalid",
        }])
        result = await graph.ainvoke({"prompt": "Find contracts"})
        self.assertIn("malformed tool calls", result["error"])
        self.client.call_tool.assert_not_awaited()

    async def test_cli_success_and_failure(self) -> None:
        for result in ({"response": "Done"}, {"error": "Failed"}):
            output, error = io.StringIO(), io.StringIO()
            with (
                self.subTest(result=result),
                patch("sys.argv", ["agent_workflow.py", "Find contracts"]),
                patch("agent_workflow.graph.ainvoke", new=AsyncMock(return_value=result)),
                patch("agent_workflow.graph.astream") as stream,
                redirect_stdout(output), redirect_stderr(error),
            ):
                if "error" in result:
                    with self.assertRaises(SystemExit) as raised:
                        await main()
                    self.assertEqual(raised.exception.code, 1)
                    self.assertIn("Failed", error.getvalue())
                    self.assertEqual(output.getvalue(), "")
                else:
                    await main()
                    self.assertEqual(output.getvalue(), "Done\n")
                    self.assertEqual(error.getvalue(), "")
                stream.assert_not_called()

    async def run_traced_cli(self) -> tuple[str, str]:
        """Run the real graph with mocked providers and capture both CLI streams."""
        output, error = io.StringIO(), io.StringIO()
        with (
            patch("sys.argv", ["agent_workflow.py", "--trace", "Find contracts"]),
            patch("agent_workflow.graph.ainvoke") as invoke,
            patch("agent_workflow.graph.astream", wraps=graph.astream) as stream,
            redirect_stdout(output), redirect_stderr(error),
        ):
            await main()
        invoke.assert_not_called()
        stream.assert_called_once_with({"prompt": "Find contracts"}, stream_mode="updates")
        return output.getvalue(), error.getvalue()

    async def test_trace_without_tools_preserves_full_final_answer(self) -> None:
        answer = "A long final answer. " * 100
        self.model.ainvoke.return_value = AIMessage(
            content=answer, response_metadata={"private_metadata": "Do not print this"}
        )
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, answer + "\n")
        self.assertIn("[trace] Node: validate_input", trace)
        self.assertIn("[trace] Node: call_llm", trace)
        self.assertIn("SystemMessage:", trace)
        self.assertIn("HumanMessage: Find contracts", trace)
        self.assertIn("AIMessage: A long final answer.", trace)
        self.assertIn("Final answer produced.", trace)
        self.assertNotIn("Node: execute_mcp_tools", trace)
        self.assertNotIn("private_metadata", trace)
        self.assertNotIn("Do not print this", trace)
        self.assertNotIn(answer, trace)

    async def test_trace_with_tool_call_summarizes_results(self) -> None:
        self.model.ainvoke.side_effect = [tool_request(), AIMessage(content="Summary")]
        self.client.call_tool.return_value = CallToolResult(content=[], structured_content={
            "result": [{"id": "C-002", "title": "Synthetic agreement",
                        "text": "Contract text that should not appear. " * 100}],
        })
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, "Summary\n")
        self.assertEqual([
            line for line in trace.splitlines() if line.startswith("[trace] Node:")
        ], [
            "[trace] Node: validate_input", "[trace] Node: choose_strategy", "[trace] Node: call_llm",
            "[trace] Node: execute_mcp_tools", "[trace] Node: call_llm",
        ])
        self.assertIn('Tool call: name=find_contracts_by_party arguments={"party": "Harbor Labs"}', trace)
        self.assertIn("tool_call_id=call-1", trace)
        self.assertIn("ToolMessage: name=find_contracts_by_party", trace)
        self.assertIn("1 item; C-002: Synthetic agreement", trace)
        self.assertIn("Tool rounds: 1/5", trace)
        self.assertNotIn("Contract text that should not appear", trace)
        self.assertNotIn("input_schema", trace)

    async def test_trace_does_not_repeat_history_across_rounds(self) -> None:
        self.model.ainvoke.side_effect = [
            tool_request("first"), tool_request("second"), AIMessage(content="Done"),
        ]
        output, trace = await self.run_traced_cli()
        self.assertEqual(output, "Done\n")
        self.assertEqual(trace.count("SystemMessage:"), 1)
        self.assertEqual(trace.count("HumanMessage:"), 1)
        self.assertEqual(trace.count("AIMessage:"), 3)
        self.assertEqual(trace.count("ToolMessage:"), 2)
        self.assertEqual(trace.count("Tool call:"), 2)
        # Each ID appears once in the request and once in its result.
        self.assertEqual(trace.count("tool_call_id=first"), 2)
        self.assertEqual(trace.count("tool_call_id=second"), 2)
        self.assertEqual(trace.count("Available tools:"), 1)

    async def test_trace_error_preserves_accumulated_error_state(self) -> None:
        self.model.ainvoke.side_effect = RuntimeError("Model unavailable")
        output, error = io.StringIO(), io.StringIO()
        with (
            patch("sys.argv", ["agent_workflow.py", "--trace", "Find contracts"]),
            redirect_stdout(output), redirect_stderr(error),
        ):
            with self.assertRaises(SystemExit) as raised:
                await main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Node: handle_error", error.getvalue())
        self.assertIn("No state changes.", error.getvalue())
        self.assertIn("Error: LLM or tool discovery failed: Model unavailable", error.getvalue())
        self.assertIn("Agent error: LLM or tool discovery failed: Model unavailable", error.getvalue())
        self.assertNotIn("Final answer produced.", error.getvalue())

    async def test_trace_tool_error_preserves_round_and_call_details(self) -> None:
        self.model.ainvoke.return_value = tool_request()
        self.client.call_tool.side_effect = RuntimeError("Transport unavailable")
        output, error = io.StringIO(), io.StringIO()
        with (
            patch("sys.argv", ["agent_workflow.py", "--trace", "Find contracts"]),
            redirect_stdout(output), redirect_stderr(error),
        ):
            with self.assertRaises(SystemExit) as raised:
                await main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(output.getvalue(), "")
        trace = error.getvalue()
        self.assertIn("Node: execute_mcp_tools", trace)
        self.assertIn("tool_call_id=call-1", trace)
        self.assertIn("Tool rounds: 0/5", trace)
        self.assertNotIn("Tool rounds: 1/5", trace)
        self.assertIn("Node: handle_error\n  No state changes.", trace)
        self.assertIn(
            "Agent error: MCP tool execution failed: Transport unavailable", trace
        )
        self.assertNotIn("Final answer produced.", trace)

    async def test_trace_invalid_input_ends_before_model_call(self) -> None:
        output, error = io.StringIO(), io.StringIO()
        with (
            patch("sys.argv", ["agent_workflow.py", "--trace", " "]),
            redirect_stdout(output), redirect_stderr(error),
        ):
            with self.assertRaises(SystemExit):
                await main()
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Prompt must not be blank", error.getvalue())
        self.assertIn("Node: handle_error", error.getvalue())
        self.assertNotIn("Node: call_llm", error.getvalue())
        self.initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
