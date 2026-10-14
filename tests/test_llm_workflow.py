"""Verify LLM workflow validation, routing, and CLI without provider calls."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_workflow import graph, main, validate_input


class LLMWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_path_messages_and_normalization(self) -> None:
        with (
            patch.dict("os.environ", {
                "LLM_PROVIDER": " ollama ", "LLM_MODEL": " qwen3:8b "
            }, clear=True),
            patch("llm_workflow.init_chat_model") as initialize,
        ):
            model = initialize.return_value
            model.ainvoke = AsyncMock(return_value=AIMessage(content="Example answer"))
            updates = [
                update async for update in graph.astream(
                    {"prompt": "  Explain MCP. \n"}, stream_mode="updates"
                )
            ]

        self.assertEqual(
            [next(iter(update)) for update in updates],
            ["validate_input", "call_llm"],
        )
        self.assertEqual(updates[0]["validate_input"]["prompt"], "Explain MCP.")
        self.assertEqual(updates[1]["call_llm"]["response"], "Example answer")
        initialize.assert_called_once_with(model="qwen3:8b", model_provider="ollama")
        model.ainvoke.assert_awaited_once()
        messages = model.ainvoke.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "Explain MCP.")

    async def test_invalid_input_never_initializes_model(self) -> None:
        for request in ({}, {"prompt": None}, {"prompt": 123},
                        {"prompt": ""}, {"prompt": " \t\n "}):
            with self.subTest(request=request), patch(
                "llm_workflow.init_chat_model"
            ) as initialize:
                updates = [
                    update async for update in graph.astream(
                        request, stream_mode="updates"
                    )
                ]
                self.assertEqual(
                    [next(iter(update)) for update in updates],
                    ["validate_input", "handle_error"],
                )
                self.assertTrue(updates[0]["validate_input"]["error"])
                initialize.assert_not_called()

    async def test_missing_configuration_returns_error(self) -> None:
        for environment in ({}, {"LLM_PROVIDER": "ollama"},
                            {"LLM_MODEL": "qwen3:8b"},
                            {"LLM_PROVIDER": " ", "LLM_MODEL": "qwen3:8b"}):
            with (
                self.subTest(environment=environment),
                patch.dict("os.environ", environment, clear=True),
                patch("llm_workflow.init_chat_model") as initialize,
            ):
                result = await graph.ainvoke({"prompt": "Explain MCP."})
                self.assertIn("Set both", result["error"])
                self.assertNotIn("response", result)
                initialize.assert_not_called()

    async def test_model_failures_follow_error_path(self) -> None:
        for stage in ("initialization", "invocation"):
            with (
                self.subTest(stage=stage),
                patch.dict("os.environ", {
                    "LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3:8b"
                }, clear=True),
                patch("llm_workflow.init_chat_model") as initialize,
            ):
                failure = RuntimeError("Provider unavailable")
                if stage == "initialization":
                    initialize.side_effect = failure
                else:
                    initialize.return_value.ainvoke = AsyncMock(side_effect=failure)
                updates = [
                    update async for update in graph.astream(
                        {"prompt": "Explain MCP."}, stream_mode="updates"
                    )
                ]
                self.assertEqual(
                    [next(iter(update)) for update in updates],
                    ["validate_input", "call_llm", "handle_error"],
                )
                self.assertIn("Provider unavailable", updates[1]["call_llm"]["error"])
                result = await graph.ainvoke({"prompt": "Explain MCP."})
                self.assertIn("Provider unavailable", result["error"])
                self.assertNotIn("response", result)

    async def test_cli_success_and_failure_output(self) -> None:
        for result in ({"response": "Example answer"}, {"error": "Provider unavailable"}):
            output, error = io.StringIO(), io.StringIO()
            with (
                self.subTest(result=result),
                patch("sys.argv", ["llm_workflow.py", "Explain MCP."]),
                patch("llm_workflow.graph.ainvoke", new=AsyncMock(return_value=result))
                as invoke,
                redirect_stdout(output), redirect_stderr(error),
            ):
                if "error" in result:
                    with self.assertRaises(SystemExit) as raised:
                        await main()
                    self.assertEqual(raised.exception.code, 1)
                    self.assertEqual(output.getvalue(), "")
                    self.assertIn(result["error"], error.getvalue())
                else:
                    await main()
                    self.assertEqual(output.getvalue(), "Example answer\n")
                    self.assertEqual(error.getvalue(), "")
                invoke.assert_awaited_once_with({"prompt": "Explain MCP."})


class InputValidationTests(unittest.TestCase):
    def test_validation_does_not_mutate_input(self) -> None:
        state = {"prompt": "  Explain MCP.  "}
        update = validate_input(state)
        self.assertEqual(state, {"prompt": "  Explain MCP.  "})
        self.assertEqual(update, {"prompt": "Explain MCP.", "error": None})


if __name__ == "__main__":
    unittest.main()
