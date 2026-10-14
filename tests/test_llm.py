"""Verify the standalone example without calling an LLM provider."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import main


class LLMExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_model_messages_and_output(self) -> None:
        for provider in ("openai", "anthropic", "ollama"):
            with self.subTest(provider=provider):
                output = io.StringIO()
                with (
                    patch.dict("os.environ", {
                        "LLM_PROVIDER": provider, "LLM_MODEL": "test-model"
                    }, clear=True),
                    patch("sys.argv", ["llm.py", "Explain MCP."]),
                    patch("llm.init_chat_model") as initialize,
                    redirect_stdout(output),
                ):
                    model = initialize.return_value
                    model.ainvoke = AsyncMock(return_value=AIMessage(content="Example answer"))
                    await main()

                initialize.assert_called_once_with(model="test-model", model_provider=provider)
                model.ainvoke.assert_awaited_once()
                messages = model.ainvoke.call_args.args[0]
                self.assertEqual(len(messages), 2)
                self.assertIsInstance(messages[0], SystemMessage)
                self.assertIsInstance(messages[1], HumanMessage)
                self.assertEqual(messages[1].content, "Explain MCP.")
                self.assertEqual(output.getvalue(), "Example answer\n")

    async def test_missing_configuration_does_not_initialize_model(self) -> None:
        for environment in ({}, {"LLM_PROVIDER": "openai"}, {"LLM_MODEL": "test-model"}):
            with self.subTest(environment=environment):
                error = io.StringIO()
                with (
                    patch.dict("os.environ", environment, clear=True),
                    patch("sys.argv", ["llm.py"]),
                    patch("llm.init_chat_model") as initialize,
                    redirect_stderr(error),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        await main()

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("Set both LLM_PROVIDER and LLM_MODEL", error.getvalue())
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
