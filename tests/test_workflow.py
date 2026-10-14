
"""Test graph execution and contract result validation."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, TextContent

from workflow import graph, validate_result, validate_request


class WorkflowIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_node_order_normalization_and_real_mcp_result(self) -> None:
        updates = [
            update
            async for update in graph.astream(
                {"party": "  HARBOR LABS  "},
                stream_mode="updates",
            )
        ]

        self.assertEqual(
            [next(iter(update)) for update in updates],
            ["validate_request", "call_mcp_tool", "validate_result"],
        )
        self.assertEqual(
            updates[0]["validate_request"]["party"],
            "harbor labs",
        )
        self.assertFalse(
            updates[1]["call_mcp_tool"]["mcp_result"].is_error
        )
        self.assertEqual(
            [
                contract.id
                for contract in updates[2]["validate_result"]["contracts"]
            ],
            ["C-002", "C-003", "C-004"],
        )

    async def test_no_matches_are_a_valid_result(self) -> None:
        for party in ("unknown party", "Harbor"):
            with self.subTest(party=party):
                result = await graph.ainvoke({"party": party})

                self.assertNotIn("error", result)
                self.assertEqual(result["contracts"], [])

    async def test_invalid_input_never_starts_mcp(self) -> None:
        requests = (
            {},
            {"party": ""},
            {"party": " \t\n "},
            {"party": None},
            {"party": 123},
            {"party": ["Harbor Labs"]},
        )

        with patch("workflow.Client") as client:
            for request in requests:
                with self.subTest(request=request):
                    result = await graph.ainvoke(request)

                    self.assertIn("error", result)
                    self.assertNotIn("contracts", result)

            client.assert_not_called()

    async def test_mcp_connection_failure(self) -> None:
        with patch("workflow.Client", side_effect=RuntimeError("Connection failed")):
            result = await graph.ainvoke({"party": "Harbor Labs"})

        self.assertIn("error", result)
        self.assertIn("Connection failed", result["error"])
        self.assertNotIn("contracts", result)

    async def test_malformed_mcp_response_uses_error_path(self) -> None:
        response = CallToolResult(content=[], structured_content={"result": [{}]})
        with patch("workflow.Client") as client:
            session = client.return_value.__aenter__.return_value
            session.call_tool = AsyncMock(return_value=response)
            updates = [
                update
                async for update in graph.astream(
                    {"party": "Harbor Labs"}, stream_mode="updates"
                )
            ]

        self.assertEqual(
            [next(iter(update)) for update in updates],
            ["validate_request", "call_mcp_tool", "validate_result", "handle_error"],
        )
        validation = updates[2]["validate_result"]
        self.assertIn("Invalid MCP response", validation["error"])
        self.assertNotIn("contracts", validation)
        self.assertIn("handle_error", updates[3])
        session.call_tool.assert_awaited_once_with(
            "find_contracts_by_party", {"party": "harbor labs"}
        )


class RequestValidationTests(unittest.TestCase):
    def test_normalizes_party(self) -> None:
        result = validate_request({"party": "  HARBOR LABS  "})

        self.assertEqual(result, {"party": "harbor labs"})

    def test_rejects_blank_party(self) -> None:
        result = validate_request({"party": "   "})

        self.assertEqual(
            result,
            {"error": "Party must not be blank."},
        )

    def test_rejects_non_string_party(self) -> None:
        result = validate_request({"party": 123})

        self.assertEqual(
            result,
            {"error": "Party must be a string."},
        )


class ResultValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "contracts.json"
        )
        self.contract = json.loads(
            dataset.read_text(encoding="utf-8")
        )[0]

    def test_mcp_tool_error(self) -> None:
        response = CallToolResult(
            is_error=True,
            content=[
                TextContent(type="text", text="Tool failed")
            ],
        )

        result = validate_result({
            "party": "northstar studio",
            "mcp_result": response,
        })

        self.assertIn("Tool failed", result["error"])
        self.assertNotIn("contracts", result)

    def test_malformed_structured_results(self) -> None:
        payloads = (
            None,
            [],
            {},
            {"result": None},
            {"result": [{}]},
            {
                "result": [
                    dict(self.contract, signed_on="invalid-date")
                ]
            },
        )

        for payload in payloads:
            with self.subTest(payload=payload):
                result = validate_result({
                    "party": "northstar studio",
                    "mcp_result": CallToolResult(
                        content=[],
                        structured_content=payload,
                    ),
                })

                self.assertIn("Invalid MCP response", result["error"])
                self.assertNotIn("contracts", result)

    def test_duplicate_ids_and_unrelated_parties(self) -> None:
        cases = (
            (
                "northstar studio",
                [self.contract, self.contract],
                "unique",
            ),
            (
                "harbor labs",
                [self.contract],
                "requested party",
            ),
        )

        for party, contracts, message in cases:
            with self.subTest(party=party):
                result = validate_result({
                    "party": party,
                    "mcp_result": CallToolResult(
                        content=[],
                        structured_content={"result": contracts},
                    ),
                })

                self.assertIn(message, result["error"])
                self.assertNotIn("contracts", result)


if __name__ == "__main__":
    unittest.main()
