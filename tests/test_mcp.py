"""Integration tests using a real server subprocess and MCP stdio."""

import sys
import unittest
from pathlib import Path

from mcp import Client, StdioServerParameters


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.server = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).resolve().parents[1] / "server.py")],
            cwd="/tmp",
        )

    async def test_tool_discovery_and_argument_schemas(self) -> None:
        async with Client(self.server, read_timeout_seconds=10) as client:
            result = await client.list_tools()
            tools = {tool.name: tool for tool in result.tools}
            self.assertEqual(set(tools), {
                "list_contracts", "get_contract", "search_contracts",
                "find_contract_by_party", "find_contracts_by_party",
            })
            for name, parameter in (
                ("get_contract", "contract_id"), ("search_contracts", "query"),
                ("find_contract_by_party", "party"), ("find_contracts_by_party", "party"),
            ):
                schema = tools[name].input_schema
                self.assertEqual(schema["required"], [parameter])
                self.assertEqual(schema["properties"][parameter]["type"], "string")

    async def test_list_and_get_contract(self) -> None:
        async with Client(self.server, read_timeout_seconds=10) as client:
            listed = await client.call_tool("list_contracts")
            self.assertFalse(listed.is_error)
            contracts = listed.structured_content["result"]
            self.assertEqual(
                [contract["id"] for contract in contracts],
                ["C-001", "C-002", "C-003", "C-004"],
            )
            for contract in contracts:
                result = await client.call_tool("get_contract", {"contract_id": contract["id"]})
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content, contract)

    async def test_search(self) -> None:
        cases = (
            ("  HARBOR  ", ["C-002", "C-003", "C-004"]),
            ("maintenance", ["C-001"]),
            ("30 days", ["C-001"]),
            ("c-003", ["C-003"]),
            ("2026-02-10", ["C-002"]),
            ("no matching contract", []),
            ("", []),
            ("   ", []),
        )
        async with Client(self.server, read_timeout_seconds=10) as client:
            for query, expected_ids in cases:
                with self.subTest(query=query):
                    result = await client.call_tool("search_contracts", {"query": query})
                    self.assertFalse(result.is_error)
                    self.assertEqual(
                        [contract["id"] for contract in result.structured_content["result"]],
                        expected_ids,
                    )

    async def test_unknown_id_and_argument_validation(self) -> None:
        async with Client(self.server, read_timeout_seconds=10) as client:
            unknown = await client.call_tool("get_contract", {"contract_id": "missing"})
            self.assertTrue(unknown.is_error)
            self.assertIn("Unknown contract ID: missing", unknown.content[0].text)
            for name, arguments in (
                ("get_contract", {}),
                ("get_contract", {"contract_id": 123}),
                ("search_contracts", {"query": ["harbor"]}),
            ):
                with self.subTest(name=name, arguments=arguments):
                    result = await client.call_tool(name, arguments)
                    self.assertTrue(result.is_error)
            healthy = await client.call_tool("list_contracts")
            self.assertFalse(healthy.is_error)

    async def test_find_contracts_by_party_and_compatibility_alias(self) -> None:
        async with Client(self.server, read_timeout_seconds=10) as client:
            for name in ("find_contracts_by_party", "find_contract_by_party"):
                for party, expected in (
                    ("  HARBOR LABS  ", ["C-002", "C-003", "C-004"]),
                    ("Northstar Studio", ["C-001"]),
                    ("Harbor", []), ("missing", []), ("   ", []),
                ):
                    with self.subTest(name=name, party=party):
                        result = await client.call_tool(name, {"party": party})
                        self.assertFalse(result.is_error)
                        self.assertEqual(
                            [item["id"] for item in result.structured_content["result"]],
                            expected,
                        )


if __name__ == "__main__":
    unittest.main()
