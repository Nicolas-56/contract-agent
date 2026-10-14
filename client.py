"""Discover and call tools on a separate MCP server process."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve().with_name("server.py"))],
    )
    async with Client(server, read_timeout_seconds=10) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")
            print(json.dumps(tool.input_schema, indent=2))

        for name, arguments in (
            ("list_contracts", {}),
            ("get_contract", {"contract_id": "C-991"}),
            ("search_contracts", {"query": "harbor"}),
            ("find_contract_by_party", {"party": "Northstar Studio"}),
        ):
            result = await client.call_tool(name, arguments)
            if result.is_error:
                raise RuntimeError(f"{name} failed: {result.content}")
            print(f"\n{name}({json.dumps(arguments)}):")
            print(json.dumps(result.structured_content, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
