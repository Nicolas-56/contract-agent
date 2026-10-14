"""Search contracts through MCP using a deterministic LangGraph workflow."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult
from pydantic import BaseModel, ValidationError

from models import Contract


class ContractSearchState(TypedDict):
    """Input, intermediate MCP response, and validated output for one run."""

    party: str
    mcp_result: NotRequired[CallToolResult]
    contracts: NotRequired[list[Contract]]
    error: NotRequired[str]


class ContractSearchResult(BaseModel):
    """Structured MCP envelope for a list-returning tool."""

    result: list[Contract]


def validate_request(state: ContractSearchState) -> dict[str, str]:
    """Reject non-string or blank input and normalize the search name."""
    party = state.get("party")
    if not isinstance(party, str):
        return {"error": "Party must be a string."}
    party = party.strip().casefold()
    if not party:
        return {"error": "Party must not be blank."}
    return {"party": party}


async def call_mcp_tool(state: ContractSearchState) -> dict[str, CallToolResult | str]:
    print("call tool")
    """Call the tool over stdio in a separate server process."""
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve().with_name("server.py"))],
    )
    try:
        async with Client(server, read_timeout_seconds=10) as client:
            result = await client.call_tool(
                "find_contracts_by_party", {"party": state["party"]}
            )
            if result.is_error:
                return {"error": "Tool call failed"}
    except Exception as exc:
        return {"error": f"MCP call failed: {exc}"}

    return {"mcp_result": result}


def validate_result(state: ContractSearchState) -> dict[str, list[Contract] | str]:
    """Check tool success, response schema, and party-search invariants."""
    response = state["mcp_result"]
    if response.is_error:
        return {"error": f"find_contracts_by_party failed: {response.content}"}
    try:
        contracts = ContractSearchResult.model_validate(response.structured_content).result
    except ValidationError as exc:
        return {"error": f"Invalid MCP response: {exc}"}
    if len({contract.id for contract in contracts}) != len(contracts):
        return {"error": "Contract IDs must be unique."}
    if any(
        not any(party.strip().casefold() == state["party"] for party in contract.parties)
        for contract in contracts
    ):
        return {"error": "Every returned contract must include the requested party."}
    return {"contracts": contracts}

def route_after_validation(state: ContractSearchState) -> str:
    if state.get("error"):
        return "error"


    return "valid"


def handle_error(state: ContractSearchState) -> dict:
    print(f"Workflow error: {state['error']}")
    return {}


builder = StateGraph(ContractSearchState)
builder.add_node("validate_request", validate_request)
builder.add_node("call_mcp_tool", call_mcp_tool)
builder.add_node("validate_result", validate_result)
builder.add_node("handle_error", handle_error)
builder.add_edge(START, "validate_request")
builder.add_conditional_edges(
    "validate_request",
    route_after_validation,
    {
        "valid": "call_mcp_tool",
        "error": "handle_error",
    },
)

builder.add_conditional_edges(
    "call_mcp_tool",
    route_after_validation,
    {
        "valid": "validate_result",
        "error": "handle_error",
    },
)
builder.add_conditional_edges(
    "validate_result",
    route_after_validation,
    {
        "valid": END,
        "error": "handle_error",
    },
)
builder.add_edge("handle_error", END)
graph = builder.compile()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("party", nargs="?", default="Northstar Studio")
    args = parser.parse_args()
    result = await graph.ainvoke({"party": args.party})
    if result.get("error"):
        print(f"Workflow main error: {result['error']}")
        return

    print(json.dumps(
        [contract.model_dump(mode="json") for contract in result["contracts"]],
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    asyncio.run(main())
