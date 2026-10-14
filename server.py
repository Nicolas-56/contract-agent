"""Expose read-only tools for synthetic contracts over MCP stdio."""

from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import TypeAdapter

from models import Contract


DATA_PATH = Path(__file__).resolve().parent / "data" / "contracts.json"
contracts = TypeAdapter(list[Contract]).validate_json(DATA_PATH.read_text(encoding="utf-8"))
if len({contract.id for contract in contracts}) != len(contracts):
    raise ValueError("Contract IDs must be unique.")

mcp = MCPServer("Synthetic Contracts")


@mcp.tool()
def list_contracts() -> list[Contract]:
    """Return all synthetic contracts in dataset order."""
    return contracts


@mcp.tool()
def get_contract(contract_id: str) -> Contract:
    """Return a contract by its exact, case-sensitive ID; fail if it is unknown."""
    for contract in contracts:
        if contract.id == contract_id:
            return contract
    raise ToolError(f"Unknown contract ID: {contract_id}")


@mcp.tool()
def search_contracts(query: str) -> list[Contract]:
    """Find a case-insensitive substring in any contract field; blank queries return []."""
    needle = query.strip().casefold()
    if not needle:
        return []
    return [
        contract
        for contract in contracts
        if any(
            needle in field.casefold()
            for field in (
                contract.id,
                contract.title,
                *contract.parties,
                contract.signed_on.isoformat(),
                contract.text,
            )
        )
    ]


@mcp.tool()
def find_contracts_by_party(party: str) -> list[Contract]:
    """Find exact party-name matches, ignoring surrounding whitespace and case."""
    contracts_list = []
    needle = party.strip().casefold()
    if not needle:
        return []

    for contract in contracts:
        for party_contract in contract.parties:
            if party_contract.strip().casefold() == needle:
                contracts_list.append(contract)
                break

    return contracts_list


@mcp.tool()
def find_contract_by_party(party: str) -> list[Contract]:
    """Compatibility alias for find_contracts_by_party."""
    return find_contracts_by_party(party)


if __name__ == "__main__":
    mcp.run(transport="stdio")
