"""Run one chat-model request through a minimal LangGraph workflow."""

import argparse
import asyncio
import os
import sys
from typing import Literal, NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class LLMWorkflowState(TypedDict):
    """Input prompt, generated response, and optional error for one run."""

    prompt: str
    response: NotRequired[str]
    error: NotRequired[str | None]


def validate_input(state: LLMWorkflowState) -> dict[str, str | None]:
    """Validate and normalize the prompt without mutating the input state."""
    prompt = state.get("prompt")
    if not isinstance(prompt, str):
        return {"error": "Prompt must be a string."}
    prompt = prompt.strip()
    if not prompt:
        return {"error": "Prompt must not be blank."}
    return {"prompt": prompt, "error": None}


async def call_llm(state: LLMWorkflowState) -> dict[str, str]:
    """Initialize and invoke the configured model, returning failures as state."""
    provider = os.environ.get("LLM_PROVIDER", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not provider or not model:
        return {"error": "Set both LLM_PROVIDER and LLM_MODEL environment variables."}

    try:
        chat_model = init_chat_model(model=model, model_provider=provider)
        response = await chat_model.ainvoke([
            SystemMessage(content="You are a helpful tutor. Keep explanations concise."),
            HumanMessage(content=state["prompt"]),
        ])
        return {"response": response.text}
    except Exception as exc:
        return {"error": f"LLM request failed: {exc}"}


def route_after_node(state: LLMWorkflowState) -> Literal["success", "error"]:
    """Select the next edge using only the error state."""
    return "error" if state.get("error") else "success"


def handle_error(state: LLMWorkflowState) -> dict[str, str]:
    """Finish the error path while preserving the error already in state."""
    return {}


builder = StateGraph(LLMWorkflowState)
builder.add_node("validate_input", validate_input)
builder.add_node("call_llm", call_llm)
builder.add_node("handle_error", handle_error)
builder.add_edge(START, "validate_input")
builder.add_conditional_edges(
    "validate_input", route_after_node,
    {"success": "call_llm", "error": "handle_error"},
)
builder.add_conditional_edges(
    "call_llm", route_after_node,
    {"success": END, "error": "handle_error"},
)
builder.add_edge("handle_error", END)
graph = builder.compile()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    args = parser.parse_args()
    result = await graph.ainvoke({"prompt": args.prompt})
    if result.get("error"):
        print(f"Workflow error: {result['error']}", file=sys.stderr)
        raise SystemExit(1)
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
