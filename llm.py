"""Run one standalone chat-model request configured through the environment."""

import argparse
import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt", nargs="?", default="Explain MCP and LangGraph in three sentences."
    )
    args = parser.parse_args()
    provider = os.environ.get("LLM_PROVIDER", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not provider or not model:
        parser.error("Set both LLM_PROVIDER and LLM_MODEL environment variables.")

    chat_model = init_chat_model(model=model, model_provider=provider)
    response = await chat_model.ainvoke([
        SystemMessage(content="You are a helpful tutor. Keep explanations concise."),
        HumanMessage(content=args.prompt),
    ])
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
