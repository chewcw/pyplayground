#!/usr/bin/env python3
"""Sandbox prototype for DeepAgents + MCP tool layer.

This script demonstrates how to load MCP tools via langchain_mcp_adapters
and pass them into a DeepAgents agent as a tool set.

It is intentionally minimal to prove feasibility in a dedicated sandbox env.
"""

import asyncio
import os

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def main() -> None:
    model_name = os.environ.get("DEEPAGENT_MODEL", "ollama:gemma4:31b-cloud")

    # Configure the MCP server that exposes Playwright actions.
    # This example uses a local stdio-based MCP process.
    client = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "stdio",
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
            }
        }
    )

    async with client.session("playwright") as session:
        tools = await load_mcp_tools(
            session,
            server_name="playwright",
            tool_name_prefix=True,
        )

        print(f"Loaded {len(tools)} MCP tools from Playwright")
        for tool in tools:
            print(f"- {tool.name}: {getattr(tool, 'description', '')}")

        agent = create_deep_agent(
            model=model_name,
            tools=tools,
            system_prompt=(
                "You are a browser automation companion. Use the available Playwright "
                "tools to satisfy the user's request and avoid making up browser actions."
            ),
        )

        inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use Playwright to navigate to example.com, let me know what do you see on the page."
                    ),
                }
            ]
        }

        print("Sending prompt to the agent...\n")
        result = await agent.ainvoke(inputs)

        print("\n=== AGENT RESPONSE ===")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
