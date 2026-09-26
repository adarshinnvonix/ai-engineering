"""
Exercise 3: Combine TWO MCP servers into one agent.

Same as exercise 2, but now the agent has tools from both "filesystem" and "fetch"
servers at once, and we give it a task that needs both.

Needs `uv` installed (for the `uvx` command) - https://docs.astral.sh/uv/
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from model import get_model

WORKSPACE = os.path.abspath("./workspace")

servers = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE],
        "transport": "stdio",
    },
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "transport": "stdio",
    },
}


async def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()  # tools from BOTH servers, combined into one list

    agent = create_agent(get_model(), tools)

    question = (
        "Fetch https://example.com and save a one-sentence summary of it "
        "to summary.txt in the workspace."
    )
    result = await agent.ainvoke({"messages": question})

    print("Answer:", result["messages"][-1].content)


asyncio.run(main())