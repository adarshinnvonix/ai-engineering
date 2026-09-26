"""
Exercise 1: Connect to an MCP server and list its tools.

We use the "filesystem" MCP server - an existing open-source server that lets an agent
read/write files in one folder. We don't write any tool code ourselves; we just connect.
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

WORKSPACE = os.path.abspath("./workspace")  # the only folder this server can touch

servers = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE],
        "transport": "stdio",
    }
}


async def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    client = MultiServerMCPClient(servers)

    tools = await client.get_tools()

    print(f"Connected to filesystem server. It exposes {len(tools)} tools:\n")
    for tool in tools:
        print(f"- {tool.name}: {tool.description[:80]}")


asyncio.run(main())