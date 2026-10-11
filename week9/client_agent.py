"""
Connects to the MCP server and lets an LLM use whatever tools it finds there - there is never
hardcoded "search_products" or "get_product_details" logic in this file. The agent
discovers them from the server at runtime.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from model import get_model

servers = {
    "catalog": {
        "command": "python",
        "args": ["server.py"],   # starts the MCP server as a subprocess automatically
        "transport": "stdio",
    }
}


async def main():
    client = MultiServerMCPClient(servers)

    # Discover the tools the server offers - no hardcoding.
    tools = await client.get_tools()
    print(f"Discovered tools: {[t.name for t in tools]}\n")

    # Resources work the same way - data the server offers to read directly.
    resources = await client.get_resources("catalog")
    print(f"Discovered {len(resources)} resource(s), e.g. the full catalog listing.\n")

    agent = create_agent(get_model(), tools)

    question = "Do you have any office furniture under $200? List what's available and the stock."
    result = await agent.ainvoke({"messages": question})

    print("Answer:", result["messages"][-1].content)


asyncio.run(main())