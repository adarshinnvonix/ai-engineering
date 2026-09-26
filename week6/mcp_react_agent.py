"""
Exercise 2: Give those MCP tools to an LLM and let it decide when to use them.

This is a ReAct agent: the model looks at the question, decides if it needs a tool,
calls it, reads the result, and repeats until it can answer.
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
    }
}


async def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()

    agent = create_agent(get_model(), tools)

    question = "Create a file called hello.txt with the text 'Hi from MCP', then read it back."
    result = await agent.ainvoke({"messages": question})

    print("Answer:", result["messages"][-1].content)


asyncio.run(main())