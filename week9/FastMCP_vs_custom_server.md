# Custom MCP Server vs FastMCP Server

This README explains both servers, compares them, describes where each one fits, and ends with a single client that uses both servers together.

## The idea

An MCP server is a small program that offers **tools** (actions) and **resources** (read-only data) to an AI application through a standard protocol. An MCP client connects, asks what the server offers, and then uses it. Since the protocol is the same, a client cannot tell how a server was written. What differs between the two approaches is how much of the plumbing is written by hand (manual configuration).

## Setup

Install the packages with:

```bash
pip install "mcp<2" fastmcp langchain-mcp-adapters
pip install langchain langchain-groq     # only for the agent version at the end
```

`mcp<2` has been pinned because `langchain-mcp-adapters` failed to import against `mcp` 2.x. The official SDK also changed in 2.x: its built-in FastMCP class was renamed `MCPServer`. The standalone `fastmcp` package keeps the `FastMCP` name.

Versions used: `mcp` 1.30.0, `fastmcp` 3.4.11, `langchain-mcp-adapters` 0.3.2.

## Server 1: the custom (low-level) MCP server

This server is built on `mcp.server.Server` from the official SDK. It exposes one tool, `search_products`. Registered a handler for each protocol request.

```python
# low_server.py
import asyncio
import mcp.types as types
import mcp.server.stdio
from mcp.server import Server

server = Server("catalog")

PRODUCTS = ["Standing Desk", "Desk Lamp", "Office Chair"]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_products",
            description="Search products by keyword.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_products":
        hits = [p for p in PRODUCTS if arguments["query"].lower() in p.lower()]
        return [types.TextContent(type="text", text=", ".join(hits) or "No match")]
    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


asyncio.run(main())
```

Summary:

1. **The tool description.** `list_tools` answers "what tools exist?" with a name, a description and a JSON schema for the input. I write the schema myself.
2. **The routing.** `call_tool` receives every tool call, so I check `name` to decide what to run.
3. **The result format.** wrapped the text in `types.TextContent`.
4. **The startup.** `main()` opens the stdin/stdout streams and runs the server loop.

Resources work the same way, with their own pair of handlers (`list_resources` and `read_resource`).

## Server 2: the FastMCP server

This server uses the `fastmcp` package. It exposes one tool, `find_employees`, and one resource, `hr://leave-policy`.

```python
# fast_server.py
from fastmcp import FastMCP

mcp = FastMCP("employees")

EMPLOYEES = [
    {"name": "Asha Patel", "team": "Data", "role": "Analyst"},
    {"name": "Ravi Shah", "team": "Data", "role": "Engineer"},
    {"name": "Meera Joshi", "team": "HR", "role": "Manager"},
]


@mcp.tool()
def find_employees(team: str) -> str:
    """List employees (name and role) on a given team, e.g. 'Data'."""
    rows = [f"{e['name']} - {e['role']}" for e in EMPLOYEES if e["team"].lower() == team.lower()]
    return "\n".join(rows) or "No employees found on that team."


@mcp.resource("hr://leave-policy")
def leave_policy() -> str:
    """Short summary of the leave policy."""
    return "12 casual leaves and 8 sick leaves per year."


mcp.run()
```

What FastMCP does here:

1. **The tool description is generated.** The name comes from the function name, the description from the docstring, and the input schema from the type hint (`team: str`).
2. **The routing is automatic.** Each decorated function is registered and called by name.
3. **The result is plain Python.** Returning a string is enough.
4. **The resource is one decorator.** `@mcp.resource("hr://leave-policy")` publishes read-only data at a URI.
5. **The startup is one line.** `mcp.run()`.

## Differences

| | Custom (low-level) server | FastMCP server |
|---|---|---|
| Declaring a tool | Handler that returns a list of `types.Tool` objects | `@mcp.tool()` above a function |
| Input schema | Written by hand as JSON | Generated from type hints |
| Tool description | Written by hand in the schema object | Taken from the docstring |
| Routing tool calls | One `call_tool` function with `if name == ...` | Automatic |
| Return value | Wrapped in `types.TextContent` | Plain Python value |
| Resources | Separate `list_resources` and `read_resource` handlers | `@mcp.resource("uri")` |
| Starting the server | Open stdio streams, run the loop | `mcp.run()` |
| Size for the same single tool | 40 lines (29 of code) | 16 lines (9 of code) |
| Control over protocol details | Full | Less direct |
| Easy mistakes | A typo in the schema, or in the tool name used for routing | Fewer places for a mismatch |
| What the client sees | Same | Same |

## Use cases

**Use of custom (low-level) server when:**
- Want to see and control every protocol message, for example to understand how MCP works underneath.
- Need behavior that a decorator does not express, such as handling requests or building responses by hand.
- The server has to be written directly against the SDK's building blocks, for example inside an existing service.

**Use of FastMCP when:**
- Exposing an existing system (an employee directory, a product catalog, a ticketing tool) through a handful of functions.
- Prototyping and want a working server quickly.
- A server has many tools and resources, where hand-written schemas and routing would be repetitive and easy to get wrong.

## Using both servers together

Since a client sees both servers the same way, one client can connect to both. The `catalog` server is the custom one and the `employees` server is the FastMCP one. I gave their tools different names (`search_products` and `find_employees`) because an agent picks tools by name.

I build the server paths from the script's own location so the script works from any folder, and I use `sys.executable` so the servers run in the same Python environment as the client.

```python
# use_both_servers.py
import asyncio
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

HERE = Path(__file__).resolve().parent

servers = {
    "catalog": {                                  # custom (low-level) server
        "command": sys.executable,
        "args": [str(HERE / "low_server.py")],
        "transport": "stdio",
    },
    "employees": {                                # FastMCP server
        "command": sys.executable,
        "args": [str(HERE / "fast_server.py")],
        "transport": "stdio",
    },
}


async def main():
    client = MultiServerMCPClient(servers)
    tools = {t.name: t for t in await client.get_tools()}

    for name, tool in tools.items():
        print(f"{name}: {tool.description} | input: {tool.args}")

    print()
    print("search_products ->", (await tools["search_products"].ainvoke({"query": "desk"}))[0]["text"])
    print("find_employees  ->", (await tools["find_employees"].ainvoke({"team": "Data"}))[0]["text"].replace("\n", "; "))

    resources = await client.get_resources("employees")
    print("resource        ->", resources[0].as_string())


if __name__ == "__main__":
    asyncio.run(main())
```

Output I got from `python use_both_servers.py`:

```text
search_products: Search products by keyword. | input: {'query': {'type': 'string'}}
find_employees: List employees (name and role) on a given team, e.g. 'Data'. | input: {'team': {'type': 'string'}}

search_products -> Standing Desk, Desk Lamp
find_employees  -> Asha Patel - Analyst; Ravi Shah - Engineer
resource        -> 12 casual leaves and 8 sick leaves per year.
```

The schema for `search_products` was written by hand in `low_server.py`, and the one for `find_employees` was generated by FastMCP from `team: str`. The client prints them in the same shape.

### Agent version

The same two servers can feed one LLM agent. A single question then uses both: the catalog server answers the product part and the employee server answers the team part. This file needs `GROQ_API_KEY` set in the environment.

```python
# agent_version.py
import asyncio
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from use_both_servers import servers


async def main():
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()              # tools from both servers, in one list

    agent = create_agent(ChatGroq(model="llama-3.3-70b-versatile"), tools)
    result = await agent.ainvoke({"messages": "Who is on the Data team, and does the catalog include any desks?"})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

## What was tested

- Ran `use_both_servers.py` as shown, including from a different folder, with the versions listed in Setup.
- Wrote the same single tool both ways and checked that the client gets the same name, description, schema and result.
- For `agent_version.py`, checked the wiring with a scripted stand-in model instead of Groq. It called one tool on each server and combined the results.
- Did not run the low-level server on `mcp` 2.x.
