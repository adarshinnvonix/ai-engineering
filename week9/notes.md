# Week 9 - Build Own MCP Server

Two files: one is an MCP **server** that exposes a product catalog, the other is an LLM
**client** that connects to it and uses its tools.

## Setup

```bash
pip install -r week9/modules.txt
```

## Run

```bash
python week9/client_agent.py
```

## Files

| File | What it is |
|---|---|
| `server.py` | The MCP server - exposes a product catalog as 1 resource + 2 tools |
| `client_agent.py` | The MCP client - connects, discovers the tools, hands them to an LLM |

### MCP architecture
Two sides: a **server** that offers capabilities, and a **client** that connects to it and
uses them. They talk over a standard protocol, so any MCP client can talk to any MCP
server.

### MCP server and MCP client
- **Server** (`server.py`) - built with `FastMCP`. Just need to write normal Python functions
  and decorate them:
  - `@mcp.resource(...)` - exposes **data** the client can read directly (here: the whole
    catalog as one block of text).
  - `@mcp.tool()` - exposes an **action** the client can call with arguments (here:
    `search_products(query)` and `get_product_details(product_id)`). The docstring becomes
    the description the LLM sees.
- **Client** (`client_agent.py`) - connects using `MultiServerMCPClient`, then calls
  `get_tools()` and `get_resources()` to **discover** what the server offers at runtime.
  Nothing about "search_products" is written into the client file - it just asks the
  server what's available and uses whatever comes back.