# Week 6 - MCP + LangChain

Connect a LangChain agent to ready-made MCP servers instead of writing custom tools.

## Setup

```bash
pip install -r week6/modules.txt
```

Also need:
- **Node.js** installed (runs the filesystem server via `npx`)
- **uv** installed, only for file 3 (runs the fetch server via `uvx`) — https://docs.astral.sh/uv/

No accounts or API keys needed for the MCP servers themselves.

## Run

```bash
python list_mcp_tools.py       # connect + see what tools the server offers
python mcp_react_agent.py      # give those tools to an LLM, let it use them
python multi_server_agent.py   # optional: two servers, one agent
```

## What's happening, in short

- **MCP server** - a program someone else built that offers ready-made tools (here:
  reading/writing files, fetching web pages). You don't write the tool code.
- **MCP client** (`MultiServerMCPClient`) — connects to one or more servers using a
  small config dict (command to run + how to talk to it).
- **Loading tools** - `await client.get_tools()` turns whatever the server(s) offer into
  normal LangChain tools, ready to hand to a model.
- **ReAct agent** - `create_agent(model, tools)` builds an agent that reasons in a loop:
  look at the question → decide if a tool is needed → call it → read the result → repeat
  until it can answer.
- **stdio transport** - the client starts the server as a local background process and
  talks to it directly (what all 3 files use here).
- **HTTP/SSE transport** - same idea, but for a server hosted elsewhere, connected via a
  URL instead of starting a local process. Not used here since our servers are local.
- **Single vs multiple servers** - just add more entries to the `servers` dict (see file 3).
  `get_tools()` combines everything into one list either way, so the agent can freely mix
  tools from different servers in a single task.