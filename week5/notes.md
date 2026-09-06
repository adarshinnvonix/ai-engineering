# Week 5 - Tools & Agents

## Setup
```bash
pip install -r week5/modules.txt
```

## Files

| File | What it shows |
|---|---|
| `prebuilt_tool.py` | Using a ready-made LangChain tool (DuckDuckGo search), bound to an LLM |
| `custom_tool.py` | Writing custom tools (`get_weather`, `get_stock_price`) with `@tool`, bound to an LLM |
| `react_agent.py` | A ReAct agent using all 3 tools together, chaining multiple tool calls to answer one question |

Run in order:
```bash
python prebuilt_tool.py
python custom_tool.py
python react_agent.py
```

## Concepts, simply

- **Pre-built tools** — LangChain ships ready-made tools (search, Wikipedia, etc.)
- **Custom tools** — any Python function can become a tool with `@tool`. The docstring is
  what the LLM reads to decide when/how to use it.
- **Tool calling** — the LLM never runs a tool itself. It just replies "call this tool with
  this input." The code actually runs it and sends the result back as a `ToolMessage`.
- **ReAct pattern** — Reason + Act: the agent loops through **Thought** (what should I do
  next?) → **Action** (call a tool) → **Observation** (read the result) → repeat, until it
  has enough to give a **Final Answer**. Run `react_agent.py` with `verbose=True` and
  this loop would be printed step by step in the terminal.