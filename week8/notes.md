# Week 8 - Checkpoints, Human-in-the-Loop & Subgraphs

One file (`workflow.py`): a draft and send an email workflow that pauses for your
approval before actually sending.

## Setup

```bash
pip install -r week8/modules.txt
```

## Run

```bash
python workflow.py
```

It will draft an email, then stop and ask: `Approve sending this email? (yes/no):`
Type `yes` and it resumes and "sends" it. Type anything else and it stops, nothing sent.

## How it's organized

```
START -> [draft subgraph: plan -> write] -> (PAUSE, wait for you) -> send -> END
```

- **plan / write** - two small steps that together draft the email. They're bundled into
  their own mini-graph (a **subgraph**), then plugged into the main graph as a single node
  called `"draft"`.
- **send** - the "critical action" (simulated here with a `print`, but imagine a real
  email API). The graph is told to pause right before this node.

## Topics covered

### Persistence & Checkpoints
Every time the graph finishes a step, LangGraph saves the current state via a
**checkpointer** - here, `MemorySaver()` (keeps it in memory while the program runs). Each
run is tagged with a `thread_id` (`"demo-1"` in the code) so the checkpointer knows which
saved progress belongs to which run.

When you call `app.invoke(None, config=config)` - passing `None` instead of a new input -
LangGraph doesn't start over. It loads the last saved checkpoint for that `thread_id` and
continues from exactly there. That's the whole idea of "interrupt and resume": the state
isn't lost just because execution paused.

> `MemorySaver` only lives as long as the Python process runs. For persistence that
> survives a restart or crash, swap it for a file-based one:
> ```python
> from langgraph.checkpoint.sqlite import SqliteSaver
> checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
> ```
> Everything else in the code stays the same.

### Human-in-the-Loop
`app.compile(checkpointer=checkpointer, interrupt_before=["send"])` tells LangGraph:
"always stop right before running the `send` node." The graph literally halts and hands
control back to your code - which is where we ask a real person to type `yes` or `no`
before deciding whether to resume it. This is the pattern for anything you don't want an
AI to do unsupervised: sending an email, making a payment, deleting something, etc.

### Subgraphs
`build_draft_subgraph()` builds and compiles a small, complete graph on its own - it has
no idea it will later be used inside a bigger graph. Then in the main graph, we just do:
```python
main_graph.add_node("draft", draft_subgraph)
```
A compiled subgraph can be added as a node exactly like a regular function. This is what
makes it "reusable" - you could drop that same `draft_subgraph` into a completely
different parent workflow without changing anything inside it.