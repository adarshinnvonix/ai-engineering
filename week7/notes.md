# Week 7 - Research Assistant with LangGraph

One simple file (`research_assistant.py`) that builds a research assistant out of 4
small agents wired together as a graph, instead of one big prompt doing everything.

## Setup

```bash
pip install -r week7/modules.txt
```

## Run

```bash
python research_assistant.py
```

## How it's organized

```
START -> plan -> (needs research?) -> retrieve_a + retrieve_b (parallel) -> summarize -> answer -> END
                                    -> answer_direct -> END   (if no research needed)
```

- **plan** - decides if the question needs web research, and if so, comes up with 2 search queries.
- **retrieve_a / retrieve_b** - each searches one of the 2 queries. They run **in parallel**.
- **summarize** - combines both search results into one summary.
- **answer** - writes the final answer using that summary.
- **answer_direct** - a shortcut for questions that don't need research at all.

## Topics covered
- **LangGraph fundamentals** - instead of one long prompt/chain, you break a task into
  small steps (nodes) and describe how data flows between them (edges). Each node is just
  a normal Python function.
- **State management** - all nodes share one `State` object (here, a `TypedDict`) that
  flows through the graph. A node reads from it and returns updates to it. The
  `Annotated[List[str], operator.add]` on `research_results` tells LangGraph: "if two
  nodes update this in the same step, combine their results instead of one overwriting
  the other" - needed because `retrieve_a` and `retrieve_b` both write to it at once.
- **Nodes and edges** - a node is a function (`plan_node`, `retrieve_a`, etc.); an edge
  (`graph.add_edge("summarize", "answer")`) says "after this node, go to that one."
- **Conditional routing** - `route_after_plan()` looks at the state and decides where to
  go next: straight to research, or skip to a direct answer. Wired in with
  `add_conditional_edges` instead of a fixed `add_edge`.
- **Parallel execution** - `route_after_plan()` can return a *list* of node names
  (`["retrieve_a", "retrieve_b"]`) instead of just one. LangGraph runs everything in that
  list at the same time, then waits for both before moving on to `summarize` - that's the
  fan-out/fan-in pattern.