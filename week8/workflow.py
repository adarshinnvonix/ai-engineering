"""
A simple draft and send an email workflow that demonstrates all three topics at once:

  1. Persistence & Checkpoints - the graph saves its progress (via a checkpointer), so it
     can pause and be resumed later from exactly where it left off.
  2. Human-in-the-Loop - it PAUSES before the critical action (actually "sending") and
     waits for a real human to approve or cancel.
  3. Subgraphs - the "drafting" part (plan -> write) is its own small graph, built once
     and plugged into the main graph as a single reusable node.

Flow:
    START -> [draft subgraph: plan -> write] -> (PAUSE for human approval) -> send -> END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from model import get_model

model = get_model()


class State(TypedDict):
    topic: str
    plan: str
    draft: str
    result: str


# --- Subgraph: drafting (plan -> write) ----------------------------------------------
def plan_node(state: State) -> dict:
    resp = model.invoke(f"In 1 sentence, plan what an email about '{state['topic']}' should cover.")
    return {"plan": resp.content}


def write_node(state: State) -> dict:
    resp = model.invoke(f"Write a short email (3-4 sentences) about: {state['topic']}\nPlan: {state['plan']}")
    return {"draft": resp.content}


def build_draft_subgraph():
    sub = StateGraph(State)
    sub.add_node("plan", plan_node)
    sub.add_node("write", write_node)
    sub.add_edge(START, "plan")
    sub.add_edge("plan", "write")
    sub.add_edge("write", END)
    return sub.compile()  # a normal, reusable graph - could be dropped into ANY parent graph


# --- Main graph: draft (subgraph) -> send (critical action) -------------------------
def send_node(state: State) -> dict:
    # In a real app this would actually call an email API. Here we just simulate it.
    print(f"\n[SENDING] ...\n{state['draft']}\n[SENT]")
    return {"result": "sent"}


draft_subgraph = build_draft_subgraph()

main_graph = StateGraph(State)
main_graph.add_node("draft", draft_subgraph)   # <- the whole subgraph, used as ONE node
main_graph.add_node("send", send_node)
main_graph.add_edge(START, "draft")
main_graph.add_edge("draft", "send")
main_graph.add_edge("send", END)

# A checkpointer saves the graph's state after every step. MemorySaver keeps it in RAM
# (good for this demo). For persistence across restarts, swap in SqliteSaver instead -
# see the README for a one-line change.
checkpointer = MemorySaver()

# interrupt_before=["send"] tells LangGraph: pause right before running "send", every time.
app = main_graph.compile(checkpointer=checkpointer, interrupt_before=["send"])


if __name__ == "__main__":
    # thread_id identifies ONE specific run - the checkpointer uses it to know which
    # saved state to resume later.
    config = {"configurable": {"thread_id": "demo-1"}}

    result = app.invoke({"topic": "requesting a 2-day deadline extension"}, config=config)

    print("Draft ready:\n", result["draft"])
    print("\n--- Paused before sending (human-in-the-loop checkpoint) ---")

    approve = input("Approve sending this email? (yes/no): ").strip().lower()

    if approve == "yes":
        # Passing None + the SAME config resumes the graph from its saved checkpoint,
        # right where it paused - it does NOT re-run "draft" again.
        final = app.invoke(None, config=config)
        print("\nFinal result:", final["result"])
    else:
        print("\nCancelled by human - nothing was sent. The saved checkpoint is still")
        print("there if you change your mind and resume it later.")