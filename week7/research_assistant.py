"""
Research Assistant built with LangGraph.

Four agents, coordinated as a graph:
  plan       -> decides if research is needed, and if so, 2 search queries
  retrieve_a \
  retrieve_b  }-> run in PARALLEL, each searches one query
  summarize  -> combines both search results into one summary
  answer     -> writes the final answer using the summary

If the question doesn't need research (e.g. "what is 2+2"), it skips straight to a
direct-answer node instead - that's the conditional routing part.
"""

import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun
from model import get_model

search = DuckDuckGoSearchRun()
model = get_model()


# --- 1. State: the shared data that flows through every node ------------------------
# Annotated[..., operator.add] tells LangGraph: when TWO nodes update this field in the
# same step (our parallel retrieve_a/retrieve_b), COMBINE their results instead of one
# overwriting the other.
class State(TypedDict):
    question: str
    needs_research: bool
    plan: List[str]
    research_results: Annotated[List[str], operator.add]
    summary: str
    final_answer: str


# --- 2. Nodes: each one is just a function that reads state and returns updates -----
def plan_node(state: State) -> dict:
    resp = model.invoke(
        f"Question: {state['question']}\n"
        "Does answering this well require web research? Reply YES or NO on the first line.\n"
        "If YES, give exactly 2 short search queries, one per line, after that."
    )
    lines = [l.strip("- ") for l in resp.content.strip().splitlines() if l.strip()]
    needs_research = lines[0].upper().startswith("YES") if lines else False
    queries = lines[1:3] if needs_research and len(lines) >= 3 else []
    return {"needs_research": needs_research, "plan": queries}


def retrieve_a(state: State) -> dict:
    query = state["plan"][0]
    result = search.invoke(query)
    return {"research_results": [f"Query: {query}\n{result}"]}


def retrieve_b(state: State) -> dict:
    query = state["plan"][1]
    result = search.invoke(query)
    return {"research_results": [f"Query: {query}\n{result}"]}


def summarize_node(state: State) -> dict:
    combined = "\n\n".join(state["research_results"])
    resp = model.invoke(f"Summarize these search results in 3-4 sentences:\n\n{combined}")
    return {"summary": resp.content}


def answer_node(state: State) -> dict:
    resp = model.invoke(
        f"Question: {state['question']}\n\nResearch summary:\n{state['summary']}\n\n"
        "Give a clear, direct final answer based on the research above."
    )
    return {"final_answer": resp.content}


def answer_direct_node(state: State) -> dict:
    resp = model.invoke(f"Answer this directly and briefly: {state['question']}")
    return {"final_answer": resp.content}


# --- 3. Conditional routing: decide the next node(s) based on state -----------------
def route_after_plan(state: State):
    if state["needs_research"]:
        return ["retrieve_a", "retrieve_b"]  # returning a LIST fans out to both in parallel
    return ["answer_direct"]


# --- 4. Build the graph: nodes + edges -----------------------------------------------
graph = StateGraph(State)
graph.add_node("plan", plan_node)
graph.add_node("retrieve_a", retrieve_a)
graph.add_node("retrieve_b", retrieve_b)
graph.add_node("summarize", summarize_node)
graph.add_node("answer", answer_node)
graph.add_node("answer_direct", answer_direct_node)

graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", route_after_plan)
graph.add_edge("retrieve_a", "summarize")
graph.add_edge("retrieve_b", "summarize")
graph.add_edge("summarize", "answer")
graph.add_edge("answer", END)
graph.add_edge("answer_direct", END)

app = graph.compile()


if __name__ == "__main__":
    question = "What is the current population of Japan and how has it changed recently?"
    result = app.invoke({"question": question})

    print("Needed research:", result["needs_research"])
    print("\nFinal answer:\n", result["final_answer"])