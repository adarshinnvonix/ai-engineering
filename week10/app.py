import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Verify loaded tracing configuration
print("--- [0] INITIALIZING ENVIRONMENT ---")
print(f"Tracing Enabled: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
print(f"Project Name:    {os.getenv('LANGCHAIN_PROJECT', 'Default-Project')}")


# Guardrails (Input & output validation)
class OutputSchema(BaseModel):
    answer: str = Field(description="The finalized response to the user")
    is_safe: bool = Field(description="Whether the response passed safety checks")


def validate_input(user_query: str) -> str:
    """Simple input guardrail for PII and prompt injection."""
    query_lower = user_query.lower()
    if "ignore previous instructions" in query_lower or "hack" in query_lower:
        return "[BLOCKED] Potential prompt injection detected."
    if "password" in query_lower or "ssn" in query_lower:
        return "[BLOCKED] Sensitive PII request blocked."
    return user_query


# Langgraph state and workflow
class State(TypedDict):
    query: str
    cleaned_query: str
    draft_response: str
    approved: bool


# Node 1: Input Guardrail Node
def input_guardrail_node(state: State):
    print("\n--- [1] RUNNING INPUT GUARDRAIL ---")
    original = state["query"]
    safe_query = validate_input(original)
    if "[BLOCKED]" in safe_query:
        print(f"Blocked input: {original}")
    return {"cleaned_query": safe_query}


# Node 2: Agent Processing Node
def agent_node(state: State):
    print("\n--- [2] RUNNING AGENT WORKFLOW ---")
    query = state["cleaned_query"]
    if "[BLOCKED]" in query:
        draft = "I cannot fulfill this request due to safety policies."
    else:
        draft = f"Processed and generated answer for: '{query}'."
    return {"draft_response": draft}


# Node 3: Human-in-the-Loop Interruption Check
def human_approval_node(state: State):
    pass


# Conditional Edge: Route based on approval status
def check_approval(state: State):
    if state.get("approved", False):
        return "finalize"
    return "interrupt"


# Node 4: Output Validation Node
def finalize_node(state: State):
    print("\n--- [3] FINALIZING & OUTPUT VALIDATION ---")
    raw_answer = state["draft_response"]
    
    # Validate output schema via Pydantic
    validated = OutputSchema(answer=raw_answer, is_safe=True)
    return {"draft_response": validated.answer}


# Build the graph with checkpoints
workflow = StateGraph(State)

workflow.add_node("guardrail", input_guardrail_node)
workflow.add_node("agent", agent_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("finalize", finalize_node)

workflow.add_edge(START, "guardrail")
workflow.add_edge("guardrail", "agent")
workflow.add_edge("agent", "human_approval")

workflow.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "finalize": "finalize",
        "interrupt": END  # Pause execution waiting for human input
    }
)
workflow.add_edge("finalize", END)

# Memory checkpointer for state persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "session-1"}}

    print("\n=== FIRST RUN (Will pause at Human Approval) ===")
    initial_input = {"query": "Tell me about LangGraph checkpointers."}
    
    for event in app.stream(initial_input, thread_config):
        print(event)

    # Check paused state
    current_state = app.get_state(thread_config)
    print(f"\n[Paused State] Draft response to review: {current_state.values.get('draft_response')}")

    print("\n=== RESUMING RUN (After Human Approval) ===")
    # Approve and resume workflow
    app.update_state(thread_config, {"approved": True})
    
    for event in app.stream(None, thread_config):
        print(event)