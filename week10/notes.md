# Simple Guarded LangGraph Application

This contains a simple, reference implementation combining concepts from **Weeks 7, 8**:
1. **LangGraph Workflows & State Management (Week 7)**: Modular nodes and state passing.
2. **Persistence & Human-in-the-Loop (Week 8)**: Uses `MemorySaver` to checkpoint runs, pause before critical actions, and resume reliably using `.update_state()`.
3. **Guardrails & Tracing**: Input filtering (blocks prompt injection/PII keywords) and Output validation (Pydantic schema checking). Integrates easily with LangSmith for token, latency, and cost tracing.

## Setup

```bash
pip install -r week10/modules.txt
```
