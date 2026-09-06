"""
react_agent.py
---------------------
LEARNING GOAL
  See the ReAct pattern (Reason + Act) in action: the agent alternates between THINKING
  ("what do I need to do next?"), ACTING (calling a tool), and OBSERVING (reading the
  result) - repeating this loop until it has enough info to give a final answer.

  With verbose=True below, you'll literally see this printed out as:
    Thought: ...
    Action: ...
    Action Input: ...
    Observation: ...
  repeating until it reaches "Final Answer:". This is the whole point of a ReAct agent -
  unlike a single tool-call round trip (previous two files), it can chain MULTIPLE tool
  calls together, deciding each next step based on what it just observed.

It give it 3 tools: web search + the two custom tools from the previous file.
"""

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from model_factory import get_model
from custom_tool import get_weather, get_stock_price

TOOLS = [DuckDuckGoSearchRun(), get_weather, get_stock_price]
model = get_model("groq")
agent = create_agent(
  model=model,
  tools=TOOLS,
  system_prompt=(
    "Answer the user's question using the available tools. "
    "Use tools when current information is needed, then provide a concise answer."
  ),
)


if __name__ == "__main__":
    question = "What's the weather in Ahmedabad, and what is the current stock price of TSLA?"
  result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("\n=== FINAL ANSWER ===")
  print(result["messages"][-1].content)