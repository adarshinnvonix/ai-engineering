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

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from model_factory import get_model
from importlib import import_module

# Reuse the custom tools defined in 02_custom_tool.py instead of redefining them
custom_tools = import_module("02_custom_tool")

TOOLS = [
    DuckDuckGoSearchRun(),
    custom_tools.get_weather,
    custom_tools.get_stock_price,
]

# The classic ReAct prompt format (Thought / Action / Action Input / Observation loop).
REACT_PROMPT = PromptTemplate.from_template("""Answer the following question as best you can.
You have access to the following tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: think about what to do next
Action: the tool to use, must be one of [{tool_names}]
Action Input: the input to that tool
Observation: the result of the tool
... (this Thought/Action/Action Input/Observation can repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

model = get_model("groq")
agent = create_react_agent(llm=model, tools=TOOLS, prompt=REACT_PROMPT)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, handle_parsing_errors=True)


if __name__ == "__main__":
    question = "What's the weather in Ahmedabad, and what is the current stock price of TSLA?"
    result = executor.invoke({"input": question})
    print("\n=== FINAL ANSWER ===")
    print(result["output"])