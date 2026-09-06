"""
prebuilt_tool.py
-----------------------
LEARNING GOAL
  See how a ready-made LangChain tool gets bound to an LLM, and how "tool calling"
  actually works: the LLM doesn't run the tool itself - it just tells you WHICH tool to
  run and with WHAT input. Your code runs it and sends the result back.

It uses LangChain's built-in DuckDuckGo search tool - no API key needed.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage
from model_factory import get_model

# 1. A pre-built tool - LangChain ships many of these (search, Wikipedia, arXiv, etc.)
search_tool = DuckDuckGoSearchRun()

# 2. Bind it to the model. This just tells the model "this tool exists and here's its
#    name/description/expected input" - the model can now CHOOSE to call it.
model = get_model("groq")
model_with_tools = model.bind_tools([search_tool])


def main():
    question = "Who won the most recent Formula 1 world championship?"
    messages = [HumanMessage(content=question)]

    # 3. Ask the model. If it decides it needs the tool, it replies with tool_calls
    #    instead of a direct answer.
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print("Did the model request a tool call?", bool(ai_msg.tool_calls))
    print("Tool calls requested:", ai_msg.tool_calls)

    # 4. Actually run the tool ourselves, and feed the result back as a ToolMessage.
    for call in ai_msg.tool_calls:
        result = search_tool.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # 5. Ask again - now the model has the search result and can give a real answer.
    final = model_with_tools.invoke(messages)
    print("\nFinal answer:", final.content)


if __name__ == "__main__":
    main()