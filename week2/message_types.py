"""
message_types.py
--------------------
LEARNING GOALS
  - Understand LangChain's message abstraction: every chat model call is a LIST of
    typed messages, not a raw string.
      SystemMessage -> sets model behavior/persona, sent once at the start
      HumanMessage  -> what the user says
      AIMessage     -> what the model said (used for conversation history / few-shot)
      ToolMessage   -> the RESULT of a tool call, fed back to the model
  - See that a model's .invoke() takes List[BaseMessage] and returns an AIMessage.
  - Understand AIMessage.tool_calls — how a model requests a tool be run.

TASKS
  1. Run this file and inspect the raw AIMessage object (not just .content) —
     check .response_metadata, .usage_metadata, .id.
  2. Build a 3-turn conversation manually using Human/AI message pairs and confirm
     the model "remembers" earlier turns because they're in the list you send.
  3. Bonus: bind a fake tool schema to the model and see it emit AIMessage.tool_calls
     instead of plain text, then construct a ToolMessage with a fake result and send
     it back to complete the loop.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from model_factory import get_model

PROVIDER = "groq"  # change to "mistral" or "gemini" to compare


def basic_message_list():
    model = get_model(PROVIDER)
    messages = [
        SystemMessage(content="You are a terse, no-fluff coding assistant. Max 2 sentences."),
        HumanMessage(content="What is a Python list comprehension?"),
    ]
    response = model.invoke(messages)
    print("Type:", type(response).__name__)
    print("Content:", response.content)
    print("Usage metadata:", response.usage_metadata)
    return response


def multi_turn_conversation():
    model = get_model(PROVIDER)
    messages = [
        SystemMessage(content="You are a helpful math tutor."),
        HumanMessage(content="What is 12 * 8?"),
        AIMessage(content="12 * 8 = 96."),
        HumanMessage(content="Now divide that by 4."),
    ]
    response = model.invoke(messages)
    print("\nMulti-turn response:", response.content)


def tool_message_demo():
    """Shows how ToolMessage completes a tool-calling round trip."""
    model = get_model(PROVIDER)

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"{city}: 31C, sunny"

    model_with_tools = model.bind_tools([get_weather])
    messages = [HumanMessage(content="What's the weather in Ahmedabad?")]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print("\nTool calls requested:", ai_msg.tool_calls)

    for call in ai_msg.tool_calls:
        result = get_weather(**call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = model_with_tools.invoke(messages)
    print("Final answer after tool round trip:", final.content)


if __name__ == "__main__":
    basic_message_list()
    multi_turn_conversation()
    tool_message_demo()