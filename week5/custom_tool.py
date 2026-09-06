"""
custom_tool.py
---------------------
LEARNING GOAL
  Build custom tools with the @tool decorator. LangChain reads the function's type
  hints and docstring to describe the tool to the LLM - so a clear docstring matters,
  it's literally the instructions the model sees.

Two simple, real (no API key needed) tools:
  - get_weather(city)       -> uses wttr.in, a free plain-text weather service
  - get_stock_price(ticker) -> uses yfinance, free, no API key
"""

import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from model_factory import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city name."""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return response.text.strip()
    except Exception as e:
        return f"Could not fetch weather for {city}: {e}"


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol, e.g. AAPL or TSLA."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        price = stock.fast_info["last_price"]
        return f"{ticker.upper()} is currently trading at ${price:.2f}"
    except Exception as e:
        return f"Could not fetch stock price for {ticker}: {e}"


TOOLS = [get_weather, get_stock_price]
model = get_model("groq")
model_with_tools = model.bind_tools(TOOLS)
tools_by_name = {t.name: t for t in TOOLS}


def ask(question: str):
    messages = [HumanMessage(content=question)]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print(f"\nQuestion: {question}")
    print("Tool calls requested:", ai_msg.tool_calls)

    for call in ai_msg.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = model_with_tools.invoke(messages)
    print("Final answer:", final.content)


if __name__ == "__main__":
    ask("What's the weather like in Ahmedabad right now?")
    ask("What is the current stock price of AAPL?")