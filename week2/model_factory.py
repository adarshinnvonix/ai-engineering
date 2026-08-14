"""
model_factory.py
-----------------
LEARNING GOALS
  - See that ChatGroq, ChatMistralAI, and ChatGoogleGenerativeAI all implement the
    SAME BaseChatModel interface. Once instantiated, they're interchangeable — any
    code written against one works against the others unchanged.
  - Understand that provider differences (model names, params) live only at
    construction time, never in how you call them downstream.

TASKS
  1. Run get_model("groq") / ("mistral") / ("gemini") and print(type(model).mro())
     to see the class hierarchy — note they all inherit BaseChatModel -> Runnable.
  2. Try changing temperature and max_tokens per provider and observe output variance.
  3. Add a 4th provider (e.g. ChatOpenAI) yourself to prove the pattern generalizes.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
}


def get_model(provider: str = "groq", temperature: float = 0.3, model_name: str | None = None):
    """Return a ready-to-use LangChain chat model for the given provider.

    provider: "groq" | "mistral" | "gemini"
    """
    provider = provider.lower()
    name = model_name or DEFAULT_MODELS.get(provider)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=name, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=name, temperature=temperature, api_key=os.getenv("MISTRAL_API_KEY"))

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, api_key=os.getenv("GOOGLE_API_KEY"))

    raise ValueError(f"Unknown provider: {provider}. Choose from groq, mistral, gemini.")


if __name__ == "__main__":
    for p in ["groq", "mistral", "gemini"]:
        try:
            m = get_model(p)
            print(f"[OK] {p:8s} -> {type(m).__name__} (model={DEFAULT_MODELS[p]})")
        except Exception as e:
            print(f"[SKIP] {p:8s} -> {e}")