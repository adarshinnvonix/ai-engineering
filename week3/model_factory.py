"""
model_factory.py
-----------------
Same pattern as the LangChain-fundamentals module, extended with an "ollama" provider
that runs entirely on your own machine (no API key, no internet call at inference time).
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",   # pull first: `ollama pull llama3.2`
}


def get_model(provider: str = "groq", temperature: float = 0.3, model_name: str | None = None):
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

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # base_url defaults to http://localhost:11434 - override if Ollama runs elsewhere
        return ChatOllama(model=name, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}. Choose from groq, mistral, gemini, ollama.")


if __name__ == "__main__":
    for p in ["groq", "mistral", "gemini", "ollama"]:
        try:
            m = get_model(p)
            print(f"[OK] {p:8s} -> {type(m).__name__} (model={DEFAULT_MODELS[p]})")
        except Exception as e:
            print(f"[SKIP] {p:8s} -> {e}")