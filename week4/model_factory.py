"""
model_factory.py
-----------------
One place to create chat models AND embedding models, so every other file just calls
get_model(...) / get_embeddings(...) instead of repeating provider setup code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",   # local - run `ollama pull llama3.2` first
}


def get_model(provider: str = "groq", temperature: float = 0.2, model_name: str | None = None):
    """Return a chat model. provider: groq | mistral | gemini | ollama"""
    provider = provider.lower()
    name = model_name or DEFAULT_CHAT_MODELS.get(provider)

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
        return ChatOllama(model=name, temperature=temperature)

    raise ValueError(f"Unknown chat provider: {provider}")


def get_embeddings(provider: str = "mistral"):
    """Return an embedding model. provider: mistral | huggingface (local, free fallback)"""
    provider = provider.lower()

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))

    if provider == "huggingface":
        # No API key needed, runs locally - good fallback if you don't have a Mistral key.
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    raise ValueError(f"Unknown embeddings provider: {provider}")


if __name__ == "__main__":
    for p in ["groq", "mistral", "gemini", "ollama"]:
        try:
            get_model(p)
            print(f"[OK]   chat model  -> {p}")
        except Exception as e:
            print(f"[SKIP] chat model  -> {p}: {e}")

    for p in ["mistral", "huggingface"]:
        try:
            get_embeddings(p)
            print(f"[OK]   embeddings  -> {p}")
        except Exception as e:
            print(f"[SKIP] embeddings  -> {p}: {e}")