"""
model_factory.py - one place to create the chat model, so every example file stays short.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_model(provider: str = "groq", temperature: float = 0.2):
    if provider == "groq":
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return ChatGroq(model=model_name, temperature=temperature,
                         api_key=os.getenv("GROQ_API_KEY"))
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", temperature=temperature,
                              api_key=os.getenv("MISTRAL_API_KEY"))
    raise ValueError(f"Unknown provider: {provider}")