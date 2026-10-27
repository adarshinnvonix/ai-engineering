import os
from dotenv import load_dotenv

load_dotenv()


def get_chat_model():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))


def get_embeddings():
    # Same embeddings as Week 4. Set EMBEDDINGS_PROVIDER=huggingface in .env to run without a Mistral key.
    if os.getenv("EMBEDDINGS_PROVIDER", "mistral") == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    from langchain_mistralai import MistralAIEmbeddings
    return MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))