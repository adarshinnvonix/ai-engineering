"""
vector_search_faiss_chroma.py
------------------------------------
LEARNING GOALS
  - An embedding model turns text into a fixed-length vector of numbers such
    that SEMANTICALLY similar text ends up as NEARBY vectors in that space —
    this is the entire trick behind "search by meaning" instead of by keyword.
  - A vector store (FAISS, Chroma, etc.) indexes many such vectors so you can
    ask "which stored vectors are closest to THIS query vector?" efficiently,
    instead of comparing against every document one by one.
  - FAISS (Facebook AI Similarity Search) is a lightweight, in-memory/local
    library — great for prototyping, no server needed.
  - Chroma is a slightly heavier "vector database" with a persistence layer and
    metadata filtering built in — closer to what you'd actually run in production.
  - This file does NOT build a RAG chain — it stops at "given a query, which raw
    chunks come back" so you can inspect similarity search in isolation before
    RAG_pipeline_explained.md wires it into a full pipeline.

PREREQUISITES
  First run downloads a small embedding model (~80MB) from Hugging Face, so you
  need internet access once. After that it's cached locally and fully offline.

TASKS
  1. Run this file and read the similarity scores (lower distance = more similar
     for FAISS's default L2 metric) — do the top matches make intuitive sense to
     you as a human reading the query?
  2. Add a query that's a NEAR-miss (uses different words but a related meaning,
     e.g. "battery doesn't last long" vs a doc about "poor battery life") and
     confirm semantic search still finds it — a plain keyword search wouldn't.
  3. Add a query that's a trap (shares KEYWORDS with a doc but different meaning,
     e.g. "I want to return this" vs a doc about "the return of a classic
     design") and see whether the embedding model gets it right or wrong.
  4. Compare FAISS vs Chroma results for the SAME query/corpus/embedding model —
     they should return the same nearest neighbors (they're just different
     index implementations over the same math), confirming the choice of vector
     store is an infra decision, not a quality decision.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, good for experimentation

CORPUS = [
    "The laptop's battery drains fast, barely lasting 3 hours on a full charge.",
    "Customer support resolved my billing issue within minutes over chat.",
    "The blender is very loud but chops vegetables extremely evenly.",
    "Shipping took two weeks longer than the estimated delivery date.",
    "This vintage-style chair is a return to classic mid-century design.",
    "I want to return this jacket, the size runs way too small.",
    "The phone's camera takes stunning photos even in low light.",
    "Assembly instructions were confusing and missing a step for the legs.",
    "Software update fixed the crashing bug I reported last month.",
    "The headphones have excellent noise cancellation for the price.",
]


def build_documents():
    return [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(CORPUS)]


def faiss_demo(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = build_documents()

    vectorstore = FAISS.from_documents(docs, embeddings)
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n[FAISS] Query: \"{query}\"")
    for doc, score in results:
        print(f"  score={score:.4f}  ->  {doc.page_content}")
    return results


def chroma_demo(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = build_documents()

    # in-memory, ephemeral Chroma collection (no persist_directory = nothing written to disk)
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="demo_collection")
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n[Chroma] Query: \"{query}\"")
    for doc, score in results:
        print(f"  score={score:.4f}  ->  {doc.page_content}")
    return results


def keyword_vs_semantic_illustration():
    """Shows why embeddings beat naive keyword matching for meaning-based search."""
    query = "the device runs out of power quickly"  # shares almost NO keywords with the battery doc
    print(f"\n--- Keyword vs Semantic illustration ---")
    print(f"Query: \"{query}\"")
    print("Naive keyword overlap with corpus would likely find NOTHING (no shared words "
          "with 'battery drains fast, barely lasting 3 hours').")
    faiss_demo(query, k=2)


if __name__ == "__main__":
    faiss_demo("battery doesn't last long")
    chroma_demo("battery doesn't last long")

    print("\n--- Trap query: shares keywords but different meaning ---")
    faiss_demo("I want to return this")

    keyword_vs_semantic_illustration()