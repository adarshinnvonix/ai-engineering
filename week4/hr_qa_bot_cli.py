"""
hr_qa_bot_cli.py
----------------------
The actual deliverable: a command-line HR Policy Q&A bot for Innvonix.

Usage:
    python hr_qa_bot_cli.py               # uses cloud model (Groq) by default
    python hr_qa_bot_cli.py --provider ollama   # uses local Ollama model instead

For every question, it PRINTS the retrieved chunks and their similarity scores before
showing the final answer, so you can see exactly what context the LLM was given - this
is the "explainability" that plain chatbots don't give you.
"""

import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from model_factory import get_model, get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
EMBEDDING_PROVIDER = "mistral"

SYSTEM_PROMPT = (
    "You are an HR policy assistant for Innvonix. Answer the employee's question using "
    "ONLY the context below, in a clear and friendly tone. If the answer isn't in the "
    "context, say you don't know and suggest they contact HR directly - do not guess or "
    "make up policy details.\n\nContext:\n{context}"
)


def build_retriever(k=3):
    """Load -> split -> embed -> index. Done once when the bot starts."""
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings(EMBEDDING_PROVIDER)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def answer_question(vectorstore, model, question, k=3):
    # Retrieve + show the evidence
    results = vectorstore.similarity_search_with_score(question, k=k)

    print("\n  Retrieved context (chunk + similarity score):")
    for i, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content.replace("\n", " ")[:120]
        print(f"    [{i}] score={score:.4f}  \"{preview}...\"")

    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    # Generate the grounded answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Innvonix HR Policy Q&A Bot")
    parser.add_argument("--provider", default="groq",
                         choices=["groq", "mistral", "gemini", "ollama"],
                         help="Which LLM to use for answering (default: groq)")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    print(f"Starting Innvonix HR Bot (LLM provider: {args.provider})...")
    vectorstore = build_retriever()
    model = get_model(args.provider)
    print("Ready! Ask a question about HR policy, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = answer_question(vectorstore, model, question, k=args.k)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()