"""
cloud_vs_local_comparison.py
------------------------------------
Runs the SAME benchmark questions, with the SAME retrieved context, through:
  - a cloud model (Groq)
  - a local model (Ollama)

Keeping the retrieved context identical for both means any difference in the answer is
caused by the LLM itself, not by different search results - a fair comparison.

Saves everything to comparison_results.json so you can review it and write up findings
in comparison_results.md.

PREREQUISITE for the local half: Ollama installed and running, with a model pulled:
    ollama pull llama3.2
"""

import json
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from model_factory import get_model, get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
QUESTIONS_PATH = "data/benchmark_questions.json"
EMBEDDING_PROVIDER = "mistral"

SYSTEM_PROMPT = (
    "You are an HR policy assistant for Innvonix. Answer using ONLY the context below. "
    "If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"
)


def build_vectorstore():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings(EMBEDDING_PROVIDER)
    return FAISS.from_documents(chunks, embeddings)


def ask(model, context, question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | model
    start = time.time()
    try:
        response = chain.invoke({"context": context, "question": question})
        elapsed = time.time() - start
        return response.content, round(elapsed, 2), None
    except Exception as e:
        return None, None, str(e)


def main():
    with open(QUESTIONS_PATH) as f:
        benchmark = json.load(f)

    vectorstore = build_vectorstore()
    cloud_model = get_model("groq")
    local_model = get_model("ollama")

    results = []

    for item in benchmark:
        question = item["question"]
        print(f"\nQuestion: {question}")

        # Same retrieval for both, so we're only comparing the LLMs
        retrieved = vectorstore.similarity_search_with_score(question, k=3)
        context = "\n\n---\n\n".join(doc.page_content for doc, _ in retrieved)
        retrieved_preview = [
            {"score": round(float(score), 4), "chunk": doc.page_content[:150]}
            for doc, score in retrieved
        ]

        cloud_answer, cloud_time, cloud_error = ask(cloud_model, context, question)
        local_answer, local_time, local_error = ask(local_model, context, question)

        print(f"  Cloud (groq)  [{cloud_time if cloud_time else 'N/A'}s]: "
              f"{cloud_answer if cloud_answer else 'ERROR - ' + str(cloud_error)}")
        print(f"  Local (ollama)[{local_time if local_time else 'N/A'}s]: "
              f"{local_answer if local_answer else 'ERROR - ' + str(local_error)}")

        results.append({
            "question": question,
            "retrieved_chunks": retrieved_preview,
            "cloud": {"answer": cloud_answer, "time_sec": cloud_time, "error": cloud_error},
            "local": {"answer": local_answer, "time_sec": local_time, "error": local_error},
        })

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Simple average latency summary
    cloud_times = [r["cloud"]["time_sec"] for r in results if r["cloud"]["time_sec"]]
    local_times = [r["local"]["time_sec"] for r in results if r["local"]["time_sec"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    if cloud_times:
        print(f"  Cloud avg latency: {sum(cloud_times)/len(cloud_times):.2f}s over {len(cloud_times)} questions")
    if local_times:
        print(f"  Local avg latency: {sum(local_times)/len(local_times):.2f}s over {len(local_times)} questions")
    print("  Full results saved to comparison_results.json")
    print("  Now manually review the answers and fill in comparison_results.md with your")
    print("  observations on quality, relevance, and any hallucinations you spot.")


if __name__ == "__main__":
    main()