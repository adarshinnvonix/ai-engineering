"""
reranking.py
-------------------
Adds a RERANKER on top of vector search.

The idea: vector search is fast but approximate - it compares pre-computed chunk vectors
to a query vector using simple math (cosine/L2 distance). A reranker is slower but much
more accurate: it reads the QUERY and EACH CANDIDATE CHUNK TOGETHER (a "cross-encoder")
and directly scores how relevant that specific pairing is.

Typical pattern: retrieve a wider net first (e.g. top 10 from vector search), then rerank
those 10 down to the best 3 with the cross-encoder. This gets you the accuracy of a
cross-encoder without the cost of running it against every chunk in the whole document.

Trade-off measured here: does reranking improve hit rate, and how much extra time does it
add per question?
"""

import json
import time
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from model_factory import get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
QUESTIONS_PATH = "data/benchmark_questions.json"
EMBEDDING_PROVIDER = "mistral"
RETRIEVE_N = 8   # wide net from vector search
FINAL_K = 3      # how many we keep after reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, fast, well-known reranker


def build_vectorstore():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings(EMBEDDING_PROVIDER)
    return FAISS.from_documents(chunks, embeddings)


def hit(docs, expected_keyword, k):
    combined_text = " ".join(d.page_content.lower() for d in docs[:k])
    return expected_keyword.lower() in combined_text


def main():
    with open(QUESTIONS_PATH) as f:
        benchmark = json.load(f)

    vectorstore = build_vectorstore()
    reranker = CrossEncoder(RERANKER_MODEL)

    before_hits = 0
    after_hits = 0
    total_rerank_time = 0.0

    for item in benchmark:
        question, expected = item["question"], item["expected_keyword"]

        # Step 1: wide net from vector search
        candidates = vectorstore.similarity_search(question, k=RETRIEVE_N)

        # "Before" = just take the top FINAL_K from vector search order, no reranking
        before_hits += hit(candidates, expected, FINAL_K)

        # Step 2: rerank all candidates against the actual question
        pairs = [[question, doc.page_content] for doc in candidates]
        start = time.time()
        scores = reranker.predict(pairs)
        total_rerank_time += time.time() - start

        reranked = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        after_hits += hit(reranked, expected, FINAL_K)

        print(f"Q: {question}")
        print(f"  top chunk BEFORE reranking: \"{candidates[0].page_content[:80]}...\"")
        print(f"  top chunk AFTER  reranking: \"{reranked[0].page_content[:80]}...\"")

    total = len(benchmark)
    avg_rerank_time = total_rerank_time / total

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Hit rate BEFORE reranking (vector order only): {before_hits}/{total} = {before_hits/total:.0%}")
    print(f"  Hit rate AFTER reranking:                      {after_hits}/{total} = {after_hits/total:.0%}")
    print(f"  Avg reranking time added per question: {avg_rerank_time*1000:.0f} ms "
          f"(for {RETRIEVE_N} candidates)")


if __name__ == "__main__":
    main()