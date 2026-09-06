"""
hybrid_search.py
-----------------------
Adds KEYWORD search (BM25 - a classic "count matching words, weighted by rarity"
algorithm) alongside vector search, then combines both rankings into one "hybrid" result.

Why bother, if vector search already understands meaning?
  Vector search is great at "similar meaning, different words" but can sometimes miss
  EXACT terms - policy names, specific numbers, acronyms (e.g. "ICC", "POSH Act") that a
  plain keyword match would catch instantly. Hybrid search hedges between the two.

Combination method used here: Reciprocal Rank Fusion (RRF) - a simple, well-known way to
merge two ranked lists without needing to compare their raw scores (which use different
scales and aren't directly comparable).
"""

import json
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from model_factory import get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
QUESTIONS_PATH = "data/benchmark_questions.json"
EMBEDDING_PROVIDER = "mistral"
TOP_K = 3
RRF_K = 60  # standard smoothing constant used in reciprocal rank fusion


def build_chunks():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    return splitter.split_documents(documents)


def build_bm25_index(chunks):
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    return BM25Okapi(tokenized)


def bm25_search(bm25, chunks, query, k=TOP_K):
    scores = bm25.get_scores(query.lower().split())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return ranked_indices  # just the chunk indices, ranked best-first


def vector_search(vectorstore, chunks, query, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    # map each returned chunk back to its index in the original chunk list
    indices = []
    for r in results:
        for i, c in enumerate(chunks):
            if c.page_content == r.page_content:
                indices.append(i)
                break
    return indices


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """rankings: list of ranked-index-lists (e.g. [bm25_indices, vector_indices])."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def hit(indices, chunks, expected_keyword, k=TOP_K):
    combined_text = " ".join(chunks[i].page_content.lower() for i in indices[:k])
    return expected_keyword.lower() in combined_text


def main():
    with open(QUESTIONS_PATH) as f:
        benchmark = json.load(f)

    chunks = build_chunks()
    bm25 = build_bm25_index(chunks)
    embeddings = get_embeddings(EMBEDDING_PROVIDER)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vector_only_hits = 0
    hybrid_hits = 0

    print(f"Indexed {len(chunks)} chunks with both BM25 and FAISS.\n")

    for item in benchmark:
        question, expected = item["question"], item["expected_keyword"]

        bm25_idx = bm25_search(bm25, chunks, question)
        vector_idx = vector_search(vectorstore, chunks, question)
        hybrid_idx = reciprocal_rank_fusion([bm25_idx, vector_idx])

        v_hit = hit(vector_idx, chunks, expected)
        h_hit = hit(hybrid_idx, chunks, expected)

        vector_only_hits += v_hit
        hybrid_hits += h_hit

        print(f"Q: {question}")
        print(f"  vector-only hit: {v_hit}   hybrid hit: {h_hit}")

    total = len(benchmark)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Vector-only hit rate: {vector_only_hits}/{total} = {vector_only_hits/total:.0%}")
    print(f"  Hybrid (BM25+vector) hit rate: {hybrid_hits}/{total} = {hybrid_hits/total:.0%}")
    print("\n  Note: on a small, well-written document like this HR policy, the gap may be")
    print("  small - hybrid search tends to matter more on messier or jargon/code-heavy data.")


if __name__ == "__main__":
    main()