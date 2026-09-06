"""
chunking_strategies_comparison.py
-----------------------------------------
Implements THREE chunking strategies over the same HR policy document, then checks which
one retrieves the "right" chunk most often for a fixed set of benchmark questions.

Strategies:
  1. FIXED       - chop every N characters, no regard for sentence/paragraph boundaries.
                   Simplest, but can cut a sentence (and its meaning) in half.
  2. RECURSIVE   - try to split on paragraph breaks first, then sentences, then words -
                   only falls back to a harder cut if a piece is still too big.
  3. SENTENCE-BASED - always split on sentence boundaries, then group whole sentences
                   together until the chunk hits the size limit. Never cuts mid-sentence.

Evaluation method (kept simple on purpose):
  For each benchmark question, we check whether the EXPECTED KEYWORD phrase (e.g. "casual
  leave") appears in any of the top-3 retrieved chunks. This is a rough "hit rate", not a
  perfect measure, but good enough to compare strategies at a glance.
"""

import json
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from model_factory import get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
QUESTIONS_PATH = "data/benchmark_questions.json"
EMBEDDING_PROVIDER = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3


def load_raw_text():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    return loader.load()[0].page_content


def fixed_chunking(text):
    """Strategy 1: dumb fixed-size split, ignores sentence boundaries."""
    splitter = CharacterTextSplitter(
        separator="",  # "" forces pure character counting, no smart boundary seeking
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


def recursive_chunking(text):
    """Strategy 2: tries paragraph -> sentence -> word boundaries, in that order."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


def sentence_based_chunking(text):
    """Strategy 3: split into sentences first, then group whole sentences into chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) <= CHUNK_SIZE:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    return [Document(page_content=c) for c in chunks]


def evaluate_strategy(name, chunks, benchmark, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)

    hits = 0
    for item in benchmark:
        results = vectorstore.similarity_search(item["question"], k=TOP_K)
        combined_text = " ".join(doc.page_content.lower() for doc in results)
        if item["expected_keyword"].lower() in combined_text:
            hits += 1

    hit_rate = hits / len(benchmark)
    avg_chunk_len = sum(len(c.page_content) for c in chunks) / len(chunks)

    print(f"\n[{name}]")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {avg_chunk_len:.0f} chars")
    print(f"  Hit rate on benchmark: {hits}/{len(benchmark)} = {hit_rate:.0%}")

    return {"name": name, "num_chunks": len(chunks), "avg_chunk_len": avg_chunk_len, "hit_rate": hit_rate}


def main():
    text = load_raw_text()
    with open(QUESTIONS_PATH) as f:
        benchmark = json.load(f)

    embeddings = get_embeddings(EMBEDDING_PROVIDER)

    strategies = {
        "Fixed-size": fixed_chunking(text),
        "Recursive": recursive_chunking(text),
        "Sentence-based": sentence_based_chunking(text),
    }

    summary = []
    for name, chunks in strategies.items():
        summary.append(evaluate_strategy(name, chunks, benchmark, embeddings))

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print(f"{'Strategy':16s} {'Chunks':>7s} {'Avg Len':>9s} {'Hit Rate':>9s}")
    for row in summary:
        print(f"{row['name']:16s} {row['num_chunks']:7d} {row['avg_chunk_len']:9.0f} {row['hit_rate']:9.0%}")


if __name__ == "__main__":
    main()