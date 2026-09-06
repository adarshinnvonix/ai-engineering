# Week 4 - RAG Fundamentals + Retrieval Quality

A basic, from-scratch RAG (Retrieval-Augmented Generation) pipeline built as an HR Policy
Q&A bot, plus hands-on experiments to see how document prep and retrieval strategy affect
answer quality.

## Setup

```bash
pip install -r week4/modules.txt
```

For the Ollama half, install Ollama and pull a small model:
```bash
ollama pull llama3.2
```

## Files, in the order to run them

| # | File | What it does |
|---|------|---------------|
| 1 | `basic_rag_pipeline.py` | Runs every RAG step one at a time with printed output |
| 2 | `hr_qa_bot_cli.py` | Interactive CLI bot, shows retrieved chunks + scores for every answer |
| 3 | `cloud_vs_local_comparison.py` | Runs the same benchmark questions through Groq (cloud) and Ollama (local), saves results to `comparison_results.json` |
| 4 | `chunking_strategies_comparison.py` | Compares fixed vs recursive vs sentence-based chunking on retrieval hit-rate |
| 5 | `hybrid_search.py` | Adds keyword search (BM25) alongside vector search, compares hit-rate |
| 6 | `reranking.py` | Adds a cross-encoder reranker on top of vector search, compares hit-rate + latency cost |

### RAG pipeline and architecture
RAG means: before asking an AI a question, first go fetch the most relevant pieces of a real
document, and give those to the AI. It splits
into two phases: **indexing** (done once - prepare and store the documents) and **query**
(done every time someone asks something - search and answer).

### Document Loaders
The tool that reads a file (PDF, Word doc, plain text, webpage) and pulls out usable text.
Here we use `TextLoader` since our HR policy is a plain markdown file — for PDFs or Word
docs, LangChain has different loader classes, but the rest of the pipeline stays identical.

### Text Splitters, Chunking, and Chunking Strategies
An LLM and an embedding model both have size limits, and retrieval works better on small,
focused pieces of text rather than a whole document at once.
- **Fixed-size:** cut every N characters, no matter what. Simplest but can slice a sentence
  in half, losing meaning right at the cut point.
- **Recursive:** try to cut on paragraph breaks first, then sentences, then words, only
  forces a harder cut if a piece is still too big. Usually the best default.
- **Sentence-based:** always keep whole sentences together, grouping them until a chunk hits
  the size limit, never cuts mid-thought.
- **Semantic:** groups sentences together based on whether they're *about the same idea* , rather than just size or punctuation. More advanced, more compute-expensive.

### Chunk Overlap
Chunks share a small amount of text at their boundary (e.g. the last 80 characters of chunk 1
are also the first 80 of chunk 2). This prevents a fact that happens to sit right at a chunk
boundary from being lost or split awkwardly.

### Embedding Models
A model that converts text into a list of numbers (a vector) such that texts with similar
MEANING end up as nearby vectors, even if they don't share the same words. This is what makes
"search by meaning" possible.

### Vector Stores and FAISS
A vector store indexes many embedded vectors so one can quickly ask "which stored vectors are
closest to this one?" without comparing against every single chunk one by one. FAISS (from
Meta) is a fast, lightweight, local option, no server needed, good for prototyping and
small-to-medium datasets.

### Retrievers
A retriever is just a wrapper around the vector store that exposes one simple job: "given a
query, give me the top-k most relevant chunks." It hides the embedding + search steps behind
one clean interface, so the rest of your code doesn't need to know HOW the search works.

### Similarity Search
The actual math operation: comparing a query vector against every stored vector and ranking
by closeness (distance). Lower distance = more similar (for the default metric FAISS uses
here). This is what a retriever does under the hood.

### Context Retrieval
Once one have the top-k chunks, it is glued together into one "context" block that
gets handed to the LLM alongside the original question, that's the "augmented" part of
Retrieval-**Augmented** Generation.

### RAG with LangChain
LangChain gives one the pieces (loaders, splitters, embeddings, vector stores, retrievers,
prompt templates, chat models) and the `|` operator to snap them together into one pipeline —
exactly what `basic_rag_pipeline.py` and `hr_qa_bot_cli.py` do, step by step.

### Reranking
Vector search is fast but approximate. A reranker reads the query and each candidate chunk TOGETHER and scores relevance much more precisely, but it's slower, so the usual pattern is: retrieve a wider net with vector search (e.g. top 10), then rerank down to the best few (e.g. top 3). `reranking.py` measures both the accuracy gain
and the latency cost of doing this.


## Suggested order to actually run things

1. `python basic_rag_pipeline.py` — watch each RAG step print its output.
2. `python hr_qa_bot_cli.py` — ask it real HR questions, read the retrieved chunks.
3. `python hr_qa_bot_cli.py --provider ollama` — same bot, local model instead.
4. `python cloud_vs_local_comparison.py` — automated head-to-head.
5. `python chunking_strategies_comparison.py`
6. `python hybrid_search.py`
7. `python reranking.py`