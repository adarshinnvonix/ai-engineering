"""
basic_rag_pipeline.py
---------------------------
A step-by-step, heavily-commented walkthrough of every RAG building block, in order.
Run this top to bottom once to SEE what each stage produces before touching the CLI bot
or any of the advanced retrieval files.

Steps covered (matches the topic list):
  1. Document Loader   - read the raw HR policy file
  2. Text Splitter     - break it into chunks (with overlap)
  3. Embedding Model   - turn each chunk into a vector (Mistral embeddings)
  4. Vector Store/FAISS- index those vectors for fast search
  5. Retriever         - wrap the vector store as a "give me relevant chunks" object
  6. Similarity Search - actually run a query and see scores
  7. Context Retrieval - assemble retrieved chunks into a context block
  8. RAG generation    - feed context + question to an LLM for the final answer
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from model_factory import get_model, get_embeddings

DOC_PATH = "data/innvonix_hr_policy.md"
EMBEDDING_PROVIDER = "mistral"   # switch to "huggingface" if you don't have a Mistral key
CHAT_PROVIDER = "groq"


def step1_load_document():
    print("\nSTEP 1: Document Loader")
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"  Loaded {len(documents)} document(s), total {len(documents[0].page_content)} characters.")
    return documents


def step2_split_into_chunks(documents):
    print("\nSTEP 2: Text Splitter (chunking with overlap)")
    # RecursiveCharacterTextSplitter tries to break on paragraph -> sentence -> word
    # boundaries (in that order) instead of chopping mid-sentence like a naive fixed splitter.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # max characters per chunk
        chunk_overlap=80,    # chunks share 80 chars so a fact near a boundary isn't lost
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (chunk_size=500, overlap=80).")
    print(f"  Example chunk:\n  \"{chunks[0].page_content[:150]}...\"")
    return chunks


def step3_and_4_embed_and_index(chunks):
    print("\nSTEP 3 & 4: Embedding Model + Vector Store (FAISS)")
    embeddings = get_embeddings(EMBEDDING_PROVIDER)
    # FAISS.from_documents does two things: embeds every chunk, then builds a fast
    # nearest-neighbor index over those vectors.
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  Embedded {len(chunks)} chunks using '{EMBEDDING_PROVIDER}' embeddings "
          f"and indexed them in FAISS.")
    return vectorstore


def step5_build_retriever(vectorstore, k=3):
    print("\nSTEP 5: Retriever")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"  Retriever ready - will return top-{k} most relevant chunks for any query.")
    return retriever


def step6_similarity_search(vectorstore, query, k=3):
    print(f"\nSTEP 6: Similarity Search for query: \"{query}\"")
    # similarity_search_with_score gives back BOTH the chunk and a distance score,
    # so we can see WHY these chunks were picked, not just trust it blindly.
    results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"  [{i}] score={score:.4f}  \"{doc.page_content[:100]}...\"")
    return results


def step7_assemble_context(results):
    print("\nSTEP 7: Context Retrieval (assembling chunks into one context block)")
    context = "\n\n---\n\n".join(doc.page_content for doc, _score in results)
    print(f"  Assembled {len(results)} chunks into a {len(context)}-character context block.")
    return context


def step8_generate_answer(context, query):
    print("\nSTEP 8: RAG Generation (context + question -> LLM -> answer)")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR policy assistant for Innvonix. Answer the employee's question using "
         "ONLY the context below. If the answer isn't in the context, say you don't know - "
         "do not guess.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    model = get_model(CHAT_PROVIDER)
    chain = prompt | model
    response = chain.invoke({"context": context, "question": query})
    print(f"  Answer:\n  {response.content}")
    return response.content


if __name__ == "__main__":
    query = "How many casual leaves am I entitled to per year?"

    documents = step1_load_document()
    chunks = step2_split_into_chunks(documents)
    vectorstore = step3_and_4_embed_and_index(chunks)
    retriever = step5_build_retriever(vectorstore)  # built here, used by the CLI bot file
    results = step6_similarity_search(vectorstore, query)
    context = step7_assemble_context(results)
    step8_generate_answer(context, query)