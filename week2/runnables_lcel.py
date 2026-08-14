"""
06_runnables_lcel.py
-----------------------
LEARNING GOALS
  - The Runnable protocol is LangChain's universal interface (invoke/ainvoke/
    stream/astream/batch). Prompts, models, parsers, AND plain Python functions
    (wrapped in RunnableLambda) all satisfy it -- that's what lets you `|` them
    together freely.
  - RunnableLambda: wrap any Python function as a pipeline step.
  - RunnableParallel: run multiple Runnables on the SAME input simultaneously,
    collect results into a dict -- e.g. get sentiment AND summary in one round
    trip's worth of chain-building (though calls still happen separately unless
    you design a single structured-output call, this is about composition).
  - RunnablePassthrough: forward the original input untouched alongside
    transformed branches -- essential for keeping context (e.g. original review
    text) available downstream after other steps have already changed the input
    shape.
  - LCEL (LangChain Expression Language): the `|` operator itself. It builds a
    RunnableSequence declaratively, and every LCEL chain automatically gets
    invoke/ainvoke/stream/astream/batch for free -- you never write that
    plumbing yourself.

TASKS
  1. Build a RunnableLambda that cleans/lowercases input text, and prepend it to
     an existing prompt|model|parser chain -- confirm preprocessing now happens
     automatically on every .invoke().
  2. Build a RunnableParallel that fans the SAME review text out to two
     differently-prompted chains (one extracts sentiment, one extracts a
     one-line summary) and returns both in a single dict result.
  3. Use RunnablePassthrough.assign(...) to enrich a dict with a new key computed
     from existing keys, without losing the original keys.
  4. Call .batch() on any chain with a LIST of inputs and compare timing vs
     looping .invoke() yourself -- batch can parallelize under the hood.
"""

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from model_factory import get_model

PROVIDER = "groq"


def runnable_lambda_demo():
    clean_text = RunnableLambda(lambda x: x.strip().lower())
    prompt = ChatPromptTemplate.from_template("Summarize in 5 words: {text}")
    model = get_model(PROVIDER)

    chain = clean_text | (lambda t: {"text": t}) | prompt | model | StrOutputParser()
    result = chain.invoke("   THIS PRODUCT IS AMAZING AND FAST SHIPPING TOO!!!   ")
    print("[RunnableLambda preprocessing] ->", result)


def runnable_parallel_demo():
    model = get_model(PROVIDER)
    parser = StrOutputParser()

    sentiment_chain = (
        ChatPromptTemplate.from_template("In one word (positive/negative/neutral), sentiment of: {review}")
        | model | parser
    )
    summary_chain = (
        ChatPromptTemplate.from_template("Summarize in under 8 words: {review}")
        | model | parser
    )

    parallel = RunnableParallel(sentiment=sentiment_chain, summary=summary_chain)
    result = parallel.invoke({"review": "Great battery life but the screen scratches way too easily."})
    print("\n[RunnableParallel] ->", result)


def runnable_passthrough_demo():
    model = get_model(PROVIDER)
    parser = StrOutputParser()

    tag_chain = (
        ChatPromptTemplate.from_template("Give a single category tag for: {review}")
        | model | parser
    )

    # Keep original 'review' key AND add a new 'tag' key computed from it
    enrich = RunnablePassthrough.assign(tag=tag_chain)
    result = enrich.invoke({"review": "The laptop overheats during video calls."})
    print("\n[RunnablePassthrough.assign] ->", result)


def batch_demo():
    model = get_model(PROVIDER)
    parser = StrOutputParser()
    chain = (
        ChatPromptTemplate.from_template("One-word sentiment for: {review}")
        | model | parser
    )

    reviews = [
        {"review": "Fast shipping, loved it."},
        {"review": "Broke after two days."},
        {"review": "It's okay, nothing special."},
    ]
    results = chain.batch(reviews)
    print("\n[batch] ->", results)


if __name__ == "__main__":
    runnable_lambda_demo()
    runnable_parallel_demo()
    runnable_passthrough_demo()
    batch_demo()