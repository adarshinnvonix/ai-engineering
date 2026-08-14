"""
03_model_invocation.py
------------------------
LEARNING GOALS
  - The 4 core ways to call any Runnable (and therefore any chat model):
      .invoke()   -> blocking, single call, returns full AIMessage at once
      .stream()   -> blocking but yields chunks (AIMessageChunk) as they arrive
      .ainvoke()  -> same as invoke but async (non-blocking, awaitable)
      .astream()  -> same as stream but async generator
  - Understand WHEN each matters:
      invoke  -> simple scripts, batch jobs
      stream  -> user-facing UIs (show tokens as they generate, lower perceived latency)
      ainvoke -> web servers handling many concurrent requests (FastAPI etc.)
      astream -> concurrent + streaming (e.g. many chatbot users at once)
  - Measure and compare latency-to-first-token vs total time across providers.

TASKS
  1. Run this file, note "time to first token" for stream() vs total time for
     invoke() -- streaming feels faster even if total time is similar/slower.
  2. Change PROVIDER between "groq", "mistral", "gemini" and compare raw speed --
     Groq's LPU inference is usually dramatically faster; this is a good exercise
     in picking the right provider for latency-sensitive apps.
  3. Use asyncio.gather() to fire ainvoke() at all 3 providers SIMULTANEOUSLY and
     compare wall-clock time vs calling them sequentially with invoke().
"""

import asyncio
import time
from model_factory import get_model

PROVIDER = "groq"
QUESTION = "List 3 benefits of using LangChain, one line each."


def sync_invoke_demo():
    model = get_model(PROVIDER)
    start = time.time()
    response = model.invoke(QUESTION)
    elapsed = time.time() - start
    print(f"[invoke] ({elapsed:.2f}s) {response.content}")


def sync_stream_demo():
    model = get_model(PROVIDER)
    start = time.time()
    first_token_time = None
    chunks = []
    for chunk in model.stream(QUESTION):
        if first_token_time is None:
            first_token_time = time.time() - start
        chunks.append(chunk.content)
        print(chunk.content, end="", flush=True)
    total = time.time() - start
    print(f"\n[stream] first_token={first_token_time:.2f}s total={total:.2f}s")


async def async_invoke_demo():
    model = get_model(PROVIDER)
    start = time.time()
    response = await model.ainvoke(QUESTION)
    elapsed = time.time() - start
    print(f"[ainvoke] ({elapsed:.2f}s) {response.content}")


async def async_stream_demo():
    model = get_model(PROVIDER)
    print("[astream] ", end="")
    async for chunk in model.astream(QUESTION):
        print(chunk.content, end="", flush=True)
    print()


async def concurrent_multi_provider_demo():
    """Fire all 3 providers at once with ainvoke and compare wall-clock time."""
    providers = ["groq", "mistral", "gemini"]
    models = {p: get_model(p) for p in providers}

    start = time.time()
    tasks = [models[p].ainvoke(QUESTION) for p in providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    print(f"\n[concurrent ainvoke across {providers}] total_wall_clock={elapsed:.2f}s")
    for p, r in zip(providers, results):
        if isinstance(r, Exception):
            print(f"  {p}: ERROR - {r}")
        else:
            print(f"  {p}: {r.content[:80]}...")


if __name__ == "__main__":
    sync_invoke_demo()
    sync_stream_demo()
    asyncio.run(async_invoke_demo())
    asyncio.run(async_stream_demo())
    asyncio.run(concurrent_multi_provider_demo())