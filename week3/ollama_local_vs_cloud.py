"""
ollama_local_vs_cloud.py
------------------------------
LEARNING GOALS
  - Ollama runs open-weight models ENTIRELY on your own machine — no API key,
    no data leaving your device, no per-token cost, but bounded by your local
    hardware (see README "Hardware & Ollama notes").
  - Because ChatOllama implements the same Runnable interface as every cloud
    model here, swapping local <-> cloud is a ONE LINE change (provider name) —
    this is the payoff of LangChain's unified interface from the previous module.
  - Compare on 3 axes: latency, output quality/coherence, and (conceptually)
    privacy/cost trade-offs — local is slower and often lower-quality for a given
    model size, but free per-call and fully private.

PREREQUISITES
  1. Install Ollama: https://ollama.com/download
  2. Pull a small model:  `ollama pull llama3.2`   (~2GB, 3B params)
  3. Make sure it's running: `ollama serve` (usually auto-starts as a background
     service after install — check with `ollama list`)

TASKS
  1. Run this file. If Ollama isn't running, you'll get a clear connection error
     instead of a crash — read it, then start Ollama and rerun.
  2. Compare the local vs cloud response for the SAME prompt — is the local
     model's answer noticeably shorter/simpler/less accurate? That gap is what
     you're paying for (or saving) by going cloud vs local.
  3. Try a bigger local model (`ollama pull llama3.1:8b`) and see if the quality
     gap narrows — note how much slower it is on your specific hardware.
  4. Time 5 consecutive local calls vs 5 cloud calls (via a simple loop + time.time())
     and compute average latency for each — cloud is usually faster per call despite
     the network round trip, because Groq/Mistral run on much bigger hardware.
"""

import time
from model_factory import get_model

LOCAL_PROVIDER = "ollama"
CLOUD_PROVIDER = "groq"
QUESTION = "Explain what a vector database is, in 3 sentences, for someone new to AI."


def try_local():
    print(f"\n--- LOCAL (Ollama, model={get_model.__module__}) ---")
    try:
        model = get_model(LOCAL_PROVIDER)
        start = time.time()
        response = model.invoke(QUESTION)
        elapsed = time.time() - start
        print(f"Response ({elapsed:.2f}s):\n{response.content}")
        return elapsed, response.content
    except Exception as e:
        print(
            "Could not reach local Ollama server.\n"
            "  -> Is Ollama installed and running? Try: `ollama serve` in a terminal,\n"
            "     and confirm a model is pulled with: `ollama list`\n"
            f"  Raw error: {e}"
        )
        return None, None


def try_cloud():
    print(f"\n--- CLOUD ({CLOUD_PROVIDER}) ---")
    try:
        model = get_model(CLOUD_PROVIDER)
        start = time.time()
        response = model.invoke(QUESTION)
        elapsed = time.time() - start
        print(f"Response ({elapsed:.2f}s):\n{response.content}")
        return elapsed, response.content
    except Exception as e:
        print(f"Could not reach {CLOUD_PROVIDER}. Check your API key in .env. Raw error: {e}")
        return None, None


def compare():
    local_time, local_output = try_local()
    cloud_time, cloud_output = try_cloud()

    print(f"\n{'=' * 60}\nCOMPARISON")
    if local_time is not None:
        print(f"  Local (Ollama) latency : {local_time:.2f}s")
    else:
        print("  Local (Ollama) latency : N/A (not running)")
    if cloud_time is not None:
        print(f"  Cloud ({CLOUD_PROVIDER}) latency  : {cloud_time:.2f}s")
    else:
        print(f"  Cloud ({CLOUD_PROVIDER}) latency  : N/A (no key configured)")

    if local_time and cloud_time:
        faster = "Local" if local_time < cloud_time else "Cloud"
        print(f"  -> {faster} was faster in this run. Try re-running a few times; "
              f"cloud latency varies with network/provider load, local varies with your CPU/GPU load.")


if __name__ == "__main__":
    compare()