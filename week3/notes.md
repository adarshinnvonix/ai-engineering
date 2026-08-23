# Prompt Engineering, RAG Concepts & Local LLMs

## Structure

| # | File | What we're exploring |
|---|------|------------------------|
| 1 | `01_prompt_experiments.py` | Same task, 5 different prompting styles to compare outputs side by side |
| 2 | `02_prompt_improvement_iteration.py` | Take one weak prompt, iteratively improve it, observe the delta at each step |
| 3 | `03_ollama_local_vs_cloud.py` | Run a model locally via Ollama, compare vs a cloud model (Groq) on quality + latency |
| 4 | `04_vector_search_faiss_chroma.py` | Embed a small corpus, run similarity search with FAISS *and* Chroma, compare |
| 5 | `RAG_pipeline_explained.md` | Conceptual write-up of the RAG pipeline with diagrams (no implementation) |
| - | `prompt_engineering_and_rag_basics.ipynb` | All of the above as one narrative notebook, for exploring interactively cell-by-cell |

## Setup

```bash
pip install -r week3/modules.txt
```

For file 3 (Ollama), Ollama is needed to be installed and run locally.

## Hardware & Ollama notes

Ollama runs models on local machine, so requirements depend entirely on which model is pulled:

| Model size | Approx RAM needed | CPU-only usable? | Notes |
|---|---|---|---|
| ~1B (e.g. `llama3.2:1b`) | ~2 GB | Yes, fast | Good for a "does this even work" smoke test |
| ~3B (e.g. `llama3.2:3b`) | ~4 GB | Yes, usable | Good default for a laptop with no GPU |
| ~7-8B (e.g. `llama3.1:8b`, `mistral:7b`) | ~8-10 GB | Yes, but slow (~2-10 tok/s on CPU) | This is the common "real quality" tier |
| ~13-14B | ~16 GB | Slow on CPU, needs GPU for comfort | |
| 70B+ | 40-64+ GB (or multi-GPU) | Not realistically CPU-only | Skip unless you have a serious GPU rig |

**Recommendation for this exercise:** install Ollama, run `ollama pull llama3.2` (3B, ~2GB),
and it will get a reasonable local model to compare against Groq/Mistral without needing any
special hardware.