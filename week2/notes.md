# LangChain Fundamentals

A file-wise walkthrough of LangChain core concepts using **Groq**, **Mistral**, and **Gemini**
as interchangeable LLM backends. Each file is self-contained, runnable on its own.
## Setup

```bash
pip install -r week2/modules.txt
```

Need at least one of these API keys:
- `GROQ_API_KEY`   - https://console.groq.com/keys
- `MISTRAL_API_KEY`- https://console.mistral.ai/api-keys
- `GOOGLE_API_KEY` - https://aistudio.google.com/apikey

## Program files

| # | File | Concept |
|---|------|---------|
| 1 | `message_types.py` | SystemMessage / HumanMessage / AIMessage / ToolMessage |
| 2 | `prompt_templates.py` | ChatPromptTemplate, reusable prompts, partials |
| 3 | `model_invocation.py` | invoke / ainvoke / stream / astream, sync vs async vs streaming |
| 4 | `structured_outputs_pydantic.py` | Pydantic schemas + `with_structured_output()` |
| 5 | `output_parsers.py` | StrOutputParser, PydanticOutputParser, JsonOutputParser |
| 6 | `runnables_lcel.py` | Runnable protocol, RunnableLambda/Parallel/Passthrough, LCEL `|` chains |
| 7 | `review_analysis_pipeline.py` | Capstone: full structured-output pipeline over 10 reviews, compares providers, saves JSON |

## Core mental model

LangChain's whole design rests on **one interface**: the `Runnable`. A prompt template, a chat
model, an output parser, and even a plain Python function (via `RunnableLambda`) all implement:

- `.invoke(input)` — run once, synchronously
- `.ainvoke(input)` — run once, asynchronously
- `.stream(input)` — yield output incrementally
- `.astream(input)` — async streaming
- `.batch([inputs])` — run many inputs, parallelized

Because every piece shares this interface, it can be **piped them together with `|`** (LCEL -
LangChain Expression Language) into a single new Runnable:

```python
chain = prompt | model | parser
chain.invoke({"topic": "black holes"})
```

`prompt | model | parser` is really `RunnableSequence(prompt, model, parser)`, the output of
each step becomes the input of the next. Once you see this, everything else in LangChain
(chains, agents, RAG pipelines) is just this pattern composed at larger scale.