"""
07_review_analysis_pipeline.py  (CAPSTONE)
---------------------------------------------
LEARNING GOALS
  - Combine everything: PromptTemplate + Model + Pydantic schema + Output Parser
    + Runnables/LCEL into one end-to-end pipeline that turns unstructured human
    text into structured, machine-readable JSON.
  - Compare TWO structured-output strategies on the same data:
      A) model.with_structured_output(schema)  -- provider-native structured output
      B) prompt (with format instructions) | model | PydanticOutputParser -- manual LCEL
  - Compare providers (Groq vs Mistral vs Gemini) on: output quality, JSON
    validity rate, and wall-clock latency for the same batch of reviews.
  - Practice using .batch() to process all 10 reviews per provider efficiently,
    and persist results as clean JSON to disk.

TASKS
  1. Run this file. It will process all 10 reviews from sample.json
     through every available provider (skips ones without an API key set) and
     write results to output/results_<provider>_<method>.json.
  2. Open the generated JSON files and compare: did all providers agree on
     sentiment for ambiguous reviews (e.g. review #1, which is mostly positive
     but mentions a fan-noise issue)? Where do they diverge?
  3. Check the printed timing table at the end -- which provider was fastest for
     10 reviews? Which method (A: with_structured_output vs B: LCEL parser) was
     faster/more reliable in your run?
  4. Extend the schema with a new field (e.g. `suggested_response: str`, a draft
     reply to the customer) and rerun -- see how easily the pipeline adapts.
"""

import json
import os
import time
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from model_factory import get_model

OUTPUT_DIR = "output"
PROVIDERS_TO_TRY = ["groq", "mistral", "gemini"]


# ---------------------------------------------------------------------------
# 1. Pydantic schema for structured extraction
# ---------------------------------------------------------------------------
class ReviewAnalysis(BaseModel):
    """Structured analysis of a single customer review."""
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment expressed in the review"
    )
    key_issues: list[str] = Field(
        description="Specific problems or complaints mentioned. Empty list if none."
    )
    key_positives: list[str] = Field(
        description="Specific things praised in the review. Empty list if none."
    )
    summary: str = Field(description="One-sentence neutral summary of the review")
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgently a business should follow up with this customer"
    )


# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
def load_reviews(path: str = "sample.json") -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3a. Method A: provider-native structured output
# ---------------------------------------------------------------------------
def build_chain_structured_output(provider: str):
    model = get_model(provider).with_structured_output(ReviewAnalysis)
    prompt = ChatPromptTemplate.from_template(
        "Analyze this customer review and extract structured details:\n\n{review}"
    )
    return prompt | model  # LCEL chain, output is a ReviewAnalysis object


# ---------------------------------------------------------------------------
# 3b. Method B: manual LCEL with PydanticOutputParser
# ---------------------------------------------------------------------------
def build_chain_lcel_parser(provider: str):
    parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You analyze customer reviews. {format_instructions}"),
        ("human", "{review}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    model = get_model(provider)
    return prompt | model | parser


# ---------------------------------------------------------------------------
# 4. Run pipeline for one provider/method over all reviews, using .batch()
# ---------------------------------------------------------------------------
def run_pipeline(provider: str, method: str, reviews: list[str]) -> dict:
    chain = (
        build_chain_structured_output(provider)
        if method == "structured_output"
        else build_chain_lcel_parser(provider)
    )

    inputs = [{"review": r} for r in reviews]

    start = time.time()
    try:
        results = chain.batch(inputs)
        elapsed = time.time() - start
        parsed = [r.model_dump() for r in results]
        for review_text, r in zip(reviews, parsed):
            r["original_review"] = review_text
        return {"provider": provider, "method": method, "elapsed_sec": round(elapsed, 2),
                "success": True, "results": parsed}
    except Exception as e:
        elapsed = time.time() - start
        return {"provider": provider, "method": method, "elapsed_sec": round(elapsed, 2),
                "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 5. Orchestrate: run every available provider x both methods, save + compare
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews.\n")

    summary_table = []

    for provider in PROVIDERS_TO_TRY:
        try:
            get_model(provider)  # fail fast if no API key configured
        except Exception as e:
            print(f"[SKIP] {provider}: {e}")
            continue

        for method in ["structured_output", "lcel_parser"]:
            print(f"Running provider={provider} method={method} ...")
            outcome = run_pipeline(provider, method, reviews)

            out_path = os.path.join(OUTPUT_DIR, f"results_{provider}_{method}.json")
            with open(out_path, "w") as f:
                json.dump(outcome, f, indent=2)

            status = "OK" if outcome["success"] else f"FAILED ({outcome.get('error')})"
            print(f"  -> {status} in {outcome['elapsed_sec']}s -> saved to {out_path}")

            summary_table.append({
                "provider": provider, "method": method,
                "elapsed_sec": outcome["elapsed_sec"], "success": outcome["success"],
            })

    print("\n=== Timing / Reliability Comparison ===")
    print(f"{'Provider':10s} {'Method':20s} {'Time(s)':>8s} {'Success':>8s}")
    for row in summary_table:
        print(f"{row['provider']:10s} {row['method']:20s} {row['elapsed_sec']:8.2f} {str(row['success']):>8s}")


if __name__ == "__main__":
    main()