"""
04_structured_outputs_pydantic.py
-----------------------------------
LEARNING GOALS
  - Pydantic BaseModel defines a SCHEMA (field names, types, descriptions) that
    LangChain can hand to the LLM provider's native structured-output / function-
    calling mechanism, guaranteeing the response matches your schema.
  - model.with_structured_output(PydanticClass) returns a NEW Runnable whose
    .invoke() returns an instance of your Pydantic class directly -- not a string
    you have to parse yourself.
  - Field descriptions matter: they're sent to the model as instructions, so
    write them like you're briefing a human.
  - Understand the difference between this approach and output PARSERS (next file):
    with_structured_output relies on the provider's native tool/JSON mode; a
    parser works on ANY model by instructing it via prompt text and parsing the
    raw string after the fact.

TASKS
  1. Run this file across all 3 providers and check: do they all correctly return
     a validated Pydantic object? Note any provider that struggles with strict
     schemas (nested lists, enums) -- this is a real trade-off you should know.
  2. Add a field with a Literal[...] or Enum type (e.g. urgency: Literal["low",
     "medium", "high"]) and confirm the model is constrained to those values.
  3. Try a NESTED Pydantic model (a field whose type is another BaseModel) and see
     if all 3 providers handle nesting correctly.
"""

from typing import Literal
from pydantic import BaseModel, Field
from model_factory import get_model

PROVIDER = "groq"


class MovieReview(BaseModel):
    """Structured extraction of a movie review's key attributes."""
    title: str = Field(description="Name of the movie being reviewed")
    sentiment: Literal["positive", "negative", "mixed", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating_out_of_10: int = Field(description="Reviewer's implied rating from 1-10")
    key_points: list[str] = Field(description="2-4 short bullet points summarizing the review's main points")


def structured_output_demo():
    model = get_model(PROVIDER)
    structured_model = model.with_structured_output(MovieReview)

    review_text = (
        "Inception blew my mind. The visuals were stunning and the concept of "
        "dream-within-a-dream heists was genuinely original. The ending left "
        "me arguing with friends for hours. Only complaint: the pacing dragged "
        "a bit in the middle act. Easily a 9/10 for me."
    )

    result = structured_model.invoke(f"Extract structured info from this review:\n{review_text}")
    print("Type:", type(result).__name__)
    print("Parsed object:", result)
    print("As dict:", result.model_dump())
    print("Title:", result.title, "| Rating:", result.rating_out_of_10)


def compare_across_providers():
    review_text = (
        "The new phone is decent but overpriced. Battery life is great, camera "
        "is average, and the software has bugs. Wouldn't recommend at this price."
    )
    for provider in ["groq", "mistral", "gemini"]:
        try:
            model = get_model(provider).with_structured_output(MovieReview)
            result = model.invoke(f"Extract structured info (treat as a general review):\n{review_text}")
            print(f"\n[{provider}] -> {result.model_dump()}")
        except Exception as e:
            print(f"\n[{provider}] ERROR -> {e}")


if __name__ == "__main__":
    structured_output_demo()
    compare_across_providers()