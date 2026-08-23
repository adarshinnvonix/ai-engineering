"""
prompt_experiments.py
--------------------------
LEARNING GOALS
  - There is no single "correct" way to prompt — the SAME task can produce very
    different quality outputs depending on structure, framing, and constraints.
  - Learn to recognize and use these core prompting techniques:
      Zero-shot        -> just ask, no examples
      Role/persona      -> "You are an expert X" framing changes tone/depth
      Few-shot          -> show 2-3 examples of input->output before the real one
      Chain-of-thought  -> ask the model to reason step by step before answering
      Constrained/format-> force a specific output shape (e.g. "answer in exactly
                            3 bullet points" or "respond ONLY in valid JSON")
  - Learn to actually EVALUATE the differences, not just eyeball them: this file
    prints length, structure, and lets you visually diff outputs side by side.

TASKS
  1. Run this file with PROVIDER="groq" then rerun with "mistral" — does the BEST
     prompting technique change per model, or does the ranking stay the same?
  2. Pick your own task (not sentiment classification) and write all 5 prompt
     variants yourself from scratch.
  3. For the few-shot variant, try with 1 example vs 5 examples — does output
     quality/consistency improve, plateau, or get worse (overfitting to examples)?
  4. Deliberately write a BAD prompt (vague, ambiguous, contradictory instructions)
     and add it as a 6th variant — use it as a baseline to appreciate the others.
"""

from model_factory import get_model

PROVIDER = "groq"

# Same underlying task for every prompt variant, so the comparison is apples-to-apples.
TASK_INPUT = "The delivery was 4 days late, but the item itself works perfectly and the packaging was nice."


PROMPTS = {
    "zero_shot": (
        "Classify the sentiment of this review and explain why:\n{input}"
    ),
    "role_based": (
        "You are a senior customer-experience analyst with 10 years reviewing "
        "e-commerce feedback for patterns businesses miss. Analyze this review's "
        "sentiment and flag anything a business should act on:\n{input}"
    ),
    "few_shot": (
        "Classify sentiment as positive, negative, or mixed. Examples:\n"
        "Review: 'Broke on day one, waste of money.' -> Sentiment: negative\n"
        "Review: 'Fast shipping, exactly as described, love it!' -> Sentiment: positive\n"
        "Review: 'Great product but way overpriced for what it is.' -> Sentiment: mixed\n"
        "Now classify:\nReview: '{input}' -> Sentiment:"
    ),
    "chain_of_thought": (
        "Analyze the sentiment of this review. First, list the distinct positive "
        "and negative signals separately. Then weigh them against each other. "
        "Finally, state your overall sentiment classification.\nReview: {input}"
    ),
    "constrained_format": (
        "Analyze this review and respond with ONLY valid JSON, no other text, "
        "matching exactly this shape: "
        '{{"sentiment": "positive|negative|mixed", "confidence": 0.0-1.0, "one_line_reason": "..."}}\n'
        "Review: {input}"
    ),
}


def run_all_variants():
    model = get_model(PROVIDER)
    results = {}

    for name, template in PROMPTS.items():
        prompt = template.format(input=TASK_INPUT)
        response = model.invoke(prompt)
        results[name] = response.content

    print(f"TASK INPUT:\n  \"{TASK_INPUT}\"\n")
    print("=" * 80)
    for name, output in results.items():
        print(f"\n--- {name.upper()} ---")
        print(output)
        print(f"[length: {len(output)} chars]")
    print("\n" + "=" * 80)

    print("\nSUMMARY (length as a rough proxy for verbosity):")
    for name, output in results.items():
        print(f"  {name:20s} {len(output):4d} chars")

    return results


if __name__ == "__main__":
    run_all_variants()