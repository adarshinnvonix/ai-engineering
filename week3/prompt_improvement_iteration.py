"""
prompt_improvement_iteration.py  ("Week 2 pipeline")
-----------------------------------------------------------
LEARNING GOALS
  - Prompt engineering is ITERATIVE, not one-shot: you write a prompt, look at
    what's wrong with the output, patch the specific failure, and re-test.
  - Practice diagnosing WHY an output is bad (too vague? wrong format? missing
    constraint? hallucinated detail?) rather than just rewriting randomly.
  - See how small, targeted additions (a constraint, an example, a role) compound
    across iterations into a much stronger final prompt.

TASKS
  1. Run this file and read each iteration's output plus the "diagnosis" comment
     above it — confirm you agree with what's wrong before seeing the fix.
  2. Pick ONE of your own prompts from the previous file (or a new task) and do
     your own 4-iteration improvement loop, writing your own diagnosis notes.
  3. Try reordering the fixes (e.g. add the format constraint FIRST instead of
     last) — does the order in which you fix things change the final quality
     much, or does it converge to the same place either way?
  4. Add a 5th iteration that intentionally over-constrains the prompt (too many
     rigid rules) and observe whether output quality starts to DEGRADE — good
     prompts have a point of diminishing/negative returns.
"""

from model_factory import get_model

PROVIDER = "groq"
TASK_CONTEXT = "Write email subject lines for a SaaS product's 'trial expiring in 3 days' reminder email."


ITERATIONS = [
    {
        "label": "v1 - naive",
        "diagnosis": "Baseline: vague ask, no length/quantity/tone constraints. Expect generic, "
                     "inconsistent-length output that's hard to use directly.",
        "prompt": TASK_CONTEXT,
    },
    {
        "label": "v2 - add quantity + length constraint",
        "diagnosis": "v1's output was unusable as-is (unknown how many, unknown length). "
                     "Fix: force a specific count and character limit so it's directly usable.",
        "prompt": TASK_CONTEXT + "\nGive exactly 5 options, each under 50 characters.",
    },
    {
        "label": "v3 - add tone + urgency guidance",
        "diagnosis": "v2 outputs were technically compliant but tonally flat/generic. "
                     "Fix: specify the emotional register we actually want (urgency without "
                     "being spammy/aggressive).",
        "prompt": TASK_CONTEXT
        + "\nGive exactly 5 options, each under 50 characters. Tone: create urgency "
          "but stay friendly and non-pushy — avoid ALL CAPS or excessive punctuation "
          "(no 'ACT NOW!!!' style spam).",
    },
    {
        "label": "v4 - add format + few-shot anchor",
        "diagnosis": "v3 outputs were good individually but inconsistent in STYLE across the "
                     "5 (some questions, some statements, some with emoji). Fix: give one "
                     "example to anchor style, and force a numbered list for easy scanning.",
        "prompt": TASK_CONTEXT
        + "\nGive exactly 5 options, each under 50 characters. Tone: urgency without being "
          "pushy, no ALL CAPS, no excessive punctuation. Match this style: "
          "'Your trial ends Friday — here's what you'll lose'. "
          "Output as a numbered list, nothing else.",
    },
]


def run_iterations():
    model = get_model(PROVIDER)

    for i, step in enumerate(ITERATIONS, start=1):
        print(f"\n{'=' * 80}\nITERATION {i}: {step['label']}")
        print(f"Diagnosis before this fix: {step['diagnosis']}")
        print(f"\nPrompt sent:\n  {step['prompt']}\n")

        response = model.invoke(step["prompt"])
        print(f"Output:\n{response.content}")

    print(f"\n{'=' * 80}\nCompare ITERATION 1 vs ITERATION {len(ITERATIONS)} above — "
          f"same underlying task, dramatically different usability.")


if __name__ == "__main__":
    run_iterations()