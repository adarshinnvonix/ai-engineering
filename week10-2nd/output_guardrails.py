"""
Output guardrails: checks that run on the LLM's answer BEFORE the user sees it.

  1. Schema      - the answer must match the HRAnswer Pydantic model
  2. Grounded    - the supporting quote must really appear in the retrieved context,
                   and every number in the answer must appear in the context (or the question)
  3. Safety      - no personal data, no prompt leakage, no unsafe content in the answer

If any check fails, the user gets a safe fallback message instead of the model's answer.
"""

import re
from pydantic import BaseModel, Field
from input_guardrails import PII_PATTERNS, UNSAFE

FALLBACK_ANSWER = (
    "I couldn't find a reliable answer to that in the Innvonix HR policy. "
    "Please contact HR directly."
)
COMPANY_EMAIL_DOMAIN = "@innvonix.example"   # policy emails like the ICC address are allowed

LEAK = re.compile(r"system prompt|my (?:hidden |system )?instructions|as instructed by the system", re.I)


class HRAnswer(BaseModel):
    """The shape the LLM must answer in. Anything else is rejected."""
    answer: str = Field(description="Short answer to the employee's question, using only the context.")
    answer_found: bool = Field(description="True only if the context contains the answer.")
    supporting_quote: str = Field(
        default="",
        description="One sentence copied word for word from the context that supports the answer. "
                    "Empty if answer_found is false.",
    )


def _words(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def quote_is_grounded(quote, context):
    """True if the quote is really in the context (ignoring case, punctuation, markdown)."""
    q, c = _words(quote), _words(context)
    if len(q) < 3:
        return False
    if " ".join(q) in " ".join(c):
        return True
    known = set(c)                                   # small slips are OK: 90% of words must exist
    return sum(w in known for w in q) / len(q) >= 0.9


def _numbers(text):
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def ungrounded_numbers(answer, context, question=""):
    """Numbers in the answer that appear nowhere in the context or the question."""
    return sorted(_numbers(answer) - _numbers(context) - _numbers(question))


def safety_issues(answer):
    issues = []
    for name, pattern in PII_PATTERNS:
        for m in pattern.finditer(answer):
            # the email pattern can swallow a sentence-ending period, so strip punctuation first
            if name == "EMAIL" and m.group(0).lower().rstrip(".,;:)").endswith(COMPANY_EMAIL_DOMAIN):
                continue
            issues.append(f"pii:{name}")
    if LEAK.search(answer):
        issues.append("prompt_leak")
    if any(p.search(answer) for p in UNSAFE):
        issues.append("unsafe_content")
    return sorted(set(issues))


def check_output(parsed, context, question=""):
    """parsed = an HRAnswer, or None if the model's output failed schema validation."""
    if parsed is None:
        return {"passed": False, "final_answer": FALLBACK_ANSWER,
                "reasons": ["schema_validation_failed"], "checks": {"schema": False}}

    checks = {"schema": True}
    if not parsed.answer_found:
        return {"passed": True, "final_answer": FALLBACK_ANSWER,
                "reasons": ["model_found_no_answer"], "checks": checks}

    reasons = []
    checks["quote_grounded"] = quote_is_grounded(parsed.supporting_quote, context)
    if not checks["quote_grounded"]:
        reasons.append("quote_not_in_context")

    bad_numbers = ungrounded_numbers(parsed.answer, context, question)
    checks["numbers_grounded"] = not bad_numbers
    if bad_numbers:
        reasons.append("ungrounded_numbers:" + ",".join(bad_numbers))

    issues = safety_issues(parsed.answer)
    checks["safe"] = not issues
    reasons += issues

    passed = not reasons
    return {"passed": passed, "final_answer": parsed.answer if passed else FALLBACK_ANSWER,
            "reasons": reasons, "checks": checks}