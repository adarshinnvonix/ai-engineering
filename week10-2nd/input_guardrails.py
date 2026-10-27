"""
Input guardrails: checks that run BEFORE a question reaches the LLM.

check_input(text) decides one of:
  allow   -> question is fine
  redact  -> question had personal data (PII); it was masked, then allowed
  block   -> question is refused (prompt injection, unsafe, off-topic, too long)

These are simple rules (regex + keywords) so you can SEE how guardrails work.
Real systems layer a classifier or LLM check on top - see the README for the gaps.
"""

import re
from dataclasses import dataclass, field

MAX_CHARS = 1000

# --- PII: personal data we mask before anything is logged or sent to the LLM ------
# Order matters: specific patterns first, so a 16-digit card isn't half-matched as an ID.
PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
]

# --- Prompt injection: attempts to override the bot's instructions ----------------
INJECTION = [re.compile(p, re.I | re.M) for p in [
    r"\b(?:ignore|disregard|forget|override)\b\s+(?:(?:all|any|the|your)\s+)*(?:previous|prior|above|earlier|preceding)\b",
    r"\boverride\b\s+(?:your|the|all)\s+(?:rules|instructions|safety|guardrails|restrictions)",
    r"\byou\s+are\s+now\b",
    r"\bfrom\s+now\s+on\s+you\b",
    r"\bpretend\s+(?:to\s+be|you\s+are|you're)\b",
    r"\b(?:reveal|show|print|repeat|output|tell\s+me|leak)\b\s+(?:me\s+)?(?:your|the\s+system|your\s+hidden|your\s+initial|your\s+original)\s+(?:system\s+)?(?:prompt|instructions?)\b",
    r"\bsystem\s+prompt\b",
    r"\b(?:developer\s+mode|jailbreak|do\s+anything\s+now)\b",
    r"^\s*(?:system|assistant)\s*:",
    r"<\|?(?:im_start|im_end|system)\|?>|\[/?INST\]|###\s*(?:system|instruction)",
    r"\b(?:print|dump|output|repeat)\b\s+(?:the\s+|all\s+)?(?:context|documents?|retrieved\s+\w+)\s+(?:verbatim|word\s+for\s+word|exactly)",
]]

# --- Unsafe requests: harmful intent, not an HR question --------------------------
UNSAFE = [re.compile(p, re.I) for p in [
    r"\b(?:make|build|create|assemble)\b.{0,20}\b(?:bomb|explosives?)\b",
    r"\b(?:write|create|build)\b.{0,20}\b(?:malware|ransomware|keylogger|virus)\b",
    r"\b(?:hack|break)\s+into\b",
    r"\bsteal\s+(?:passwords?|credentials|data|money)\b",
    r"\b(?:poison|murder)\s+(?:someone|him|her|my|a|the)\b",
]]

# --- Off-topic: block only CLEAR off-topic requests --------------------------------
# A question is blocked only if it looks off-topic (code, trivia, recipes...) AND has no HR word.
# Why not "block anything without an HR keyword"? I tried that first: it wrongly blocked 7 of 15
# normal HR questions - including bullying, harassment and burnout questions. Refusing those is
# worse than answering a trivia question, so unclear cases are allowed through, and the output
# guardrails (grounded answer or "not found") deal with them later.
OFF_TOPIC_HINTS = re.compile(
    r"\b(?:python|javascript|java|c\+\+|code|coding|script|sql|program(?:ming)?|algorithm|debug|"
    r"recipe|ingredients?|capital\s+of|president\s+of|prime\s+minister|who\s+won|weather|forecast|"
    r"movie|song|lyrics|joke|poem|story|translate|horoscope|bitcoin|cricket|football|homework|"
    r"equation|solve)\b", re.I)

HR_STEMS = [
    "leave", "vacation", "holiday", "day off", "days off", "salary", "salaries", "pay", "paid", "payroll",
    "payslip", "bonus", "policy", "policies", "hr", "human resources", "work", "office", "remote",
    "wfh", "hours", "attendance", "insurance", "benefit", "expense", "reimburs", "notice", "resign",
    "terminat", "harass", "conduct", "provident", "pf", "wellness", "learning", "maternity",
    "paternity", "bereavement", "sick", "casual", "probation", "employee", "manager", "company",
    "innvonix", "appraisal", "promotion", "dress code", "timesheet", "overtime", "shift",
]
HR_TOPIC = re.compile(r"\b(?:" + "|".join(HR_STEMS) + r")", re.I)


@dataclass
class InputVerdict:
    action: str                      # "allow" | "redact" | "block"
    reason: str                      # "ok", "pii_redacted", "prompt_injection", ...
    clean_text: str                  # PII-masked text (safe to log and to send on)
    pii_found: list = field(default_factory=list)


def redact_pii(text):
    found = []
    for name, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(name)
            text = pattern.sub(f"[{name}]", text)
    return text, found


def check_input(text):
    too_long = len(text) > MAX_CHARS
    clean, pii = redact_pii(text[:MAX_CHARS])   # always mask first, so logs never hold raw PII

    if too_long:
        return InputVerdict("block", "input_too_long", clean, pii)
    if any(p.search(clean) for p in INJECTION):
        return InputVerdict("block", "prompt_injection", clean, pii)
    if any(p.search(clean) for p in UNSAFE):
        return InputVerdict("block", "unsafe_request", clean, pii)
    if OFF_TOPIC_HINTS.search(clean) and not HR_TOPIC.search(clean):
        return InputVerdict("block", "off_topic", clean, pii)
    if pii:
        return InputVerdict("redact", "pii_redacted", clean, pii)
    return InputVerdict("allow", "ok", clean, pii)