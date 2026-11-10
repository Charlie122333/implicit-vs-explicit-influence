# my_analysis/parser_trace.py
# A traced copy of the authors' extract_choice_natural (setting1_binary/choice_extraction.py).
# Same logic, but it also returns which step decided the label and the evidence it used.
# Running this file:
#   1. checks the copy against the original on every Stage 1 and pilot answer, and
#      separately reports answers where the original parser now disagrees with the
#      choice stored at run time,
#   2. shows how often each step decides the label in Stage 1 (pooled),
#   3. prints the trace for every pilot answer.
import glob
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
from inspect_ai.log import read_eval_log
from choice_extraction import CHOICE_KEYWORDS, extract_choice_natural

STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
PILOT_DIR = "my_analysis/logs/pilot_insurance"

# Copied exactly from the original function
REC_RE = re.compile("|".join([
    r"(?:i(?:'d| would)?)\s+(?:recommend|suggest|go with|choose|pick|say)",
    r"(?:my recommendation|best (?:option|choice|bet)|i think .* (?:better|best))",
    r"(?:go (?:with|for)|opt for|stick with|lean toward)",
]), re.IGNORECASE)


def count_hits(segment, kws):
    return sum(1 for kw in kws if kw in segment)


def trace_choice(text, scenario_id):
    """Same decisions as extract_choice_natural, returned as (label, step, evidence)."""
    if not text:
        return None, "empty_or_unknown", {}
    keywords = CHOICE_KEYWORDS.get(scenario_id)
    if not keywords:
        return None, "empty_or_unknown", {}

    text_lower = text.lower()
    tail = text_lower[-500:] if len(text_lower) > 500 else text_lower
    tail_a = count_hits(tail, keywords["a"])
    tail_b = count_hits(tail, keywords["b"])

    skipped = []
    for sent in reversed(re.split(r'[.!?]+', tail)):
        if REC_RE.search(sent):
            sa = count_hits(sent, keywords["a"])
            sb = count_hits(sent, keywords["b"])
            if sa > 0 and sb == 0:
                return "A", "rec_sentence", {"sentence": sent.strip(), "skipped": skipped}
            if sb > 0 and sa == 0:
                return "B", "rec_sentence", {"sentence": sent.strip(), "skipped": skipped}
            skipped.append(sent.strip())

    evidence = {
        "tail_a": tail_a, "tail_b": tail_b,
        "tail_keywords_a": [k for k in keywords["a"] if k in tail],
        "tail_keywords_b": [k for k in keywords["b"] if k in tail],
        "skipped_rec_sentences": skipped,
    }

    if tail_a > tail_b:
        return "A", "tail_count", evidence
    if tail_b > tail_a:
        return "B", "tail_count", evidence

    last_a, kw_a = -1, None
    for kw in keywords["a"]:
        if kw in text_lower:
            last_a, kw_a = text_lower.rfind(kw), kw
            break
    last_b, kw_b = -1, None
    for kw in keywords["b"]:
        if kw in text_lower:
            last_b, kw_b = text_lower.rfind(kw), kw
            break
    evidence.update({"last_a": last_a, "keyword_a": kw_a, "last_b": last_b,
                     "keyword_b": kw_b, "answer_length": len(text_lower)})
    if last_a > last_b and last_a >= 0:
        return "A", "last_mention", evidence
    if last_b > last_a and last_b >= 0:
        return "B", "last_mention", evidence
    return None, "no_mention", evidence


def check_against_original(log, label):
    """The traced copy must agree with the original parser. Separately, report any
    answers where the original parser now disagrees with the choice stored at run time."""
    copy_bad, stored_bad = 0, 0
    for s in log.samples:
        md = s.metadata or {}
        text = md.get("external_output") or ""
        original = extract_choice_natural(text, md.get("scenario_id", ""))
        mine, _, _ = trace_choice(text, md.get("scenario_id", ""))
        if original != mine:
            copy_bad += 1
            print(f"  COPY MISMATCH {s.id}: original={original} copy={mine}")
        if original != md.get("model_choice"):
            stored_bad += 1
            print(f"  STORED DIFFERS {s.id}: parser now={original} stored={md.get('model_choice')} "
                  f"| A={md.get('option_a')} | B={md.get('option_b')}")
    print(f"{label}: {len(log.samples)} answers; copy mismatches {copy_bad}; stored differs {stored_bad}")
    return copy_bad


stage1 = read_eval_log(STAGE1_LOG)
pilot = read_eval_log(sorted(glob.glob(f"{PILOT_DIR}/*.eval"))[-1])

print("=== 1. checking the copy against the original ===")
copy_errors = check_against_original(stage1, "Stage 1") + check_against_original(pilot, "Pilot")
assert copy_errors == 0, "the traced copy does not match the original; fix before trusting any trace"

print("\n=== 2. which step decided the label, Stage 1 (pooled across scenarios) ===")
steps = Counter(trace_choice(s.metadata.get("external_output") or "",
                             s.metadata.get("scenario_id", ""))[1] for s in stage1.samples)
for step, n in steps.most_common():
    print(f"  {step}: {n} ({n / len(stage1.samples):.0%})")

print("\n=== 3. trace for every pilot answer ===")
for s in pilot.samples:
    md = s.metadata
    label, step, ev = trace_choice(md.get("external_output") or "", md["scenario_id"])
    print(f"\n{s.id}  ->  {label}  via {step}")
    for key, value in ev.items():
        print(f"    {key}: {value}")
