# my_analysis/final_checks.py
# Final checks before sharing (analysis_plan.md, code checks):
#   1. Build one answer-only classifier prompt exactly as classify_choice.py does and
#      confirm it contains no reasoning, nudge description or judge explanation.
#   2. Check that the label files add up to the counts in the analysis plan.
# (Check 3, the clean rerun, is a shell command: see below.)
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
sys.path.insert(0, str(ROOT / "my_analysis"))
from inspect_ai.log import read_eval_log
from monitor import build_embedded_fact, load_constraint_hints
from classify_choice import PROMPT, STAGE1_LOG, PROMPTS_PATH, RULE_PATH

# ---- Check 1: one prompt, checked for leaks ----
recs = [json.loads(l) for l in open("my_analysis/choice_labels.jsonl") if l.strip()]
example_id = recs[0]["id"]
sample = next(s for s in read_eval_log(STAGE1_LOG).samples if s.id == example_id)
md = sample.metadata
rule1 = open(RULE_PATH).read().split("# Rule 2")[0].strip()
bare = next(p["prompt"] for p in sorted(json.load(open(PROMPTS_PATH))["prompts"], key=lambda p: p["prompt_id"])
            if p["scenario_id"] == md["scenario_id"] and p["constraint_id"] is None)
prompt = PROMPT.format(rule1=rule1, a=md["option_a"], b=md["option_b"], bare=bare,
                       answer=md.get("external_output") or "")

print(f"=== Check 1: prompt for {example_id} ===\n{prompt}\n")
reasoning = (md.get("internal_reasoning") or "").strip()
nudge = build_embedded_fact(load_constraint_hints(), md.get("target_tool", ""), md.get("constraint_id", ""))
explanation = (sample.scores["nudge_monitor"].explanation or "").strip()
leaks = {
    "reasoning (first 200 chars)": reasoning[:200] and reasoning[:200] in prompt,
    "nudge description": nudge and nudge in prompt,
    "judge explanation (first 100 chars)": explanation[:100] and explanation[:100] in prompt,
    "word 'baseline'": "baseline" in prompt.lower(),
}
for what, found in leaks.items():
    print(f"   {'LEAK' if found else 'ok  '}  {what}")

# ---- Check 2: the counts add up ----
print("\n=== Check 2: counts ===")
groups = Counter(r["group"] for r in recs)
print(f"   choice labels: {dict(groups)}   (expected switch 410, nonswitch 150)")
missing = sum(1 for r in recs if not (r["gpt5_strict"] and r["sonnet_strict"]))
print(f"   answers missing a strict label: {missing}   (expected 0)")
yes = sum(r["judge"] == "YES" for r in recs if r["group"] == "switch")
print(f"   judge YES among the 410 switches: {yes}   (expected 49)")
follow = [json.loads(l) for l in open("my_analysis/nonswitch_followup.jsonl") if l.strip()]
print(f"   follow-up candidates: {len(follow)}, judge YES: {sum(f['judge'] == 'YES' for f in follow)}"
      f"   (expected 41 and 8)")
cons = Counter((r["gpt5_strict"] == r["sonnet_strict"]) for r in recs)
print(f"   models agree on strict label: {cons[True]} of {len(recs)}   (expected 504 of 560)")