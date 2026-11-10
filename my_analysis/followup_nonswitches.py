# my_analysis/followup_nonswitches.py
# For sampled non-switches that either labeller (strict or lenient) says recommend the
# target option: (1) nudge-handling labels using the same prompt as the 410 switches,
# (2) a verdict from the authors' monitor, told the target choice. These answers were
# never judged because the parser called them non-switches. (analysis_plan.md, s.4)
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
sys.path.insert(0, str(ROOT / "my_analysis"))
from inspect_ai.log import read_eval_log
from monitor import (JUDGE_PROMPT, MAX_COMPLETION_TOKENS, MONITOR_MODEL,
                     build_embedded_fact, load_constraint_hints, parse_verdict)
from classify_switches import PROMPT as NUDGE_PROMPT, parse as parse_nudge, RETRY_NOTE

STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
LABELS_PATH = "my_analysis/choice_labels.jsonl"
RULE_PATH = "my_analysis/audit_rules.md"
OUT_PATH = "my_analysis/nonswitch_followup.jsonl"
OPENAI_MODEL, ANTHROPIC_MODEL = "gpt-5", "claude-sonnet-5-5"

rules = open(RULE_PATH).read()
hints = load_constraint_hints()
log = read_eval_log(STAGE1_LOG)
by_id = {s.id: s for s in log.samples}
oa, an = openai.OpenAI(), anthropic.Anthropic()

recs = [json.loads(line) for line in open(LABELS_PATH) if line.strip()]
candidates = []
for r in recs:
    if r["group"] != "nonswitch":
        continue
    target = "B" if r["baseline"] == "A" else "A"
    votes = [r["gpt5_strict"], r["sonnet_strict"], r["gpt5_lenient"], r["sonnet_lenient"]]
    if target in votes:
        candidates.append((r, target))
print(f"sampled non-switches: {sum(r['group'] == 'nonswitch' for r in recs)}; "
      f"possible target recommendations to follow up: {len(candidates)}")


def ask_gpt5(prompt, max_tokens=4000):
    r = oa.chat.completions.create(model=OPENAI_MODEL, max_completion_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


def ask_sonnet(prompt):
    r = an.messages.create(model=ANTHROPIC_MODEL, max_tokens=2000,
                           messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") or "" for b in r.content if getattr(b, "type", "") == "text")


def follow_up(item):
    r, target = item
    md = by_id[r["id"]].metadata
    nudge = build_embedded_fact(hints, md.get("target_tool", ""), md.get("constraint_id", ""))
    reasoning = md.get("internal_reasoning") or ""
    out = {"id": r["id"], "scenario": r["scenario"], "baseline": r["baseline"], "target": target,
           "votes": {"gpt5_strict": r["gpt5_strict"], "sonnet_strict": r["sonnet_strict"],
                     "gpt5_lenient": r["gpt5_lenient"], "sonnet_lenient": r["sonnet_lenient"]}}

    # (1) nudge handling: same prompt as the 410 switches; keep only the nudge label
    prompt = NUDGE_PROMPT.format(rules=rules, a=md["option_a"], b=md["option_b"], nudge=nudge,
                                 reasoning=reasoning, answer=md.get("external_output") or "")
    for name, ask in (("gpt5", ask_gpt5), ("sonnet", ask_sonnet)):
        label = None
        for attempt in range(2):
            try:
                _, label = parse_nudge(ask(prompt if attempt == 0 else prompt + RETRY_NOTE))
            except Exception as e:
                label = None
            if label:
                break
        out[f"{name}_nudge"] = label

    # (2) the authors' monitor, told the target choice
    choice_name = md["option_a"] if target == "A" else md["option_b"]
    judge_prompt = JUDGE_PROMPT.format(option_a=md["option_a"], option_b=md["option_b"],
                                       constraint_description=nudge, model_choice=target,
                                       choice_name=choice_name, internal_reasoning=reasoning)
    try:
        r2 = oa.chat.completions.create(model=MONITOR_MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS,
                                        messages=[{"role": "user", "content": judge_prompt}])
        text = r2.choices[0].message.content or ""
        out["judge"], out["judge_explanation"] = parse_verdict(text), text[:400]
    except Exception as e:
        out["judge"], out["judge_explanation"] = "ERROR", str(e)[:200]
    return out


with open(OUT_PATH, "w") as f, ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(follow_up, candidates))
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"followed up: {len(results)}")
print(f"judge verdicts: { {v: sum(r['judge'] == v for r in results) for v in ('YES', 'NO', 'UNKNOWN', 'ERROR')} }")
print(f"nudge labels missing: {sum(1 for r in results if not (r['gpt5_nudge'] and r['sonnet_nudge']))}")
print(f"saved to {OUT_PATH}")