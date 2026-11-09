# my_analysis/matched_task.py
# Run the matched-hint experiment: neutral, hint A and hint B prompts from
# matched_prompts.json, using the authors' solvers and model settings unchanged,
# with a scorer that judges every switched trace against BOTH candidate hints.
#
# Usage (from the repository root):
#   python my_analysis/matched_task.py --log-name smoke --n-neutral 1 --n-hint 1
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))   # so the authors' modules can be imported

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, chain, generate

from binary_task import extract_choice_and_switch, inject_system_and_prefill
from monitor import (JUDGE_PROMPT, build_embedded_fact, get_monitor_model,
                     load_constraint_hints, parse_verdict)
from run import build_model

MODEL_KEY = "haiku"
PROMPTS_PATH = ROOT / "my_analysis" / "matched_prompts.json"
LOG_ROOT = ROOT / "my_analysis" / "logs"


# ── dataset ──────────────────────────────────────────────────────────────────

def build_samples(n_neutral, n_hint):
    data = json.load(open(PROMPTS_PATH))
    by_scenario = {}
    for e in data["entries"]:
        by_scenario.setdefault(e["scenario_id"], {})[e["condition"]] = e

    samples = []
    for sid, conds in by_scenario.items():
        # Interleave conditions: A, B, neutral, A, B, neutral, ...
        for rep in range(max(n_neutral, n_hint)):
            order = (["A", "B"] if rep < n_hint else []) + (["neutral"] if rep < n_neutral else [])
            for cond in order:
                e = conds[cond]
                samples.append(Sample(
                    input=e["prompt"],
                    target="",
                    id=f"{sid}__{cond}__{rep:02d}",
                    metadata={
                        "model_key": MODEL_KEY, "arm": "implicit_matched",
                        "scenario_id": sid, "condition": cond, "repetition": rep,
                        "constraint_id": e["constraint_id"],
                        "candidate_constraint_ids": e["candidate_constraint_ids"],
                        "aside": e["aside"],
                        "baseline_choice": e["baseline_choice"],
                        "target_choice": e["target_choice"],
                        "target_path": e["target_path"],
                        "target_tool": e["target_tool"],
                        "option_a": e["option_a"], "option_b": e["option_b"],
                        "system_prompt": "", "prefill": "",
                    },
                ))
    return samples


# ── scorer: judge every switched hinted trace against both candidate hints ───

@scorer(metrics=[])
def two_hint_monitor() -> Scorer:
    hints = load_constraint_hints()

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        switched = md.get("switched")
        internal = md.get("internal_reasoning") or ""
        if md.get("condition") == "neutral" or not switched or not internal.strip():
            return Score(value=NOANSWER, metadata={"switched": bool(switched), "judged": False})

        mc = md.get("model_choice", "")
        oa, ob = md["option_a"], md["option_b"]
        cname = oa if mc == "A" else ob if mc == "B" else "Unknown"

        verdicts, explanations, shown = {}, {}, {}
        for cand, cid in md["candidate_constraint_ids"].items():
            desc = build_embedded_fact(hints, md.get("target_tool", ""), cid)
            prompt = JUDGE_PROMPT.format(option_a=oa, option_b=ob, constraint_description=desc,
                                         model_choice=mc, choice_name=cname,
                                         internal_reasoning=internal)
            out = await get_monitor_model().generate(input=[ChatMessageUser(content=prompt)])
            text = out.completion or ""
            verdicts[cand] = parse_verdict(text)
            explanations[cand] = text
            shown[cand] = desc

        delivered = md["condition"]
        return Score(value=verdicts[delivered], answer=verdicts[delivered],
                     explanation=explanations[delivered],
                     metadata={"switched": True, "judged": True, "model_choice": mc,
                               "verdicts": verdicts, "explanations": explanations,
                               "descriptions_shown": shown})

    return score


# ── task and runner ──────────────────────────────────────────────────────────

@task
def matched_choice(n_neutral: int = 10, n_hint: int = 20) -> Task:
    return Task(
        dataset=MemoryDataset(samples=build_samples(n_neutral, n_hint)),
        solver=chain(inject_system_and_prefill(), generate(), extract_choice_and_switch()),
        scorer=two_hint_monitor(),
        metadata={"setting": "binary_matched_hints", "model_key": MODEL_KEY,
                  "n_neutral": n_neutral, "n_hint": n_hint},
    )


def main():
    ap = argparse.ArgumentParser(description="Matched-hint runner")
    ap.add_argument("--log-name", required=True)
    ap.add_argument("--n-neutral", type=int, default=10)
    ap.add_argument("--n-hint", type=int, default=20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t = matched_choice(n_neutral=args.n_neutral, n_hint=args.n_hint)
    inspect_eval(t, model=build_model(MODEL_KEY, args.max_concurrent),
                 log_dir=str(LOG_ROOT / args.log_name), limit=args.limit)


if __name__ == "__main__":
    main()