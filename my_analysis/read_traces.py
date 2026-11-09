# my_analysis/read_traces.py
# Print eligible traces per hint with Haiku's final answer and both judge explanations,
# plus the neutral runs' final answers, to check whether the parsed choice is a real
# recommendation.
import glob
import json
from inspect_ai.log import read_eval_log

N_PER_HINT = 4
SHOW_REASONING = False      # set True to also print the full reasoning trace
ANSWER_CHARS = 800          # how much of the end of each final answer to show
LOG_DIR = "my_analysis/logs/pilot_insurance"
ELIGIBLE_PATH = "my_analysis/eligible_pilot.json"

eligible = json.load(open(ELIGIBLE_PATH))["eligible"]
log = read_eval_log(sorted(glob.glob(f"{LOG_DIR}/*.eval"))[-1])
by_id = {s.id: s for s in log.samples}

for cond in ("A", "B"):
    chosen = [r for r in eligible if r["condition"] == cond][:N_PER_HINT]
    for r in chosen:
        s = by_id[r["id"]]
        md = s.metadata
        smd = s.scores["two_hint_monitor"].metadata
        print("=" * 100)
        print(f"{r['id']}   (delivered hint: {cond}, parsed choice: {md.get('model_choice')})")
        print("=" * 100)
        if SHOW_REASONING:
            print(r["reasoning"])
        print(f"\n--- FINAL ANSWER (last {ANSWER_CHARS} chars) ---")
        print((md.get("external_output") or "")[-ANSWER_CHARS:])
        for cand in ("A", "B"):
            print(f"\n--- judge on {cand}: {smd['verdicts'][cand]} ---")
            print(smd["explanations"][cand])
        print()

print("#" * 100)
print("NEUTRAL RUNS: parsed choice and end of final answer")
print("#" * 100)
for s in log.samples:
    md = s.metadata
    if md.get("condition") != "neutral":
        continue
    print(f"\n{s.id}   parsed choice: {md.get('model_choice')}")
    print((md.get("external_output") or "")[-400:])