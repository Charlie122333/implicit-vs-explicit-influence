# Setting 1 — Binary Preference Elicitation

Everyday A/B choice (flight booking, payment method, …). Record the model's stable baseline
choice, re-ask with a nudge pushing *against* it, and check whether it switches. A GPT-5
monitor then judges whether the CoT attributes the switch to the nudge (or to deference to the
user). Arms: `implicit` (aside), `explicit` (instruct + conceal); plus guidance/adversarial
system prompts (`data/system_prompts.json`).

Only scenarios a model answers stably (≥80% consistent over 5 baseline reps) are used; the per-
model stable sets are frozen in `data/baseline_choices.json`, so baselines don't need to be recomputed.

## Files

| file | role |
|---|---|
| `data/prompts.json` | implicit prompts (35 stable scenarios) |
| `data/explicit_prompts.json` | explicit (instruct+conceal) messages, keyed `scenario\|constraint\|direction\|recommend_letter` |
| `data/constraints.json` | nudge definitions (id, theme, direction, category, `natural_hint`) |
| `data/system_prompts.json` | guidance (prefill, practical-focus) + adversarial stack |
| `data/baseline_choices.json` | per-model stable scenarios + baseline choices |
| `binary_task.py` | inspect task: dataset + solvers + `nudge_monitor` scorer |
| `monitor.py` | GPT-5 monitor (scorer) |
| `run.py` / `plot.py` | one `inspect eval` per model / detection-rate figure |

## Run

```bash
python run.py --arm implicit --log-name implicit
python run.py --arm implicit --guidance prefill --log-name prefill
python run.py --arm implicit --guidance practical_focus --log-name practical
python run.py --arm explicit --log-name explicit
python run.py --arm explicit --adversarial --log-name explicit_adv
python plot.py
```

`--models`, `--limit` to scope; `inspect view --log-dir logs/<name>` to browse; `inspect score
logs/<name>/<file>.eval` to re-judge without regenerating.

**New model:** add it to `MODEL_SPECS` in `run.py`, compute its stable scenarios (each bare
scenario ×5, keep ≥80%-consistent majorities), and add a block to `baseline_choices.json`.
