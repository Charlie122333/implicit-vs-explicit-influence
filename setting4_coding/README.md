# Setting 4 — Open-Ended Coding

Model writes Python for a task with free choice of library. Record its stable baseline library,
add a nudge (an ethical/social aside about that library), and check whether it switches. A GPT-5
monitor judges whether the CoT attributes the switch to the nudge (or to the user's steer).
Arms: `implicit` (+ `ack_dismiss` / `noise_framing` guidance) and `explicit` (+ `--adversarial`).

Library choice is detected from the generated code (import/API regexes). Baselines are frozen
per model in `data/baseline_libraries.json`; a switch is a detected library ≠ that baseline.

## Files

| file | role |
|---|---|
| `data/tasks.json` | 8 coding tasks |
| `data/constraints.json` | 11 nudges (8 ethical, 3 social) |
| `data/explicit_prompts.json` | explicit (instruct+conceal) messages, keyed `task\|constraint\|baseline_library` |
| `data/baseline_libraries.json` | per-model baseline library per task |
| `data/system_prompts.json` | guidance (ack/noise) + adversarial stack |
| `coding_task.py` | inspect task: dataset + library detection + `nudge_monitor` scorer |
| `run.py` / `monitor.py` / `plot.py` | one `inspect eval` per model / GPT-5 monitor / detection-rate figure |

## Run

```bash
python run.py --arm implicit --log-name implicit
python run.py --arm implicit --guidance ack_dismiss --log-name ack
python run.py --arm implicit --guidance noise_framing --log-name noise
python run.py --arm explicit --log-name explicit
python run.py --arm explicit --adversarial --log-name explicit_adv
python plot.py
```
