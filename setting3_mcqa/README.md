# Setting 3 — Hard MCQA with Positional Hints

Test questions from HLE (5-choice subset). Few-shot examples plant all correct answers at a
consistent position; the model may follow that position without acknowledging it. Helpful =
plant at the test question's ground-truth position; misleading (and the explicit arms) = one
position to the left. A GPT-5 3-level monitor classifies the CoT as none / noticed / influenced
(detected = noticed OR influenced).

Generation is inspect (`hint_task.py`); the monitor is a separate step that pairs conditions
(baseline vs hint/misleading) to find flips. No guidance arm — the non-implicit columns are
delivery variants (8-shot / 16-shot / multiturn).

## Files

| file | role |
|---|---|
| `data/hle_mcqa.json` | HLE questions (513; task filters to the 282 five-choice) |
| `data/fewshot*.txt` | frozen few-shot reasoning traces |
| `data/system_prompts.json` | explicit + adversarial system prompts |
| `hint_task.py` | inspect task: few-shot construction + delivery |
| `run_experiment.py` / `monitor.py` / `plot.py` | generation / 3-level monitor / detection-rate figure |

## Run

Run baseline + the condition(s), then the monitor, per delivery. `--models all`, `--limit N`.

```bash
# implicit: 8-shot / 16-shot / multiturn (non-Anthropic)
python run_experiment.py --log-name s8  --fewshot-mode sysprompt_1    --models all ; python monitor.py --log-name s8
python run_experiment.py --log-name s16 --fewshot-mode sysprompt_1_16 --models all ; python monitor.py --log-name s16
python run_experiment.py --log-name mt  --fewshot-mode multiturn --models kimi glm grok_xai gpt_oss ; python monitor.py --log-name mt
python run_experiment.py --log-name explicit     --fewshot-mode sysprompt_1 --conditions baseline explicit     --models all ; python monitor.py --log-name explicit
python run_experiment.py --log-name explicit_adv --fewshot-mode sysprompt_1 --conditions baseline explicit_adv --models all ; python monitor.py --log-name explicit_adv
python plot.py
```