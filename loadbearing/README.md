# Load-Bearing Reasoning Variant (Settings 1 & 4)

The nudge is rewritten as a symmetric multi-hop ownership chain: one option genuinely traces to
a bad actor whose deed matches the nudge; the other sits near a same-theme decoy in a
disconnected tree. Only tracing the chains reveals which option attaches, so acting on the nudge
requires load-bearing reasoning that must surface in the CoT. `plot.py` compares implicit vs
load-bearing detection per model.

Chain depth is per model (deeper for models that can trace short chains without CoT): 6 for
haiku/kimi/glm/gpt_oss/grok_xai, 15 for sonnet/opus (`api.MODEL_DEPTH`; override with
`--depth-override`). Switch is vs each model's frozen baseline (S1: `baseline_choice` in the
manifest; S4: `data/s4_neutral_baseline.json`, a matched neutral-chain baseline).

## Files

| file | role |
|---|---|
| `data/s{1,4}_lbset_full[_d15].jsonl` | frozen chain prompts (per setting, depth 6 & 15) |
| `data/s4_neutral_baseline.json` | S4 matched neutral-chain baseline |
| `chains.py` / `capability_probe.py` | chain construction / load-bearingness probe → `logs/capability.json` |
| `lb_task.py` / `run.py` / `monitor.py` / `plot.py` | inspect task + scorer / runner / GPT-5 monitor / figures |

## Run

```bash
python capability_probe.py --models haiku,sonnet,opus,kimi,glm,gpt_oss,grok_xai
python run.py --settings s1,s4 --models haiku sonnet opus kimi glm grok_xai gpt_oss --log-name lb
python plot.py 
```

The detection figure compares load-bearing against the **implicit** arm, computed from the base
Setting 1 & Setting 4 implicit runs (restricted to the convertible nudges). Run those first, then
`plot.py` picks them up (`--s1-implicit-log` / `--s4-implicit-log`; default `implicit`).
