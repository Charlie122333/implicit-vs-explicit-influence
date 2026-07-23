# Implicit vs Explicit Influence CoT Monitorability Benchmark

Code and data for *"Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence
Settings."* A benchmark that compares how well a CoT monitor detects a behavior shift
when the same biasing nudge is delivered **explicitly** (an instruction to act on it and hide
it) versus **implicitly** (a casual aside).

![Explicit vs implicit influence](assets/regimes.png)

Each item runs a task, adds a nudge, and asks (a) did the behavior shift, and (b) did the
monitor catch the nudge's influence in the CoT. The headline metric is **detection rate**:
of the cases where the nudge shifted behavior, what fraction did the monitor catch.

## The four settings

![The four settings, shown in the implicit form](assets/four_settings.png)

*Shown in the implicit form (a casual nudge). Each setting also has an explicit form — the user
instructs the model to act on the nudge and conceal it — as in the figure above.*

Everything runs through [`inspect_ai`](https://inspect.aisi.org.uk/): `run.py` drives one
`inspect eval` per model. For S1/S4/load-bearing the monitor is an in-eval scorer; for S2/S3 it
is a separate `monitor.py` step (their shift is defined across a cell/condition, not per sample).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
```

## Quickstart

```bash
cd setting1_binary
python run.py  --arm implicit --models haiku --log-name implicit --limit 12
python plot.py --conditions implicit --out figures/quickstart.pdf
```

## Run and reproduce figures

Each setting's `README` lists the exact commands per condition. General flow:

```bash
# S1, S4, load-bearing: run.py writes verdict-bearing .eval logs
python run.py --arm explicit --adversarial --log-name explicit_adv
# S2, S3: generation then the monitor step
python run.py ... ; python monitor.py --log-name ...
```

Then recreate the paper's per-model detection-rate figures from your results:

```bash
python make_figures.py                 # all settings that have results
python setting1_binary/plot.py         # or one at a time
```

The g-mean² metric (paper Appendix B) is in `analysis/gmean2.py`.

