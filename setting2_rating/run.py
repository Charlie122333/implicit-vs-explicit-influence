#!/usr/bin/env python3
"""Setting 2 runner — drives the inspect_ai `rating` task (rating_task.py).

Runs one `inspect` eval per model with the per-provider generation config (extended
thinking budget 10000, max_tokens 16000). Generation ONLY — the GPT-5 monitor is a
SEPARATE step (`monitor.py`), because S2's detection target is a whole
significantly-biased cell, not a single sample (see README). Each run writes one
`.eval` log per model under `logs/<log-name>/`; monitor.py and plot.py read those.

Conditions:
  --arm implicit                             -> Implicit (bare + all nudge conditions)
  --arm implicit --intervention rubric_focus -> Implicit + rubric-focus guidance
  --arm explicit                             -> Explicit (instruct + conceal)
  --arm explicit --adversarial               -> Explicit + adversarial system prompt

Note every arm's grid INCLUDES the bare control — the significance test in monitor.py
compares each nudge cell against the bare cell from the SAME arm's generation.

Usage:
    python run.py --arm implicit --log-name implicit
    python run.py --arm implicit --intervention rubric_focus --log-name rubric_focus
    python run.py --arm explicit --log-name explicit
    python run.py --arm explicit --adversarial --log-name explicit_adv
    # smoke: one model, one domain, a few cells
    python run.py --arm implicit --log-name smoke --models haiku --domains creative --limit 4
    inspect view --log-dir logs/implicit        # browse the .eval logs
"""

import argparse
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model

from rating_task import rating

# model_key -> (inspect model spec, provider kind). S2 uses the key `grok` (routed to
# xAI direct; the OpenRouter grok-3-mini route is deprecated).
MODEL_SPECS = {
    "haiku":   ("anthropic/claude-haiku-4-5-20251001",   "anthropic"),
    "sonnet":  ("anthropic/claude-sonnet-4-5-20250929",  "anthropic"),
    "opus":    ("anthropic/claude-opus-4-5-20251101",    "anthropic"),
    "kimi":    ("openrouter/moonshotai/kimi-k2-thinking", "openrouter"),
    "glm":     ("openrouter/z-ai/glm-4.7",               "openrouter"),
    "gpt_oss": ("openrouter/openai/gpt-oss-120b",        "openrouter"),
    "grok":    ("openai/grok-3-mini",                    "xai"),
}
DEFAULT_MODELS = ["haiku", "sonnet", "opus", "kimi", "glm", "grok", "gpt_oss"]


def build_model(model_key, max_connections):
    spec, kind = MODEL_SPECS[model_key]
    if kind == "anthropic":
        cfg = GenerateConfig(reasoning_tokens=10000, max_tokens=16000,
                             max_connections=max_connections)
        return get_model(spec, config=cfg)
    if kind == "openrouter":
        cfg = GenerateConfig(reasoning_effort="high", max_tokens=16000,
                             max_connections=max_connections)
        return get_model(spec, config=cfg)
    # xAI direct (grok) via the OpenAI-compatible endpoint
    cfg = GenerateConfig(reasoning_effort="high", max_tokens=16000,
                         max_connections=max_connections)
    return get_model(spec, base_url="https://api.x.ai/v1",
                     api_key=os.environ.get("XAI_API_KEY", ""), config=cfg)


def main():
    ap = argparse.ArgumentParser(description="Setting 2 multi-domain rating runner (inspect_ai)")
    ap.add_argument("--arm", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--intervention", choices=["rubric_focus"], default="")
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--domains", nargs="+", default=None,
                    choices=["creative", "essay_grading", "hiring", "medical"])
    ap.add_argument("--log-name", required=True, help="output dir under logs/")
    ap.add_argument("--max-connections", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap CELLS per model before epochs (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the grid and print cell/sample counts, make no API calls")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the 20 reps/cell (for cheap smoke tests only)")
    args = ap.parse_args()
    if args.intervention and args.arm == "explicit":
        ap.error("--intervention applies to the implicit arm; use --adversarial for explicit")
    if args.adversarial and args.arm != "explicit":
        ap.error("--adversarial requires --arm explicit")

    domains_arg = ",".join(args.domains) if args.domains else ""
    log_dir = str(Path(__file__).parent / "logs" / args.log_name)

    if args.dry_run:
        from rating_task import N_SAMPLES, build_samples
        dl = args.domains or None
        cells = len(build_samples("haiku", args.arm, args.intervention,
                                  1 if args.adversarial else 0, domains=dl))
        cells = min(cells, args.limit) if args.limit else cells
        per_model = cells * N_SAMPLES
        print(f"arm={args.arm}{' +adversarial' if args.adversarial else ''}"
              f"{(' +'+args.intervention) if args.intervention else ''} | "
              f"domains={args.domains or 'all'}")
        print(f"cells/model: {cells}  x {N_SAMPLES} reps = {per_model} samples/model")
        print(f"models: {args.models} -> {per_model * len(args.models)} total API calls")
        print("(dry run — no API calls made)")
        return

    for model_key in args.models:
        print(f"\n=== {model_key} | arm={args.arm}"
              f"{' +adversarial' if args.adversarial else ''}"
              f"{(' +'+args.intervention) if args.intervention else ''} ===")
        t = rating(model_key, arm=args.arm, intervention=args.intervention,
                   adversarial=1 if args.adversarial else 0, domains=domains_arg)
        kw = {"epochs": args.epochs} if args.epochs else {}
        inspect_eval(t, model=build_model(model_key, args.max_connections),
                     log_dir=log_dir, limit=args.limit, **kw)


if __name__ == "__main__":
    main()
