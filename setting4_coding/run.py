#!/usr/bin/env python3
"""Setting 4 runner — drives the inspect_ai `coding_task` task (coding_task.py).

Runs one `inspect` eval per model (each needs its own frozen-baseline dataset), with the
per-provider generation config (anthropic: extended thinking budget 10000, max_tokens
16000; openrouter/xai: reasoning_effort "high", max_tokens 16000). Generation AND the
GPT-5 monitor both run inside the eval — the monitor is the `nudge_monitor` scorer — so
each run produces a single .eval log holding behavior + verdict. plot.py reads those logs.

Conditions:
  --arm explicit                               -> Explicit
  --arm explicit --adversarial                 -> Explicit + adversarial system prompt
  --arm implicit                               -> Implicit (bare aside)
  --arm implicit --guidance ack_dismiss        -> Implicit + ack-and-dismiss
  --arm implicit --guidance noise_framing      -> Implicit + noise-framing

Usage:
    python run.py --arm implicit --log-name implicit
    python run.py --arm implicit --guidance ack_dismiss --log-name ack
    python run.py --arm explicit --adversarial --log-name explicit_adv --models haiku
    inspect view --log-dir logs/implicit        # inspect the .eval logs
"""

import argparse
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model

from coding_task import coding_task

# model_key -> (inspect model spec, provider kind). Thinking/reasoning + max_tokens are
# set below in build_model().
MODEL_SPECS = {
    "haiku":    ("anthropic/claude-haiku-4-5-20251001",  "anthropic"),
    "sonnet":   ("anthropic/claude-sonnet-4-5-20250929", "anthropic"),
    "opus":     ("anthropic/claude-opus-4-5-20251101",   "anthropic"),
    "kimi":     ("openrouter/moonshotai/kimi-k2-thinking", "openrouter"),
    "glm":      ("openrouter/z-ai/glm-4.7",              "openrouter"),
    "gpt_oss":  ("openrouter/openai/gpt-oss-120b",       "openrouter"),
    "grok_xai": ("openai/grok-3-mini",                   "xai"),
}
DEFAULT_MODELS = ["haiku", "sonnet", "opus", "kimi", "glm", "grok_xai", "gpt_oss"]


def build_model(model_key, max_connections):
    spec, kind = MODEL_SPECS[model_key]
    if kind == "anthropic":
        cfg = GenerateConfig(reasoning_tokens=10000, max_tokens=16000, max_connections=max_connections)
        return get_model(spec, config=cfg)
    if kind == "openrouter":
        cfg = GenerateConfig(reasoning_effort="high", max_tokens=16000, max_connections=max_connections)
        return get_model(spec, config=cfg)
    # xAI direct (grok) via the OpenAI-compatible endpoint
    cfg = GenerateConfig(reasoning_effort="high", max_tokens=16000, max_connections=max_connections)
    return get_model(spec, base_url="https://api.x.ai/v1",
                     api_key=os.environ.get("XAI_API_KEY", ""), config=cfg)


def main():
    ap = argparse.ArgumentParser(description="Setting 4 open-ended coding runner (inspect_ai)")
    ap.add_argument("--arm", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--guidance", choices=["ack_dismiss", "noise_framing"], default="")
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--log-name", required=True, help="output dir under logs/")
    ap.add_argument("--max-concurrent", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="cap samples per model (smoke tests)")
    args = ap.parse_args()
    if args.guidance and args.arm == "explicit":
        ap.error("--guidance applies to the implicit arm; use --adversarial for explicit")
    if args.adversarial and args.arm != "explicit":
        ap.error("--adversarial requires --arm explicit")

    log_dir = str(Path(__file__).parent / "logs" / args.log_name)
    for model_key in args.models:
        print(f"\n=== {model_key} | arm={args.arm}"
              f"{' +adversarial' if args.adversarial else ''}"
              f"{(' +'+args.guidance) if args.guidance else ''} ===")
        t = coding_task(model_key, arm=args.arm, guidance=args.guidance,
                        adversarial=1 if args.adversarial else 0)
        inspect_eval(t, model=build_model(model_key, args.max_concurrent),
                     log_dir=log_dir, limit=args.limit)


if __name__ == "__main__":
    main()
