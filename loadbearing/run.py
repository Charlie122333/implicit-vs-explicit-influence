#!/usr/bin/env python3
"""Load-bearing runner (Settings 1 & 4) — drives the inspect_ai `loadbearing` task (lb_task.py).

Runs one `inspect` eval per (setting, model): each needs its own frozen manifest (at the
model's chain depth) and frozen baseline. The per-provider generation config uses extended
thinking budget 10000, max_tokens 16000. Generation AND the GPT-5 monitor both run inside
the eval — the monitor is the `lb_monitor` scorer — so each run produces a single .eval log
holding behavior (choice + `switched`) + verdict. plot.py reads those logs.

Depth per model: sonnet/opus use the 15-hop manifests, everyone else 6-hop (set inside
lb_task.MODEL_DEPTH).

Usage:
    source ../../keys.sh
    python run.py --settings s1,s4 --models haiku sonnet opus kimi glm grok_xai gpt_oss
    python run.py --settings s1 --models haiku --log-name smoke --limit 8   # smoke test
    inspect view --log-dir logs/lb        # inspect the .eval logs
"""

import argparse
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model

from lb_task import loadbearing

# model_key -> (inspect model spec, provider kind). Thinking/reasoning + max_tokens are set
# below in build_model().
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
    ap = argparse.ArgumentParser(description="Load-bearing runner (Settings 1 & 4, inspect_ai)")
    ap.add_argument("--settings", default="s1,s4", help="comma-separated: s1,s4")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--log-name", default="lb", help="output dir under logs/ (default: lb)")
    ap.add_argument("--max-concurrent", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="cap samples per (setting,model) (smoke tests)")
    args = ap.parse_args()

    log_dir = str(Path(__file__).parent / "logs" / args.log_name)
    for setting in args.settings.split(","):
        assert setting in ("s1", "s4"), f"unknown setting {setting}"
        for model_key in args.models:
            print(f"\n=== {setting} | {model_key} ===")
            task = loadbearing(setting, model_key)
            inspect_eval(task, model=build_model(model_key, args.max_concurrent),
                         log_dir=log_dir, limit=args.limit)


if __name__ == "__main__":
    main()
