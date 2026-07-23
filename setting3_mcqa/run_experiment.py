#!/usr/bin/env python3
"""Runner for HLE few-shot hint experiment.

Usage:
    python run_experiment.py --log-name exp1
    python run_experiment.py --models haiku --limit 5 --log-name test
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HLE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = HLE_DIR.parent.parent

MODEL_CONFIGS = {
    "haiku": {
        "model_id": "anthropic/claude-haiku-4-5-20251001",
        "display_name": "Haiku 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "sonnet": {
        "model_id": "anthropic/claude-sonnet-4-5-20250929",
        "display_name": "Sonnet 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "opus": {
        "model_id": "anthropic/claude-opus-4-5-20251101",
        "display_name": "Opus 4.5",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "kimi": {
        "model_id": "openrouter/moonshotai/kimi-k2-thinking",
        "display_name": "Kimi K2",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    "glm": {
        "model_id": "openrouter/z-ai/glm-4.7",
        "display_name": "GLM 4.7",
        "thinking_args": ["-M", "reasoning_enabled=true"],
    },
    "grok": {
        "model_id": "openrouter/x-ai/grok-3-mini",
        "display_name": "Grok 3 Mini",
        "thinking_args": ["--reasoning-tokens", "10000"],
    },
    "grok_xai": {
        "model_id": "openai/grok-3-mini",
        "display_name": "Grok 3 Mini (xAI)",
        "thinking_args": ["--reasoning-tokens", "10000"],
        "env_override": {
            "OPENAI_API_KEY": "$XAI_API_KEY",
            "OPENAI_BASE_URL": "https://api.x.ai/v1",
        },
    },
    "gpt_oss": {
        "model_id": "openrouter/openai/gpt-oss-120b",
        "display_name": "GPT-OSS 120B",
        "thinking_args": ["--reasoning-effort", "high"],
    },
}

MODEL_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm", "grok", "grok_xai", "gpt_oss"]
CONDITIONS = ["baseline", "hint", "misleading"]
# Explicit arm: misleading direction + a custom system prompt (see hint_task.py).
# These force fewshot_mode=sysprompt_1 inside the task, so the few-shot user block
# matches the misleading condition; only the system prompt differs.
EXPLICIT_CONDITIONS = ["explicit", "explicit_adv"]
ALL_CONDITIONS = CONDITIONS + EXPLICIT_CONDITIONS


FEWSHOT_MODES = ["standard", "prefill_1", "prefill_2", "sysprompt_1", "sysprompt_2", "sysprompt_1_16", "sysprompt_2_16", "multiturn"]


def resolve_env_overrides(env_override: dict[str, str]) -> dict[str, str]:
    """Build a modified env dict with overrides applied.

    Values starting with '$' are resolved from the current environment.
    E.g. "$XAI_API_KEY" -> os.environ["XAI_API_KEY"]
    """
    env = os.environ.copy()
    for key, value in env_override.items():
        if value.startswith("$"):
            var_name = value[1:]
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise ValueError(
                    f"env_override references ${var_name} but it is not set "
                    f"in the environment. Did you run 'source keys.sh'?"
                )
            env[key] = resolved
        else:
            env[key] = value
    return env


async def run_single(
    condition: str,
    model_id: str,
    model_config: dict,
    log_dir: Path,
    limit: int = 100,
    timeout: int = 3600,
    max_connections: int = 10,
    fewshot_mode: str = "standard",
) -> Path | None:
    """Run a single evaluation via inspect eval subprocess."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve `inspect` from the RUNNING interpreter's venv, not the shell PATH, so
    # `python run_experiment.py` always uses the matching inspect (no venv-activation
    # needed, and a different inspect earlier on PATH can't be picked up by mistake).
    inspect_bin = Path(sys.executable).parent / "inspect"
    inspect_cmd = str(inspect_bin) if inspect_bin.exists() else "inspect"

    cmd = [
        inspect_cmd,
        "eval",
        "hint_task.py",
        "--model",
        model_config["model_id"],
        "--limit",
        str(limit),
        "--log-dir",
        str(log_dir),
        "--max-connections",
        str(max_connections),
        "--max-retries",
        "3",
        "-T",
        f"condition={condition}",
        "-T",
        f"fewshot_mode={fewshot_mode}",
    ]

    cmd.extend(model_config.get("thinking_args", []))

    mode_tag = f" ({fewshot_mode})" if fewshot_mode != "standard" else ""
    display = f"{model_config['display_name']} [{condition}{mode_tag}]"
    print(f"  Starting {display}...", flush=True)

    # Build env with overrides if specified (e.g. for xAI direct API)
    proc_env = None
    if "env_override" in model_config:
        proc_env = resolve_env_overrides(model_config["env_override"])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(HLE_DIR),
        env=proc_env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        print(f"  TIMEOUT: {display}", flush=True)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode()[:300] if stderr else "unknown error"
        print(f"  ERROR: {display} - {err_msg}", flush=True)
        return None

    eval_files = list(log_dir.glob("*.eval"))
    if eval_files:
        result = max(eval_files, key=lambda p: p.stat().st_mtime)
        print(f"  DONE: {display} -> {result.name}", flush=True)
        return result

    print(f"  NO OUTPUT: {display}", flush=True)
    return None


async def run_experiment(args):
    """Run the full experiment with parallel execution."""
    models = MODEL_ORDER if args.models == ["all"] else args.models
    conditions = args.conditions or CONDITIONS
    fewshot_mode = args.fewshot_mode

    base_log_dir = HLE_DIR / "logs" / args.log_name

    print(f"\n{'='*70}")
    print(f"HLE FEW-SHOT HINT EXPERIMENT: {args.log_name}")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Fewshot mode: {fewshot_mode}")
    print(f"Samples per config: {args.limit}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {base_log_dir}")
    print(f"{'='*70}\n")

    sem = asyncio.Semaphore(args.concurrency)
    results = {}

    async def run_with_sem(condition, model_id):
        async with sem:
            config = MODEL_CONFIGS[model_id]
            log_dir = base_log_dir / condition / model_id
            return (
                condition,
                model_id,
                await run_single(
                    condition=condition,
                    model_id=model_id,
                    model_config=config,
                    log_dir=log_dir,
                    limit=args.limit,
                    timeout=args.timeout,
                    max_connections=args.max_connections,
                    fewshot_mode=fewshot_mode,
                ),
            )

    tasks = []
    for condition in conditions:
        for model_id in models:
            tasks.append(run_with_sem(condition, model_id))

    completed = await asyncio.gather(*tasks)

    for condition, model_id, eval_path in completed:
        if condition not in results:
            results[condition] = {}
        if eval_path:
            results[condition][model_id] = str(eval_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "conditions": conditions,
            "limit": args.limit,
        },
        "eval_files": results,
    }

    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Condition':<12} {'Status'}")
    print("-" * 40)
    for condition in conditions:
        for model_id in models:
            status = (
                "OK" if results.get(condition, {}).get(model_id) else "FAILED"
            )
            display = MODEL_CONFIGS[model_id]["display_name"]
            print(f"{display:<15} {condition:<12} {status}")

    print(f"\nResults saved to: {base_log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HLE few-shot hint experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run (haiku, sonnet, opus, kimi, glm, grok, grok_xai, or 'all')",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=ALL_CONDITIONS,
        help="Conditions to run (default: baseline hint misleading). Also accepts "
             "explicit / explicit_adv (the explicit arm; pair with baseline and "
             "--fewshot-mode sysprompt_1).",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Samples per evaluation (default: 100)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per eval in seconds (default: 3600)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max concurrent evaluations (default: 6)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Max concurrent API connections per eval (default: 10)",
    )
    parser.add_argument(
        "--fewshot-mode",
        type=str,
        default="standard",
        choices=FEWSHOT_MODES,
        help="Fewshot mode: standard, prefill_1, prefill_2 (default: standard)",
    )
    parser.add_argument(
        "--log-name", type=str, required=True, help="Name for log subdirectory"
    )

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
