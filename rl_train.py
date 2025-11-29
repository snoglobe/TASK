#!/usr/bin/env python3
"""
Prime-RL launcher for TASK-format environments.

Configures the PRIME-RL `rl` entrypoint with the local `task-trace` verifiers
environment and forwards common overrides (model path, dataset, batch sizing).
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE_DIR = Path("configs/prime_rl/task_trace")


def ensure_prime_rl_installed():
    if importlib.util.find_spec("prime_rl") is None:
        raise RuntimeError(
            "prime_rl is not installed. Install from git with:\n"
            "  uv pip install \"prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl.git\""
        )


def build_command(args: argparse.Namespace) -> list[str]:
    trainer_cfg = TEMPLATE_DIR / "trainer.toml"
    orch_cfg = TEMPLATE_DIR / "orchestrator.toml"
    infer_cfg = TEMPLATE_DIR / "inference.toml"

    for cfg in (trainer_cfg, orch_cfg, infer_cfg):
        if not cfg.exists():
            raise FileNotFoundError(f"Missing PRIME-RL config template: {cfg}")

    cmd = [
        "uv",
        "run",
        "rl",
        "--trainer",
        "@",
        str(trainer_cfg),
        "--orchestrator",
        "@",
        str(orch_cfg),
        "--inference",
        "@",
        str(infer_cfg),
        "--output-dir",
        str(args.output),
    ]

    overrides: list[tuple[str, str | int | float]] = [
        ("--trainer.model.name", args.model),
        ("--orchestrator.model.name", args.model),
        ("--inference.model.name", args.model),
        ("--orchestrator.env.0.args.data_path", args.data),
        ("--trainer.max_steps", args.max_steps),
        ("--orchestrator.max_steps", args.max_steps),
        ("--trainer.buffer.batch_size", args.batch_size),
        ("--trainer.buffer.seq_len", args.seq_len),
        ("--orchestrator.batch_size", args.batch_size),
        ("--orchestrator.rollouts_per_example", args.rollouts_per_example),
        ("--orchestrator.seq_len", args.seq_len),
        ("--trainer.optim.lr", args.learning_rate),
        ("--trainer.optim.weight_decay", args.weight_decay),
        ("--orchestrator.sampling.max_tokens", args.max_tokens),
        ("--orchestrator.sampling.temperature", args.temperature),
    ]

    if args.max_samples is not None:
        overrides.append(("--orchestrator.env.0.args.max_samples", args.max_samples))
    if args.max_trace_chars is not None:
        overrides.append(("--orchestrator.env.0.args.max_trace_chars", args.max_trace_chars))

    for flag, value in overrides:
        cmd.extend([flag, str(value)])

    cmd.extend(args.extra_prime_rl_args)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PRIME-RL on TASK data.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Base or checkpoint HF model id.")
    parser.add_argument("--data", default="traces.processed.jsonl", help="Prompt dataset file.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory for PRIME-RL artifacts.")
    parser.add_argument("--max-steps", type=int, default=200, help="Number of trainer/orchestrator steps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Effective batch size per step.")
    parser.add_argument("--rollouts-per-example", type=int, default=8, help="Rollouts per sampled prompt.")
    parser.add_argument("--seq-len", type=int, default=4096, help="Maximum sequence length tokens.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Generation cap for orchestrator sampling.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Trainer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Trainer weight decay.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--max-samples", type=int, help="Optional cap on dataset size.")
    parser.add_argument(
        "--max-trace-chars",
        type=int,
        default=50000,
        help="Skip traces whose assistant output exceeds this char count (0 disables).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the PRIME-RL command without launching.",
    )
    parser.add_argument(
        "extra_prime_rl_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded verbatim to `uv run rl`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_trace_chars == 0:
        args.max_trace_chars = None

    ensure_prime_rl_installed()

    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to launch PRIME-RL. See https://docs.astral.sh/uv/")

    args.output.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args)
    print("[Prime-RL] " + " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)