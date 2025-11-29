## PRIME-RL Integration

This project now delegates all RL fine-tuning to [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl). The legacy GRPO trainer has been removed in favour of PRIME-RL’s orchestrator/trainer/inference stack.

### Setup

1. Install dependencies (this uses `uv` by default):
   ```bash
   uv pip install -r requirements-train.txt
   ```
   This pulls the official `prime-rl` package plus the local TASK environment helper modules.

2. (Optional) Generate/refresh prompts:
   ```bash
   uv run python generate_prompts.py --count 2000 --output traces.processed.jsonl
   ```

### Local Environment

- The TASK verifier/reward logic now lives in `task_format/`.
- The PRIME-RL verifiers environment entrypoint is `task_trace.load_environment`.
- Config templates are under `configs/prime_rl/task_trace/`.

### Launching RL

Use the new wrapper which configures PRIME-RL with the local environment:

```bash
uv run python rl_train.py \
  --model Qwen/Qwen3-4B \
  --data traces.processed.jsonl \
  --output outputs \
  --max-steps 400 \
  --batch-size 192 \
  --rollouts-per-example 12 \
  --seq-len 4096
```

Flags such as `--max-samples`, `--max-trace-chars`, `--max-tokens`, etc. forward directly to the PRIME-RL orchestrator/trainer. Use `--dry-run` to print the command without launching, and append extra PRIME-RL arguments after `--` to pass them straight through.

### Direct PRIME-RL Invocation

Advanced users can interact with PRIME-RL directly:

```bash
uv run rl \
  --trainer @ configs/prime_rl/task_trace/trainer.toml \
  --orchestrator @ configs/prime_rl/task_trace/orchestrator.toml \
  --inference @ configs/prime_rl/task_trace/inference.toml \
  --orchestrator.env.0.args.data_path traces.processed.jsonl
```

This is equivalent to what `rl_train.py` builds automatically.

