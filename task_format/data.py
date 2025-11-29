from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_prompts(
    data_path: str,
    max_samples: int | None = None,
    max_trace_chars: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """Load prompts from JSONL file with optional length filtering."""

    prompts: list[dict] = []
    stats = {
        "total": 0,
        "skipped_long": 0,
        "skipped_malformed": 0,
    }

    def within_limit(assistant_segment: str) -> bool:
        if not max_trace_chars:
            return True
        return len(assistant_segment) <= max_trace_chars

    with Path(data_path).open(encoding="utf-8") as f:
        for line in f:
            if max_samples and len(prompts) >= max_samples:
                break
            stats["total"] += 1
            try:
                item = json.loads(line)

                if "prompt" in item:
                    prompt = item["prompt"]
                    if not prompt.startswith("<|im_start|>"):
                        prompt = f"<|im_start|>system\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                    prompts.append({"prompt": prompt})
                    continue

                trace = item.get("trace", "")
                split = trace.split("<|im_start|>assistant", 1)
                if len(split) != 2:
                    stats["skipped_malformed"] += 1
                    continue

                prompt_part, assistant_segment = split
                if not within_limit(assistant_segment):
                    stats["skipped_long"] += 1
                    continue

                prompt_part += "<|im_start|>assistant\n"
                prompts.append({"prompt": prompt_part})
            except Exception:
                stats["skipped_malformed"] += 1

    return prompts, stats

