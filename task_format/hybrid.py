from __future__ import annotations

import random
from typing import Optional

from .core import TaskVerifier
from .judge import LLMJudge


class HybridRewardFunction:
    """Combines deterministic verifiers with optional LLM judge."""

    def __init__(
        self,
        judge_rate: float = 0.0,
        judge_model: str = "MiniMaxAI/MiniMax-M2",
        judge_backend: str = "vllm_server",
        judge_weight: float = 1.0,
        judge_gpus: int = 6,
        judge_dtype: str = "bfloat16",
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        judge_timeout: float = 30.0,
        judge_max_chars: int = 4000,
        judge_retries: int = 1,
    ):
        self.verifier = TaskVerifier()
        self.judge_rate = judge_rate
        self.judge_weight = judge_weight
        self.judge: Optional[LLMJudge] = None

        if judge_rate > 0:
            self.judge = LLMJudge(
                model=judge_model,
                backend=judge_backend,
                api_key=api_key,
                tensor_parallel_size=judge_gpus,
                dtype=judge_dtype,
                server_url=server_url,
                timeout=judge_timeout,
                max_completion_chars=judge_max_chars,
                max_retries=judge_retries,
            )

    def __call__(self, completions: list[str], prompts: list[str], **_) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            verifier_score, _ = self.verifier.compute_reward(completion)
            if self.judge and random.random() < self.judge_rate:
                judge_score, _ = self.judge.judge(prompt, completion)
                scaled_judge = (judge_score * 3) - 1
                rewards.append(verifier_score + (scaled_judge * self.judge_weight))
            else:
                rewards.append(verifier_score)
        return rewards

    def get_stats(self) -> dict:
        stats = {"judge_enabled": self.judge is not None}
        if self.judge:
            stats.update(self.judge.stats())
        return stats

