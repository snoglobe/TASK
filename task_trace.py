from __future__ import annotations

from datasets import Dataset

import verifiers as vf
from task_format import TaskVerifier, load_prompts


async def _deterministic_reward(completion, **kwargs) -> float:
    verifier: TaskVerifier = kwargs["verifier"]
    text = completion if isinstance(completion, str) else completion[-1]["content"]
    score, _ = verifier.compute_reward(text)
    return score


def load_environment(
    data_path: str = "traces.processed.jsonl",
    max_samples: int | None = None,
    max_trace_chars: int | None = 50000,
    env_id: str = "task-trace",
) -> vf.Environment:
    """Load a TASK-format verifiers environment backed by local prompts."""

    prompts, _ = load_prompts(
        data_path=data_path,
        max_samples=max_samples,
        max_trace_chars=max_trace_chars,
    )
    dataset = Dataset.from_list(
        [
            {
                "prompt": item["prompt"],
                "task": env_id,
            }
            for item in prompts
        ]
    )

    parser = vf.Parser()
    verifier = TaskVerifier()
    rubric = vf.Rubric(parser=parser)
    rubric.class_objects["verifier"] = verifier
    rubric.add_reward_func(_deterministic_reward, weight=1.0)

    env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        message_type="completion",
        env_id=env_id,
    )
    return env

