from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        BadRequestError,
        APIError,
    )

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore
    APIConnectionError = (
        APITimeoutError
    ) = RateLimitError = BadRequestError = APIError = Exception  # type: ignore

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating AI assistant responses in a specialized format called TASK.

TASK format uses:
- plan { todo ↦ { 1 ↦ "task" } • rationale ↦ "why" } - planning
- act { think ↦ 「reasoning」 } or act { call ↦ { tool ↦ name • args } } - actions
- result { data ↦ "output" } - tool results
- response「final answer」 - response to user

Evaluate the response on these criteria. Score each 0-10:

1. COHERENCE: Is the response internally consistent? Does it make logical sense? Is it readable?
2. RELEVANCE: Does it actually address what the user asked for?
3. REASONING: Is the thinking/planning sensible and well-structured?
4. COMPLETION: Does it fully complete the task with a proper response block?
5. QUALITY: Is the final response helpful and well-written?

Respond with JSON only:
{"coherence": N, "relevance": N, "reasoning": N, "completion": N, "quality": N, "explanation": "brief reason"}"""

JUDGE_USER_TEMPLATE = """## User Request
{user_request}

## Assistant Response (TASK format)
{response}

Evaluate this response. JSON only:"""

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "integer", "minimum": 0, "maximum": 10},
        "relevance": {"type": "integer", "minimum": 0, "maximum": 10},
        "reasoning": {"type": "integer", "minimum": 0, "maximum": 10},
        "completion": {"type": "integer", "minimum": 0, "maximum": 10},
        "quality": {"type": "integer", "minimum": 0, "maximum": 10},
        "explanation": {"type": "string"},
    },
    "required": ["coherence", "relevance", "reasoning", "completion", "quality"],
}


class LLMJudge:
    """LLM-as-a-judge for semantic quality evaluation."""

    def __init__(
        self,
        model: str = "MiniMaxAI/MiniMax-M2",
        backend: str = "vllm",
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/judge",
        tensor_parallel_size: int = 6,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        server_url: Optional[str] = None,
        timeout: float = 30.0,
        max_completion_chars: int = 4000,
        max_retries: int = 1,
    ):
        self.model = model
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout if timeout and timeout > 0 else None
        self.max_completion_chars = (
            max_completion_chars if max_completion_chars and max_completion_chars > 0 else 4000
        )
        self.max_retries = max(0, max_retries)
        self.calls = 0
        self.cache_hits = 0

        client_kwargs = {}
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        if backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required. Install with: pip install openai")
            self.client = OpenAI(api_key=api_key, **client_kwargs) if api_key else OpenAI(**client_kwargs)
            self._llm = None
            self._sampling_params = None
        elif backend == "vllm_server":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for vllm_server backend")
            base_url = server_url or "http://localhost:8100/v1"
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", **client_kwargs)
            self._llm = None
            self._sampling_params = None
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm package required. Install with: pip install vllm")
            is_distributed = os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_LAUNCHED")
            if is_distributed:
                tensor_parallel_size = 1
                gpu_memory_utilization = 0.3
            self._llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                trust_remote_code=False,
                max_model_len=4096,
                enforce_eager=True,
            )
            try:
                self._sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=256,
                    guided_json=JUDGE_JSON_SCHEMA,
                )
            except TypeError:
                self._sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=256,
                )
            self.client = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _cache_key(self, prompt: str, completion: str) -> str:
        content = f"{prompt}|||{completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[dict]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with cache_file.open() as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, result: dict):
        cache_file = self.cache_dir / f"{key}.json"
        with cache_file.open("w") as f:
            json.dump(result, f)

    def _extract_user_request(self, prompt: str) -> str:
        match = re.search(r'user「([^」]+)」', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        match = re.search(r"<\|im_start\|>user\s*(.+?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        return prompt[-500:]

    def _truncate_completion(self, completion: str) -> str:
        if self.max_completion_chars and self.max_completion_chars > 0:
            return completion[: self.max_completion_chars]
        return completion

    def _parse_json_response(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        match = re.search(r'\{[^{}]*"coherence"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    def _judge_openai(self, user_request: str, completion: str) -> dict:
        truncated = self._truncate_completion(completion)
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        user_request=user_request,
                        response=truncated,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 256,
        }
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout

        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(**request_kwargs)  # type: ignore[union-attr]
                return json.loads(response.choices[0].message.content)
            except (APITimeoutError, APIConnectionError, RateLimitError, BadRequestError, APIError) as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                else:
                    raise exc
        if last_error:
            raise last_error
        raise RuntimeError("Unknown judge failure without exception")

    def _judge_vllm(self, user_request: str, completion: str) -> dict:
        prompt = f"""<|im_start|>system
{JUDGE_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{JUDGE_USER_TEMPLATE.format(user_request=user_request, response=self._truncate_completion(completion))}<|im_end|>
<|im_start|>assistant
"""
        outputs = self._llm.generate([prompt], self._sampling_params)  # type: ignore[union-attr]
        response_text = outputs[0].outputs[0].text
        return self._parse_json_response(response_text)

    def judge(self, prompt: str, completion: str) -> tuple[float, dict]:
        cache_key = self._cache_key(prompt, completion)
        cached = self._get_cached(cache_key)
        if cached:
            self.cache_hits += 1
            return cached["score"], cached["breakdown"]

        self.calls += 1
        user_request = self._extract_user_request(prompt)

        if self.backend in ("openai", "vllm_server"):
            result = self._judge_openai(user_request, completion)
        else:
            result = self._judge_vllm(user_request, completion)

        breakdown = {
            "coherence": result.get("coherence", 5) / 10,
            "relevance": result.get("relevance", 5) / 10,
            "reasoning": result.get("reasoning", 5) / 10,
            "completion": result.get("completion", 5) / 10,
            "quality": result.get("quality", 5) / 10,
        }

        score = (
            breakdown["coherence"] * 0.15
            + breakdown["relevance"] * 0.25
            + breakdown["reasoning"] * 0.2
            + breakdown["completion"] * 0.25
            + breakdown["quality"] * 0.15
        )

        self._save_cache(cache_key, {"score": score, "breakdown": breakdown})
        return score, breakdown

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_rate": self.cache_hits / max(1, self.calls + self.cache_hits),
            "backend": self.backend,
            "model": self.model,
        }

