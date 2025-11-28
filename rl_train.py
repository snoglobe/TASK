#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for TASK format.

Uses hybrid reward signals:
1. Deterministic verifiers:
   - Bracket balance
   - Todo satisfaction
   - Tag reference validity
   - Response completion
   - Structural correctness

2. LLM-as-a-judge (optional, sampled):
   - Coherence and relevance
   - Reasoning quality
   - Task completion
   - Prevents reward hacking

Usage:
    accelerate launch --config_file accelerate_config.yaml rl_train.py \
        --model ./checkpoints/final \
        --output ./checkpoints/rl

    # With LLM judge (requires OPENAI_API_KEY in config)
    accelerate launch --config_file accelerate_config.yaml rl_train.py \
        --model ./checkpoints/final \
        --judge-rate 0.1 \
        --judge-model gpt-4o-mini
"""

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Optional backends for LLM judge
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# =============================================================================
# TASK Format Verifiers (Reward Functions)
# =============================================================================

class TaskVerifier:
    """Deterministic verifier for TASK format traces."""
    
    def __init__(self):
        # Patterns
        self.todo_def_pattern = re.compile(r'(\d+)\s*↦\s*[「"]')
        self.satisfies_pattern = re.compile(r'⊨\s*(\d+)')
        self.tag_def_pattern = re.compile(r'🏷\s*["\']?([a-zA-Z_][a-zA-Z0-9_-]*)["\']?')
        self.ref_pattern = re.compile(r'※\s*["\']?([a-zA-Z_][a-zA-Z0-9_-]*)["\']?')
        self.ref_array_pattern = re.compile(r'※\s*\[([^\]]+)\]')
    
    def check_brackets_balanced(self, trace: str) -> tuple[bool, str]:
        """Check if all brackets are balanced."""
        counts = {
            '{': 0, '}': 0,
            '[': 0, ']': 0,
            '「': 0, '」': 0,
        }
        for char in trace:
            if char in counts:
                counts[char] += 1
        
        if counts['{'] != counts['}']:
            return False, f"Brace mismatch: {counts['{']} open, {counts['}']} close"
        if counts['['] != counts[']']:
            return False, f"Bracket mismatch: {counts['[']} open, {counts[']']} close"
        if counts['「'] != counts['」']:
            return False, f"CJK bracket mismatch: {counts['「']} open, {counts['」']} close"
        return True, "Brackets balanced"
    
    def check_todos_satisfied(self, trace: str) -> tuple[bool, str, set, set]:
        """Check if all todos are satisfied."""
        # Find todo definitions in plan block
        plan_match = re.search(r'plan\s*\{.*?todo\s*↦\s*\{([^}]*)\}', trace, re.DOTALL)
        defined_todos = set()
        if plan_match:
            todo_block = plan_match.group(1)
            for match in self.todo_def_pattern.finditer(todo_block):
                defined_todos.add(int(match.group(1)))
        
        # Find satisfied todos
        satisfied_todos = set()
        for match in self.satisfies_pattern.finditer(trace):
            satisfied_todos.add(int(match.group(1)))
        
        unsatisfied = defined_todos - satisfied_todos
        if unsatisfied:
            return False, f"Unsatisfied todos: {unsatisfied}", defined_todos, satisfied_todos
        return True, "All todos satisfied", defined_todos, satisfied_todos
    
    def check_todos_start_at_one(self, trace: str) -> tuple[bool, str]:
        """Check if todos start at 1."""
        plan_match = re.search(r'plan\s*\{.*?todo\s*↦\s*\{([^}]*)\}', trace, re.DOTALL)
        if plan_match:
            todo_block = plan_match.group(1)
            todos = [int(m.group(1)) for m in self.todo_def_pattern.finditer(todo_block)]
            if todos and min(todos) != 1:
                return False, f"Todos start at {min(todos)}, not 1"
        return True, "Todos start at 1"
    
    def check_refs_valid(self, trace: str) -> tuple[bool, str, set, set]:
        """Check if all references point to defined tags."""
        # Find all defined tags
        defined_tags = set()
        for match in self.tag_def_pattern.finditer(trace):
            defined_tags.add(match.group(1))
        
        # Find all references
        referenced_tags = set()
        for match in self.ref_pattern.finditer(trace):
            referenced_tags.add(match.group(1))
        for match in self.ref_array_pattern.finditer(trace):
            for item in match.group(1).split('•'):
                item = item.strip().strip('"\'')
                if item and re.match(r'[a-zA-Z_]', item):
                    referenced_tags.add(item)
        
        # Check for orphan references
        orphans = referenced_tags - defined_tags
        # Filter out usr1, sys1 etc which are typically defined at the start
        orphans = {t for t in orphans if not re.match(r'^(usr|sys)\d+$', t)}
        
        if orphans:
            return False, f"Orphan references: {orphans}", defined_tags, referenced_tags
        return True, "All references valid", defined_tags, referenced_tags
    
    def check_has_response(self, trace: str) -> tuple[bool, str]:
        """Check if trace ends with a response block."""
        if 'response「' in trace or 'response 「' in trace:
            return True, "Has response"
        return False, "Missing response block"
    
    def check_has_plan(self, trace: str) -> tuple[bool, str]:
        """Check if trace has a plan block."""
        if 'plan {' in trace or 'plan{' in trace:
            return True, "Has plan"
        return False, "Missing plan block"
    
    def check_structure_order(self, trace: str) -> tuple[bool, str]:
        """Check if structure follows correct order: plan → act → response."""
        plan_pos = trace.find('plan {')
        if plan_pos == -1:
            plan_pos = trace.find('plan{')
        
        response_pos = trace.find('response「')
        if response_pos == -1:
            response_pos = trace.find('response 「')
        
        if plan_pos != -1 and response_pos != -1:
            if plan_pos > response_pos:
                return False, "Plan comes after response"
        return True, "Structure order correct"
    
    def compute_reward(self, trace: str) -> tuple[float, dict]:
        """Compute total reward and breakdown."""
        rewards = {}
        total = 0.0
        
        # Hard constraints (must pass or big penalty)
        balanced, msg = self.check_brackets_balanced(trace)
        rewards['brackets_balanced'] = 1.0 if balanced else -2.0
        total += rewards['brackets_balanced']
        
        has_response, msg = self.check_has_response(trace)
        rewards['has_response'] = 0.5 if has_response else -1.5
        total += rewards['has_response']
        
        has_plan, msg = self.check_has_plan(trace)
        rewards['has_plan'] = 0.3 if has_plan else -0.5
        total += rewards['has_plan']
        
        # Soft constraints
        todos_satisfied, msg, defined, satisfied = self.check_todos_satisfied(trace)
        if defined:
            ratio = len(satisfied) / len(defined) if defined else 0
            rewards['todos_satisfied'] = ratio * 0.5
        else:
            rewards['todos_satisfied'] = 0.0
        total += rewards['todos_satisfied']
        
        todos_start_one, msg = self.check_todos_start_at_one(trace)
        rewards['todos_start_at_1'] = 0.3 if todos_start_one else -0.5
        total += rewards['todos_start_at_1']
        
        refs_valid, msg, defined_tags, referenced_tags = self.check_refs_valid(trace)
        rewards['refs_valid'] = 0.3 if refs_valid else -0.3
        total += rewards['refs_valid']
        
        structure_ok, msg = self.check_structure_order(trace)
        rewards['structure_order'] = 0.1 if structure_ok else -0.2
        total += rewards['structure_order']
        
        return total, rewards


# =============================================================================
# LLM-as-a-Judge
# =============================================================================

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


class LLMJudge:
    """LLM-as-a-judge for semantic quality evaluation.
    
    Supports two backends:
    - "openai": Uses OpenAI API (requires openai package)
    - "vllm": Uses local vLLM server (requires vllm package)
    
    For local models on H200 cluster:
    - 6x H200 (846GB) can run Qwen2.5-72B-Instruct at BF16 or Llama-3.1-405B at FP8
    """
    
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        backend: str = "vllm",  # "vllm" or "openai"
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/judge",
        tensor_parallel_size: int = 6,  # Number of GPUs for judge
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",  # "bfloat16", "float16", "auto"
    ):
        self.model = model
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.calls = 0
        self.cache_hits = 0
        
        if backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required. Install with: pip install openai")
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self._llm = None
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm package required. Install with: pip install vllm")
            print(f"[Judge] Loading {model} on {tensor_parallel_size} GPUs...")
            self._llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                trust_remote_code=True,
                max_model_len=8192,  # Limit context for judge efficiency
            )
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=256,
            )
            print(f"[Judge] Model loaded successfully")
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'openai'")
    
    def _cache_key(self, prompt: str, completion: str) -> str:
        """Generate cache key from prompt + completion."""
        content = f"{prompt}|||{completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached judgment if exists."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _save_cache(self, key: str, result: dict):
        """Save judgment to cache."""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f)
    
    def _extract_user_request(self, prompt: str) -> str:
        """Extract user request from prompt."""
        # Look for user block
        match = re.search(r'user「([^」]+)」', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()[:500]  # Truncate for prompt efficiency
        # Fallback: look for im_start user
        match = re.search(r'<\|im_start\|>user\s*(.+?)(?:<\|im_end\|>|$)', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        return prompt[-500:]  # Last resort
    
    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except:
            pass
        
        # Try extracting from code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Try finding JSON object anywhere
        match = re.search(r'\{[^{}]*"coherence"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        return {}
    
    def _judge_openai(self, user_request: str, completion: str) -> dict:
        """Judge using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    user_request=user_request,
                    response=completion[:4000]
                )}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
        return json.loads(response.choices[0].message.content)
    
    def _judge_vllm(self, user_request: str, completion: str) -> dict:
        """Judge using local vLLM model."""
        # Build chat prompt
        prompt = f"""<|im_start|>system
{JUDGE_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{JUDGE_USER_TEMPLATE.format(user_request=user_request, response=completion[:4000])}<|im_end|>
<|im_start|>assistant
"""
        outputs = self._llm.generate([prompt], self._sampling_params)
        response_text = outputs[0].outputs[0].text
        return self._parse_json_response(response_text)
    
    def judge(self, prompt: str, completion: str) -> tuple[float, dict]:
        """
        Judge a completion's semantic quality.
        
        Returns:
            (score, breakdown) where score is 0-1 normalized and breakdown has individual scores.
        """
        cache_key = self._cache_key(prompt, completion)
        cached = self._get_cached(cache_key)
        if cached:
            self.cache_hits += 1
            return cached['score'], cached['breakdown']
        
        self.calls += 1
        user_request = self._extract_user_request(prompt)
        
        try:
            if self.backend == "openai":
                result = self._judge_openai(user_request, completion)
            else:
                result = self._judge_vllm(user_request, completion)
            
            # Normalize scores (0-10) to 0-1
            breakdown = {
                'coherence': result.get('coherence', 5) / 10,
                'relevance': result.get('relevance', 5) / 10,
                'reasoning': result.get('reasoning', 5) / 10,
                'completion': result.get('completion', 5) / 10,
                'quality': result.get('quality', 5) / 10,
            }
            
            # Weighted average (completion and relevance weighted higher)
            score = (
                breakdown['coherence'] * 0.15 +
                breakdown['relevance'] * 0.25 +
                breakdown['reasoning'] * 0.2 +
                breakdown['completion'] * 0.25 +
                breakdown['quality'] * 0.15
            )
            
            # Cache
            self._save_cache(cache_key, {'score': score, 'breakdown': breakdown})
            
            return score, breakdown
            
        except Exception as e:
            print(f"[Judge] Error: {e}")
            # Return neutral score on error
            return 0.5, {'error': str(e)}
    
    def judge_batch(self, prompts: list[str], completions: list[str]) -> list[tuple[float, dict]]:
        """Batch judge for efficiency with vLLM."""
        results = []
        uncached_indices = []
        uncached_requests = []
        
        # Check cache first
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            cache_key = self._cache_key(prompt, completion)
            cached = self._get_cached(cache_key)
            if cached:
                self.cache_hits += 1
                results.append((cached['score'], cached['breakdown']))
            else:
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                user_request = self._extract_user_request(prompt)
                uncached_requests.append((user_request, completion[:4000]))
        
        if not uncached_requests:
            return results
        
        self.calls += len(uncached_requests)
        
        if self.backend == "vllm" and self._llm is not None:
            # Batch process with vLLM
            batch_prompts = []
            for user_request, completion in uncached_requests:
                p = f"""<|im_start|>system
{JUDGE_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{JUDGE_USER_TEMPLATE.format(user_request=user_request, response=completion)}<|im_end|>
<|im_start|>assistant
"""
                batch_prompts.append(p)
            
            outputs = self._llm.generate(batch_prompts, self._sampling_params)
            
            for idx, (i, output) in enumerate(zip(uncached_indices, outputs)):
                try:
                    response_text = output.outputs[0].text
                    result = self._parse_json_response(response_text)
                    
                    breakdown = {
                        'coherence': result.get('coherence', 5) / 10,
                        'relevance': result.get('relevance', 5) / 10,
                        'reasoning': result.get('reasoning', 5) / 10,
                        'completion': result.get('completion', 5) / 10,
                        'quality': result.get('quality', 5) / 10,
                    }
                    
                    score = (
                        breakdown['coherence'] * 0.15 +
                        breakdown['relevance'] * 0.25 +
                        breakdown['reasoning'] * 0.2 +
                        breakdown['completion'] * 0.25 +
                        breakdown['quality'] * 0.15
                    )
                    
                    # Cache
                    cache_key = self._cache_key(prompts[i], completions[i])
                    self._save_cache(cache_key, {'score': score, 'breakdown': breakdown})
                    
                    results[i] = (score, breakdown)
                except Exception as e:
                    results[i] = (0.5, {'error': str(e)})
        else:
            # Fall back to sequential for OpenAI
            for idx, i in enumerate(uncached_indices):
                user_request, completion = uncached_requests[idx]
                score, breakdown = self.judge(prompts[i], completions[i])
                results[i] = (score, breakdown)
        
        return results
    
    def stats(self) -> dict:
        """Get judge statistics."""
        return {
            'calls': self.calls,
            'cache_hits': self.cache_hits,
            'cache_rate': self.cache_hits / max(1, self.calls + self.cache_hits),
            'backend': self.backend,
            'model': self.model,
        }


# =============================================================================
# Hybrid Reward Function
# =============================================================================

class HybridRewardFunction:
    """Combines deterministic verifiers with optional LLM judge."""
    
    def __init__(
        self,
        judge_rate: float = 0.0,  # Fraction of samples to judge (0 = disabled)
        judge_model: str = "Qwen/Qwen2.5-72B-Instruct",
        judge_backend: str = "vllm",  # "vllm" or "openai"
        judge_weight: float = 1.0,  # Weight of LLM score vs verifier score
        judge_gpus: int = 6,  # GPUs for judge model
        judge_dtype: str = "bfloat16",
        api_key: Optional[str] = None,
    ):
        self.verifier = TaskVerifier()
        self.judge_rate = judge_rate
        self.judge_weight = judge_weight
        self.judge = None
        
        if judge_rate > 0:
            self.judge = LLMJudge(
                model=judge_model,
                backend=judge_backend,
                api_key=api_key,
                tensor_parallel_size=judge_gpus,
                dtype=judge_dtype,
            )
    
    def __call__(
        self,
        completions: list[str],
        prompts: list[str],
        **kwargs
    ) -> list[float]:
        """Compute hybrid rewards for completions."""
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # Always compute deterministic reward
            verifier_score, verifier_breakdown = self.verifier.compute_reward(completion)
            
            # Optionally use LLM judge
            if self.judge and random.random() < self.judge_rate:
                judge_score, judge_breakdown = self.judge.judge(prompt, completion)
                
                # Scale judge score to similar range as verifier (-2 to 3 roughly)
                # Judge gives 0-1, scale to -1 to 2
                scaled_judge = (judge_score * 3) - 1
                
                # Combine scores
                combined = verifier_score + (scaled_judge * self.judge_weight)
                rewards.append(combined)
            else:
                rewards.append(verifier_score)
        
        return rewards
    
    def get_stats(self) -> dict:
        """Get reward function statistics."""
        stats = {'judge_enabled': self.judge is not None}
        if self.judge:
            stats.update(self.judge.stats())
        return stats


# Global reward function instance (set in main)
_reward_fn: Optional[HybridRewardFunction] = None


def reward_function_wrapper(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Wrapper for GRPO trainer compatibility."""
    global _reward_fn
    if _reward_fn is None:
        # Fallback to basic verifier if not initialized
        verifier = TaskVerifier()
        return [verifier.compute_reward(c)[0] for c in completions]
    return _reward_fn(completions, prompts, **kwargs)


# =============================================================================
# Dataset and Prompt Building
# =============================================================================

def load_prompts(data_path: str, max_samples: int = None) -> list[dict]:
    """Load prompts from processed traces (extract system + user only)."""
    prompts = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                trace = item.get("trace", "")
                
                # Extract system and user parts (before assistant)
                if "<|im_start|>assistant" in trace:
                    prompt_part = trace.split("<|im_start|>assistant")[0]
                    prompt_part += "<|im_start|>assistant\n"
                    prompts.append({"prompt": prompt_part})
            except:
                pass
    
    print(f"Loaded {len(prompts)} prompts from {data_path}")
    return prompts


# =============================================================================
# Main Training
# =============================================================================

@dataclass
class RLConfig:
    model_path: str = "./checkpoints/final"
    data_path: str = "traces.processed.jsonl"
    output_dir: str = "./checkpoints/rl"
    
    # GRPO config
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    
    # Generation
    max_new_tokens: int = 4096
    temperature: float = 0.8
    num_generations: int = 4  # Number of completions per prompt for GRPO
    
    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 10
    
    # LLM Judge
    judge_rate: float = 0.0  # Fraction of samples to judge (0 = disabled)
    judge_model: str = "Qwen/Qwen2.5-72B-Instruct"
    judge_backend: str = "vllm"  # "vllm" for local, "openai" for API
    judge_weight: float = 1.0  # Weight of LLM score vs verifier score
    judge_gpus: int = 6  # GPUs for judge model (when using vllm)
    judge_dtype: str = "bfloat16"  # "bfloat16", "float16", "auto"
    openai_api_key: Optional[str] = None
    
    max_samples: int = None  # Limit training samples


def main():
    global _reward_fn
    
    parser = argparse.ArgumentParser(
        description="GRPO training for TASK format with hybrid rewards"
    )
    parser.add_argument("--model", type=str, default="./checkpoints/final",
                        help="Path to base model")
    parser.add_argument("--data", type=str, default="traces.processed.jsonl",
                        help="Training data path")
    parser.add_argument("--output", type=str, default="./checkpoints/rl",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of training samples")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    
    # LLM Judge arguments
    parser.add_argument("--judge-rate", type=float, default=0.0,
                        help="Fraction of samples to evaluate with LLM judge (0=disabled, 0.1=10%%)")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model for LLM judge (HF model ID for vllm, or OpenAI model name)")
    parser.add_argument("--judge-backend", type=str, default="vllm", choices=["vllm", "openai"],
                        help="Backend for LLM judge: 'vllm' (local) or 'openai' (API)")
    parser.add_argument("--judge-weight", type=float, default=1.0,
                        help="Weight of LLM judge score relative to verifier")
    parser.add_argument("--judge-gpus", type=int, default=6,
                        help="Number of GPUs for judge model (vllm backend only)")
    parser.add_argument("--judge-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "auto"],
                        help="Data type for judge model")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key (for openai backend)")
    
    args = parser.parse_args()
    
    config = RLConfig(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        judge_rate=args.judge_rate,
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        judge_weight=args.judge_weight,
        judge_gpus=args.judge_gpus,
        judge_dtype=args.judge_dtype,
        openai_api_key=args.openai_api_key,
    )
    
    print("="*60)
    print("GRPO Training for TASK Format")
    print("="*60)
    print(f"Model: {config.model_path}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Num generations per prompt: {config.num_generations}")
    print("-"*60)
    if config.judge_rate > 0:
        print(f"LLM Judge: ENABLED")
        print(f"  Backend: {config.judge_backend}")
        print(f"  Model: {config.judge_model}")
        print(f"  Rate: {config.judge_rate*100:.0f}% of samples")
        print(f"  Weight: {config.judge_weight}")
        if config.judge_backend == "vllm":
            print(f"  GPUs: {config.judge_gpus}")
            print(f"  Dtype: {config.judge_dtype}")
    else:
        print(f"LLM Judge: DISABLED (verifier only)")
    print("="*60)
    
    # Initialize hybrid reward function
    print("\nInitializing reward function...")
    _reward_fn = HybridRewardFunction(
        judge_rate=config.judge_rate,
        judge_model=config.judge_model,
        judge_backend=config.judge_backend,
        judge_weight=config.judge_weight,
        judge_gpus=config.judge_gpus,
        judge_dtype=config.judge_dtype,
        api_key=config.openai_api_key,
    )
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    print("\nLoading prompts...")
    prompts = load_prompts(config.data_path, config.max_samples)
    dataset = Dataset.from_list(prompts)
    
    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        max_completion_length=config.max_new_tokens,
        num_generations=config.num_generations,
        temperature=config.temperature,
        report_to="tensorboard",
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function_wrapper,
    )
    
    # Train
    print("\nStarting GRPO training...")
    try:
        trainer.train()
    finally:
        # Print judge stats if used
        if _reward_fn:
            stats = _reward_fn.get_stats()
            if stats.get('judge_enabled'):
                print("\n" + "="*60)
                print("LLM Judge Statistics")
                print("="*60)
                print(f"Total API calls: {stats.get('calls', 0)}")
                print(f"Cache hits: {stats.get('cache_hits', 0)}")
                print(f"Cache rate: {stats.get('cache_rate', 0)*100:.1f}%")
    
    # Save
    print(f"\nSaving model to {config.output_dir}/final...")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    print("Done!")


if __name__ == "__main__":
    main()

