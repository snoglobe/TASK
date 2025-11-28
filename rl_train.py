#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for TASK format.

Uses deterministic verifiers as reward signals:
- Bracket balance
- Todo satisfaction
- Tag reference validity
- Response completion
- Structural correctness

Usage:
    accelerate launch --config_file accelerate_config.yaml rl_train.py \
        --model ./checkpoints/final \
        --output ./checkpoints/rl
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


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


def reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Compute rewards for completions using TASK verifier."""
    verifier = TaskVerifier()
    rewards = []
    
    for completion in completions:
        reward, breakdown = verifier.compute_reward(completion)
        rewards.append(reward)
    
    return rewards


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
    
    max_samples: int = None  # Limit training samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/final")
    parser.add_argument("--data", type=str, default="traces.processed.jsonl")
    parser.add_argument("--output", type=str, default="./checkpoints/rl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
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
    )
    
    print("="*60)
    print("GRPO Training for TASK Format")
    print("="*60)
    print(f"Model: {config.model_path}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Num generations per prompt: {config.num_generations}")
    print("="*60)
    
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
        reward_funcs=reward_function,
    )
    
    # Train
    print("\nStarting GRPO training...")
    trainer.train()
    
    # Save
    print(f"\nSaving model to {config.output_dir}/final...")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    print("Done!")


if __name__ == "__main__":
    main()

