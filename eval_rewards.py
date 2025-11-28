#!/usr/bin/env python3
"""
Standalone evaluation script for TASK format rewards.

Tests deterministic verifiers and optional LLM judge on model outputs.
Useful for debugging reward hacking and tuning reward weights.

Usage:
    # Evaluate a single output
    python eval_rewards.py --text "plan { ... } response「...」"
    
    # Evaluate from file
    python eval_rewards.py --file output.txt
    
    # With LLM judge
    python eval_rewards.py --file output.txt --judge
    
    # Interactive mode
    python eval_rewards.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

# Import from rl_train
from rl_train import TaskVerifier, LLMJudge, OPENAI_AVAILABLE


def print_verifier_results(trace: str):
    """Print detailed verifier results."""
    verifier = TaskVerifier()
    
    print("\n" + "="*60)
    print("DETERMINISTIC VERIFIER RESULTS")
    print("="*60)
    
    # Individual checks
    checks = [
        ("Brackets Balanced", verifier.check_brackets_balanced(trace)),
        ("Has Response", verifier.check_has_response(trace)),
        ("Has Plan", verifier.check_has_plan(trace)),
        ("Todos Start at 1", verifier.check_todos_start_at_one(trace)),
        ("Structure Order", verifier.check_structure_order(trace)),
    ]
    
    for name, (passed, msg) in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {msg}")
    
    # Todo satisfaction (special handling)
    todos_ok, msg, defined, satisfied = verifier.check_todos_satisfied(trace)
    status = "✓" if todos_ok else "✗"
    print(f"  {status} Todos Satisfied: {msg}")
    if defined:
        print(f"      Defined: {sorted(defined)}")
        print(f"      Satisfied: {sorted(satisfied)}")
        unsatisfied = defined - satisfied
        if unsatisfied:
            print(f"      Missing: {sorted(unsatisfied)}")
    
    # Tag references
    refs_ok, msg, defined_tags, referenced_tags = verifier.check_refs_valid(trace)
    status = "✓" if refs_ok else "✗"
    print(f"  {status} References Valid: {msg}")
    if defined_tags or referenced_tags:
        print(f"      Defined tags: {sorted(defined_tags)[:10]}{'...' if len(defined_tags) > 10 else ''}")
        orphans = referenced_tags - defined_tags
        orphans = {t for t in orphans if not t.startswith(('usr', 'sys'))}
        if orphans:
            print(f"      Orphan refs: {sorted(orphans)}")
    
    # Total score
    total, breakdown = verifier.compute_reward(trace)
    print("-"*60)
    print(f"  TOTAL SCORE: {total:.2f}")
    print("  Breakdown:")
    for key, val in breakdown.items():
        print(f"    {key}: {val:+.2f}")


def print_judge_results(prompt: str, trace: str, model: str = "gpt-4o-mini"):
    """Print LLM judge results."""
    if not OPENAI_AVAILABLE:
        print("\n[Judge] OpenAI not available. Install with: pip install openai")
        return
    
    print("\n" + "="*60)
    print(f"LLM JUDGE RESULTS ({model})")
    print("="*60)
    
    try:
        judge = LLMJudge(model=model)
        score, breakdown = judge.judge(prompt, trace)
        
        print(f"  Overall Score: {score:.2f} (0-1 scale)")
        print("  Breakdown:")
        for key, val in breakdown.items():
            if key != 'error':
                print(f"    {key}: {val:.2f}")
        
        if 'error' in breakdown:
            print(f"  Error: {breakdown['error']}")
            
    except Exception as e:
        print(f"  Error: {e}")


def interactive_mode(use_judge: bool = False, judge_model: str = "gpt-4o-mini"):
    """Interactive evaluation mode."""
    print("Interactive TASK Evaluation")
    print("="*60)
    print("Enter TASK format text (type 'END' on a new line to evaluate)")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        print("\n[Enter trace]:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() == 'quit':
                return
            if line.strip() == 'END':
                break
            lines.append(line)
        
        trace = '\n'.join(lines)
        if not trace.strip():
            continue
        
        print_verifier_results(trace)
        
        if use_judge:
            # For interactive, use trace as prompt too
            print_judge_results(trace, trace, judge_model)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TASK format rewards")
    parser.add_argument("--text", type=str, help="TASK format text to evaluate")
    parser.add_argument("--file", type=str, help="File containing TASK format text")
    parser.add_argument("--prompt-file", type=str, help="File containing the prompt (for judge)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="LLM judge model")
    parser.add_argument("--jsonl", type=str, help="Evaluate traces from JSONL file")
    parser.add_argument("--limit", type=int, default=10, help="Limit for JSONL evaluation")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.judge, args.judge_model)
        return
    
    if args.jsonl:
        # Batch evaluation of JSONL
        print(f"Evaluating {args.jsonl} (limit: {args.limit})")
        print("="*60)
        
        verifier = TaskVerifier()
        scores = []
        
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                try:
                    item = json.loads(line)
                    trace = item.get("trace", "")
                    score, _ = verifier.compute_reward(trace)
                    scores.append(score)
                    
                    # Show per-item
                    status = "✓" if score > 1.5 else ("~" if score > 0 else "✗")
                    print(f"  {status} Trace {i+1}: {score:.2f}")
                except Exception as e:
                    print(f"  ✗ Trace {i+1}: Error - {e}")
        
        if scores:
            print("-"*60)
            print(f"  Average: {sum(scores)/len(scores):.2f}")
            print(f"  Min: {min(scores):.2f}")
            print(f"  Max: {max(scores):.2f}")
            print(f"  Passing (>1.5): {sum(1 for s in scores if s > 1.5)}/{len(scores)}")
        return
    
    # Get trace text
    if args.text:
        trace = args.text
    elif args.file:
        trace = Path(args.file).read_text()
    else:
        print("Reading from stdin... (Ctrl+D to end)")
        trace = sys.stdin.read()
    
    if not trace.strip():
        print("No input provided.")
        return
    
    # Get prompt for judge
    prompt = ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    else:
        prompt = trace  # Use trace as prompt if not specified
    
    # Evaluate
    print_verifier_results(trace)
    
    if args.judge:
        print_judge_results(prompt, trace, args.judge_model)


if __name__ == "__main__":
    main()

