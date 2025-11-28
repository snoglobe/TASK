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
    
    # Detailed parse output
    python eval_rewards.py --file output.txt --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import from rl_train
from rl_train import TaskVerifier, TaskParser, LLMJudge, OPENAI_AVAILABLE


def print_verifier_results(trace: str, verbose: bool = False):
    """Print detailed verifier results using the parser."""
    verifier = TaskVerifier()
    
    print("\n" + "="*60)
    print("TASK FORMAT PARSER RESULTS")
    print("="*60)
    
    # Get detailed results
    details = verifier.detailed_check(trace)
    
    # Blocks found
    print(f"\n  Blocks parsed: {len(details['blocks'])}")
    for block_type, size in details['blocks']:
        print(f"    • {block_type} ({size} chars)")
    
    # Todos
    print(f"\n  Todos:")
    defined = details['defined_todos']
    satisfied = details['satisfied_todos']
    if defined:
        print(f"    Defined: {sorted(defined)}")
        print(f"    Satisfied: {sorted(satisfied)}")
        unsatisfied = set(defined) - set(satisfied)
        if unsatisfied:
            print(f"    ✗ Missing: {sorted(unsatisfied)}")
        else:
            print(f"    ✓ All todos satisfied")
    else:
        print(f"    (no todos defined)")
    
    # Tags
    print(f"\n  Tags:")
    defined_tags = set(details['defined_tags'])
    referenced_tags = set(details['referenced_tags'])
    print(f"    Defined: {len(defined_tags)}")
    if verbose and defined_tags:
        print(f"      {sorted(defined_tags)[:15]}{'...' if len(defined_tags) > 15 else ''}")
    print(f"    Referenced: {len(referenced_tags)}")
    
    # Check for orphan references
    filtered_refs = {t for t in referenced_tags if not re.match(r'^(usr|sys)\d+$', t)}
    orphans = filtered_refs - defined_tags
    if orphans:
        print(f"    ✗ Orphan refs: {sorted(orphans)}")
    else:
        print(f"    ✓ All references valid")
    
    # Tools
    print(f"\n  Tools:")
    defined_tools = set(details.get('defined_tools', []))
    called_tools = set(details.get('called_tools', []))
    call_ids = set(details.get('call_ids', []))
    result_ids = set(details.get('result_ids', []))
    
    print(f"    Defined: {sorted(defined_tools) if defined_tools else '(none)'}")
    print(f"    Called: {sorted(called_tools) if called_tools else '(none)'}")
    
    # Check for hallucinated tools
    if called_tools:
        if defined_tools:
            hallucinated = called_tools - defined_tools
            if hallucinated:
                print(f"    ✗ HALLUCINATED tools: {sorted(hallucinated)}")
            else:
                print(f"    ✓ All called tools are defined")
        else:
            print(f"    ✗ HALLUCINATED: tools called but none defined!")
    
    # Check call/result matching
    if call_ids or result_ids:
        print(f"    Call IDs: {sorted(call_ids) if call_ids else '(none)'}")
        print(f"    Result IDs: {sorted(result_ids) if result_ids else '(none)'}")
        orphan_results = result_ids - call_ids
        missing_results = call_ids - result_ids
        if orphan_results:
            print(f"    ✗ Orphan results (no matching call): {sorted(orphan_results)}")
        if missing_results:
            print(f"    ⚠ Calls without results: {sorted(missing_results)}")
        if not orphan_results and not missing_results and call_ids:
            print(f"    ✓ All calls have matching results")
    
    # Errors and warnings
    if details['errors']:
        print(f"\n  ✗ Errors ({len(details['errors'])}):")
        for err in details['errors'][:5]:
            print(f"    • {err}")
        if len(details['errors']) > 5:
            print(f"    ... and {len(details['errors']) - 5} more")
    
    if details['warnings']:
        print(f"\n  ⚠ Warnings ({len(details['warnings'])}):")
        for warn in details['warnings'][:5]:
            print(f"    • {warn}")
        if len(details['warnings']) > 5:
            print(f"    ... and {len(details['warnings']) - 5} more")
    
    # Reward breakdown
    print("\n" + "-"*60)
    print(f"  TOTAL SCORE: {details['score']:.2f}")
    print("  Reward breakdown:")
    for key, val in details['rewards'].items():
        status = "✓" if val > 0 else ("~" if val == 0 else "✗")
        print(f"    {status} {key}: {val:+.2f}")


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


def interactive_mode(use_judge: bool = False, judge_model: str = "gpt-4o-mini", verbose: bool = False):
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
        
        print_verifier_results(trace, verbose)
        
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output with more details")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.judge, args.judge_model, args.verbose)
        return
    
    if args.jsonl:
        # Batch evaluation of JSONL
        print(f"Evaluating {args.jsonl} (limit: {args.limit})")
        print("="*60)
        
        verifier = TaskVerifier()
        scores = []
        parse_errors = 0
        
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                try:
                    item = json.loads(line)
                    trace = item.get("trace", "")
                    score, rewards = verifier.compute_reward(trace)
                    scores.append(score)
                    
                    # Check for parse errors
                    if rewards.get('parse_valid', 1.0) < 0:
                        parse_errors += 1
                    
                    # Show per-item
                    status = "✓" if score > 1.5 else ("~" if score > 0 else "✗")
                    print(f"  {status} Trace {i+1}: {score:.2f}")
                    
                    if args.verbose and score < 1.0:
                        for key, val in rewards.items():
                            if val < 0:
                                print(f"      {key}: {val:+.2f}")
                                
                except Exception as e:
                    print(f"  ✗ Trace {i+1}: Error - {e}")
        
        if scores:
            print("-"*60)
            print(f"  Total evaluated: {len(scores)}")
            print(f"  Average score: {sum(scores)/len(scores):.2f}")
            print(f"  Min: {min(scores):.2f}")
            print(f"  Max: {max(scores):.2f}")
            print(f"  Parse errors: {parse_errors}/{len(scores)}")
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
    print_verifier_results(trace, args.verbose)
    
    if args.judge:
        print_judge_results(prompt, trace, args.judge_model)


if __name__ == "__main__":
    main()

