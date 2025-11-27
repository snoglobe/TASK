#!/usr/bin/env python3
"""
Count tokens in TASK traces JSONL file.
"""

import argparse
import json
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def estimate_tokens_simple(text: str) -> int:
    """Simple estimation: ~4 chars per token."""
    return len(text) // 4


def estimate_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """Accurate token count using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def analyze_traces(file_path: Path, use_tiktoken: bool = True):
    """Analyze token counts in traces JSONL."""
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return
    
    print(f"Analyzing {file_path}...")
    
    if use_tiktoken and not HAS_TIKTOKEN:
        print("Warning: tiktoken not installed, using simple estimation")
        print("Install with: pip install tiktoken")
        use_tiktoken = False
    
    traces = []
    total_tokens = 0
    
    with open(file_path) as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                trace_text = item.get("trace", "")
                
                if use_tiktoken:
                    tokens = estimate_tokens_tiktoken(trace_text)
                else:
                    tokens = estimate_tokens_simple(trace_text)
                
                traces.append({
                    "index": i,
                    "tokens": tokens,
                    "chars": len(trace_text)
                })
                total_tokens += tokens
            except Exception as e:
                print(f"  Error on line {i + 1}: {e}")
    
    if not traces:
        print("No traces found")
        return
    
    # Stats
    trace_tokens = [t["tokens"] for t in traces]
    min_tokens = min(trace_tokens)
    max_tokens = max(trace_tokens)
    avg_tokens = sum(trace_tokens) // len(trace_tokens)
    median_tokens = sorted(trace_tokens)[len(trace_tokens) // 2]
    
    print(f"\n{'='*60}")
    print(f"Total traces: {len(traces):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total chars:  {sum(t['chars'] for t in traces):,}")
    print(f"\nPer-trace stats:")
    print(f"  Min:     {min_tokens:,} tokens")
    print(f"  Max:     {max_tokens:,} tokens")
    print(f"  Average: {avg_tokens:,} tokens")
    print(f"  Median:  {median_tokens:,} tokens")
    
    # Training estimates (rough)
    print(f"\n{'='*60}")
    print("Training estimates (very rough):")
    print(f"  GPT-3.5-turbo fine-tune cost: ~${total_tokens * 0.008 / 1000:.2f}")
    print(f"  GPT-4o-mini fine-tune cost:   ~${total_tokens * 0.0003 / 1000:.2f}")
    print(f"  Base model SFT (1 epoch):     ~{total_tokens:,} tokens")
    
    # Distribution
    print(f"\n{'='*60}")
    print("Token distribution:")
    buckets = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
    bucket_names = ["0-100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K+"]
    counts = [0] * len(bucket_names)
    
    for t in trace_tokens:
        for i, (low, high) in enumerate(zip(buckets[:-1], buckets[1:])):
            if low <= t < high:
                counts[i] += 1
                break
    
    for name, count in zip(bucket_names, counts):
        pct = 100 * count / len(traces)
        bar = "█" * int(pct / 2)
        print(f"  {name:>10}: {count:4} ({pct:5.1f}%) {bar}")
    
    print(f"{'='*60}\n")
    
    # Largest traces
    print("Top 10 largest traces:")
    largest = sorted(traces, key=lambda x: x["tokens"], reverse=True)[:10]
    for t in largest:
        print(f"  trace_{t['index']:04d}: {t['tokens']:,} tokens ({t['chars']:,} chars)")


def main():
    parser = argparse.ArgumentParser(description="Count tokens in TASK traces")
    parser.add_argument(
        "file",
        nargs="?",
        default="traces.jsonl",
        help="Path to traces JSONL file (default: traces.jsonl)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple estimation instead of tiktoken"
    )
    
    args = parser.parse_args()
    file_path = Path(args.file)
    
    analyze_traces(file_path, use_tiktoken=not args.simple)


if __name__ == "__main__":
    main()
