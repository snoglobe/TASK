#!/usr/bin/env python3
"""
Generate training prompts for RL training.

Runs independently of accelerate to avoid NCCL timeout issues.
Output can be used directly with rl_train.py --data.

Usage:
    # Generate 1000 prompts using vLLM server
    python generate_prompts.py --count 1000 --output prompts.jsonl \
        --backend vllm_server --url http://localhost:8000/v1
    
    # Then run RL training with generated prompts
    accelerate launch rl_train.py --data prompts.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# Prompt Generation System
# =============================================================================

DOMAINS = ['coding', 'research', 'writing', 'math', 'planning', 'general', 'data_analysis']
COMPLEXITIES = ['simple', 'medium', 'complex']

PROMPT_GEN_SYSTEM = """You generate training scenarios for an AI assistant that uses a structured reasoning format.

Generate a scenario with:
1. A system prompt describing the assistant's role
2. Optionally 0-3 tools the assistant can use
3. A user request

## CRITICAL: Tool Guidelines

Tools must be DETERMINISTIC, VERIFIABLE operations - NOT hidden LLM calls.

✅ ALLOWED tools (real operations):
- File ops: read_file, write_file, list_dir, delete_file
- Code: run_code, run_tests, execute_shell, compile
- Data: query_database, execute_sql, fetch_url, call_api
- Search: search_web, search_codebase, grep
- System: get_time, get_weather, get_env_var
- Explicit LLM: call_llm (when you explicitly need another LLM)

❌ FORBIDDEN tools (hidden LLM magic):
- critique_writing, analyze_sentiment, summarize_text
- improve_code, refactor_function, fix_bugs  
- generate_ideas, brainstorm, suggest_alternatives
- explain_concept, translate_text, paraphrase
- review_code, assess_quality, evaluate_answer

If the task needs reasoning/analysis, the model should use its own thinking, NOT call a tool.
If you genuinely need an external LLM call, use `call_llm` explicitly.

## Parameter Types (MUST be valid JSON Schema types):
- string: text values
- number: floating point numbers
- integer: whole numbers
- boolean: true/false
- array: lists of values
- object: nested objects

Generate diverse, realistic scenarios. Vary domains, complexity, and whether tools are needed."""


def format_tools_task(tools: list[dict]) -> str:
    """Format tools into TASK format with proper JSON Schema types."""
    if not tools:
        return ""
    
    VALID_TYPES = {'string', 'number', 'integer', 'boolean', 'array', 'object'}
    
    result = []
    for tool in tools:
        name = tool.get('name', 'unknown')
        desc = tool.get('description', '')
        params = tool.get('params', {})
        
        params_str = ""
        if params:
            param_items = []
            for pname, ptype in params.items():
                if ptype not in VALID_TYPES:
                    ptype = 'string'
                param_items.append(f"{pname} ↦ {{ type ↦ {ptype} }}")
            params_str = f" • params ↦ {{ {' • '.join(param_items)} }}"
        
        tool_str = f'tool {{ name ↦ {name} • description ↦ "{desc}"{params_str} }}'
        result.append(tool_str)
    
    return '\n'.join(result)


def format_prompt(scenario: dict) -> Optional[str]:
    """Format a scenario into a TASK format prompt."""
    system = scenario.get('system_prompt', '')
    tools = scenario.get('tools', [])
    user_request = scenario.get('user_request', '')
    
    if not system or not user_request:
        return None
    
    parts = [f'system「{system}」🏷 sys1']
    
    if tools:
        tools_str = format_tools_task(tools)
        if tools_str:
            parts.append(tools_str)
    
    parts.append(f'user「{user_request}」🏷 usr1')
    
    return '\n\n'.join(parts)


def generate_batch(client, model: str, batch_size: int = 20) -> list[str]:
    """Generate multiple scenarios in a single API call."""
    
    batch_prompt = f"""Generate {batch_size} diverse training scenarios. Return a JSON array.

Requirements for ALL scenarios:
- Domain: vary between {', '.join(DOMAINS)}
- Include tools: vary (some with 1-3 tools, some without)
- Complexity: vary between simple, medium, complex
- Make each scenario DIFFERENT from the others

Return ONLY a JSON array like:
[
  {{"system_prompt": "...", "tools": [{{"name": "tool_name", "description": "...", "params": {{"arg1": "string"}}}}], "user_request": "...", "domain": "...", "complexity": "..."}},
  {{"system_prompt": "...", "tools": [], "user_request": "...", "domain": "...", "complexity": "..."}},
  ...
]

{batch_size} scenarios, JSON array only:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_GEN_SYSTEM},
                {"role": "user", "content": batch_prompt}
            ],
            temperature=0.9,
            max_tokens=batch_size * 400,
        )
    except Exception as e:
        print(f"  API error: {e}")
        return []
    
    response_text = response.choices[0].message.content if response.choices else None
    if not response_text:
        print(f"  Empty response")
        return []
    
    # Parse JSON array
    try:
        scenarios = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', response_text)
        if match:
            try:
                scenarios = json.loads(match.group(0))
            except:
                print(f"  Failed to parse array")
                return []
        else:
            print(f"  No array found in response")
            return []
    
    if not isinstance(scenarios, list):
        print(f"  Response is not an array")
        return []
    
    # Format each scenario
    results = []
    for scenario in scenarios:
        try:
            if isinstance(scenario, dict) and 'system_prompt' in scenario and 'user_request' in scenario:
                formatted = format_prompt(scenario)
                if formatted:
                    results.append(formatted)
        except:
            pass
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate training prompts for RL")
    parser.add_argument("--count", type=int, default=1000,
                        help="Number of prompts to generate")
    parser.add_argument("--output", type=str, default="prompts.jsonl",
                        help="Output file (JSONL format)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of scenarios per API call")
    parser.add_argument("--backend", type=str, default="vllm_server",
                        choices=["vllm_server", "openai"],
                        help="Backend: vllm_server or openai")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1",
                        help="URL for vLLM server")
    parser.add_argument("--model", type=str, default="",
                        help="Model name (auto-detected for vllm_server)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for OpenAI backend")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    args = parser.parse_args()
    
    if not OPENAI_AVAILABLE:
        print("Error: openai package required. Install with: pip install openai")
        return 1
    
    # Create client
    if args.backend == "vllm_server":
        print(f"Connecting to vLLM server at {args.url}...")
        client = OpenAI(base_url=args.url, api_key="not-needed")
        
        # Try to get model name from server
        if not args.model:
            try:
                models = client.models.list()
                if models.data:
                    args.model = models.data[0].id
                    print(f"  Detected model: {args.model}")
            except:
                print("  Warning: Could not detect model, using 'default'")
                args.model = "default"
    else:
        print("Using OpenAI API...")
        client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()
        if not args.model:
            args.model = "gpt-4o-mini"
    
    print(f"Model: {args.model}")
    print(f"Generating {args.count} prompts in batches of {args.batch_size}...")
    print()
    
    # Check existing file
    existing_count = 0
    if args.append and Path(args.output).exists():
        with open(args.output) as f:
            existing_count = sum(1 for _ in f)
        print(f"Appending to {args.output} ({existing_count} existing prompts)")
    
    # Generate in batches
    results = []
    num_batches = (args.count + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        remaining = args.count - len(results)
        current_batch_size = min(args.batch_size, remaining)
        
        print(f"[Batch {batch_idx+1}/{num_batches}] Requesting {current_batch_size} scenarios...")
        
        batch_results = generate_batch(client, args.model, current_batch_size)
        
        for prompt in batch_results:
            results.append(prompt)
            preview = prompt[:80].replace('\n', ' ')
            print(f"  [{len(results)}/{args.count}] {preview}...")
        
        if len(batch_results) < current_batch_size:
            print(f"  Warning: Only got {len(batch_results)}/{current_batch_size} valid scenarios")
        
        if len(results) >= args.count:
            break
    
    # Write output
    mode = 'a' if args.append else 'w'
    with open(args.output, mode) as f:
        for prompt in results[:args.count]:
            # Format as JSONL with "prompt" key for rl_train.py compatibility
            f.write(json.dumps({"prompt": prompt}) + '\n')
    
    total = existing_count + len(results[:args.count])
    print()
    print(f"Done! Wrote {len(results[:args.count])} prompts to {args.output}")
    print(f"Total prompts in file: {total}")
    print()
    print(f"To use with RL training:")
    print(f"  accelerate launch rl_train.py --data {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

