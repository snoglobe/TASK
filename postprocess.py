#!/usr/bin/env python3
"""
Post-processor for TASK format traces.
- Merges sequential think blocks into single blocks
- Adds special tokens for training
"""

import argparse
import json
import re
from pathlib import Path


# Special tokens for different models
QWEN3_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "system": "system",
    "user": "user", 
    "assistant": "assistant",
}

LLAMA3_TOKENS = {
    "bos": "<|begin_of_text|>",
    "eos": "<|end_of_text|>",
    "eot": "<|eot_id|>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def collect_act_block(lines: list[str], start_idx: int) -> tuple[list[str], str, int]:
    """Collect a full act block and return (lines, block_text, next_idx)."""
    block_lines = [lines[start_idx]]
    brace_count = lines[start_idx].count('{') - lines[start_idx].count('}')
    j = start_idx + 1
    
    while j < len(lines) and brace_count > 0:
        block_lines.append(lines[j])
        brace_count += lines[j].count('{') - lines[j].count('}')
        j += 1
    
    return block_lines, '\n'.join(block_lines), j


def merge_sequential_acts(trace: str) -> str:
    """Merge sequential act blocks:
    
    1. Sequential think-only blocks → merge into array format with operators preserved
    2. Think block followed by call block → merge into single act with both fields
    """
    
    lines = trace.split('\n')
    result_lines = []
    i = 0
    
    # Pattern to extract think content and trailing operators
    # Handles both "..." quotes and 「...」 brackets
    think_extract_quoted = re.compile(r'think\s*↦\s*"([^"]*)"(.*?)(?=\s*\})', re.DOTALL)
    think_extract_bracket = re.compile(r'think\s*↦\s*「([^」]*)」(.*?)(?=\s*\})', re.DOTALL)
    # Pattern to extract call block content (everything between call ↦ { and the matching })
    call_extract = re.compile(r'call\s*↦\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})(.*?)(?=\s*\})', re.DOTALL)
    # Operators pattern - matches ⊨, 🏷, ※, 𝑝 and their values
    has_operators = re.compile(r'[⊨🏷※𝑝]')
    
    def extract_think(block_text):
        """Extract think content and operators from a block."""
        match = think_extract_quoted.search(block_text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        match = think_extract_bracket.search(block_text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this starts an act block
        if line.strip().startswith('act {'):
            block_lines, block, j = collect_act_block(lines, i)
            
            # Classify the block
            has_think = 'think ↦' in block
            has_call = 'call ↦' in block
            is_think_only = has_think and not has_call
            is_call_only = has_call and not has_think
            
            if is_think_only:
                # Collect all sequential think blocks and check for trailing call block
                think_items = []  # List of (text, operators_string)
                trailing_call = None  # (call_content, call_operators)
                
                while True:
                    # Extract think content and operators from current block
                    text, ops = extract_think(block)
                    if text is not None:
                        think_items.append((text, ops if ops else ''))
                    
                    # Check next block
                    next_i = j
                    while next_i < len(lines) and not lines[next_i].strip():
                        next_i += 1
                    
                    if next_i >= len(lines):
                        break
                    
                    if not lines[next_i].strip().startswith('act {'):
                        break
                    
                    # Collect next block
                    next_block_lines, next_block, k = collect_act_block(lines, next_i)
                    
                    next_has_think = 'think ↦' in next_block
                    next_has_call = 'call ↦' in next_block
                    
                    if next_has_think and not next_has_call:
                        # Another think-only block - continue collecting
                        block = next_block
                        j = k
                    elif next_has_call and not next_has_think:
                        # Call-only block - merge with thinks
                        call_match = call_extract.search(next_block)
                        if call_match:
                            trailing_call = (call_match.group(1), call_match.group(2).strip())
                        j = k
                        break
                    else:
                        # Mixed or other - don't merge
                        break
                
                # Build merged block
                if len(think_items) > 1 or trailing_call:
                    # Need to merge something
                    
                    # First, build the think part
                    if len(think_items) == 1:
                        text, ops = think_items[0]
                        think_part = f'think ↦ 「{text}」{" " + ops if ops else ""}'
                    else:
                        # Group: concatenate texts without operators onto next text with operators
                        merged_items = []
                        pending_texts = []
                        
                        for text, ops in think_items:
                            pending_texts.append(text)
                            if has_operators.search(ops):
                                combined = ' '.join(pending_texts)
                                merged_items.append((combined, ops))
                                pending_texts = []
                        
                        # Handle remaining texts
                        if pending_texts:
                            if merged_items:
                                last_text, last_ops = merged_items[-1]
                                merged_items[-1] = (last_text + ' ' + ' '.join(pending_texts), last_ops)
                            else:
                                merged_items.append((' '.join(pending_texts), ''))
                        
                        if len(merged_items) == 1:
                            text, ops = merged_items[0]
                            think_part = f'think ↦ 「{text}」{" " + ops if ops else ""}'
                        else:
                            array_items = []
                            for text, ops in merged_items:
                                item = f'「{text}」'
                                if ops:
                                    item += f' {ops}'
                                array_items.append(item)
                            array_str = ' •\n        '.join(array_items)
                            think_part = f'think ↦ [\n        {array_str}\n    ]'
                    
                    # Build final block
                    if trailing_call:
                        call_content, call_ops = trailing_call
                        call_part = f'call ↦ {call_content}{" " + call_ops if call_ops else ""}'
                        merged_block = f'act {{\n    {think_part} •\n    {call_part}\n}}'
                    else:
                        merged_block = f'act {{\n    {think_part}\n}}'
                    
                    result_lines.append(merged_block)
                else:
                    result_lines.extend(block_lines)
                
                i = j
            else:
                result_lines.extend(block_lines)
                i = j
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)


def collect_bracket_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """Collect a block delimited by 「...」 that may span multiple lines."""
    block_lines = [lines[start_idx]]
    # Count open brackets (「) minus close brackets (」)
    bracket_count = lines[start_idx].count('「') - lines[start_idx].count('」')
    j = start_idx + 1
    while j < len(lines) and bracket_count > 0:
        block_lines.append(lines[j])
        bracket_count += lines[j].count('「') - lines[j].count('」')
        j += 1
    return block_lines, j


def collect_brace_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """Collect a block delimited by { ... } that may span multiple lines."""
    block_lines = [lines[start_idx]]
    brace_count = lines[start_idx].count('{') - lines[start_idx].count('}')
    j = start_idx + 1
    while j < len(lines) and brace_count > 0:
        block_lines.append(lines[j])
        brace_count += lines[j].count('{') - lines[j].count('}')
        j += 1
    return block_lines, j


def add_special_tokens_qwen3(trace: str) -> str:
    """Add Qwen3 special tokens for training.
    
    Structure:
    - <|im_start|>system\n{system msg + tool defs}<|im_end|>
    - <|im_start|>user\n{user msg}<|im_end|>
    - <|im_start|>assistant\n{plan + acts + results + response}<|im_end|>
    - (repeat user/assistant for multi-turn)
    """
    T = QWEN3_TOKENS
    lines = trace.split('\n')
    
    # Collect system content (system message + tool definitions)
    system_content = []
    user_messages = []  # List of (message_lines)
    assistant_turns = []
    current_assistant = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # System message (may span multiple lines with 「...」)
        if stripped.startswith('system「'):
            block_lines, i = collect_bracket_block(lines, i)
            system_content.extend(block_lines)
            continue
        
        # Tool definitions - part of system
        if stripped.startswith('tool {'):
            block_lines, i = collect_brace_block(lines, i)
            system_content.extend(block_lines)
            continue
        
        # User message - starts a new turn (may span multiple lines)
        if stripped.startswith('user「'):
            # Save any pending assistant content
            if current_assistant:
                assistant_turns.append('\n'.join(current_assistant))
                current_assistant = []
            block_lines, i = collect_bracket_block(lines, i)
            user_messages.append('\n'.join(block_lines))
            continue
        
        # Plan/act/result blocks (brace-delimited)
        if stripped.startswith(('plan {', 'act {', 'result {')):
            block_lines, i = collect_brace_block(lines, i)
            current_assistant.extend(block_lines)
            continue
        
        # Response block (bracket-delimited, may span multiple lines)
        if stripped.startswith('response「'):
            block_lines, i = collect_bracket_block(lines, i)
            current_assistant.extend(block_lines)
            continue
        
        # Preserve non-empty lines in current context
        if stripped:
            if current_assistant:
                current_assistant.append(line)
            elif system_content:
                system_content.append(line)
        elif current_assistant:
            # Preserve blank lines within assistant content
            current_assistant.append('')
        
        i += 1
    
    # Save final assistant content
    if current_assistant:
        assistant_turns.append('\n'.join(current_assistant))
    
    # Build output
    result = []
    
    # System turn
    if system_content:
        result.append(f"{T['im_start']}{T['system']}")
        result.extend(system_content)
        result.append(T['im_end'])
        result.append('')
    
    # Interleave user/assistant turns
    for idx, user_msg in enumerate(user_messages):
        # User turn
        result.append(f"{T['im_start']}{T['user']}")
        result.append(user_msg)
        result.append(T['im_end'])
        result.append('')
        
        # Assistant turn
        if idx < len(assistant_turns):
            result.append(f"{T['im_start']}{T['assistant']}")
            result.append(assistant_turns[idx])
            result.append(T['im_end'])
            result.append('')
    
    return '\n'.join(result)


def add_special_tokens_llama3(trace: str) -> str:
    """Add Llama3 special tokens for training (granular turn markers)."""
    T = LLAMA3_TOKENS
    lines = trace.split('\n')
    result = [T["bos"]]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith('system「'):
            result.append(T["system"])
            block_lines, i = collect_bracket_block(lines, i)
            result.extend(block_lines)
            result.append(T["eot"])
            continue
        elif stripped.startswith('tool {'):
            block_lines, i = collect_brace_block(lines, i)
            result.extend(block_lines)
            continue
        elif stripped.startswith('user「'):
            result.append(T["user"])
            block_lines, i = collect_bracket_block(lines, i)
            result.extend(block_lines)
            result.append(T["eot"])
            continue
        elif stripped.startswith(('plan {', 'act {', 'result {')):
            result.append(T["assistant"])
            block_lines, i = collect_brace_block(lines, i)
            result.extend(block_lines)
            result.append(T["eot"])
            continue
        elif stripped.startswith('response「'):
            result.append(T["assistant"])
            block_lines, i = collect_bracket_block(lines, i)
            result.extend(block_lines)
            result.append(T["eot"])
            continue
        elif stripped:
            result.append(line)
        
        i += 1
    
    result.append(T["eos"])
    return '\n'.join(result)


def add_special_tokens(trace: str, token_style: str = "qwen3") -> str:
    """Add special tokens for training."""
    if token_style == "none":
        return trace
    elif token_style == "qwen3":
        return add_special_tokens_qwen3(trace)
    elif token_style == "llama3":
        return add_special_tokens_llama3(trace)
    else:
        return trace


def count_unresolved_todos(trace: str) -> tuple[set, set]:
    """Count todo items that are defined but never satisfied.
    
    Returns (defined_todos, satisfied_todos).
    """
    # Find todo definitions: N ↦ "description" in plan blocks
    todo_def_pattern = re.compile(r'(\d+)\s*↦\s*[「"]')
    
    # Find satisfies markers: ⊨ N or ⊨ [N • M]
    satisfies_single = re.compile(r'⊨\s*(\d+)')
    satisfies_array = re.compile(r'⊨\s*\[([^\]]+)\]')
    
    # Extract defined todos (only from plan blocks)
    defined_todos = set()
    plan_match = re.search(r'plan\s*\{.*?todo\s*↦\s*\{([^}]*)\}', trace, re.DOTALL)
    if plan_match:
        todo_block = plan_match.group(1)
        for match in todo_def_pattern.finditer(todo_block):
            defined_todos.add(int(match.group(1)))
    
    # Extract satisfied todos
    satisfied_todos = set()
    for match in satisfies_single.finditer(trace):
        satisfied_todos.add(int(match.group(1)))
    for match in satisfies_array.finditer(trace):
        for item in match.group(1).split('•'):
            item = item.strip()
            if item.isdigit():
                satisfied_todos.add(int(item))
    
    return defined_todos, satisfied_todos


def filter_unreferenced_tags(trace: str) -> tuple[str, int]:
    """Remove tags (🏷) that are never referenced (※).
    
    Returns (filtered_trace, count_of_removed_tags).
    """
    # Find all defined tags: 🏷 tagname or 🏷 "tagname" or 🏷 'tagname'
    tag_pattern = re.compile(r'🏷\s*["\']?([a-zA-Z_][a-zA-Z0-9_-]*)["\']?')
    
    # Find all references: ※ tagname or ※ [tag1 • tag2 • ...]
    ref_single = re.compile(r'※\s*["\']?([a-zA-Z_][a-zA-Z0-9_-]*)["\']?')
    ref_array = re.compile(r'※\s*\[([^\]]+)\]')
    
    # Collect all defined tags
    defined_tags = set()
    for match in tag_pattern.finditer(trace):
        defined_tags.add(match.group(1))
    
    # Collect all referenced tags
    referenced_tags = set()
    for match in ref_single.finditer(trace):
        referenced_tags.add(match.group(1))
    for match in ref_array.finditer(trace):
        # Parse array contents: tag1 • tag2 • ...
        array_content = match.group(1)
        for item in array_content.split('•'):
            item = item.strip().strip('"\'')
            if item:
                referenced_tags.add(item)
    
    # Find unreferenced tags
    unreferenced = defined_tags - referenced_tags
    
    if not unreferenced:
        return trace, 0
    
    # Remove unreferenced tag definitions
    # Pattern: 🏷 tagname (with optional quotes) and surrounding whitespace
    removed_count = 0
    result = trace
    for tag in unreferenced:
        # Match 🏷 followed by the tag (quoted or not), with optional surrounding space
        patterns = [
            re.compile(r'\s*🏷\s*"' + re.escape(tag) + r'"'),
            re.compile(r'\s*🏷\s*\'' + re.escape(tag) + r'\''),
            re.compile(r'\s*🏷\s*' + re.escape(tag) + r'(?![a-zA-Z0-9_-])'),
        ]
        for pattern in patterns:
            new_result, count = pattern.subn('', result)
            if count > 0:
                removed_count += count
                result = new_result
                break
    
    return result, removed_count


def process_trace(trace: str, merge_thinks: bool = True, add_tokens: bool = True, 
                  token_style: str = "qwen3", filter_tags: bool = True) -> tuple[str, dict]:
    """Process a single trace. Returns (processed_trace, stats_dict)."""
    result = trace
    stats = {"tags_removed": 0, "unresolved_todos": 0, "total_todos": 0}
    
    # Count unresolved todos (before any processing)
    defined, satisfied = count_unresolved_todos(trace)
    stats["total_todos"] = len(defined)
    stats["unresolved_todos"] = len(defined - satisfied)
    
    if merge_thinks:
        result = merge_sequential_acts(result)
    
    if filter_tags:
        result, tags_removed = filter_unreferenced_tags(result)
        stats["tags_removed"] = tags_removed
    
    if add_tokens:
        result = add_special_tokens(result, token_style)
    
    return result, stats


def main():
    parser = argparse.ArgumentParser(
        description="Post-process TASK traces for training"
    )
    parser.add_argument(
        "input",
        help="Input traces JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: input with .processed suffix)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge sequential think blocks"
    )
    parser.add_argument(
        "--no-tokens",
        action="store_true",
        help="Don't add special tokens"
    )
    parser.add_argument(
        "--token-style",
        choices=["qwen3", "llama3", "none"],
        default="qwen3",
        help="Special token style (default: qwen3)"
    )
    parser.add_argument(
        "--no-filter-tags",
        action="store_true",
        help="Don't filter unreferenced tags"
    )
    parser.add_argument(
        "--purge-unresolved",
        action="store_true",
        help="Remove traces with unresolved todo items"
    )
    parser.add_argument(
        "--preview",
        type=int,
        metavar="N",
        help="Preview first N traces without writing"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return
    
    output_path = Path(args.output) if args.output else input_path.with_suffix('.processed.jsonl')
    
    print(f"Loading traces from {input_path}...")
    
    traces = []
    with open(input_path) as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except:
                pass
    
    print(f"Loaded {len(traces)} traces")
    
    # Process
    merge_thinks = not args.no_merge
    add_tokens = not args.no_tokens
    filter_tags = not args.no_filter_tags
    
    print(f"Processing (merge={merge_thinks}, tokens={add_tokens}, filter_tags={filter_tags}, style={args.token_style})...")
    
    processed = []
    merged_count = 0
    total_tags_removed = 0
    total_todos = 0
    total_unresolved = 0
    traces_with_unresolved = 0
    
    for i, item in enumerate(traces):
        trace_text = item.get("trace", "")
        original_acts = trace_text.count('act {')
        
        processed_text, stats = process_trace(
            trace_text,
            merge_thinks=merge_thinks,
            filter_tags=filter_tags,
            add_tokens=add_tokens,
            token_style=args.token_style
        )
        
        new_acts = processed_text.count('act {')
        if new_acts < original_acts:
            merged_count += 1
        
        total_tags_removed += stats.get("tags_removed", 0)
        total_todos += stats.get("total_todos", 0)
        unresolved = stats.get("unresolved_todos", 0)
        total_unresolved += unresolved
        if unresolved > 0:
            traces_with_unresolved += 1
            if args.purge_unresolved:
                continue  # Skip this trace
        
        processed_item = {**item, "trace": processed_text}
        processed.append(processed_item)
        
        if args.preview and i < args.preview:
            print(f"\n{'='*60}")
            print(f"Trace {i} (tags removed: {stats.get('tags_removed', 0)}, unresolved todos: {unresolved}):")
            print(f"{'='*60}")
            print(processed_text[:3000])
            if len(processed_text) > 3000:
                print("... (truncated)")
    
    print(f"\nMerged acts in {merged_count}/{len(traces)} traces")
    print(f"Removed {total_tags_removed} unreferenced tags")
    print(f"Todos: {total_unresolved}/{total_todos} unresolved ({traces_with_unresolved} traces with issues)")
    if args.purge_unresolved:
        print(f"Purged {traces_with_unresolved} traces with unresolved todos")
    
    if args.preview:
        print(f"\nPreview mode - not writing output")
        return
    
    # Write output
    with open(output_path, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
    
    print(f"Wrote {len(processed)} processed traces to {output_path}")
    
    # Stats
    total_chars_before = sum(len(t.get("trace", "")) for t in traces)
    total_chars_after = sum(len(t.get("trace", "")) for t in processed)
    
    print(f"\nSize change: {total_chars_before:,} → {total_chars_after:,} chars ({100*total_chars_after/total_chars_before:.1f}%)")


if __name__ == "__main__":
    main()

