#!/usr/bin/env python3
"""
TASK Format Synthetic Data Generator

Generates diverse synthetic training traces in the TASK format using OpenAI.
Supports both real-time and batch API (50% cheaper) modes.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from schema import Scenario, ScenarioBatch
from validator import validate_trace


# ============================================================================
# Prompts
# ============================================================================

SCENARIO_SYSTEM_PROMPT = """You are a synthetic data generator creating diverse, realistic scenarios for training an AI assistant.

Generate scenarios that are:
- Grounded in real-world tasks people actually need help with
- Varied across domains (coding, research, data analysis, creative writing, planning, troubleshooting, etc.)
- Appropriately complex (mix of simple and multi-step tasks)
- Clear and specific (not vague or overly abstract)

Each scenario should define:
- A realistic system prompt
- 1-3 useful tools relevant to the task (OR empty tools list if no_tools=true)
- A natural user request
- Expected complexity level

Avoid:
- Contrived or unrealistic scenarios
- Overly simple or trivially obvious tasks"""

SCENARIO_NO_TOOLS_PROMPT = """You are a synthetic data generator creating diverse, realistic scenarios for training an AI assistant.

Generate scenarios that DO NOT require any tools - pure reasoning, knowledge, or creative tasks.

Good domains for tool-free scenarios:
- Creative writing (stories, poems, scripts)
- Explanations and teaching (explain concepts, how things work)
- Analysis and reasoning (logic puzzles, ethical dilemmas, comparisons)
- Advice and recommendations (career, relationships, decisions)
- Brainstorming and ideation
- Summarization and paraphrasing
- Translation and language help
- Math and logic (that can be done mentally)

Each scenario should define:
- A realistic system prompt
- An EMPTY tools list (no tools needed)
- A natural user request that can be answered through reasoning/knowledge alone
- Expected complexity level

Avoid:
- Tasks that would genuinely benefit from tools (real-time data, file operations, searches)
- Overly simple single-sentence responses"""

SCENARIO_USER_PROMPT = """Generate {count} diverse, realistic scenarios for synthetic trace generation.

Ensure variety across:
- Domains: coding, data analysis, research, writing, planning, math, troubleshooting
- Complexity: simple (1-2 steps), medium (2-3 steps), complex (3-5 steps)
- Tool types: search, calculation, file operations, API calls, data processing
- Follow-ups: {follow_up_instruction}

Each scenario needs tools that would realistically help accomplish the task."""

FOLLOW_UP_INSTRUCTION_YES = "Include 1-3 realistic follow-up requests in follow_up_requests. These should be natural continuations (clarifications, additional questions, or building on the response)."
FOLLOW_UP_INSTRUCTION_NO = "Set follow_up_requests to an empty list (single-turn conversation)."


TRACE_SYSTEM_PROMPT = """You generate training data in the TASK format - a structured message format for LLM task completion.

# TASK Format Syntax

## Literals
- Objects: { key ↦ value • key2 ↦ value2 }
- Arrays: [ a • b • c ]
- Strings: unquoted if no spaces, "quoted" otherwise
- Long strings: 「special quotes for multi-line content」

## Operators (postfix)
- 🏷 tag - semantic tag for later reference
- ※ ref or ※ [ref1 • ref2] - references to prior tags
- ⊨ N - satisfies todo item N
- 𝑝 0.0-1.0 - confidence score (1.0 = certain, lower = inference)

## Structure Types

system「message」🏷 sys1

tool {
    name ↦ get_data •
    description ↦ "Fetches data from a source" •
    params ↦ {
        source ↦ { type ↦ string • description ↦ "Data source path" } •
        format ↦ { enum ↦ [ json • csv • xml ] • required ↦ false • description ↦ "Output format" }
    }
}

user「message」🏷 usr1

plan {
    todo ↦ {
        1 ↦ "First task." •
        2 ↦ "Second task." ※ usr1
    } •
    rationale ↦ "Explanation of approach."
}

act {
    call ↦ {
        tool ↦ get_data •
        source ↦ "/path/to/data" •
        id ↦ result1
    } ⊨ 1
}

result {
    data ↦ "Tool output here." 🏷 result1
}

// OR if tool fails:
result {
    data ↦ "Error: Connection timeout - service unavailable" 🏷 result1
}

act {
    think ↦ "Reasoning about the result - can be a longer, multi-sentence thought process. All sequential reasoning goes in ONE think, not multiple act blocks." 𝑝 0.85 ※ result1 🏷 thought1 ⊨ 2
}

response「Final answer to user.」※ [result1 • thought1] ⊨ 3

## Rules
- Sequence: system → tools → user → plan → (act/result cycles) → response
- User tags: usr1, usr2, etc.
- Every todo must be satisfied (⊨) by end
- Tool results tagged with call's id
- Confidence: 1.0 for facts/tool outputs, 0.7-0.95 for inferences
- Tool names: snake_case or camelCase, NO SPACES (e.g. get_weather, searchWeb, NOT "Get Weather")
- Param types: always specify type OR enum for every parameter
- Tags go on EXPRESSIONS, not blocks (e.g. data ↦ "value" 🏷 tag, NOT result { ... } 🏷 tag)
- Tool errors: result data can contain error messages (e.g. data ↦ "Error: timeout") - model should handle gracefully (retry, fallback, or inform user)

## CRITICAL - Common Mistakes to Avoid
1. BRACKET BALANCE: Every { must have matching }. Every [ must have matching ]. Every 「 must have matching 」. Count carefully!
2. COMPLETE THE TRACE: Always end with a response「...」 - never truncate mid-trace
3. SATISFY ALL TODOS: If you define todos 1,2,3 in the plan, you MUST have ⊨ 1, ⊨ 2, ⊨ 3 somewhere in the trace
4. DEFINE BEFORE REFERENCE: Only use ※ ref for tags you already defined with 🏷 earlier in the trace
5. RESULT TAGS: Tag the data expression inside result, NOT the result block itself
6. TOOL NAMES: No spaces! Use snake_case (get_weather) or camelCase (getWeather)
7. PARAM DEFINITIONS: Every param needs type ↦ X or enum ↦ [...], never leave undefined
8. ONE ACT PER PHASE: Don't split sequential thinking into multiple act blocks. One act block can have one think step - if you need extended reasoning, put it all in ONE think, not separate acts"""


TRACE_USER_PROMPT = """Generate a complete TASK format trace for this scenario:

**Domain:** {domain}
**System Prompt:** {system_prompt}
**Initial User Request:** {user_request}
{follow_ups_section}
**Target Length:** {num_steps} total act phases across all turns
{tool_error_instruction}
**Tools:**
{tools_desc}

Generate the trace with:
1. System message {tools_instruction}
2. User message (tagged usr1)
3. Plan with todo list and rationale
4. Act phases ({act_instruction}) - aim for {num_steps} total
5. Response referencing sources
{multi_turn_instruction}

Output ONLY the TASK format trace, no explanations."""

MULTI_TURN_INSTRUCTION = """
6. Then for each follow-up: user message (usr2, usr3...) → new plan → more act phases → response
   - Each turn should have its own plan that references the prior context
   - Continue numbering todos across turns or start fresh per turn
   - Final response should wrap up the entire conversation"""

TOOL_ERROR_INSTRUCTION = """
**IMPORTANT - Tool Error Scenario:**
One of the tool calls should FAIL with a realistic error (e.g. timeout, rate limit, invalid input, service unavailable, permission denied).
The result data should contain the error message, e.g. `data ↦ "Error: Connection timeout - service unavailable"`.
The model must then handle this gracefully: retry with different params, use a fallback approach, or explain the limitation to the user.
This teaches error handling - make the recovery realistic and helpful.
"""

# Inline scenario generation (no separate scenario API call)
INLINE_TRACE_PROMPT = """Generate a complete, realistic TASK format trace.

**Parameters:**
- Domain: {domain}
- Has tools: {has_tools}
- Has follow-ups: {has_follow_ups}
- Target length: {num_steps} act phases total
- Include tool error: {include_error}

**Your task:**
1. Invent a realistic scenario in the {domain} domain
2. Create appropriate tools (if has_tools=True) with proper names (snake_case), descriptions, and typed params
3. Write a natural user request
4. Generate the complete trace with {num_steps} act phases
{follow_up_instruction}
{error_instruction}

Output ONLY the TASK format trace, no explanations. Make sure the scenario feels real and grounded."""

DOMAINS = ["coding", "data_analysis", "research", "writing", "planning", "math", "troubleshooting", "creative", "education", "business"]


EXAMPLE_TRACE = '''system「You are a helpful assistant.」🏷 sys1

tool {
    name ↦ get_weather •
    description ↦ "Fetches current weather for a location" •
    params ↦ {
        zip_code ↦ { type ↦ string • description ↦ "US ZIP code" } •
        unit ↦ { enum ↦ [ metric • imperial ] • required ↦ false • description ↦ "Temperature unit" }
    }
}

user「What is the weather in 94103 today?」🏷 usr1

plan {
    todo ↦ {
        1 ↦ "Fetch the weather information." •
        2 ↦ "Determine the best clothing for the weather." ※ usr1 •
        3 ↦ "Present to the user."
    } •
    rationale ↦ "The user wants to know the weather, and the get_weather tool will give us live weather data. Then, they want to know the best clothing to wear. I will consider what the best clothing for the weather is given the output of the get_weather tool, and finally present it to the user."
}

act {
    call ↦ {
        tool ↦ get_weather •
        zip_code ↦ "94103" •
        id ↦ weather_result
    } ⊨ 1
}

result {
    data ↦ "The weather at 94103 is 68 degrees Fahrenheit, and overcast." 🏷 weather_result
}

act {
    think ↦ "Given the weather data, I need to determine appropriate clothing. 68°F is mild - not hot, not cold. Overcast means no direct sun but also no rain expected. For this temperature range, layers work well: a t-shirt as a base, with a light sweater or jacket that can be removed if it warms up. Sweatpants or jeans would be comfortable. No need for rain gear or heavy outerwear." 𝑝 0.9 ※ weather_result 🏷 rationale ⊨ 2
}

response「The weather in 94103 today is 68 degrees Fahrenheit and overcast. Today would be a good day for a light sweater, a t-shirt, and sweatpants.」※ [weather_result • rationale] ⊨ 3'''


# ============================================================================
# Generation
# ============================================================================

def build_trace_messages(scenario: Scenario, include_tool_error: bool = False, num_steps: int = 5) -> list[dict]:
    """Build the messages for trace generation."""
    has_tools = len(scenario.tools) > 0
    has_follow_ups = len(scenario.follow_up_requests) > 0
    
    if has_tools:
        tools_desc = "\n".join(
            f"- {t.name}({', '.join(p.name for p in t.params)}): {t.description}"
            for t in scenario.tools
        )
        tools_instruction = "and tool definitions"
        act_instruction = "tool calls with realistic results, and/or thinking"
        tool_error_instruction = TOOL_ERROR_INSTRUCTION if include_tool_error else ""
    else:
        tools_desc = "(No tools - pure reasoning/knowledge task)"
        tools_instruction = "(no tools needed for this task)"
        act_instruction = "thinking/reasoning steps only - NO tool calls"
        tool_error_instruction = ""
    
    # Follow-ups section
    if has_follow_ups:
        follow_ups_section = "**Follow-up requests:**\n" + "\n".join(
            f"  - usr{i+2}: \"{req}\"" for i, req in enumerate(scenario.follow_up_requests)
        )
        multi_turn_instruction = MULTI_TURN_INSTRUCTION
    else:
        follow_ups_section = ""
        multi_turn_instruction = ""
    
    return [
        {"role": "system", "content": TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here's an example trace:\n\n{EXAMPLE_TRACE}"},
        {"role": "assistant", "content": "I understand the TASK format. I'll generate traces following this exact syntax with proper tags, references, confidence scores, and todo satisfaction markers."},
        {"role": "user", "content": TRACE_USER_PROMPT.format(
            domain=scenario.domain,
            system_prompt=scenario.system_prompt,
            user_request=scenario.user_request,
            num_steps=num_steps,
            tools_desc=tools_desc,
            tools_instruction=tools_instruction,
            act_instruction=act_instruction,
            tool_error_instruction=tool_error_instruction,
            follow_ups_section=follow_ups_section,
            multi_turn_instruction=multi_turn_instruction
        )}
    ]


class TraceGenerator:
    def __init__(self, model: str = "gpt-5-mini", concurrency: int = 5, error_rate: float = 0.2, 
                 no_tools_rate: float = 0.2, follow_up_rate: float = 0.3,
                 min_steps: int = 5, max_steps: int = 50):
        self.async_client = AsyncOpenAI()
        self.sync_client = OpenAI()
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.error_rate = error_rate
        self.no_tools_rate = no_tools_rate
        self.follow_up_rate = follow_up_rate
        self.min_steps = min_steps
        self.max_steps = max_steps
    
    async def generate_scenarios(self, count: int, no_tools: bool = False, include_follow_ups: bool = False) -> list[Scenario]:
        """Generate diverse scenarios for trace generation."""
        system_prompt = SCENARIO_NO_TOOLS_PROMPT if no_tools else SCENARIO_SYSTEM_PROMPT
        follow_up_instruction = FOLLOW_UP_INSTRUCTION_YES if include_follow_ups else FOLLOW_UP_INSTRUCTION_NO
        async with self.semaphore:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": SCENARIO_USER_PROMPT.format(count=count, follow_up_instruction=follow_up_instruction)}
                ],
                response_format=ScenarioBatch,
            )
            return response.choices[0].message.parsed.scenarios
    
    async def generate_trace(self, scenario: Scenario, include_tool_error: bool | None = None, num_steps: int | None = None) -> str:
        """Generate a complete trace as TASK format text (real-time)."""
        # Use configured error_rate if not explicitly set
        if include_tool_error is None:
            include_tool_error = random.random() < self.error_rate and len(scenario.tools) > 0
        
        # Sample number of steps if not explicitly set
        if num_steps is None:
            num_steps = random.randint(self.min_steps, self.max_steps)
        
        async with self.semaphore:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=build_trace_messages(scenario, include_tool_error=include_tool_error, num_steps=num_steps),
            )
            return response.choices[0].message.content.strip()
    
    async def generate_trace_inline(self, has_tools: bool, has_follow_ups: bool, 
                                     include_error: bool, num_steps: int) -> str:
        """Generate trace with inline scenario (no separate scenario generation)."""
        domain = random.choice(DOMAINS)
        
        follow_up_instruction = MULTI_TURN_INSTRUCTION if has_follow_ups else ""
        error_instruction = "Include a realistic tool error and recovery." if include_error else ""
        
        prompt = INLINE_TRACE_PROMPT.format(
            domain=domain,
            has_tools=has_tools,
            has_follow_ups=has_follow_ups,
            num_steps=num_steps,
            include_error=include_error,
            follow_up_instruction=follow_up_instruction,
            error_instruction=error_instruction
        )
        
        async with self.semaphore:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here's an example trace:\n\n{EXAMPLE_TRACE}"},
                    {"role": "assistant", "content": "I understand the TASK format. I'll generate traces following this exact syntax."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content.strip()
    
    async def generate_batch_realtime(self, count: int, progress_callback=None, max_retries: int = 3) -> list[dict]:
        """Generate traces in real-time (more expensive, immediate results)."""
        all_scenarios = await self._generate_all_scenarios(count, progress_callback)
        print()
        
        async def generate_with_retry(scenario: Scenario, idx: int) -> dict | None:
            for attempt in range(max_retries):
                try:
                    if progress_callback:
                        retry_str = f" (retry {attempt})" if attempt > 0 else ""
                        progress_callback(f"Generating trace {idx + 1}/{count}{retry_str}...")
                    
                    trace_text = await self.generate_trace(scenario)
                    result = self._validate_and_wrap(scenario, trace_text, idx, silent=(attempt < max_retries - 1))
                    
                    if result is not None:
                        return result
                    # Invalid trace, retry
                except Exception as e:
                    if attempt == max_retries - 1:
                        import traceback
                        print(f"\nError generating trace {idx + 1}:")
                        traceback.print_exc()
            return None
        
        tasks = [generate_with_retry(s, i) for i, s in enumerate(all_scenarios)]
        outputs = await asyncio.gather(*tasks)
        return [r for r in outputs if r is not None]
    
    async def _generate_all_scenarios(self, count: int, progress_callback=None) -> list[Scenario]:
        """Generate all scenarios in batches of 10, with variety in tools and follow-ups."""
        # Calculate splits
        no_tools_count = int(count * self.no_tools_rate)
        tools_count = count - no_tools_count
        
        # Within each category, split by follow-ups
        tools_with_followup = int(tools_count * self.follow_up_rate)
        tools_without_followup = tools_count - tools_with_followup
        no_tools_with_followup = int(no_tools_count * self.follow_up_rate)
        no_tools_without_followup = no_tools_count - no_tools_with_followup
        
        all_scenarios = []
        
        async def generate_batch_type(target_count: int, no_tools: bool, include_follow_ups: bool, label: str):
            generated = 0
            while generated < target_count:
                batch_size = min(10, target_count - generated)
                scenarios = await self.generate_scenarios(batch_size, no_tools=no_tools, include_follow_ups=include_follow_ups)
                all_scenarios.extend(scenarios)
                generated += len(scenarios)
                if progress_callback:
                    progress_callback(f"Generated {len(all_scenarios)}/{count} scenarios ({label})")
        
        # Generate all types
        if tools_without_followup > 0:
            await generate_batch_type(tools_without_followup, no_tools=False, include_follow_ups=False, label="tools, single-turn")
        if tools_with_followup > 0:
            await generate_batch_type(tools_with_followup, no_tools=False, include_follow_ups=True, label="tools, multi-turn")
        if no_tools_without_followup > 0:
            await generate_batch_type(no_tools_without_followup, no_tools=True, include_follow_ups=False, label="no-tools, single-turn")
        if no_tools_with_followup > 0:
            await generate_batch_type(no_tools_with_followup, no_tools=True, include_follow_ups=True, label="no-tools, multi-turn")
        
        # Shuffle to mix them up
        random.shuffle(all_scenarios)
        return all_scenarios[:count]
    
    def _validate_and_wrap(self, scenario: Scenario, trace_text: str, idx: int, silent: bool = False) -> dict | None:
        """Validate trace and wrap with metadata."""
        validation = validate_trace(trace_text)
        if not validation.valid:
            if not silent:
                print(f"\n  Trace {idx + 1} invalid: {validation.errors}")
            return None
        
        if validation.warnings and not silent:
            print(f"\n  Trace {idx + 1} warnings: {validation.warnings}")
        
        return {
            "scenario": scenario.model_dump(),
            "trace": trace_text,
            "validation": {
                "tags_defined": list(validation.tags_defined),
                "todos_defined": list(validation.todos_defined),
                "todos_satisfied": list(validation.todos_satisfied),
            }
        }
    
    # =========================================================================
    # Batch API Methods (50% cheaper, async processing)
    # =========================================================================
    
    def submit_batch_inline(self, count: int, output_dir: Path) -> str:
        """Submit a batch job with inline scenario generation (no separate scenario API calls)."""
        print(f"Creating {count} batch requests with inline scenarios...")
        
        # Calculate distribution
        no_tools_count = int(count * self.no_tools_rate)
        tools_count = count - no_tools_count
        follow_up_count = int(count * self.follow_up_rate)
        error_count = int(tools_count * self.error_rate)
        
        batch_requests = []
        for i in range(count):
            # Determine characteristics for this trace
            has_tools = i >= no_tools_count  # First no_tools_count are without tools
            has_follow_ups = random.random() < self.follow_up_rate
            include_error = has_tools and random.random() < self.error_rate
            num_steps = random.randint(self.min_steps, self.max_steps)
            domain = random.choice(DOMAINS)
            
            follow_up_instruction = MULTI_TURN_INSTRUCTION if has_follow_ups else ""
            error_instruction = "Include a realistic tool error and recovery." if include_error else ""
            
            prompt = INLINE_TRACE_PROMPT.format(
                domain=domain,
                has_tools=has_tools,
                has_follow_ups=has_follow_ups,
                num_steps=num_steps,
                include_error=include_error,
                follow_up_instruction=follow_up_instruction,
                error_instruction=error_instruction
            )
            
            batch_requests.append({
                "custom_id": f"trace-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Here's an example trace:\n\n{EXAMPLE_TRACE}"},
                        {"role": "assistant", "content": "I understand the TASK format. I'll generate traces following this exact syntax."},
                        {"role": "user", "content": prompt}
                    ],
                }
            })
        
        print(f"  ~{no_tools_count} no-tools, ~{follow_up_count} multi-turn, ~{error_count} with errors")
        
        # Write batch file
        batch_input_file = output_dir / "batch_input.jsonl"
        with open(batch_input_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"Created batch input file: {batch_input_file}")
        
        # Upload and create batch
        print("Uploading batch file...")
        with open(batch_input_file, "rb") as f:
            uploaded = self.sync_client.files.create(file=f, purpose="batch")
        print(f"Uploaded file: {uploaded.id}")
        
        print("Creating batch job...")
        batch = self.sync_client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"TASK format synthetic data (inline): {count} traces"}
        )
        
        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "status": batch.status,
            "count": count,
            "model": self.model,
            "inline": True,
        }
        batch_info_file = output_dir / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)
        
        print(f"\nBatch submitted!")
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Info saved to: {batch_info_file}")
        print(f"\nTo check status and retrieve results:")
        print(f"  python generate.py --retrieve {batch.id} -o {output_dir / 'traces.jsonl'}")
        
        return batch.id
    
    async def submit_batch(self, count: int, output_dir: Path, progress_callback=None) -> str:
        """Submit a batch job for trace generation (with scenario generation). Returns batch ID."""
        all_scenarios = await self._generate_all_scenarios(count, progress_callback)
        print()
        
        # Save scenarios for later retrieval
        scenarios_file = output_dir / "batch_scenarios.json"
        with open(scenarios_file, "w") as f:
            json.dump([s.model_dump() for s in all_scenarios], f)
        print(f"Saved {len(all_scenarios)} scenarios to {scenarios_file}")
        
        # Create batch request file (some with tool errors, variable steps)
        batch_requests = []
        tool_error_count = 0
        follow_up_count = sum(1 for s in all_scenarios if s.follow_up_requests)
        for i, scenario in enumerate(all_scenarios):
            include_tool_error = random.random() < self.error_rate and len(scenario.tools) > 0
            num_steps = random.randint(self.min_steps, self.max_steps)
            if include_tool_error:
                tool_error_count += 1
            batch_requests.append({
                "custom_id": f"trace-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": build_trace_messages(scenario, include_tool_error=include_tool_error, num_steps=num_steps),
                }
            })
        print(f"  {tool_error_count}/{len(all_scenarios)} with tool errors, {follow_up_count}/{len(all_scenarios)} multi-turn")
        
        batch_input_file = output_dir / "batch_input.jsonl"
        with open(batch_input_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"Created batch input file: {batch_input_file}")
        
        # Upload file
        print("Uploading batch file...")
        with open(batch_input_file, "rb") as f:
            uploaded = self.sync_client.files.create(file=f, purpose="batch")
        print(f"Uploaded file: {uploaded.id}")
        
        # Create batch
        print("Creating batch job...")
        batch = self.sync_client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"TASK format synthetic data: {count} traces"}
        )
        
        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "status": batch.status,
            "count": count,
            "model": self.model,
            "scenarios_file": str(scenarios_file),
        }
        batch_info_file = output_dir / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)
        
        print(f"\nBatch submitted!")
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Info saved to: {batch_info_file}")
        print(f"\nTo check status and retrieve results:")
        print(f"  python generate.py --retrieve {batch.id} -o {output_dir / 'traces.jsonl'}")
        
        return batch.id
    
    def retrieve_batch(self, batch_id: str, output_dir: Path) -> list[dict]:
        """Retrieve and process batch results."""
        print(f"Checking batch {batch_id}...")
        batch = self.sync_client.batches.retrieve(batch_id)
        
        print(f"  Status: {batch.status}")
        print(f"  Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        
        if batch.status == "in_progress":
            print("\nBatch still processing. Check back later.")
            return [], [], []
        
        if batch.status == "failed":
            print(f"\nBatch failed!")
            if batch.errors:
                for err in batch.errors.data:
                    print(f"  Error: {err.message}")
            return [], [], []
        
        if batch.status != "completed":
            print(f"\nUnexpected status: {batch.status}")
            return [], [], []
        
        # Check if inline mode (no scenarios)
        batch_info_file = output_dir / "batch_info.json"
        is_inline = False
        if batch_info_file.exists():
            with open(batch_info_file) as f:
                batch_info = json.load(f)
                is_inline = batch_info.get("inline", False)
        
        # Load scenarios (if not inline mode)
        scenarios_file = output_dir / "batch_scenarios.json"
        if is_inline:
            print("Inline mode - no scenario data")
            scenarios = []
        elif not scenarios_file.exists():
            print(f"Warning: {scenarios_file} not found, results won't include scenario data")
            scenarios = []
        else:
            with open(scenarios_file) as f:
                scenarios = [Scenario(**s) for s in json.load(f)]
        
        # Download results
        print("Downloading results...")
        output_file_id = batch.output_file_id
        content = self.sync_client.files.content(output_file_id)
        
        # Parse results
        results = []
        failed_indices = []
        
        for line in content.text.strip().split("\n"):
            item = json.loads(line)
            custom_id = item["custom_id"]
            idx = int(custom_id.split("-")[1])
            
            if item["error"]:
                print(f"  Request {idx} failed: {item['error']}")
                failed_indices.append(idx)
                continue
            
            trace_text = item["response"]["body"]["choices"][0]["message"]["content"].strip()
            scenario = scenarios[idx] if idx < len(scenarios) else None
            
            if scenario:
                result = self._validate_and_wrap(scenario, trace_text, idx)
                if result:
                    results.append(result)
                else:
                    failed_indices.append(idx)
            else:
                # No scenario data (inline mode), just save trace with validation
                validation = validate_trace(trace_text)
                if validation.valid:
                    results.append({
                        "trace": trace_text,
                        "validation": {
                            "tags_defined": list(validation.tags_defined),
                            "todos_defined": list(validation.todos_defined),
                            "todos_satisfied": list(validation.todos_satisfied),
                        }
                    })
                else:
                    print(f"  Trace {idx} invalid: {validation.errors}")
                    failed_indices.append(idx)
        
        print(f"\nBatch complete: {len(results)} valid, {len(failed_indices)} failed/invalid")
        return results, scenarios, failed_indices
    
    async def cleanup_batch_failures(self, scenarios: list[Scenario], results: list[dict], 
                                      failed_indices: list[int], max_retries: int = 3) -> list[dict]:
        """Retry failed batch traces in real-time."""
        if not failed_indices:
            return results
        
        print(f"\nRetrying {len(failed_indices)} failed traces in real-time...")
        
        async def retry_one(idx: int) -> dict | None:
            scenario = scenarios[idx]
            for attempt in range(max_retries):
                try:
                    trace_text = await self.generate_trace(scenario)
                    result = self._validate_and_wrap(scenario, trace_text, idx, silent=(attempt < max_retries - 1))
                    if result is not None:
                        print(f"  Trace {idx + 1} recovered on attempt {attempt + 1}")
                        return result
                except Exception:
                    pass
            print(f"  Trace {idx + 1} failed after {max_retries} retries")
            return None
        
        tasks = [retry_one(i) for i in failed_indices]
        recovered = await asyncio.gather(*tasks)
        results.extend([r for r in recovered if r is not None])
        
        return results
    
    def check_batch_status(self, batch_id: str):
        """Check batch status without downloading results."""
        batch = self.sync_client.batches.retrieve(batch_id)
        print(f"Batch: {batch_id}")
        print(f"  Status: {batch.status}")
        print(f"  Created: {batch.created_at}")
        print(f"  Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        if batch.request_counts.failed:
            print(f"  Failed: {batch.request_counts.failed}")
        return batch.status


# ============================================================================
# CLI
# ============================================================================

def print_progress(msg: str):
    print(f"\r{msg}", end="", flush=True)


def get_next_trace_index(render_dir: Path) -> int:
    """Find the next available trace index in the rendered folder."""
    if not render_dir.exists():
        return 0
    
    existing = list(render_dir.glob("trace_*.task"))
    if not existing:
        return 0
    
    # Extract numbers from filenames like trace_0042.task
    indices = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[1])
            indices.append(num)
        except (IndexError, ValueError):
            pass
    
    return max(indices) + 1 if indices else 0


def count_existing_lines(file_path: Path) -> int:
    """Count existing lines in a JSONL file."""
    if not file_path.exists():
        return 0
    with open(file_path) as f:
        return sum(1 for _ in f)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic TASK format training data"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of traces to generate (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="traces.jsonl",
        help="Output file path (default: traces.jsonl)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also save rendered TASK format (as .task files)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Batch API (50%% cheaper, results in ~24h)"
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Skip scenario generation, generate scenarios inline with traces (faster, cheaper)"
    )
    parser.add_argument(
        "--retrieve",
        type=str,
        metavar="BATCH_ID",
        help="Retrieve results from a batch job"
    )
    parser.add_argument(
        "--status",
        type=str,
        metavar="BATCH_ID",
        help="Check status of a batch job"
    )
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Max retries for invalid traces (default: 3)"
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.2,
        help="Fraction of traces with tool errors (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--no-tools-rate",
        type=float,
        default=0.2,
        help="Fraction of traces without tools (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--follow-up-rate",
        type=float,
        default=0.3,
        help="Fraction of traces with user follow-ups (default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=5,
        help="Minimum act phases per trace (default: 5)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum act phases per trace (default: 50)"
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = TraceGenerator(
        model=args.model, 
        concurrency=args.concurrency, 
        error_rate=args.error_rate, 
        no_tools_rate=args.no_tools_rate,
        follow_up_rate=args.follow_up_rate,
        min_steps=args.min_steps,
        max_steps=args.max_steps
    )
    
    # Check batch status
    if args.status:
        generator.check_batch_status(args.status)
        return
    
    # Retrieve batch results
    if args.retrieve:
        results, scenarios, failed_indices = generator.retrieve_batch(args.retrieve, output_dir)
        
        # Cleanup pass: retry failed traces in real-time
        if failed_indices and scenarios:
            results = await generator.cleanup_batch_failures(scenarios, results, failed_indices, max_retries=args.retries)
        
        if results:
            existing_count = count_existing_lines(output_path)
            with open(output_path, "a") as f:
                for item in results:
                    f.write(json.dumps(item) + "\n")
            print(f"Appended {len(results)} traces to {output_path} (total: {existing_count + len(results)})")
            
            if args.render:
                render_dir = output_dir / "rendered"
                render_dir.mkdir(exist_ok=True)
                start_idx = get_next_trace_index(render_dir)
                for i, item in enumerate(results):
                    task_path = render_dir / f"trace_{start_idx + i:04d}.task"
                    task_path.write_text(item["trace"])
                print(f"Rendered {len(results)} traces to {render_dir}/ (starting at {start_idx:04d})")
        return
    
    # Submit batch
    if args.batch:
        print(f"Submitting batch of {args.count} traces using {args.model}...")
        if args.inline:
            generator.submit_batch_inline(args.count, output_dir)
        else:
            await generator.submit_batch(args.count, output_dir, progress_callback=print_progress)
        return
    
    # Real-time generation (default)
    print(f"Generating {args.count} traces using {args.model}...")
    results = await generator.generate_batch_realtime(args.count, progress_callback=print_progress, max_retries=args.retries)
    print()
    
    # Write JSONL (append)
    existing_count = count_existing_lines(output_path)
    with open(output_path, "a") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"Appended {len(results)}/{args.count} valid traces to {output_path} (total: {existing_count + len(results)})")
    
    # Render (continue from last index)
    if args.render:
        render_dir = output_dir / "rendered"
        render_dir.mkdir(exist_ok=True)
        start_idx = get_next_trace_index(render_dir)
        for i, item in enumerate(results):
            task_path = render_dir / f"trace_{start_idx + i:04d}.task"
            task_path.write_text(item["trace"])
        print(f"Rendered {len(results)} traces to {render_dir}/ (starting at {start_idx:04d})")
    
    # Sample
    if results:
        print("\n--- Sample trace ---")
        print(results[0]["trace"][:2000])
        if len(results[0]["trace"]) > 2000:
            print("... (truncated)")
    else:
        print("\nNo traces generated. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())
