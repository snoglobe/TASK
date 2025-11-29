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
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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

# Dashboard for live training visualization
try:
    from dashboard import create_dashboard, TrainingDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class DashboardCallback(TrainerCallback):
    """Callback to update dashboard with training stats."""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self._last_loss = 0.0
        self._last_reward = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if self.dashboard is None or logs is None:
            return
        
        loss = logs.get('loss', self._last_loss)
        reward_mean = logs.get('reward_mean', logs.get('rewards/mean', self._last_reward))
        
        self._last_loss = loss
        self._last_reward = reward_mean
        
        self.dashboard.update_stats(
            step=state.global_step,
            loss=loss,
            reward_mean=reward_mean,
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step."""
        if self.dashboard is None:
            return
        
        self.dashboard.update_stats(step=state.global_step)


# =============================================================================
# TASK Format Parser and Verifier
# =============================================================================

class ParseError(Exception):
    """Error during TASK format parsing."""
    def __init__(self, message: str, position: int = -1):
        self.message = message
        self.position = position
        super().__init__(f"{message} at position {position}" if position >= 0 else message)


class TaskParser:
    """
    Recursive descent parser for TASK format.
    
    Grammar (simplified):
        trace       := block*
        block       := system_block | tool_block | user_block | plan_block | act_block | result_block | response_block
        system_block:= 'system' string postfix*
        user_block  := 'user' string postfix*
        tool_block  := 'tool' object postfix*
        plan_block  := 'plan' object postfix*
        act_block   := 'act' object postfix*
        result_block:= 'result' object postfix*
        response_block := 'response' string postfix*
        
        object      := '{' (key '↦' value ('•' key '↦' value)*)? '}'
        array       := '[' (value ('•' value)*)? ']'
        value       := string | number | object | array | identifier
        string      := '「' ... '」' | '"' ... '"' | unquoted_word
        postfix     := tag | ref | satisfies | confidence
        tag         := '🏷' identifier
        ref         := '※' (identifier | array)
        satisfies   := '⊨' number
        confidence  := '𝑝' number
    """
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)
        
        # Collected data during parse
        self.blocks = []  # List of (type, content, postfixes)
        self.defined_tags = set()
        self.referenced_tags = set()
        self.defined_todos = set()
        self.satisfied_todos = set()
        
        # Tool/call/result tracking
        self.defined_tools = set()  # Tool names from tool { name ↦ ... } blocks
        self.called_tools = set()   # Tool names from act { call ↦ { tool ↦ name } }
        self.call_ids = set()       # IDs from act { call ↦ { id ↦ "xxx" } }
        self.result_ids = set()     # IDs from result block data tags
        
        self.errors = []
        self.warnings = []
    
    def parse(self) -> dict:
        """Parse the trace and return analysis results."""
        try:
            self._parse_trace()
        except ParseError as e:
            self.errors.append(str(e))
        
        return {
            'blocks': self.blocks,
            'defined_tags': self.defined_tags,
            'referenced_tags': self.referenced_tags,
            'defined_todos': self.defined_todos,
            'satisfied_todos': self.satisfied_todos,
            'defined_tools': self.defined_tools,
            'called_tools': self.called_tools,
            'call_ids': self.call_ids,
            'result_ids': self.result_ids,
            'errors': self.errors,
            'warnings': self.warnings,
            'valid': len(self.errors) == 0,
        }
    
    def _peek(self, n: int = 1) -> str:
        """Peek at next n characters."""
        return self.text[self.pos:self.pos + n]
    
    def _advance(self, n: int = 1) -> str:
        """Advance and return consumed characters."""
        result = self.text[self.pos:self.pos + n]
        self.pos += n
        return result
    
    def _skip_whitespace(self):
        """Skip whitespace and newlines."""
        while self.pos < self.length and self.text[self.pos] in ' \t\n\r':
            self.pos += 1
    
    def _at_end(self) -> bool:
        return self.pos >= self.length
    
    def _match(self, s: str) -> bool:
        """Check if current position matches string."""
        return self.text[self.pos:self.pos + len(s)] == s
    
    def _expect(self, s: str):
        """Expect and consume a string."""
        if not self._match(s):
            raise ParseError(f"Expected '{s}'", self.pos)
        self._advance(len(s))
    
    def _parse_trace(self):
        """Parse top-level trace structure."""
        while not self._at_end():
            self._skip_whitespace()
            if self._at_end():
                break
            
            # Try to parse a block
            block_type = self._try_parse_block_keyword()
            if block_type:
                self._parse_block(block_type)
            else:
                # Skip unknown content until next block or end
                self._skip_to_next_block()
    
    def _try_parse_block_keyword(self) -> Optional[str]:
        """Try to match a block keyword."""
        keywords = ['system', 'tool', 'user', 'plan', 'act', 'result', 'response']
        for kw in keywords:
            if self._match(kw):
                # Make sure it's not part of a longer word
                end_pos = self.pos + len(kw)
                if end_pos >= self.length or not self.text[end_pos].isalnum():
                    self._advance(len(kw))
                    return kw
        return None
    
    def _skip_to_next_block(self):
        """Skip content until we find a block keyword or end."""
        keywords = ['system', 'tool', 'user', 'plan', 'act', 'result', 'response']
        start = self.pos
        while not self._at_end():
            for kw in keywords:
                if self._match(kw):
                    end_pos = self.pos + len(kw)
                    if end_pos >= self.length or not self.text[end_pos].isalnum():
                        return
            self._advance()
        if self.pos > start:
            self.warnings.append(f"Skipped unparseable content from {start} to {self.pos}")
    
    def _parse_block(self, block_type: str):
        """Parse a specific block type."""
        self._skip_whitespace()
        
        content = None
        postfixes = []
        
        if block_type in ('system', 'user', 'response'):
            # These expect a string
            content = self._parse_string()
            postfixes = self._parse_postfixes()
        elif block_type in ('tool', 'plan', 'act', 'result'):
            # These expect an object
            content = self._parse_object()
            postfixes = self._parse_postfixes()
            
            # Extract semantic info from content
            if block_type == 'tool' and isinstance(content, dict):
                self._extract_tool_info(content)
            elif block_type == 'plan' and isinstance(content, dict):
                self._extract_todos(content)
            elif block_type == 'act' and isinstance(content, dict):
                self._extract_act_info(content, postfixes)
            elif block_type == 'result' and isinstance(content, dict):
                self._extract_result_info(content, postfixes)
        
        self.blocks.append((block_type, content, postfixes))
    
    def _parse_string(self) -> str:
        """Parse a string value (「」, "", or unquoted)."""
        self._skip_whitespace()
        
        if self._match('「'):
            return self._parse_cjk_string()
        elif self._match('"'):
            return self._parse_quoted_string()
        else:
            return self._parse_unquoted_word()
    
    def _parse_cjk_string(self) -> str:
        """Parse 「...」 string."""
        self._expect('「')
        result = []
        depth = 1
        while not self._at_end() and depth > 0:
            ch = self._peek()
            if ch == '「':
                depth += 1
                result.append(self._advance())
            elif ch == '」':
                depth -= 1
                if depth > 0:
                    result.append(self._advance())
                else:
                    self._advance()
            else:
                result.append(self._advance())
        return ''.join(result)
    
    def _parse_quoted_string(self) -> str:
        """Parse "..." string."""
        self._expect('"')
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch == '"':
                self._advance()
                break
            elif ch == '\\' and self.pos + 1 < self.length:
                self._advance()  # Skip backslash
                result.append(self._advance())  # Add escaped char
            else:
                result.append(self._advance())
        return ''.join(result)
    
    def _parse_unquoted_word(self) -> str:
        """Parse unquoted identifier/word."""
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch in ' \t\n\r{}[]•↦🏷※⊨𝑝「」"':
                break
            result.append(self._advance())
        return ''.join(result)
    
    def _parse_object(self) -> dict:
        """Parse { key ↦ value • ... } object."""
        self._skip_whitespace()
        if not self._match('{'):
            raise ParseError("Expected '{'", self.pos)
        self._advance()
        
        result = {}
        while True:
            self._skip_whitespace()
            if self._match('}'):
                self._advance()
                break
            if self._at_end():
                self.errors.append(f"Unclosed object at position {self.pos}")
                break
            
            # Parse key
            key = self._parse_key()
            self._skip_whitespace()
            
            # Expect ↦
            if self._match('↦'):
                self._advance()
            else:
                self.errors.append(f"Expected '↦' after key '{key}' at position {self.pos}")
                self._skip_to_separator()
                continue
            
            self._skip_whitespace()
            
            # Parse value with postfixes
            value = self._parse_value()
            postfixes = self._parse_postfixes()
            
            result[key] = {'value': value, 'postfixes': postfixes}
            
            self._skip_whitespace()
            
            # Check for separator or end
            if self._match('•'):
                self._advance()
            elif self._match('}'):
                continue  # Will exit on next iteration
            elif not self._at_end():
                # Try to recover
                self._skip_to_separator()
        
        return result
    
    def _parse_array(self) -> list:
        """Parse [ a • b • c ] array."""
        self._skip_whitespace()
        self._expect('[')
        
        result = []
        while True:
            self._skip_whitespace()
            if self._match(']'):
                self._advance()
                break
            if self._at_end():
                self.errors.append(f"Unclosed array at position {self.pos}")
                break
            
            value = self._parse_value()
            postfixes = self._parse_postfixes()
            result.append({'value': value, 'postfixes': postfixes})
            
            self._skip_whitespace()
            if self._match('•'):
                self._advance()
            elif self._match(']'):
                continue
        
        return result
    
    def _parse_value(self):
        """Parse any value type."""
        self._skip_whitespace()
        
        if self._match('{'):
            return self._parse_object()
        elif self._match('['):
            return self._parse_array()
        elif self._match('「'):
            return self._parse_cjk_string()
        elif self._match('"'):
            return self._parse_quoted_string()
        else:
            # Number or identifier
            word = self._parse_unquoted_word()
            # Try to parse as number
            try:
                if '.' in word:
                    return float(word)
                return int(word)
            except ValueError:
                return word
    
    def _parse_key(self) -> str:
        """Parse object key (identifier or number)."""
        self._skip_whitespace()
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch in ' \t\n\r↦•{}[]':
                break
            result.append(self._advance())
        return ''.join(result)
    
    def _parse_postfixes(self) -> list:
        """Parse postfix operators (🏷, ※, ⊨, 𝑝)."""
        postfixes = []
        while True:
            self._skip_whitespace()
            
            if self._match('🏷'):
                self._advance()
                self._skip_whitespace()
                tag = self._parse_tag_name()
                postfixes.append(('tag', tag))
                self.defined_tags.add(tag)
                
            elif self._match('※'):
                self._advance()
                self._skip_whitespace()
                if self._match('['):
                    refs = self._parse_ref_array()
                    postfixes.append(('ref', refs))
                    for r in refs:
                        self.referenced_tags.add(r)
                else:
                    ref = self._parse_tag_name()
                    postfixes.append(('ref', [ref]))
                    self.referenced_tags.add(ref)
                    
            elif self._match('⊨'):
                self._advance()
                self._skip_whitespace()
                num = self._parse_number()
                postfixes.append(('satisfies', num))
                self.satisfied_todos.add(num)
                
            elif self._match('𝑝'):
                self._advance()
                self._skip_whitespace()
                num = self._parse_float()
                postfixes.append(('confidence', num))
            else:
                break
        
        return postfixes
    
    def _parse_tag_name(self) -> str:
        """Parse a tag/ref name."""
        # Handle quoted names
        if self._match('"'):
            return self._parse_quoted_string()
        if self._match("'"):
            self._advance()
            result = []
            while not self._at_end() and self._peek() != "'":
                result.append(self._advance())
            if self._match("'"):
                self._advance()
            return ''.join(result)
        
        # Unquoted name
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch.isalnum() or ch in '_-':
                result.append(self._advance())
            else:
                break
        return ''.join(result)
    
    def _parse_ref_array(self) -> list:
        """Parse [ref1 • ref2 • ...] reference array."""
        self._expect('[')
        refs = []
        while True:
            self._skip_whitespace()
            if self._match(']'):
                self._advance()
                break
            if self._at_end():
                break
            
            ref = self._parse_tag_name()
            if ref:
                refs.append(ref)
            
            self._skip_whitespace()
            if self._match('•'):
                self._advance()
        
        return refs
    
    def _parse_number(self) -> int:
        """Parse integer."""
        result = []
        while not self._at_end() and self._peek().isdigit():
            result.append(self._advance())
        return int(''.join(result)) if result else 0
    
    def _parse_float(self) -> float:
        """Parse float."""
        result = []
        while not self._at_end() and (self._peek().isdigit() or self._peek() == '.'):
            result.append(self._advance())
        try:
            return float(''.join(result))
        except ValueError:
            return 0.0
    
    def _skip_to_separator(self):
        """Skip to next • or }."""
        while not self._at_end():
            if self._match('•') or self._match('}') or self._match(']'):
                return
            self._advance()
    
    def _extract_tool_info(self, tool_obj: dict):
        """Extract tool name from tool definition block."""
        if 'name' in tool_obj:
            name_val = tool_obj['name'].get('value', '')
            if isinstance(name_val, str) and name_val:
                self.defined_tools.add(name_val)
    
    def _extract_todos(self, plan_obj: dict):
        """Extract todo definitions from plan object."""
        if 'todo' in plan_obj:
            todo_val = plan_obj['todo'].get('value', {})
            if isinstance(todo_val, dict):
                for key in todo_val:
                    try:
                        self.defined_todos.add(int(key))
                    except ValueError:
                        pass
    
    def _extract_act_info(self, act_obj: dict, postfixes: list):
        """Extract info from act block."""
        # Postfixes on the act block itself can satisfy todos
        for op, val in postfixes:
            if op == 'satisfies':
                self.satisfied_todos.add(val)
        
        # Check for satisfies in nested content
        for key, val_obj in act_obj.items():
            if isinstance(val_obj, dict) and 'postfixes' in val_obj:
                for op, val in val_obj['postfixes']:
                    if op == 'satisfies':
                        self.satisfied_todos.add(val)
        
        # Extract tool calls: act { call ↦ { tool ↦ name • id ↦ "xxx" } }
        if 'call' in act_obj:
            call_val = act_obj['call'].get('value', {})
            if isinstance(call_val, dict):
                # Get tool name
                if 'tool' in call_val:
                    tool_name = call_val['tool'].get('value', '')
                    if isinstance(tool_name, str) and tool_name:
                        self.called_tools.add(tool_name)
                
                # Get call ID
                if 'id' in call_val:
                    call_id = call_val['id'].get('value', '')
                    if isinstance(call_id, str) and call_id:
                        self.call_ids.add(call_id)
    
    def _extract_result_info(self, result_obj: dict, postfixes: list):
        """Extract info from result block."""
        # Check for tag on data value and collect result IDs
        if 'data' in result_obj:
            data_obj = result_obj['data']
            if isinstance(data_obj, dict) and 'postfixes' in data_obj:
                has_tag = False
                for op, val in data_obj['postfixes']:
                    if op == 'tag':
                        has_tag = True
                        self.result_ids.add(val)
                if not has_tag:
                    self.warnings.append("Result data value should have a 🏷 tag")
        
        # Also check for tag on the result block itself (less preferred but valid)
        for op, val in postfixes:
            if op == 'tag':
                self.result_ids.add(val)


class TaskVerifier:
    """Deterministic verifier for TASK format traces using actual parsing."""
    
    def __init__(self):
        pass
    
    def parse(self, trace: str) -> dict:
        """Parse trace and return analysis."""
        parser = TaskParser(trace)
        return parser.parse()
    
    def compute_reward(self, trace: str) -> tuple[float, dict]:
        """Compute total reward and breakdown based on parsing."""
        result = self.parse(trace)
        rewards = {}
        total = 0.0
        
        # Parse success (fundamental)
        parse_valid = result['valid']
        rewards['parse_valid'] = 1.0 if parse_valid else -3.0
        total += rewards['parse_valid']
        
        # Block structure
        block_types = [b[0] for b in result['blocks']]
        
        has_plan = 'plan' in block_types
        rewards['has_plan'] = 0.3 if has_plan else -0.5
        total += rewards['has_plan']
        
        has_response = 'response' in block_types
        rewards['has_response'] = 0.5 if has_response else -1.5
        total += rewards['has_response']
        
        # Structure order: plan should come before response
        if has_plan and has_response:
            plan_idx = block_types.index('plan')
            response_idx = block_types.index('response')
            order_ok = plan_idx < response_idx
            rewards['structure_order'] = 0.2 if order_ok else -0.3
        else:
            rewards['structure_order'] = 0.0
        total += rewards['structure_order']
        
        # Todo satisfaction
        defined_todos = result['defined_todos']
        satisfied_todos = result['satisfied_todos']
        
        if defined_todos:
            # Check todos start at 1
            if min(defined_todos) == 1:
                rewards['todos_start_at_1'] = 0.3
            else:
                rewards['todos_start_at_1'] = -0.5
            total += rewards['todos_start_at_1']
            
            # Check all todos satisfied
            unsatisfied = defined_todos - satisfied_todos
            if not unsatisfied:
                rewards['todos_satisfied'] = 0.5
            else:
                ratio = len(satisfied_todos) / len(defined_todos)
                rewards['todos_satisfied'] = (ratio * 0.5) - 0.3  # Partial credit minus penalty
            total += rewards['todos_satisfied']
        else:
            rewards['todos_start_at_1'] = 0.0
            rewards['todos_satisfied'] = 0.0
        
        # Tag/reference validity
        defined_tags = result['defined_tags']
        referenced_tags = result['referenced_tags']
        
        # Filter standard tags (usr1, sys1, etc.) from orphan check
        filtered_refs = {t for t in referenced_tags if not re.match(r'^(usr|sys)\d+$', t)}
        orphan_refs = filtered_refs - defined_tags
        
        if not orphan_refs:
            rewards['refs_valid'] = 0.3
        else:
            # Penalty proportional to number of orphans
            penalty = min(len(orphan_refs) * 0.1, 0.5)
            rewards['refs_valid'] = -penalty
        total += rewards['refs_valid']
        
        # Tool call validity - check for hallucinated tools
        defined_tools = result['defined_tools']
        called_tools = result['called_tools']
        
        if called_tools:
            if defined_tools:
                # Check if all called tools are defined
                hallucinated_tools = called_tools - defined_tools
                if not hallucinated_tools:
                    rewards['tools_valid'] = 0.5  # Good reward for not hallucinating
                else:
                    # Heavy penalty for hallucinating tools
                    penalty = min(len(hallucinated_tools) * 0.5, 1.5)
                    rewards['tools_valid'] = -penalty
            else:
                # Called tools but none defined - all are hallucinated
                rewards['tools_valid'] = -1.5
        else:
            # No tool calls - neutral (tools might not be needed)
            rewards['tools_valid'] = 0.0
        total += rewards['tools_valid']
        
        # Result/call ID matching - ensure results correspond to calls
        call_ids = result['call_ids']
        result_ids = result['result_ids']
        
        if call_ids or result_ids:
            # Check for orphan results (results without matching calls)
            orphan_results = result_ids - call_ids
            # Check for missing results (calls without matching results)
            missing_results = call_ids - result_ids
            
            if not orphan_results and not missing_results:
                rewards['call_result_match'] = 0.4
            else:
                penalty = 0.0
                if orphan_results:
                    # Hallucinated results
                    penalty += min(len(orphan_results) * 0.3, 0.6)
                if missing_results:
                    # Missing results for calls (less severe - might be intentional)
                    penalty += min(len(missing_results) * 0.1, 0.3)
                rewards['call_result_match'] = -penalty
        else:
            rewards['call_result_match'] = 0.0
        total += rewards['call_result_match']
        
        # Warnings penalty (minor issues)
        warning_count = len(result['warnings'])
        rewards['warnings'] = -min(warning_count * 0.05, 0.3)
        total += rewards['warnings']
        
        # Error count penalty (beyond parse failure)
        error_count = len(result['errors'])
        if error_count > 0 and parse_valid:  # Additional errors beyond parse failure
            rewards['additional_errors'] = -min(error_count * 0.2, 0.5)
            total += rewards['additional_errors']
        
        return total, rewards
    
    def detailed_check(self, trace: str) -> dict:
        """Return detailed parsing results for debugging."""
        result = self.parse(trace)
        score, rewards = self.compute_reward(trace)
        return {
            'score': score,
            'rewards': rewards,
            'blocks': [(b[0], len(str(b[1]))) for b in result['blocks']],
            'defined_tags': list(result['defined_tags']),
            'referenced_tags': list(result['referenced_tags']),
            'defined_todos': list(result['defined_todos']),
            'satisfied_todos': list(result['satisfied_todos']),
            'defined_tools': list(result['defined_tools']),
            'called_tools': list(result['called_tools']),
            'call_ids': list(result['call_ids']),
            'result_ids': list(result['result_ids']),
            'errors': result['errors'],
            'warnings': result['warnings'],
        }


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
    
    Supports three backends:
    - "openai": Uses OpenAI API (requires openai package)
    - "vllm": Uses local vLLM in-process (conflicts with distributed training!)
    - "vllm_server": Connects to external vLLM server (recommended for distributed training)
    
    For distributed training (accelerate launch), use either:
    1. --judge-backend openai (simplest)
    2. --judge-backend vllm_server --judge-url http://localhost:8000
       After starting server: vllm serve MiniMaxAI/MiniMax-M2 --tensor-parallel-size 6
    
    For local models on H200 cluster:
    - 6x H200 (846GB) can run MiniMaxAI/MiniMax-M2 at BF16 or Llama-3.1-405B at FP8
    """
    
    def __init__(
        self,
        model: str = "MiniMaxAI/MiniMax-M2",
        backend: str = "vllm",  # "vllm", "vllm_server", or "openai"
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/judge",
        tensor_parallel_size: int = 6,  # Number of GPUs for judge
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",  # "bfloat16", "float16", "auto"
        server_url: Optional[str] = None,  # For vllm_server backend
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
        
        elif backend == "vllm_server":
            # Connect to external vLLM server via OpenAI-compatible API
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for vllm_server backend")
            base_url = server_url or "http://localhost:8000/v1"
            print(f"[Judge] Connecting to vLLM server at {base_url}")
            self.client = OpenAI(base_url=base_url, api_key="not-needed")
            self._llm = None
            self._sampling_params = None
            print(f"[Judge] Connected to vLLM server (model: {model})")
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm package required. Install with: pip install vllm")
            
            # Check if running under accelerate/distributed - vLLM conflicts with it
            is_distributed = os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_LAUNCHED")
            if is_distributed:
                print(f"[Judge] WARNING: Running under distributed training (accelerate/deepspeed)")
                print(f"[Judge] vLLM tensor parallelism conflicts with training distributed backend")
                print(f"[Judge] Options:")
                print(f"         1. Use --judge-backend openai instead")
                print(f"         2. Run vLLM as separate server: vllm serve {model}")
                print(f"         3. Reduce --judge-gpus to 1 (single GPU, may OOM on large models)")
                print(f"[Judge] Attempting single-GPU mode with reduced memory...")
                tensor_parallel_size = 1
                gpu_memory_utilization = 0.3  # Leave room for training
            
            print(f"[Judge] Loading {model} on {tensor_parallel_size} GPU(s)...")
            
            # Use enforce_eager to avoid CUDA graph conflicts
            self._llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                trust_remote_code=False,
                max_model_len=4096,  # Reduced for memory
                enforce_eager=True,  # Avoid CUDA graph conflicts with training
            )
            # Use guided JSON generation to enforce schema
            # vLLM uses guided_json parameter (dict or JSON string)
            try:
                self._sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=256,
                    guided_json=JUDGE_JSON_SCHEMA,
                )
                print(f"[Judge] Model loaded with guided JSON output")
            except TypeError:
                # Fallback for older vLLM versions without guided generation
                self._sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=256,
                )
                print(f"[Judge] Model loaded (no guided generation - older vLLM)")
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
            if self.backend in ("openai", "vllm_server"):
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
# On-the-fly Prompt Generator
# =============================================================================

# JSON Schema for judge output (used with vLLM guided generation)
JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "integer", "minimum": 0, "maximum": 10},
        "relevance": {"type": "integer", "minimum": 0, "maximum": 10},
        "reasoning": {"type": "integer", "minimum": 0, "maximum": 10},
        "completion": {"type": "integer", "minimum": 0, "maximum": 10},
        "quality": {"type": "integer", "minimum": 0, "maximum": 10},
        "explanation": {"type": "string"}
    },
    "required": ["coherence", "relevance", "reasoning", "completion", "quality"]
}

# JSON Schema for prompt generator output
PROMPT_GEN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "system_prompt": {"type": "string", "minLength": 10, "maxLength": 500},
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                    "description": {"type": "string"},
                    "params": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                            "enum": ["string", "number", "integer", "boolean", "array", "object"]
                        }
                    }
                },
                "required": ["name", "description", "params"]
            },
            "maxItems": 3
        },
        "user_request": {"type": "string", "minLength": 10},
        "domain": {"type": "string", "enum": ["coding", "research", "writing", "math", "planning", "general", "data_analysis"]},
        "complexity": {"type": "string", "enum": ["simple", "medium", "complex"]}
    },
    "required": ["system_prompt", "tools", "user_request", "domain", "complexity"]
}

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

## Output Format (JSON)

{
  "system_prompt": "You are a helpful assistant that...",
  "tools": [
    {
      "name": "tool_name_snake_case",
      "description": "What it does",
      "params": {"param1": "string", "param2": "integer", "param3": "boolean"}
    }
  ],
  "user_request": "The user's task or question",
  "domain": "coding|research|writing|math|planning|general|data_analysis",
  "complexity": "simple|medium|complex"
}

Generate diverse, realistic scenarios. Vary domains, complexity, and whether tools are needed."""

PROMPT_GEN_USER = """Generate a training scenario.

Requirements:
- Domain hint: {domain}
- Include tools: {include_tools}
- Complexity: {complexity}

JSON only:"""


class PromptGenerator:
    """Generates training prompts on-the-fly using an LLM.
    
    Can share the same vLLM instance as the judge for efficiency.
    """
    
    DOMAINS = ['coding', 'research', 'writing', 'math', 'planning', 'general', 'data_analysis']
    COMPLEXITIES = ['simple', 'medium', 'complex']
    
    def __init__(
        self,
        llm_instance=None,  # Can share vLLM instance with judge
        model: str = "MiniMaxAI/MiniMax-M2",
        backend: str = "vllm",
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 6,
        dtype: str = "bfloat16",
        server_url: Optional[str] = None,  # For vllm_server backend
    ):
        self.backend = backend
        self.model = model
        self.client = None
        self._llm = None
        self._owns_llm = False
        
        if llm_instance is not None:
            # Share existing vLLM instance
            self._llm = llm_instance
        elif backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm required for prompt generation")
            print(f"[PromptGen] Loading {model}...")
            self._llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,
                dtype=dtype,
                trust_remote_code=False,
                max_model_len=4096,
            )
            self._owns_llm = True
        elif backend == "vllm_server":
            # Connect to external vLLM server via OpenAI-compatible API
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for vllm_server backend")
            base_url = server_url or "http://localhost:8000/v1"
            print(f"[PromptGen] Connecting to vLLM server at {base_url}")
            self.client = OpenAI(base_url=base_url, api_key="not-needed")
        elif backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai required")
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        
        # Use guided JSON generation to enforce schema
        # This ensures tool param types are valid JSON Schema types
        if VLLM_AVAILABLE:
            try:
                self._sampling_params = SamplingParams(
                    temperature=0.9,  # Higher temp for diversity
                    max_tokens=512,
                    guided_json=PROMPT_GEN_JSON_SCHEMA,
                )
            except TypeError:
                # Fallback for older vLLM versions
                self._sampling_params = SamplingParams(
                    temperature=0.9,
                    max_tokens=512,
                )
        else:
            self._sampling_params = None
        
        # Stats
        self.generated = 0
    
    def _parse_scenario(self, text: str) -> Optional[dict]:
        """Parse JSON scenario from LLM output.
        
        With guided JSON generation, output should always be valid JSON.
        Fallbacks are for non-vLLM backends or edge cases.
        """
        if not text:
            return None
        
        # With guided generation, this should always work
        try:
            scenario = json.loads(text)
            # Validate tool param types are correct
            if 'tools' in scenario:
                valid_types = {'string', 'number', 'integer', 'boolean', 'array', 'object'}
                for tool in scenario['tools']:
                    if 'params' in tool:
                        for param_name, param_type in tool['params'].items():
                            if param_type not in valid_types:
                                # Fix invalid type to string
                                tool['params'][param_name] = 'string'
            return scenario
        except json.JSONDecodeError:
            pass
        
        # Fallback for non-guided backends: try extracting from code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        return None
    
    def _format_tools_task(self, tools: list[dict]) -> str:
        """Format tools into TASK format with proper JSON Schema types."""
        if not tools:
            return ""
        
        # Valid JSON Schema types
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
                    # Ensure type is valid JSON Schema type
                    if ptype not in VALID_TYPES:
                        ptype = 'string'  # Default to string for invalid types
                    param_items.append(f"{pname} ↦ {{ type ↦ {ptype} }}")
                params_str = f" • params ↦ {{ {' • '.join(param_items)} }}"
            
            tool_str = f'tool {{ name ↦ {name} • description ↦ "{desc}"{params_str} }}'
            result.append(tool_str)
        
        return '\n'.join(result)
    
    def _format_prompt(self, scenario: dict) -> str:
        """Convert scenario to TASK-format prompt."""
        system = scenario.get('system_prompt', 'You are a helpful assistant.')
        user = scenario.get('user_request', 'Help me.')
        tools = scenario.get('tools', [])
        
        parts = [
            f'<|im_start|>system',
            f'system「{system}」🏷 sys1',
        ]
        
        if tools:
            parts.append(self._format_tools_task(tools))
        
        parts.extend([
            '<|im_end|>',
            '',
            '<|im_start|>user',
            f'user「{user}」🏷 usr1',
            '<|im_end|>',
            '',
            '<|im_start|>assistant',
        ])
        
        return '\n'.join(parts)
    
    def generate_one(
        self,
        domain: Optional[str] = None,
        include_tools: Optional[bool] = None,
        complexity: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a single prompt."""
        # Randomize if not specified
        domain = domain or random.choice(self.DOMAINS)
        include_tools = include_tools if include_tools is not None else random.random() < 0.6
        complexity = complexity or random.choice(self.COMPLEXITIES)
        
        user_msg = PROMPT_GEN_USER.format(
            domain=domain,
            include_tools="yes (1-3 tools)" if include_tools else "no",
            complexity=complexity,
        )
        
        if self._llm is not None:
            # Use in-process vLLM
            prompt = f"""<|im_start|>system
{PROMPT_GEN_SYSTEM}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
            outputs = self._llm.generate([prompt], self._sampling_params)
            response_text = outputs[0].outputs[0].text
        elif self.client is not None:
            # Use OpenAI-compatible API (openai or vllm_server)
            extra_body = None
            response_format = None
            
            if self.backend == "vllm_server":
                # vLLM: try without guided_json first as it can cause issues
                # Just ask for JSON in the prompt and parse it
                pass
            else:
                # OpenAI uses response_format
                response_format = {"type": "json_object"}
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PROMPT_GEN_SYSTEM},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format=response_format,
                    temperature=0.9,
                    max_tokens=1024,  # Increased from 512
                    extra_body=extra_body,
                )
            except Exception as e:
                print(f"    [PromptGen] API error: {e}")
                return None
            
            # Debug: inspect the full response
            if not response.choices:
                print(f"    [PromptGen] No choices in response: {response}")
                return None
            
            choice = response.choices[0]
            response_text = choice.message.content
            finish_reason = choice.finish_reason
            
            if response_text is None:
                print(f"    [PromptGen] None content, finish_reason={finish_reason}")
                print(f"    [PromptGen] Full message: {choice.message}")
                print(f"    [PromptGen] Full choice: {choice}")
                # Try to get any other info
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    print(f"    [PromptGen] Has tool_calls instead of content")
                if hasattr(choice.message, 'refusal') and choice.message.refusal:
                    print(f"    [PromptGen] Refusal: {choice.message.refusal}")
                return None
            
            if finish_reason == "length":
                print(f"    [PromptGen] Warning: truncated (hit max_tokens)")
                # Still try to parse partial JSON
        else:
            raise RuntimeError("PromptGenerator has no LLM or client configured")
        
        if not response_text:
            print(f"    [PromptGen] Warning: Empty response")
            return None
        
        scenario = self._parse_scenario(response_text)
        if scenario:
            self.generated += 1
            return self._format_prompt(scenario)
        else:
            # Debug: show what failed to parse
            preview = response_text[:200].replace('\n', ' ') if response_text else "None"
            print(f"    [PromptGen] Failed to parse: {preview}...")
        return None
    
    def generate_batch(self, n: int, batch_size: int = 20, **kwargs) -> list[str]:
        """Generate multiple prompts.
        
        Args:
            n: Total number of prompts to generate
            batch_size: Number of scenarios to request per API call (default 20)
        """
        if self._llm is not None:
            # Batch generation for efficiency
            prompts_to_gen = []
            for _ in range(n):
                domain = kwargs.get('domain') or random.choice(self.DOMAINS)
                include_tools = kwargs.get('include_tools')
                if include_tools is None:
                    include_tools = random.random() < 0.6
                complexity = kwargs.get('complexity') or random.choice(self.COMPLEXITIES)
                
                user_msg = PROMPT_GEN_USER.format(
                    domain=domain,
                    include_tools="yes (1-3 tools)" if include_tools else "no",
                    complexity=complexity,
                )
                
                prompt = f"""<|im_start|>system
{PROMPT_GEN_SYSTEM}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
                prompts_to_gen.append(prompt)
            
            outputs = self._llm.generate(prompts_to_gen, self._sampling_params)
            
            results = []
            for output in outputs:
                response_text = output.outputs[0].text
                scenario = self._parse_scenario(response_text)
                if scenario:
                    self.generated += 1
                    results.append(self._format_prompt(scenario))
            
            return results
        else:
            # Use batch API calls for OpenAI/vllm_server
            results = []
            failures = 0
            num_batches = (n + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                remaining = n - len(results)
                current_batch_size = min(batch_size, remaining)
                
                print(f"    [Batch {batch_idx+1}/{num_batches}] Requesting {current_batch_size} scenarios...")
                
                try:
                    batch_results = self._generate_batch_api(current_batch_size, **kwargs)
                    for prompt in batch_results:
                        if prompt:
                            results.append(prompt)
                            preview = prompt[:100].replace('\n', ' ')
                            print(f"      [{len(results)}/{n}] ✓ {preview}...")
                    
                    batch_failures = current_batch_size - len(batch_results)
                    if batch_failures > 0:
                        failures += batch_failures
                        print(f"      {batch_failures} failed to parse in this batch")
                        
                except Exception as e:
                    failures += current_batch_size
                    print(f"      Batch failed: {e}")
                
                if len(results) >= n:
                    break
            
            print(f"    Done: {len(results)} prompts generated ({failures} failures)")
            return results[:n]  # Trim to exact count
    
    def _generate_batch_api(self, batch_size: int, **kwargs) -> list[str]:
        """Generate multiple scenarios in a single API call."""
        if self.client is None:
            raise RuntimeError("No client configured for batch API generation")
        
        # Build prompt asking for multiple scenarios
        domain = kwargs.get('domain') or random.choice(self.DOMAINS)
        include_tools = kwargs.get('include_tools')
        if include_tools is None:
            include_tools = random.random() < 0.6
        complexity = kwargs.get('complexity') or random.choice(self.COMPLEXITIES)
        
        batch_prompt = f"""Generate {batch_size} diverse training scenarios. Return a JSON array.

Requirements for ALL scenarios:
- Domain: vary between {', '.join(self.DOMAINS)}
- Include tools: vary (some with 1-3 tools, some without)
- Complexity: vary between simple, medium, complex
- Make each scenario DIFFERENT from the others

Return ONLY a JSON array like:
[
  {{"system_prompt": "...", "tools": [...], "user_request": "...", "domain": "...", "complexity": "..."}},
  {{"system_prompt": "...", "tools": [], "user_request": "...", "domain": "...", "complexity": "..."}},
  ...
]

{batch_size} scenarios, JSON array only:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_GEN_SYSTEM},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.9,
                max_tokens=batch_size * 300,  # ~300 tokens per scenario
            )
        except Exception as e:
            print(f"    [PromptGen] API error: {e}")
            return []
        
        response_text = response.choices[0].message.content if response.choices else None
        if not response_text:
            print(f"    [PromptGen] Empty response")
            return []
        
        # Parse JSON array
        try:
            # Try direct parse
            scenarios = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract array from response
            match = re.search(r'\[[\s\S]*\]', response_text)
            if match:
                try:
                    scenarios = json.loads(match.group(0))
                except:
                    print(f"    [PromptGen] Failed to parse array: {response_text[:200]}...")
                    return []
            else:
                print(f"    [PromptGen] No array found: {response_text[:200]}...")
                return []
        
        if not isinstance(scenarios, list):
            print(f"    [PromptGen] Response is not an array")
            return []
        
        # Format each scenario
        results = []
        for scenario in scenarios:
            try:
                if isinstance(scenario, dict) and 'system_prompt' in scenario and 'user_request' in scenario:
                    formatted = self._format_prompt(scenario)
                    if formatted:
                        results.append(formatted)
                        self.generated += 1
            except Exception as e:
                pass  # Skip malformed scenarios
        
        return results
    
    def stats(self) -> dict:
        return {'generated': self.generated}


# =============================================================================
# Hybrid Reward Function
# =============================================================================

class HybridRewardFunction:
    """Combines deterministic verifiers with optional LLM judge."""
    
    def __init__(
        self,
        judge_rate: float = 0.0,  # Fraction of samples to judge (0 = disabled)
        judge_model: str = "MiniMaxAI/MiniMax-M2",
        judge_backend: str = "vllm_server",  # "vllm_server", "vllm", or "openai"
        judge_weight: float = 1.0,  # Weight of LLM score vs verifier score
        judge_gpus: int = 6,  # GPUs for judge model
        judge_dtype: str = "bfloat16",
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,  # URL for vllm_server backend
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
                server_url=server_url,
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


# Global instances (set in main)
_reward_fn: Optional[HybridRewardFunction] = None
_dashboard = None
_step_counter = 0
_is_main_process = True  # Set in main()


def reward_function_wrapper(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Wrapper for GRPO trainer compatibility with dashboard logging."""
    import time
    global _reward_fn, _dashboard, _step_counter, _is_main_process
    
    # DEBUG: Print when reward function is called
    t0 = time.time()
    print(f"\n[DEBUG] ===== REWARD FUNCTION CALLED =====", flush=True)
    print(f"[DEBUG] Num completions: {len(completions)}", flush=True)
    if completions:
        print(f"[DEBUG] Completion lengths (chars): {[len(c) for c in completions]}", flush=True)
        print(f"[DEBUG] First 300 chars of completion 0:\n{completions[0][:300]}", flush=True)
    
    verifier = TaskVerifier()
    
    if _reward_fn is None:
        print("[DEBUG] _reward_fn is None, using basic verifier", flush=True)
        # Fallback to basic verifier if not initialized
        return [verifier.compute_reward(c)[0] for c in completions]
    
    # Get rewards with dashboard logging
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Get verifier score and breakdown
        verifier_score, verifier_breakdown = verifier.compute_reward(completion)
        
        # Get judge score if enabled (only on main process to avoid duplicate API calls)
        judge_score = None
        judge_breakdown = None
        if _is_main_process and _reward_fn.judge and random.random() < _reward_fn.judge_rate:
            try:
                judge_score, judge_breakdown = _reward_fn.judge.judge(prompt, completion)
                if _dashboard:
                    _dashboard.log(f"[Judge] Score: {judge_score:.2f}")
            except Exception as e:
                if _dashboard:
                    _dashboard.log(f"[Judge] Error: {e}")
                judge_score = None
        
        # Compute final reward
        final_reward = verifier_score
        if judge_score is not None:
            # Judge score is already 0-1, scale to similar range as verifier
            final_reward += _reward_fn.judge_weight * (judge_score * 5 - 2.5)
        
        rewards.append(final_reward)
        
        # Log to dashboard (only on main process)
        if _is_main_process and _dashboard is not None:
            _step_counter += 1
            _dashboard.log_generation(
                step=_step_counter,
                prompt=prompt,
                output=completion,
                verifier_score=verifier_score,
                verifier_breakdown=verifier_breakdown,
                judge_score=judge_score,
            )
    
    print(f"[DEBUG] Reward computation took {time.time() - t0:.2f}s", flush=True)
    print(f"[DEBUG] Rewards: {rewards}", flush=True)
    return rewards


# =============================================================================
# Dataset and Prompt Building
# =============================================================================

def load_prompts(data_path: str, max_samples: int = None) -> list[dict]:
    """Load prompts from JSONL file.
    
    Supports two formats:
    1. {"prompt": "..."} - direct prompts from generate_prompts.py
    2. {"trace": "..."} - processed traces with special tokens
    """
    prompts = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                
                # Format 1: Direct prompt (from generate_prompts.py)
                if "prompt" in item:
                    prompt = item["prompt"]
                    # Wrap in chat template if not already
                    if not prompt.startswith("<|im_start|>"):
                        prompt = f"<|im_start|>system\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                    prompts.append({"prompt": prompt})
                    continue
                
                # Format 2: Trace format (from traces.processed.jsonl)
                trace = item.get("trace", "")
                if "<|im_start|>assistant" in trace:
                    prompt_part = trace.split("<|im_start|>assistant")[0]
                    prompt_part += "<|im_start|>assistant\n"
                    prompts.append({"prompt": prompt_part})
            except Exception as e:
                pass  # Skip malformed lines
    
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
    judge_model: str = "MiniMaxAI/MiniMax-M2"
    judge_backend: str = "vllm_server"  # "vllm_server", "vllm", or "openai"
    judge_url: str = "http://localhost:8000/v1"  # URL for vllm_server
    judge_weight: float = 1.0  # Weight of LLM score vs verifier score
    judge_gpus: int = 6  # GPUs for judge model (when using vllm)
    judge_dtype: str = "bfloat16"  # "bfloat16", "float16", "auto"
    openai_api_key: Optional[str] = None
    
    # Dynamic prompt generation
    generate_prompts: bool = False  # Generate prompts on-the-fly instead of using data file
    prompt_gen_count: int = 1000  # Number of prompts to generate (if generate_prompts=True)
    prompt_tool_rate: float = 0.6  # Fraction of generated prompts that include tools
    
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
    parser.add_argument("--judge-model", type=str, default="MiniMaxAI/MiniMax-M2",
                        help="Model for LLM judge (HF model ID for vllm, or OpenAI model name)")
    parser.add_argument("--judge-backend", type=str, default="vllm_server", 
                        choices=["vllm", "vllm_server", "openai"],
                        help="Backend: 'vllm_server' (external server, recommended), 'vllm' (in-process, conflicts with distributed!), 'openai' (API)")
    parser.add_argument("--judge-url", type=str, default="http://localhost:8000/v1",
                        help="URL for vllm_server backend (default: http://localhost:8000/v1)")
    parser.add_argument("--judge-weight", type=float, default=1.0,
                        help="Weight of LLM judge score relative to verifier")
    parser.add_argument("--judge-gpus", type=int, default=6,
                        help="Number of GPUs for judge model (vllm backend only)")
    parser.add_argument("--judge-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "auto"],
                        help="Data type for judge model")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key (for openai backend)")
    
    # Note: For prompt generation, use generate_prompts.py separately
    # This avoids NCCL timeout issues during distributed training
    
    # Dashboard
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable live training dashboard (requires 'rich' library)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable dashboard even if available")
    
    args = parser.parse_args()
    
    # Helper to only print on main process (routes to dashboard if active)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    def log(msg, force_print=False):
        global _dashboard
        if local_rank == 0:
            # Always print before dashboard is created, or if force_print
            if _dashboard is None or force_print:
                print(msg)
            # Also log to dashboard if it exists
            if _dashboard is not None:
                _dashboard.log(str(msg))
    
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
        judge_url=args.judge_url,
        judge_weight=args.judge_weight,
        judge_gpus=args.judge_gpus,
        judge_dtype=args.judge_dtype,
        openai_api_key=args.openai_api_key,
    )
    
    log("="*60)
    log("GRPO Training for TASK Format")
    log("="*60)
    log(f"Model: {config.model_path}")
    log(f"Data: {config.data_path}")
    log(f"Output: {config.output_dir}")
    log(f"Learning rate: {config.learning_rate}")
    log(f"Num generations per prompt: {config.num_generations}")
    log("-"*60)
    if config.judge_rate > 0:
        log(f"LLM Judge: ENABLED")
        log(f"  Backend: {config.judge_backend}")
        log(f"  Model: {config.judge_model}")
        log(f"  Judge rate: {config.judge_rate*100:.0f}% of samples")
        log(f"  Judge weight: {config.judge_weight}")
        if config.judge_backend == "vllm":
            log(f"  GPUs: {config.judge_gpus}")
            log(f"  Dtype: {config.judge_dtype}")
    else:
        log(f"LLM Judge: DISABLED (verifier only)")
    log("="*60)
    
    # Determine if this is the main process (rank 0)
    global _dashboard, _is_main_process
    _is_main_process = (local_rank == 0)
    
    # Initialize dashboard if requested (only on main process)
    use_dashboard = args.dashboard or (DASHBOARD_AVAILABLE and not args.no_dashboard)
    if use_dashboard and _is_main_process:
        if not DASHBOARD_AVAILABLE:
            log("Warning: Dashboard requested but 'rich' not installed. Run: pip install rich")
            _dashboard = None
        else:
            _dashboard = create_dashboard(use_rich=True)
            log("\n✓ Dashboard enabled (live TUI)")
    elif not _is_main_process:
        _dashboard = None  # Explicitly disable on non-main processes
    
    # Initialize hybrid reward function
    log("\nInitializing reward function...")
    _reward_fn = HybridRewardFunction(
        judge_rate=config.judge_rate,
        judge_model=config.judge_model,
        judge_backend=config.judge_backend,
        judge_weight=config.judge_weight,
        judge_gpus=config.judge_gpus,
        judge_dtype=config.judge_dtype,
        api_key=config.openai_api_key,
        server_url=config.judge_url,
    )
    
    # Load model and tokenizer
    log("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts from file
    # For generating prompts, use generate_prompts.py first (avoids NCCL timeout)
    log("\nLoading prompts from file...")
    prompts = load_prompts(config.data_path, config.max_samples)
    
    if len(prompts) == 0:
        log("ERROR: No prompts loaded!")
        log("Generate prompts first with:")
        log(f"  python generate_prompts.py --count 1000 --output prompts.jsonl")
        log(f"Then run: accelerate launch rl_train.py --data prompts.jsonl")
        return
    
    dataset = Dataset.from_list(prompts)
    
    # GRPO config
    # Note: For multi-GPU, FSDP + generation is extremely slow.
    # Options:
    # 1. Use single GPU (model fits on one H200)
    # 2. Use use_vllm=True for fast generation with separate vLLM process
    # 3. Use DeepSpeed ZeRO-2 (keeps full model for inference)
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
        # Reduce sync overhead - log less frequently
        logging_first_step=True,
    )
    
    # Trainer callbacks
    callbacks = []
    if _dashboard:
        callbacks.append(DashboardCallback(_dashboard))
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function_wrapper,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    log("\nStarting GRPO training...")
    print(f"[DEBUG] Dataset size: {len(dataset)}", flush=True)
    print(f"[DEBUG] Num epochs: {config.num_train_epochs}", flush=True)
    print(f"[DEBUG] Batch size: {config.per_device_train_batch_size}", flush=True)
    print(f"[DEBUG] Num generations: {config.num_generations}", flush=True)
    print(f"[DEBUG] Max new tokens: {config.max_new_tokens}", flush=True)
    
    # Start dashboard
    total_steps = len(dataset) * config.num_train_epochs
    print(f"[DEBUG] Total steps: {total_steps}", flush=True)
    if _dashboard:
        _dashboard.start(total_steps=total_steps)
        print("[DEBUG] Dashboard started", flush=True)
    
    print("[DEBUG] Calling trainer.train()...", flush=True)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        # Stop dashboard
        if _dashboard:
            _dashboard.stop()
        
        # Print stats (only on main process)
        if _is_main_process:
            print("\n" + "="*60)
            print("Training Statistics")
            print("="*60)
            
            if _reward_fn:
                stats = _reward_fn.get_stats()
                if stats.get('judge_enabled'):
                    print(f"Judge API calls: {stats.get('calls', 0)}")
                    print(f"Judge cache hits: {stats.get('cache_hits', 0)}")
                    print(f"Judge cache rate: {stats.get('cache_rate', 0)*100:.1f}%")
    
    # Save (trainer handles distributed saving)
    log(f"\nSaving model to {config.output_dir}/final...")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    log("Done!")


if __name__ == "__main__":
    main()

