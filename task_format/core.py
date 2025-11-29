from __future__ import annotations

import json
import random
import re
from typing import Optional


class ParseError(Exception):
    """Error during TASK format parsing."""

    def __init__(self, message: str, position: int = -1):
        self.message = message
        self.position = position
        super().__init__(f"{message} at position {position}" if position >= 0 else message)


class TaskParser:
    """Recursive descent parser for TASK format traces."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)
        self.blocks = []
        self.defined_tags = set()
        self.referenced_tags = set()
        self.defined_todos = set()
        self.satisfied_todos = set()
        self.defined_tools = set()
        self.called_tools = set()
        self.call_ids = set()
        self.result_ids = set()
        self.errors = []
        self.warnings = []

    def parse(self) -> dict:
        """Parse the trace and return analysis results."""
        try:
            self._parse_trace()
        except ParseError as e:
            self.errors.append(str(e))

        return {
            "blocks": self.blocks,
            "defined_tags": self.defined_tags,
            "referenced_tags": self.referenced_tags,
            "defined_todos": self.defined_todos,
            "satisfied_todos": self.satisfied_todos,
            "defined_tools": self.defined_tools,
            "called_tools": self.called_tools,
            "call_ids": self.call_ids,
            "result_ids": self.result_ids,
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": len(self.errors) == 0,
        }

    def _peek(self, n: int = 1) -> str:
        return self.text[self.pos : self.pos + n]

    def _advance(self, n: int = 1) -> str:
        result = self.text[self.pos : self.pos + n]
        self.pos += n
        return result

    def _skip_whitespace(self):
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _at_end(self) -> bool:
        return self.pos >= self.length

    def _match(self, s: str) -> bool:
        return self.text[self.pos : self.pos + len(s)] == s

    def _expect(self, s: str):
        if not self._match(s):
            raise ParseError(f"Expected '{s}'", self.pos)
        self._advance(len(s))

    def _parse_trace(self):
        while not self._at_end():
            self._skip_whitespace()
            if self._at_end():
                break

            block_type = self._try_parse_block_keyword()
            if block_type:
                self._parse_block(block_type)
            else:
                self._skip_to_next_block()

    def _try_parse_block_keyword(self) -> Optional[str]:
        keywords = ["system", "tool", "user", "plan", "act", "result", "response"]
        for kw in keywords:
            if self._match(kw):
                end_pos = self.pos + len(kw)
                if end_pos >= self.length or not self.text[end_pos].isalnum():
                    self._advance(len(kw))
                    return kw
        return None

    def _skip_to_next_block(self):
        keywords = ["system", "tool", "user", "plan", "act", "result", "response"]
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
        self._skip_whitespace()

        content = None
        postfixes = []

        if block_type in ("system", "user", "response"):
            content = self._parse_string()
            postfixes = self._parse_postfixes()
        elif block_type in ("tool", "plan", "act", "result"):
            content = self._parse_object()
            postfixes = self._parse_postfixes()

            if block_type == "tool" and isinstance(content, dict):
                self._extract_tool_info(content)
            elif block_type == "plan" and isinstance(content, dict):
                self._extract_todos(content)
            elif block_type == "act" and isinstance(content, dict):
                self._extract_act_info(content, postfixes)
            elif block_type == "result" and isinstance(content, dict):
                self._extract_result_info(content, postfixes)

        self.blocks.append((block_type, content, postfixes))

    def _parse_string(self) -> str:
        self._skip_whitespace()

        if self._match("「"):
            return self._parse_cjk_string()
        elif self._match('"'):
            return self._parse_quoted_string()
        else:
            return self._parse_unquoted_word()

    def _parse_cjk_string(self) -> str:
        self._expect("「")
        result = []
        depth = 1
        while not self._at_end() and depth > 0:
            ch = self._peek()
            if ch == "「":
                depth += 1
                result.append(self._advance())
            elif ch == "」":
                depth -= 1
                if depth > 0:
                    result.append(self._advance())
                else:
                    self._advance()
            else:
                result.append(self._advance())
        return "".join(result)

    def _parse_quoted_string(self) -> str:
        self._expect('"')
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch == '"':
                self._advance()
                break
            elif ch == "\\" and self.pos + 1 < self.length:
                self._advance()
                result.append(self._advance())
            else:
                result.append(self._advance())
        return "".join(result)

    def _parse_unquoted_word(self) -> str:
        result = []
        while not self._at_end():
            ch = self._peek()
            if ch in ' \t\n\r{}[]•↦🏷※⊨𝑝「」"':
                break
            result.append(self._advance())
        return "".join(result)

    def _parse_object(self) -> dict:
        self._skip_whitespace()
        if not self._match("{"):
            raise ParseError("Expected '{'", self.pos)
        self._advance()

        result = {}
        while True:
            self._skip_whitespace()
            if self._match("}"):
                self._advance()
                break
            if self._at_end():
                self.errors.append(f"Unclosed object at position {self.pos}")
                break

            key = self._parse_key()
            self._skip_whitespace()

            if self._match("↦"):
                self._advance()
            else:
                self.errors.append(f"Expected '↦' after key '{key}' at position {self.pos}")
                self._skip_to_separator()
                continue

            self._skip_whitespace()
            value = self._parse_value()
            postfixes = self._parse_postfixes()
            result[key] = {"value": value, "postfixes": postfixes}

            self._skip_whitespace()
            if self._match("•"):
                self._advance()
            elif self._match("}"):
                continue
            elif not self._at_end():
                self._skip_to_separator()

        return result

    def _parse_array(self) -> list:
        self._skip_whitespace()
        if not self._match("["):
            raise ParseError("Expected '['", self.pos)
        self._advance()

        result = []
        while True:
            self._skip_whitespace()
            if self._match("]"):
                self._advance()
                break
            if self._at_end():
                self.errors.append(f"Unclosed array at position {self.pos}")
                break

            value = self._parse_value()
            result.append(value)

            self._skip_whitespace()
            if self._match("•"):
                self._advance()
            elif self._match("]"):
                continue
            elif not self._at_end():
                self._skip_to_separator()

        return result

    def _parse_key(self) -> str:
        return self._parse_string()

    def _parse_value(self):
        self._skip_whitespace()

        if self._match("{"):
            return self._parse_object()
        if self._match("["):
            return self._parse_array()
        if self._match("「"):
            return self._parse_cjk_string()
        if self._match('"'):
            return self._parse_quoted_string()
        if self._match(("-", "+")):
            return self._parse_number()

        next_char = self._peek()
        if next_char.isdigit():
            return self._parse_number()
        word = self._parse_unquoted_word()
        if word in {"true", "false"}:
            return word == "true"
        if word == "null":
            return None
        return word

    def _parse_number(self):
        num_str = []
        has_decimal = False
        has_exponent = False
        while not self._at_end():
            ch = self._peek()
            if ch.isdigit():
                num_str.append(self._advance())
            elif ch == "." and not has_decimal:
                has_decimal = True
                num_str.append(self._advance())
            elif ch in ("e", "E") and not has_exponent:
                has_exponent = True
                num_str.append(self._advance())
                if self._peek() in ("+", "-"):
                    num_str.append(self._advance())
            else:
                break
        num = "".join(num_str)
        if not num:
            raise ParseError("Expected number", self.pos)
        return float(num) if "." in num or "e" in num.lower() else int(num)

    def _parse_postfixes(self) -> list[tuple[str, str]]:
        postfixes = []
        while True:
            self._skip_whitespace()
            if self._match("🏷"):
                self._advance()
                postfixes.append(("tag", self._parse_unquoted_word()))
            elif self._match("※"):
                self._advance()
                postfixes.append(("ref", self._parse_reference_value()))
            elif self._match("⊨"):
                self._advance()
                postfixes.append(("satisfies", self._parse_number()))
            elif self._match("𝑝"):
                self._advance()
                postfixes.append(("confidence", self._parse_number()))
            else:
                break
        return postfixes

    def _parse_reference_value(self):
        self._skip_whitespace()
        if self._match("["):
            return self._parse_array()
        return self._parse_unquoted_word()

    def _skip_to_separator(self):
        while not self._at_end():
            ch = self._peek()
            if ch in "•}]":
                break
            self._advance()

    def _extract_tool_info(self, tool_obj: dict):
        name_val = tool_obj.get("name", {})
        if isinstance(name_val, dict):
            tool_name = name_val.get("value")
            if tool_name:
                self.defined_tools.add(tool_name)

    def _extract_todos(self, plan_obj: dict):
        todo_section = plan_obj.get("todo", {})
        if isinstance(todo_section, dict):
            for key, val_obj in todo_section.items():
                try:
                    todo_id = int(key)
                    self.defined_todos.add(todo_id)
                except ValueError:
                    self.warnings.append(f"Invalid todo ID: {key}")

    def _extract_act_info(self, act_obj: dict, postfixes: list):
        for op, val in postfixes:
            if op == "satisfies":
                self.satisfied_todos.add(val)

        if "call" in act_obj:
            call_val = act_obj["call"].get("value", {})
            if isinstance(call_val, dict):
                if "tool" in call_val:
                    tool_name = call_val["tool"].get("value", "")
                    if isinstance(tool_name, str) and tool_name:
                        self.called_tools.add(tool_name)
                if "id" in call_val:
                    call_id = call_val["id"].get("value", "")
                    if isinstance(call_id, str) and call_id:
                        self.call_ids.add(call_id)

    def _extract_result_info(self, result_obj: dict, postfixes: list):
        if "data" in result_obj:
            data_obj = result_obj["data"]
            if isinstance(data_obj, dict) and "postfixes" in data_obj:
                has_tag = False
                for op, val in data_obj["postfixes"]:
                    if op == "tag":
                        has_tag = True
                        self.result_ids.add(val)
                if not has_tag:
                    self.warnings.append("Result data value should have a 🏷 tag")

        for op, val in postfixes:
            if op == "tag":
                self.result_ids.add(val)


class TaskVerifier:
    """Deterministic verifier for TASK format traces using actual parsing."""

    def parse(self, trace: str) -> dict:
        parser = TaskParser(trace)
        return parser.parse()

    def compute_reward(self, trace: str) -> tuple[float, dict]:
        result = self.parse(trace)
        rewards = {}
        total = 0.0

        parse_valid = result["valid"]
        rewards["parse_valid"] = 1.0 if parse_valid else -3.0
        total += rewards["parse_valid"]

        block_types = [b[0] for b in result["blocks"]]

        has_plan = "plan" in block_types
        rewards["has_plan"] = 0.3 if has_plan else -0.5
        total += rewards["has_plan"]

        has_response = "response" in block_types
        rewards["has_response"] = 0.5 if has_response else -1.5
        total += rewards["has_response"]

        if has_plan and has_response:
            plan_idx = block_types.index("plan")
            response_idx = block_types.index("response")
            order_ok = plan_idx < response_idx
            rewards["structure_order"] = 0.2 if order_ok else -0.3
        else:
            rewards["structure_order"] = 0.0
        total += rewards["structure_order"]

        defined_todos = result["defined_todos"]
        satisfied_todos = result["satisfied_todos"]

        if defined_todos:
            if min(defined_todos) == 1:
                rewards["todos_start_at_1"] = 0.3
            else:
                rewards["todos_start_at_1"] = -0.5
            total += rewards["todos_start_at_1"]

            unsatisfied = defined_todos - satisfied_todos
            if not unsatisfied:
                rewards["todos_satisfied"] = 0.5
            else:
                ratio = len(satisfied_todos) / len(defined_todos)
                rewards["todos_satisfied"] = (ratio * 0.5) - 0.3
            total += rewards["todos_satisfied"]
        else:
            rewards["todos_start_at_1"] = 0.0
            rewards["todos_satisfied"] = 0.0

        defined_tags = result["defined_tags"]
        referenced_tags = result["referenced_tags"]
        filtered_refs = {t for t in referenced_tags if not re.match(r"^(usr|sys)\d+$", t)}
        orphan_refs = filtered_refs - defined_tags

        if not orphan_refs:
            rewards["refs_valid"] = 0.3
        else:
            penalty = min(len(orphan_refs) * 0.1, 0.5)
            rewards["refs_valid"] = -penalty
        total += rewards["refs_valid"]

        defined_tools = result["defined_tools"]
        called_tools = result["called_tools"]

        if called_tools:
            if defined_tools:
                hallucinated_tools = called_tools - defined_tools
                if not hallucinated_tools:
                    rewards["tools_valid"] = 0.5
                else:
                    penalty = min(len(hallucinated_tools) * 0.5, 1.5)
                    rewards["tools_valid"] = -penalty
            else:
                rewards["tools_valid"] = -1.5
        else:
            rewards["tools_valid"] = 0.0
        total += rewards["tools_valid"]

        call_ids = result["call_ids"]
        result_ids = result["result_ids"]

        if call_ids or result_ids:
            orphan_results = result_ids - call_ids
            missing_results = call_ids - result_ids
            if not orphan_results and not missing_results:
                rewards["call_result_match"] = 0.4
            else:
                penalty = 0.0
                if orphan_results:
                    penalty += min(len(orphan_results) * 0.3, 0.6)
                if missing_results:
                    penalty += min(len(missing_results) * 0.1, 0.3)
                rewards["call_result_match"] = -penalty
        else:
            rewards["call_result_match"] = 0.0
        total += rewards["call_result_match"]

        warning_count = len(result["warnings"])
        rewards["warnings"] = -min(warning_count * 0.05, 0.3)
        total += rewards["warnings"]

        error_count = len(result["errors"])
        if error_count > 0 and parse_valid:
            rewards["additional_errors"] = -min(error_count * 0.2, 0.5)
            total += rewards["additional_errors"]

        if has_response:
            response_idx = block_types.index("response")
            blocks_after_response = len(block_types) - response_idx - 1
            if blocks_after_response == 0:
                rewards["clean_ending"] = 0.3
            elif blocks_after_response <= 2:
                rewards["clean_ending"] = 0.1
            else:
                rewards["clean_ending"] = -0.5
            total += rewards["clean_ending"]

            if len(trace) > 100000:
                rewards["output_length"] = -1.0
                total += rewards["output_length"]

        return total, rewards

    def detailed_check(self, trace: str) -> dict:
        result = self.parse(trace)
        score, rewards = self.compute_reward(trace)
        return {
            "score": score,
            "rewards": rewards,
            "blocks": [(b[0], len(str(b[1]))) for b in result["blocks"]],
            "defined_tags": list(result["defined_tags"]),
            "referenced_tags": list(result["referenced_tags"]),
            "defined_todos": list(result["defined_todos"]),
            "satisfied_todos": list(result["satisfied_todos"]),
            "defined_tools": list(result["defined_tools"]),
            "called_tools": list(result["called_tools"]),
            "call_ids": list(result["call_ids"]),
            "result_ids": list(result["result_ids"]),
            "errors": result["errors"],
            "warnings": result["warnings"],
        }

