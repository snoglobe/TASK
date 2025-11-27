"""
Pydantic models for the TASK format structured output.
Used with OpenAI's structured output feature for synthetic data generation.

Note: OpenAI structured output requires all fields to be required (no Optional).
We use empty strings/lists as "not present" markers.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ============================================================================
# Tool Definitions (simplified for structured output)
# ============================================================================

class ToolParamDef(BaseModel):
    """A single parameter in a tool definition."""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type: string, number, boolean, or enum")
    enum_values: list[str] = Field(description="Enum values if type is enum, empty list otherwise")
    required: bool = Field(description="Whether the parameter is required")
    description: str = Field(description="Parameter description")


class ToolDef(BaseModel):
    """A tool definition available to the model."""
    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    params: list[ToolParamDef] = Field(description="Tool parameters")


# ============================================================================
# Messages
# ============================================================================

class SystemMessage(BaseModel):
    """System message setting context."""
    content: str = Field(description="System message content")
    tag: str = Field(description="Semantic tag, e.g. 'sys1' (empty string if none)")


class UserMessage(BaseModel):
    """User message/request."""
    content: str = Field(description="User message content")
    tag: str = Field(description="Message tag in format usrN, e.g. 'usr1'")


# ============================================================================
# Todo & Planning
# ============================================================================

class TodoItem(BaseModel):
    """A single todo item."""
    number: int = Field(description="Todo item number (1-indexed)")
    text: str = Field(description="Todo item description")


class PlanPhase(BaseModel):
    """Model's planning phase - creates the todo list."""
    todos: list[TodoItem] = Field(description="The todo list for this task")
    rationale: str = Field(description="Explanation of the approach")


# ============================================================================
# Action Phase
# ============================================================================

class ToolCallArg(BaseModel):
    """A single argument to a tool call."""
    name: str = Field(description="Argument name")
    value: str = Field(description="Argument value (as string)")


class ToolCall(BaseModel):
    """A tool call within an action phase."""
    tool: str = Field(description="Name of the tool to call")
    args: list[ToolCallArg] = Field(description="Tool arguments")
    id: str = Field(description="Unique ID for this call's result")
    refs: list[str] = Field(description="References to prior tags (※), empty list if none")
    satisfies: int = Field(description="Todo item number this satisfies (⊨), 0 if none")


class ThinkStep(BaseModel):
    """A thinking step within an action phase."""
    content: str = Field(description="The thought content")
    confidence: float = Field(description="Confidence value 0.0-1.0 (𝑝), use -1 if not applicable")
    refs: list[str] = Field(description="References to prior tags (※), empty list if none")
    tag: str = Field(description="Semantic tag for this thought (🏷), empty string if none")
    satisfies: int = Field(description="Todo item number this satisfies (⊨), 0 if none")


class ActPhase(BaseModel):
    """Model's action phase - can think and/or call tools."""
    think: ThinkStep = Field(description="Thinking step (use empty content if not thinking)")
    calls: list[ToolCall] = Field(description="Tool calls to make, empty list if just thinking")


# ============================================================================
# Results
# ============================================================================

class ResultPhase(BaseModel):
    """Tool result injected by the inference system."""
    data: str = Field(description="The tool result data")
    tag: str = Field(description="Tag matching the tool call ID (🏷)")


# ============================================================================
# Response
# ============================================================================

class ResponsePhase(BaseModel):
    """Model's final response to the user."""
    content: str = Field(description="Response content")
    refs: list[str] = Field(description="References to prior tags (※), empty list if none")
    satisfies: int = Field(description="Todo item number this satisfies (⊨), 0 if none")


# ============================================================================
# Complete Trace
# ============================================================================

class TraceStep(BaseModel):
    """A single step in a trace - can be plan, act, result, or response."""
    step_type: str = Field(description="Step type: 'plan', 'act', 'result', or 'response'")
    plan: PlanPhase = Field(description="Plan data (ignore if step_type != 'plan')")
    act: ActPhase = Field(description="Act data (ignore if step_type != 'act')")
    result: ResultPhase = Field(description="Result data (ignore if step_type != 'result')")
    response: ResponsePhase = Field(description="Response data (ignore if step_type != 'response')")


# Default empty values for unused fields
_EMPTY_THINK = ThinkStep(content="", confidence=-1, refs=[], tag="", satisfies=0)
_EMPTY_PLAN = PlanPhase(todos=[], rationale="")
_EMPTY_ACT = ActPhase(think=_EMPTY_THINK, calls=[])
_EMPTY_RESULT = ResultPhase(data="", tag="")
_EMPTY_RESPONSE = ResponsePhase(content="", refs=[], satisfies=0)


class Trace(BaseModel):
    """A complete TASK format trace."""
    system: SystemMessage = Field(description="System message")
    tools: list[ToolDef] = Field(description="Available tools")
    user: UserMessage = Field(description="User message/request")
    steps: list[TraceStep] = Field(description="Sequence of plan/act/result/response steps")

    def render(self) -> str:
        """Render the trace to TASK format string."""
        lines = []
        
        # System message
        tag = f" 🏷 {self.system.tag}" if self.system.tag else ""
        lines.append(f"system「{self.system.content}」{tag}")
        lines.append("")
        
        # Tools
        for tool in self.tools:
            lines.append("tool {")
            lines.append(f"    name ↦ {tool.name} •")
            if tool.params:
                lines.append("    params ↦ {")
                for i, param in enumerate(tool.params):
                    lines.append(f"        {param.name} ↦ {{")
                    param_parts = []
                    if param.type != "enum":
                        param_parts.append(f"            type ↦ {param.type}")
                    if param.enum_values:
                        enum_str = " • ".join(param.enum_values)
                        param_parts.append(f"            enum ↦ [ {enum_str} ]")
                    if not param.required:
                        param_parts.append("            required ↦ false")
                    if param.description:
                        param_parts.append(f"            description ↦ \"{param.description}\"")
                    lines.append(" •\n".join(param_parts))
                    sep = " •" if i < len(tool.params) - 1 else ""
                    lines.append(f"        }}{sep}")
                lines.append("    }")
            lines.append("}")
            lines.append("")
        
        # User message
        lines.append(f"user「{self.user.content}」🏷 {self.user.tag}")
        lines.append("")
        
        # Steps
        for step in self.steps:
            if step.step_type == "plan":
                lines.append("plan {")
                lines.append("    todo ↦ {")
                for i, item in enumerate(step.plan.todos):
                    sep = " •" if i < len(step.plan.todos) - 1 else ""
                    lines.append(f"        {item.number} ↦ \"{item.text}\"{sep}")
                lines.append("    } •")
                lines.append(f"    rationale ↦ \"{step.plan.rationale}\"")
                lines.append("}")
                
            elif step.step_type == "act":
                lines.append("act {")
                parts = []
                if step.act.think.content:
                    t = step.act.think
                    think_str = f"    think ↦ \"{t.content}\""
                    if t.confidence >= 0:
                        think_str += f" 𝑝 {t.confidence}"
                    if t.refs:
                        refs = " • ".join(f'"{r}"' for r in t.refs)
                        think_str += f" ※ [{refs}]" if len(t.refs) > 1 else f" ※ \"{t.refs[0]}\""
                    if t.tag:
                        think_str += f" 🏷 {t.tag}"
                    if t.satisfies > 0:
                        think_str += f" ⊨ {t.satisfies}"
                    parts.append(think_str)
                for call in step.act.calls:
                    call_lines = ["    call ↦ {"]
                    call_lines.append(f"        tool ↦ {call.tool} •")
                    for arg in call.args:
                        call_lines.append(f"        {arg.name} ↦ \"{arg.value}\" •")
                    call_lines.append(f"        id ↦ \"{call.id}\"")
                    call_lines.append("    }")
                    call_str = "\n".join(call_lines)
                    if call.satisfies > 0:
                        call_str += f" ⊨ {call.satisfies}"
                    parts.append(call_str)
                lines.append(" •\n".join(parts))
                lines.append("}")
                
            elif step.step_type == "result":
                lines.append("result {")
                lines.append(f"    data ↦ \"{step.result.data}\"")
                lines.append(f"}} 🏷 \"{step.result.tag}\"")
                
            elif step.step_type == "response":
                r = step.response
                resp_str = f"response「{r.content}」"
                if r.refs:
                    refs = " • ".join(f'"{ref}"' for ref in r.refs)
                    resp_str += f" ※ [{refs}]" if len(r.refs) > 1 else f" ※ \"{r.refs[0]}\""
                if r.satisfies > 0:
                    resp_str += f" ⊨ {r.satisfies}"
                lines.append(resp_str)
            
            lines.append("")
        
        return "\n".join(lines).strip()


# ============================================================================
# Scenario Generation
# ============================================================================

class Scenario(BaseModel):
    """A generated scenario for trace generation."""
    domain: str = Field(description="Domain/category: coding, research, data_analysis, writing, planning, math, troubleshooting")
    system_prompt: str = Field(description="System prompt for this scenario")
    tools: list[ToolDef] = Field(description="Tools available in this scenario (1-3 tools)")
    user_request: str = Field(description="The user's initial request/task")
    complexity: str = Field(description="Expected complexity: simple, medium, or complex")
    expected_steps: int = Field(description="Expected number of act steps per turn (1-5)")
    follow_up_requests: list[str] = Field(description="Optional follow-up user messages (0-3). Empty list if single-turn.")


class ScenarioBatch(BaseModel):
    """Batch of scenarios for generation."""
    scenarios: list[Scenario] = Field(description="List of generated scenarios")
