#!/usr/bin/env python3
"""
Rich TUI Dashboard for TASK RL Training.

Shows live training progress with:
- Reward graph (sparkline)
- Streaming prompt → output → judge → verifier flow
- Training statistics

Usage:
    from dashboard import TrainingDashboard
    
    dashboard = TrainingDashboard()
    dashboard.start()
    
    # During training:
    dashboard.log_generation(prompt, output, judge_score, verifier_score, rewards_breakdown)
    dashboard.update_stats(step=100, loss=0.5, reward_mean=2.1)
    
    dashboard.stop()
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import sys

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.style import Style
    from rich.box import ROUNDED, HEAVY
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class GenerationLog:
    """Single generation log entry."""
    step: int
    prompt_preview: str
    output_preview: str
    verifier_score: float
    verifier_breakdown: dict
    judge_score: Optional[float] = None
    judge_breakdown: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)


class RewardGraph:
    """ASCII sparkline graph for rewards."""
    
    BARS = " ▁▂▃▄▅▆▇█"
    
    def __init__(self, width: int = 60, history: int = 5000):
        self.width = width
        self.history = history
        self.values = deque(maxlen=history)
        self.min_val = -3.0
        self.max_val = 4.0
        self._total_count = 0  # Total values ever added
    
    def add(self, value: float):
        self.values.append(value)
        self._total_count += 1
        # Auto-adjust scale based on actual values
        if self.values:
            actual_min = min(self.values)
            actual_max = max(self.values)
            self.min_val = min(actual_min - 0.5, -3.0)
            self.max_val = max(actual_max + 0.5, 4.0)
    
    def render(self) -> str:
        if not self.values:
            return "─" * self.width
        
        # Normalize values to 0-8 range for bar chars
        range_val = self.max_val - self.min_val
        if range_val == 0:
            range_val = 1
        
        # Take last `width` values
        display_values = list(self.values)[-self.width:]
        
        # Build sparkline
        line = ""
        for v in display_values:
            normalized = (v - self.min_val) / range_val
            idx = int(normalized * 8)
            idx = max(0, min(8, idx))
            line += self.BARS[idx]
        
        # Pad if needed
        if len(line) < self.width:
            line = "─" * (self.width - len(line)) + line
        
        return line
    
    def stats(self) -> tuple[float, float, float, int]:
        """Return (min, max, mean, count) of all values."""
        if not self.values:
            return 0.0, 0.0, 0.0, 0
        vals = list(self.values)
        return min(vals), max(vals), sum(vals) / len(vals), self._total_count


class TrainingDashboard:
    """Rich TUI dashboard for RL training visualization."""
    
    def __init__(self, max_logs: int = 50, max_console_logs: int = 100):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required. Install with: pip install rich")
        
        self.console = Console()
        self.logs: deque[GenerationLog] = deque(maxlen=max_logs)
        self.console_logs: deque[str] = deque(maxlen=max_console_logs)  # For log messages
        self.reward_graph = RewardGraph(width=40)
        self.judge_graph = RewardGraph(width=40)
        
        # Training stats
        self.step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.reward_mean = 0.0
        self.generations = 0
        self.valid_rate = 0.0
        self.judge_calls = 0
        self.start_time = time.time()
        
        # Live display
        self._live: Optional[Live] = None
        self._running = False
        self._lock = threading.Lock()
    
    def _truncate(self, text: str, max_len: int = 200) -> str:
        """Truncate text with ellipsis."""
        text = text.replace('\n', ' ').strip()
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text
    
    def _format_breakdown(self, breakdown: dict) -> str:
        """Format reward breakdown for display."""
        parts = []
        for k, v in breakdown.items():
            if isinstance(v, float):
                color = "green" if v > 0 else ("red" if v < 0 else "dim")
                parts.append(f"[{color}]{k}:{v:+.2f}[/]")
        return " ".join(parts[:6])  # Limit to first 6
    
    def _make_layout(self) -> Layout:
        """Create the dashboard layout with fixed sizes to prevent jumping."""
        layout = Layout()
        
        # Fixed heights prevent layout shifts
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=25),  # Main content
            Layout(name="logs", size=12),  # Log box
            Layout(name="footer", size=3),
        )
        
        layout["main"].split_row(
            Layout(name="stats", size=50),  # Fixed width for stats
            Layout(name="flow"),  # Latest completion takes remaining space
        )
        
        return layout
    
    def _render_header(self) -> Panel:
        """Render header with title."""
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        
        title = Text()
        title.append("🎯 ", style="bold")
        title.append("TASK RL Training Dashboard", style="bold cyan")
        title.append(f"  │  ", style="dim")
        title.append(f"Step {self.step}/{self.total_steps or '?'}", style="yellow")
        title.append(f"  │  ", style="dim")
        title.append(f"⏱ {int(hours):02d}:{int(mins):02d}:{int(secs):02d}", style="green")
        
        return Panel(title, box=HEAVY, style="bold")
    
    def _render_stats(self) -> Panel:
        """Render statistics panel with graphs."""
        # Stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_column("Key", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Generations", f"{self.generations:,}")
        stats_table.add_row("Valid Rate", f"{self.valid_rate*100:.1f}%")
        stats_table.add_row("Judge Calls", f"{self.judge_calls:,}")
        stats_table.add_row("Loss", f"{self.loss:.4f}")
        stats_table.add_row("Reward Mean", f"{self.reward_mean:.2f}")
        
        # Recent scores (last 10)
        recent_section = Text()
        recent_section.append("\nRecent Scores: ", style="bold")
        recent_scores = [log.verifier_score for log in list(self.logs)[-10:]]
        for score in recent_scores:
            color = "green" if score > 1.5 else ("yellow" if score > 0 else "red")
            recent_section.append(f"{score:.1f} ", style=color)
        
        # Reward graph
        r_min, r_max, r_mean, r_count = self.reward_graph.stats()
        reward_section = Text()
        reward_section.append("\n\nVerifier Reward", style="bold yellow")
        reward_section.append(f" (n={r_count})\n", style="dim")
        reward_section.append(f"mean:{r_mean:.2f} range:[{r_min:.1f},{r_max:.1f}]\n", style="dim")
        reward_section.append(self.reward_graph.render(), style="green")
        
        # Judge graph (only if we have judge calls)
        judge_section = Text()
        if self.judge_calls > 0:
            j_min, j_max, j_mean, j_count = self.judge_graph.stats()
            judge_section.append("\n\nJudge Score", style="bold magenta")
            judge_section.append(f" (n={j_count})\n", style="dim")
            judge_section.append(f"mean:{j_mean:.2f} range:[{j_min:.1f},{j_max:.1f}]\n", style="dim")
            judge_section.append(self.judge_graph.render(), style="magenta")
        
        content = Group(
            stats_table,
            recent_section,
            reward_section,
            judge_section,
        )
        
        return Panel(content, title="[bold]📊 Statistics[/]", border_style="blue", box=ROUNDED)
    
    def _render_flow(self) -> Panel:
        """Render the latest completion panel."""
        if not self.logs:
            return Panel(
                "[dim]Waiting for generations...[/]",
                title="[bold]🔄 Latest Completion[/]",
                border_style="green",
                box=ROUNDED,
            )
        
        # Get the latest log
        latest = self.logs[-1]
        
        content = Text()
        
        # Header with step and scores
        v_color = "green" if latest.verifier_score > 1.5 else ("yellow" if latest.verifier_score > 0 else "red")
        content.append(f"Step {latest.step}", style="bold cyan")
        content.append(f"  │  ", style="dim")
        content.append(f"Verifier: {latest.verifier_score:.2f}", style=f"bold {v_color}")
        if latest.judge_score is not None:
            j_color = "green" if latest.judge_score > 0.7 else ("yellow" if latest.judge_score > 0.4 else "red")
            content.append(f"  │  ", style="dim")
            content.append(f"Judge: {latest.judge_score:.2f}", style=f"bold {j_color}")
        content.append("\n")
        content.append("─" * 60 + "\n", style="dim")
        
        # Prompt (truncated)
        content.append("📝 ", style="bold yellow")
        prompt_preview = latest.prompt_preview[:200].replace('\n', ' ')
        content.append(f"{prompt_preview}...\n\n", style="dim")
        
        # Full output (or as much as fits)
        content.append("🤖 Output:\n", style="bold green")
        
        # Show the completion - limit to ~25 lines to fit in panel
        output_lines = latest.output_preview.split('\n')
        max_lines = 22
        if len(output_lines) > max_lines:
            # Show first and last parts
            shown = output_lines[:max_lines-2]
            shown.append(f"... ({len(output_lines) - max_lines + 2} more lines) ...")
            shown.extend(output_lines[-1:])
            output_text = '\n'.join(shown)
        else:
            output_text = latest.output_preview
        
        content.append(output_text, style="white")
        
        # Reward breakdown at bottom
        content.append("\n\n")
        content.append("─" * 60 + "\n", style="dim")
        breakdown_str = self._format_breakdown(latest.verifier_breakdown)
        content.append(f"Breakdown: {breakdown_str}", style="dim")
        
        return Panel(
            content,
            title=f"[bold]🔄 Latest Completion ({self.generations} total)[/]",
            border_style="green",
            box=ROUNDED,
        )
    
    def _render_logs(self) -> Panel:
        """Render the log box panel."""
        if not self.console_logs:
            content = Text("[dim]No logs yet...[/]")
        else:
            content = Text()
            # Show last N logs that fit
            for log_line in list(self.console_logs)[-10:]:
                # Color based on content
                if "[ERROR]" in log_line or "Error" in log_line:
                    style = "red"
                elif "[WARN]" in log_line or "Warning" in log_line:
                    style = "yellow"
                elif "[Judge]" in log_line:
                    style = "magenta"
                elif "[Reward]" in log_line:
                    style = "green"
                elif "✓" in log_line:
                    style = "green"
                elif "✗" in log_line:
                    style = "red"
                else:
                    style = "dim"
                
                # Truncate long lines
                display_line = log_line[:120] + "..." if len(log_line) > 120 else log_line
                content.append(f"{display_line}\n", style=style)
        
        return Panel(
            content,
            title=f"[bold]📋 Logs ({len(self.console_logs)})[/]",
            border_style="yellow",
            box=ROUNDED,
        )
    
    def _render_footer(self) -> Panel:
        """Render footer with controls."""
        footer = Text()
        footer.append("  Press ", style="dim")
        footer.append("Ctrl+C", style="bold yellow")
        footer.append(" to stop  │  ", style="dim")
        footer.append("Samples: ", style="dim")
        footer.append(f"{len(self.logs)}", style="cyan")
        footer.append(f"/{self.logs.maxlen}", style="dim")
        footer.append("  │  ", style="dim")
        footer.append("Logs: ", style="dim")
        footer.append(f"{len(self.console_logs)}", style="cyan")
        
        return Panel(footer, box=ROUNDED, style="dim")
    
    def _render(self) -> Layout:
        """Render the full dashboard."""
        layout = self._make_layout()
        
        with self._lock:
            layout["header"].update(self._render_header())
            layout["stats"].update(self._render_stats())
            layout["flow"].update(self._render_flow())
            layout["logs"].update(self._render_logs())
            layout["footer"].update(self._render_footer())
        
        return layout
    
    def start(self, total_steps: int = 0):
        """Start the live dashboard."""
        self.total_steps = total_steps
        self.start_time = time.time()
        self._running = True
        
        # Use vertical_overflow="visible" to prevent layout shifts
        # Use refresh_per_second=4 for smoother updates
        # screen=False to avoid full screen clear which causes flicker
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()
        # Clear screen once at start
        self.console.clear()
    
    def stop(self):
        """Stop the live dashboard."""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None
    
    def update(self):
        """Force update the display (called automatically by Live)."""
        if self._live and self._live.is_started:
            try:
                self._live.update(self._render(), refresh=True)
            except Exception:
                pass  # Ignore update errors during shutdown
    
    def log_generation(
        self,
        step: int,
        prompt: str,
        output: str,
        verifier_score: float,
        verifier_breakdown: dict,
        judge_score: Optional[float] = None,
        judge_breakdown: Optional[dict] = None,
    ):
        """Log a generation for display."""
        with self._lock:
            log = GenerationLog(
                step=step,
                prompt_preview=prompt,
                output_preview=output,
                verifier_score=verifier_score,
                verifier_breakdown=verifier_breakdown,
                judge_score=judge_score,
                judge_breakdown=judge_breakdown,
            )
            self.logs.append(log)
            self.generations += 1
            
            # Update graphs
            self.reward_graph.add(verifier_score)
            if judge_score is not None:
                self.judge_graph.add(judge_score)
                self.judge_calls += 1
            
            # Update valid rate
            valid_count = sum(1 for l in self.logs if l.verifier_score > 0)
            self.valid_rate = valid_count / len(self.logs) if self.logs else 0
        
        # Don't call update() here - let Live's auto-refresh handle it
        # This prevents flicker from too-frequent redraws
    
    def update_stats(
        self,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        reward_mean: Optional[float] = None,
    ):
        """Update training statistics."""
        with self._lock:
            if step is not None:
                self.step = step
            if loss is not None:
                self.loss = loss
            if reward_mean is not None:
                self.reward_mean = reward_mean
        
        # Don't call update() - let Live's auto-refresh handle it
    
    def log(self, message: str):
        """Add a log message to the log box."""
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.console_logs.append(f"[{timestamp}] {message}")


# =============================================================================
# Simple fallback for non-rich environments
# =============================================================================

class SimpleDashboard:
    """Simple text-based dashboard fallback."""
    
    def __init__(self, max_logs: int = 10):
        self.logs = deque(maxlen=max_logs)
        self.step = 0
        self.generations = 0
    
    def start(self, total_steps: int = 0):
        print("=" * 60)
        print("TASK RL Training")
        print("=" * 60)
    
    def stop(self):
        print("\nTraining complete!")
    
    def update(self):
        pass
    
    def log_generation(
        self,
        step: int,
        prompt: str,
        output: str,
        verifier_score: float,
        verifier_breakdown: dict,
        judge_score: Optional[float] = None,
        judge_breakdown: Optional[dict] = None,
    ):
        self.generations += 1
        status = "✓" if verifier_score > 1.5 else ("~" if verifier_score > 0 else "✗")
        judge_str = f" | judge={judge_score:.2f}" if judge_score else ""
        print(f"[{step}] {status} verifier={verifier_score:.2f}{judge_str}")
    
    def update_stats(self, step=None, loss=None, reward_mean=None):
        if step is not None:
            self.step = step
        if loss is not None and step and step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}, reward_mean={reward_mean:.2f}")


def create_dashboard(use_rich: bool = True, **kwargs):
    """Factory function to create appropriate dashboard."""
    if use_rich and RICH_AVAILABLE:
        return TrainingDashboard(**kwargs)
    else:
        return SimpleDashboard(**kwargs)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("Testing dashboard...")
    
    dashboard = create_dashboard(use_rich=True)
    dashboard.start(total_steps=100)
    
    try:
        for i in range(100):
            # Simulate a generation
            verifier_score = random.gauss(1.5, 1.0)
            judge_score = random.random() if random.random() < 0.3 else None
            
            dashboard.log_generation(
                step=i,
                prompt="user「Write a function to sort a list of numbers」🏷 usr1",
                output=f"plan {{ todo ↦ {{ 1 ↦ \"Write sorting function\" }} }} act {{ think ↦ 「I'll implement quicksort...」 }} response「Here's a sorting function...」",
                verifier_score=verifier_score,
                verifier_breakdown={
                    'parse_valid': 1.0,
                    'has_plan': 0.3,
                    'has_response': 0.5,
                    'todos_satisfied': 0.5 if verifier_score > 1 else -0.3,
                    'refs_valid': 0.3 if verifier_score > 0.5 else -0.2,
                },
                judge_score=judge_score,
            )
            
            dashboard.update_stats(
                step=i,
                loss=max(0.1, 2.0 - i * 0.015 + random.gauss(0, 0.1)),
                reward_mean=min(2.5, 0.5 + i * 0.02 + random.gauss(0, 0.2)),
            )
            
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
    
    print("\nDemo complete!")

