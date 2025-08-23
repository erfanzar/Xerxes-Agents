# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import json
import re
import time
import typing
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .types import AssistantMessage, MessagesHistory, SystemMessage, UserMessage

if typing.TYPE_CHECKING:
    from calute import Calute


class CaluteTerminal:
    """Terminal interface for existing Calute instances"""

    def __init__(self, calute_instance: Calute | None = None):
        self.console = Console()
        self.calute = calute_instance
        self.history = MessagesHistory(messages=[])
        self.session_dir = Path.home() / ".calute_terminal" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.streaming = True
        self.show_function_calls = True
        self.show_thinking = True
        self.context_variables = {}
        self.use_chain_of_thought = False
        self.require_reflection = False
        self.current_agent_id = None

        # Command list for autocomplete
        self.commands = {
            "/help": "Show help message",
            "/clear": "Clear conversation",
            "/reset": "Reset conversation and context",
            "/exit": "Exit terminal",
            "/quit": "Exit terminal",
            "/agents": "List available agents",
            "/agent": "Switch to agent",
            "/memory": "Show memory statistics",
            "/context": "Manage context variables",
            "/cot": "Toggle chain of thought",
            "/reflect": "Toggle reflection mode",
            "/thinking": "Toggle thinking display",
            "/save": "Save session",
            "/load": "Load session",
            "/stream": "Toggle streaming",
            "/functions": "Toggle function display",
            "/history": "Show conversation history",
        }

    def set_calute(self, calute_instance: Calute):
        """Set the Calute instance to use"""
        self.calute = calute_instance

    def display_welcome(self):
        """Display welcome message"""
        agents_info = ""
        memory_info = ""

        # Get agent info
        if self.calute and hasattr(self.calute, "orchestrator"):
            agents_list = []
            orchestrator = self.calute.orchestrator
            for agent_id, agent in orchestrator.agents.items():
                model = getattr(agent, "model", "Unknown")
                is_current = "✓" if agent_id == self.current_agent_id else " "
                agents_list.append(f"  {is_current} **{agent_id}** (model: {model})")
            if agents_list:
                agents_info = "\n## 🤖 Available Agents:\n" + "\n".join(agents_list)

        # Get memory info if enabled
        if self.calute and self.calute.enable_memory:
            memory_stats = self.calute.memory_store.get_statistics()
            memory_info = f"""
## 🧠 Memory System:
• Total memories: {memory_stats.get("total_memories", 0)}
• Cache hit rate: {memory_stats.get("cache_hit_rate", 0):.1%}
• Short-term: {memory_stats.get("short_term_count", 0)} | Long-term: {memory_stats.get("long_term_count", 0)}"""

        # Current settings
        settings = f"""
## ⚙️ Settings:
• Streaming: {"✓" if self.streaming else "✗"}
• Functions: {"✓" if self.show_function_calls else "✗"}
• Thinking: {"✓" if self.show_thinking else "✗"}
• Chain of Thought: {"✓" if self.use_chain_of_thought else "✗"}
• Reflection: {"✓" if self.require_reflection else "✗"}"""

        # Create command groups for better organization
        navigation_cmds = [
            ("/help", "Show this help"),
            ("/clear", "Clear conversation"),
            ("/reset", "Reset all"),
            ("/exit", "Exit terminal"),
        ]

        agent_cmds = [
            ("/agents", "List agents"),
            ("/agent <id>", "Switch agent"),
            ("/memory", "Memory stats"),
            ("/context", "Variables"),
        ]

        mode_cmds = [
            ("/cot", "Chain of thought"),
            ("/reflect", "Reflection mode"),
            ("/thinking", "Show thinking"),
            ("/functions", "Show functions"),
        ]

        session_cmds = [
            ("/save <name>", "Save session"),
            ("/load <name>", "Load session"),
            ("/stream", "Toggle stream"),
            ("/history", "Show history"),
        ]

        # Main header
        header = Text()
        header.append("🚀 CALUTE TERMINAL", style="bold cyan")
        header.append(" - ", style="dim")
        header.append("Advanced AI Agent Interface", style="dim italic")

        # Create commands table for better alignment
        cmd_table = Table(box=None, show_header=False, padding=(0, 2))
        cmd_table.add_column("Navigation", style="cyan", no_wrap=True)
        cmd_table.add_column("Agents", style="green", no_wrap=True)
        cmd_table.add_column("Modes", style="yellow", no_wrap=True)
        cmd_table.add_column("Session", style="magenta", no_wrap=True)

        # Add rows
        for i in range(4):
            row_data = []
            for cmds in [navigation_cmds, agent_cmds, mode_cmds, session_cmds]:
                if i < len(cmds):
                    cmd, desc = cmds[i]
                    # Create formatted text for each cell
                    cell_text = Text()
                    cell_text.append(cmd, style="bold")
                    cell_text.append(f"\n{desc}", style="dim")
                    row_data.append(cell_text)
                else:
                    row_data.append("")
            cmd_table.add_row(*row_data)

        # Create the full display
        layout = Group(
            Panel(header, box=box.MINIMAL, border_style="cyan", padding=(0, 1)),
            Panel(
                cmd_table,
                title="Commands",
                title_align="left",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 1),
            ),
        )

        if agents_info or memory_info:
            info_text = Text()
            if agents_info:
                info_text.append("Agents: ", style="bold green")
                for agent_id, _ in self.calute.orchestrator.agents.items():
                    is_current = "✓" if agent_id == self.current_agent_id else ""
                    info_text.append(f"{is_current}{agent_id} ", style="green")
                info_text.append("  ")

            if memory_info and self.calute.enable_memory:
                stats = self.calute.memory_store.get_statistics()
                info_text.append("Memory: ", style="bold yellow")
                info_text.append(f"{stats.get('total_memories', 0)} items", style="yellow")

            layout = Group(layout, Panel(info_text, box=box.MINIMAL, border_style="dim", padding=(0, 1)))

        # Add compact settings display
        settings_text = Text()
        settings_text.append("Settings: ", style="bold magenta")
        settings = [
            ("Stream", self.streaming),
            ("Functions", self.show_function_calls),
            ("Thinking", self.show_thinking),
            ("CoT", self.use_chain_of_thought),
            ("Reflect", self.require_reflection),
        ]
        for name, enabled in settings:
            icon = "✓" if enabled else "✗"
            color = "green" if enabled else "red"
            settings_text.append(f"{name}:{icon} ", style=color)

        layout = Group(layout, Panel(settings_text, box=box.MINIMAL, border_style="dim", padding=(0, 1)))

        self.console.print(layout)
        self.console.print("[dim]Type your message or use / for commands[/dim]\n")

    def format_usage_info(self, usage_info=None, processing_time: float = 0.0) -> str:
        """Format usage information for display"""
        if not usage_info:
            return ""

        # Calculate TPS if we have timing info
        tps = getattr(usage_info, "tokens_per_second", 0) or 0

        # Extract token counts
        prompt_tokens = getattr(usage_info, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_info, "completion_tokens", 0) or 0
        total_tokens = getattr(usage_info, "total_tokens", 0) or 0

        # If total_tokens is 0, calculate it
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        # Format the usage info
        usage_parts = []
        if prompt_tokens > 0:
            usage_parts.append(f"Prompt: {prompt_tokens:,}")
        if completion_tokens > 0:
            usage_parts.append(f"Completion: {completion_tokens:,}")
        if total_tokens > 0:
            usage_parts.append(f"Total: {total_tokens:,}")
        if tps > 0:
            usage_parts.append(f"TPS: {tps:.1f}")

        if usage_parts:
            return " | ".join(usage_parts)
        return ""

    def parse_and_display_content(self, content: str, live_mode: bool = False, skip_thinking: bool = False):
        """Parse content for XML tags and display appropriately"""
        segments = []
        last_end = 0

        # Find all XML tags
        tag_regex = r"<(think|thinking|reason|reasoning|reflection|tool_call|function_call|invoke)>(.*?)</\1>"
        for match in re.finditer(tag_regex, content, re.DOTALL):
            # Add any text before the tag
            if match.start() > last_end:
                text_before = content[last_end : match.start()].strip()
                if text_before:
                    segments.append(("text", text_before))

            tag_name = match.group(1)
            tag_content = match.group(2).strip()

            if tag_name in ["think", "thinking", "reason", "reasoning"]:
                if self.show_thinking and not skip_thinking:  # Skip if already shown
                    segments.append(("thinking", tag_content))
            elif tag_name == "reflection":
                segments.append(("reflection", tag_content))
            else:
                segments.append(("tool", tag_content))

            last_end = match.end()

        # Add any remaining text
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining:
                segments.append(("text", remaining))

        # If no tags found, treat as plain text
        if not segments:
            segments = [("text", content)]

        # Display segments
        panels = []
        for seg_type, seg_content in segments:
            if seg_type == "thinking" and self.show_thinking:
                # Compact thinking display - only print if not in live mode
                if not live_mode:
                    self.console.print()
                    self.console.print("[dim cyan]💭 Thinking:[/dim cyan]")
                    # Add left padding to thinking content
                    padded_content = "\n".join(f"   {line}" for line in seg_content.split("\n"))
                    self.console.print(f"[dim]{padded_content}[/dim]")
                continue  # Don't add to panels
            elif seg_type == "reflection":
                # Compact reflection display
                panel = Panel(
                    Markdown(seg_content),
                    title="🔍 Reflection",
                    border_style="blue",
                    box=box.MINIMAL,
                    padding=(0, 1),
                )
            elif seg_type == "tool":
                # Compact tool display
                try:
                    tool_data = json.loads(seg_content)
                    formatted = json.dumps(tool_data, indent=2)
                    content_display = Syntax(formatted, "json", theme="monokai")
                except (json.JSONDecodeError, ValueError):
                    content_display = seg_content

                panel = Panel(
                    content_display,
                    title="🔧 Tool",
                    border_style="yellow",
                    box=box.MINIMAL,
                    padding=(0, 1),
                )
            else:
                # Regular assistant text - no panel, just formatted text
                if not live_mode:
                    self.console.print()
                    self.console.print("[green]Assistant:[/green] ", end="")
                    self.console.print(Markdown(seg_content))
                    continue
                else:
                    # For live mode, still use panel but minimal
                    panel = Panel(
                        Markdown(seg_content),
                        border_style="green",
                        box=box.MINIMAL,
                        padding=(0, 1),
                    )

            if live_mode:
                panels.append(panel)
            else:
                self.console.print(panel)

        return panels

    def display_message(self, role: str, content: str, function_calls=None):
        """Display a formatted message"""
        if role == "user":
            # Simple user message with minimal styling
            self.console.print(f"\n[bold blue]You:[/bold blue] {content}")
        elif role == "assistant":
            # Show function calls if enabled
            if function_calls and self.show_function_calls:
                for call in function_calls:
                    # Handle both dict and object types
                    if hasattr(call, "name"):
                        # It's an object (like FunctionCallInfo)
                        call_name = getattr(call, "name", "unknown")
                        call_args = getattr(call, "arguments", None)
                    else:
                        # It's a dictionary
                        call_name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
                        call_args = call.get("arguments") if isinstance(call, dict) else None

                    call_info = f"**Function:** `{call_name}`\n"
                    if call_args:
                        call_info += f"**Arguments:**\n```json\n{json.dumps(call_args, indent=2)}\n```"

                    self.console.print(
                        Panel(
                            Markdown(call_info),
                            title="[yellow]⚡ Function Call[/yellow]",
                            border_style="yellow",
                            box=box.MINIMAL,
                        )
                    )

            # Parse and display content with XML tag handling
            if content:
                self.parse_and_display_content(content)
        elif role == "system":
            self.console.print(
                Panel(content, title="[bold yellow]System[/bold yellow]", border_style="yellow", box=box.MINIMAL)
            )

    def handle_command(self, command: str) -> bool:
        """Handle commands. Returns True to continue, False to exit"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ["/exit", "/quit", "/q"]:
            return False

        elif cmd == "/clear":
            self.history = MessagesHistory(messages=[])
            self.console.clear()
            self.console.print("[green]✓ Conversation cleared[/green]")

        elif cmd == "/reset":
            self.history = MessagesHistory(messages=[])
            self.context_variables = {}
            self.console.print("[green]✓ Conversation and context reset[/green]")

        elif cmd == "/help":
            self.display_welcome()

        elif cmd == "/agents":
            # List all agents
            if self.calute and hasattr(self.calute, "orchestrator"):
                table = Table(title="Available Agents", box=box.SIMPLE)
                table.add_column("ID", style="cyan")
                table.add_column("Model", style="green")
                table.add_column("Functions", style="yellow")
                table.add_column("Current", style="magenta")

                for agent_id, agent in self.calute.orchestrator.agents.items():
                    func_count = len(agent.functions) if agent.functions else 0
                    is_current = "✓" if agent_id == self.current_agent_id else ""
                    table.add_row(agent_id, getattr(agent, "model", "Unknown"), str(func_count), is_current)

                self.console.print(table)
            else:
                self.console.print("[yellow]No agents available[/yellow]")

        elif cmd == "/agent":
            # Switch agent
            if args and self.calute:
                if hasattr(self.calute, "orchestrator"):
                    if args in self.calute.orchestrator.agents:
                        self.current_agent_id = args
                        self.console.print(f"[green]✓ Switched to agent: {args}[/green]")
                    else:
                        self.console.print(f"[red]Agent not found: {args}[/red]")
                        self.console.print("[yellow]Use /agents to see available agents[/yellow]")
            else:
                self.console.print("[yellow]Usage: /agent <agent_id>[/yellow]")

        elif cmd == "/memory":
            # Show memory statistics
            if self.calute and self.calute.enable_memory:
                stats = self.calute.memory_store.get_statistics()
                table = Table(title="Memory Statistics", box=box.SIMPLE)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                for key, value in stats.items():
                    if isinstance(value, float):
                        table.add_row(key, f"{value:.2f}")
                    else:
                        table.add_row(key, str(value))

                self.console.print(table)

                # Show recent memories
                if self.current_agent_id:
                    recent = self.calute.memory_store.get_memories(agent_id=self.current_agent_id, limit=5)
                    if recent:
                        self.console.print("\n[bold]Recent Memories:[/bold]")
                        for mem in recent:
                            self.console.print(f"  • {mem.content[:100]}...")
            else:
                self.console.print("[yellow]Memory system not enabled[/yellow]")

        elif cmd == "/context":
            # Manage context variables
            if args:
                # Parse key=value
                if "=" in args:
                    key, value = args.split("=", 1)
                    self.context_variables[key.strip()] = value.strip()
                    self.console.print(f"[green]✓ Set context: {key} = {value}[/green]")
                elif args == "clear":
                    self.context_variables = {}
                    self.console.print("[green]✓ Context cleared[/green]")
                else:
                    self.console.print("[yellow]Usage: /context key=value or /context clear[/yellow]")
            else:
                # Show current context
                if self.context_variables:
                    table = Table(title="Context Variables", box=box.SIMPLE)
                    table.add_column("Key", style="cyan")
                    table.add_column("Value", style="green")

                    for key, value in self.context_variables.items():
                        table.add_row(key, str(value)[:100])

                    self.console.print(table)
                else:
                    self.console.print("[yellow]No context variables set[/yellow]")

        elif cmd == "/cot":
            self.use_chain_of_thought = not self.use_chain_of_thought
            status = "enabled" if self.use_chain_of_thought else "disabled"
            self.console.print(f"[green]✓ Chain of thought {status}[/green]")

        elif cmd == "/reflect":
            self.require_reflection = not self.require_reflection
            status = "enabled" if self.require_reflection else "disabled"
            self.console.print(f"[green]✓ Reflection mode {status}[/green]")

        elif cmd == "/thinking":
            self.show_thinking = not self.show_thinking
            status = "shown" if self.show_thinking else "hidden"
            self.console.print(f"[green]✓ Thinking tags {status}[/green]")

        elif cmd == "/history":
            if not self.history.messages:
                self.console.print("[yellow]No messages in history[/yellow]")
            else:
                for msg in self.history.messages:
                    role = msg.__class__.__name__.replace("Message", "")
                    self.console.print(f"[cyan]{role}:[/cyan] {msg.content[:100]}...")

        elif cmd == "/stream":
            self.streaming = not self.streaming
            status = "enabled" if self.streaming else "disabled"
            self.console.print(f"[green]✓ Streaming {status}[/green]")

        elif cmd == "/functions":
            self.show_function_calls = not self.show_function_calls
            status = "shown" if self.show_function_calls else "hidden"
            self.console.print(f"[green]✓ Function calls {status}[/green]")

        elif cmd == "/save":
            filename = args or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if not filename.endswith(".json"):
                filename += ".json"

            filepath = self.session_dir / filename
            session_data = {
                "messages": [
                    {"role": msg.__class__.__name__.replace("Message", "").lower(), "content": msg.content}
                    for msg in self.history.messages
                ],
                "timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2)

            self.console.print(f"[green]✓ Saved to {filename}[/green]")

        elif cmd == "/load":
            if not args:
                # List available sessions
                sessions = list(self.session_dir.glob("*.json"))
                if sessions:
                    table = Table(title="Available Sessions", box=box.SIMPLE)
                    table.add_column("Filename", style="cyan")
                    table.add_column("Created", style="green")

                    for session_file in sorted(sessions, reverse=True)[:10]:
                        mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                        table.add_row(session_file.name, mtime.strftime("%Y-%m-%d %H:%M"))

                    self.console.print(table)
                else:
                    self.console.print("[yellow]No saved sessions found[/yellow]")
            else:
                filename = args if args.endswith(".json") else args + ".json"
                filepath = self.session_dir / filename

                if filepath.exists():
                    with open(filepath, "r") as f:
                        session_data = json.load(f)

                    self.history = MessagesHistory(messages=[])
                    for msg_data in session_data.get("messages", []):
                        if msg_data["role"] == "user":
                            self.history.messages.append(UserMessage(content=msg_data["content"]))
                        elif msg_data["role"] == "system":
                            self.history.messages.append(SystemMessage(content=msg_data["content"]))

                    self.console.print(f"[green]✓ Loaded {filename}[/green]")
                else:
                    self.console.print(f"[red]File not found: {filename}[/red]")

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")

        return True

    def autocomplete_command(self, partial: str) -> str | None:
        """Autocomplete command from partial input"""
        if not partial.startswith("/"):
            return None

        matches = [cmd for cmd in self.commands if cmd.startswith(partial)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Show available completions
            self.console.print("[dim]Available commands:[/dim]")
            for cmd in matches:
                self.console.print(f"  {cmd} - {self.commands[cmd]}")
            return partial  # Return original partial
        return None

    def get_prompt(self) -> str:
        """Get user input with smart features"""
        try:
            # Create a custom prompt with agent info
            prompt_text = "\n"
            if self.current_agent_id:
                prompt_text += f"[dim]({self.current_agent_id})[/dim] "
            prompt_text += "[cyan]>[/cyan]"

            user_input = Prompt.ask(prompt_text)

            # Handle autocomplete for commands
            if user_input.startswith("/") and not user_input.startswith("//"):
                completed = self.autocomplete_command(user_input)
                if completed and completed != user_input:
                    self.console.print(f"[dim]→ {completed}[/dim]")
                    return completed

            return user_input
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    def chat_loop(self):
        """Main chat loop"""
        if not self.calute:
            self.console.print("[red]Error: No Calute instance provided![/red]")
            self.console.print("Please provide a Calute instance when initializing CaluteTerminal")
            return

        self.display_welcome()

        while True:
            try:
                # Get user input with autocomplete
                user_input = self.get_prompt()

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                # Add to history and display
                self.history.messages.append(UserMessage(content=user_input))
                self.display_message("user", user_input)

                # Get response from Calute
                try:
                    # Handle chain of thought and reflection by modifying the prompt
                    user_prompt = user_input

                    # Add chain of thought instruction if enabled
                    if self.use_chain_of_thought:
                        user_prompt = (
                            "Please think through this step-by-step before responding:\n"
                            f"{user_input}\n\n"
                            "Show your reasoning process."
                        )

                    # Add reflection instruction if enabled
                    if self.require_reflection:
                        user_prompt += (
                            "\n\nAfter your primary response, add a reflection section in `<reflection>` tags:\n"
                            "- Assumptions made.\n"
                            "- Potential limitations of your response."
                        )

                    if self.streaming:
                        # Stream response
                        response_content = ""
                        function_calls = []
                        displayed_thinking = False
                        usage_info = None
                        start_time = time.time()

                        # Create initial minimal display
                        with Live(
                            Text("", style="green"),
                            console=self.console,
                            refresh_per_second=20,
                            transient=False,  # Keep the display for scrolling
                        ) as live:
                            # Use run method with correct parameters
                            for chunk in self.calute.run(
                                prompt=user_prompt,
                                messages=self.history,
                                context_variables=self.context_variables,
                                agent_id=self.current_agent_id,
                                stream=True,
                            ):
                                # Handle different streaming event types
                                chunk_type = chunk.__class__.__name__

                                # Function execution events
                                if chunk_type == "FunctionDetection":
                                    live.update(
                                        Panel(
                                            "[yellow]🔍 Detecting function calls...[/yellow]",
                                            border_style="yellow",
                                            box=box.MINIMAL,
                                        )
                                    )

                                elif chunk_type == "FunctionExecutionStart" and self.show_function_calls:
                                    live.update(
                                        Panel(
                                            f"[yellow]⚡ Executing: {chunk.function_name}[/yellow]\n"
                                            f"Progress: {chunk.progress}",
                                            border_style="yellow",
                                            box=box.MINIMAL,
                                        )
                                    )

                                elif chunk_type == "FunctionExecutionComplete" and self.show_function_calls:
                                    status_color = "green" if chunk.status == "success" else "red"
                                    result_text = str(chunk.result)[:200] if chunk.result else chunk.error
                                    live.update(
                                        Panel(
                                            f"[{status_color}]✓ {chunk.function_name}[/{status_color}]\n"
                                            f"Status: {chunk.status}\n"
                                            f"Result: {result_text}",
                                            border_style=status_color,
                                            box=box.MINIMAL,
                                        )
                                    )

                                elif chunk_type == "AgentSwitch":
                                    live.update(
                                        Panel(
                                            f"[magenta]🔄 Switching agent: {chunk.from_agent} → "
                                            f"{chunk.to_agent}[/magenta]\n"
                                            f"Reason: {chunk.reason}",
                                            border_style="magenta",
                                            box=box.MINIMAL,
                                        )
                                    )
                                    self.current_agent_id = chunk.to_agent

                                elif chunk_type == "ReinvokeSignal":
                                    live.update(
                                        Panel(
                                            "[blue]🔁 Reinvoking with function results...[/blue]",
                                            border_style="blue",
                                            box=box.MINIMAL,
                                        )
                                    )

                                # Regular content streaming
                                elif hasattr(chunk, "content") and chunk.content:
                                    response_content += chunk.content

                                    # Detect start of thinking/reasoning tags early
                                    thinking_started = any(tag in response_content for tag in ["<think", "<reason"])

                                    if thinking_started and self.show_thinking and not displayed_thinking:
                                        # Check if we have a complete thinking tag
                                        if "<think>" in response_content and "</think>" in response_content:
                                            # Complete thinking tag - display it once and mark as done
                                            thinking_match = re.search(
                                                r"<think>(.*?)</think>", response_content, re.DOTALL
                                            )
                                            if thinking_match:
                                                thinking_content = thinking_match.group(1).strip()
                                                # Display thinking once with padding
                                                self.console.print("\n[dim cyan]💭 Thinking:[/dim cyan]")
                                                # Add left padding to thinking content
                                                padded_thinking = "\n".join(
                                                    f"   {line}" for line in thinking_content.split("\n")
                                                )
                                                self.console.print(f"[dim]{padded_thinking}[/dim]\n")
                                                displayed_thinking = True
                                                # Now show the response without thinking
                                                display_content = re.sub(
                                                    r"<think>.*?</think>", "", response_content, flags=re.DOTALL
                                                ).strip()
                                                if display_content:
                                                    live.update(Text(f"{display_content}▌", style="green"))
                                                continue
                                        elif "<think" in response_content and "</think>" not in response_content:
                                            # Still accumulating thinking content - show with cursor
                                            thinking_match = re.search(r"<think[^>]*>(.*)", response_content, re.DOTALL)
                                            if thinking_match:
                                                partial_thinking = thinking_match.group(1)
                                                # Add padding to partial thinking display
                                                padded_partial = "\n".join(
                                                    f"   {line}" for line in partial_thinking.split("\n")
                                                )
                                                live.update(Text(f"💭 Thinking:\n{padded_partial}▌", style="dim cyan"))
                                                continue

                                    # Check for complete tags
                                    complete_tags = []
                                    tag_patterns = [
                                        r"<(think|thinking)>.*?</\1>",
                                        r"<(reason|reasoning)>.*?</\1>",
                                        r"<(reflection)>.*?</\1>",
                                        r"<(tool_call|function_call|invoke)>.*?</\1>",
                                    ]

                                    for pattern in tag_patterns:
                                        complete_tags.extend(re.findall(pattern, response_content, re.DOTALL))

                                    if complete_tags:
                                        # Show parsed content, but skip thinking if already displayed
                                        panels = self.parse_and_display_content(
                                            response_content + "▌", live_mode=True, skip_thinking=displayed_thinking
                                        )
                                        if panels:
                                            live.update(Group(*panels))
                                    else:
                                        # Regular streaming display - minimal
                                        # Remove thinking tags from display if already shown
                                        display_content = response_content
                                        if displayed_thinking:
                                            # Strip thinking tags from display
                                            display_content = re.sub(
                                                r"<think[^>]*>.*?</think>", "", display_content, flags=re.DOTALL
                                            )
                                            display_content = display_content.strip()

                                        # Just show the text with a cursor
                                        if display_content:
                                            live.update(Text(f"{display_content}▌", style="green"))

                                if hasattr(chunk, "function_calls") and chunk.function_calls:
                                    function_calls.extend(chunk.function_calls)

                                # Check for usage info in chunks
                                if hasattr(chunk, "chunk") and hasattr(chunk.chunk, "usage"):
                                    usage_info = chunk.chunk.usage
                                elif hasattr(chunk, "usage"):
                                    usage_info = chunk.usage
                        
                        # Final update to remove cursor and show clean output
                        if response_content:
                            # Remove thinking tags if present
                            display_content = response_content
                            if displayed_thinking:
                                display_content = re.sub(
                                    r"<think[^>]*>.*?</think>", "", display_content, flags=re.DOTALL
                                ).strip()
                            # Show final content without cursor
                            if display_content:
                                live.update(Text(display_content, style="green"))

                        # Calculate processing time
                        processing_time = time.time() - start_time

                        # Display usage info if available
                        if usage_info:
                            usage_str = self.format_usage_info(usage_info, processing_time)
                            if usage_str:
                                self.console.print(f"\n[dim cyan]📊 {usage_str}[/dim cyan]")

                        # Show function calls if any
                        if function_calls and self.show_function_calls:
                            for call in function_calls:
                                # Handle both dict and object types
                                if hasattr(call, "name"):
                                    # It's an object (like FunctionCallInfo)
                                    call_name = getattr(call, "name", "unknown")
                                    call_args = getattr(call, "arguments", None)
                                else:
                                    # It's a dictionary
                                    call_name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
                                    call_args = call.get("arguments") if isinstance(call, dict) else None

                                call_info = f"**Function:** `{call_name}`\n"
                                if call_args:
                                    args_str = json.dumps(call_args, indent=2)
                                    call_info += f"**Arguments:**\n```json\n{args_str}\n```"

                                self.console.print(
                                    Panel(
                                        Markdown(call_info),
                                        title="[yellow]⚡ Function Call[/yellow]",
                                        border_style="yellow",
                                        box=box.MINIMAL,
                                    )
                                )

                        # Display final response only if not already shown during streaming
                        if response_content:
                            # Parse content but skip thinking if it was already displayed
                            self.parse_and_display_content(
                                response_content, live_mode=False, skip_thinking=displayed_thinking
                            )
                            # Add assistant's response to history for context
                            self.history.messages.append(AssistantMessage(content=response_content))

                    else:
                        # Non-streaming response
                        start_time = time.time()
                        with self.console.status("[dim]Thinking...[/dim]", spinner="dots"):
                            response = self.calute.run(
                                prompt=user_prompt,
                                messages=self.history,
                                context_variables=self.context_variables,
                                agent_id=self.current_agent_id,
                                stream=False,
                            )
                        processing_time = time.time() - start_time

                        if response:
                            content = getattr(response, "content", "")
                            calls = getattr(response, "function_calls", None)

                            # Check for agent switch in execution history
                            if hasattr(response, "execution_history"):
                                for event in response.execution_history:
                                    if event.get("type") == "agent_switch":
                                        self.console.print(
                                            Panel(
                                                f"[magenta]🔄 Agent switched: {event.get('from')} → "
                                                f"{event.get('to')}[/magenta]",
                                                border_style="magenta",
                                                box=box.MINIMAL,
                                            )
                                        )
                                        self.current_agent_id = event.get("to")

                            # Show function calls first if any
                            if calls and self.show_function_calls:
                                for call in calls:
                                    # Handle both dict and object types
                                    if hasattr(call, "name"):
                                        # It's an object (like FunctionCallInfo)
                                        call_name = getattr(call, "name", "unknown")
                                        call_args = getattr(call, "arguments", None)
                                    else:
                                        # It's a dictionary
                                        call_name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
                                        call_args = call.get("arguments") if isinstance(call, dict) else None

                                    call_info = f"**Function:** `{call_name}`\n"
                                    if call_args:
                                        args_str = json.dumps(call_args, indent=2)
                                        call_info += f"**Arguments:**\n```json\n{args_str}\n```"

                                    self.console.print(
                                        Panel(
                                            Markdown(call_info),
                                            title="[yellow]⚡ Function Call[/yellow]",
                                            border_style="yellow",
                                            box=box.MINIMAL,
                                        )
                                    )

                            # Parse and display content with XML handling
                            if content:
                                self.parse_and_display_content(content)
                                # Add assistant's response to history for context
                                self.history.messages.append(AssistantMessage(content=content))

                            # Display usage info if available
                            usage_info = None
                            if hasattr(response, "response") and hasattr(response.response, "usage"):
                                usage_info = response.response.usage
                            elif hasattr(response, "usage"):
                                usage_info = response.usage

                            if usage_info:
                                usage_str = self.format_usage_info(usage_info, processing_time)
                                if usage_str:
                                    self.console.print(f"\n[dim cyan]📊 {usage_str}[/dim cyan]")

                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Use /exit to quit[/yellow]")
                continue
            except Exception as e:
                self.console.print(
                    Panel(
                        f"[red]⚠ Error:[/red] {str(e)[:200]}",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                )
                self.console.print("[dim]Try again or use /help for assistance[/dim]")
                continue

    def run(self):
        """Run the terminal interface"""
        try:
            self.chat_loop()
        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
        finally:
            self.console.print("\n[dim]Goodbye! 👋[/dim]\n")

    @staticmethod
    def run_terminal(calute_instance: Calute):
        """Convenience function to run terminal with a Calute instance"""
        terminal = CaluteTerminal(calute_instance)
        terminal.run()
