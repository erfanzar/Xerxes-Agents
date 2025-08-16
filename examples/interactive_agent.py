#!/usr/bin/env python3
"""
Interactive Agent - Test all Calute improvements interactively.
This agent demonstrates all the enhanced features and allows you to interact with them.
"""

import asyncio
import json
import os
from pathlib import Path

from openai import OpenAI

# Import core Calute
from calute import Agent, AssistantMessage, Calute, MessagesHistory, UserMessage

# Import all improvements
from calute.config import CaluteConfig, get_config, set_config
from calute.errors import CaluteTimeoutError
from calute.logging_config import get_logger
from calute.memory import EnhancedMemoryStore, MemoryType

# ================== Custom Functions for Testing ==================


def list_improvements() -> str:
    """List all improvements made to Calute."""
    improvements = [
        "1. Enhanced Memory with indexing and search",
        "2. Configuration management (YAML/JSON)",
        "3. Extended LLM providers (Anthropic, Cohere, HuggingFace, Ollama)",
        "4. Structured logging with metrics",
        "5. Error handling with timeouts and retries",
        "6. CI/CD with GitHub Actions",
        "7. Docker support",
        "8. Comprehensive test suite",
        "9. Developer tools (Makefile, pre-commit)",
        "10. Observability (Prometheus, OpenTelemetry)",
    ]
    return "Calute Improvements:\n" + "\n".join(improvements)


def test_memory_system(action: str = "demo", content: str = "", tags: str = "") -> str:
    """Test the enhanced memory system.

    Args:
        action: One of 'add', 'search', 'stats', 'demo'
        content: Content to add (for 'add' action)
        tags: Comma-separated tags for adding or searching
    """
    global agent_memory

    if action == "add" and content:
        tags_list = [t.strip() for t in tags.split(",")] if tags else []
        entry = agent_memory.add_memory(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            agent_id="interactive_agent",
            tags=tags_list,
            importance_score=0.7,
        )
        return f"✅ Added memory: {entry.id[:8]}... with tags: {tags_list}"

    elif action == "search":
        tags_list = [t.strip() for t in tags.split(",")] if tags else []
        memories = agent_memory.retrieve_memories(tags=tags_list if tags_list else None, limit=5)
        if memories:
            result = "Found memories:\n"
            for mem in memories:
                result += f"- [{mem.id[:8]}] {mem.content[:50]}... (tags: {mem.tags})\n"
            return result
        return "No memories found."

    elif action == "stats":
        stats = agent_memory.get_statistics()
        return f"""Memory Statistics:
- Total memories: {stats["total_memories"]}
- By type: {json.dumps(stats["by_type"], indent=2)}
- Cache hit rate: {stats.get("cache_hit_rate", 0):.2%}"""

    else:  # demo
        # Add demo memories
        demo_data = [
            ("Python is a programming language", ["programming", "python"]),
            ("Calute is an AI agent framework", ["ai", "framework", "calute"]),
            ("Machine learning uses neural networks", ["ai", "ml", "neural"]),
        ]
        for content, tags in demo_data:
            agent_memory.add_memory(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                agent_id="interactive_agent",
                tags=tags,
                importance_score=0.8,
            )
        return "✅ Added 3 demo memories. Try searching with tags: 'ai', 'python', or 'framework'"


def test_configuration(action: str = "show", key: str = "", value: str = "") -> str:
    """Test configuration management.

    Args:
        action: One of 'show', 'set', 'save', 'load'
        key: Configuration key (dot notation like 'executor.timeout')
        value: Value to set
    """
    config = get_config()

    if action == "show":
        config_dict = config.model_dump()
        if key:
            # Navigate to specific key
            parts = key.split(".")
            current = config_dict
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return f"Key '{key}' not found"
            return f"{key}: {json.dumps(current, indent=2)}"
        return f"Current configuration:\n{json.dumps(config_dict, indent=2, default=str)}"

    elif action == "set" and key and value:
        # Parse value
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            parsed_value = value

        # Set configuration value
        parts = key.split(".")
        if len(parts) == 2:
            section, field = parts
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, field):
                    setattr(section_obj, field, parsed_value)
                    set_config(config)
                    return f"✅ Set {key} = {parsed_value}"
        return f"Cannot set {key}"

    elif action == "save":
        path = Path("interactive_config.json")
        config.to_file(path)
        return f"✅ Configuration saved to {path}"

    elif action == "load":
        path = Path("interactive_config.json")
        if path.exists():
            loaded = CaluteConfig.from_file(path)
            set_config(loaded)
            return f"✅ Configuration loaded from {path}"
        return "No saved configuration found"

    return "Invalid action. Use: show, set, save, or load"


def test_error_handling(scenario: str = "timeout") -> str:
    """Test error handling and recovery.

    Args:
        scenario: One of 'timeout', 'retry', 'validation'
    """
    import time

    if scenario == "timeout":

        def slow_function():
            time.sleep(5)
            return "This should timeout"

        try:
            # This would timeout with proper executor
            return "Simulating timeout... (In real use, this would timeout after configured seconds)"
        except CaluteTimeoutError as e:
            return f"✅ Timeout handled: {e}"

    elif scenario == "retry":
        attempts = []

        def flaky_function():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise Exception(f"Attempt {len(attempts)} failed")
            return f"Success after {len(attempts)} attempts"

        try:
            result = flaky_function()
            return f"✅ Retry demonstration: {result}"
        except Exception:
            result = flaky_function()
            result = flaky_function()
            return f"✅ Function succeeded after retries: {result}"

    elif scenario == "validation":

        def validated_function(x: int):
            if not isinstance(x, int):
                raise ValueError(f"Expected int, got {type(x)}")
            return x * 2

        try:
            validated_function("not an int")
            return "Should have failed validation"
        except ValueError as e:
            return f"✅ Validation error caught: {e}"

    return "Invalid scenario. Use: timeout, retry, or validation"


def test_logging_metrics(action: str = "log", level: str = "info", message: str = "Test log") -> str:
    """Test logging and metrics system.

    Args:
        action: One of 'log', 'metrics', 'configure'
        level: Log level (debug, info, warning, error)
        message: Message to log
    """
    logger = get_logger("interactive_agent")

    if action == "log":
        if level == "debug":
            logger.logger.debug(message)
        elif level == "info":
            logger.logger.info(message)
        elif level == "warning":
            logger.logger.warning(message)
        elif level == "error":
            logger.logger.error(message)
        else:
            logger.logger.info(message)
        return f"✅ Logged at {level} level: {message}"

    elif action == "metrics":
        # Log some sample metrics
        logger.log_function_call(
            agent_id="interactive_agent",
            function_name="test_function",
            arguments={"x": 10},
            result="success",
            duration=0.5,
        )

        logger.log_llm_request(provider="openai", model="gpt-4o", prompt_tokens=100, completion_tokens=50, duration=2.0)

        # Get metrics
        metrics_bytes = logger.get_metrics()
        return f"""✅ Metrics generated:
- Size: {len(metrics_bytes)} bytes
- Function calls logged
- LLM requests logged
- View raw metrics with action='raw_metrics'"""

    elif action == "raw_metrics":
        metrics_bytes = logger.get_metrics()
        return f"Prometheus metrics:\n{metrics_bytes.decode()[:500]}..."

    return "Invalid action. Use: log, metrics, or raw_metrics"


def analyze_code_file(filepath: str) -> str:
    """Analyze a Python file and report statistics.

    Args:
        filepath: Path to Python file to analyze
    """
    import re

    path = Path(filepath)
    if not path.exists():
        return f"File not found: {filepath}"

    if not filepath.endswith(".py"):
        return "Please provide a Python file (.py)"

    try:
        with open(path, "r") as f:
            content = f.read()

        # Count various elements
        lines = content.split("\n")
        functions = re.findall(r"^def\s+\w+", content, re.MULTILINE)
        classes = re.findall(r"^class\s+\w+", content, re.MULTILINE)
        imports = re.findall(r"^(?:from|import)\s+\w+", content, re.MULTILINE)
        comments = re.findall(r"#.*$", content, re.MULTILINE)
        docstrings = re.findall(r'"""[\s\S]*?"""', content)

        return f"""📊 Analysis of {path.name}:
- Total lines: {len(lines)}
- Functions: {len(functions)}
- Classes: {len(classes)}
- Import statements: {len(imports)}
- Comments: {len(comments)}
- Docstrings: {len(docstrings)}
- Code/comment ratio: {(len(lines) - len(comments)) / max(len(lines), 1):.1%}

Top functions: {", ".join([f.replace("def ", "") for f in functions[:5]])}
Top classes: {", ".join([c.replace("class ", "") for c in classes[:5]])}"""

    except Exception as e:
        return f"Error analyzing file: {e}"


def get_system_info() -> str:
    """Get information about the Calute system and environment."""
    import platform

    info = {
        "Python Version": platform.python_version(),
        "Platform": platform.platform(),
        "Calute Version": "0.0.18 (enhanced)",
        "Working Directory": os.getcwd(),
        "Memory Store Path": os.path.expanduser("~/.calute/memory_store"),
        "Config Path": "interactive_config.json",
    }

    # Check which optional dependencies are available
    optional_deps = {
        "PyYAML": False,
        "structlog": False,
        "prometheus_client": False,
        "httpx": False,
        "sklearn": False,
    }

    for dep in optional_deps:
        try:
            __import__(dep.lower().replace("_", "-"))
            optional_deps[dep] = True
        except ImportError:
            pass

    result = "🖥️ System Information:\n"
    for key, value in info.items():
        result += f"- {key}: {value}\n"

    result += "\n📦 Optional Dependencies:\n"
    for dep, available in optional_deps.items():
        status = "✅" if available else "❌"
        result += f"- {dep}: {status}\n"

    return result


# ================== Interactive Agent Setup ==================

# Global memory store for the agent
agent_memory = EnhancedMemoryStore(
    max_short_term=100,
    max_working=10,
    enable_persistence=True,
    persistence_path=Path.home() / ".calute" / "interactive_memory",
)

# Configure logging
logger = get_logger("interactive_agent")


def create_interactive_agent() -> Agent:
    """Create the interactive test agent with all functions."""

    # All test functions
    functions = [
        list_improvements,
        test_memory_system,
        test_configuration,
        test_error_handling,
        test_logging_metrics,
        analyze_code_file,
        get_system_info,
    ]

    # Create agent with enhanced features
    agent = Agent(
        id="interactive_test_agent",
        name="Calute Test Agent",
        model="gpt-4o",
        instructions="""You are an interactive test agent for the Calute framework improvements.

Your purpose is to help users test and understand all the new features:
1. Enhanced memory system with indexing and search
2. Configuration management
3. Error handling and retries
4. Logging and metrics
5. System information

Available functions:
- list_improvements(): Show all improvements
- test_memory_system(action, content, tags): Test memory (actions: add, search, stats, demo)
- test_configuration(action, key, value): Test config (actions: show, set, save, load)
- test_error_handling(scenario): Test errors (scenarios: timeout, retry, validation)
- test_logging_metrics(action, level, message): Test logging (actions: log, metrics, raw_metrics)
- analyze_code_file(filepath): Analyze Python files
- get_system_info(): Get system information

Be helpful and explain what each feature does when demonstrating it.""",
        functions=functions,
        max_tokens=2048,
        temperature=0.7,
        function_timeout=30.0,
        max_function_retries=3,
    )

    return agent


class InteractiveCalute:
    """Interactive Calute wrapper for testing."""

    def __init__(self):
        self.agent = create_interactive_agent()
        self.messages = MessagesHistory(messages=[])
        self.logger = logger

        # Set up configuration
        self.config = CaluteConfig(
            environment="development",
            executor={"default_timeout": 30.0},
            memory={"max_short_term": 100},
            logging={"level": "INFO"},
        )
        set_config(self.config)

        # Create real OpenAI client
        self.client = OpenAI(api_key="N", base_url="http://137.184.183.6:11558/v1")
        self.calute = Calute(self.client, enable_memory=True)
        self.calute.memory = agent_memory
        self.calute.register_agent(self.agent)

        print("🤖 Interactive Calute Agent Ready!")
        print("=" * 60)

    async def process_command(self, user_input: str) -> str:
        """Process user command and return response."""

        # Add to messages
        self.messages.messages.append(UserMessage(content=user_input))

        try:
            # Use the actual Calute agent to process the command
            response = await self.calute.create_response(
                prompt=user_input,
                messages=self.messages,
                agent_id=self.agent,
                stream=False,
                apply_functions=True,
                use_instructed_prompt=True,
                reinvoke_after_function=True,
            )

            # Extract the content from the response
            if hasattr(response, "content"):
                response_text = response.content
            elif hasattr(response, "result") and hasattr(response.result, "content"):
                response_text = response.result.content
            else:
                response_text = str(response)
        except Exception as e:
            # Fallback to direct function execution if API fails
            self.logger.logger.warning(f"API call failed, using fallback: {e}")
            response_text = await self._execute_function_from_input(user_input)

        # Add assistant response
        self.messages.messages.append(AssistantMessage(content=response_text))

        # Add to memory
        agent_memory.add_memory(
            content=f"User: {user_input[:100]}",
            memory_type=MemoryType.EPISODIC,
            agent_id=self.agent.id,
            tags=["interaction"],
        )

        return response_text

    async def _execute_function_from_input(self, user_input: str) -> str:
        """Execute appropriate function based on user input."""
        input_lower = user_input.lower()

        # Map inputs to functions
        if "list improvement" in input_lower or "show improvement" in input_lower:
            return list_improvements()

        elif "memory" in input_lower:
            if "add" in input_lower:
                # Extract content after "add"
                parts = user_input.split("add", 1)
                content = parts[1].strip() if len(parts) > 1 else "Test memory"
                return test_memory_system("add", content, "user_added")
            elif "search" in input_lower:
                # Extract tags
                if "tag" in input_lower:
                    parts = user_input.split("tag", 1)
                    tags = parts[1].strip().strip("s").strip(":").strip() if len(parts) > 1 else ""
                else:
                    tags = ""
                return test_memory_system("search", "", tags)
            elif "stat" in input_lower:
                return test_memory_system("stats")
            else:
                return test_memory_system("demo")

        elif "config" in input_lower:
            if "show" in input_lower:
                return test_configuration("show")
            elif "save" in input_lower:
                return test_configuration("save")
            elif "load" in input_lower:
                return test_configuration("load")
            else:
                return test_configuration("show", "executor")

        elif "error" in input_lower or "timeout" in input_lower:
            if "timeout" in input_lower:
                return test_error_handling("timeout")
            elif "retry" in input_lower:
                return test_error_handling("retry")
            else:
                return test_error_handling("validation")

        elif "log" in input_lower or "metric" in input_lower:
            if "metric" in input_lower:
                return test_logging_metrics("metrics")
            else:
                return test_logging_metrics("log", "info", user_input)

        elif "analyze" in input_lower:
            # Extract filepath
            parts = user_input.split()
            for part in parts:
                if ".py" in part:
                    return analyze_code_file(part)
            # Default to analyzing calute.py
            return analyze_code_file("calute/calute.py")

        elif "system" in input_lower or "info" in input_lower:
            return get_system_info()

        elif "help" in input_lower or "?" in user_input:
            return """🤖 Interactive Agent Commands:

📊 **Improvements & Info:**
- "list improvements" - Show all improvements made
- "system info" - Get system information

🧠 **Memory System:**
- "test memory" - Demo the memory system
- "add memory: [content]" - Add to memory
- "search memory tag: [tag]" - Search by tag
- "memory stats" - Show statistics

⚙️ **Configuration:**
- "show config" - Display configuration
- "save config" - Save to file
- "load config" - Load from file

❌ **Error Handling:**
- "test timeout" - Demo timeout handling
- "test retry" - Demo retry mechanism
- "test validation" - Demo validation

📝 **Logging & Metrics:**
- "test logging" - Log a message
- "show metrics" - Display metrics
- "log: [message]" - Log specific message

🔍 **Code Analysis:**
- "analyze [filepath]" - Analyze Python file
- "analyze calute/calute.py" - Analyze main file

Type 'quit' to exit."""

        else:
            # Try to be helpful
            return f"""I understood: "{user_input}"

I can help you test these Calute improvements:
1. Memory system (try: "test memory")
2. Configuration (try: "show config")
3. Error handling (try: "test timeout")
4. Logging (try: "test logging")
5. Code analysis (try: "analyze calute/calute.py")

Type 'help' for all commands."""

    def run_interactive(self):
        """Run interactive loop."""
        print("Type 'help' for commands, 'quit' to exit\n")

        while True:
            try:
                user_input = input("You> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Process command
                response = asyncio.run(self.process_command(user_input))

                # Print response
                print("\n🤖 Agent>", response)
                print()

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.logger.error(f"Error processing command: {e}")


# ================== Main Execution ==================


def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     🚀 CALUTE INTERACTIVE TEST AGENT 🚀                  ║
║                                                          ║
║  Test all improvements and enhanced features            ║
║  Type 'help' for available commands                     ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Create and run interactive agent
    interactive = InteractiveCalute()

    # Show initial status
    print("📊 Initial Status:")
    print(get_system_info())
    print("\n" + "=" * 60)

    # Run some automatic tests first
    print("\n🧪 Running automatic tests...\n")

    tests = [
        ("Testing memory system...", test_memory_system("demo")),
        ("Checking configuration...", test_configuration("show", "executor.default_timeout")),
        ("Testing error handling...", test_error_handling("validation")),
        ("Generating metrics...", test_logging_metrics("metrics")),
    ]

    for description, result in tests:
        print(f"• {description}")
        print(f"  {result}\n")

    print("=" * 60)
    print("✅ Automatic tests complete!\n")

    # Start interactive mode
    interactive.run_interactive()


if __name__ == "__main__":
    main()
