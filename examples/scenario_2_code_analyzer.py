#!/usr/bin/env python3
"""
Scenario 2: Intelligent Code Analysis Agent
An agent that analyzes code, finds bugs, suggests improvements, and learns patterns.
"""

import ast
import asyncio
import os
import re
from pathlib import Path

import openai

from calute import Agent, Calute
from calute.executors import EnhancedAgentOrchestrator, EnhancedFunctionExecutor
from calute.memory import MemoryStore, MemoryType

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR-KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", None),
)

# Initialize memory for code patterns
code_memory = MemoryStore(
    max_short_term=200, enable_persistence=True, persistence_path=Path.home() / ".calute" / "code_analysis_memory"
)


def analyze_python_code(code: str) -> str:
    """Analyze Python code for syntax, style, and potential issues."""
    issues = []
    suggestions = []

    try:
        # Parse the code
        tree = ast.parse(code)

        # Analyze AST
        class CodeAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.stats = {
                    "functions": 0,
                    "classes": 0,
                    "imports": 0,
                    "variables": 0,
                    "loops": 0,
                    "conditions": 0,
                }
                self.issues = []
                self.complexity = 0

            def visit_FunctionDef(self, node):
                self.stats["functions"] += 1
                # Check function length
                if len(node.body) > 20:
                    self.issues.append(f"Function '{node.name}' is too long ({len(node.body)} lines)")
                # Check for docstring
                if not ast.get_docstring(node):
                    self.issues.append(f"Function '{node.name}' lacks docstring")
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.stats["classes"] += 1
                if not ast.get_docstring(node):
                    self.issues.append(f"Class '{node.name}' lacks docstring")
                self.generic_visit(node)

            def visit_Import(self, node):
                self.stats["imports"] += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.stats["loops"] += 1
                self.complexity += 1
                self.generic_visit(node)

            def visit_If(self, node):
                self.stats["conditions"] += 1
                self.complexity += 1
                self.generic_visit(node)

        analyzer = CodeAnalyzer()
        analyzer.visit(tree)

        # Store analysis in memory
        code_memory.add_memory(
            content=f"Code analysis: {analyzer.stats}",
            memory_type=MemoryType.SEMANTIC,
            agent_id="code_analyzer",
            tags=["analysis", "python", "statistics"],
            importance_score=0.7,
        )

        # Build response
        result = "📊 Code Statistics:\n"
        for key, value in analyzer.stats.items():
            result += f"  - {key}: {value}\n"

        result += f"\n🔍 Complexity Score: {analyzer.complexity}\n"

        if analyzer.issues:
            result += "\n⚠️ Issues Found:\n"
            for issue in analyzer.issues:
                result += f"  - {issue}\n"
                issues.append(issue)

        # Check for common patterns
        if "import *" in code:
            issues.append("Avoid wildcard imports")
        if "except:" in code or "except Exception:" in code:
            issues.append("Use specific exception handling")
        if re.search(r"print\s*\(", code):
            suggestions.append("Consider using logging instead of print statements")

        if suggestions:
            result += "\n💡 Suggestions:\n"
            for suggestion in suggestions:
                result += f"  - {suggestion}\n"

        return result

    except SyntaxError as e:
        return f"❌ Syntax Error: {e}"
    except Exception as e:
        return f"❌ Analysis Error: {e}"


def find_security_issues(code: str) -> str:
    """Find potential security vulnerabilities in code."""
    vulnerabilities = []

    # Check for common security issues
    patterns = {
        r"eval\s*\(": "Dangerous use of eval() - can execute arbitrary code",
        r"exec\s*\(": "Dangerous use of exec() - can execute arbitrary code",
        r"pickle\.loads": "Pickle deserialization can execute arbitrary code",
        r"os\.system": "Command injection risk with os.system()",
        r"subprocess\.call\s*\(.*shell\s*=\s*True": "Shell injection risk",
        r'password\s*=\s*["\']': "Hardcoded password detected",
        r'api_key\s*=\s*["\']': "Hardcoded API key detected",
        r'SECRET\s*=\s*["\']': "Hardcoded secret detected",
    }

    for pattern, description in patterns.items():
        if re.search(pattern, code, re.IGNORECASE):
            vulnerabilities.append(description)

    # Store findings in memory
    if vulnerabilities:
        code_memory.add_memory(
            content=f"Security issues found: {', '.join(vulnerabilities[:3])}",
            memory_type=MemoryType.EPISODIC,
            agent_id="code_analyzer",
            tags=["security", "vulnerability", "alert"],
            importance_score=0.9,
        )

    if vulnerabilities:
        result = "🔐 Security Issues Found:\n"
        for vuln in vulnerabilities:
            result += f"  ⚠️ {vuln}\n"
        result += "\n🛡️ Recommendation: Review and fix these security issues immediately."
    else:
        result = "✅ No obvious security issues detected."

    return result


def suggest_refactoring(code: str) -> str:
    """Suggest code refactoring improvements."""
    suggestions = []

    # Analyze code patterns
    lines = code.split("\n")

    # Check for long lines
    for i, line in enumerate(lines, 1):
        if len(line) > 100:
            suggestions.append(f"Line {i}: Consider breaking long line (>{100} chars)")

    # Check for nested conditions
    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if indent_levels and max(indent_levels) > 12:  # 3 levels of 4-space indentation
        suggestions.append("Deep nesting detected - consider extracting methods")

    # Check for duplicate code patterns
    code_blocks = re.findall(r"def \w+\(.*?\):.*?(?=def|\Z)", code, re.DOTALL)
    if len(code_blocks) > 1:
        # Simple duplicate detection
        for i, block1 in enumerate(code_blocks):
            for block2 in code_blocks[i + 1 :]:
                similarity = sum(1 for a, b in zip(block1.split(), block2.split(), strict=False) if a == b)
                if similarity > 20:  # More than 20 similar tokens
                    suggestions.append("Possible code duplication detected - consider extracting common code")
                    break

    # Check for magic numbers
    magic_numbers = re.findall(r"\b\d{2,}\b", code)
    if magic_numbers:
        suggestions.append(f"Magic numbers found ({', '.join(set(magic_numbers)[:3])}) - consider using constants")

    # Store refactoring suggestions
    if suggestions:
        code_memory.add_memory(
            content=f"Refactoring suggestions: {len(suggestions)} improvements",
            memory_type=MemoryType.SEMANTIC,
            agent_id="code_analyzer",
            tags=["refactoring", "improvement", "code_quality"],
            importance_score=0.6,
        )

    if suggestions:
        result = "🔧 Refactoring Suggestions:\n"
        for suggestion in suggestions[:10]:  # Limit to 10 suggestions
            result += f"  • {suggestion}\n"
    else:
        result = "✨ Code looks well-structured!"

    return result


def generate_tests(code: str) -> str:
    """Generate test cases for the given code."""
    test_cases = []

    try:
        # Parse code to find functions
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                params = [arg.arg for arg in node.args.args]

                # Generate basic test template
                test_case = f"""
def test_{func_name}():
    # Test case for {func_name}
"""

                if not params:
                    test_case += f"    result = {func_name}()\n"
                    test_case += "    assert result is not None  # Add specific assertion\n"
                else:
                    # Generate parameter examples
                    param_values = []
                    for param in params:
                        if "id" in param.lower() or "count" in param.lower():
                            param_values.append("1")
                        elif "name" in param.lower() or "text" in param.lower():
                            param_values.append('"test"')
                        elif "flag" in param.lower() or "is_" in param:
                            param_values.append("True")
                        else:
                            param_values.append("None")

                    test_case += f"    result = {func_name}({', '.join(param_values)})\n"
                    test_case += "    # Add assertions here\n"
                    test_case += "    assert result is not None\n"

                test_cases.append(test_case)

        if test_cases:
            result = "🧪 Generated Test Templates:\n"
            result += "```python\nimport pytest\n\n"
            for test in test_cases[:5]:  # Limit to 5 tests
                result += test
            result += "```"

            # Store test generation in memory
            code_memory.add_memory(
                content=f"Generated {len(test_cases)} test templates",
                memory_type=MemoryType.PROCEDURAL,
                agent_id="code_analyzer",
                tags=["testing", "test_generation", "quality"],
                importance_score=0.7,
            )
        else:
            result = "No functions found to generate tests for."

    except Exception as e:
        result = f"Could not generate tests: {e}"

    return result


def check_best_practices(code: str) -> str:
    """Check if code follows Python best practices."""
    practices = {"passed": [], "failed": []}

    # PEP 8 checks (simplified)
    if re.search(r"^[A-Z][a-zA-Z]*", code, re.MULTILINE):
        practices["passed"].append("✓ Class names use CapWords convention")

    if re.search(r"^[a-z_][a-z0-9_]*\s*=", code, re.MULTILINE):
        practices["passed"].append("✓ Variable names use snake_case")

    if re.search(r"^def [a-z_][a-z0-9_]*\(", code, re.MULTILINE):
        practices["passed"].append("✓ Function names use snake_case")

    if '"""' in code or "'''" in code:
        practices["passed"].append("✓ Uses docstrings")
    else:
        practices["failed"].append("✗ Missing docstrings")

    if re.search(r"^\s{4}", code, re.MULTILINE):
        practices["passed"].append("✓ Uses 4-space indentation")
    elif re.search(r"^\t", code, re.MULTILINE):
        practices["failed"].append("✗ Uses tabs instead of spaces")

    if not re.search(r"\s+$", code, re.MULTILINE):
        practices["passed"].append("✓ No trailing whitespace")
    else:
        practices["failed"].append("✗ Has trailing whitespace")

    # Type hints check
    if re.search(r"def \w+\([^)]*:.*?\)", code):
        practices["passed"].append("✓ Uses type hints")
    else:
        practices["failed"].append("✗ Consider adding type hints")

    result = "📋 Best Practices Check:\n"
    if practices["passed"]:
        result += "\n✅ Following:\n"
        for practice in practices["passed"]:
            result += f"  {practice}\n"

    if practices["failed"]:
        result += "\n❌ Needs Improvement:\n"
        for practice in practices["failed"]:
            result += f"  {practice}\n"

    score = len(practices["passed"]) / (len(practices["passed"]) + len(practices["failed"]))
    result += f"\n📊 Score: {score:.0%}"

    return result


async def main():
    """Run the code analysis scenario."""
    print("=" * 60)
    print("🔍 INTELLIGENT CODE ANALYSIS AGENT")
    print("=" * 60)
    print()

    # Create the code analysis agent
    agent = Agent(
        id="code_analyzer",
        name="Code Analysis Expert",
        model="gpt-4o",
        instructions="""You are an expert code analyzer specializing in Python.
        Your role is to:
        1. Analyze code for quality, security, and performance
        2. Suggest improvements and refactoring
        3. Generate test cases
        4. Check best practices
        5. Learn from patterns you see

        Be thorough but concise in your analysis.""",
        functions=[analyze_python_code, find_security_issues, suggest_refactoring, generate_tests, check_best_practices],
        max_tokens=1000,
        temperature=0.3,  # Lower temperature for more consistent analysis
    )

    # Set up enhanced executor for better function handling
    orchestrator = EnhancedAgentOrchestrator(enable_metrics=True)
    await orchestrator.register_agent(agent)

    executor = EnhancedFunctionExecutor(orchestrator=orchestrator, default_timeout=10.0, max_concurrent_executions=5)

    # Initialize Calute
    calute = Calute(client, enable_memory=True)
    calute.memory = code_memory
    calute.register_agent(agent)

    # Example code to analyze
    test_code = """
import os
import pickle

class DataProcessor:
    def __init__(self):
        self.api_key = "sk-1234567890"
        self.data = []

    def process_data(self, input_data):
        # Process the data
        for item in input_data:
            if item > 100:
                self.data.append(item * 2)
            elif item > 50:
                self.data.append(item * 1.5)
            else:
                self.data.append(item)

        return self.data

    def save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self, filename):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)

    def execute_command(self, cmd):
        os.system(cmd)

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)

def main():
    processor = DataProcessor()
    data = [10, 55, 105, 200, 45, 88, 150, 33, 99, 175]
    result = processor.process_data(data)
    print(result)
    avg = calculate_average(result)
    print("Average:", avg)
"""

    print("📝 Analyzing the following code:")
    print("-" * 40)
    print(test_code)
    print("-" * 40)
    print()

    # Analyze different aspects using the enhanced executor
    from calute.types import RequestFunctionCall

    # Create function calls for analysis
    analysis_calls = [
        RequestFunctionCall(name="analyze_python_code", arguments={"code": test_code}, id="analysis_1"),
        RequestFunctionCall(name="find_security_issues", arguments={"code": test_code}, id="security_1"),
        RequestFunctionCall(name="suggest_refactoring", arguments={"code": test_code}, id="refactor_1"),
        RequestFunctionCall(name="generate_tests", arguments={"code": test_code}, id="tests_1"),
        RequestFunctionCall(name="check_best_practices", arguments={"code": test_code}, id="practices_1"),
    ]

    print("🔎 Running analysis with Enhanced Executor...")

    # Execute all analyses in parallel using the enhanced executor
    from calute.types import FunctionCallStrategy

    try:
        results = await executor.execute_function_calls(
            calls=analysis_calls,
            strategy=FunctionCallStrategy.PARALLEL,
            context_variables={"code": test_code},
            agent=agent,
        )

        # Display results
        analysis_names = {
            "analyze_python_code": "Code Analysis",
            "find_security_issues": "Security Check",
            "suggest_refactoring": "Refactoring Suggestions",
            "generate_tests": "Test Generation",
            "check_best_practices": "Best Practices",
        }

        for call, result in zip(analysis_calls, results, strict=False):
            print(f"\n📊 {analysis_names.get(call.name, call.name)}:")
            print("-" * 40)
            if hasattr(result, "result"):
                print(result.result)
            else:
                print(str(result))

    except Exception as e:
        print(f"Error during enhanced execution: {e}")
        # Fallback to direct function calls
        print("\nFalling back to direct function execution...")
        print(analyze_python_code(test_code))
        print(find_security_issues(test_code))
        print(suggest_refactoring(test_code))
        print(generate_tests(test_code))
        print(check_best_practices(test_code))

        print("\n" + "-" * 40 + "\n")
        await asyncio.sleep(1)

    # Show learned patterns
    print("=" * 60)
    print("📚 LEARNED CODE PATTERNS")
    print("=" * 60)

    patterns = code_memory.retrieve_memories(tags=["analysis"], limit=10)

    if patterns:
        print("Stored analysis patterns:")
        for pattern in patterns:
            print(f"  • {pattern.content}")

    # Show metrics
    if hasattr(orchestrator, "function_registry"):
        print("\n📊 Function Execution Metrics:")
        for func_name in [
            "analyze_python_code",
            "find_security_issues",
            "suggest_refactoring",
            "generate_tests",
            "check_best_practices",
        ]:
            metrics = orchestrator.function_registry.get_metrics(func_name)
            if metrics and metrics.total_calls > 0:
                print(f"  {func_name}:")
                print(f"    - Calls: {metrics.total_calls}")
                print(f"    - Success rate: {metrics.successful_calls / metrics.total_calls:.0%}")
                print(f"    - Avg duration: {metrics.average_duration:.2f}s")

    # Save memory
    code_memory.save()
    print("\n💾 Analysis patterns saved for future use!")

    print("\n✅ Code analysis scenario completed!")


if __name__ == "__main__":
    asyncio.run(main())
