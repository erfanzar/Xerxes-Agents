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


"""Pre-built Coder Agent for intelligent code generation and analysis.

This module provides a pre-configured AI agent specialized in software development
tasks within the Calute framework. The coder agent is designed to assist with
various programming tasks including writing, reviewing, debugging, and optimizing
code across different programming languages.

The agent is equipped with specialized tools for:
- Executing Python code in a sandboxed environment
- Reading and writing files for code manipulation
- Executing shell commands for build and test operations
- File system operations for project navigation

Agent Capabilities:
    - Code Generation: Writing clean, efficient, and maintainable code
    - Debugging: Identifying and fixing complex issues in codebases
    - Refactoring: Restructuring code for better organization and readability
    - Test Generation: Creating comprehensive test suites
    - Documentation: Generating clear code documentation
    - Code Review: Analyzing code quality and suggesting improvements

Typical usage example:
    from calute import Calute
    from calute.agents import code_agent

    calute = Calute(llm=your_llm)
    response = calute.run(
        prompt="Write a function to calculate fibonacci numbers",
        agent_id=code_agent
    )

Note:
    The agent uses a moderate temperature (0.6) to balance creativity with
    accuracy in code generation. It has a higher token limit (4096) to
    accommodate complex code outputs.
"""

from ..tools import ExecutePythonCode, ExecuteShell, FileSystemTools, ReadFile, WriteFile
from ..types import Agent

code_agent = Agent(
    id="coder_agent",
    name="Coder Assistant",
    model=None,
    instructions="""You are an expert software engineer and code architect.

Your specialties include:
- Writing clean, efficient, and maintainable code
- Debugging and fixing complex issues
- Refactoring code for better structure
- Generating comprehensive tests
- Creating clear documentation
- Analyzing code quality and suggesting improvements

Guidelines:
1. Always follow best practices for the language
2. Write secure code with proper error handling
3. Optimize for readability first, performance second
4. Include helpful comments and documentation
5. Consider edge cases and error scenarios
6. Follow established style guides
7. Write testable, modular code

When generating code:
- Ask for clarification if requirements are unclear
- Provide multiple solutions when appropriate
- Explain trade-offs between different approaches
- Include example usage when helpful

You have access to various code analysis and generation tools.
Use them strategically to provide the best assistance.""",
    functions=[ExecutePythonCode, ReadFile, WriteFile, ExecuteShell, FileSystemTools],
    temperature=0.6,
    max_tokens=4096,
)
