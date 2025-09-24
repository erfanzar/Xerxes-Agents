
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

"""
Coder Agent - An intelligent code generation and analysis agent.

This agent specializes in:
- Code generation and refactoring
- Code review and optimization
- Bug fixing and debugging
- Test generation
- Documentation generation
- Code analysis and metrics
"""

from calute import Agent
from calute.tools import ExecutePythonCode, ExecuteShell, FileSystemTools, ReadFile, WriteFile

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
