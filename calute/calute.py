# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import json
import re
import typing as tp
from dataclasses import dataclass
from enum import Enum

from .client import GeminiClient, OpenAIClient
from .executors import AgentOrchestrator, FunctionExecutor
from .memory import MemoryStore, MemoryType
from .types import (
    Agent,
    AgentFunction,
    AgentSwitch,
    AgentSwitchTrigger,
    Completion,
    ExecutionStatus,
    FunctionCall,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
)
from .utils import function_to_json

SEP = "  "  # two spaces
add_depth = (  # noqa
    lambda x, ep=False: SEP + x.replace("\n", f"\n{SEP}") if ep else x.replace("\n", f"\n{SEP}")
)


class PromptSection(Enum):
    SYSTEM = "system"
    PERSONA = "persona"
    RULES = "rules"
    FUNCTIONS = "functions"
    TOOLS = "tools"
    EXAMPLES = "examples"
    CONTEXT = "context"
    HISTORY = "history"
    PROMPT = "prompt"


@dataclass
class PromptTemplate:
    """Configurable template for structuring agent prompts"""

    sections: dict[PromptSection, str] = None
    section_order: list[PromptSection] = None

    def __post_init__(self):
        self.sections = self.sections or {
            PromptSection.SYSTEM: f"SYSTEM:\n{SEP}",
            PromptSection.PERSONA: f"PERSONA:\n{SEP}Your style and approach:",
            PromptSection.RULES: f"RULES:\n{SEP}",
            PromptSection.FUNCTIONS: f"FUNCTIONS:\n{SEP}The available functions are listed with their schemas:",
            PromptSection.TOOLS: f"TOOLS:\n{SEP}When using tools, follow this format:",
            PromptSection.EXAMPLES: f"EXAMPLES:\n{SEP}",
            PromptSection.CONTEXT: f"CONTEXT:\n{SEP}Current variables:\n",
            PromptSection.HISTORY: f"HISTORY:\n{SEP}Conversation so far:\n",
            PromptSection.PROMPT: f"PROMPT:\n{SEP}",
        }

        self.section_order = self.section_order or [
            PromptSection.SYSTEM,
            PromptSection.PERSONA,
            PromptSection.RULES,
            PromptSection.FUNCTIONS,
            PromptSection.TOOLS,
            PromptSection.EXAMPLES,
            PromptSection.CONTEXT,
            PromptSection.HISTORY,
            PromptSection.PROMPT,
        ]


class Calute:
    """Calute with orchestration"""

    def __init__(self, client, template: PromptTemplate | None = None, enable_memory: bool = True):
        """
        Initialize Calute with an LLM client.

        Args:
            client: An instance of OpenAI client or Google Gemini client
            template: Optional prompt template
        """
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            self.llm_client = OpenAIClient(client)
        elif hasattr(client, "GenerativeModel"):
            self.llm_client = GeminiClient(client)
        else:
            raise ValueError("Unsupported client type. Must be OpenAI or Gemini.")

        self.template = template or PromptTemplate()
        self.orchestrator = AgentOrchestrator()
        self.executor = FunctionExecutor(self.orchestrator)
        self.enable_memory = enable_memory
        if enable_memory:
            self.memory_store = MemoryStore()
        self._setup_default_triggers()

    def _setup_default_triggers(self):
        """Setup default agent switching triggers"""

        def capability_based_switch(context, agents, current_agent_id):
            """Switch agent based on required capabilities"""
            required_capability = context.get("required_capability")
            if not required_capability:
                return None

            best_agent = None
            best_score = 0

            for agent_id, agent in agents.items():
                if agent.has_capability(required_capability):
                    for cap in agent.capabilities:
                        if cap.name == required_capability and cap.performance_score > best_score:
                            best_agent = agent_id
                            best_score = cap.performance_score

            return best_agent

        def error_recovery_switch(context, agents, current_agent_id):
            """Switch agent on function execution errors"""
            if context.get("execution_error") and current_agent_id:
                current_agent = agents[current_agent_id]
                if current_agent.fallback_agent_id:
                    return current_agent.fallback_agent_id
            return None

        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.CAPABILITY_BASED, capability_based_switch)
        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.ERROR_RECOVERY, error_recovery_switch)

    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        self.orchestrator.register_agent(agent)

    def _update_memory_from_response(
        self,
        content: str,
        agent_id: str,
        context_variables: dict | None = None,
        function_calls: list[FunctionCall] | None = None,
    ):
        """Update memory based on response"""
        if not self.enable_memory:
            return

        # Add response to short-term memory
        self.memory_store.add_memory(
            content=f"Assistant response: {content[:200]}...",
            memory_type=MemoryType.SHORT_TERM,
            agent_id=agent_id,
            context=context_variables or {},
            importance_score=0.6,
        )

        if function_calls:
            for call in function_calls:
                self.memory_store.add_memory(
                    content=f"Function called: {call.name} with args: {call.arguments}",
                    memory_type=MemoryType.WORKING,
                    agent_id=agent_id,
                    context={"function_id": call.id, "status": call.status.value},
                    importance_score=0.7,
                    tags=["function_call", call.name],
                )

    def _update_memory_from_prompt(self, prompt: str, agent_id: str):
        """Update memory from user prompt"""
        if not self.enable_memory:
            return

        self.memory_store.add_memory(
            content=f"User prompt: {prompt}",
            memory_type=MemoryType.SHORT_TERM,
            agent_id=agent_id,
            importance_score=0.8,
            tags=["user_input"],
        )

    def _extract_from_markdown(self, content: str, field: str) -> list[FunctionCall]:
        """Extract function calls from response content"""

        pattern = rf"```{field}\s*\n(.*?)\n```"
        return re.findall(pattern, content, re.DOTALL)

    def _extract_function_calls(self, content: str) -> list[FunctionCall]:
        """Extract function calls from response content"""
        function_calls = []

        matches = self._extract_from_markdown(content=content, field="tool_call")

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                function_call = FunctionCall(
                    name=call_data.get("name"),
                    arguments=call_data.get("content", {}),
                    id=f"call_{i}_{hash(match)}",
                    timeout=self.orchestrator.get_current_agent().function_timeout,
                    max_retries=self.orchestrator.get_current_agent().max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                continue

        return function_calls

    def generate_prompt(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: dict | None = None,
        history: list[dict] | None = None,
        include_memory: bool = True,
        use_chain_of_thought: bool = False,
        require_reflection: bool = False,
    ) -> str:
        """
        Generates a structured prompt using the enhanced template with memory integration

        Args:
            agent: The agent to generate prompt for
            prompt: The user's prompt/query
            context_variables: Additional context variables
            history: Conversation history
            include_memory: Whether to include memory context
            use_chain_of_thought: Whether to wrap prompt in chain-of-thought format
            require_reflection: Whether to require reflection in response

        Returns:
            Formatted prompt string
        """

        if not agent:
            return prompt or "You are a helpful assistant."

        # Update memory with the prompt if enabled
        if self.enable_memory and prompt:
            self._update_memory_from_prompt(prompt, agent.id)

        # Check what components are being used
        rules_being_used = agent.rules is not None and len(agent.rules) > 0
        functions_being_used = agent.functions is not None and len(agent.functions) > 0
        examples_being_used = agent.examples is not None and len(agent.examples) > 0
        tools_being_used = agent.tool_choice is not None and len(agent.tool_choice) > 0

        sections = {}

        # SYSTEM SECTION
        system_content = agent.name or self.template.sections[PromptSection.SYSTEM]
        if agent.capabilities:
            capabilities_str = ", ".join([cap.name for cap in agent.capabilities])
            system_content += f"\nCapabilities: {capabilities_str}"
        sections[PromptSection.SYSTEM] = f"{self.template.sections[PromptSection.SYSTEM]}\n{SEP}{system_content}"

        instructions = str((agent.instructions() if callable(agent.instructions) else agent.instructions) or "")

        if use_chain_of_thought:
            instructions += (
                f"\n{SEP}Approach every task systematically:\n"
                f"{SEP * 2}1. Understand the request fully\n"
                f"{SEP * 2}2. Break down complex problems\n"
                f"{SEP * 2}3. Consider edge cases\n"
                f"{SEP * 2}4. Validate your approach\n"
                f"{SEP * 2}5. Provide clear, structured output"
            )

        sections[PromptSection.PERSONA] = f"{self.template.sections[PromptSection.PERSONA]} {instructions}"

        if rules_being_used or functions_being_used or include_memory:
            rules = []

            if rules_being_used:
                agent_rules = agent.rules if isinstance(agent.rules, list) else [agent.rules]
                rules.extend(agent_rules)

            if functions_being_used:
                rules.append("If a function can satisfy the user, respond only with a tool_call JSON and nothing else.")

            # Add memory-related rules if enabled
            if self.enable_memory and include_memory:
                rules.extend(
                    [
                        "Consider previous context and interactions when relevant",
                        "Build upon earlier conversations when appropriate",
                        "Acknowledge when you're using information from memory",
                    ]
                )

            rules_string = "\n".join([f"{SEP}- {rule}" for rule in rules])
            sections[PromptSection.RULES] = f"{self.template.sections[PromptSection.RULES]}\n{rules_string}"

        if functions_being_used:
            fn_docs = self.generate_function_section(agent.functions)
            fn_docs_formatted = "\n" + SEP + add_depth(fn_docs, ep=True)

            sections[PromptSection.FUNCTIONS] = f"{self.template.sections[PromptSection.FUNCTIONS]}{fn_docs_formatted}"

        if functions_being_used:
            fn_docs = self.generate_function_section(agent.functions)
            fn_docs = "\n" + SEP + add_depth(add_depth(fn_docs, ep=True))
            sections[PromptSection.FUNCTIONS] = f"{self.template.sections[PromptSection.FUNCTIONS]} {fn_docs}"

        if tools_being_used:
            sections[PromptSection.TOOLS] = f"{self.template.sections[PromptSection.TOOLS]} {agent.tool_choice}"

        example_text = ""

        if examples_being_used:
            example_text = "\n\n".join([add_depth(ex, True) for i, ex in enumerate(agent.examples)])

        if functions_being_used:
            example_text += (
                "When calling a function, you must include the function name and all required inputs "
                "inside XML tags. The tag name should be the function name, and parameters should be nested within `<arguments>` tags.\n\n"  # noqa
                "Here is an example:\n"
                f'{SEP * 2}User Prompt: "..."\n'
                f"{SEP * 2}Assistant: <{agent.functions[0].__name__}>\n"
                f"{SEP * 3}<arguments>\n"
                f"{SEP * 4}{{...}}\n"
                f"{SEP * 3}</arguments>\n"
                f"{SEP * 2}</{agent.functions[0].__name__}>"
            )

        if example_text:
            sections[PromptSection.EXAMPLES] = f"{self.template.sections[PromptSection.EXAMPLES]}\n{example_text}"

        # CONTEXT SECTION
        context_parts = []

        if self.enable_memory and include_memory:
            memory_context = self.memory_store.consolidate_memories(agent.id)
            if memory_context:
                context_parts.append(f"MEMORY:\n{add_depth(memory_context, True)}")

        if context_variables:
            ctx_vars_formatted = self.format_context_variables(context_variables)
            if ctx_vars_formatted:
                context_parts.append(f"VARIABLES:\n{add_depth(ctx_vars_formatted, True)}")

        if hasattr(self, "executor") and self.executor.execution_history.executions:
            recent_functions = self.executor.execution_history.get_successful_results()
            if recent_functions:
                func_history = "RECENT FUNCTION RESULTS:\n"
                for fname, result in list(recent_functions.items())[-3:]:  # Last 3 results
                    result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    func_history += f"{SEP}- {fname}: {result_str}\n"
                context_parts.append(func_history.rstrip())

        if context_parts:
            sections[PromptSection.CONTEXT] = f"{self.template.sections[PromptSection.CONTEXT]}\n" + "\n\n".join(
                context_parts
            )

        if history:
            formatted_history = self.format_chat_history(history)
            if formatted_history:
                sections[PromptSection.HISTORY] = (
                    f"{self.template.sections[PromptSection.HISTORY]}\n{add_depth(formatted_history, True)}"
                )

        if prompt is not None:
            formatted_prompt = prompt

            if use_chain_of_thought:
                formatted_prompt = (
                    f"{prompt}\n\n"
                    f"{SEP}Please approach this systematically:\n"
                    f"{SEP * 2}1. First, clearly understand what is being asked\n"
                    f"{SEP * 2}2. Identify any constraints or requirements\n"
                    f"{SEP * 2}3. Determine if any functions would be helpful\n"
                    f"{SEP * 2}4. Formulate your response\n"
                    f"{SEP * 2}5. Verify your response addresses the request fully"
                )

            if require_reflection:
                formatted_prompt += (
                    f"\n\n{SEP}After your response, please briefly reflect on:\n"
                    f"{SEP * 2}- Any assumptions you made\n"
                    f"{SEP * 2}- Potential limitations of your approach\n"
                    f"{SEP * 2}- Alternative solutions that might exist"
                )

            sections[PromptSection.PROMPT] = (
                f"{self.template.sections[PromptSection.PROMPT]} {add_depth(formatted_prompt, True)}"
            )

        parts = []
        for section in self.template.section_order:
            if sections.get(section):
                parts.append(sections[section])

        return "\n\n".join(parts).strip()

    def _extract_function_calls_from_xml(self, content: str) -> list[FunctionCall]:
        """Extract function calls from response content using XML tags"""
        function_calls = []
        pattern = r"<(\w+)>\s*<arguments>(.*?)</arguments>\s*</\w+>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            name = match[0]
            arguments_str = match[1].strip()
            try:
                arguments = json.loads(arguments_str)
                function_call = FunctionCall(
                    name=name,
                    arguments=arguments,
                    id=f"call_{i}_{hash(match)}",
                    timeout=self.orchestrator.get_current_agent().function_timeout,
                    max_retries=self.orchestrator.get_current_agent().max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                # Handle cases where arguments are not valid JSON
                continue

        return function_calls

    def _extract_function_calls(self, content: str) -> list[FunctionCall]:
        """Extract function calls from response content"""
        function_calls = self._extract_function_calls_from_xml(content)
        if function_calls:
            return function_calls

        function_calls = []
        matches = self._extract_from_markdown(content=content, field="tool_call")

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                function_call = FunctionCall(
                    name=call_data.get("name"),
                    arguments=call_data.get("content", {}),
                    id=f"call_{i}_{hash(match)}",
                    timeout=self.orchestrator.get_current_agent().function_timeout,
                    max_retries=self.orchestrator.get_current_agent().max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                continue

        return function_calls

    @staticmethod
    def extract_from_markdown(format: str, string: str) -> str | None | dict:  # noqa:A002
        search_mour = f"```{format}"
        index = string.find(search_mour)

        if index != -1:
            choosen = string[index + len(search_mour) :]
            if choosen.endswith("```"):
                choosen = choosen[:-3]
            try:
                return json.loads(choosen)
            except Exception:
                return choosen
        return None

    @staticmethod
    def get_thoughts(response: str, tag: str = "think") -> str:
        inside = None
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, flags=re.S)
        if match:
            inside = match.group(1).strip()
        return inside

    @staticmethod
    def filter_thoughts(response: str, tag: str = "think") -> str:
        before, after = re.split(rf"<{tag}>.*?</{tag}>", response, maxsplit=1, flags=re.S)
        string = "".join(before) + "".join(after)
        return string.strip()

    def format_function_parameters(self, parameters: dict) -> str:
        """Formats function parameters in a clear, structured way"""
        if not parameters.get("properties"):
            return ""

        formatted_params = []
        required_params = parameters.get("required", [])

        for param_name, param_info in parameters["properties"].items():
            if param_name == "context_variables":
                continue

            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            required = "(required)" if param_name in required_params else "(optional)"

            param_str = f"    - {param_name}: {param_type} {required}"
            if param_desc:
                param_str += f"\n      Description: {param_desc}"
            if "enum" in param_info:
                param_str += f"\n      Allowed values: {', '.join(str(v) for v in param_info['enum'])}"

            formatted_params.append(param_str)

        return "\n".join(formatted_params)

    def generate_function_section(self, functions: list[AgentFunction]) -> str:
        """Generates detailed function documentation with improved formatting and strict schema requirements"""
        if not functions:
            return ""

        function_docs = []
        for func in functions:
            try:
                schema = function_to_json(func)["function"]
                doc = [f"Function: {schema['name']}", f"Purpose: {schema['description']}"]
                params = self.format_function_parameters(schema["parameters"])
                if params:
                    doc.append("Parameters:")
                    doc.append(params)
                if "returns" in schema:
                    doc.append(f"Returns: {schema['returns']}")

                function_docs.append("\n".join(doc))

            except Exception as e:
                func_name = getattr(func, "__name__", str(func))
                function_docs.append(f"Warning: Unable to parse function {func_name}: {e!s}")

        return "\n\n".join(function_docs)

    def format_context_variables(self, variables: dict[str, tp.Any]) -> str:
        """Formats context variables with type information and improved readability"""
        if not variables:
            return ""

        formatted_vars = []
        for key, value in variables.items():
            var_type = type(value).__name__
            formatted_value = str(value) if len(str(value)) < 50 else f"{str(value)[:47]}..."
            formatted_vars.append(f"- {key} ({var_type}): {formatted_value}")
        return "\n".join(formatted_vars)

    def format_prompt(self, prompt: str | None) -> str:
        if not prompt:
            return ""
        return prompt

    def format_chat_history(self, history: list[dict]) -> str:
        """Formats chat history with improved readability and metadata"""
        if not history:
            return ""

        formatted_messages = []
        for msg in history:
            role_display = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System",
                "tool": "Tool",
            }.get(msg.role, msg.role.capitalize())

            timestamp = getattr(msg, "timestamp", "")
            time_str = f" at {timestamp}" if timestamp else ""

            formatted_messages.append(f"{role_display}{time_str}:\n{msg.content}")

        return "\n\n".join(formatted_messages)

    async def create_response(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        history: list[dict] | None = None,
        agent_id: str | None = None,
        stream: bool = True,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
    ) -> ResponseResult | tp.AsyncIterator[StreamingResponseType]:
        """Create response with enhanced function calling and agent switching"""

        if agent_id:
            self.orchestrator.switch_agent(agent_id, "User specified agent")

        agent = self.orchestrator.get_current_agent()
        context_variables = context_variables or {}

        formatted_prompt = self.generate_prompt(
            agent=agent,
            prompt=prompt,
            context_variables=context_variables,
            history=history,
        )

        if print_formatted_prompt:
            print(formatted_prompt)

        response = await self.llm_client.generate_completion(
            prompt=formatted_prompt,
            model=agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            top_p=agent.top_p,
            stop=agent.stop,
            stream=stream,
        )

        if not apply_functions:
            return response

        if stream:
            return self._handle_streaming_with_functions(
                response,
                agent,
                context_variables,
            )
        else:
            return await self._handle_response_with_functions(
                response,
                agent,
                context_variables,
            )

    async def _handle_response_with_functions(
        self,
        response: tp.Any,
        agent: "Agent",
        context: dict,
    ) -> ResponseResult:
        """Handle non-streaming response with function calls"""

        content = self.llm_client.extract_content(response)
        function_calls = self._extract_function_calls(content)

        if function_calls:
            results = await self.executor.execute_function_calls(
                function_calls,
                agent.function_call_strategy,
                context,
            )

            switch_context = SwitchContext(
                function_results=results,
                execution_error=any(r.status == ExecutionStatus.FAILURE for r in results),
            )

            target_agent = self.orchestrator.should_switch_agent(switch_context.__dict__)
            if target_agent:
                self.orchestrator.switch_agent(target_agent, "Post-execution switch")

        return ResponseResult(
            content=content,
            response=response,
            function_calls=function_calls if function_calls else [],
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-5:],
        )

    async def _handle_streaming_with_functions(
        self,
        response: tp.Any,
        agent: "Agent",
        context: dict,
    ) -> tp.AsyncIterator[StreamingResponseType]:
        """Handle streaming response with function calls"""
        buffered_content = ""
        function_calls_detected = False
        function_calls = []

        if isinstance(self.llm_client, OpenAIClient):
            for chunk in response:  # shouldn't be async
                content = None
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        buffered_content += delta.content
                        content = delta.content

                        if "<" in buffered_content and not function_calls_detected:
                            function_calls_detected = True

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                )
        elif isinstance(self.llm_client, GeminiClient):
            for chunk in response:  # shouldn't be async
                content = None
                if hasattr(chunk, "text") and chunk.text:
                    buffered_content += chunk.text
                    content = chunk.text

                    if "<" in buffered_content and not function_calls_detected:
                        function_calls_detected = True

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                )

        if function_calls_detected:
            yield FunctionDetection(
                message="Processing function calls...",
                agent_id=self.orchestrator.current_agent_id,
            )

            function_calls = self._extract_function_calls(buffered_content)

            if function_calls:
                yield FunctionCallsExtracted(
                    function_calls=[FunctionCallInfo(name=fc.name, id=fc.id) for fc in function_calls],
                    agent_id=self.orchestrator.current_agent_id,
                )

                results = []
                for i, call in enumerate(function_calls):
                    yield FunctionExecutionStart(
                        function_name=call.name,
                        function_id=call.id,
                        progress=f"{i + 1}/{len(function_calls)}",
                        agent_id=self.orchestrator.current_agent_id,
                    )

                    result = await self.executor._execute_single_call(call, context)
                    results.append(result)

                    yield FunctionExecutionComplete(
                        function_name=call.name,
                        function_id=call.id,
                        status=result.status.value,
                        result=result.result if result.status == ExecutionStatus.SUCCESS else None,
                        error=result.error,
                        agent_id=self.orchestrator.current_agent_id,
                    )

                switch_context = SwitchContext(
                    function_results=results,
                    execution_error=any(r.status == ExecutionStatus.FAILURE for r in results),
                    buffered_content=buffered_content,
                )

                target_agent = self.orchestrator.should_switch_agent(switch_context.__dict__)
                if target_agent:
                    old_agent = self.orchestrator.current_agent_id
                    self.orchestrator.switch_agent(target_agent, "Post-execution switch")

                    yield AgentSwitch(
                        from_agent=old_agent,
                        to_agent=target_agent,
                        reason="Post-execution switch",
                    )

        yield Completion(
            final_content=buffered_content,
            function_calls_executed=len(function_calls),
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-3:],
        )

    async def _process_streaming_chunks(self, response, callback):
        """Process streaming chunks and yield results"""
        chunks = []

        def wrapper_callback(content, chunk):
            result = callback(content, chunk)
            chunks.append(result)

        await self.llm_client.process_streaming_response(response, wrapper_callback)

        for chunk in chunks:
            yield chunk


__all__ = ("Calute", "PromptSection", "PromptTemplate")
