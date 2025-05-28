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

    def __init__(self, client, template: PromptTemplate | None = None):
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

        self.orchestrator.register_switch_trigger(
            AgentSwitchTrigger.CAPABILITY_BASED,
            capability_based_switch,
        )
        self.orchestrator.register_switch_trigger(
            AgentSwitchTrigger.ERROR_RECOVERY,
            error_recovery_switch,
        )

    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        self.orchestrator.register_agent(agent)

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
    ) -> str:
        """
        Generates a structured prompt using the enhanced template with clear rules and examples
        """

        if not agent:
            return prompt or "You are a helpful assistant."
        rules_being_used = agent.rules is not None and len(agent.rules) > 0
        functions_being_used = agent.functions is not None and len(agent.functions) > 0
        examples_being_used = agent.examples is not None and len(agent.examples) > 0
        tools_being_used = agent.tool_choice is not None and len(agent.tool_choice) > 0
        sections = {}

        sections[PromptSection.SYSTEM] = agent.name or self.template.sections[PromptSection.SYSTEM]
        instructions = agent.instructions() if callable(agent.instructions) else agent.instructions
        sections[PromptSection.PERSONA] = f"{self.template.sections[PromptSection.PERSONA]} {instructions}"

        if rules_being_used or functions_being_used:
            if rules_being_used:
                rules = agent.rules if isinstance(agent.rules, list) else [agent.rules]
            else:
                rules = []
            if functions_being_used:
                rules.append("If a function can satisfy the user, respond only with a tool_call JSON and nothing else.")
            rules_string = "\n".join(rules).replace("\n", f"\n{SEP}")
            _str = f"{self.template.sections[PromptSection.RULES]} {rules_string}"
            sections[PromptSection.RULES] = _str

        if functions_being_used:
            fn_docs = self.generate_function_section(agent.functions)
            fn_docs = "\n" + SEP + add_depth(add_depth(fn_docs, ep=True))
            sections[PromptSection.FUNCTIONS] = f"{self.template.sections[PromptSection.FUNCTIONS]} {fn_docs}"

        if tools_being_used:
            sections[PromptSection.TOOLS] = f"{self.template.sections[PromptSection.TOOLS]} {agent.tool_choice}"

        example_text = ""
        if examples_being_used:
            example_text = "\n\n".join(agent.examples)
        if functions_being_used:
            example_text += (
                "When calling a function, you must include the function name and all required inputs "
                "inside a markdown code block labeled `tool_call`.\n\n"
                "Here is an example:\n"
                f'{SEP * 2}User Prompt: "..."\n'
                f"{SEP * 2}Assistant: ```tool_call\n"
                f"{SEP * 3}{{\n"
                f'{SEP * 4}"name": "{agent.functions[0].__name__}",\n'
                f'{SEP * 4}"content": {{...}}\n'
                f"{SEP * 3}}}\n"
                f"{SEP * 2}```"
            )

        if example_text != "":
            sections[PromptSection.EXAMPLES] = (
                f"{self.template.sections[PromptSection.EXAMPLES]}{add_depth(example_text, True)}"
            )

        if context_variables:
            _ctx_layout = add_depth(self.format_context_variables(context_variables), True)
            sections[PromptSection.CONTEXT] = f"{self.template.sections[PromptSection.CONTEXT]} {_ctx_layout}"

        if history:
            sections[PromptSection.HISTORY] = (
                f"{self.template.sections[PromptSection.HISTORY]} {self.format_chat_history(history)}"
            )

        if prompt is not None:
            sections[PromptSection.PROMPT] = (
                f"{self.template.sections[PromptSection.PROMPT]} {self.format_prompt(prompt)}"
            )

        parts = []
        for sec in self.template.section_order:
            if sec in sections:
                parts.append(sections[sec])
        return "\n\n".join(parts).strip()

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
            for chunk in response:
                content = None
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        buffered_content += delta.content
                        content = delta.content

                        if "```tool_call" in buffered_content and not function_calls_detected:
                            function_calls_detected = True

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                )
        elif isinstance(self.llm_client, GeminiClient):
            for chunk in response:
                content = None
                if hasattr(chunk, "text") and chunk.text:
                    buffered_content += chunk.text
                    content = chunk.text

                    if "```tool_call" in buffered_content and not function_calls_detected:
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
