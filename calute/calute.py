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
import textwrap
import typing as tp
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from calute.types.function_execution_types import ReinvokeSignal
from calute.types.messages import ChatMessage, MessagesHistory, SystemMessage, UserMessage

from .client import GeminiClient, OpenAIClient
from .executors import AgentOrchestrator, FunctionExecutor
from .memory import MemoryStore, MemoryType
from .types import (
    Agent,
    AgentFunction,
    AgentSwitch,
    AgentSwitchTrigger,
    AssistantMessage,
    Completion,
    ExecutionStatus,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    RequestFunctionCall,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
    ToolCall,
    ToolMessage,
)
from .types.tool_calls import FunctionCall
from .utils import debug_print, function_to_json

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
            PromptSection.SYSTEM: "SYSTEM:",
            PromptSection.RULES: "RULES:",
            PromptSection.FUNCTIONS: "FUNCTIONS:",
            PromptSection.TOOLS: f"TOOLS:\n{SEP}When using tools, follow this format:",
            PromptSection.EXAMPLES: f"EXAMPLES:\n{SEP}",
            PromptSection.CONTEXT: "CONTEXT:\n",
            PromptSection.HISTORY: f"HISTORY:\n{SEP}Conversation so far:\n",
            PromptSection.PROMPT: "PROMPT:\n",
        }

        self.section_order = self.section_order or [
            PromptSection.SYSTEM,
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

    SEP: tp.ClassVar = SEP

    def __init__(self, client: tp.Any, template: PromptTemplate | None = None, enable_memory: bool = False):
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

        def capability_based_switch(context, agents, current_agent_id):  # type:ignore
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
        function_calls: list[RequestFunctionCall] | None = None,
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

    def _format_section(
        self,
        header: str,
        content: str | list[str] | None,
        item_prefix: str | None = "- ",
    ) -> str | None:
        """
        Formats a section of the prompt with a header and indented content.
        Returns None if the content is empty.
        """
        if not content:
            return None

        if isinstance(content, list):
            content_str = "\n".join(f"{item_prefix or ''}{str(line).strip()}" for line in content)
        else:
            content_str = str(content).strip()

        if not content_str:
            return None

        indented_content = textwrap.indent(content_str, SEP)

        return f"{header}\n{indented_content}"

    def _extract_from_markdown(self, content: str, field: str) -> list[RequestFunctionCall]:
        """Extract function calls from response content"""

        pattern = rf"```{field}\s*\n(.*?)\n```"
        return re.findall(pattern, content, re.DOTALL)

    def manage_messages(
        self,
        agent: Agent | None,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        include_memory: bool = True,
        use_chain_of_thought: bool = False,
        require_reflection: bool = False,
    ) -> MessagesHistory:
        """
        Generates a structured list of ChatMessage objects for the LLM.
        This version uses a helper function to ensure clean and consistent indentation.
        """
        if not agent:
            return MessagesHistory(messages=[UserMessage(content=prompt or "You are a helpful assistant.")])

        system_parts = []

        persona_header = self.template.sections.get(PromptSection.SYSTEM, "SYSTEM:")
        instructions = str((agent.instructions() if callable(agent.instructions) else agent.instructions) or "")
        if use_chain_of_thought:
            instructions += (
                "\n\nApproach every task systematically:\n"
                "- Understand the request fully.\n"
                "- Break down complex problems.\n"
                "- If functions are available, determine if they are needed.\n"
                "- Formulate your response or function call.\n"
                "- Verify your output addresses the request completely."
            )
        system_parts.append(self._format_section(persona_header, instructions, item_prefix=None))
        rules_header = self.template.sections.get(PromptSection.RULES, "RULES:")
        rules = agent.rules if isinstance(agent.rules, list) else ([agent.rules] if agent.rules else [])
        if agent.functions:
            rules.append(
                "If a function can satisfy the user request, you MUST respond only with a valid tool call in the"
                " specified format. Do not add any conversational text before or after the tool call."
            )
        if self.enable_memory and include_memory:
            rules.extend(
                [
                    "Consider previous context and conversation history.",
                    "Build upon earlier interactions when appropriate.",
                ]
            )
        system_parts.append(self._format_section(rules_header, rules))

        if agent.functions:
            functions_header = self.template.sections.get(PromptSection.FUNCTIONS, "FUNCTIONS:")

            tool_format_instruction = textwrap.dedent(
                """
                When calling a function, you must use the following XML format.
                The tag name is the function name and parameters are a JSON object within <arguments> tags.

                Example:
                    <my_function_name><arguments>{"param1": "value1"}</arguments></my_function_name>

                The available functions are listed with their schemas:
                """
            ).strip()

            fn_docs_raw = self.generate_function_section(agent.functions)
            indented_fn_docs = textwrap.indent(fn_docs_raw, SEP)
            full_function_content = f"{tool_format_instruction}\n\n{indented_fn_docs}"
            system_parts.append(self._format_section(functions_header, full_function_content, item_prefix=None))

        if agent.examples:
            examples_header = self.template.sections.get(PromptSection.EXAMPLES, "EXAMPLES:")
            example_content = "\n\n".join(ex.strip() for ex in agent.examples)
            system_parts.append(self._format_section(examples_header, example_content, item_prefix=None))

        context_header = self.template.sections.get(PromptSection.CONTEXT, "CONTEXT:")
        context_content_list = []
        if self.enable_memory and include_memory:
            memory_context = self.memory_store.consolidate_memories(agent.id)
            if memory_context:
                context_content_list.append(f"Relevant information from memory:\n{memory_context}")
        if context_variables:
            ctx_vars_formatted = self.format_context_variables(context_variables)
            if ctx_vars_formatted:
                context_content_list.append(f"Current variables:\n{ctx_vars_formatted}")

        if context_content_list:
            system_parts.append(
                self._format_section(context_header, "\n\n".join(context_content_list), item_prefix=None)
            )

        instructed_messages: list[ChatMessage] = []

        final_system_content = "\n\n".join(part for part in system_parts if part)
        instructed_messages.append(SystemMessage(content=final_system_content))

        if messages and messages.messages:
            instructed_messages.extend(messages.messages)

        if prompt is not None:
            final_prompt_content = prompt
            if require_reflection:
                final_prompt_content += (
                    f"\n\nAfter your primary response, add a reflection section in `<reflection>` tags:\n"
                    f"{self.SEP}- Assumptions made.\n"
                    f"{self.SEP}- Potential limitations of your response."
                )
            instructed_messages.append(UserMessage(content=final_prompt_content))

        return MessagesHistory(messages=instructed_messages)

    def _build_reinvoke_messages(
        self,
        original_messages: MessagesHistory,
        assistant_content: str,
        function_calls: list[RequestFunctionCall],
        results: list[RequestFunctionCall],
    ) -> MessagesHistory:
        """Build message history for reinvocation including function results"""

        messages = original_messages.messages.copy()

        tool_calls = []
        for fc in function_calls:
            tool_call = ToolCall(
                id=fc.id,
                function=FunctionCall(
                    name=fc.name,
                    arguments=json.dumps(fc.arguments) if isinstance(fc.arguments, dict) else fc.arguments,
                ),
            )
            tool_calls.append(tool_call)

        clean_content = self._remove_function_calls_from_content(assistant_content)
        assistant_msg = AssistantMessage(
            content=clean_content if clean_content.strip() else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        messages.append(assistant_msg)

        for fc, result in zip(function_calls, results, strict=False):
            if result.status == ExecutionStatus.SUCCESS:
                tool_content = json.dumps(result.result) if not isinstance(result.result, str) else result.result
            else:
                tool_content = f"Error: {result.error}"

            tool_msg = ToolMessage(content=tool_content, tool_call_id=fc.id)
            messages.append(tool_msg)

        return MessagesHistory(messages=messages)

    @staticmethod
    def extract_md_block(input_string: str) -> list[tuple[str, str]]:
        """
        Extract Markdown code blocks from a string.

        This function finds all Markdown code blocks (delimited by triple backticks)
        in the input string and returns their content along with the optional language
        specifier (if present).

        Args:
            input_string (str): The input string containing one or more Markdown code blocks.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains:
                - The language specifier (e.g., 'xml', 'python', or '' if not specified).
                - The content of the code block.

        Example:
            >>> text = '''```xml
            ... <web_research>
            ...   <arguments>
            ...     {"query": "quantum computing breakthroughs 2024"}
            ...   </arguments>
            ... </web_research>
            ... ```'''
            >>> extract_md_block(text)
            [('xml', '<web_research>\n  <arguments>\n    {"query": "quantum computing breakthroughs 2024"}\n  </arguments>\n</web_research>')]
        """  # noqa
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [(lang, content.strip()) for lang, content in matches]

    def _remove_function_calls_from_content(self, content: str) -> str:
        """Remove function call XML blocks from content"""
        pattern = r"<(\w+)>\s*<arguments>.*?</arguments>\s*</\w+>"
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
        pattern = r"```tool_call.*?```"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        return cleaned.strip()

    def _extract_function_calls_from_xml(self, content: str) -> list[RequestFunctionCall]:
        """Extract function calls from response content using XML tags"""
        function_calls = []
        pattern = r"<(\w+)>\s*<arguments>(.*?)</arguments>\s*</\w+>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            name = match[0]
            arguments_str = match[1].strip()
            try:
                arguments = json.loads(arguments_str)
                function_call = RequestFunctionCall(
                    name=name,
                    arguments=arguments,
                    id=f"call_{i}_{hash(match)}",
                    timeout=self.orchestrator.get_current_agent().function_timeout,
                    max_retries=self.orchestrator.get_current_agent().max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                continue

        return function_calls

    def _extract_function_calls(self, content: str) -> list[RequestFunctionCall]:
        """Extract function calls from response content"""
        function_calls = self._extract_function_calls_from_xml(content)
        if function_calls:
            return function_calls

        function_calls = []
        matches = self._extract_from_markdown(content=content, field="tool_call")

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                function_call = RequestFunctionCall(
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

    def _detect_function_calls(self, content: str, agent: Agent) -> bool:
        """Detect if content contains valid function calls"""
        if not agent.functions:
            return False
        function_names = [func.__name__ for func in agent.functions]
        for func_name in function_names:
            if f"<{func_name}>" in content or f"<{func_name} " in content:
                if "<arguments>" in content:
                    return True
        if "```tool_call" in content:
            return True

        return False

    def _detect_function_calls_regex(self, content: str, agent: Agent) -> bool:
        """Detect function calls using regex for more precision"""
        if not agent.functions:
            return False
        function_names = [func.__name__ for func in agent.functions]
        for func_name in function_names:
            pattern = rf"<{func_name}(?:\s[^>]*)?>.*?<arguments>"
            if re.search(pattern, content, re.DOTALL):
                return True
        return False

    @staticmethod
    def get_thoughts(response: str, tag: str = "think") -> str | None:
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
        categorized_functions = {}
        uncategorized = []

        for func in functions:
            if hasattr(func, "category"):
                category = func.category  # type:ignore
                if category not in categorized_functions:
                    categorized_functions[category] = []
                categorized_functions[category].append(func)
            else:
                uncategorized.append(func)

        for category, funcs in categorized_functions.items():
            function_docs.append(f"## {category} Functions\n")
            for func in funcs:
                try:
                    schema = function_to_json(func)["function"]
                    doc = self._format_function_doc(schema)
                    function_docs.append(doc)
                except Exception as e:
                    func_name = getattr(func, "__name__", str(func))
                    function_docs.append(f"Warning: Unable to parse function {func_name}: {e!s}")
        if uncategorized:
            if categorized_functions:
                function_docs.append("## Other Functions\n")
            for func in uncategorized:
                try:
                    schema = function_to_json(func)["function"]
                    doc = self._format_function_doc(schema)
                    function_docs.append(doc)
                except Exception as e:
                    func_name = getattr(func, "__name__", str(func))
                    function_docs.append(f"Warning: Unable to parse function {func_name}: {e!s}")

        return "\n\n".join(function_docs)

    def _format_function_doc(self, schema: dict) -> str:
        """
        Format a single function's documentation block.

        Layout produced, e.g.:

            Function: get_temperature
            Purpose: get city name and return temperature in C

            Parameters:
                - city (string, required)
                Description : Name of the city to look up.

            Returns : float   # temperature in Celsius

            Call-pattern:
                <get_temperature>
                <arguments>
                    {"city": "Isfahan"}
                </arguments>
                </get_temperature>
        """
        ind1 = SEP
        ind2 = SEP * 2
        ind3 = SEP * 3

        doc_lines: list[str] = []
        doc_lines.append(f"Function: {schema['name']}")
        if desc := schema.get("description", "").strip():
            doc_lines.append(f"{ind1}Purpose: {desc}")
        params_block = []
        params = schema.get("parameters", {})
        properties: dict = params.get("properties", {})
        required = set(params.get("required", []))

        for pname, pinfo in properties.items():
            if pname == "context_variables":
                continue

            ptype = pinfo.get("type", "any")
            req = "required" if pname in required else "optional"

            params_block.append(f"{ind2}- {pname} ({ptype}, {req})")

            if pdesc := pinfo.get("description", "").strip():
                params_block.append(f"{ind3}Description : {pdesc}")

            if enum_vals := pinfo.get("enum"):
                joined = ", ".join(map(str, enum_vals))
                params_block.append(f"{ind3}Allowed values : {joined}")

        if params_block:
            doc_lines.append(f"\n{ind1}Parameters:")
            doc_lines.extend(params_block)
        if ret := schema.get("returns"):
            doc_lines.append(f"\n{ind1}Returns : {ret}")
        call_example = textwrap.dedent(
            f'<{schema["name"]}><arguments>{{"param": "value"}}</arguments></{schema["name"]}>'.rstrip()
        )
        doc_lines.append(f"\n{ind1}Call-pattern:")
        doc_lines.append(textwrap.indent(call_example, ind2))
        if schema_examples := schema.get("examples"):
            doc_lines.append(f"\n{ind1}Examples:")
            for example in schema_examples:
                json_example = json.dumps(example, indent=2)
                doc_lines.append(textwrap.indent(f"```json\n{json_example}\n```", ind2))

        return "\n".join(doc_lines)

    def format_context_variables(self, variables: dict[str, tp.Any]) -> str:
        """Formats context variables with type information and improved readability"""
        if not variables:
            return ""
        formatted_vars = []
        for key, value in variables.items():
            if not callable(value):
                var_type = type(value).__name__
                formatted_value = str(value)
                formatted_vars.append(f"- {key} ({var_type}): {formatted_value}")
        return "\n".join(formatted_vars)

    def format_prompt(self, prompt: str | None) -> str:
        if not prompt:
            return ""
        return prompt

    def format_chat_history(self, messages: MessagesHistory) -> str:
        """Formats chat messages with improved readability and metadata"""
        formatted_messages = []
        for msg in messages.messages:
            formatted_messages.append(f"## {msg.role}:\n{msg.content}")
        return "\n\n".join(formatted_messages)

    async def create_response(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        stream: bool = True,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = True,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
    ) -> ResponseResult | AsyncIterator[StreamingResponseType]:
        """Create response with enhanced function calling and agent switching"""
        if isinstance(agent_id, Agent):
            agent = agent_id
        else:
            if agent_id:
                self.orchestrator.switch_agent(agent_id, "User specified agent")
            agent = self.orchestrator.get_current_agent()

        context_variables = context_variables or {}
        original_messages = messages.model_copy() if messages else MessagesHistory(messages=[])
        prompt_messages: MessagesHistory = self.manage_messages(
            agent=agent,
            prompt=prompt,
            context_variables=context_variables,
            messages=messages,
        )

        if use_instructed_prompt:
            prompt_str = prompt_messages.make_instruction_prompt(
                conversation_name_holder=conversation_name_holder,
                mention_last_turn=mention_last_turn,
            )
        else:
            prompt_str = prompt_messages.to_openai()["messages"]

        if print_formatted_prompt:
            if use_instructed_prompt:
                print(prompt_str)
            else:
                for msg in prompt_messages.messages:
                    debug_print(f"--- ROLE: {msg.role} ---\n{msg.content}\n---------------------\n")

        response = await self.llm_client.generate_completion(
            prompt=prompt_str,
            model=agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            top_p=agent.top_p,
            stop=agent.stop,
            top_k=agent.top_k,
            min_p=agent.min_p,
            presence_penalty=agent.presence_penalty,
            frequency_penalty=agent.frequency_penalty,
            repetition_penalty=agent.repetition_penalty,
            extra_body=agent.extra_body,
            stream=stream,
        )

        if not apply_functions:
            if stream:
                return self._handle_streaming(response, reinvoked_runtime)
            else:
                return await self._handle_response(response, reinvoked_runtime)

        if stream:
            return self._handle_streaming_with_functions(
                response,
                agent,
                context_variables,
                original_messages,
                reinvoke_after_function,
                reinvoked_runtime,
            )
        else:
            return await self._handle_response_with_functions(
                response,
                agent,
                context_variables,
                original_messages,
                reinvoke_after_function,
                reinvoked_runtime,
            )

    async def _handle_streaming_with_functions(
        self,
        response: tp.Any,
        agent: Agent,
        context: dict,
        original_messages: MessagesHistory,
        reinvoke_after_function: bool,
        reinvoked_runtime: bool,
    ) -> AsyncIterator[StreamingResponseType]:
        """Handle streaming response with function calls and optional reinvocation"""
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

                        if not function_calls_detected:
                            function_calls_detected = self._detect_function_calls(buffered_content, agent)

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=function_calls_detected,
                    reinvoked=reinvoked_runtime,
                )
        elif isinstance(self.llm_client, GeminiClient):
            async for chunk in response:
                content = None
                if hasattr(chunk, "text") and chunk.text:
                    buffered_content += chunk.text
                    content = chunk.text

                    if not function_calls_detected:
                        function_calls_detected = self._detect_function_calls(buffered_content, agent)

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=function_calls_detected,
                    reinvoked=reinvoked_runtime,
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

                if reinvoke_after_function and function_calls:
                    updated_messages = self._build_reinvoke_messages(
                        original_messages,
                        buffered_content,
                        function_calls,
                        results,
                    )
                    yield ReinvokeSignal(
                        message="Reinvoking agent with function results...",
                        agent_id=self.orchestrator.current_agent_id,
                    )

                    reinvoke_response = await self.create_response(
                        prompt=None,
                        context_variables=context,
                        messages=updated_messages,
                        agent_id=agent,
                        stream=True,
                        apply_functions=True,
                        print_formatted_prompt=False,
                        use_instructed_prompt=True,
                        reinvoke_after_function=True,
                        reinvoked_runtime=True,
                    )
                    try:
                        for chunk in reinvoke_response:
                            yield chunk
                    except TypeError:
                        async for chunk in reinvoke_response:
                            yield chunk
                    return

        yield Completion(
            final_content=buffered_content,
            function_calls_executed=len(function_calls),
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-3:],
        )

    async def _handle_response_with_functions(
        self,
        response: tp.Any,
        agent: Agent,
        context: dict,
        original_messages: MessagesHistory,
        reinvoke_after_function: bool,
        reinvoked_runtime: bool,
    ) -> ResponseResult:
        """Handle non-streaming response with function calls and optional reinvocation"""

        content = self.llm_client.extract_content(response)
        function_calls = self._extract_function_calls(content)

        if function_calls:
            results = await self.executor.execute_function_calls(function_calls, agent.function_call_strategy, context)

            switch_context = SwitchContext(
                function_results=results,
                execution_error=any(r.status == ExecutionStatus.FAILURE for r in results),
            )

            target_agent = self.orchestrator.should_switch_agent(switch_context.__dict__)
            if target_agent:
                self.orchestrator.switch_agent(target_agent, "Post-execution switch")

            if reinvoke_after_function:
                updated_messages = self._build_reinvoke_messages(original_messages, content, function_calls, results)
                reinvoke_response = await self.create_response(
                    prompt=None,
                    context_variables=context,
                    messages=updated_messages,
                    agent_id=agent,
                    stream=False,
                    apply_functions=True,
                    print_formatted_prompt=False,
                    use_instructed_prompt=True,
                    reinvoke_after_function=True,
                    reinvoked_runtime=True,
                )

                if isinstance(reinvoke_response, ResponseResult):
                    return ResponseResult(
                        content=reinvoke_response.content,
                        response=reinvoke_response.response,
                        function_calls=function_calls,
                        agent_id=reinvoke_response.agent_id,
                        execution_history=self.orchestrator.execution_history[-5:],
                        reinvoked=True,
                    )

        return ResponseResult(
            content=content,
            response=response,
            function_calls=function_calls if function_calls else [],
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-5:],
            reinvoked=False,
        )

    async def _handle_streaming(self, response: tp.Any, reinvoked_runtime) -> AsyncIterator[StreamingResponseType]:
        """Handle streaming response with function calls and optional reinvocation"""
        buffered_content = ""

        if isinstance(self.llm_client, OpenAIClient):
            for chunk in response:
                content = None
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        buffered_content += delta.content
                        content = delta.content

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=False,
                    reinvoked=reinvoked_runtime,
                )
        elif isinstance(self.llm_client, GeminiClient):
            async for chunk in response:
                content = None
                if hasattr(chunk, "text") and chunk.text:
                    buffered_content += chunk.text
                    content = chunk.text

                yield StreamChunk(
                    chunk=chunk,
                    agent_id=self.orchestrator.current_agent_id,
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=False,
                    reinvoked=reinvoked_runtime,
                )
        yield Completion(
            final_content=buffered_content,
            function_calls_executed=0,
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-3:],
        )

    async def _handle_response(self, response: tp.Any, reinvoked_runtime) -> ResponseResult:
        """Handle non-streaming response"""

        content = self.llm_client.extract_content(response)

        return ResponseResult(
            content=content,
            response=response,
            function_calls=[],
            agent_id=self.orchestrator.current_agent_id,
            execution_history=self.orchestrator.execution_history[-5:],
            reinvoked=reinvoked_runtime,
        )

    async def _process_streaming_chunks(self, response: tp.Any, callback: tp.Any):
        """Process streaming chunks and yield results"""
        chunks = []

        def wrapper_callback(content: tp.Any, chunk: tp.Any):
            result = callback(content, chunk)
            chunks.append(result)

        await self.llm_client.process_streaming_response(response, wrapper_callback)

        for chunk in chunks:
            yield chunk


__all__ = ("Calute", "PromptSection", "PromptTemplate")
