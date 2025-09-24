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

"""Agent definition for Cortex framework"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from calute.llms.base import BaseLLM

from ..loggings import get_logger, log_delegation, log_error, log_thinking, log_warning
from ..types import Agent as CaluteAgent
from ..types import AgentCapability
from .templates import PromptTemplate
from .tool import CortexTool

if TYPE_CHECKING:
    from ..calute import Calute
    from .cortex import Cortex
    from .memory_integration import CortexMemory


@dataclass
class CortexAgent:
    """Agent with specific role, goal, and capabilities"""

    role: str
    goal: str
    backstory: str
    model: str | None = None
    instructions: str | None = None
    tools: list[CortexTool] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = True
    allow_delegation: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    memory_enabled: bool = True
    capabilities: list[AgentCapability] = field(default_factory=list)
    calute_instance: "Calute | None" = None
    memory: Optional["CortexMemory"] = None
    llm: BaseLLM | None = None
    reinvoke_after_function: bool = True
    cortex_instance: Optional["Cortex"] = None

    max_execution_time: int | None = None
    max_rpm: int | None = None
    step_callback: Callable | None = None
    config: dict = field(default_factory=dict)
    knowledge: dict = field(default_factory=dict)
    knowledge_sources: list = field(default_factory=list)

    auto_format_guidance: bool = True
    output_format_preference: str = "xml"

    _internal_agent: CaluteAgent | None = None
    _logger = None
    _template_engine: PromptTemplate | None = None
    _delegation_count: int = 0
    _times_executed: int = 0
    _execution_times: list = field(default_factory=list)
    _last_rpm_window: float = 0
    _rpm_requests: list = field(default_factory=list)

    def __post_init__(self):
        """Initialize the internal Calute agent"""
        self._logger = get_logger() if self.verbose else None
        self._template_engine = PromptTemplate()

        if not self.instructions:
            self.instructions = self._template_engine.render_agent_prompt(
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                tools=self.tools if self.tools else None,
            )

        functions = []
        for tool in self.tools:
            if callable(tool) and not isinstance(tool, CortexTool):
                functions.append(tool)
            elif callable(tool.function):
                functions.append(tool.function)
        if self.calute_instance is None and self.llm is not None:
            from calute.calute import Calute

            self.calute_instance = Calute(llm=self.llm)
        self._internal_agent = CaluteAgent(
            name=self.role,
            instructions=self.instructions,
            model=self.model,
            functions=functions,
            capabilities=self.capabilities if self.capabilities else [],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            parallel_tool_calls=True,
        )

    def _check_rate_limit(self) -> bool:
        """Check if agent is within rate limit"""
        if not self.max_rpm:
            return True

        import time

        current_time = time.time()

        self._rpm_requests = [req_time for req_time in self._rpm_requests if current_time - req_time < 60]

        if len(self._rpm_requests) >= self.max_rpm:
            if self.verbose:
                log_warning(f"Rate limit reached ({self.max_rpm} RPM)")
            return False

        return True

    def _record_request(self):
        """Record a request for rate limiting"""
        if self.max_rpm:
            import time

            self._rpm_requests.append(time.time())

    def _check_execution_timeout(self, start_time: float) -> bool:
        """Check if execution has exceeded timeout"""
        if not self.max_execution_time:
            return False

        import time

        elapsed = time.time() - start_time
        if elapsed > self.max_execution_time:
            if self.verbose:
                log_error(f"Execution timeout after {elapsed:.2f}s")
            return True
        return False

    def _execute_step_callback(self, step_info: dict):
        """Execute step callback if provided"""
        if self.step_callback and callable(self.step_callback):
            try:
                self.step_callback(step_info)
            except Exception as e:
                if self.verbose:
                    log_error(f"Step callback failed: {e}")

    def _build_knowledge_context(self, task_description: str) -> str:
        """Build context from knowledge sources"""
        if not self.knowledge and not self.knowledge_sources:
            return ""

        context_parts = []

        if self.knowledge:
            context_parts.append("Available Knowledge:")
            for key, value in self.knowledge.items():
                context_parts.append(f"- {key}: {value}")

        if self.knowledge_sources:
            context_parts.append("Knowledge Sources:")
            for source in self.knowledge_sources:
                context_parts.append(f"- {source}")

        return "\n".join(context_parts) + "\n\n" if context_parts else ""

    def _generate_format_guidance(self, output_model) -> str:
        """Generate format guidance from Pydantic model"""
        if not output_model or not self.auto_format_guidance:
            return ""

        try:
            from pydantic import BaseModel

            if not issubclass(output_model, BaseModel):
                return ""

            if hasattr(output_model, "model_json_schema"):
                schema = output_model.model_json_schema()
            else:
                schema = output_model.schema()
            model_name = schema.get("title", "Output")

            example_data = self._create_example_from_schema(schema)

            if self.output_format_preference == "xml":
                nested_arrays = self._count_nested_structures(schema)

                format_instruction = f"""

OUTPUT FORMAT REQUIREMENT:
Please provide your response in the following XML format to ensure proper parsing:

<json>
{self._format_json_example(example_data)}
</json>

CRITICAL FORMATTING RULES:
1. Follow the EXACT structure shown above - do not simplify or modify it
2. {"Arrays of objects must contain FULL objects with all required fields - do not use simple strings" if nested_arrays > 0 else ""}
3. Each nested object must have ALL the fields shown in the example
4. Use realistic values but maintain the exact JSON structure
5. Numbers must be within the specified ranges (check min/max constraints)

This format is required for validation against the {model_name} schema.
FAILURE TO FOLLOW THE EXACT STRUCTURE WILL RESULT IN VALIDATION ERRORS.
"""
            else:
                format_instruction = f"""

OUTPUT FORMAT REQUIREMENT:
Please provide your response as valid JSON matching this schema for {model_name}:

{self._format_json_example(example_data)}

Ensure your response is valid JSON that can be parsed directly.
"""

            return format_instruction

        except Exception:
            return ""

    def _resolve_schema_refs(self, schema: dict, definitions: dict | None = None) -> dict:
        """Resolve $ref references in schema (supports both Pydantic v1 and v2)"""
        if definitions is None:
            definitions = schema.get("definitions", schema.get("$defs", {}))

        if "$ref" in schema:
            ref_path = schema["$ref"]

            if ref_path.startswith("#/definitions/") or ref_path.startswith("#/$defs/"):
                ref_name = ref_path.split("/")[-1]
                if ref_name in definitions:
                    return self._resolve_schema_refs(definitions[ref_name], definitions)

        resolved = schema.copy()
        if "properties" in resolved:
            resolved["properties"] = {
                k: self._resolve_schema_refs(v, definitions) for k, v in resolved["properties"].items()
            }

        if "items" in resolved:
            resolved["items"] = self._resolve_schema_refs(resolved["items"], definitions)

        return resolved

    def _create_example_from_schema(self, schema: dict) -> dict:
        """Create realistic example data structure from JSON schema with full resolution"""

        resolved_schema = self._resolve_schema_refs(schema)

        properties = resolved_schema.get("properties", {})
        definitions = schema.get("definitions", {})
        example = {}

        for field_name, field_info in properties.items():
            field_type = field_info.get("type")
            field_description = field_info.get("description", "")

            if field_type == "string":
                if "name" in field_name.lower() or "title" in field_name.lower():
                    example[field_name] = f"Example {field_name.replace('_', ' ').title()}"
                else:
                    example[field_name] = f"Example {field_description or field_name.replace('_', ' ')}"
            elif field_type == "integer":
                min_val = field_info.get("minimum", 1)
                max_val = field_info.get("maximum", 10)
                example[field_name] = min(max_val, max(min_val, 5))
            elif field_type == "number":
                min_val = field_info.get("minimum", 1.0)
                max_val = field_info.get("maximum", 10.0)
                example[field_name] = min(max_val, max(min_val, 7.5))
            elif field_type == "boolean":
                example[field_name] = True
            elif field_type == "array":
                items = field_info.get("items", {})
                min_items = field_info.get("minItems", 2)

                if items.get("type") == "string":
                    if "industries" in field_name.lower():
                        example[field_name] = ["technology", "healthcare", "finance"][:min_items]
                    elif "tags" in field_name.lower():
                        example[field_name] = ["tag1", "tag2", "tag3"][:min_items]
                    else:
                        example[field_name] = [f"item{i + 1}" for i in range(min_items)]

                elif items.get("type") == "object" or "$ref" in items:
                    resolved_items = self._resolve_schema_refs(items, definitions)

                    if "properties" in resolved_items:
                        nested_examples = []
                        for i in range(max(min_items, 3)):
                            nested_example = self._create_nested_example(resolved_items, i + 1)
                            nested_examples.append(nested_example)
                        example[field_name] = nested_examples
                    else:
                        nested_example = self._create_example_from_schema(resolved_items)
                        example[field_name] = [nested_example] * max(min_items, 3)
                else:
                    example[field_name] = [f"item{i + 1}" for i in range(min_items)]

            elif field_type == "object":
                if "properties" in field_info:
                    example[field_name] = self._create_example_from_schema(field_info)
                else:
                    example[field_name] = {}
            else:
                example[field_name] = f"Example {field_description or field_name.replace('_', ' ')}"

        return example

    def _create_nested_example(self, schema: dict, index: int = 1) -> dict:
        """Create a realistic nested object example with variation"""
        properties = schema.get("properties", {})

        example = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type")
            field_description = field_info.get("description", "")

            if field_type == "string":
                if "name" in field_name.lower() or "title" in field_name.lower():
                    trend_names = [
                        "Generative AI Enterprise Integration",
                        "Edge AI and IoT Convergence",
                        "AI-Powered Cybersecurity",
                    ]
                    example[field_name] = trend_names[(index - 1) % len(trend_names)]
                elif "description" in field_name.lower():
                    descriptions = [
                        "Large enterprises are integrating generative AI into core business processes",
                        "AI processing moving closer to data sources for real-time decision making",
                        "Advanced threat detection and response using machine learning algorithms",
                    ]
                    example[field_name] = descriptions[(index - 1) % len(descriptions)]
                else:
                    example[field_name] = f"Example {field_description or field_name} {index}"
            elif field_type == "number":
                min_val = field_info.get("minimum", 1.0)
                max_val = field_info.get("maximum", 10.0)
                base_val = (min_val + max_val) / 2
                example[field_name] = round(base_val + (index - 1) * 0.5, 1)
            elif field_type == "integer":
                min_val = field_info.get("minimum", 1)
                max_val = field_info.get("maximum", 10)
                example[field_name] = min(max_val, max(min_val, index + 4))
            elif field_type == "boolean":
                example[field_name] = index % 2 == 1
            elif field_type == "array" and field_info.get("items", {}).get("type") == "string":
                if "industries" in field_name.lower():
                    industry_groups = [
                        ["technology", "finance", "retail"],
                        ["manufacturing", "automotive", "healthcare"],
                        ["cybersecurity", "finance", "government"],
                    ]
                    example[field_name] = industry_groups[(index - 1) % len(industry_groups)]
                else:
                    example[field_name] = [f"item{index}a", f"item{index}b"]
            else:
                example[field_name] = f"value{index}"

        return example

    def _count_nested_structures(self, schema: dict) -> int:
        """Count nested arrays of objects to provide better guidance"""
        count = 0
        properties = schema.get("properties", {})

        for field_info in properties.values():
            if field_info.get("type") == "array":
                items = field_info.get("items", {})
                if items.get("type") == "object" or "$ref" in items:
                    count += 1

        return count

    def _format_json_example(self, data: dict, indent: int = 0) -> str:
        """Format JSON data as a readable example"""
        import json

        try:
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

    def execute(self, task_description: str, context: str | None = None) -> str:
        """Execute a task using the agent with advanced controls"""
        if not self.calute_instance:
            raise ValueError(f"Agent {self.role} not connected to Cortex")

        if not self._check_rate_limit():
            import time

            sleep_time = 60 / self.max_rpm if self.max_rpm else 1
            if self.verbose:
                if self._logger:
                    self._logger.info(f"Sleeping {sleep_time:.1f}s for rate limit")
            time.sleep(sleep_time)

        self._record_request()

        import time

        start_time = time.time()
        self._times_executed += 1

        self._execute_step_callback(
            {
                "step": "execution_start",
                "agent": self.role,
                "task": task_description,
                "execution_count": self._times_executed,
            }
        )

        try:
            knowledge_context = self._build_knowledge_context(task_description)

            memory_context = ""
            if self.memory_enabled and self.memory:
                memory_context = self.memory.build_context_for_task(
                    task_description=task_description,
                    agent_role=self.role,
                    additional_context=context,
                    max_items=10,
                )

            full_context = ""
            contexts = [knowledge_context, memory_context, context]
            contexts = [ctx for ctx in contexts if ctx]
            if contexts:
                full_context = "\n\n".join(contexts)

            prompt = self._template_engine.render_task_prompt(
                description=task_description,
                expected_output="",
                context=full_context,
            )

            if self.verbose:
                log_thinking(self.role)

            response = None
            iteration = 0
            while iteration < self.max_iterations:
                if self._check_execution_timeout(start_time):
                    raise TimeoutError(f"Agent execution timed out after {self.max_execution_time}s")

                try:
                    response_generator = self.calute_instance.run(
                        prompt=prompt,
                        agent_id=self._internal_agent,
                        stream=True,  # Always use stream=True for reliable function execution
                        apply_functions=True,
                        reinvoke_after_function=self.reinvoke_after_function,
                    )

                    # Collect the streaming response
                    response_content = []
                    for chunk in response_generator:
                        if hasattr(chunk, "content") and chunk.content is not None:
                            response_content.append(chunk.content)

                    # Create a response object with the full content
                    class StreamedResponse:
                        def __init__(self, content):
                            self.content = content

                    response = StreamedResponse("".join(response_content))
                    break

                except Exception as e:
                    iteration += 1
                    if iteration >= self.max_iterations:
                        raise e

                    self._execute_step_callback(
                        {"step": "retry", "agent": self.role, "iteration": iteration, "error": str(e)}
                    )

            if not response:
                raise RuntimeError("Failed to get response after maximum iterations")

            if hasattr(response, "content"):
                result = response.content
            elif hasattr(response, "completion"):
                result = response.completion.content
            else:
                result = str(response)

            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)

            self._execute_step_callback(
                {
                    "step": "execution_complete",
                    "agent": self.role,
                    "execution_time": execution_time,
                    "result_length": len(result),
                }
            )

            if self.memory_enabled and self.memory:
                self.memory.save_agent_interaction(
                    agent_role=self.role,
                    action="execute_task",
                    content=f"Task: {task_description[:512]} - Result: {result}",
                    importance=0.5,
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)

            self._execute_step_callback(
                {"step": "execution_error", "agent": self.role, "execution_time": execution_time, "error": str(e)}
            )

            if self.verbose:
                log_error(f"Agent {self.role}: {e!s}")
            raise

    def _check_delegation_needed(self, task_description: str, initial_response: str) -> tuple[bool, str]:
        """Check if delegation is needed based on task and initial response"""
        if not self.allow_delegation or not self.cortex_instance or self._delegation_count >= 3:
            return False, ""

        delegation_indicators = [
            "i need help with",
            "i'm not sure",
            "i cannot",
            "beyond my expertise",
            "would require",
            "need assistance",
            "delegate",
            "ask another agent",
        ]

        response_lower = initial_response.lower()
        for indicator in delegation_indicators:
            if indicator in response_lower:
                return True, "Agent indicated need for assistance"

        if len(task_description.split()) > 50:
            return True, "Task complexity suggests delegation might help"

        return False, ""

    def _select_delegate_agent(self, task_description: str, reason: str) -> Optional["CortexAgent"]:
        """Select the best agent to delegate to"""
        if not self.cortex_instance:
            return None

        available_agents = [agent for agent in self.cortex_instance.agents if agent.role != self.role]

        if not available_agents:
            return None

        selection_prompt = f"""
        Current agent: {self.role}
        Task requiring delegation: {task_description}
        Reason for delegation: {reason}

        Available agents to delegate to:
        {chr(10).join([f"- {agent.role}: {agent.goal}" for agent in available_agents])}

        Which agent would be best suited for this task?
        Respond with ONLY the role name of the selected agent, nothing else.
        """

        try:
            response_generator = self.calute_instance.run(
                prompt=selection_prompt,
                agent_id=self._internal_agent,
                stream=True,  # Always use stream=True for reliable execution
                apply_functions=False,
                reinvoke_after_function=False,
            )

            # Collect the streaming response
            response_content = []
            for chunk in response_generator:
                if hasattr(chunk, "content") and chunk.content is not None:
                    response_content.append(chunk.content)

            class StreamedResponse:
                def __init__(self, content):
                    self.content = content

            response = StreamedResponse("".join(response_content))

            selected_role = response.content.strip() if hasattr(response, "content") else str(response).strip()

            for agent in available_agents:
                if agent.role.lower() == selected_role.lower() or selected_role.lower() in agent.role.lower():
                    return agent
        except Exception as e:
            if self.verbose:
                log_error(f"Agent {self.role} - Failed to select delegate: {e}")

        return available_agents[0] if available_agents else None

    def delegate_task(self, task_description: str, context: str | None = None) -> str:
        """Delegate a task to another agent"""
        if not self.allow_delegation or not self.cortex_instance:
            return "Delegation not available"

        self._delegation_count += 1

        delegate = self._select_delegate_agent(task_description, "Delegation requested")

        if not delegate:
            self._delegation_count -= 1
            return "No suitable agent found for delegation"

        if self.verbose:
            log_delegation(self.role, delegate.role)

        delegation_context = f"""
        This task has been delegated from {self.role}.
        Original context: {context or "No additional context"}
        Please complete this task to the best of your ability.
        """

        delegate._delegation_count = self._delegation_count
        result = delegate.execute(task_description, delegation_context)

        if self.verbose:
            if self._logger:
                self._logger.info(f"✅ Delegation from {self.role} to {delegate.role} complete")

        if self.memory_enabled and self.memory:
            self.memory.save_agent_interaction(
                agent_role=self.role,
                action="delegated_task",
                content=f"Delegated to {delegate.role}: {task_description[:100]}",
                importance=0.6,
            )

        self._delegation_count -= 1
        return result

    def execute_with_delegation(self, task_description: str, context: str | None = None) -> str:
        """Execute task with automatic delegation if needed and allowed"""

        initial_result = self.execute(task_description, context)

        needs_delegation, reason = self._check_delegation_needed(task_description, initial_result)

        if needs_delegation and self.allow_delegation and self.cortex_instance:
            if self.verbose:
                if self._logger:
                    self._logger.info(f"Agent {self.role} considering delegation: {reason}")

            delegated_result = self.delegate_task(task_description, context)

            final_prompt = f"""
            You initially responded with: {initial_result}

            After delegating to another agent, they provided: {delegated_result}

            Please provide a final, comprehensive response combining both insights.
            """

            try:
                response_generator = self.calute_instance.run(
                    prompt=final_prompt,
                    agent_id=self._internal_agent,
                    stream=True,  # Always use stream=True for reliable execution
                    apply_functions=False,
                    reinvoke_after_function=False,
                )

                # Collect the streaming response
                response_content = []
                for chunk in response_generator:
                    if hasattr(chunk, "content") and chunk.content is not None:
                        response_content.append(chunk.content)

                class StreamedResponse:
                    def __init__(self, content):
                        self.content = content

                final_response = StreamedResponse("".join(response_content))

                result = final_response.content if hasattr(final_response, "content") else str(final_response)
                return result
            except Exception:
                return delegated_result

        return initial_result

    def get_execution_stats(self) -> dict:
        """Get execution statistics"""
        if not self._execution_times:
            return {
                "times_executed": self._times_executed,
                "avg_execution_time": 0,
                "total_execution_time": 0,
                "min_execution_time": 0,
                "max_execution_time": 0,
            }

        return {
            "times_executed": self._times_executed,
            "avg_execution_time": sum(self._execution_times) / len(self._execution_times),
            "total_execution_time": sum(self._execution_times),
            "min_execution_time": min(self._execution_times),
            "max_execution_time": max(self._execution_times),
            "recent_execution_times": self._execution_times[-5:],
        }

    def reset_stats(self):
        """Reset execution statistics"""
        self._times_executed = 0
        self._execution_times.clear()
        self._rpm_requests.clear()
        self._delegation_count = 0

    def add_knowledge(self, key: str, value: str):
        """Add knowledge to the agent's knowledge base"""
        self.knowledge[key] = value

    def add_knowledge_source(self, source: str):
        """Add a knowledge source"""
        if source not in self.knowledge_sources:
            self.knowledge_sources.append(source)

    def update_config(self, key: str, value):
        """Update agent configuration"""
        self.config[key] = value

    def get_config(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set_step_callback(self, callback: Callable):
        """Set step callback function"""
        self.step_callback = callback

    def is_rate_limited(self) -> bool:
        """Check if agent is currently rate limited"""
        return not self._check_rate_limit()

    def get_rate_limit_status(self) -> dict:
        """Get current rate limiting status"""
        if not self.max_rpm:
            return {"rate_limited": False, "max_rpm": None, "current_requests": 0}

        import time

        current_time = time.time()
        recent_requests = [req for req in self._rpm_requests if current_time - req < 60]

        return {
            "rate_limited": len(recent_requests) >= self.max_rpm,
            "max_rpm": self.max_rpm,
            "current_requests": len(recent_requests),
            "requests_remaining": max(0, self.max_rpm - len(recent_requests)),
        }
