from typing import Optional, Generator, Any
from jinja2 import Environment, BaseLoader
import os
import json
from ..base_agent import BaseAgent

PROMPT = open(f"{os.path.dirname(__file__)}/prompt.jinja2", "r").read().strip()


class ActionAgent(BaseAgent):
    @staticmethod
    def render(
            conversation: list[str], full_context: Optional[str] = None
    ) -> str:
        template = Environment(loader=BaseLoader()).from_string(PROMPT)
        return template.render(
            conversation=conversation,
            full_context=full_context or ""
        )

    def format_prompt(
            self,
            conversation: list[str],
            full_context: Optional[str] = None,
    ):
        sample = [{"role": "user", "content": self.render(conversation, full_context)}]
        if self.prompter is None:
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = self.prompter.render(sample)
        return prompt

    def execute(
            self,
            conversation: list[str],
            full_context: Optional[str] = None,
            *args,
            **kwargs
    ) -> str:
        prompt = self.format_prompt(conversation=conversation, full_context=full_context)
        return self.engine.execute(prompt, **kwargs)

    def stream(
            self,
            conversation: list[str],
            full_context: Optional[str] = None,
            *args,
            **kwargs
    ) -> Generator[Any, Any, Any]:
        prompt = self.prompter.render(
            [
                {
                    "role": "user",
                    "content": self.render(conversation, full_context)
                }
            ],
        )
        for response in self.engine.process(
                prompt, **kwargs
        ):
            yield response

    @staticmethod
    def validate_response(response: str) -> bool | tuple[str, str]:
        response = response.strip().replace("```json", "```")

        if response.startswith("`json") and response.endswith("```"):
            response = response[4:-3].strip()

        if response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()

        try:
            response = json.loads(response)
        except json.JSONDecodeError as _:
            return False

        if "response" not in response and "action" not in response:
            return False
        else:
            return response["response"], response["action"]
