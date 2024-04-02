from typing import Optional, Generator, Any
from ...prompt_templates import PromptTemplates
from jinja2 import Environment, BaseLoader
import os

PROMPT = open(f"{os.path.dirname(__file__)}/prompt.jinja2", "r").read().strip()


class ChatAgent:
    def __init__(
            self,
            engine: "ServeEngine",
            prompter: PromptTemplates
    ):
        self.engine = engine
        self.prompter = prompter

    @staticmethod
    def render(
            conversation: list[str], full_context: Optional[str] = None
    ) -> str:
        template = Environment(loader=BaseLoader()).from_string(PROMPT)
        return template.render(
            conversation=conversation,
            full_context=full_context or ""
        )

    def execute(
            self,
            conversation: list[str],
            full_context: Optional[str] = None,
            *args,
            **kwargs
    ) -> str:
        prompt = self.prompter.render(
            [
                {
                    "role": "user",
                    "content": self.render(conversation, full_context)
                }
            ],
        )
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
