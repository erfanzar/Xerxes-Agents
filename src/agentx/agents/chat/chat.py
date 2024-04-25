from typing import Optional, Generator, Any
from ..base_agent import BaseAgent
from jinja2 import Environment, BaseLoader
import os

PROMPT = open(f"{os.path.dirname(__file__)}/prompt.jinja2", "r").read().strip()


class ChatAgent(BaseAgent):

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
        prompt = self.format_prompt(
            conversation=conversation,
            full_context=full_context
        )
        return self.engine.execute(prompt, **kwargs)

    def stream(
            self,
            conversation: list[str],
            full_context: Optional[str] = None,
            *args,
            **kwargs
    ) -> Generator[Any, Any, Any]:
        prompt = self.format_prompt(
            conversation=conversation,
            full_context=full_context
        )

        for response in self.engine.process(
                prompt, **kwargs
        ):
            yield response
