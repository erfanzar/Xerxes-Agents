import os

from jinja2 import Environment, BaseLoader, Template
from typing import Optional


class PromptTemplates(object):
    def __init__(
            self,
            template: Template,
            bos_token: str,
            eos_token: str
    ):
        self.template = template
        self.bos_token = bos_token
        self.eos_token = eos_token

    def get_template(self) -> Template:
        return self.template

    def render(
            self,
            messages: list[dict],
            eos_token: Optional[str] = None,
            bos_token: Optional[str] = None,
            add_generation_prompt: bool = True
    ):
        return self.template.render(
            messages=messages,
            bos_token=bos_token or self.bos_token,
            eos_token=eos_token or self.eos_token,
            add_generation_prompt=add_generation_prompt
        )

    @classmethod
    def from_prompt_templates(
            cls,
            prompt_template: str,
            bos_token: str,
            eos_token: str
    ) -> 'PromptTemplates':
        available_formats = [
            s.replace("prompt_template_", "").replace(".jinja2", "") for s
            in os.listdir(os.path.dirname(__file__)) if
            os.path.exists(os.path.join(os.path.dirname(__file__), s))
        ]

        assert prompt_template in available_formats, (
            f"couldn't find {prompt_template} in available templates {available_formats}"
        )
        template_string = open(f"{os.path.dirname(__file__)}/prompt_template_{prompt_template}.jinja2", "r").read()
        template = Environment(loader=BaseLoader()).from_string(
            template_string
        )

        return cls(
            template,
            bos_token=bos_token,
            eos_token=eos_token
        )
