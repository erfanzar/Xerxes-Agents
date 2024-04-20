import os
import warnings

from jinja2 import Environment, BaseLoader, Template
from typing import Optional


bos_eos_token_templates = {
    "chatml": {
        "eos": "<|im_end|>",
        "bos": "<|im_start|>"
    },
    "gemma": {
        "eos": "<end_of_turn>",
        "bos": "<start_of_turn>"
    },
    "llama": {
        "eos": "</s>",
        "bos": "<s>"
    },
    "llama_3": {
        "eos": "<|eot_id|>",
        "bos": "<|begin_of_text|>"
    },
    "open_chat": {
        "eos": "</s>",
        "bos": "<s>"
    },
    "zephyr": {
        "eos": "<|endoftext|>",
        "bos": "<|startoftext|>"
    },
}


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
            bos_token: Optional[str] = None,
            eos_token: Optional[str] = None
    ) -> 'PromptTemplates':
        available_formats = [
            s.replace("prompt_template_", "").replace(".jinja2", "") for s
            in os.listdir(os.path.dirname(__file__)) if
            os.path.exists(os.path.join(os.path.dirname(__file__), s))
        ]
        assert prompt_template in available_formats, (
            f"couldn't find {prompt_template} in available templates {available_formats}"
        )

        if bos_token is None and eos_token is None:
            if prompt_template not in list(bos_eos_token_templates.keys()):
                raise ValueError(
                    "No bos_eos_token_templates available for given prompt_template, "
                    "you should provide bos and eos tokens."
                )
            temp = bos_eos_token_templates[prompt_template]
            bos_token = temp["bos"]
            eos_token = temp["eos"]
            warnings.warn(
                f"Since no eos and bos token is provided the eos and tokens for {prompt_template} will be set as "
                f"{eos_token=}, {bos_token=}"
            )
        elif bos_token is None and eos_token is not None:
            raise ValueError("if you are passing eos token you should provide bos token too.")
        elif bos_token is not None and eos_token is None:
            raise ValueError("if you are passing bos token you should provide eos token too.")

        template_string = open(f"{os.path.dirname(__file__)}/prompt_template_{prompt_template}.jinja2", "r").read()
        template = Environment(loader=BaseLoader()).from_string(
            template_string,
        )

        return cls(
            template,
            bos_token=bos_token,
            eos_token=eos_token
        )
