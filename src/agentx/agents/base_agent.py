from ..prompt_templates import PromptTemplates
import transformers
from abc import abstractmethod


class BaseAgent:
    def __init__(
            self,
            engine: "ServeEngine",  # type:ignore
            prompter: PromptTemplates,
            tokenizer: transformers.PreTrainedTokenizer
    ):
        self.engine = engine
        self.prompter = prompter
        self.tokenizer = tokenizer

    @abstractmethod
    def render(self, **kwargs) -> str:
        ...

    def format_prompt(
            self, **kwargs
    ):
        sample = [{"role": "user", "content": self.render(**kwargs)}]
        if self.prompter is None:
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = self.prompter.render(sample)
        return prompt
