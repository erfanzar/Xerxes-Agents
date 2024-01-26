from typing import List, Optional

from .base import BaseInteract


class OpenChatInteract(BaseInteract):
    def __init__(
            self,
            user_name: str,
            assistant_name: str,
            mode: str = "GPT4 Correct"
    ):
        user_prefix = f"{mode} User:"
        assistant_prefix = f"{mode} Assistant: "
        super().__init__(
            user_name=user_name,
            user_message_token=user_prefix,
            assistant_name=assistant_name,
            assistant_message_token=assistant_prefix,
            prompter_type="openchat",
            end_of_turn_token="<end_of_turn>",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = ""
        for user, assistant in history:
            prompt += f"{self.user_message_token}{user} "
            prompt += f"{self.assistant_message_token}{assistant} "

        return prompt

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:

        dialogs = prefix if prefix is not None else ""

        for user, assistant in history:
            dialogs += f"{self.user_message_token}{user}"
            dialogs += f"{self.assistant_message_token}{assistant}"

        dialogs += f"{self.user_message_token}{prompt}"
        dialogs += self.assistant_message_token
        return dialogs
