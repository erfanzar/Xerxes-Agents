from typing import List, Optional

from ._base import BaseInteract


class CargoInteract(BaseInteract):
    def __init__(
            self,
            user_name: str,
            assistant_name: str,
    ):
        user_prefix = f"\n{user_name}=> "
        assistant_prefix = f"\n{assistant_name}=> "
        super().__init__(
            user_name=user_name,
            user_message_token=user_name,
            assistant_name=assistant_name,
            assistant_message_token=assistant_name,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            prompter_type="cargo",
            end_of_turn_token="<end_of_turn>",
        )

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ):
        prompt = system_message + "\n\n"
        for user, assistant in history:
            prompt += f"{self.user_prefix}{user}"
            prompt += f"{self.assistant_prefix}{assistant}"

        return prompt

    def get_stop_signs(self) -> List[str]:
        return [self.user_prefix]

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: str
    ) -> str:
        dialogs = prefix

        for user, assistant in history:
            dialogs += f"{self.user_prefix}{user}"
            dialogs += f"{self.assistant_prefix}{assistant}"

        dialogs += f"{self.user_prefix}{prompt}"
        dialogs += self.assistant_prefix
        return dialogs
