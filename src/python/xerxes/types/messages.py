# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Messages module for Xerxes.

Exports:
    - ChunkTypes
    - BaseContentChunk
    - ImageChunk
    - ImageURL
    - ImageURLChunk
    - TextChunk
    - ContentChunk
    - Roles
    - BaseMessage
    - UserMessage
    - ... and 10 more."""

import re
import textwrap
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import ConfigDict, Field

from ..core.multimodal import SerializableImage
from ..core.utils import XerxesBase
from .tool_calls import ToolCall


class ChunkTypes(StrEnum):
    """Chunk types.

    Inherits from: StrEnum
    """

    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(XerxesBase):
    """Base content chunk.

    Inherits from: XerxesBase

    Attributes:
        type (Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]): type."""

    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | dict[str, str]]: OUT: Result of the operation."""

        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "BaseContentChunk":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_chunk (dict[str, str | dict[str, str]]): IN: openai chunk. OUT: Consumed during execution.
        Returns:
            'BaseContentChunk': OUT: Result of the operation."""

        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    """Image chunk.

    Inherits from: BaseContentChunk

    Attributes:
        type (Literal[ChunkTypes.image]): type.
        image (SerializableImage): image.
        model_config (Any): model config."""

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | dict[str, str]]: OUT: Result of the operation."""

        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageChunk":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_chunk (dict[str, str | dict[str, str]]): IN: openai chunk. OUT: Consumed during execution.
        Returns:
            'ImageChunk': OUT: Result of the operation."""

        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(XerxesBase):
    """Image url.

    Inherits from: XerxesBase

    Attributes:
        url (str): url.
        detail (str | None): detail."""

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    """Image urlchunk.

    Inherits from: BaseContentChunk

    Attributes:
        type (Literal[ChunkTypes.image_url]): type.
        image_url (ImageURL | str): image url.
        model_config (Any): model config."""

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        """Retrieve the url.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | dict[str, str]]: OUT: Result of the operation."""

        image_url_dict = {"url": self.get_url()}
        if isinstance(self.image_url, ImageURL) and self.image_url.detail is not None:
            image_url_dict["detail"] = self.image_url.detail

        out_dict: dict[str, str | dict[str, str]] = {
            "type": "image_url",
            "image_url": image_url_dict,
        }
        return out_dict

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageURLChunk":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_chunk (dict[str, str | dict[str, str]]): IN: openai chunk. OUT: Consumed during execution.
        Returns:
            'ImageURLChunk': OUT: Result of the operation."""

        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class TextChunk(BaseContentChunk):
    """Text chunk.

    Inherits from: BaseContentChunk

    Attributes:
        type (Literal[ChunkTypes.text]): type.
        text (str): text."""

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | dict[str, str]]: OUT: Result of the operation."""

        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | dict[str, str]]) -> "TextChunk":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            messages (dict[str, str | dict[str, str]]): IN: messages. OUT: Consumed during execution.
        Returns:
            'TextChunk': OUT: Result of the operation."""

        return cls.model_validate(messages)


ContentChunk = Annotated[TextChunk | ImageChunk | ImageURLChunk, Field(discriminator="type")]


def _convert_openai_content_chunks(openai_content_chunks: dict[str, str | dict[str, str]]) -> ContentChunk:
    """Internal helper to convert openai content chunks.

    Args:
        openai_content_chunks (dict[str, str | dict[str, str]]): IN: openai content chunks. OUT: Consumed during execution.
    Returns:
        ContentChunk: OUT: Result of the operation."""

    content_type_str = openai_content_chunks.get("type")

    if content_type_str is None:
        raise ValueError("Content chunk must have a type field.")

    if not isinstance(content_type_str, str):
        raise ValueError("Content chunk type must be a string.")

    content_type = ChunkTypes(content_type_str)

    if content_type == ChunkTypes.text:
        return TextChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image_url:
        return ImageURLChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image:
        return ImageChunk.from_openai(openai_content_chunks)
    else:
        raise ValueError(f"Unknown content chunk type: {content_type}")


class Roles(StrEnum):
    """Roles.

    Inherits from: StrEnum
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(XerxesBase):
    """Base message.

    Inherits from: XerxesBase

    Attributes:
        role (Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]): role."""

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | list[dict[str, str | dict[str, Any]]]]: OUT: Result of the operation."""

        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "BaseMessage":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_message (dict[str, str | list[dict[str, str | dict[str, Any]]]]): IN: openai message. OUT: Consumed during execution.
        Returns:
            'BaseMessage': OUT: Result of the operation."""

        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    """User message.

    Inherits from: BaseMessage

    Attributes:
        role (Literal[Roles.user]): role.
        content (str | list[ContentChunk]): content."""

    role: Literal[Roles.user] = Roles.user
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | list[dict[str, str | dict[str, Any]]]]: OUT: Result of the operation."""

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [chunk.to_openai() for chunk in self.content],
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "UserMessage":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_message (dict[str, str | list[dict[str, str | dict[str, Any]]]]): IN: openai message. OUT: Consumed during execution.
        Returns:
            'UserMessage': OUT: Result of the operation."""

        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            },
        )


class SystemMessage(BaseMessage):
    """System message.

    Inherits from: BaseMessage

    Attributes:
        role (Literal[Roles.system]): role.
        content (str | list[ContentChunk]): content."""

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | list[dict[str, str | dict[str, Any]]]]: OUT: Result of the operation."""

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "SystemMessage":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_message (dict[str, str | list[dict[str, str | dict[str, Any]]]]): IN: openai message. OUT: Consumed during execution.
        Returns:
            'SystemMessage': OUT: Result of the operation."""

        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            }
        )


class AssistantMessage(BaseMessage):
    """Assistant message.

    Inherits from: BaseMessage

    Attributes:
        role (Literal[Roles.assistant]): role.
        content (str | None): content.
        tool_calls (list[ToolCall] | None): tool calls.
        prefix (bool): prefix."""

    role: Literal[Roles.assistant] = Roles.assistant
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | list[dict[str, str | dict[str, Any]]]]: OUT: Result of the operation."""

        out_dict: dict[str, str | list[dict[str, str | dict[str, Any]]]] = {
            "role": self.role,
            "content": self.content if self.content is not None else "",
        }
        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        return out_dict

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "AssistantMessage":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_message (dict[str, str | list[dict[str, str | dict[str, Any]]]]): IN: openai message. OUT: Consumed during execution.
        Returns:
            'AssistantMessage': OUT: Result of the operation."""

        openai_tool_calls = openai_message.get("tool_calls")
        if isinstance(openai_tool_calls, list):
            tools_calls = [ToolCall.from_openai(openai_tool_call) for openai_tool_call in openai_tool_calls]
        else:
            tools_calls = None
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": openai_message.get("content"),
                "tool_calls": tools_calls,
            }
        )


class ToolMessage(BaseMessage):
    """Tool message.

    Inherits from: BaseMessage

    Attributes:
        content (str): content.
        role (Literal[Roles.tool]): role.
        tool_call_id (str | None): tool call id."""

    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str | list[dict[str, str | dict[str, Any]]]]: OUT: Result of the operation."""

        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "ToolMessage":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            messages (dict[str, str | list[dict[str, str | dict[str, Any]]]]): IN: messages. OUT: Consumed during execution.
        Returns:
            'ToolMessage': OUT: Result of the operation."""

        tool_message = cls.model_validate(
            dict(
                content=messages["content"],
                role=messages["role"],
                tool_call_id=messages.get("tool_call_id", None),
            )
        )
        assert tool_message.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return tool_message


_map_type_to_role = {
    ToolMessage: Roles.tool,
    UserMessage: Roles.user,
    AssistantMessage: Roles.assistant,
    SystemMessage: Roles.system,
}

_map_role_to_type = {
    Roles.tool: ToolMessage,
    Roles.user: UserMessage,
    Roles.assistant: AssistantMessage,
    Roles.system: SystemMessage,
}


class MessagesHistory(XerxesBase):
    """Messages history.

    Inherits from: XerxesBase

    Attributes:
        messages (list[Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator='role')]]): messages."""

    messages: list[
        Annotated[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage,
            Field(discriminator="role"),
        ]
    ]

    def to_openai(self) -> dict[str, list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]]: OUT: Result of the operation."""

        message = []
        for msg in self.messages:
            msg_dict = msg.to_openai()
            if msg_dict.get("role", "") == "system" and msg_dict.get("content", "default") == "":
                ...
            else:
                message.append(msg_dict)
        return {"messages": message}

    @classmethod
    def from_openai(
        cls,
        openai_messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]],
    ) -> "MessagesHistory":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_messages (list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]): IN: openai messages. OUT: Consumed during execution.
        Returns:
            'MessagesHistory': OUT: Result of the operation."""

        messages: list[SystemMessage | UserMessage | AssistantMessage | ToolMessage] = []
        for message in openai_messages:
            role = message.get("role")
            if not isinstance(role, str):
                raise ValueError("Message role must be a string.")
            if role == Roles.system:
                messages.append(SystemMessage.from_openai(message))
            elif role == Roles.user:
                messages.append(UserMessage.from_openai(message))
            elif role == Roles.assistant:
                messages.append(AssistantMessage.from_openai(message))
            elif role == Roles.tool:
                messages.append(ToolMessage.from_openai(message))
            else:
                raise ValueError(f"Unknown message role: {role}")
        return MessagesHistory(messages=messages)

    def make_instruction_prompt(
        self,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
    ) -> str:
        """Make instruction prompt.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        ind1 = "  "
        prompt_parts: list[str] = []
        system_msg: SystemMessage | None = next((m for m in self.messages if isinstance(m, SystemMessage)), None)
        prompt_parts.append("# Instruction")
        if system_msg and system_msg.content:
            if isinstance(system_msg.content, str):
                prompt_parts.append(textwrap.indent(system_msg.content.strip(), ind1))
            else:
                content_text = "".join(chunk.text for chunk in system_msg.content if isinstance(chunk, TextChunk))
                prompt_parts.append(textwrap.indent(content_text.strip(), ind1))
        else:
            prompt_parts.append(f"{ind1}(No system prompt provided)")

        other_msgs = [m for m in self.messages if not isinstance(m, SystemMessage)]

        def _capitalize_role(role: str | Roles) -> str:
            """Internal helper to capitalize role.

            Args:
                role (str | Roles): IN: role. OUT: Consumed during execution.
            Returns:
                str: OUT: Result of the operation."""

            if hasattr(role, "value"):
                return role.value.capitalize()
            return role.capitalize()

        if other_msgs:
            prompt_parts.append(f"\n# {conversation_name_holder}")
            formatted_msgs = []
            for msg in other_msgs:
                role_title = f"## {_capitalize_role(msg.role)}"
                inner: list[str] = []
                if isinstance(msg, UserMessage | SystemMessage):
                    if isinstance(msg.content, str):
                        inner.append(msg.content)
                    else:
                        for chunk in msg.content:
                            if hasattr(chunk, "text"):
                                inner.append(chunk.text)
                            elif hasattr(chunk, "image"):
                                inner.append("[IMAGE CHUNK]")
                            elif hasattr(chunk, "image_url"):
                                inner.append(f"[IMAGE URL: {chunk.get_url()}]")

                elif isinstance(msg, AssistantMessage):
                    if msg.content:
                        inner.append(msg.content)

                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            xml_call = (
                                f"<{tc.function.name}>"
                                f"<arguments>{tc.function.arguments}</arguments>"
                                f"</{tc.function.name}>"
                            )
                            inner.append(xml_call)
                elif isinstance(msg, ToolMessage):
                    tool_res = textwrap.indent(str(msg.content), ind1)
                    inner.append(f"Tool Result (ID: {msg.tool_call_id}):\n{tool_res}")

                formatted_block = textwrap.indent("\n".join(inner).strip(), ind1)
                formatted_msgs.append(f"{role_title}\n{formatted_block}")

            prompt_parts.append("\n\n".join(formatted_msgs))

        if mention_last_turn and other_msgs:
            last = other_msgs[-1]
            preview = last.content if isinstance(last, UserMessage | ToolMessage) else last.content or "[tool calls]"
            prompt_parts.append(f"\nLast Message from {_capitalize_role(last.role)}: {preview}")

        return "\n\n".join(prompt_parts)


ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]

ChatMessageType = TypeVar("ChatMessageType", bound=ChatMessage)

UserMessageType = TypeVar("UserMessageType", bound=UserMessage)

AssistantMessageType = TypeVar("AssistantMessageType", bound=AssistantMessage)

ToolMessageType = TypeVar("ToolMessageType", bound=ToolMessage)

SystemMessageType = TypeVar("SystemMessageType", bound=SystemMessage)

ConversionType: TypeAlias = list[ChatMessage]
