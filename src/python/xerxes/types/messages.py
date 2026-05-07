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
    """Discriminators for the three supported content chunk variants.

    Attributes:
        text: Plain text content.
        image: Binary image content.
        image_url: Image referenced by URL.
    """

    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(XerxesBase):
    """Abstract base for content chunks carried in messages.

    Subclasses (``TextChunk``, ``ImageChunk``, ``ImageURLChunk``) correspond to
    the three variants defined in ``ChunkTypes``.

    Attributes:
        type: Discriminator selecting the chunk variant.
    """

    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Serialize this chunk to an OpenAI content-block dictionary.

        Returns:
            A dictionary compatible with the OpenAI ``content`` block format.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "BaseContentChunk":
        """Construct a content chunk from an OpenAI content-block dictionary.

        Args:
            openai_chunk: A dictionary conforming to the OpenAI ``content`` block format.

        Returns:
            A subclass instance of ``BaseContentChunk`` corresponding to the input.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    """A content chunk containing binary image data encoded as a string.

    Attributes:
        type: Discriminator set to ``ChunkTypes.image``.
        image: Serializable image data.
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Serialize this chunk to an OpenAI image_url content block.

        Returns:
            An OpenAI-compatible ``{"type": "image_url", "image_url": {"url": ...}}`` dict.
        """
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageChunk":
        """Construct an ImageChunk from an OpenAI image_url content block.

        Args:
            openai_chunk: An OpenAI content block with ``type: "image_url"``.

        Returns:
            An ``ImageChunk`` instance.

        Raises:
            AssertionError: If the input does not have ``type: "image_url"``.
        """
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(XerxesBase):
    """A reference to an image via URL with optional detail level.

    Attributes:
        url: The image URL.
        detail: Optional detail level hint (e.g., ``"low"``, ``"high"``, ``"auto"``).
    """

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    """A content chunk referencing an image by URL.

    Attributes:
        type: Discriminator set to ``ChunkTypes.image_url``.
        image_url: Either an ``ImageURL`` object or a raw URL string.
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        """Return the URL string, extracting it from an ImageURL if needed.

        Returns:
            The URL as a string.
        """
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Serialize this chunk to an OpenAI image_url content block.

        Returns:
            An OpenAI-compatible ``{"type": "image_url", "image_url": {"url": ...}}`` dict.
        """
        image_url_dict: dict[str, str] = {"url": self.get_url()}
        if isinstance(self.image_url, ImageURL) and self.image_url.detail is not None:
            image_url_dict["detail"] = self.image_url.detail

        out_dict: dict[str, str | dict[str, str]] = {
            "type": "image_url",
            "image_url": image_url_dict,
        }
        return out_dict

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageURLChunk":
        """Construct an ImageURLChunk from an OpenAI image_url content block.

        Args:
            openai_chunk: An OpenAI content block with ``type: "image_url"``.

        Returns:
            An ``ImageURLChunk`` instance.
        """
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class TextChunk(BaseContentChunk):
    """A content chunk containing plain text.

    Attributes:
        type: Discriminator set to ``ChunkTypes.text``.
        text: The text content.
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Serialize this chunk to an OpenAI text content block.

        Returns:
            An OpenAI-compatible ``{"type": "text", "text": "..."}`` dict.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | dict[str, str]]) -> "TextChunk":
        """Construct a TextChunk from an OpenAI text content block.

        Args:
            messages: An OpenAI content block with ``type: "text"``.

        Returns:
            A ``TextChunk`` instance.
        """
        return cls.model_validate(messages)


ContentChunk = Annotated[TextChunk | ImageChunk | ImageURLChunk, Field(discriminator="type")]


def _convert_openai_content_chunks(openai_content_chunks: dict[str, str | dict[str, str]]) -> ContentChunk:
    """Convert a single OpenAI content block to a Xerxes ``ContentChunk``.

    Args:
        openai_content_chunks: A content block dict with a ``type`` field.

    Returns:
        The appropriate ``ContentChunk`` subclass for the given type.

    Raises:
        ValueError: If the content type is missing or unrecognized.
    """

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
    """Message role discriminators for the Xerxes chat protocol.

    Attributes:
        system: System-level instruction message.
        user: User-authored message.
        assistant: LLM-generated response message.
        tool: Result returned from a tool call.
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(XerxesBase):
    """Abstract base for all message types in the Xerxes chat protocol.

    Subclasses correspond to the roles defined in ``Roles`` and provide
    OpenAI-compatible serialization.

    Attributes:
        role: Discriminator selecting the message variant.
    """

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Serialize this message to an OpenAI messages API dictionary.

        Returns:
            A dictionary compatible with the OpenAI messages format.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "BaseMessage":
        """Construct a message from an OpenAI messages API dictionary.

        Args:
            openai_message: A dictionary conforming to the OpenAI messages format.

        Returns:
            A subclass instance of ``BaseMessage`` corresponding to the input.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    """A message authored by the user.

    Attributes:
        role: Discriminator fixed to ``Roles.user``.
        content: Text content or a list of content chunks.
    """

    role: Literal[Roles.user] = Roles.user
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Serialize to an OpenAI messages API dictionary.

        Returns:
            An OpenAI-compatible ``{"role": "user", "content": ...}`` dict.
        """
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [chunk.to_openai() for chunk in self.content],
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "UserMessage":
        """Construct a UserMessage from an OpenAI messages API dictionary.

        Args:
            openai_message: An OpenAI ``{"role": "user", "content": ...}`` dict.

        Returns:
            A ``UserMessage`` instance.
        """
        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            },
        )


class SystemMessage(BaseMessage):
    """A system-level instruction message.

    Attributes:
        role: Discriminator fixed to ``Roles.system``.
        content: Text content or a list of content chunks.
    """

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Serialize to an OpenAI messages API dictionary.

        Returns:
            An OpenAI-compatible ``{"role": "system", "content": ...}`` dict.
        """
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "SystemMessage":
        """Construct a SystemMessage from an OpenAI messages API dictionary.

        Args:
            openai_message: An OpenAI ``{"role": "system", "content": ...}`` dict.

        Returns:
            A ``SystemMessage`` instance.
        """
        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            }
        )


class AssistantMessage(BaseMessage):
    """A message generated by the LLM assistant.

    Attributes:
        role: Discriminator fixed to ``Roles.assistant``.
        content: Text content of the response, or None if only tool calls are present.
        tool_calls: Optional list of tool calls requested by the assistant.
        prefix: Whether this message uses the ``prefix`` tool use format.
    """

    role: Literal[Roles.assistant] = Roles.assistant
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Serialize to an OpenAI messages API dictionary.

        Returns:
            An OpenAI-compatible ``{"role": "assistant", "content": ..., "tool_calls": ...}`` dict.
        """
        out_dict: dict[str, str | list[dict[str, str | dict[str, Any]]]] = {
            "role": self.role,
            "content": self.content if self.content is not None else "",
        }
        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        return out_dict

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "AssistantMessage":
        """Construct an AssistantMessage from an OpenAI messages API dictionary.

        Args:
            openai_message: An OpenAI ``{"role": "assistant", "content": ..., "tool_calls": ...}`` dict.

        Returns:
            An ``AssistantMessage`` instance.
        """
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
    """A message carrying the result of a tool invocation.

    Attributes:
        content: The result returned by the tool as a string.
        role: Discriminator fixed to ``Roles.tool``.
        tool_call_id: Identifier linking this result to the originating tool call.
    """

    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Serialize to an OpenAI messages API dictionary.

        Returns:
            An OpenAI-compatible ``{"role": "tool", "content": ..., "tool_call_id": ...}`` dict.

        Raises:
            AssertionError: If ``tool_call_id`` is not set.
        """
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "ToolMessage":
        """Construct a ToolMessage from an OpenAI messages API dictionary.

        Args:
            messages: An OpenAI ``{"role": "tool", "content": ..., "tool_call_id": ...}`` dict.

        Returns:
            A ``ToolMessage`` instance.

        Raises:
            AssertionError: If ``tool_call_id`` is not present in the input.
        """
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
    """A sequence of typed messages forming a conversation history.

    Provides OpenAI-compatible serialization and a utility to render the history
    as an instruction prompt for models that do not natively support message arrays.

    Attributes:
        messages: The ordered list of messages in this history.
    """

    messages: list[
        Annotated[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage,
            Field(discriminator="role"),
        ]
    ]

    def to_openai(self) -> dict[str, list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]]:
        """Serialize this history to an OpenAI messages API request body.

        Empty system messages are omitted from the output.

        Returns:
            A ``{"messages": [...]}`` dictionary ready for the OpenAI API.
        """
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
        """Reconstruct a MessagesHistory from a list of OpenAI message dictionaries.

        Args:
            openai_messages: A list of messages in OpenAI's messages API format.

        Returns:
            A ``MessagesHistory`` containing the deserialized messages.

        Raises:
            ValueError: If a message lacks a valid role string.
        """
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
        """Render this conversation as a plain-text instruction prompt.

        This is useful for models that accept a single prompt string rather than
        a structured message list. The output is Markdown-formatted and includes
        a system-prompt section, a conversation log, and an optional preview of
        the final turn.

        Args:
            conversation_name_holder: Heading text for the conversation section.
            mention_last_turn: Whether to append a "Last Message" trailer.

        Returns:
            A Markdown-formatted prompt string combining system context and conversation.
        """
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
            """Capitalize a role value for display."""
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
