# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Message type definitions for Xerxes.

This module provides a comprehensive message system for handling
conversations with LLM models. It includes:
- Content chunk types for text, images, and image URLs
- Message types for different conversation roles (system, user, assistant, tool)
- Message history management with OpenAI format conversion
- Support for multimodal content including images and text

The message system follows a type-safe design using Pydantic models
with discriminated unions for proper serialization and validation.

Example:
    >>> from xerxes_agent.types.messages import UserMessage, MessagesHistory
    >>> user_msg = UserMessage(content="Hello, how can I help you?")
    >>> history = MessagesHistory(messages=[user_msg])
    >>> openai_format = history.to_openai()
"""

import re
import textwrap
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import ConfigDict, Field

from ..core.multimodal import SerializableImage
from ..core.utils import XerxesBase
from .tool_calls import ToolCall


class ChunkTypes(StrEnum):
    """Enumeration of content chunk types supported in messages.

    Defines the different types of content chunks that can be included
    in multimodal messages sent to LLM models.

    Attributes:
        text: A plain text content chunk.
        image: A binary image content chunk (PIL Image or base64-encoded).
        image_url: An image referenced by URL or base64 data URI.

    Example:
        >>> from xerxes_agent.types.messages import ChunkTypes
        >>> chunk_type = ChunkTypes.text
        >>> chunk_type.value
        'text'
    """

    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(XerxesBase):
    """Abstract base class for all content chunks in multimodal messages.

    Content chunks represent individual pieces of content within a message,
    such as text passages, images, or image URLs. Subclasses must implement
    the ``to_openai`` and ``from_openai`` methods for format conversion.

    Attributes:
        type: The discriminator type of the chunk, one of 'text', 'image',
            or 'image_url'.
    """

    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Convert this content chunk to the OpenAI API format.

        Must be implemented by subclasses to provide the appropriate
        dictionary representation for the OpenAI messages API.

        Returns:
            A dictionary in OpenAI content chunk format with a 'type' key
            and type-specific content keys.

        Raises:
            NotImplementedError: Always raised in the base class; subclasses
                must override this method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "BaseContentChunk":  # type:ignore
        """Create a content chunk from an OpenAI-format dictionary.

        Must be implemented by subclasses to parse the OpenAI content chunk
        format into the appropriate Xerxes chunk type.

        Args:
            openai_chunk: A dictionary in OpenAI content chunk format containing
                at minimum a 'type' key.

        Returns:
            An instance of the appropriate BaseContentChunk subclass.

        Raises:
            NotImplementedError: Always raised in the base class; subclasses
                must override this method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    """Content chunk containing a binary image.

    Wraps a PIL Image or base64-encoded image data for inclusion in
    multimodal messages. When converted to OpenAI format, the image is
    serialized as a base64 data URI within an ``image_url`` structure.

    Attributes:
        type: Chunk type discriminator, always ``ChunkTypes.image``.
        image: The image data, either a PIL Image object or a base64 string,
            wrapped in a SerializableImage for Pydantic compatibility.

    Example:
        >>> from PIL import Image
        >>> image_chunk = ImageChunk(image=Image.new('RGB', (200, 200), color='blue'))
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Convert the image chunk to OpenAI API format.

        Serializes the image to a base64 data URI and wraps it in the
        OpenAI ``image_url`` content chunk format.

        Returns:
            A dictionary with ``type`` set to ``"image_url"`` and an
            ``image_url`` sub-dictionary containing the base64 data URI.
        """
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageChunk":
        """Create an ImageChunk from an OpenAI-format image_url dictionary.

        Parses the base64 data URI from the OpenAI format, stripping the
        ``data:image/...;base64,`` prefix if present, and creates an
        ImageChunk with the raw base64 image data.

        Args:
            openai_chunk: A dictionary with ``type`` set to ``"image_url"``
                and an ``image_url`` sub-dictionary containing a ``url`` key.

        Returns:
            A new ImageChunk instance with the parsed image data.

        Raises:
            AssertionError: If the chunk type is not ``"image_url"`` or the
                ``image_url`` dictionary is malformed.
        """
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(XerxesBase):
    """Represents an image reference by URL or base64-encoded data URI.

    Used within ``ImageURLChunk`` to specify the image source and optional
    detail level for vision-capable models.

    Attributes:
        url: The URL of the image or a base64-encoded data URI string
            (e.g., ``"data:image/png;base64,..."`` or ``"https://..."``).
        detail: Optional detail level hint for the model (e.g., ``"low"``,
            ``"high"``, ``"auto"``). Controls image processing resolution.

    Example:
        >>> image_url = ImageURL(url="https://example.com/image.png", detail="high")
    """

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    """Content chunk containing an image referenced by URL or data URI.

    Supports both plain URL strings and structured ``ImageURL`` objects
    with optional detail level hints for vision-capable models.

    Attributes:
        type: Chunk type discriminator, always ``ChunkTypes.image_url``.
        image_url: The image reference, either an ``ImageURL`` object with
            ``url`` and optional ``detail`` fields, or a plain URL string.

    Example:
        >>> chunk = ImageURLChunk(image_url="https://example.com/photo.jpg")
        >>> chunk_with_detail = ImageURLChunk(
        ...     image_url=ImageURL(url="https://example.com/photo.jpg", detail="high")
        ... )
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        """Extract the URL string from the image_url attribute.

        Handles both ``ImageURL`` objects and plain string URLs, providing
        a uniform way to access the underlying URL.

        Returns:
            The URL string for the image, regardless of whether ``image_url``
            is an ``ImageURL`` instance or a plain string.
        """
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Convert the image URL chunk to OpenAI API format.

        Creates an OpenAI-compatible content chunk dictionary with the image
        URL and optional detail level.

        Returns:
            A dictionary with ``type`` set to ``"image_url"`` and an
            ``image_url`` sub-dictionary containing the URL and optionally
            a ``detail`` key.
        """
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
        """Create an ImageURLChunk from an OpenAI-format content chunk dictionary.

        Args:
            openai_chunk: A dictionary with an ``image_url`` key containing
                either a URL string or a dictionary with ``url`` and optional
                ``detail`` keys.

        Returns:
            A new ImageURLChunk instance parsed from the OpenAI format.
        """
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class TextChunk(BaseContentChunk):
    """Content chunk containing plain text.

    The most common content chunk type, representing a segment of text
    within a multimodal message.

    Attributes:
        type: Chunk type discriminator, always ``ChunkTypes.text``.
        text: The text content string.

    Example:
        >>> text_chunk = TextChunk(text="Hello, how can I help you?")
        >>> text_chunk.to_openai()
        {'type': 'text', 'text': 'Hello, how can I help you?'}
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        """Convert the text chunk to OpenAI API format.

        Returns:
            A dictionary with ``type`` set to ``"text"`` and a ``text`` key
            containing the text content.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | dict[str, str]]) -> "TextChunk":
        """Create a TextChunk from an OpenAI-format content chunk dictionary.

        Args:
            messages: A dictionary with ``type`` set to ``"text"`` and a
                ``text`` key containing the text content.

        Returns:
            A new TextChunk instance with the parsed text.
        """
        return cls.model_validate(messages)


ContentChunk = Annotated[TextChunk | ImageChunk | ImageURLChunk, Field(discriminator="type")]


def _convert_openai_content_chunks(openai_content_chunks: dict[str, str | dict[str, str]]) -> ContentChunk:
    """Convert an OpenAI format content chunk to the appropriate Xerxes content chunk type.

    This internal function handles the conversion of content chunks from the OpenAI
    message format to their corresponding Xerxes ContentChunk types based on the
    'type' field.

    Args:
        openai_content_chunks: A dictionary representing an OpenAI format content chunk.
            Must contain a 'type' field indicating the chunk type.

    Returns:
        The appropriate ContentChunk subclass instance (TextChunk, ImageURLChunk,
        or ImageChunk) based on the chunk type.

    Raises:
        ValueError: If the content chunk has no 'type' field or an unknown type.
    """
    content_type_str = openai_content_chunks.get("type")

    if content_type_str is None:
        raise ValueError("Content chunk must have a type field.")

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
    """Enumeration of message roles in a chat conversation.

    Defines the four standard roles used in chat completion APIs to
    distinguish between different participants in a conversation.

    Attributes:
        system: The system role, used for initial instructions and context.
        user: The user role, representing the human participant.
        assistant: The assistant role, representing the AI model's responses.
        tool: The tool role, used for function/tool execution results.

    Example:
        >>> role = Roles.user
        >>> role.value
        'user'
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(XerxesBase):
    """Abstract base class for all chat message types.

    Provides the common interface for message serialization to and from
    OpenAI format. Subclasses implement role-specific content handling
    and serialization logic.

    Attributes:
        role: The role of the message sender, one of ``system``, ``user``,
            ``assistant``, or ``tool``.
    """

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Convert this message to the OpenAI API format.

        Must be implemented by subclasses to produce a dictionary compatible
        with the OpenAI chat completion messages format.

        Returns:
            A dictionary with at minimum ``role`` and ``content`` keys,
            formatted for the OpenAI API.

        Raises:
            NotImplementedError: Always raised in the base class; subclasses
                must override this method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "BaseMessage":  # type:ignore
        """Create a message instance from an OpenAI-format dictionary.

        Must be implemented by subclasses to parse OpenAI-format message
        dictionaries into the appropriate Xerxes message type.

        Args:
            openai_message: A dictionary in OpenAI message format containing
                at minimum a ``role`` key.

        Returns:
            An instance of the appropriate BaseMessage subclass.

        Raises:
            NotImplementedError: Always raised in the base class; subclasses
                must override this method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    """Message from the user in a chat conversation.

    Supports both plain text content and multimodal content consisting of
    a list of content chunks (text, images, image URLs).

    Attributes:
        role: The message role, always ``Roles.user``.
        content: The message content, either a plain text string or a list of
            ``ContentChunk`` objects for multimodal messages.

    Example:
        >>> message = UserMessage(content="Can you help me to write a poem?")
        >>> message.role
        <Roles.user: 'user'>
    """

    role: Literal[Roles.user] = Roles.user
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Convert the user message to OpenAI API format.

        Handles both plain text content (returned as a string) and multimodal
        content (returned as a list of content chunk dictionaries).

        Returns:
            A dictionary with ``role`` and ``content`` keys in OpenAI format.
        """
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [chunk.to_openai() for chunk in self.content],
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "UserMessage":
        """Create a UserMessage from an OpenAI-format message dictionary.

        Handles both plain text content strings and lists of content chunk
        dictionaries for multimodal messages.

        Args:
            openai_message: A dictionary with ``role`` set to ``"user"`` and
                a ``content`` key containing either a string or a list of
                content chunk dictionaries.

        Returns:
            A new UserMessage instance with properly typed content.
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
    """System-level instruction message for configuring model behavior.

    System messages set the context, personality, and constraints for
    the AI assistant. They are typically placed at the beginning of a
    conversation and support both plain text and multimodal content.

    Attributes:
        role: The message role, always ``Roles.system``.
        content: The system instruction content, either a plain text string
            or a list of ``ContentChunk`` objects.

    Example:
        >>> message = SystemMessage(content="You are a helpful assistant.")
        >>> message.to_openai()
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    """

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Convert the system message to OpenAI API format.

        Returns:
            A dictionary with ``role`` and ``content`` keys in OpenAI format.
        """
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "SystemMessage":
        """Create a SystemMessage from an OpenAI-format message dictionary.

        Args:
            openai_message: A dictionary with ``role`` set to ``"system"`` and
                a ``content`` key containing either a string or a list of
                content chunk dictionaries.

        Returns:
            A new SystemMessage instance with properly typed content.
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
    """Message generated by the AI assistant.

    Represents the model's response in a conversation. May contain text
    content, tool/function calls, or both. The ``prefix`` flag indicates
    whether this message should be treated as a prefix-fill for the next
    generation (used by some model APIs).

    Attributes:
        role: The message role, always ``Roles.assistant``.
        content: The text content of the assistant's response, or None if
            the response consists only of tool calls.
        tool_calls: Optional list of ``ToolCall`` objects representing
            function/tool invocations requested by the model.
        prefix: Whether this message serves as a prefix for continued
            generation (default: False).

    Example:
        >>> message = AssistantMessage(content="Hello, how can I help you?")
        >>> message.to_openai()
        {'role': 'assistant', 'content': 'Hello, how can I help you?'}
    """

    role: Literal[Roles.assistant] = Roles.assistant
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Convert the assistant message to OpenAI API format.

        Always includes ``content`` so OpenAI-compatible servers that require
        the field for tool-call turns still accept the payload.

        Returns:
            A dictionary with ``role``, ``content``, and optionally
            ``tool_calls`` keys in OpenAI format.
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
        """Create an AssistantMessage from an OpenAI-format message dictionary.

        Parses both the text content and any tool calls from the OpenAI
        message format.

        Args:
            openai_message: A dictionary with ``role`` set to ``"assistant"``,
                optional ``content`` string, and optional ``tool_calls`` list.

        Returns:
            A new AssistantMessage instance with parsed content and tool calls.
        """
        openai_tool_calls = openai_message.get("tool_calls", None)
        tools_calls = (
            [ToolCall.from_openai(openai_tool_call) for openai_tool_call in openai_tool_calls]
            if openai_tool_calls is not None
            else None
        )
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": openai_message.get("content"),
                "tool_calls": tools_calls,
            }
        )


class ToolMessage(BaseMessage):
    """Message containing the result of a tool/function call execution.

    Tool messages are sent after a function has been executed to provide
    the result back to the model. Each tool message must reference the
    specific tool call it responds to via ``tool_call_id``.

    Attributes:
        content: The string result of the tool execution.
        role: The message role, always ``Roles.tool``.
        tool_call_id: The unique identifier of the tool call this message
            responds to. Must not be None when converting to OpenAI format.

    Example:
        >>> message = ToolMessage(
        ...     content='{"temperature": 72, "unit": "fahrenheit"}',
        ...     tool_call_id="call_abc123"
        ... )
    """

    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        """Convert the tool message to OpenAI API format.

        Returns:
            A dictionary with ``role``, ``content``, and ``tool_call_id`` keys.

        Raises:
            AssertionError: If ``tool_call_id`` is None, as OpenAI requires
                tool messages to reference a specific tool call.
        """
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "ToolMessage":
        """Create a ToolMessage from an OpenAI-format message dictionary.

        Args:
            messages: A dictionary with ``role`` set to ``"tool"``, a
                ``content`` string, and a ``tool_call_id`` string.

        Returns:
            A new ToolMessage instance with the parsed content and tool call ID.

        Raises:
            AssertionError: If ``tool_call_id`` is not present in the input,
                as it is required for tool messages.
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

_map_role_to_type = {v: k for k, v in _map_type_to_role.items()}


class MessagesHistory(XerxesBase):
    """Container for managing a sequence of chat messages.

    Provides a structured container for storing and manipulating chat
    conversation history. Supports conversion to/from OpenAI format
    and generation of instruction prompts for model input.

    Attributes:
        messages: List of chat messages in the conversation. Each message
            is discriminated by its role (system, user, assistant, or tool).

    Example:
        >>> from xerxes_agent.types.messages import UserMessage, AssistantMessage, MessagesHistory
        >>> history = MessagesHistory(messages=[
        ...     UserMessage(content="Hello!"),
        ...     AssistantMessage(content="Hi there! How can I help you?"),
        ... ])
        >>> openai_format = history.to_openai()
    """

    messages: list[
        Annotated[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage,
            Field(discriminator="role"),
        ]
    ]

    def to_openai(self) -> list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]:
        """Convert all messages to OpenAI API format.

        Iterates through all messages, converts each to OpenAI format,
        and filters out system messages with empty content.

        Returns:
            A dictionary with a ``messages`` key containing a list of
            OpenAI-format message dictionaries.
        """
        message = []
        for msg in self.messages:
            msg = msg.to_openai()
            if msg.get("role", "") == "system" and msg.get("content", "default") == "":
                ...
            else:
                message.append(msg)
        return {"messages": message}  # type:ignore

    @classmethod
    def from_openai(
        cls,
        openai_messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]],
    ) -> "MessagesHistory":
        """Create a MessagesHistory from OpenAI format messages.

        Converts a list of OpenAI format message dictionaries into a
        MessagesHistory instance with properly typed message objects.

        Args:
            openai_messages: List of message dictionaries in OpenAI format.
                Each dictionary must contain a 'role' field to determine
                the message type.

        Returns:
            A MessagesHistory instance containing the converted messages.

        Example:
            >>> openai_msgs = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ... ]
            >>> history = MessagesHistory.from_openai(openai_msgs)
        """
        messages = []
        for message in openai_messages:
            messages.append(_map_role_to_type[message.get("role")].from_openai(message))
        return MessagesHistory(messages=messages)

    def make_instruction_prompt(
        self,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
    ) -> str:
        """Format the message history into a human-readable instruction prompt.

        Converts the entire message history into a single, structured string
        suitable for LLM input. Tool calls are rendered in canonical XML format
        to encourage the LLM to follow the same pattern.

        Args:
            conversation_name_holder: The section header name for the conversation
                history section (default: "Messages").
            mention_last_turn: Whether to append a summary of the last message
                at the end of the prompt (default: True).

        Returns:
            A formatted string containing the instruction prompt with all
            messages properly structured and indented.
        """
        ind1 = "  "
        prompt_parts: list[str] = []
        system_msg: SystemMessage | None = next((m for m in self.messages if isinstance(m, SystemMessage)), None)
        prompt_parts.append("# Instruction")
        if system_msg and system_msg.content:
            prompt_parts.append(textwrap.indent(system_msg.content.strip(), ind1))
        else:
            prompt_parts.append(f"{ind1}(No system prompt provided)")

        other_msgs = [m for m in self.messages if not isinstance(m, SystemMessage)]

        def _capitalize_role(role: str | Roles) -> str:
            """Capitalize a role name for display purposes.

            Args:
                role: A Roles enum member or plain string.

            Returns:
                Capitalized string representation of the role.
            """
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
