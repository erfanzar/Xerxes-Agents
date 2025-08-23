import re
import textwrap
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import ConfigDict, Field

from ..multimodal import SerializableImage
from ..utils import CaluteBase
from .tool_calls import ToolCall


class ChunkTypes(str, Enum):
    r"""Enum for the types of chunks that can be sent to the model.

    Attributes:
       text: A text chunk.
       image: An image chunk.
       image_url: An image url chunk.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import ChunkTypes
        >>> chunk_type = ChunkTypes.text
    """

    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(CaluteBase):
    r"""Base class for all content chunks.

    Content chunks are used to send different types of content to the model.

    Attributes:
       type: The type of the chunk.
    """

    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        r"""Converts the chunk to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "BaseContentChunk":  # type:ignore
        r"""Converts the OpenAI chunk to the Calute format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    r"""Image chunk.

    Attributes:
       image: The image to be sent to the model.

    Examples:
        >>> from PIL import Image
        >>> image_chunk = ImageChunk(image=Image.new('RGB', (200, 200), color='blue'))
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        r"""Converts the chunk to the OpenAI format."""
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, str | dict[str, str]]) -> "ImageChunk":
        r"""Converts the OpenAI chunk to the Calute format."""
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):  # Remove the prefix if it exists
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(CaluteBase):
    r"""Image URL or a base64 encoded image.

    Attributes:
       url: The URL of the image.
       detail: The detail of the image.

    Examples:
       >>> image_url = ImageURL(url="https://example.com/image.png")
    """

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    r"""Image URL chunk.

    Attributes:
       image_url: The URL of the image or a base64 encoded image to be sent to the model.

    Examples:
        >>> image_url_chunk = ImageURLChunk(image_url="data:image/png;base64,iVBORw0")
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        r"""Converts the chunk to the OpenAI format."""
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
        r"""Converts the OpenAI chunk to the Calute format."""
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class TextChunk(BaseContentChunk):
    r"""Text chunk.

    Attributes:
      text: The text to be sent to the model.

    Examples:
        >>> text_chunk = TextChunk(text="Hello, how can I help you?")
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> dict[str, str | dict[str, str]]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | dict[str, str]]) -> "TextChunk":
        r"""Converts the OpenAI chunk to the Calute format."""
        return cls.model_validate(messages)


ContentChunk = Annotated[TextChunk | ImageChunk | ImageURLChunk, Field(discriminator="type")]


def _convert_openai_content_chunks(openai_content_chunks: dict[str, str | dict[str, str]]) -> ContentChunk:
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


class Roles(str, Enum):
    r"""Enum for the roles of the messages.

    Attributes:
       system: The system role.
       user: The user role.
       assistant: The assistant role.
       tool: The tool role.

    Examples:
        >>> role = Roles.user
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(CaluteBase):
    r"""Base class for all messages.

    Attributes:
       role: The role of the message.
    """

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        r"""Converts the message to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "BaseMessage":  # type:ignore
        r"""Converts the OpenAI message to the Calute format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    r"""User message.

    Attributes:
        content: The content of the message.

    Examples:
        >>> message = UserMessage(content="Can you help me to write a poem?")
    """

    role: Literal[Roles.user] = Roles.user
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        r"""Converts the message to the OpenAI format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [chunk.to_openai() for chunk in self.content],
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "UserMessage":
        r"""Converts the OpenAI message to the Calute format."""
        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            },
        )


class SystemMessage(BaseMessage):
    r"""System message.

    Attributes:
        content: The content of the message.

    Examples:
        >>> message = SystemMessage(content="You are a helpful assistant.")
    """

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        r"""Converts the message to the OpenAI format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "SystemMessage":
        r"""Converts the OpenAI message to the Calute format."""
        if isinstance(openai_message["content"], str):
            return cls.model_validate(dict(role=openai_message["role"], content=openai_message["content"]))
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            }
        )


class AssistantMessage(BaseMessage):
    r"""Assistant message.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        tool_calls: The tool calls of the message.
        prefix: Whether the message is a prefix.

    Examples:
        >>> message = AssistantMessage(content="Hello, how can I help you?")
    """

    role: Literal[Roles.assistant] = Roles.assistant
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        r"""Converts the message to the OpenAI format."""
        out_dict: dict[str, str | list[dict[str, str | dict[str, Any]]]] = {
            "role": self.role,
        }
        if self.content is not None:
            out_dict["content"] = self.content
        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        return out_dict

    @classmethod
    def from_openai(cls, openai_message: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "AssistantMessage":
        r"""Converts the OpenAI message to the Calute format."""
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
    r"""Tool message.

    Attributes:
        content: The content of the message.
        tool_call_id: The tool call id of the message.

    Examples:
       >>> message = ToolMessage(content="Hello, how can I help you?", tool_call_id="123")
    """

    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, str | list[dict[str, str | dict[str, Any]]]]:
        r"""Converts the message to the OpenAI format."""
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "ToolMessage":
        r"""Converts the OpenAI message to the Calute format."""
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


class MessagesHistory(CaluteBase):
    messages: list[
        Annotated[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage,
            Field(discriminator="role"),
        ]
    ]

    def to_openai(self) -> list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]:
        r"""Converts the message to the OpenAI format."""
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
        messages = []
        for message in openai_messages:
            messages.append(_map_role_to_type[message.get("role")].from_openai(message))
        return MessagesHistory(messages=messages)

    def make_instruction_prompt(
        self,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
    ) -> str:
        """
        Formats the entire message history into a single, human-readable string
        while ensuring any tool calls are rendered in the canonical XML format
        so the LLM continues to use it.
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

        def _capitalize_role(role):
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
