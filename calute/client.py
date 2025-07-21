# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

if tp.TYPE_CHECKING:
    from openai import Client


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str] | None,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        repetition_penalty: float,
        stream: bool,
        **kwargs,
    ) -> tp.Any:
        """Generate a completion using the LLM"""
        pass

    @abstractmethod
    def extract_content(self, response: tp.Any) -> str:
        """Extract text content from a non-streaming response"""
        pass

    @abstractmethod
    async def process_streaming_response(
        self,
        response: tp.Any,
        callback: tp.Callable[[str, tp.Any], None],
    ) -> str:
        """Process a streaming response, calling the callback for each chunk"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI client implementation"""

    def __init__(self, client: Client):
        self.client = client

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str] | None,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        repetition_penalty: float,
        stream: bool,
        extra_body: dict | None = None,
        **kwargs,
    ) -> tp.Any:
        """Generate a completion using OpenAI"""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if extra_body is None:
            extra_body = {}
        extra_body = {"repetition_penalty": repetition_penalty, "top_k": top_k, "min_p": min_p, **extra_body}

        return self.client.chat.completions.create(
            messages=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=stream,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=extra_body,
        )

    def extract_content(self, response: tp.Any) -> str:
        """Extract text content from an OpenAI response"""
        return response.choices[0].message.content

    async def process_streaming_response(
        self,
        response: tp.Any,
        callback: tp.Callable[[str, tp.Any], None],
    ) -> str:
        """Process an OpenAI streaming response"""
        buffered_content = ""

        for chunk in response:
            content = None
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    buffered_content += delta.content
                    content = delta.content

            if content:
                callback(content, chunk)

        return buffered_content


class GeminiClient(LLMClient):
    """Google Gemini client implementation"""

    def __init__(self, client):
        self.client = client

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str] | None,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        repetition_penalty: float,
        stream: bool,
        **kwargs,
    ) -> tp.Any:
        """Generate a completion using Gemini"""
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty,
        }

        if stop:
            generation_config["stop_sequences"] = stop

        model_instance = self.client.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

        return model_instance.generate_content(prompt, stream=stream)

    def extract_content(self, response: tp.Any) -> str:
        """Extract text content from a Gemini response"""
        return response.text

    async def process_streaming_response(
        self,
        response: tp.Any,
        callback: tp.Callable[[str, tp.Any], None],
    ) -> str:
        """Process a Gemini streaming response"""
        buffered_content = ""

        async for chunk in response:
            content = None
            if hasattr(chunk, "text") and chunk.text:
                buffered_content += chunk.text
                content = chunk.text

            if content:
                callback(content, chunk)

        return buffered_content
