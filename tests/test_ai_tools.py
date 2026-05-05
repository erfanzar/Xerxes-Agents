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
"""Tests for xerxes.tools.ai_tools module."""

from xerxes.tools.ai_tools import (
    EntityExtractor,
    TextClassifier,
    TextEmbedder,
    TextSimilarity,
    TextSummarizer,
)


class TestTextEmbedder:
    def test_tfidf_single(self):
        result = TextEmbedder.static_call("Hello world this is a test")
        assert "embeddings" in result
        assert len(result["embeddings"]) == 1

    def test_tfidf_multiple(self):
        result = TextEmbedder.static_call(["Hello world", "Foo bar baz"])
        assert "embeddings" in result
        assert len(result["embeddings"]) == 2

    def test_max_length(self):
        result = TextEmbedder.static_call("x" * 1000, max_length=50)
        assert "embeddings" in result


class TestTextSimilarity:
    def test_basic(self):
        result = TextSimilarity.static_call(
            text1="Python programming language",
            text2="Python coding tutorial",
        )
        assert "similarity" in result or "score" in result or "error" not in result

    def test_different_texts(self):
        result = TextSimilarity.static_call(
            text1="The cat sat on the mat",
            text2="Quantum physics is fascinating",
        )
        assert result is not None


class TestTextSummarizer:
    def test_basic(self):
        long_text = " ".join(["This is a sentence about programming."] * 20)
        result = TextSummarizer.static_call(text=long_text)
        assert "summary" in result or "error" not in result

    def test_short_text(self):
        result = TextSummarizer.static_call(text="Short text.")
        assert result is not None


class TestTextClassifier:
    def test_basic(self):
        result = TextClassifier.static_call(
            text="I love this product! It's amazing!",
            categories=["positive", "negative", "neutral"],
        )
        assert result is not None
        assert "classification" in result or "category" in result or "error" not in result

    def test_no_categories(self):
        result = TextClassifier.static_call(text="Some text")
        assert result is not None


class TestEntityExtractor:
    def test_basic(self):
        result = EntityExtractor.static_call(
            text="John Smith visited New York City on January 15th.",
        )
        assert "entities" in result or "error" not in result

    def test_empty_text(self):
        result = EntityExtractor.static_call(text="")
        assert result is not None
