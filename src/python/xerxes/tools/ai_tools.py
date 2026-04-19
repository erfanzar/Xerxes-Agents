# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""AI and machine learning tools for text processing and analysis.

This module provides a comprehensive set of AI-powered text processing
tools for the Xerxes framework. It includes:
- Text embedding generation using TF-IDF, sentence-transformers, or OpenAI
- Text similarity calculation with multiple metrics (cosine, Jaccard, Levenshtein, semantic)
- Text classification with keyword, sentiment, language, and topic detection
- Text summarization using extractive and keyword-based methods
- Named entity extraction for emails, URLs, phone numbers, dates, and more

All tools are implemented as AgentBaseFn subclasses for seamless integration
with Xerxes agents and support context_variables for runtime configuration.

Example:
    >>> from xerxes.tools.ai_tools import TextSummarizer, TextSimilarity
    >>> summary = TextSummarizer.static_call("Long article text...", method="extractive")
    >>> similarity = TextSimilarity.static_call("text one", "text two", method="cosine")
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..types import AgentBaseFn


class TextEmbedder(AgentBaseFn):
    """Generate text embeddings using various methods.

    Supports multiple embedding backends including TF-IDF, sentence-transformers,
    and OpenAI embeddings. Falls back to simple word frequency vectors when
    sklearn is not available.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Methods:
        static_call: Generate embeddings for one or more texts.
    """

    @staticmethod
    def static_call(
        text: str | list[str],
        method: str = "tfidf",
        model_name: str | None = None,
        max_length: int = 512,
        **context_variables,
    ) -> dict[str, Any]:
        """Generate text embeddings using the specified method.

        Converts one or more text strings into numerical vector representations.
        Supports TF-IDF (with sklearn fallback to word frequency), sentence-transformers
        for dense semantic embeddings, and OpenAI embedding API.

        Args:
            text: A single text string or a list of text strings to embed.
            method: Embedding method to use. Options:
                - "tfidf": TF-IDF vectorization via sklearn (falls back to word
                  frequency vectors if sklearn is not installed).
                - "sentence-transformers": Dense semantic embeddings using the
                  sentence-transformers library.
                - "openai": Embeddings via the OpenAI API (requires an OpenAI
                  client in context_variables).
            model_name: Model identifier for the embedding backend. Used by
                sentence-transformers (default: "all-MiniLM-L6-v2") and OpenAI
                (default: "text-embedding-ada-002"). Ignored for TF-IDF.
            max_length: Maximum number of characters per text. Texts longer
                than this are truncated before embedding.
            **context_variables: Runtime context from the agent. For the "openai"
                method, must contain an "openai_client" key with an initialized
                OpenAI client instance.

        Returns:
            A dictionary containing:
                - embeddings: List of embedding vectors (list of lists of floats).
                - shape: Tuple of (num_texts, embedding_dimension).
                - features: Top feature names (for TF-IDF method).
                - model: Model name used (for sentence-transformers and OpenAI).
                - usage: Token usage information (for OpenAI method).
                - error: Error message if the operation failed.

        Example:
            >>> result = TextEmbedder.static_call("Hello world", method="tfidf")
            >>> print(result["shape"])
            (1, 2)
        """
        result = {}

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        texts = [t[:max_length] for t in texts]

        if method == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                vectorizer = TfidfVectorizer(max_features=100)
                embeddings = vectorizer.fit_transform(texts).toarray()

                result["embeddings"] = embeddings.tolist()
                result["shape"] = embeddings.shape
                result["features"] = vectorizer.get_feature_names_out().tolist()[:20]

            except ImportError:
                all_words = []
                for t in texts:
                    all_words.extend(t.lower().split())

                word_freq = Counter(all_words)
                top_words = [w for w, _ in word_freq.most_common(50)]

                embeddings = []
                for t in texts:
                    vec = []
                    t_words = t.lower().split()
                    for word in top_words:
                        vec.append(t_words.count(word) / len(t_words) if t_words else 0)
                    embeddings.append(vec)

                result["embeddings"] = embeddings
                result["shape"] = (len(embeddings), len(top_words))
                result["features"] = top_words[:20]

        elif method == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
                embeddings = model.encode(texts)

                result["embeddings"] = embeddings.tolist()
                result["shape"] = embeddings.shape
                result["model"] = model_name or "all-MiniLM-L6-v2"

            except ImportError:
                return {"error": "sentence-transformers required. Install with: pip install xerxes[vectors]"}

        elif method == "openai":
            try:
                client = context_variables.get("openai_client")
                if not client:
                    return {"error": "OpenAI client required in context_variables"}

                response = client.embeddings.create(input=texts, model=model_name or "text-embedding-ada-002")

                embeddings = [e.embedding for e in response.data]
                result["embeddings"] = embeddings
                result["shape"] = (len(embeddings), len(embeddings[0]))
                result["model"] = model_name or "text-embedding-ada-002"
                result["usage"] = response.usage._asdict() if hasattr(response, "usage") else None

            except Exception as e:
                return {"error": f"OpenAI embedding failed: {e!s}"}

        else:
            return {"error": f"Unknown embedding method: {method}"}

        return result


class TextSimilarity(AgentBaseFn):
    """Calculate text similarity using various metrics.

    Provides multiple similarity calculation methods including cosine
    similarity, Jaccard index, Levenshtein distance, and semantic
    similarity using sentence embeddings.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Methods:
        static_call: Calculate similarity between two texts.
    """

    @staticmethod
    def static_call(
        text1: str,
        text2: str,
        method: str = "cosine",
        **context_variables,
    ) -> dict[str, Any]:
        """Calculate the similarity between two text strings.

        Computes a similarity score using the chosen metric. All methods
        produce a normalized score in the range [0, 1] (or [-1, 1] for
        semantic), along with a human-readable interpretation of the result.

        Args:
            text1: The first text to compare.
            text2: The second text to compare.
            method: Similarity metric to use. Options:
                - "cosine": Cosine similarity on word frequency vectors.
                  Scale: 0 to 1 (1 = identical).
                - "jaccard": Jaccard index on word sets (intersection / union).
                  Scale: 0 to 1. Also returns common words.
                - "levenshtein": Normalized Levenshtein edit distance.
                  Scale: 0 to 1 (1 = identical). Also returns raw distance.
                - "semantic": Cosine similarity on sentence-transformer
                  embeddings. Scale: -1 to 1. Requires the
                  sentence-transformers package.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - similarity (float): The computed similarity score.
                - method (str): The method used for comparison.
                - scale (str): Description of the score range.
                - interpretation (str): Human-readable strength label
                  ("Very high", "High", "Moderate", "Low", "Very low").
                - common_words (list[str]): Shared words (Jaccard only).
                - distance (int): Raw edit distance (Levenshtein only).
                - model (str): Embedding model used (semantic only).
                - error (str): Error message if the operation failed.

        Example:
            >>> result = TextSimilarity.static_call("hello world", "hello there")
            >>> print(result["similarity"])
            0.5
        """
        result = {}

        if method == "cosine":
            words1 = text1.lower().split()
            words2 = text2.lower().split()

            vocab = list(set(words1 + words2))

            vec1 = [words1.count(w) for w in vocab]
            vec2 = [words2.count(w) for w in vocab]

            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))

            if norm1 * norm2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm1 * norm2)

            result["similarity"] = similarity
            result["method"] = "cosine"
            result["scale"] = "0 to 1 (1 = identical)"

        elif method == "jaccard":
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())

            intersection = set1.intersection(set2)
            union = set1.union(set2)

            similarity = len(intersection) / len(union) if union else 0

            result["similarity"] = similarity
            result["method"] = "jaccard"
            result["scale"] = "0 to 1 (1 = identical)"
            result["common_words"] = list(intersection)[:20]

        elif method == "levenshtein":

            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)

                if len(s2) == 0:
                    return len(s1)

                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row

                return previous_row[-1]

            distance = levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            similarity = 1 - (distance / max_len) if max_len > 0 else 1

            result["similarity"] = similarity
            result["distance"] = distance
            result["method"] = "levenshtein"
            result["scale"] = "0 to 1 (1 = identical)"

        elif method == "semantic":
            try:
                from sentence_transformers import SentenceTransformer, util

                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode([text1, text2])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

                result["similarity"] = similarity
                result["method"] = "semantic"
                result["model"] = "all-MiniLM-L6-v2"
                result["scale"] = "-1 to 1 (1 = identical)"

            except ImportError:
                return {"error": "sentence-transformers required for semantic similarity"}

        else:
            return {"error": f"Unknown similarity method: {method}"}

        sim = result.get("similarity", 0)
        if sim > 0.9:
            result["interpretation"] = "Very high similarity"
        elif sim > 0.7:
            result["interpretation"] = "High similarity"
        elif sim > 0.5:
            result["interpretation"] = "Moderate similarity"
        elif sim > 0.3:
            result["interpretation"] = "Low similarity"
        else:
            result["interpretation"] = "Very low similarity"

        return result


class TextClassifier(AgentBaseFn):
    """Classify text into categories using various methods.

    Supports keyword-based classification, sentiment analysis,
    language detection, and topic classification. Uses simple
    heuristic methods that work without external ML dependencies.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Methods:
        static_call: Classify text into categories.
    """

    @staticmethod
    def static_call(
        text: str,
        categories: list[str] | None = None,
        method: str = "keyword",
        **context_variables,
    ) -> dict[str, Any]:
        """Classify text into categories using heuristic methods.

        Applies the selected classification method to determine the category,
        sentiment, language, or topic of the input text. All methods are
        lightweight and do not require external ML models.

        Args:
            text: The text to classify.
            categories: List of candidate category labels. Required when
                method is "keyword"; ignored for other methods.
            method: Classification method to use. Options:
                - "keyword": Match category labels against text content.
                  Requires the ``categories`` argument.
                - "sentiment": Simple lexicon-based sentiment analysis
                  returning positive, negative, or neutral.
                - "language": Detect the language of the text using common
                  word indicators (supports English, Spanish, French,
                  German, Italian).
                - "topic": Classify into predefined topics (technology,
                  business, science, health, education) using keyword
                  matching.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing method-specific results:
                For "keyword":
                    - category (str): Best matching category.
                    - confidence (float): Confidence score (0 to 1).
                    - scores (dict): Per-category match counts.
                For "sentiment":
                    - sentiment (str): "positive", "negative", or "neutral".
                    - confidence (float): Confidence score (0 to 1).
                    - positive_score (int): Count of positive word matches.
                    - negative_score (int): Count of negative word matches.
                For "language":
                    - language (str): Detected language name.
                    - confidence (float): Confidence score.
                    - scores (dict): Per-language match counts.
                For "topic":
                    - topic (str): Detected topic label.
                    - confidence (float): Confidence score (0 to 1).
                    - scores (dict): Per-topic match counts.
                - error (str): Error message if the operation failed.

        Example:
            >>> result = TextClassifier.static_call(
            ...     "The algorithm processes data efficiently",
            ...     method="topic"
            ... )
            >>> print(result["topic"])
            'technology'
        """
        result = {}

        if method == "keyword":
            if not categories:
                return {"error": "categories required for keyword classification"}

            scores = {}
            text_lower = text.lower()

            for category in categories:
                category_words = category.lower().split()
                score = sum(1 for word in category_words if word in text_lower)
                scores[category] = score

            if scores:
                top_category = max(scores, key=scores.get)
                result["category"] = top_category
                result["confidence"] = scores[top_category] / sum(scores.values()) if sum(scores.values()) > 0 else 0
                result["scores"] = scores
            else:
                result["category"] = "unknown"
                result["confidence"] = 0

        elif method == "sentiment":
            positive_words = [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "love",
                "best",
                "happy",
                "joy",
            ]
            negative_words = [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "hate",
                "worst",
                "sad",
                "angry",
                "frustrating",
                "disappointing",
            ]

            text_lower = text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)

            if positive_score > negative_score:
                sentiment = "positive"
                confidence = (
                    positive_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
                )
            elif negative_score > positive_score:
                sentiment = "negative"
                confidence = (
                    negative_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
                )
            else:
                sentiment = "neutral"
                confidence = 0.5

            result["sentiment"] = sentiment
            result["confidence"] = confidence
            result["positive_score"] = positive_score
            result["negative_score"] = negative_score

        elif method == "language":
            lang_indicators = {
                "english": ["the", "is", "and", "to", "of", "in", "that", "it", "with", "for"],
                "spanish": ["el", "la", "de", "que", "en", "los", "las", "por", "con", "para"],
                "french": ["le", "de", "la", "et", "les", "des", "en", "un", "une", "pour"],
                "german": ["der", "die", "und", "das", "ist", "den", "dem", "mit", "zu", "ein"],
                "italian": ["il", "di", "la", "che", "e", "le", "della", "per", "con", "del"],
            }

            text_words = text.lower().split()
            scores = {}

            for lang, indicators in lang_indicators.items():
                score = sum(1 for word in text_words if word in indicators)
                scores[lang] = score

            if scores:
                detected_lang = max(scores, key=scores.get)
                result["language"] = detected_lang
                result["confidence"] = scores[detected_lang] / len(text_words) if text_words else 0
                result["scores"] = scores
            else:
                result["language"] = "unknown"
                result["confidence"] = 0

        elif method == "topic":
            topics = {
                "technology": [
                    "computer",
                    "software",
                    "hardware",
                    "internet",
                    "digital",
                    "data",
                    "algorithm",
                    "programming",
                    "code",
                    "app",
                ],
                "business": [
                    "company",
                    "market",
                    "sales",
                    "revenue",
                    "profit",
                    "customer",
                    "product",
                    "service",
                    "management",
                    "strategy",
                ],
                "science": [
                    "research",
                    "study",
                    "experiment",
                    "hypothesis",
                    "theory",
                    "discovery",
                    "analysis",
                    "evidence",
                    "method",
                    "result",
                ],
                "health": [
                    "medical",
                    "health",
                    "doctor",
                    "patient",
                    "treatment",
                    "disease",
                    "medicine",
                    "hospital",
                    "symptom",
                    "diagnosis",
                ],
                "education": [
                    "student",
                    "teacher",
                    "school",
                    "learn",
                    "education",
                    "course",
                    "class",
                    "university",
                    "study",
                    "knowledge",
                ],
            }

            text_lower = text.lower()
            topic_scores = {}

            for topic, keywords in topics.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                topic_scores[topic] = score

            if topic_scores:
                top_topic = max(topic_scores, key=topic_scores.get)
                result["topic"] = top_topic
                result["confidence"] = (
                    topic_scores[top_topic] / sum(topic_scores.values()) if sum(topic_scores.values()) > 0 else 0
                )
                result["scores"] = topic_scores
            else:
                result["topic"] = "general"
                result["confidence"] = 0

        else:
            return {"error": f"Unknown classification method: {method}"}

        return result


class TextSummarizer(AgentBaseFn):
    """Summarize text using various techniques.

    Provides extractive summarization, keyword extraction, and
    statistical analysis of text. Uses sentence scoring based on
    word frequency for extractive summaries.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Methods:
        static_call: Generate a summary of the input text.
    """

    @staticmethod
    def static_call(
        text: str,
        method: str = "extractive",
        max_sentences: int = 3,
        max_length: int | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Generate a summary of the input text.

        Supports extractive summarization (selecting important sentences),
        keyword extraction (identifying key terms and phrases), and
        statistical analysis (computing text metrics).

        Args:
            text: The text to summarize.
            method: Summarization method to use. Options:
                - "extractive": Select the most important sentences based
                  on word frequency scoring. Returns a condensed version
                  of the original text.
                - "keywords": Extract the most frequent meaningful words
                  and bigram phrases from the text.
                - "statistics": Compute text statistics including word
                  count, sentence count, vocabulary richness, and
                  sentence length metrics.
            max_sentences: Maximum number of sentences to include in an
                extractive summary. Defaults to 3.
            max_length: Maximum character length for the summary output.
                If the summary exceeds this, it is truncated with "...".
                Only applies to the "extractive" method. None means no limit.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing method-specific results:
                For "extractive":
                    - summary (str): The extracted summary text.
                    - original_length (int): Character count of original text.
                    - summary_length (int): Character count of summary.
                    - compression_ratio (float): Summary length / original length.
                For "keywords":
                    - keywords (list[str]): Top 10 most frequent words.
                    - key_phrases (list[str]): Top 5 bigram phrases.
                    - summary (str): Brief description of key topics.
                For "statistics":
                    - summary (dict): Dictionary with total_characters,
                      total_words, unique_words, vocabulary_richness,
                      total_sentences, avg_sentence_length,
                      longest_sentence, shortest_sentence.
                - error (str): Error message if the operation failed.

        Example:
            >>> result = TextSummarizer.static_call(
            ...     "Long article text here...",
            ...     method="extractive",
            ...     max_sentences=2
            ... )
            >>> print(result["summary"])
        """
        result = {}

        if method == "extractive":
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return {"error": "No sentences found in text"}

            words = text.lower().split()
            word_freq = Counter(words)

            common_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "was",
                "are",
                "were",
            }
            word_freq = {w: f for w, f in word_freq.items() if w not in common_words}

            sentence_scores = []
            for sentence in sentences:
                score = 0
                words_in_sentence = sentence.lower().split()
                for word in words_in_sentence:
                    score += word_freq.get(word, 0)
                sentence_scores.append((sentence, score / len(words_in_sentence) if words_in_sentence else 0))

            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            summary_sentences = [s for s, _ in sentence_scores[:max_sentences]]

            summary = ". ".join(summary_sentences)
            if not summary.endswith("."):
                summary += "."

            if max_length and len(summary) > max_length:
                summary = summary[:max_length] + "..."

            result["summary"] = summary
            result["original_length"] = len(text)
            result["summary_length"] = len(summary)
            result["compression_ratio"] = len(summary) / len(text) if text else 0

        elif method == "keywords":
            words = text.lower().split()

            common_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "was",
                "are",
                "were",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
            }
            words = [w for w in words if w not in common_words and len(w) > 3]

            word_freq = Counter(words)
            keywords = [w for w, _ in word_freq.most_common(10)]

            bigrams = []
            words_list = text.lower().split()
            for i in range(len(words_list) - 1):
                if words_list[i] not in common_words and words_list[i + 1] not in common_words:
                    bigrams.append(f"{words_list[i]} {words_list[i + 1]}")

            bigram_freq = Counter(bigrams)
            key_phrases = [p for p, _ in bigram_freq.most_common(5)]

            result["keywords"] = keywords
            result["key_phrases"] = key_phrases
            result["summary"] = f"Key topics: {', '.join(keywords[:5])}"

        elif method == "statistics":
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            words = text.split()
            unique_words = set(w.lower() for w in words)

            result["summary"] = {
                "total_characters": len(text),
                "total_words": len(words),
                "unique_words": len(unique_words),
                "vocabulary_richness": len(unique_words) / len(words) if words else 0,
                "total_sentences": len(sentences),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "longest_sentence": max(len(s.split()) for s in sentences) if sentences else 0,
                "shortest_sentence": min(len(s.split()) for s in sentences) if sentences else 0,
            }

        else:
            return {"error": f"Unknown summarization method: {method}"}

        return result


class EntityExtractor(AgentBaseFn):
    """Extract named entities from text.

    Uses regex patterns to identify and extract various entity types
    including emails, URLs, phone numbers, dates, times, currency
    values, hashtags, mentions, and proper names.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Methods:
        static_call: Extract entities from the input text.
    """

    @staticmethod
    def static_call(
        text: str,
        entity_types: list[str] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Extract named entities from text using regex pattern matching.

        Scans the input text for various entity types using predefined
        regular expression patterns. Returns deduplicated matches for
        each requested entity type.

        Args:
            text: The text to extract entities from.
            entity_types: List of entity types to extract. If None, extracts
                all supported types. Supported types:
                - "emails": Email addresses.
                - "urls": HTTP/HTTPS URLs.
                - "phone_numbers": Phone numbers in various formats.
                - "dates": Dates in common formats (YYYY-MM-DD, MM/DD/YYYY, etc.).
                - "times": Time expressions (HH:MM, HH:MM:SS, with optional AM/PM).
                - "numbers": Integer and decimal numbers.
                - "hashtags": Hashtag expressions (#word).
                - "mentions": At-mention expressions (@user).
                - "currency": Currency values ($, EUR, GBP, JPY prefixed).
                - "names": Proper names (capitalized multi-word sequences).
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - entities (dict[str, list[str]]): Mapping of entity type
                  to a list of unique extracted values (max 20 per type).
                - total_entities (int): Total number of extracted entities
                  across all types.

        Example:
            >>> result = EntityExtractor.static_call(
            ...     "Contact john@example.com or visit https://example.com",
            ...     entity_types=["emails", "urls"]
            ... )
            >>> print(result["entities"]["emails"])
            ['john@example.com']
        """
        result = {"entities": {}}

        patterns = {
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "urls": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "phone_numbers": r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}",
            "dates": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            "times": r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b",
            "numbers": r"\b\d+(?:\.\d+)?\b",
            "hashtags": r"#\w+",
            "mentions": r"@\w+",
            "currency": r"[$€£¥][\d,]+(?:\.\d{2})?",
        }

        name_pattern = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b"

        if not entity_types:
            entity_types = [*list(patterns.keys()), "names"]

        for entity_type in entity_types:
            if entity_type == "names":
                matches = re.findall(name_pattern, text)
                result["entities"]["names"] = list(set(matches))[:20]
            elif entity_type in patterns:
                matches = re.findall(patterns[entity_type], text)
                result["entities"][entity_type] = list(set(matches))[:20]

        total = sum(len(v) for v in result["entities"].values())
        result["total_entities"] = total

        return result
