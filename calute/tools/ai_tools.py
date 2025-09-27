# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""AI and machine learning tools for text processing and analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..types import AgentBaseFn


class TextEmbedder(AgentBaseFn):
    """Generate text embeddings using various methods."""

    @staticmethod
    def static_call(
        text: str | list[str],
        method: str = "tfidf",
        model_name: str | None = None,
        max_length: int = 512,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Generate text embeddings.

        Args:
            text: Text or list of texts to embed
            method: Embedding method (tfidf, sentence-transformers, openai)
            model_name: Model name for embedding
            max_length: Maximum text length

        Returns:
            Embeddings and metadata
        """
        result = {}

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        texts = [t[:max_length] for t in texts]

        if method == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

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
                from sentence_transformers import SentenceTransformer  # type:ignore

                model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
                embeddings = model.encode(texts)

                result["embeddings"] = embeddings.tolist()
                result["shape"] = embeddings.shape
                result["model"] = model_name or "all-MiniLM-L6-v2"

            except ImportError:
                return {"error": "sentence-transformers required. Install with: pip install calute[vectors]"}

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
    """Calculate text similarity using various metrics."""

    @staticmethod
    def static_call(
        text1: str,
        text2: str,
        method: str = "cosine",
        **context_variables,
    ) -> dict[str, Any]:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            method: Similarity method (cosine, jaccard, levenshtein, semantic)

        Returns:
            Similarity scores and analysis
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
                from sentence_transformers import SentenceTransformer, util  # type:ignore

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
    """Classify text into categories using various methods."""

    @staticmethod
    def static_call(
        text: str,
        categories: list[str] | None = None,
        method: str = "keyword",
        **context_variables,
    ) -> dict[str, Any]:
        """
        Classify text into categories.

        Args:
            text: Text to classify
            categories: List of possible categories
            method: Classification method (keyword, sentiment, language, topic)

        Returns:
            Classification results
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
    """Summarize text using various techniques."""

    @staticmethod
    def static_call(
        text: str,
        method: str = "extractive",
        max_sentences: int = 3,
        max_length: int | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Summarize text.

        Args:
            text: Text to summarize
            method: Summarization method (extractive, keywords, statistics)
            max_sentences: Maximum sentences in summary
            max_length: Maximum character length

        Returns:
            Summary and metadata
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
    """Extract named entities from text."""

    @staticmethod
    def static_call(
        text: str,
        entity_types: list[str] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Extract entities from text.

        Args:
            text: Text to process
            entity_types: Types of entities to extract

        Returns:
            Extracted entities
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
