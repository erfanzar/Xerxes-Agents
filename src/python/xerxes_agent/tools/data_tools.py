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


"""Data processing and manipulation tools for Xerxes agents.

This module provides a comprehensive set of data processing tools
for the Xerxes framework. It includes:
- JSON data processing with load, save, query, and validation operations
- CSV file processing with read, write, analyze, and convert capabilities
- Advanced text processing with statistics, extraction, and formatting
- Data format conversion between JSON, YAML, Base64, Hex, and hashes
- Date and time processing with parsing, formatting, and delta calculations

Each tool is implemented as a class inheriting from AgentBaseFn,
making them directly usable as agent tools for data manipulation tasks.

Example:
    >>> processor = JSONProcessor()
    >>> result = processor(operation="load", file_path="data.json")
    >>> print(result["data"])
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any

from ..types import AgentBaseFn


class JSONProcessor(AgentBaseFn):
    """JSON data processing and manipulation tool.

    Provides operations for loading, saving, validating, querying,
    and transforming JSON data. Supports both file-based and in-memory
    JSON operations with simple dot-notation queries.

    Supported operations:
        load: Load JSON data from a file.
        save: Save JSON data to a file.
        validate: Check if data is valid JSON.
        query: Extract data using dot-notation paths (e.g., "user.name").
        transform: Get metadata and formatted output of JSON data.
    """

    @staticmethod
    def static_call(
        operation: str,
        data: Any = None,
        file_path: str | None = None,
        query: str | None = None,
        pretty: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Process JSON data with various operations.

        Performs load, save, validate, query, or transform operations on
        JSON data. Supports both file-based and in-memory JSON manipulation.

        Args:
            operation: The operation to perform. Options:
                - "load": Load JSON from a file. Requires ``file_path``.
                - "save": Save data to a JSON file. Requires ``file_path``
                  and ``data``.
                - "validate": Check if ``data`` is valid JSON (accepts
                  both string and object inputs).
                - "query": Extract a value from ``data`` using dot-notation
                  paths (e.g., "user.name", "items[0].id"). Requires
                  ``query`` and ``data``.
                - "transform": Get metadata about ``data`` including type,
                  keys, length, and optionally pretty-printed output.
            data: The JSON data to process. Can be a Python dict/list or
                a JSON string (for validate). Required for save, validate,
                query, and transform operations.
            file_path: Path to the JSON file for load/save operations.
            query: Dot-notation query path for data extraction. Supports
                bracket notation for array indexing (e.g., "items[0]").
            pretty: Whether to use indented formatting when saving or
                transforming JSON. Defaults to True.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing operation-specific results:
                For "load": data, success.
                For "save": success, file_path.
                For "validate": valid (bool), error (if invalid).
                For "query": result (extracted value).
                For "transform": keys, type, length, formatted (if pretty).
                - error (str): Error message if the operation failed.

        Example:
            >>> result = JSONProcessor.static_call("validate", data='{"key": 1}')
            >>> print(result["valid"])
            True
        """
        result = {}

        if operation == "load":
            if not file_path:
                return {"error": "file_path required for load operation"}
            try:
                with open(file_path, "r") as f:
                    result["data"] = json.load(f)
                result["success"] = True
            except Exception as e:
                return {"error": f"Failed to load JSON: {e!s}"}

        elif operation == "save":
            if not file_path or data is None:
                return {"error": "file_path and data required for save operation"}
            try:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2 if pretty else None)
                result["success"] = True
                result["file_path"] = file_path
            except Exception as e:
                return {"error": f"Failed to save JSON: {e!s}"}

        elif operation == "validate":
            try:
                if isinstance(data, str):
                    json.loads(data)
                else:
                    json.dumps(data)
                result["valid"] = True
            except Exception as e:
                result["valid"] = False
                result["error"] = str(e)

        elif operation == "query":
            if not query or data is None:
                return {"error": "query and data required for query operation"}
            try:
                parts = query.split(".")
                current = data
                for part in parts:
                    if "[" in part and "]" in part:
                        key = part[: part.index("[")]
                        index = int(part[part.index("[") + 1 : part.index("]")])
                        current = current[key][index] if key else current[index]
                    else:
                        current = current[part]
                result["result"] = current
            except Exception as e:
                return {"error": f"Query failed: {e!s}"}

        elif operation == "transform":
            if data:
                result["keys"] = list(data.keys()) if isinstance(data, dict) else None
                result["type"] = type(data).__name__
                result["length"] = len(data) if hasattr(data, "__len__") else None
                if pretty:
                    result["formatted"] = json.dumps(data, indent=2)

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class CSVProcessor(AgentBaseFn):
    """CSV data processing and manipulation tool.

    Provides operations for reading, writing, analyzing, and converting
    CSV files. Supports custom delimiters, headers, and row limits.

    Supported operations:
        read: Read CSV file into a list of dictionaries.
        write: Write list of dictionaries to a CSV file.
        analyze: Get statistics about a CSV file structure.
        convert: Convert CSV data to JSON format.
    """

    @staticmethod
    def static_call(
        operation: str,
        file_path: str | None = None,
        data: list[dict] | None = None,
        delimiter: str = ",",
        headers: list[str] | None = None,
        has_header: bool = True,
        max_rows: int | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Process CSV data with various operations.

        Performs read, write, analyze, or convert operations on CSV files.
        Supports custom delimiters, header configuration, and row limits.

        Args:
            operation: The operation to perform. Options:
                - "read": Read a CSV file into a list of dictionaries.
                  Requires ``file_path``.
                - "write": Write a list of dictionaries to a CSV file.
                  Requires ``file_path`` and ``data``.
                - "analyze": Get structural statistics about a CSV file
                  including row/column counts, headers, sample data, and
                  empty cell count. Requires ``file_path``.
                - "convert": Convert a CSV file to a list of JSON-like
                  dictionaries. Requires ``file_path``.
            file_path: Path to the CSV file for read/write/analyze/convert.
            data: List of dictionaries to write. Each dict represents a row
                with column names as keys. Required for the "write" operation.
            delimiter: Column delimiter character. Defaults to ",".
            headers: Explicit column headers. For "write", used as fieldnames;
                if not provided, inferred from the first data dict. For "read"
                with ``has_header=False``, used as the column names.
            has_header: Whether the CSV file's first row is a header row.
                If False and no ``headers`` are provided, columns are
                auto-named as "col_0", "col_1", etc. Defaults to True.
            max_rows: Maximum number of rows to read. None reads all rows.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing operation-specific results:
                For "read": data (list[dict]), count (int), columns (list[str]).
                For "write": success (bool), rows_written (int), file_path (str).
                For "analyze": total_rows, total_columns, headers, sample_data,
                    empty_cells.
                For "convert": json (list[dict]), count (int).
                - error (str): Error message if the operation failed.

        Example:
            >>> result = CSVProcessor.static_call("read", file_path="data.csv", max_rows=5)
            >>> print(result["count"])
            5
        """
        result = {}

        if operation == "read":
            if not file_path:
                return {"error": "file_path required for read operation"}
            try:
                rows = []
                with open(file_path, "r", newline="", encoding="utf-8") as f:
                    fieldnames = None
                    if not has_header:
                        if headers:
                            fieldnames = headers
                        else:
                            first_line = f.readline()
                            col_count = len(first_line.split(delimiter))
                            fieldnames = [f"col_{i}" for i in range(col_count)]
                            f.seek(0)
                    reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter)
                    for i, row in enumerate(reader):
                        if max_rows and i >= max_rows:
                            break
                        rows.append(row)
                result["data"] = rows
                result["count"] = len(rows)
                if rows:
                    result["columns"] = list(rows[0].keys())
            except Exception as e:
                return {"error": f"Failed to read CSV: {e!s}"}

        elif operation == "write":
            if not file_path or not data:
                return {"error": "file_path and data required for write operation"}
            try:
                if not headers and data:
                    headers = list(data[0].keys())

                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(data)
                result["success"] = True
                result["rows_written"] = len(data)
                result["file_path"] = file_path
            except Exception as e:
                return {"error": f"Failed to write CSV: {e!s}"}

        elif operation == "analyze":
            if not file_path:
                return {"error": "file_path required for analyze operation"}
            try:
                with open(file_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    rows = list(reader)

                result["total_rows"] = len(rows)
                result["total_columns"] = len(rows[0]) if rows else 0

                if rows:
                    result["headers"] = rows[0]
                    result["sample_data"] = rows[1 : min(6, len(rows))]

                    result["empty_cells"] = sum(1 for row in rows[1:] for cell in row if not cell.strip())

            except Exception as e:
                return {"error": f"Failed to analyze CSV: {e!s}"}

        elif operation == "convert":
            if not file_path:
                return {"error": "file_path required for convert operation"}
            try:
                rows = []
                with open(file_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows = list(reader)
                result["json"] = rows
                result["count"] = len(rows)
            except Exception as e:
                return {"error": f"Failed to convert CSV: {e!s}"}

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class TextProcessor(AgentBaseFn):
    """Advanced text processing and manipulation tool.

    Provides operations for analyzing, cleaning, extracting patterns,
    replacing content, and formatting text. Supports regular expressions
    for pattern matching and extraction.

    Supported operations:
        stats: Get text statistics (length, word count, character frequency).
        clean: Remove extra whitespace and optionally matched patterns.
        extract: Extract patterns like emails, URLs, phone numbers, or custom regex.
        replace: Replace patterns in text using regex.
        split: Split text by pattern or whitespace.
        format: Apply formatting (title, upper, lower, sentence case).
    """

    @staticmethod
    def static_call(
        text: str,
        operation: str,
        pattern: str | None = None,
        replacement: str | None = None,
        case_sensitive: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Process text with various operations.

        Applies the specified text processing operation, ranging from
        statistical analysis to pattern-based extraction and formatting.

        Args:
            text: The input text to process.
            operation: The operation to perform. Options:
                - "stats": Compute text statistics including length, word
                  count, line count, character frequency, and word frequency.
                - "clean": Remove extra whitespace and optionally remove
                  content matching ``pattern``.
                - "extract": Extract patterns from text. ``pattern`` can be
                  a named shortcut ("emails", "urls", "phones", "numbers")
                  or a custom regular expression.
                - "replace": Replace occurrences of ``pattern`` in text with
                  ``replacement``. Uses regex matching.
                - "split": Split text by ``pattern`` (regex) or by
                  whitespace if no pattern is given.
                - "format": Apply text formatting. ``pattern`` specifies the
                  format: "title", "upper", "lower", "sentence", or
                  "no_punctuation".
            pattern: Regex pattern or named shortcut for extract/replace/split/
                format operations. Required for "extract" and "replace".
            replacement: Replacement string for the "replace" operation.
                Defaults to empty string if None.
            case_sensitive: Whether pattern matching is case-sensitive.
                Defaults to True.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing operation-specific results:
                For "stats": length, words, lines, characters_no_spaces,
                    most_common_chars, most_common_words.
                For "clean": cleaned_text, original_length, cleaned_length.
                For "extract": matches (list[str]), count (int).
                For "replace": replaced_text, replacements_made (int).
                For "split": parts (list[str]), count (int).
                For "format": formatted_text (str).
                - error (str): Error message if the operation failed.

        Example:
            >>> result = TextProcessor.static_call("Hello World!", "stats")
            >>> print(result["words"])
            2
        """
        result = {}

        if operation == "stats":
            result["length"] = len(text)
            result["words"] = len(text.split())
            result["lines"] = len(text.splitlines())
            result["characters_no_spaces"] = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

            char_freq = {}
            for char in text.lower():
                if char.isalpha():
                    char_freq[char] = char_freq.get(char, 0) + 1
            result["most_common_chars"] = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:5]

            words = re.findall(r"\b\w+\b", text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            result["most_common_words"] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        elif operation == "clean":
            cleaned = text

            cleaned = re.sub(r"\s+", " ", cleaned)

            if pattern:
                cleaned = re.sub(pattern, "", cleaned)
            cleaned = cleaned.strip()
            result["cleaned_text"] = cleaned
            result["original_length"] = len(text)
            result["cleaned_length"] = len(cleaned)

        elif operation == "extract":
            if not pattern:
                return {"error": "pattern required for extract operation"}

            flags = 0 if case_sensitive else re.IGNORECASE

            if pattern == "emails":
                pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            elif pattern == "urls":
                pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            elif pattern == "phones":
                pattern = r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}"
            elif pattern == "numbers":
                pattern = r"-?\d+\.?\d*"

            matches = re.findall(pattern, text, flags)
            result["matches"] = matches
            result["count"] = len(matches)

        elif operation == "replace":
            if not pattern:
                return {"error": "pattern required for replace operation"}
            if replacement is None:
                replacement = ""

            flags = 0 if case_sensitive else re.IGNORECASE
            replaced = re.sub(pattern, replacement, text, flags=flags)
            result["replaced_text"] = replaced
            result["replacements_made"] = len(re.findall(pattern, text, flags))

        elif operation == "split":
            if pattern:
                parts = re.split(pattern, text)
            else:
                parts = text.split()
            result["parts"] = parts
            result["count"] = len(parts)

        elif operation == "format":
            formatted = text

            if pattern == "title":
                formatted = text.title()

            elif pattern == "upper":
                formatted = text.upper()

            elif pattern == "lower":
                formatted = text.lower()

            elif pattern == "sentence":
                formatted = ". ".join(s.capitalize() for s in text.split(". "))

            elif pattern == "no_punctuation":
                formatted = re.sub(r"[^\w\s]", "", text)

            result["formatted_text"] = formatted

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class DataConverter(AgentBaseFn):
    """Convert data between different formats.

    Provides conversion between various data formats including
    JSON, YAML, Base64, hexadecimal, and cryptographic hashes.
    Supports bidirectional conversion where applicable.

    Supported formats:
        json: JSON string format.
        yaml: YAML format (requires PyYAML).
        base64: Base64 encoded string.
        hex: Hexadecimal string representation.
        hash: Generate MD5, SHA1, SHA256, and SHA512 hashes (output only).
    """

    @staticmethod
    def static_call(
        data: Any,
        from_format: str,
        to_format: str,
        encoding: str = "utf-8",
        **context_variables,
    ) -> dict[str, Any]:
        """Convert data between different formats.

        First parses the input data from the source format into an
        intermediate Python object, then serializes it to the target format.

        Args:
            data: Input data to convert. Can be a string (for json, yaml,
                base64, hex source formats) or a Python object (dict, list).
            from_format: Source format of the data. Options:
                - "json": JSON string or Python dict/list.
                - "yaml": YAML string or Python object. Requires PyYAML.
                - "base64": Base64-encoded string.
                - "hex": Hexadecimal-encoded string.
            to_format: Target format to convert to. Options:
                - "json": Pretty-printed JSON string.
                - "yaml": YAML string. Requires PyYAML.
                - "base64": Base64-encoded string.
                - "hex": Hexadecimal string.
                - "hash": Dictionary of cryptographic hashes (MD5, SHA1,
                  SHA256, SHA512). Output only.
            encoding: Character encoding for encoding/decoding operations.
                Defaults to "utf-8".
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - output: The converted data in the target format. For
                  "hash" target, this is a dict with md5, sha1, sha256,
                  and sha512 hex digest strings.
                - success (bool): True if conversion succeeded.
                - error (str): Error message if the conversion failed.

        Example:
            >>> result = DataConverter.static_call(
            ...     '{"key": "value"}', from_format="json", to_format="base64"
            ... )
            >>> print(result["success"])
            True
        """
        result = {}

        try:
            parsed_data = None

            if from_format == "json":
                if isinstance(data, str):
                    parsed_data = json.loads(data)
                else:
                    parsed_data = data

            elif from_format == "yaml":
                try:
                    import yaml

                    if isinstance(data, str):
                        parsed_data = yaml.safe_load(data)
                    else:
                        parsed_data = data
                except ImportError:
                    return {"error": "PyYAML required for YAML operations"}

            elif from_format == "base64":
                if isinstance(data, str):
                    parsed_data = base64.b64decode(data).decode(encoding)
                else:
                    return {"error": "Base64 input must be string"}

            elif from_format == "hex":
                if isinstance(data, str):
                    parsed_data = bytes.fromhex(data).decode(encoding)
                else:
                    return {"error": "Hex input must be string"}

            else:
                parsed_data = data

            if to_format == "json":
                result["output"] = json.dumps(parsed_data, indent=2)

            elif to_format == "yaml":
                try:
                    import yaml

                    result["output"] = yaml.dump(parsed_data, default_flow_style=False)
                except ImportError:
                    return {"error": "PyYAML required for YAML operations"}

            elif to_format == "base64":
                if isinstance(parsed_data, str):
                    result["output"] = base64.b64encode(parsed_data.encode(encoding)).decode("ascii")
                else:
                    result["output"] = base64.b64encode(json.dumps(parsed_data).encode(encoding)).decode("ascii")

            elif to_format == "hex":
                if isinstance(parsed_data, str):
                    result["output"] = parsed_data.encode(encoding).hex()
                else:
                    result["output"] = json.dumps(parsed_data).encode(encoding).hex()

            elif to_format == "hash":
                if not isinstance(parsed_data, str):
                    parsed_data = json.dumps(parsed_data)
                data_bytes = parsed_data.encode(encoding)
                result["output"] = {
                    "md5": hashlib.md5(data_bytes).hexdigest(),
                    "sha1": hashlib.sha1(data_bytes).hexdigest(),
                    "sha256": hashlib.sha256(data_bytes).hexdigest(),
                    "sha512": hashlib.sha512(data_bytes).hexdigest(),
                }

            else:
                return {"error": f"Unknown target format: {to_format}"}

            result["success"] = True

        except Exception as e:
            return {"error": f"Conversion failed: {e!s}"}

        return result


class DateTimeProcessor(AgentBaseFn):
    """Date and time processing utilities.

    Provides operations for parsing, formatting, and manipulating
    dates and times. Supports multiple date formats and time delta
    calculations.

    Supported operations:
        now: Get current date and time in various formats.
        parse: Parse a date string into components.
        delta: Add or subtract time from a date.
        format: Format a date in various output styles.
    """

    @staticmethod
    def static_call(
        operation: str,
        date_string: str | None = None,
        format: str | None = None,  # noqa: A002
        timezone: str | None = None,
        delta_days: int = 0,
        delta_hours: int = 0,
        delta_minutes: int = 0,
        **context_variables,
    ) -> dict[str, Any]:
        """Process dates and times with various operations.

        Provides operations for getting the current time, parsing date
        strings, computing time deltas, and formatting dates in various
        output styles.

        Args:
            operation: The operation to perform. Options:
                - "now": Get current date and time in multiple formats.
                - "parse": Parse a date string into components. Tries
                  common formats automatically; use ``format`` for a
                  specific strptime format. Falls back to dateutil if
                  available.
                - "delta": Add or subtract time from a date. Uses
                  ``date_string`` as the base (defaults to now).
                - "format": Format a date in various output styles. Uses
                  ``date_string`` as input (defaults to now). If ``format``
                  is provided, uses it as a strftime pattern; otherwise
                  returns all common formats.
            date_string: Date string to parse, use as base for delta, or
                format. Expected to be in ISO format for delta/format
                operations. For parse, accepts many common formats.
            format: Explicit strftime/strptime format pattern. For "parse",
                used as the preferred parsing format. For "format", used as
                the output format pattern.
            timezone: Timezone name. Currently reserved for future use.
            delta_days: Number of days to add (positive) or subtract
                (negative) from the base date. Defaults to 0.
            delta_hours: Number of hours to add or subtract. Defaults to 0.
            delta_minutes: Number of minutes to add or subtract. Defaults to 0.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing operation-specific results:
                For "now": datetime (ISO), timestamp, formatted (dict with
                    date, time, datetime, iso, human keys).
                For "parse": parsed (ISO), timestamp, components (dict with
                    year, month, day, hour, minute, second, weekday).
                For "delta": original (ISO), new (ISO), delta (dict with
                    days, hours, minutes, total_seconds).
                For "format": formats (dict of format name to value) or
                    formatted (str) when a specific format is provided.
                - error (str): Error message if the operation failed.

        Example:
            >>> result = DateTimeProcessor.static_call("parse", date_string="2024-01-15")
            >>> print(result["components"]["weekday"])
            'Monday'
        """
        result = {}

        if operation == "now":
            now = datetime.now()
            result["datetime"] = now.isoformat()
            result["timestamp"] = now.timestamp()
            result["formatted"] = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "iso": now.isoformat(),
                "human": now.strftime("%B %d, %Y at %I:%M %p"),
            }

        elif operation == "parse":
            if not date_string:
                return {"error": "date_string required for parse operation"}

            try:
                formats = [
                    "%Y-%m-%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]

                if format:
                    formats.insert(0, format)

                parsed_date = None
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_string, fmt)
                        break
                    except Exception:
                        continue

                if not parsed_date:
                    try:
                        from dateutil import parser  # type:ignore

                        parsed_date = parser.parse(date_string)
                    except Exception:
                        return {"error": "Could not parse date string"}

                result["parsed"] = parsed_date.isoformat()
                result["timestamp"] = parsed_date.timestamp()
                result["components"] = {
                    "year": parsed_date.year,
                    "month": parsed_date.month,
                    "day": parsed_date.day,
                    "hour": parsed_date.hour,
                    "minute": parsed_date.minute,
                    "second": parsed_date.second,
                    "weekday": parsed_date.strftime("%A"),
                }

            except Exception as e:
                return {"error": f"Failed to parse date: {e!s}"}

        elif operation == "delta":
            base_date = datetime.now()
            if date_string:
                try:
                    base_date = datetime.fromisoformat(date_string)
                except Exception:
                    return {"error": "Invalid date_string for delta operation"}

            delta = timedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
            new_date = base_date + delta

            result["original"] = base_date.isoformat()
            result["new"] = new_date.isoformat()
            result["delta"] = {
                "days": delta_days,
                "hours": delta_hours,
                "minutes": delta_minutes,
                "total_seconds": delta.total_seconds(),
            }

        elif operation == "format":
            if not date_string:
                date_string = datetime.now().isoformat()

            try:
                dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))

                if not format:
                    result["formats"] = {
                        "iso": dt.isoformat(),
                        "date": dt.strftime("%Y-%m-%d"),
                        "time": dt.strftime("%H:%M:%S"),
                        "us": dt.strftime("%m/%d/%Y"),
                        "eu": dt.strftime("%d/%m/%Y"),
                        "human": dt.strftime("%B %d, %Y at %I:%M %p"),
                        "short": dt.strftime("%b %d, %Y"),
                        "timestamp": dt.timestamp(),
                    }
                else:
                    result["formatted"] = dt.strftime(format)

            except Exception as e:
                return {"error": f"Failed to format date: {e!s}"}

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result
