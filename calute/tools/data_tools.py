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

"""Data processing and manipulation tools."""

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
    """JSON data processing and manipulation."""

    @staticmethod
    def static_call(
        operation: str,
        data: Any = None,
        file_path: str | None = None,
        query: str | None = None,
        pretty: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Process JSON data with various operations.

        Args:
            operation: Operation to perform (load, save, query, validate, transform)
            data: JSON data to process
            file_path: Path to JSON file
            query: JSONPath or simple query for data extraction
            pretty: Whether to format output prettily

        Returns:
            Processed JSON data or operation result
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
    """CSV data processing and manipulation."""

    @staticmethod
    def static_call(
        operation: str,
        file_path: str | None = None,
        data: list[dict] | None = None,
        delimiter: str = ",",
        headers: list[str] | None = None,
        max_rows: int | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Process CSV data with various operations.

        Args:
            operation: Operation to perform (read, write, analyze, convert)
            file_path: Path to CSV file
            data: Data to write (list of dictionaries)
            delimiter: CSV delimiter
            headers: Column headers for writing
            max_rows: Maximum rows to read

        Returns:
            Processed CSV data or operation result
        """
        result = {}

        if operation == "read":
            if not file_path:
                return {"error": "file_path required for read operation"}
            try:
                rows = []
                with open(file_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
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
    """Advanced text processing and manipulation."""

    @staticmethod
    def static_call(
        text: str,
        operation: str,
        pattern: str | None = None,
        replacement: str | None = None,
        case_sensitive: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """
        Process text with various operations.

        Args:
            text: Input text to process
            operation: Operation to perform (stats, clean, extract, replace, split, format)
            pattern: Regex pattern for operations
            replacement: Replacement text for replace operation
            case_sensitive: Whether operations are case-sensitive

        Returns:
            Processed text or analysis results
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
    """Convert data between different formats."""

    @staticmethod
    def static_call(
        data: Any,
        from_format: str,
        to_format: str,
        encoding: str = "utf-8",
        **context_variables,
    ) -> dict[str, Any]:
        """
        Convert data between formats.

        Args:
            data: Input data to convert
            from_format: Source format (json, csv, yaml, xml, base64, hex)
            to_format: Target format
            encoding: Character encoding

        Returns:
            Converted data
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
    """Date and time processing utilities."""

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
        """
        Process dates and times.

        Args:
            operation: Operation to perform (parse, format, now, delta, compare)
            date_string: Date string to parse
            format: Date format string
            timezone: Timezone name
            delta_days: Days to add/subtract
            delta_hours: Hours to add/subtract
            delta_minutes: Minutes to add/subtract

        Returns:
            Processed date/time information
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
