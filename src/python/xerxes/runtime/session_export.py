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
"""Export persisted daemon sessions as complete trace artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.paths import xerxes_subdir

EXPORT_SCHEMA = "xerxes.session.export.v1"
DEFAULT_EXPORT_FORMAT = "json"
LOVELY_PIRATE_FORMAT = "lovely-pirate"
EXPORT_FORMATS = ("json", "jsonl", "md", LOVELY_PIRATE_FORMAT)


class SessionExportError(RuntimeError):
    """Raised when a session export request cannot be satisfied."""


@dataclass(frozen=True)
class SavedSession:
    """A persisted daemon session record plus its source path."""

    path: Path
    record: dict[str, Any]
    mtime: float


def default_session_store_dir() -> Path:
    """Return the daemon session directory used by ``xerxes -r``."""
    return xerxes_subdir("sessions")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def _record_project_dir(record: dict[str, Any]) -> Path | None:
    raw = record.get("cwd") or record.get("project_dir") or ""
    if not raw:
        return None
    return _resolve_path(str(raw))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content or "")


def _metadata_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _title_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = " ".join(_message_text(message.get("content")).split())
        if text:
            return text[:77] + "..." if len(text) > 80 else text
    return ""


def _record_title(record: dict[str, Any]) -> str:
    metadata = _metadata_dict(record.get("metadata"))
    title = str(metadata.get("title") or "").strip()
    if title:
        return title
    return _title_from_messages(record.get("messages", []))


def _record_message_count(record: dict[str, Any]) -> int:
    messages = record.get("messages", [])
    return len(messages) if isinstance(messages, list) else 0


def _record_turn_count(record: dict[str, Any]) -> int:
    try:
        return int(record.get("turn_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _record_has_history(record: dict[str, Any]) -> bool:
    return _record_message_count(record) > 0 or _record_turn_count(record) > 0


def list_saved_sessions(
    *,
    store_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> list[SavedSession]:
    """Return saved daemon sessions newest first.

    Args:
        store_dir: Optional override for the persisted session directory.
        project_dir: Optional project directory filter. Matching is done
            against the persisted ``cwd``/``project_dir`` fields.

    Returns:
        Saved sessions sorted by persisted ``updated_at`` and filesystem mtime.
    """
    root = _resolve_path(store_dir) if store_dir else default_session_store_dir()
    if not root.exists():
        return []
    project = _resolve_path(project_dir) if project_dir else None
    sessions: list[SavedSession] = []
    for path in root.glob("*.json"):
        if path.name.startswith("."):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not _record_has_history(record):
            continue
        if project is not None and _record_project_dir(record) != project:
            continue
        sessions.append(SavedSession(path=path, record=record, mtime=mtime))
    sessions.sort(key=lambda saved: (str(saved.record.get("updated_at") or ""), saved.mtime), reverse=True)
    return sessions


def saved_session_summary(saved: SavedSession) -> dict[str, Any]:
    """Return concise metadata for a saved session."""
    record = saved.record
    session_id = str(record.get("session_id") or saved.path.stem)
    return {
        "id": session_id,
        "session_id": session_id,
        "key": str(record.get("key") or ""),
        "title": _record_title(record),
        "agent_id": str(record.get("agent_id") or ""),
        "project_dir": str(_record_project_dir(record) or ""),
        "updated_at": str(record.get("updated_at") or ""),
        "turn_count": _record_turn_count(record),
        "messages": _record_message_count(record),
        "path": str(saved.path),
    }


def select_saved_session(
    query: str = "",
    *,
    store_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> SavedSession:
    """Find one saved session by id/key/title, or return the latest match."""
    sessions = list_saved_sessions(store_dir=store_dir, project_dir=project_dir)
    if not sessions:
        scope = f" for project {_resolve_path(project_dir)}" if project_dir else ""
        raise SessionExportError(f"No saved Xerxes sessions found{scope}.")

    needle = query.strip()
    if not needle:
        return sessions[0]

    lower = needle.lower()
    exact: list[SavedSession] = []
    prefix: list[SavedSession] = []
    for saved in sessions:
        summary = saved_session_summary(saved)
        values = [
            str(summary.get("id") or ""),
            str(summary.get("session_id") or ""),
            str(summary.get("key") or ""),
            str(summary.get("title") or ""),
        ]
        if any(needle == value or lower == value.lower() for value in values):
            exact.append(saved)
        elif any(value.startswith(needle) or value.lower().startswith(lower) for value in values):
            prefix.append(saved)

    matches = exact or prefix
    if not matches:
        scope = f" in project {_resolve_path(project_dir)}" if project_dir else ""
        raise SessionExportError(f"No saved Xerxes session matched `{needle}`{scope}.")
    if len(matches) > 1:
        ids = ", ".join(str(saved_session_summary(item)["id"]) for item in matches[:8])
        raise SessionExportError(f"Session query `{needle}` matched multiple sessions: {ids}")
    return matches[0]


def _archive_path_for(record_path: Path) -> Path:
    return record_path.with_name(f"{record_path.stem}.archive.jsonl")


def _read_archive_messages(record_path: Path) -> list[dict[str, Any]]:
    archive_path = _archive_path_for(record_path)
    if not archive_path.exists():
        return []
    messages: list[dict[str, Any]] = []
    try:
        lines = archive_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            messages.append(value)
    return messages


def build_session_export(saved: SavedSession, *, include_archive: bool = True) -> dict[str, Any]:
    """Build a complete JSON-serialisable session export."""
    record = saved.record
    metadata = _metadata_dict(record.get("metadata"))
    live_messages = record.get("messages", [])
    if not isinstance(live_messages, list):
        live_messages = []
    archive_messages = _read_archive_messages(saved.path) if include_archive else []
    messages = [*archive_messages, *live_messages]
    summary = saved_session_summary(saved)
    archive_path = _archive_path_for(saved.path)
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": datetime.now(UTC).isoformat(),
        "session": summary,
        "record_path": str(saved.path),
        "archive_path": str(archive_path) if archive_path.exists() else "",
        "archive_included": include_archive,
        "messages": messages,
        "live_messages": live_messages,
        "archive_messages": archive_messages,
        "metadata": metadata,
        "thinking_content": record.get("thinking_content", []),
        "tool_executions": record.get("tool_executions", []),
        "usage": {
            "total_input_tokens": int(record.get("total_input_tokens", 0) or 0),
            "total_output_tokens": int(record.get("total_output_tokens", 0) or 0),
        },
        "runtime": {
            "interaction_mode": record.get("interaction_mode") or record.get("mode") or "",
            "plan_mode": bool(record.get("plan_mode", False)),
            "agent_id": record.get("agent_id") or "",
            "workspace": record.get("workspace") or "",
            "model": _first_text(record.get("model"), metadata.get("model")),
            "model_provider": _first_text(
                record.get("model_provider"), record.get("provider"), metadata.get("provider")
            ),
        },
    }


def format_session_export(export: dict[str, Any], output_format: str) -> str:
    """Render a session export as JSON, JSONL, Markdown, or Lovely Pirate JSONL."""
    fmt = output_format.strip().lower()
    if fmt == "json":
        return json.dumps(export, ensure_ascii=False, indent=2, default=str) + "\n"
    if fmt == "jsonl":
        return _format_jsonl(export)
    if fmt in {LOVELY_PIRATE_FORMAT, "lp-jsonl"}:
        return _format_lovely_pirate_jsonl(export)
    if fmt == "md":
        return _format_markdown(export)
    raise ValueError(f"unsupported export format: {output_format}")


def _format_jsonl(export: dict[str, Any]) -> str:
    lines = [
        json.dumps(
            {
                "type": "session",
                "schema": export.get("schema"),
                "exported_at": export.get("exported_at"),
                "session": export.get("session"),
                "usage": export.get("usage"),
                "runtime": export.get("runtime"),
            },
            ensure_ascii=False,
            default=str,
        )
    ]
    archive_count = len(export.get("archive_messages") or [])
    for index, message in enumerate(export.get("messages") or []):
        source = "archive" if index < archive_count else "live"
        lines.append(
            json.dumps(
                {"type": "message", "index": index, "source": source, "message": message},
                ensure_ascii=False,
                default=str,
            )
        )
    for index, item in enumerate(export.get("tool_executions") or []):
        lines.append(json.dumps({"type": "tool_execution", "index": index, "tool_execution": item}, default=str))
    return "\n".join(lines) + "\n"


def _lovely_pirate_meta_event(export: dict[str, Any]) -> dict[str, Any]:
    session = _metadata_dict(export.get("session"))
    metadata = _metadata_dict(export.get("metadata"))
    runtime = _metadata_dict(export.get("runtime"))
    usage = _metadata_dict(export.get("usage"))
    input_tokens = int(usage.get("total_input_tokens") or 0)
    output_tokens = int(usage.get("total_output_tokens") or 0)
    payload: dict[str, Any] = {
        "id": _first_text(session.get("id"), session.get("session_id")),
        "session_id": _first_text(session.get("session_id"), session.get("id")),
        "source": "xerxes",
        "schema": export.get("schema") or EXPORT_SCHEMA,
        "cwd": _first_text(session.get("project_dir")),
        "title": _first_text(session.get("title")),
        "model_provider": _first_text(
            runtime.get("model_provider"), metadata.get("model_provider"), metadata.get("provider")
        ),
        "model": _first_text(runtime.get("model"), metadata.get("model")),
        "cli_version": _first_text(metadata.get("cli_version"), metadata.get("version")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "message_count": len(export.get("messages") or []),
        "record_path": export.get("record_path") or "",
        "archive_path": export.get("archive_path") or "",
        "updated_at": session.get("updated_at") or "",
        "exported_at": export.get("exported_at") or "",
    }
    tools = metadata.get("tools") or metadata.get("available_tools")
    if tools:
        payload["tools"] = tools
    return {
        "timestamp": _first_text(metadata.get("started_at"), metadata.get("created_at"), export.get("exported_at")),
        "type": "external_session_meta",
        "payload": payload,
    }


def _lovely_pirate_message_event(message: Any, *, index: int, source: str) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = _first_text(message.get("role"))
    if not role:
        return None
    event: dict[str, Any] = {
        "type": "external_message",
        "role": role,
        "content": _message_text(message.get("content")),
        "index": index,
        "source": source,
    }
    timestamp = _first_text(message.get("timestamp"), message.get("created_at"), message.get("updated_at"))
    if timestamp:
        event["timestamp"] = timestamp
    reasoning = _first_text(message.get("reasoning_content"), message.get("reasoning"))
    if reasoning:
        event["reasoning_content"] = reasoning
    tool_calls = message.get("tool_calls")
    if tool_calls:
        event["tool_calls"] = tool_calls
    tool_call_id = _first_text(message.get("tool_call_id"))
    if tool_call_id:
        event["tool_call_id"] = tool_call_id
    tool_name = _first_text(message.get("name"), message.get("tool_name"))
    if tool_name:
        event["name"] = tool_name
    return event


def _format_lovely_pirate_jsonl(export: dict[str, Any]) -> str:
    lines = [json.dumps(_lovely_pirate_meta_event(export), ensure_ascii=False, default=str)]
    archive_count = len(export.get("archive_messages") or [])
    for index, message in enumerate(export.get("messages") or []):
        source = "archive" if index < archive_count else "live"
        event = _lovely_pirate_message_event(message, index=index, source=source)
        if event is not None:
            lines.append(json.dumps(event, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n"


def _format_markdown(export: dict[str, Any]) -> str:
    session = export.get("session") or {}
    lines = [
        f"# Xerxes Session Export: {session.get('id', '')}",
        "",
        f"- Project: `{session.get('project_dir', '')}`",
        f"- Title: {session.get('title', '')}",
        f"- Updated: {session.get('updated_at', '')}",
        f"- Exported: {export.get('exported_at', '')}",
        f"- Messages: {len(export.get('messages') or [])}",
        f"- Record: `{export.get('record_path', '')}`",
    ]
    archive_path = str(export.get("archive_path") or "")
    if archive_path:
        lines.append(f"- Archive: `{archive_path}`")
    lines.extend(["", "## Messages", ""])
    archive_count = len(export.get("archive_messages") or [])
    for index, message in enumerate(export.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "message")
        source = "archive" if index < archive_count else "live"
        lines.extend([f"### {index + 1}. {role} ({source})", "", _message_text(message.get("content")).rstrip(), ""])
        tool_calls = message.get("tool_calls")
        if tool_calls:
            lines.extend(["```json", json.dumps(tool_calls, ensure_ascii=False, indent=2, default=str), "```", ""])
    return "\n".join(lines).rstrip() + "\n"
