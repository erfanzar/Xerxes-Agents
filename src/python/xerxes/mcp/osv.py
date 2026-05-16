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
"""Gate MCP package installs against the OSV vulnerability database.

Before running ``npx`` or ``uvx`` against an unfamiliar package, the MCP
install path asks OSV.dev whether any GHSA/MAL advisories cover the
``(ecosystem, name, version)`` tuple. :func:`check_package` is the only
network IO point — tests can stub it directly. Results are cached for 24h
in-memory and on-disk to keep OSV from being hammered. :func:`is_blocked`
classifies a result list into block/allow based on the advisory severity.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

OSV_API_URL = "https://api.osv.dev/v1/query"

DEFAULT_CACHE_TTL = 24 * 3600  # 24h


@dataclass
class Vulnerability:
    """One OSV advisory record relevant to a package.

    Attributes:
        id: GHSA / MAL / CVE identifier.
        summary: Short title.
        severity: Free-form severity (CVSS string, ``"MAL"``, etc.).
        aliases: Alternate identifiers the advisory is also known as.
    """

    id: str
    summary: str = ""
    severity: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class _CacheEntry:
    """One on-disk cache row: timestamp plus the cached advisories."""

    fetched_at: float
    vulnerabilities: list[Vulnerability]


def _cache_path(cache_dir: Path) -> Path:
    """Return the on-disk cache file path inside ``cache_dir``."""
    return cache_dir / "osv_cache.json"


def _load_cache(cache_dir: Path) -> dict[str, _CacheEntry]:
    """Read the OSV cache from ``cache_dir`` (empty dict on miss/corruption)."""
    path = _cache_path(cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, _CacheEntry] = {}
    for key, entry in raw.items():
        try:
            out[key] = _CacheEntry(
                fetched_at=float(entry["fetched_at"]),
                vulnerabilities=[Vulnerability(**v) for v in entry["vulnerabilities"]],
            )
        except (KeyError, TypeError):
            continue
    return out


def _save_cache(cache_dir: Path, cache: dict[str, _CacheEntry]) -> None:
    """Persist ``cache`` to ``osv_cache.json`` under ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: {
            "fetched_at": entry.fetched_at,
            "vulnerabilities": [
                {"id": v.id, "summary": v.summary, "severity": v.severity, "aliases": v.aliases}
                for v in entry.vulnerabilities
            ],
        }
        for key, entry in cache.items()
    }
    _cache_path(cache_dir).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _parse_response(payload: dict[str, Any]) -> list[Vulnerability]:
    """Convert an OSV ``/v1/query`` response into :class:`Vulnerability` records."""
    out: list[Vulnerability] = []
    for vuln in payload.get("vulns", []) or []:
        out.append(
            Vulnerability(
                id=vuln.get("id", ""),
                summary=vuln.get("summary", ""),
                severity=(vuln.get("database_specific", {}) or {}).get("severity", ""),
                aliases=list(vuln.get("aliases", []) or []),
            )
        )
    return out


def check_package(
    ecosystem: str,
    name: str,
    version: str | None = None,
    *,
    cache_dir: Path | None = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    client: httpx.Client | None = None,
) -> list[Vulnerability]:
    """Return vulnerabilities covering ``(ecosystem, name, version)``.

    Responses cached younger than ``cache_ttl`` seconds are reused. Network
    failures degrade gracefully to an empty list so install isn't blocked
    by a transient OSV outage.
    """

    cache: dict[str, _CacheEntry] = {}
    cache_key = f"{ecosystem}::{name}::{version or ''}"
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        entry = cache.get(cache_key)
        if entry is not None and time.time() - entry.fetched_at < cache_ttl:
            return list(entry.vulnerabilities)

    payload: dict[str, Any] = {"package": {"name": name, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version

    own_client = client is None
    c = client or httpx.Client(timeout=10.0)
    vulns: list[Vulnerability]
    try:
        try:
            resp = c.post(OSV_API_URL, json=payload)
            resp.raise_for_status()
            vulns = _parse_response(resp.json())
        except (httpx.HTTPError, ValueError):
            return []
    finally:
        if own_client:
            c.close()

    if cache_dir is not None:
        cache[cache_key] = _CacheEntry(fetched_at=time.time(), vulnerabilities=vulns)
        _save_cache(cache_dir, cache)
    return vulns


def is_blocked(vulnerabilities: list[Vulnerability]) -> bool:
    """True when any advisory is a malware feed entry or CRITICAL/HIGH severity.

    Lower-severity vulns aren't blocking here — callers can make their own
    policy decision based on the returned list.
    """
    for v in vulnerabilities:
        if v.id.startswith("MAL-"):
            return True
        if v.severity.upper().startswith(("CRITICAL", "HIGH")):
            return True
    return False


__all__ = ["DEFAULT_CACHE_TTL", "OSV_API_URL", "Vulnerability", "check_package", "is_blocked"]
