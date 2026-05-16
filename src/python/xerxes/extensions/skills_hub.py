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
"""Skill installation, uninstallation, and search hub.

``SkillsHub`` coordinates multiple ``SkillSource`` backends (local, GitHub,
official) so users can install reusable agent skills from a URI like
``github:owner/repo/path``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xerxes.core.paths import xerxes_subdir

logger = logging.getLogger(__name__)

SKILLS_DIR = xerxes_subdir("skills")
HUB_DIR = SKILLS_DIR / ".hub"
LOCK_FILE = HUB_DIR / "lock.json"
QUARANTINE_DIR = HUB_DIR / "quarantine"
AUDIT_LOG = HUB_DIR / "audit.log"


@dataclass
class SkillHubEntry:
    """Record of an installed skill in the hub lock file.

    Attributes:
        name: Skill identifier and lock key.
        source: Backend name (``local``, ``github``, ``official``).
        identifier: Backend-specific lookup string.
        installed_at: Unix timestamp of installation.
        path: Filesystem location of the installed skill.
        metadata: Extra backend-specific metadata.
    """

    name: str
    source: str
    identifier: str
    installed_at: float
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillSource(ABC):
    """Abstract backend that can fetch and search for skills."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique backend identifier (e.g. ``"local"``)."""

    @abstractmethod
    def fetch(self, identifier: str) -> dict[str, Any]:
        """Retrieve a skill bundle keyed by ``identifier``.

        Returns:
            Bundle dict with keys ``name``, ``content``, ``files``, and ``metadata``.
        """

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search this backend for skills matching ``query``.

        Returns:
            Result dicts with ``name``, ``source``, and ``identifier``.
        """


class LocalSkillSource(SkillSource):
    """Skill source backed by the local filesystem."""

    @property
    def name(self) -> str:
        """Return ``"local"``."""
        return "local"

    def fetch(self, identifier: str) -> dict[str, Any]:
        """Read a local skill from a directory path or direct ``SKILL.md`` file.

        Args:
            identifier: Path to the skill directory or ``SKILL.md`` file.

        Raises:
            FileNotFoundError: The path or ``SKILL.md`` does not exist.
        """
        path = Path(identifier).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local skill not found: {identifier}")
        skill_md = path / "SKILL.md" if path.is_dir() else path
        if not skill_md.exists():
            raise FileNotFoundError(f"No SKILL.md at {skill_md}")
        content = skill_md.read_text(encoding="utf-8")

        from xerxes.extensions.skills import parse_skill_md

        skill = parse_skill_md(content, skill_md)
        return {
            "name": skill.name,
            "content": content,
            "files": {},
            "metadata": {"path": str(skill_md.parent)},
        }

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Case-insensitively search installed local skills by content."""
        results = []
        if SKILLS_DIR.exists():
            for skill_md in SKILLS_DIR.rglob("SKILL.md"):
                try:
                    text = skill_md.read_text(encoding="utf-8")
                    if query.lower() in text.lower():
                        results.append(
                            {
                                "name": skill_md.parent.name,
                                "source": "local",
                                "identifier": str(skill_md.parent),
                            }
                        )
                        if len(results) >= limit:
                            break
                except Exception:
                    pass
        return results


class GitHubSkillSource(SkillSource):
    """Skill source that fetches from GitHub repositories."""

    @property
    def name(self) -> str:
        """Return ``"github"``."""
        return "github"

    def _resolve_token(self) -> str | None:
        """Return a GitHub token from ``GITHUB_TOKEN``/``GH_TOKEN`` or ``gh auth token``."""

        token = __import__("os").environ.get("GITHUB_TOKEN") or __import__("os").environ.get("GH_TOKEN")
        if token:
            return token
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _api_headers(self) -> dict[str, str]:
        """Build GitHub REST API headers, attaching ``Authorization`` when available."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = self._resolve_token()
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def fetch(self, identifier: str) -> dict[str, Any]:
        """Fetch ``SKILL.md`` from a GitHub repo via the Contents API.

        Args:
            identifier: ``owner/repo/path`` string locating the skill.

        Raises:
            ValueError: ``identifier`` is malformed or points to a directory.
        """
        import urllib.request

        parts = identifier.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid GitHub identifier: {identifier}")
        owner, repo = parts[0], parts[1]
        path = "/".join(parts[2:])

        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}/SKILL.md"
        req = urllib.request.Request(url, headers=self._api_headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if isinstance(data, list):
            raise ValueError(f"Expected a file, got directory: {identifier}")

        import base64

        content = base64.b64decode(data["content"]).decode("utf-8")
        return {
            "name": Path(path).name,
            "content": content,
            "files": {},
            "metadata": {
                "repo": f"{owner}/{repo}",
                "path": path,
                "sha": data.get("sha", ""),
            },
        }

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search GitHub code for ``SKILL.md`` files matching ``query``.

        Returns an empty list when the search fails.
        """
        import urllib.parse
        import urllib.request

        q = urllib.parse.quote(f"{query} filename:SKILL.md")
        url = f"https://api.github.com/search/code?q={q}&per_page={limit}"
        req = urllib.request.Request(url, headers=self._api_headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items", [])
            return [
                {
                    "name": item["repository"]["name"],
                    "source": "github",
                    "identifier": f"{item['repository']['full_name']}/{item['path']}",
                }
                for item in items
            ]
        except Exception as exc:
            logger.warning("GitHub skill search failed: %s", exc)
            return []


class OfficialSkillSource(SkillSource):
    """Skill source that reads skills bundled with the Xerxes package."""

    @property
    def name(self) -> str:
        """Return ``"official"``."""
        return "official"

    def _official_dir(self) -> Path | None:
        """Return the path to the bundled ``skills`` directory, or ``None``."""
        import xerxes

        pkg_dir = Path(xerxes.__file__).parent
        official = pkg_dir / "skills"
        return official if official.exists() else None

    def fetch(self, identifier: str) -> dict[str, Any]:
        """Read an official skill identified by its directory name.

        Raises:
            FileNotFoundError: The official directory or skill is missing.
        """
        official = self._official_dir()
        if official is None:
            raise FileNotFoundError("No official skills directory found")
        skill_dir = official / identifier
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Official skill not found: {identifier}")
        content = skill_md.read_text(encoding="utf-8")
        return {
            "name": identifier,
            "content": content,
            "files": {},
            "metadata": {"official": True},
        }

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Case-insensitively search bundled official skills by content."""
        official = self._official_dir()
        if official is None:
            return []
        results = []
        for skill_md in official.rglob("SKILL.md"):
            try:
                text = skill_md.read_text(encoding="utf-8")
                if query.lower() in text.lower():
                    results.append(
                        {
                            "name": skill_md.parent.name,
                            "source": "official",
                            "identifier": skill_md.parent.name,
                        }
                    )
                    if len(results) >= limit:
                        break
            except Exception:
                pass
        return results


def _ensure_hub_dirs() -> None:
    """Create ``HUB_DIR`` and ``QUARANTINE_DIR`` if they do not exist."""
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def _load_lock() -> dict[str, dict[str, Any]]:
    """Return the parsed contents of ``LOCK_FILE``, or an empty dict on failure."""
    if not LOCK_FILE.exists():
        return {}
    try:
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_lock(data: dict[str, dict[str, Any]]) -> None:
    """Serialize ``data`` to ``LOCK_FILE`` as JSON."""
    _ensure_hub_dirs()
    LOCK_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _audit(event: str, detail: str) -> None:
    """Append a timestamped ``event`` line with ``detail`` to ``AUDIT_LOG``."""
    _ensure_hub_dirs()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"{timestamp}  {event:12s}  {detail}\n"
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class SkillsHub:
    """Coordinates skill sources and manages on-disk installation state."""

    def __init__(self, sources: dict[str, SkillSource] | None = None):
        """Initialize the hub with custom or default skill sources.

        Args:
            sources: Mapping from source name to backend; defaults to local,
                GitHub, and official sources.
        """
        self._sources = sources or {
            "local": LocalSkillSource(),
            "github": GitHubSkillSource(),
            "official": OfficialSkillSource(),
        }
        _ensure_hub_dirs()

    def install(self, uri: str, *, force: bool = False) -> str:
        """Fetch and install a skill from ``uri``.

        Args:
            uri: ``source:identifier`` string, or a plain identifier for local.
            force: When True, overwrite an existing installation.

        Returns:
            Human-readable status message.
        """

        if ":" in uri:
            source_name, identifier = uri.split(":", 1)
        else:
            source_name, identifier = "local", uri

        source = self._sources.get(source_name)
        if source is None:
            return f"[Error] Unknown source: {source_name}"

        try:
            bundle = source.fetch(identifier)
        except Exception as exc:
            logger.warning("Failed to fetch skill from %s: %s", uri, exc)
            return f"[Error] Failed to fetch {uri}: {exc}"

        skill_name = bundle["name"]
        target_dir = SKILLS_DIR / skill_name

        if target_dir.exists() and not force:
            return f"[Error] Skill '{skill_name}' already installed. Use force=True to overwrite."

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "SKILL.md").write_text(bundle["content"], encoding="utf-8")

        lock = _load_lock()
        lock[skill_name] = {
            "source": source_name,
            "identifier": identifier,
            "installed_at": time.time(),
            "metadata": bundle.get("metadata", {}),
        }
        _save_lock(lock)
        _audit("install", f"{skill_name} from {source_name}:{identifier}")

        return f"Installed skill '{skill_name}' from {source_name}:{identifier}"

    def uninstall(self, skill_name: str) -> str:
        """Remove ``skill_name`` from disk and clear its lock entry."""

        target_dir = SKILLS_DIR / skill_name
        if not target_dir.exists():
            return f"[Error] Skill '{skill_name}' is not installed."

        shutil.rmtree(target_dir)
        lock = _load_lock()
        if skill_name in lock:
            del lock[skill_name]
            _save_lock(lock)
        _audit("uninstall", skill_name)
        return f"Uninstalled skill '{skill_name}'"

    def list_installed(self) -> list[dict[str, Any]]:
        """Return summary dicts for every skill recorded in the lock file."""

        lock = _load_lock()
        results = []
        for skill_name, entry in lock.items():
            results.append(
                {
                    "name": skill_name,
                    "source": entry.get("source", "unknown"),
                    "identifier": entry.get("identifier", ""),
                    "installed_at": entry.get("installed_at", 0),
                }
            )
        return results

    def search(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Aggregate search results from every configured source."""

        results: list[dict[str, Any]] = []
        for source_name, source in self._sources.items():
            try:
                hits = source.search(query, limit=limit)
                for hit in hits:
                    hit["source"] = source_name
                results.extend(hits)
            except Exception as exc:
                logger.warning("Search on %s failed: %s", source_name, exc)
        return results

    def get_source(self, name: str) -> SkillSource | None:
        """Return the registered source backend for ``name``, or ``None``."""

        return self._sources.get(name)
