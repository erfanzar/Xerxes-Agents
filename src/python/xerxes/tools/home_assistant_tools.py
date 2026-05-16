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
"""Home Assistant integration tools for controlling smart home devices and querying entity states.

This module provides tools for interacting with Home Assistant instances, including
listing entities, getting states, and calling services.

Example:
    >>> from xerxes.tools.home_assistant_tools import HomeAssistantClient, ha_list_entities
    >>> entities = ha_list_entities.static_call(domain="light")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import typing as tp

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for interacting with Home Assistant REST API.

    This class provides a thread-safe interface to Home Assistant's API, enabling
    agents to query entity states, list services, and control smart home devices.

    Uses environment variables HASS_BASE_URL and HASS_TOKEN for configuration,
    or can be instantiated with explicit parameters.

    Attributes:
        _instance: Singleton instance for the client.
        _lock: Thread synchronization lock.

    Example:
        >>> client = HomeAssistantClient.instance()
        >>> states = client.list_states()
        >>> client.call_service("light", "turn_on", {"entity_id": "light.living_room"})
    """

    _instance: HomeAssistantClient | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Initialize the Home Assistant client.

        Args:
            base_url: Home Assistant URL. Defaults to HASS_BASE_URL env var.
            token: Long-lived access token. Defaults to HASS_TOKEN env var.
            http_client: Optional custom HTTP client for testing.
        """
        self.base_url = (base_url or os.environ.get("HASS_BASE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("HASS_TOKEN", "")
        self._http = http_client

    @classmethod
    def instance(cls) -> HomeAssistantClient:
        """Get the singleton client instance.

        Returns:
            The shared HomeAssistantClient instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def install_for_test(
        cls,
        *,
        base_url: str = "http://hass.test",
        token: str = "tok",
        http_client: tp.Any,
    ) -> HomeAssistantClient:
        """Install a test client for unit testing.

        Args:
            base_url: Test server URL.
            token: Test authentication token.
            http_client: Mock HTTP client.

        Returns:
            The configured test client instance.
        """
        with cls._lock:
            cls._instance = cls(base_url=base_url, token=token, http_client=http_client)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance to None."""
        with cls._lock:
            cls._instance = None

    def list_states(self) -> list[dict[str, tp.Any]]:
        """Get all entity states from Home Assistant.

        Returns:
            List of entity state dictionaries.
        """
        return self._get("/api/states")

    def get_state(self, entity_id: str) -> dict[str, tp.Any] | None:
        """Get the state of a specific entity.

        Args:
            entity_id: Full entity ID (e.g., "light.living_room").

        Returns:
            Entity state dictionary, or None if not found.
        """
        try:
            return self._get(f"/api/states/{entity_id}")
        except Exception:
            return None

    def list_services(self) -> list[dict[str, tp.Any]]:
        """Get available services from Home Assistant.

        Returns:
            List of service domain dictionaries.
        """
        return self._get("/api/services")

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., "light", "switch").
            service: Service name (e.g., "turn_on", "toggle").
            data: Optional service data dictionary.

        Returns:
            List of service call results.
        """
        return self._post(f"/api/services/{domain}/{service}", json_body=data or {})

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests."""
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        """Build full URL from path.

        Args:
            path: API endpoint path.

        Returns:
            Full URL string.
        """
        if not self.base_url:
            raise RuntimeError("HomeAssistantClient: HASS_BASE_URL is not configured")
        return f"{self.base_url}{path}"

    def _get(self, path: str) -> tp.Any:
        """Perform GET request to Home Assistant API.

        Args:
            path: API endpoint path.

        Returns:
            Parsed JSON response.
        """
        url = self._url(path)
        if self._http is not None:
            resp = self._http.get(url, headers=self._headers())
            return _parse(resp)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for HomeAssistantClient") from exc
        resp = httpx.get(url, headers=self._headers(), timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_body: dict[str, tp.Any]) -> tp.Any:
        """Perform POST request to Home Assistant API.

        Args:
            path: API endpoint path.
            json_body: JSON body for the request.

        Returns:
            Parsed JSON response.
        """
        url = self._url(path)
        if self._http is not None:
            resp = self._http.post(url, json=json_body, headers=self._headers())
            return _parse(resp)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for HomeAssistantClient") from exc
        resp = httpx.post(url, json=json_body, headers=self._headers(), timeout=15.0)
        resp.raise_for_status()
        return resp.json()


def _parse(resp: tp.Any) -> tp.Any:
    """Parse response to JSON.

    Args:
        resp: HTTP response object.

    Returns:
        Parsed JSON or raw body.
    """
    if hasattr(resp, "json") and callable(resp.json):
        try:
            return resp.json()
        except Exception:
            pass
    body = getattr(resp, "text", None) or getattr(resp, "body", None) or ""
    if isinstance(body, bytes):
        body = body.decode()
    try:
        return json.loads(body)
    except Exception:
        return body


def _filter_entities(
    states: list[dict[str, tp.Any]],
    *,
    domain: str | None = None,
    area: str | None = None,
) -> list[dict[str, tp.Any]]:
    """Filter entity states by domain or area.

    Args:
        states: List of entity states.
        domain: Filter by domain (e.g., "light").
        area: Filter by area ID.

    Returns:
        Filtered list of states.
    """
    out = []
    for s in states or []:
        eid = s.get("entity_id", "")
        if domain and not eid.startswith(f"{domain}."):
            continue
        if area:
            attrs = s.get("attributes") or {}
            entity_area = attrs.get("area_id") or attrs.get("area")
            if entity_area != area:
                continue
        out.append(s)
    return out


class ha_list_entities(AgentBaseFn):
    """List Home Assistant entities with optional filtering.

    Retrieves all or filtered entity states from Home Assistant.

    Example:
        >>> ha_list_entities.static_call(domain="light")
        >>> ha_list_entities.static_call(area="Living Room", limit=50)
    """

    @staticmethod
    def static_call(
        domain: str | None = None,
        area: str | None = None,
        limit: int = 200,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """List Home Assistant entities.

        Args:
            domain: Filter by entity domain (e.g., "light", "switch").
            area: Filter by area name.
            limit: Maximum number of entities to return. Defaults to 200.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'count' and 'entities' list.
        """
        try:
            limit_n = int(limit) if limit is not None else 200
        except (TypeError, ValueError):
            limit_n = 200
        states = HomeAssistantClient.instance().list_states()
        filtered = _filter_entities(states or [], domain=domain, area=area)[:limit_n]
        return {
            "count": len(filtered),
            "entities": [
                {
                    "entity_id": e.get("entity_id"),
                    "state": e.get("state"),
                    "attributes": e.get("attributes", {}),
                }
                for e in filtered
            ],
        }


class ha_list_services(AgentBaseFn):
    """List available Home Assistant services.

    Retrieves all service domains or filters by specific domain.

    Example:
        >>> ha_list_services.static_call()
        >>> ha_list_services.static_call(domain="light")
    """

    @staticmethod
    def static_call(
        domain: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """List Home Assistant services.

        Args:
            domain: Filter by service domain.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'domains' list.
        """
        catalog = HomeAssistantClient.instance().list_services() or []
        if domain:
            catalog = [d for d in catalog if d.get("domain") == domain]
        return {"domains": catalog}


class ha_get_state(AgentBaseFn):
    """Get the current state of a Home Assistant entity.

    Example:
        >>> ha_get_state.static_call(entity_id="light.living_room")
    """

    @staticmethod
    def static_call(
        entity_id: str,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Get entity state.

        Args:
            entity_id: Full entity ID to query.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with entity_id, state, attributes, and timestamps.
        """
        state = HomeAssistantClient.instance().get_state(entity_id)
        if not state or not isinstance(state, dict) or not state.get("entity_id"):
            return {"error": "not_found", "entity_id": entity_id}
        return {
            "entity_id": state.get("entity_id"),
            "state": state.get("state"),
            "attributes": state.get("attributes", {}),
            "last_changed": state.get("last_changed", ""),
            "last_updated": state.get("last_updated", ""),
        }


class ha_call_service(AgentBaseFn):
    """Call a Home Assistant service to control devices.

    Example:
        >>> ha_call_service.static_call(
        ...     domain="light",
        ...     service="turn_on",
        ...     data={"entity_id": "light.living_room"}
        ... )
    """

    @staticmethod
    def static_call(
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., "light", "switch", "automation").
            service: Service to call (e.g., "turn_on", "toggle").
            data: Optional service data dictionary with entity_id and other parameters.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status and service response.
        """
        try:
            changed = HomeAssistantClient.instance().call_service(domain, service, data or {})
            return {"ok": True, "domain": domain, "service": service, "changed": changed}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "domain": domain, "service": service}


__all__ = [
    "HomeAssistantClient",
    "ha_call_service",
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
]
