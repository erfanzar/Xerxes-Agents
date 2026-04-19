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

"""Home Assistant agent tools.

Four ``ha_*`` tools that bridge agents to a Home Assistant instance:

- :class:`ha_list_entities` — list entities by domain/area
- :class:`ha_list_services` — list available service actions
- :class:`ha_get_state`     — read entity attributes + state
- :class:`ha_call_service`  — invoke a service (turn lights on, etc.)

Each tool reads ``HASS_BASE_URL`` and ``HASS_TOKEN`` from the
environment by default; tests inject a fake HTTP client through
:class:`HomeAssistantClient.install_for_test`.
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
    """Singleton HTTP client wrapping the Home Assistant REST API.

    Reads ``HASS_BASE_URL`` (e.g. ``http://homeassistant.local:8123``)
    and ``HASS_TOKEN`` (long-lived access token) at construction
    time. Tests inject a fake transport via
    :meth:`install_for_test` to avoid any network I/O.

    Example:
        >>> client = HomeAssistantClient.instance()
        >>> client.list_states()
    """

    _instance: HomeAssistantClient | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Initialise with an explicit base URL / token, or read them from env vars."""
        self.base_url = (base_url or os.environ.get("HASS_BASE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("HASS_TOKEN", "")
        self._http = http_client

    @classmethod
    def instance(cls) -> HomeAssistantClient:
        """Return the process-wide singleton, creating it on first use."""
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
        """Replace the singleton with one backed by a fake HTTP client (tests)."""
        with cls._lock:
            cls._instance = cls(base_url=base_url, token=token, http_client=http_client)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton so the next :meth:`instance` call rebuilds it."""
        with cls._lock:
            cls._instance = None

    def list_states(self) -> list[dict[str, tp.Any]]:
        """Return every entity state (``GET /api/states``)."""
        return self._get("/api/states")

    def get_state(self, entity_id: str) -> dict[str, tp.Any] | None:
        """Fetch a single entity's state record, or ``None`` on any failure."""
        try:
            return self._get(f"/api/states/{entity_id}")
        except Exception:
            return None

    def list_services(self) -> list[dict[str, tp.Any]]:
        """Return the list of available service domains and their services."""
        return self._get("/api/services")

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Invoke ``domain.service`` with optional payload and return affected states."""
        return self._post(f"/api/services/{domain}/{service}", json_body=data or {})

    def _headers(self) -> dict[str, str]:
        """Build the JSON content-type and bearer-auth headers for HA requests."""
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        """Join *path* onto the configured base URL, raising if unset."""
        if not self.base_url:
            raise RuntimeError("HomeAssistantClient: HASS_BASE_URL is not configured")
        return f"{self.base_url}{path}"

    def _get(self, path: str) -> tp.Any:
        """GET *path* via the injected client or httpx and return parsed JSON."""
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
        """POST JSON to *path* via the injected client or httpx and return parsed JSON."""
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
    """Best-effort JSON decode for both real ``httpx`` and fake responses."""
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
    """Filter *states* by entity-id *domain* prefix and/or matching area attribute."""
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
    """List Home Assistant entities, optionally filtered by domain/area."""

    @staticmethod
    def static_call(
        domain: str | None = None,
        area: str | None = None,
        limit: int = 200,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Return the list of entities the connected HA instance exposes.

        Use this to discover what the agent can control before calling
        :class:`ha_call_service`. Filtering keeps responses small in
        large installs.

        Args:
            domain: Optional domain prefix (``"light"``, ``"switch"``,
                ``"climate"``, ...). When set, only entities whose
                ``entity_id`` starts with ``"{domain}."`` are returned.
            area: Optional area ID (e.g. ``"living_room"``). Matched
                against ``attributes.area_id`` / ``attributes.area``.
            limit: Max number of entities to include in the response.

        Returns:
            ``{"count": int, "entities": [{entity_id, state, attributes}]}``.
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
    """List available HA service actions per domain."""

    @staticmethod
    def static_call(
        domain: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Return the catalog of services the HA instance can perform.

        Helpful before calling :class:`ha_call_service` so the agent
        can confirm a service exists and inspect its expected fields.

        Args:
            domain: Optional domain filter (e.g. ``"light"``).

        Returns:
            ``{"domains": [{"domain", "services": {name → {"description","fields"}}}]}``.
        """
        catalog = HomeAssistantClient.instance().list_services() or []
        if domain:
            catalog = [d for d in catalog if d.get("domain") == domain]
        return {"domains": catalog}


class ha_get_state(AgentBaseFn):
    """Fetch state + attributes for a single entity."""

    @staticmethod
    def static_call(
        entity_id: str,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Return the current state object for ``entity_id``.

        Args:
            entity_id: Fully qualified entity id, e.g.
                ``"light.kitchen_main"``.

        Returns:
            ``{"entity_id", "state", "attributes", "last_changed",
            "last_updated"}`` or ``{"error": "not_found"}``.
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
    """Invoke a Home Assistant service action."""

    @staticmethod
    def static_call(
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Call ``{domain}.{service}`` with optional service data.

        Use this when the agent needs to actually change device state
        (turn lights on/off, set thermostat, run a script). Verify the
        service exists with :class:`ha_list_services` first.

        Args:
            domain: Service domain (``"light"``, ``"switch"``,
                ``"climate"``, ``"script"``, ``"automation"``, ...).
            service: Service name within the domain (``"turn_on"``,
                ``"turn_off"``, ``"set_temperature"``, ...).
            data: Service data payload. ``entity_id`` is the most
                common key (``{"entity_id": "light.kitchen_main"}``).

        Returns:
            ``{"ok": True, "changed": [...]}`` listing entities the
            service modified, or ``{"ok": False, "error": ...}``.
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
