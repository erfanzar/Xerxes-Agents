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
"""Home assistant tools module for Xerxes.

Exports:
    - logger
    - HomeAssistantClient
    - ha_list_entities
    - ha_list_services
    - ha_get_state
    - ha_call_service"""

from __future__ import annotations

import json
import logging
import os
import threading
import typing as tp

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Home assistant client.

    Attributes:
        _instance (HomeAssistantClient | None): instance."""

    _instance: HomeAssistantClient | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            base_url (str | None, optional): IN: base url. Defaults to None. OUT: Consumed during execution.
            token (str | None, optional): IN: token. Defaults to None. OUT: Consumed during execution.
            http_client (tp.Any | None, optional): IN: http client. Defaults to None. OUT: Consumed during execution."""

        self.base_url = (base_url or os.environ.get("HASS_BASE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("HASS_TOKEN", "")
        self._http = http_client

    @classmethod
    def instance(cls) -> HomeAssistantClient:
        """Instance.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
        Returns:
            HomeAssistantClient: OUT: Result of the operation."""

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
        """Install for test.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            base_url (str, optional): IN: base url. Defaults to 'http://hass.test'. OUT: Consumed during execution.
            token (str, optional): IN: token. Defaults to 'tok'. OUT: Consumed during execution.
            http_client (tp.Any): IN: http client. OUT: Consumed during execution.
        Returns:
            HomeAssistantClient: OUT: Result of the operation."""

        with cls._lock:
            cls._instance = cls(base_url=base_url, token=token, http_client=http_client)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset.

        Args:
            cls: IN: The class. OUT: Used for class-level operations."""

        with cls._lock:
            cls._instance = None

    def list_states(self) -> list[dict[str, tp.Any]]:
        """List states.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, tp.Any]]: OUT: Result of the operation."""

        return self._get("/api/states")

    def get_state(self, entity_id: str) -> dict[str, tp.Any] | None:
        """Retrieve the state.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            entity_id (str): IN: entity id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any] | None: OUT: Result of the operation."""

        try:
            return self._get(f"/api/states/{entity_id}")
        except Exception:
            return None

    def list_services(self) -> list[dict[str, tp.Any]]:
        """List services.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, tp.Any]]: OUT: Result of the operation."""

        return self._get("/api/services")

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Call service.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            domain (str): IN: domain. OUT: Consumed during execution.
            service (str): IN: service. OUT: Consumed during execution.
            data (dict[str, tp.Any] | None, optional): IN: data. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[dict[str, tp.Any]]: OUT: Result of the operation."""

        return self._post(f"/api/services/{domain}/{service}", json_body=data or {})

    def _headers(self) -> dict[str, str]:
        """Internal helper to headers.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        """Internal helper to url.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            path (str): IN: path. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if not self.base_url:
            raise RuntimeError("HomeAssistantClient: HASS_BASE_URL is not configured")
        return f"{self.base_url}{path}"

    def _get(self, path: str) -> tp.Any:
        """Internal helper to get.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            path (str): IN: path. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""

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
        """Internal helper to post.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            path (str): IN: path. OUT: Consumed during execution.
            json_body (dict[str, tp.Any]): IN: json body. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""

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
    """Internal helper to parse.

    Args:
        resp (tp.Any): IN: resp. OUT: Consumed during execution.
    Returns:
        tp.Any: OUT: Result of the operation."""

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
    """Internal helper to filter entities.

    Args:
        states (list[dict[str, tp.Any]]): IN: states. OUT: Consumed during execution.
        domain (str | None, optional): IN: domain. Defaults to None. OUT: Consumed during execution.
        area (str | None, optional): IN: area. Defaults to None. OUT: Consumed during execution.
    Returns:
        list[dict[str, tp.Any]]: OUT: Result of the operation."""

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
    """Ha list entities.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        domain: str | None = None,
        area: str | None = None,
        limit: int = 200,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            domain (str | None, optional): IN: domain. Defaults to None. OUT: Consumed during execution.
            area (str | None, optional): IN: area. Defaults to None. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 200. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Ha list services.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        domain: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            domain (str | None, optional): IN: domain. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        catalog = HomeAssistantClient.instance().list_services() or []
        if domain:
            catalog = [d for d in catalog if d.get("domain") == domain]
        return {"domains": catalog}


class ha_get_state(AgentBaseFn):
    """Ha get state.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        entity_id: str,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            entity_id (str): IN: entity id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Ha call service.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        domain: str,
        service: str,
        data: dict[str, tp.Any] | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            domain (str): IN: domain. OUT: Consumed during execution.
            service (str): IN: service. OUT: Consumed during execution.
            data (dict[str, tp.Any] | None, optional): IN: data. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
