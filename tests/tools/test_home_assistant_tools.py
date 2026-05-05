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
"""Tests for Home Assistant agent tools."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from xerxes.tools.home_assistant_tools import (
    HomeAssistantClient,
    ha_call_service,
    ha_get_state,
    ha_list_entities,
    ha_list_services,
)


@dataclass
class _R:
    payload: object
    status_code: int = 200

    def json(self):
        return self.payload


class _FakeHTTP:
    def __init__(self):
        self.calls: list[dict] = []
        self.states = [
            {"entity_id": "light.kitchen_main", "state": "off", "attributes": {"area_id": "kitchen"}},
            {"entity_id": "light.living_room", "state": "on", "attributes": {"area_id": "living_room"}},
            {"entity_id": "switch.coffee", "state": "off", "attributes": {"area_id": "kitchen"}},
        ]
        self.services = [
            {"domain": "light", "services": {"turn_on": {"description": "turn on"}, "turn_off": {}}},
            {"domain": "switch", "services": {"turn_on": {}, "turn_off": {}}},
        ]

    def get(self, url, headers=None):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        if url.endswith("/api/states"):
            return _R(self.states)
        if "/api/states/" in url:
            eid = url.rsplit("/", 1)[-1]
            for s in self.states:
                if s["entity_id"] == eid:
                    return _R(s)
            return _R({}, status_code=404)
        if url.endswith("/api/services"):
            return _R(self.services)
        return _R({})

    def post(self, url, json=None, headers=None):
        self.calls.append({"method": "POST", "url": url, "json": json})
        return _R([{"entity_id": (json or {}).get("entity_id", "light.kitchen_main"), "state": "on"}])


@pytest.fixture
def http():
    h = _FakeHTTP()
    HomeAssistantClient.install_for_test(http_client=h)
    yield h
    HomeAssistantClient.reset()


class TestListEntities:
    def test_lists_all(self, http):
        out = ha_list_entities.static_call()
        assert out["count"] == 3

    def test_filter_by_domain(self, http):
        out = ha_list_entities.static_call(domain="light")
        assert {e["entity_id"] for e in out["entities"]} == {"light.kitchen_main", "light.living_room"}

    def test_filter_by_area(self, http):
        out = ha_list_entities.static_call(area="kitchen")
        assert {e["entity_id"] for e in out["entities"]} == {"light.kitchen_main", "switch.coffee"}

    def test_combined_filters(self, http):
        out = ha_list_entities.static_call(domain="light", area="kitchen")
        assert {e["entity_id"] for e in out["entities"]} == {"light.kitchen_main"}

    def test_limit(self, http):
        out = ha_list_entities.static_call(limit=1)
        assert out["count"] == 1


class TestListServices:
    def test_returns_all(self, http):
        out = ha_list_services.static_call()
        assert {d["domain"] for d in out["domains"]} == {"light", "switch"}

    def test_domain_filter(self, http):
        out = ha_list_services.static_call(domain="light")
        assert [d["domain"] for d in out["domains"]] == ["light"]


class TestGetState:
    def test_existing(self, http):
        out = ha_get_state.static_call(entity_id="light.kitchen_main")
        assert out["state"] == "off"
        assert out["attributes"]["area_id"] == "kitchen"

    def test_missing(self, http):
        out = ha_get_state.static_call(entity_id="light.does_not_exist")
        assert out["error"] == "not_found"


class TestCallService:
    def test_invokes_correct_endpoint(self, http):
        out = ha_call_service.static_call(domain="light", service="turn_on", data={"entity_id": "light.kitchen_main"})
        assert out["ok"] is True
        assert out["domain"] == "light"
        post_call = next(c for c in http.calls if c["method"] == "POST")
        assert "/api/services/light/turn_on" in post_call["url"]
        assert post_call["json"]["entity_id"] == "light.kitchen_main"

    def test_propagates_error(self, http):
        class Broken:
            def post(self, *a, **kw):
                raise RuntimeError("hass down")

            def get(self, *a, **kw):
                raise RuntimeError("hass down")

        HomeAssistantClient.install_for_test(http_client=Broken())
        out = ha_call_service.static_call(domain="light", service="turn_on")
        assert out["ok"] is False
        assert "hass down" in out["error"]
