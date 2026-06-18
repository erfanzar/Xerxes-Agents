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
"""URL safety / SSRF prevention.

Refuses URLs pointing at RFC1918 / loopback / link-local addresses
unless they're on an explicit allowlist (e.g. ``localhost`` during
local development).

Exports:
    - UrlSafetyDecision
    - is_url_safe
    - check_url"""

from __future__ import annotations

import ipaddress
import urllib.parse
from dataclasses import dataclass


@dataclass
class UrlSafetyDecision:
    """Result of evaluating a URL against the SSRF policy.

    Attributes:
        url: the URL that was evaluated (echoed back for caller context).
        allowed: whether the URL is permitted by current policy.
        reason: short human-readable explanation of the decision.
    """

    url: str
    allowed: bool
    reason: str = ""


_DEFAULT_ALLOWLIST: frozenset[str] = frozenset()
_DENY_SCHEMES: frozenset[str] = frozenset({"file", "ftp", "gopher", "data"})


def _is_internal_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``addr`` is a private/internal/non-routable address.

    Uses the stdlib ``ipaddress`` classification rather than a hand-rolled
    network list so that we correctly catch IPv4-mapped IPv6 literals
    (e.g. ``::ffff:169.254.169.254``), the unspecified addresses
    (``0.0.0.0`` / ``::``), and the full set of reserved/link-local ranges."""
    # Unwrap IPv4-mapped IPv6 literals (e.g. ::ffff:169.254.169.254) so the
    # embedded IPv4 address is classified, not the IPv6 wrapper.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
        or addr.is_multicast
    )


def _classify_host(host: str) -> bool:
    """Return True if ``host`` resolves to a private/internal address.

    We DON'T do DNS lookup — only literal IPs and a small set of well-
    known hostnames. Production deployments should plug in a DNS check
    via a wrapper before allowing external traffic."""
    if not host:
        return True
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # hostname — let caller decide based on policy
    return _is_internal_address(addr)


def is_url_safe(url: str, *, allowlist: frozenset[str] = _DEFAULT_ALLOWLIST) -> bool:
    """Return True if the URL is safe to fetch."""
    return check_url(url, allowlist=allowlist).allowed


def check_url(url: str, *, allowlist: frozenset[str] = _DEFAULT_ALLOWLIST) -> UrlSafetyDecision:
    """Detailed safety check returning a ``UrlSafetyDecision``."""
    if not url:
        return UrlSafetyDecision(url=url, allowed=False, reason="empty URL")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() in _DENY_SCHEMES:
        return UrlSafetyDecision(url=url, allowed=False, reason=f"denied scheme: {parsed.scheme}")
    if not parsed.scheme or not parsed.hostname:
        return UrlSafetyDecision(url=url, allowed=False, reason="missing scheme or host")
    host = parsed.hostname
    if host in allowlist:
        return UrlSafetyDecision(url=url, allowed=True, reason="allowlisted host")
    if _classify_host(host):
        return UrlSafetyDecision(url=url, allowed=False, reason=f"private/internal host: {host}")
    return UrlSafetyDecision(url=url, allowed=True, reason="public host")


__all__ = ["UrlSafetyDecision", "check_url", "is_url_safe"]
