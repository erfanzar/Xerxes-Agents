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
"""Gws bridge module for Xerxes.

Exports:
    - get_hermes_home
    - get_token_path
    - refresh_token
    - get_valid_token
    - main"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_hermes_home() -> Path:
    """Retrieve the hermes home.

    Returns:
        Path: OUT: Result of the operation."""
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def get_token_path() -> Path:
    """Retrieve the token path.

    Returns:
        Path: OUT: Result of the operation."""
    return get_hermes_home() / "google_token.json"


def refresh_token(token_data: dict) -> dict:
    """Refresh token.

    Args:
        token_data (dict): IN: token data. OUT: Consumed during execution.
    Returns:
        dict: OUT: Result of the operation."""

    import urllib.error
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "client_id": token_data["client_id"],
            "client_secret": token_data["client_secret"],
            "refresh_token": token_data["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode()

    req = urllib.request.Request(token_data["token_uri"], data=params)
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Token refresh failed (HTTP {e.code}): {body}", file=sys.stderr)
        print("Re-run setup.py to re-authenticate.", file=sys.stderr)
        sys.exit(1)

    token_data["token"] = result["access_token"]
    token_data["expiry"] = datetime.fromtimestamp(
        datetime.now(UTC).timestamp() + result["expires_in"],
        tz=UTC,
    ).isoformat()

    get_token_path().write_text(json.dumps(token_data, indent=2))
    return token_data


def get_valid_token() -> str:
    """Retrieve the valid token.

    Returns:
        str: OUT: Result of the operation."""

    token_path = get_token_path()
    if not token_path.exists():
        print("ERROR: No Google token found. Run setup.py --auth-url first.", file=sys.stderr)
        sys.exit(1)

    token_data = json.loads(token_path.read_text())

    expiry = token_data.get("expiry", "")
    if expiry:
        exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        now = datetime.now(UTC)
        if now >= exp_dt:
            token_data = refresh_token(token_data)

    return token_data["token"]


def main():
    """Main.

    Returns:
        Any: OUT: Result of the operation."""

    if len(sys.argv) < 2:
        print("Usage: gws_bridge.py <gws args...>", file=sys.stderr)
        sys.exit(1)

    access_token = get_valid_token()
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = access_token

    result = subprocess.run(["gws", *sys.argv[1:]], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
