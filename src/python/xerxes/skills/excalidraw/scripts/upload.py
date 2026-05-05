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
"""Upload module for Xerxes.

Exports:
    - UPLOAD_URL
    - concat_buffers
    - upload
    - main"""

import base64
import json
import os
import struct
import sys
import urllib.request
import zlib

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("Error: 'cryptography' package is required for upload.")
    print("Install it with: pip install cryptography")
    sys.exit(1)

UPLOAD_URL = "https://json.excalidraw.com/api/v2/post/"


def concat_buffers(*buffers: bytes) -> bytes:
    """Concat buffers.

    Args:
        *buffers: IN: Additional positional arguments. OUT: Passed through to downstream calls.
    Returns:
        bytes: OUT: Result of the operation."""

    parts = [struct.pack(">I", 1)]
    for buf in buffers:
        parts.append(struct.pack(">I", len(buf)))
        parts.append(buf)
    return b"".join(parts)


def upload(excalidraw_json: str) -> str:
    """Upload.

    Args:
        excalidraw_json (str): IN: excalidraw json. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    file_metadata = json.dumps({}).encode("utf-8")
    data_bytes = excalidraw_json.encode("utf-8")
    inner_payload = concat_buffers(file_metadata, data_bytes)

    compressed = zlib.compress(inner_payload)

    raw_key = os.urandom(16)
    iv = os.urandom(12)
    aesgcm = AESGCM(raw_key)
    encrypted = aesgcm.encrypt(iv, compressed, None)

    encoding_meta = json.dumps(
        {
            "version": 2,
            "compression": "pako@1",
            "encryption": "AES-GCM",
        }
    ).encode("utf-8")

    payload = concat_buffers(encoding_meta, iv, encrypted)

    req = urllib.request.Request(UPLOAD_URL, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Upload failed with HTTP {resp.status}")
        result = json.loads(resp.read().decode("utf-8"))

    file_id = result.get("id")
    if not file_id:
        raise RuntimeError(f"Upload returned no file ID. Response: {result}")

    key_b64 = base64.urlsafe_b64encode(raw_key).rstrip(b"=").decode("ascii")

    return f"https://excalidraw.com/#json={file_id},{key_b64}"


def main():
    """Main.

    Returns:
        Any: OUT: Result of the operation."""
    if len(sys.argv) < 2:
        print("Usage: python upload.py <path-to-file.excalidraw>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        doc = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: File is not valid JSON: {e}")
        sys.exit(1)

    if "elements" not in doc:
        print("Warning: File does not contain an 'elements' key. Uploading anyway.")

    url = upload(content)
    print(url)


if __name__ == "__main__":
    main()
