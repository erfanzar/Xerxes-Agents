#!/bin/bash
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
PORT="${1:-8080}"
DIR="${2:-.}"

echo "=== p5.js Dev Server ==="
echo "Serving: $(cd "$DIR" && pwd)"
echo "URL:     http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

cd "$DIR" && python3 -m http.server "$PORT" 2>/dev/null || {
  echo "Python3 not found. Trying Node.js..."
  npx serve -l "$PORT" "$DIR" 2>/dev/null || {
    echo "Error: Need python3 or npx (Node.js) for local server"
    exit 1
  }
}
