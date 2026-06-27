---
name: fix-license-headers
description: Normalize Apache-2.0 copyright headers across Python, shell, YAML, and Dockerfile files in the repository. Uses scripts/fix_license_headers.py.
version: 1.0.0
tags: [license, headers, hygiene, xerxes]
required_tools: [exec_command, ReadFile]
---

# When to use

Use this skill when:
- You've created new files that need the Apache-2.0 copyright header.
- You've modified existing files and want to verify the header is correct.
- You're preparing a release or PR and need to normalize headers across the repo.
- A lint or CI check flagged a missing or malformed header.

# How to use

## 1. Run the existing script

The repository includes a script that handles this automatically:

```bash
uv run python scripts/fix_license_headers.py
```

This script:
- Recursively scans the repo for `.py`, `.sh`, `.yml`, `.yaml`, and `Dockerfile` files.
- Skips `.git/`, `.venv/`, `__pycache__/`, `dist/`, `build/`, and other generated directories.
- Adds the Apache-2.0 header if missing.
- Fixes the copyright line if it contains the wrong article (e.g., `a author` → `The author`).
- Splices the Apache 2.0 trailer block (`Unless required by applicable law...`) after the `LICENSE-2.0` line if missing.

## 2. Verify the header format

Every Python file should start with exactly this header:

```python
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
```

## 3. Check specific files

If you want to check a specific file or directory without running the full script:

```bash
# Check if a file has the header
grep -c "Copyright 2026" src/python/xerxes/my_new_module.py

# Find all Python files missing the header
find src/python/xerxes/ -name "*.py" -exec sh -c 'head -1 "$1" | grep -q "Copyright" || echo "$1"' _ {} \;
```

## 4. Manual fix for a single file

If the script misses a file or you need to add the header manually:

```python
# Read the file
with open("src/python/xerxes/my_new_module.py") as f:
    content = f.read()

# Add the header if missing
header = """# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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

"""

if not content.startswith("# Copyright"):
    with open("src/python/xerxes/my_new_module.py", "w") as f:
        f.write(header + content)
```

## 5. Verify after running

After running the script, check the diff to confirm changes:

```bash
git diff --stat
```

If the script touched files you didn't expect, review the changes before committing.

## Common pitfalls

- **Script modifies generated files:** The script should skip `__pycache__`, `.venv`, `dist`, `build`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, but if you have custom generated directories, add them to the skip list in `scripts/fix_license_headers.py`.
- **Wrong year:** The header says `2026`. If the year changes, update the script's `YEAR` constant.
- **Missing newline after header:** The script ensures a blank line after the header. If you add it manually, include the blank line.
- **Shell scripts with shebang:** For `.sh` files, the shebang (`#!/bin/bash`) must come BEFORE the copyright header. The script handles this correctly; manual edits must preserve this order.
- **YAML files with `---`:** For `.yml`/`.yaml` files, the copyright header is a comment block. The `---` YAML document separator should come AFTER the header, not before it.
