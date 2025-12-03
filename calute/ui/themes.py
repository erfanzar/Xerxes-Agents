# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Theme configuration for Calute Chainlit UI.

Generates Chainlit configuration files for theming and styling.
"""

import json
from pathlib import Path

APP_TITLE = "Calute"
APP_SUBTITLE = "Advanced conversational AI with reasoning and tool capabilities"

# OpenWebUI-inspired dark theme colors
COLORS = {
    "background": "#171717",
    "surface": "#262626",
    "surface_light": "#333333",
    "border": "#4e4e4e",
    "text": "#e3e3e3",
    "text_secondary": "#b4b4b4",
    "text_muted": "#9b9b9b",
    "success": "#10b981",
    "error": "#ef4444",
    "warning": "#f59e0b",
    "primary": "#676767",
    "code_bg": "#0d0d0d",
}

# Chainlit theme.json - HSL values without hsl() wrapper
CHAINLIT_THEME = {
    "variables": {
        "dark": {
            "--font-sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "--font-mono": "'JetBrains Mono', 'Fira Code', monospace",
            "--background": "0 0% 9%",
            "--foreground": "0 0% 89%",
            "--card": "0 0% 15%",
            "--card-foreground": "0 0% 89%",
            "--popover": "0 0% 15%",
            "--popover-foreground": "0 0% 89%",
            "--primary": "0 0% 40%",
            "--primary-foreground": "0 0% 89%",
            "--secondary": "0 0% 20%",
            "--secondary-foreground": "0 0% 89%",
            "--muted": "0 0% 15%",
            "--muted-foreground": "0 0% 61%",
            "--accent": "0 0% 20%",
            "--accent-foreground": "0 0% 89%",
            "--destructive": "0 62% 30%",
            "--destructive-foreground": "0 0% 98%",
            "--border": "0 0% 31%",
            "--input": "0 0% 31%",
            "--ring": "0 0% 40%",
            "--radius": "0.75rem",
        },
        "light": {
            "--font-sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "--font-mono": "'JetBrains Mono', 'Fira Code', monospace",
            "--background": "0 0% 100%",
            "--foreground": "0 0% 9%",
            "--card": "0 0% 100%",
            "--card-foreground": "0 0% 9%",
            "--popover": "0 0% 100%",
            "--popover-foreground": "0 0% 9%",
            "--primary": "0 0% 40%",
            "--primary-foreground": "0 0% 100%",
            "--secondary": "0 0% 96%",
            "--secondary-foreground": "0 0% 9%",
            "--muted": "0 0% 96%",
            "--muted-foreground": "0 0% 45%",
            "--accent": "0 0% 96%",
            "--accent-foreground": "0 0% 9%",
            "--destructive": "0 84% 60%",
            "--destructive-foreground": "0 0% 98%",
            "--border": "0 0% 90%",
            "--input": "0 0% 90%",
            "--ring": "0 0% 40%",
            "--radius": "0.5rem",
        },
    }
}

# Custom CSS for additional styling
CUSTOM_CSS = """
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Thinking step styling */
[data-testid="step"][data-type="llm"] {
    background: linear-gradient(135deg, hsl(0, 0%, 15%) 0%, hsl(0, 0%, 10%) 100%) !important;
    border-left: 3px solid hsl(0, 0%, 40%) !important;
    font-style: italic;
}

[data-testid="step"][data-type="llm"] .step-content {
    color: hsl(0, 0%, 61%) !important;
}

/* Tool step styling */
[data-testid="step"][data-type="tool"] {
    border-left: 3px solid hsl(142, 76%, 36%) !important;
}

/* Running tool animation */
[data-testid="step"][data-type="tool"].running {
    border-left-color: hsl(38, 92%, 50%) !important;
    animation: pulse-border 1.5s ease-in-out infinite;
}

@keyframes pulse-border {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Code block styling */
pre, code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

pre {
    background: hsl(0, 0%, 5%) !important;
    border: 1px solid hsl(0, 0%, 20%) !important;
    border-radius: 6px !important;
}

/* Message styling */
.message-content {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Hide watermark */
.watermark {
    display: none !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: hsl(0, 0%, 9%);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb {
    background: hsl(0, 0%, 31%);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: hsl(0, 0%, 40%);
}
"""

# Chainlit config.toml content - must match chainlit's expected format
CONFIG_TOML = f"""[project]
user_env = []
session_timeout = 3600
cache = false
allow_origins = ["*"]

[features]
unsafe_allow_html = true
latex = false

[features.spontaneous_file_upload]
enabled = true
accept = ["*/*"]
max_files = 20
max_size_mb = 500

# MCP (Model Context Protocol) configuration
[features.mcp]
enabled = true

[features.mcp.sse]
enabled = true

[features.mcp.streamable-http]
enabled = true

[features.mcp.stdio]
enabled = true
allowed_executables = ["npx", "uvx", "python", "node"]

[UI]
name = "{APP_TITLE}"
default_theme = "dark"
cot = "full"

[meta]
generated_by = "2.9.2"
"""


def setup_chainlit_theme(base_path: Path | str | None = None) -> None:
    """Generate Chainlit configuration files.

    Creates the necessary theme.json, custom.css, and config.toml files
    for Chainlit theming. Removes any existing .chainlit directory first
    to avoid version conflicts.

    Args:
        base_path: Base directory for config files. Defaults to current working directory.
    """
    import shutil

    base = Path(base_path) if base_path else Path.cwd()

    # Remove existing .chainlit directory to avoid version conflicts
    chainlit_dir = base / ".chainlit"
    if chainlit_dir.exists():
        shutil.rmtree(chainlit_dir)

    # Create fresh .chainlit directory
    chainlit_dir.mkdir(exist_ok=True)

    # Create public directory
    public_dir = base / "public"
    public_dir.mkdir(exist_ok=True)

    # Write theme.json
    theme_path = public_dir / "theme.json"
    with open(theme_path, "w") as f:
        json.dump(CHAINLIT_THEME, f, indent=2)

    # Write custom.css
    css_path = public_dir / "custom.css"
    with open(css_path, "w") as f:
        f.write(CUSTOM_CSS)

    # Write config.toml
    config_path = chainlit_dir / "config.toml"
    with open(config_path, "w") as f:
        f.write(CONFIG_TOML)


def get_theme_colors() -> dict[str, str]:
    """Get theme colors for programmatic use.

    Returns:
        Dictionary of color name to hex value.
    """
    return COLORS.copy()
