#!/usr/bin/env sh
# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.
#
# Xerxes installer — zero-dependency bootstrap.
#
#   curl -fsSL https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/scripts/install.sh | sh
#
# What it does, in order:
#   1. Detect OS + arch, refuse on unsupported targets.
#   2. Install uv (Astral's Python manager) for fast resolution + venv.
#   3. Install the `xerxes` CLI as an isolated uv tool from GitHub.
#   4. Ensure Node.js ≥20 is available (required by the TypeScript CLI at runtime).
#   5. Ensure ~/.local/bin is on PATH.
#
# Environment overrides:
#   XERXES_VERSION      Pin a specific PyPI version (default: latest).
#   XERXES_REF          Install from a git ref instead of PyPI.
#   XERXES_NO_NODE_CHECK=1  Skip the Node.js version check.
#   XERXES_NO_MODIFY_PATH=1  Do not touch shell rc files.
#   XERXES_INSTALL_EXTRAS Comma-separated PEP 508 extras (e.g. "tui,server").

set -eu

REPO_URL="https://github.com/erfanzar/Xerxes-Agents"
RAW_URL="https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main"

RED=""
GREEN=""
YELLOW=""
BLUE=""
BOLD=""
RESET=""
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    RED="$(tput setaf 1)"
    GREEN="$(tput setaf 2)"
    YELLOW="$(tput setaf 3)"
    BLUE="$(tput setaf 4)"
    BOLD="$(tput bold)"
    RESET="$(tput sgr0)"
fi

info()    { printf '%s==>%s %s\n'   "$BLUE"   "$RESET" "$*"; }
ok()      { printf '%s✓%s %s\n'     "$GREEN"  "$RESET" "$*"; }
warn()    { printf '%s!%s %s\n'     "$YELLOW" "$RESET" "$*" >&2; }
die()     { printf '%sx%s %s\n'     "$RED"    "$RESET" "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

download() {
    # download URL -> stdout, preferring curl, falling back to wget.
    url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 1 "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url"
    else
        die "neither curl nor wget is available"
    fi
}

detect_platform() {
    uname_s="$(uname -s 2>/dev/null || echo unknown)"
    uname_m="$(uname -m 2>/dev/null || echo unknown)"
    case "$uname_s" in
        Linux)  os=linux ;;
        Darwin) os=macos ;;
        MINGW*|MSYS*|CYGWIN*)
            die "Windows is not supported by this script. Use WSL2 or PowerShell with uv + rustup-init.exe." ;;
        *) die "unsupported OS: $uname_s" ;;
    esac
    case "$uname_m" in
        x86_64|amd64) arch=x86_64 ;;
        arm64|aarch64) arch=aarch64 ;;
        *) die "unsupported architecture: $uname_m" ;;
    esac
    PLATFORM="${os}-${arch}"
}

ensure_build_prereqs() {
    # Only surface missing-header warnings; don't hard-fail. uv will
    # try to use prebuilt wheels first.
    case "$PLATFORM" in
        linux-*)
            if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
                warn "no C compiler found on PATH; native builds will fail."
                warn "install build-essential (Debian/Ubuntu) or gcc (RHEL/Fedora) first if prebuilt wheels are missing."
            fi
            ;;
        macos-*)
            if ! xcode-select -p >/dev/null 2>&1; then
                warn "Xcode command-line tools not found. Run: xcode-select --install"
            fi
            ;;
    esac
}

check_node() {
    if [ "${XERXES_NO_NODE_CHECK:-0}" = "1" ]; then
        info "skipping Node.js check (XERXES_NO_NODE_CHECK=1)"
        return 0
    fi
    if command -v node >/dev/null 2>&1; then
        node_version=$(node --version 2>/dev/null | sed 's/^v//')
        major=$(echo "$node_version" | cut -d. -f1)
        if [ "$major" -ge 20 ] 2>/dev/null; then
            ok "Node.js $node_version already present (≥20 required)"
            return 0
        fi
        warn "Node.js $node_version found but ≥20 required"
    fi
    warn "Node.js ≥20 not found. The Xerxes CLI requires Node.js at runtime."
    warn "Install from https://nodejs.org or run: brew install node"
}

install_bun() {
    if command -v bun >/dev/null 2>&1; then
        ok "bun already present ($(bun --version))"
        return 0
    fi
    info "installing bun (required for CLI build)"
    # Try curl install first.
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://bun.sh/install | bash
    else
        warn "curl not found — cannot auto-install bun"
        warn "Install manually from https://bun.sh"
        return 0
    fi
    # shellcheck disable=SC1091
    if [ -f "$HOME/.bun/bin/bun" ]; then
        export PATH="$HOME/.bun/bin:$PATH"
    fi
    if command -v bun >/dev/null 2>&1; then
        ok "bun installed ($(bun --version))"
    else
        warn "bun installed but not on PATH; you may need to restart your shell"
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        ok "uv already present ($(uv --version))"
        return 0
    fi
    info "installing uv (Astral)"
    download "https://astral.sh/uv/install.sh" | sh
    # Common install locations — surface them so the rest of the script can find `uv`.
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        case ":$PATH:" in
            *":$candidate:"*) ;;
            *) [ -d "$candidate" ] && PATH="$candidate:$PATH" ;;
        esac
    done
    export PATH
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; restart your shell and re-run"
    ok "uv installed ($(uv --version))"
}

install_xerxes() {
    # Clean up stale/broken installs from previous attempts.
    info "cleaning up stale installs"
    uv tool uninstall xerxes-agent 2>/dev/null || true
    # Also remove any old pip-installed entry points that might shadow uv's.
    for old_bin in "$HOME/.local/bin/xerxes" "$HOME/.cargo/bin/xerxes"; do
        [ -f "$old_bin" ] && rm -f "$old_bin" && ok "removed stale binary: $old_bin"
    done

    # Default to git install since the package is not yet on PyPI.
    spec="xerxes-agent @ git+${REPO_URL}.git"
    if [ -n "${XERXES_REF:-}" ]; then
        spec="xerxes-agent @ git+${REPO_URL}.git@${XERXES_REF}"
    elif [ -n "${XERXES_VERSION:-}" ]; then
        spec="xerxes-agent==${XERXES_VERSION}"
    fi

    if [ -n "${XERXES_INSTALL_EXTRAS:-}" ]; then
        case "$spec" in
            "xerxes-agent @ "*) spec="xerxes-agent[${XERXES_INSTALL_EXTRAS}] ${spec#xerxes-agent }" ;;
            "xerxes-agent=="*)  spec="xerxes-agent[${XERXES_INSTALL_EXTRAS}]==${spec#xerxes-agent==}" ;;
            "xerxes-agent")     spec="xerxes-agent[${XERXES_INSTALL_EXTRAS}]" ;;
        esac
    fi

    info "installing $spec via uv tool"
    # --force lets re-runs upgrade in place; uv tool sandboxes into ~/.local/share/uv/tools.
    if ! uv tool install --force --python ">=3.10,<3.14" "$spec"; then
        die "uv tool install failed"
    fi
    ok "xerxes installed"
}

modify_path() {
    if [ "${XERXES_NO_MODIFY_PATH:-0}" = "1" ]; then
        return 0
    fi
    bin_dir=""
    for candidate in "$HOME/.local/bin"; do
        [ -d "$candidate" ] || continue
        case ":$PATH:" in
            *":$candidate:"*) ;;
            *) bin_dir="$candidate"; break ;;
        esac
    done
    [ -z "$bin_dir" ] && return 0

    # Figure out which rc file the user's shell reads.
    rc_file=""
    case "${SHELL:-}" in
        */zsh)  rc_file="$HOME/.zshrc" ;;
        */bash) rc_file="$HOME/.bashrc"; [ -f "$HOME/.bash_profile" ] && rc_file="$HOME/.bash_profile" ;;
        */fish) rc_file="$HOME/.config/fish/config.fish" ;;
    esac
    [ -z "$rc_file" ] && rc_file="$HOME/.profile"

    line="export PATH=\"$bin_dir:\$PATH\""
    case "$rc_file" in
        *config.fish) line="set -gx PATH $bin_dir \$PATH" ;;
    esac

    if [ -f "$rc_file" ] && grep -Fq "$bin_dir" "$rc_file"; then
        return 0
    fi
    mkdir -p "$(dirname "$rc_file")"
    {
        printf '\n# Added by Xerxes installer\n'
        printf '%s\n' "$line"
    } >> "$rc_file"
    ok "added $bin_dir to PATH in $rc_file"
    warn "restart your shell or run: source $rc_file"
}

verify() {
    if ! command -v xerxes >/dev/null 2>&1; then
        warn "xerxes binary not on PATH yet — restart your shell or source your rc file."
        return 0
    fi
    if xerxes --version >/dev/null 2>&1; then
        ok "xerxes --version => $(xerxes --version 2>&1 | head -n1)"
    else
        warn "xerxes found but --version failed; the install may still be functional."
    fi
}

print_banner() {
    printf '%s\n' "${BOLD}"
    cat <<'BANNER'
  ________________________________________________
 /   __  __   ___   __  __   ____                  \
|   \ \/ /___  _ __   ___| |_ ___                |
|    \  // _ \|  _ \ / _ \ __/ __|               |
|    /  \ (_) | | | |  __/ |_\__ \               |
|   /_/\_\___/|_| |_|\___|\__|___/               |
  ------------------------------------------------
  Multi-agent orchestration framework v0.2.0
BANNER
    printf '%s\n' "${RESET}"
}

main() {
    print_banner
    detect_platform
    info "target: $PLATFORM"
    ensure_build_prereqs
    check_node
    install_uv
    install_bun
    install_xerxes
    modify_path
    verify
    printf '\n'
    ok "done. Run %sxerxes --help%s to get started." "$BOLD" "$RESET"
    printf '   docs:  %s\n' "https://erfanzar.github.io/Xerxes"
    printf '   issues: %s/issues\n' "$REPO_URL"
}

main "$@"
