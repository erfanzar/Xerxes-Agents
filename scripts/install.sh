#!/usr/bin/env sh
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
            die "Windows is not supported by this script. Use WSL2 or install uv from https://astral.sh/uv." ;;
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
    if ! uv tool install --force --python ">=3.11,<3.14" "$spec"; then
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
██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗
╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝
 ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗
 ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║
██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
BANNER
    printf '%s\n' "${RESET}"
}

main() {
    print_banner
    detect_platform
    info "target: $PLATFORM"
    ensure_build_prereqs
    install_uv
    install_xerxes
    modify_path
    verify
    printf '\n'
    ok "done. Run xerxes --help to get started." "$BOLD" "$RESET"
    printf '   docs:  %s\n' "https://erfanzar.github.io/Xerxes-Agents"
    printf '   issues: %s/issues\n' "$REPO_URL"
}

main "$@"
