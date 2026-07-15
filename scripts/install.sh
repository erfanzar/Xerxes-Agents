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

# Install the locked Bun workspace and production launchers.
set -eu

REPO_URL="https://github.com/erfanzar/Xerxes-Agents.git"
INSTALL_DIRECTORY="${XERXES_INSTALL_DIRECTORY:-$HOME/.xerxes-bun}"
BIN_DIRECTORY="${XERXES_BIN_DIRECTORY:-$HOME/.local/bin}"

info() { printf '%s\n' "==> $*"; }
ok() { printf '%s\n' "✓ $*"; }
die() { printf '%s\n' "x $*" >&2; exit 1; }

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

local_checkout_root() {
    script_path="${1:-$0}"
    case "$script_path" in
        */*) script_directory="$(CDPATH= cd "$(dirname "$script_path")" 2>/dev/null && pwd -P)" || return 1 ;;
        *) return 1 ;;
    esac
    repository_root="$(CDPATH= cd "$script_directory/.." 2>/dev/null && pwd -P)" || return 1
    [ -f "$repository_root/package.json" ] || return 1
    [ -f "$repository_root/bun.lock" ] || return 1
    [ -d "$repository_root/src/typescript" ] || return 1
    printf '%s\n' "$repository_root"
}

resolve_source() {
    if [ -n "${XERXES_SOURCE_DIRECTORY:-}" ]; then
        [ -d "$XERXES_SOURCE_DIRECTORY" ] || die "XERXES_SOURCE_DIRECTORY does not exist: $XERXES_SOURCE_DIRECTORY"
        (CDPATH= cd "$XERXES_SOURCE_DIRECTORY" 2>/dev/null && pwd -P) || die "cannot resolve XERXES_SOURCE_DIRECTORY"
        return 0
    fi
    if source_root="$(local_checkout_root "$0" 2>/dev/null)"; then
        printf '%s\n' "$source_root"
        return 0
    fi

    need_command git
    if [ -e "$INSTALL_DIRECTORY" ]; then
        die "install directory already exists: $INSTALL_DIRECTORY (remove it or set XERXES_INSTALL_DIRECTORY)"
    fi
    info "cloning native Bun source into $INSTALL_DIRECTORY"
    git clone --depth 1 "${XERXES_REPOSITORY_URL:-$REPO_URL}" "$INSTALL_DIRECTORY"
    printf '%s\n' "$INSTALL_DIRECTORY"
}

write_launcher() {
    source_root="$1"
    launcher_name="$2"
    command_prefix="${3:-}"
    mkdir -p "$BIN_DIRECTORY"
    launcher="$BIN_DIRECTORY/$launcher_name"
    temporary_launcher="$launcher.tmp.$$"
    cat > "$temporary_launcher" <<EOF
#!/usr/bin/env sh
exec bun "$source_root/src/typescript/dist/cli.js" $command_prefix "\$@"
EOF
    chmod 755 "$temporary_launcher"
    mv "$temporary_launcher" "$launcher"
    ok "installed native launcher at $launcher"
    case ":$PATH:" in
        *":$BIN_DIRECTORY:"*) ;;
        *) printf '%s\n' "Add $BIN_DIRECTORY to PATH to invoke xerxes directly." ;;
    esac
}

main() {
    need_command bun
    source_root="$(resolve_source)"
    [ -f "$source_root/package.json" ] || die "native package manifest is missing: $source_root/package.json"
    [ -f "$source_root/bun.lock" ] || die "native lockfile is missing: $source_root/bun.lock"

    info "installing locked Bun workspace dependencies"
    (
        cd "$source_root"
        bun install --frozen-lockfile
        bun run build
    )
    [ -f "$source_root/src/typescript/dist/cli.js" ] || die "runtime build is missing: $source_root/src/typescript/dist/cli.js"
    [ -f "$source_root/src/typescript/dist/ui/entry.js" ] || die "TUI build is missing: $source_root/src/typescript/dist/ui/entry.js"
    write_launcher "$source_root" xerxes
    write_launcher "$source_root" xerxes-acp acp
    "$BIN_DIRECTORY/xerxes" --help >/dev/null
    ok "Xerxes Bun runtime is ready"
}

if [ "${XERXES_INSTALLER_SOURCE_ONLY:-0}" != "1" ]; then
    main "$@"
fi
