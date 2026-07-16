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

shell_single_quote() {
    escaped_value="$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
    printf "'%s'" "$escaped_value"
}

fish_single_quote() {
    escaped_value="$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e "s/'/\\\\'/g")"
    printf "'%s'" "$escaped_value"
}

prepare_bin_directory() {
    [ -n "$BIN_DIRECTORY" ] || die "XERXES_BIN_DIRECTORY cannot be empty"
    case "$BIN_DIRECTORY" in
        /*) ;;
        *) die "XERXES_BIN_DIRECTORY must be an absolute path: $BIN_DIRECTORY" ;;
    esac
    case "$BIN_DIRECTORY" in
        *:*) die "XERXES_BIN_DIRECTORY cannot contain a colon: $BIN_DIRECTORY" ;;
    esac
    case "$BIN_DIRECTORY" in
        *'
'*) die "XERXES_BIN_DIRECTORY cannot contain control characters" ;;
    esac
    carriage_return="$(printf '\r')"
    case "$BIN_DIRECTORY" in
        *"$carriage_return"*) die "XERXES_BIN_DIRECTORY cannot contain control characters" ;;
    esac
    if LC_ALL=C printf '%s' "$BIN_DIRECTORY" | grep '[[:cntrl:]]' >/dev/null 2>&1; then
        die "XERXES_BIN_DIRECTORY cannot contain control characters"
    fi
    mkdir -p "$BIN_DIRECTORY" || die "cannot create launcher directory: $BIN_DIRECTORY"
    BIN_DIRECTORY="$(CDPATH= cd "$BIN_DIRECTORY" 2>/dev/null && pwd -P)" \
        || die "cannot resolve launcher directory: $BIN_DIRECTORY"
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
    [ -d "$repository_root/xerxes" ] || return 1
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
        managed_root="$(CDPATH= cd "$INSTALL_DIRECTORY" 2>/dev/null && pwd -P)" \
            || die "cannot resolve managed install directory: $INSTALL_DIRECTORY"
        git_root="$(git -C "$managed_root" rev-parse --show-toplevel 2>/dev/null)" \
            || die "install directory is not a managed Git checkout: $managed_root"
        [ "$git_root" = "$managed_root" ] \
            || die "install directory is nested inside another Git checkout: $managed_root"
        [ -f "$managed_root/package.json" ] \
            || die "managed checkout package manifest is missing: $managed_root/package.json"
        [ -f "$managed_root/bun.lock" ] \
            || die "managed checkout lockfile is missing: $managed_root/bun.lock"
        [ -d "$managed_root/xerxes" ] \
            || die "managed checkout runtime directory is missing: $managed_root/xerxes"

        expected_remote="${XERXES_REPOSITORY_URL:-$REPO_URL}"
        actual_remote="$(git -C "$managed_root" remote get-url origin 2>/dev/null)" \
            || die "managed checkout has no origin remote: $managed_root"
        [ "$actual_remote" = "$expected_remote" ] \
            || die "managed checkout origin does not match $expected_remote: $actual_remote"
        managed_branch="$(git -C "$managed_root" symbolic-ref --quiet --short HEAD 2>/dev/null)" \
            || die "managed checkout is detached; refusing to update: $managed_root"
        [ "$managed_branch" = "main" ] \
            || die "managed checkout is on $managed_branch, expected main: $managed_root"
        managed_status="$(git -C "$managed_root" status --porcelain --untracked-files=normal)" \
            || die "cannot inspect managed checkout state: $managed_root"
        [ -z "$managed_status" ] \
            || die "managed checkout has local changes; refusing to update: $managed_root"

        info "updating native Bun source in $managed_root" >&2
        git -C "$managed_root" pull --ff-only origin main 1>&2 \
            || die "managed checkout cannot be fast-forwarded: $managed_root"
        printf '%s\n' "$managed_root"
        return 0
    fi
    info "cloning native Bun source into $INSTALL_DIRECTORY" >&2
    git clone --depth 1 "${XERXES_REPOSITORY_URL:-$REPO_URL}" "$INSTALL_DIRECTORY" 1>&2 \
        || die "could not clone native Bun source into $INSTALL_DIRECTORY"
    managed_root="$(CDPATH= cd "$INSTALL_DIRECTORY" 2>/dev/null && pwd -P)" \
        || die "cannot resolve managed install directory after clone: $INSTALL_DIRECTORY"
    printf '%s\n' "$managed_root"
}

write_launcher() {
    source_root="$1"
    launcher_name="$2"
    command_prefix="${3:-}"
    case "$command_prefix" in
        ""|acp) ;;
        *) die "unsupported launcher command prefix: $command_prefix" ;;
    esac
    launcher="$BIN_DIRECTORY/$launcher_name"
    temporary_launcher="$launcher.tmp.$$"
    quoted_entry="$(shell_single_quote "$source_root/xerxes/dist/cli.js")"
    if [ -n "$command_prefix" ]; then
        printf '%s\n' '#!/usr/bin/env sh' "exec bun $quoted_entry $command_prefix \"\$@\"" > "$temporary_launcher"
    else
        printf '%s\n' '#!/usr/bin/env sh' "exec bun $quoted_entry \"\$@\"" > "$temporary_launcher"
    fi
    chmod 755 "$temporary_launcher"
    mv "$temporary_launcher" "$launcher"
    ok "installed native launcher at $launcher"
}

write_path_block() {
    destination="$1"
    syntax="$2"
    if [ "$syntax" = "fish" ]; then
        quoted_bin="$(fish_single_quote "$BIN_DIRECTORY")"
        cat >> "$destination" <<EOF
# >>> xerxes PATH >>>
if contains -- $quoted_bin \$PATH
    set -e PATH[(contains -i -- $quoted_bin \$PATH)]
end
set -gx PATH $quoted_bin \$PATH
# <<< xerxes PATH <<<
EOF
        return 0
    fi
    quoted_bin="$(shell_single_quote "$BIN_DIRECTORY")"
    cat >> "$destination" <<EOF
# >>> xerxes PATH >>>
case "\$PATH" in
    $quoted_bin|$quoted_bin:*) ;;
    *) export PATH=$quoted_bin":\$PATH" ;;
esac
# <<< xerxes PATH <<<
EOF
}

configure_path_file() {
    shell_file="$1"
    syntax="$2"
    shell_directory="$(dirname "$shell_file")"
    mkdir -p "$shell_directory" || die "cannot create shell configuration directory: $shell_directory"
    if [ -e "$shell_file" ] && [ ! -f "$shell_file" ]; then
        die "shell configuration is not a regular file: $shell_file"
    fi
    [ -f "$shell_file" ] || : > "$shell_file"

    temporary_file="$shell_file.xerxes-path.$$"
    if ! (umask 077; awk '
        $0 == "# >>> xerxes PATH >>>" {
            if (in_block) invalid = 1
            in_block = 1
            next
        }
        $0 == "# <<< xerxes PATH <<<" {
            if (!in_block) invalid = 1
            in_block = 0
            next
        }
        !in_block { print }
        END { if (in_block || invalid) exit 2 }
    ' "$shell_file" > "$temporary_file"); then
        rm -f "$temporary_file"
        die "malformed Xerxes PATH block in $shell_file"
    fi
    write_path_block "$temporary_file" "$syntax"
    cat "$temporary_file" > "$shell_file"
    rm -f "$temporary_file"
    ok "configured $BIN_DIRECTORY on PATH in $shell_file"
}

persist_bin_path() {
    shell_path="${SHELL:-sh}"
    shell_name="${shell_path##*/}"
    case "$shell_name" in
        zsh)
            configure_path_file "${ZDOTDIR:-$HOME}/.zshrc" posix
            ;;
        bash)
            configure_path_file "$HOME/.bashrc" posix
            if [ -f "$HOME/.bash_profile" ]; then
                configure_path_file "$HOME/.bash_profile" posix
            elif [ -f "$HOME/.bash_login" ]; then
                configure_path_file "$HOME/.bash_login" posix
            else
                configure_path_file "$HOME/.profile" posix
            fi
            ;;
        fish)
            configure_path_file "${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/xerxes.fish" fish
            ;;
        *)
            configure_path_file "$HOME/.profile" posix
            ;;
    esac
}

remove_legacy_xerxes_aliases() {
    for shell_file in "${ZDOTDIR:-$HOME}/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.profile"; do
        [ -f "$shell_file" ] || continue
        grep -F '.xerxes-venv/bin/xerxes' "$shell_file" >/dev/null 2>&1 || continue

        temporary_file="$shell_file.xerxes.$$"
        if ! (umask 077; awk '
            function is_legacy_alias(line) {
                return line ~ /^[[:space:]]*alias[[:space:]]+xerxes=/ \
                    && index(line, ".xerxes-venv/bin/xerxes") > 0
            }
            $0 == "# >>> xerxes installer >>>" {
                in_block = 1
                block = $0 ORS
                legacy = 0
                next
            }
            in_block {
                block = block $0 ORS
                if (is_legacy_alias($0)) legacy = 1
                if ($0 == "# <<< xerxes installer <<<") {
                    if (!legacy) printf "%s", block
                    in_block = 0
                    block = ""
                    legacy = 0
                }
                next
            }
            is_legacy_alias($0) { next }
            { print }
            END { if (in_block) printf "%s", block }
        ' "$shell_file" > "$temporary_file"); then
            rm -f "$temporary_file"
            die "could not remove the retired Xerxes alias from $shell_file"
        fi
        cat "$temporary_file" > "$shell_file"
        rm -f "$temporary_file"
        ok "removed retired Xerxes alias from $shell_file"
    done
}

warn_running_xerxes_processes() {
    source_root="$1"
    if [ "${XERXES_INSTALLER_PROCESS_LIST+x}" = "x" ]; then
        process_listing="$XERXES_INSTALLER_PROCESS_LIST"
    elif command -v ps >/dev/null 2>&1; then
        process_listing="$(ps -Ao pid=,args= 2>/dev/null || true)"
    else
        return 0
    fi

    cli_entry="$source_root/xerxes/dist/cli.js"
    ui_entry="$source_root/xerxes/dist/ui/entry.js"
    running_count="$(printf '%s\n' "$process_listing" | awk -v cli="$cli_entry" -v ui="$ui_entry" '
        index($0, cli) || index($0, ui) { count += 1 }
        END { print count + 0 }
    ')"
    [ "$running_count" -gt 0 ] || return 0

    printf '%s\n' \
        "! $running_count running Xerxes process(es) still have the previous build loaded." \
        "! Exit open Xerxes TUI/daemon processes, then launch xerxes again to use this install." \
        "! The installer leaves active sessions running so it cannot destroy in-progress work." >&2
}

main() {
    need_command bun
    prepare_bin_directory
    source_root="$(resolve_source)"
    [ -f "$source_root/package.json" ] || die "native package manifest is missing: $source_root/package.json"
    [ -f "$source_root/bun.lock" ] || die "native lockfile is missing: $source_root/bun.lock"

    info "installing locked Bun workspace dependencies"
    (
        cd "$source_root"
        bun install --frozen-lockfile
        bun run build
    )
    [ -f "$source_root/xerxes/dist/cli.js" ] || die "runtime build is missing: $source_root/xerxes/dist/cli.js"
    [ -f "$source_root/xerxes/dist/ui/entry.js" ] || die "TUI build is missing: $source_root/xerxes/dist/ui/entry.js"
    remove_legacy_xerxes_aliases
    write_launcher "$source_root" xerxes
    write_launcher "$source_root" xerxes-acp acp
    persist_bin_path
    warn_running_xerxes_processes "$source_root"
    "$BIN_DIRECTORY/xerxes" --help >/dev/null
    ok "Xerxes Bun runtime is ready; open a new terminal to invoke xerxes"
}

if [ "${XERXES_INSTALLER_SOURCE_ONLY:-0}" != "1" ]; then
    main "$@"
fi
