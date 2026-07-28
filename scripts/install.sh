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

# Presentation. Colour is emitted only for an interactive stdout with a capable
# TERM, and never when NO_COLOR is set; the glyphs degrade to ASCII without a
# UTF-8 locale. Piped and CI output therefore stays plain, which matters because
# this script's output is read by humans watching a curl|sh and by CI logs.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    C_RESET="$(printf '\033[0m')"
    C_BOLD="$(printf '\033[1m')"
    C_DIM="$(printf '\033[2m')"
    C_BLUE="$(printf '\033[38;5;69m')"
    C_CYAN="$(printf '\033[38;5;80m')"
    C_GREEN="$(printf '\033[38;5;71m')"
    C_YELLOW="$(printf '\033[38;5;179m')"
    C_RED="$(printf '\033[38;5;167m')"
else
    C_RESET='' C_BOLD='' C_DIM='' C_BLUE='' C_CYAN='' C_GREEN='' C_YELLOW='' C_RED=''
fi

case "${LC_ALL:-}${LC_CTYPE:-}${LANG:-}" in
    *UTF-8*|*utf8*|*UTF8*|*utf-8*) G_OK='✓' G_RUN='▸' G_WARN='!' G_BAD='✗' G_RULE='─' ;;
    *) G_OK='ok' G_RUN='>' G_WARN='!' G_BAD='x' G_RULE='-' ;;
esac

STEP_INDEX=0
TOTAL_STEPS=6

banner() {
    rule=''
    width=64
    i=0
    while [ "$i" -lt "$width" ]; do
        rule="$rule$G_RULE"
        i=$((i + 1))
    done
    printf '%s\n' "${C_BOLD}${C_CYAN}Xerxes installer${C_RESET} ${C_DIM}${rule}${C_RESET}"
}

# A numbered phase heading, so a reader watching a long install knows both where
# they are and how much is left. The old output was an undifferentiated list of
# "==>" lines with no sense of progress.
info() {
    STEP_INDEX=$((STEP_INDEX + 1))
    printf '%s\n' "${C_BLUE}${G_RUN}${C_RESET} ${C_DIM}[$STEP_INDEX/$TOTAL_STEPS]${C_RESET} ${C_BOLD}$*${C_RESET}"
}
ok() { printf '%s\n' "  ${C_GREEN}${G_OK}${C_RESET} $*"; }
note() { printf '%s\n' "  ${C_DIM}$*${C_RESET}"; }
warn() { printf '%s\n' "  ${C_YELLOW}${G_WARN}${C_RESET} $*" >&2; }
# Detail lines belonging to a warning. They follow the warning to stderr rather
# than stdout so a caller capturing stdout still sees a clean result and gets the
# whole diagnostic on the stream it belongs to.
warn_note() { printf '%s\n' "    ${C_DIM}$*${C_RESET}" >&2; }
die() {
    printf '%s\n' "${C_RED}${G_BAD} $*${C_RESET}" >&2
    exit 1
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

# Bun is the runtime everything else here depends on, so the installer bootstraps
# it rather than telling the user to go install it and come back. Auto-installing
# a language runtime is a real side effect, so it is announced and can be
# declined with XERXES_SKIP_BUN_INSTALL=1.
ensure_bun() {
    if command -v bun >/dev/null 2>&1; then
        ok "bun $(bun --version 2>/dev/null || printf 'present')"
        return 0
    fi
    if [ "${XERXES_SKIP_BUN_INSTALL:-0}" = "1" ]; then
        die "bun is not installed and XERXES_SKIP_BUN_INSTALL=1; install Bun from https://bun.sh and re-run"
    fi

    note "bun was not found; installing it from https://bun.sh"
    # bun.sh/install is a bash script that fetches and unpacks a release archive,
    # so all three are prerequisites even though this installer itself is POSIX sh.
    need_command curl
    need_command unzip
    need_command bash
    # Deliberately NOT `curl … | bash`: a pipeline's status is the last command's,
    # so a failed download would be invisible — bash would receive empty input,
    # exit 0, and the failure would surface later as a confusing "bun is not on
    # PATH". Fetching first makes the download's own status checkable, and `sh`
    # cannot rely on pipefail.
    bun_install_script="$(mktemp "${TMPDIR:-/tmp}/xerxes-bun-install.XXXXXX")" \
        || die "cannot create a temporary file for the Bun installer"
    bun_install_log="$(mktemp "${TMPDIR:-/tmp}/xerxes-bun-install-log.XXXXXX")" \
        || die "cannot create a temporary file for the Bun installer log"
    if ! curl -fsSL https://bun.sh/install -o "$bun_install_script"; then
        rm -f "$bun_install_script" "$bun_install_log"
        die "could not download the Bun installer from https://bun.sh/install; check network access and re-run"
    fi
    # Output is captured so a successful install stays quiet, and replayed
    # verbatim on failure: "the Bun installer failed" with no detail leaves the
    # user nothing to act on.
    if bash "$bun_install_script" >"$bun_install_log" 2>&1; then
        rm -f "$bun_install_script" "$bun_install_log"
    else
        printf '%s\n' "${C_DIM}--- Bun installer output ---${C_RESET}" >&2
        cat "$bun_install_log" >&2
        rm -f "$bun_install_script" "$bun_install_log"
        die "the Bun installer failed; install Bun manually from https://bun.sh and re-run"
    fi

    # The Bun installer edits shell rc files, which cannot affect this already
    # running shell, so the new binary is put on PATH explicitly for the rest of
    # the run.
    BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
    export BUN_INSTALL
    PATH="$BUN_INSTALL/bin:$PATH"
    export PATH

    command -v bun >/dev/null 2>&1 \
        || die "Bun installed to $BUN_INSTALL but is still not on PATH; open a new terminal and re-run"
    ok "installed bun $(bun --version 2>/dev/null || printf 'unknown')"
    note "Bun was added to PATH for this run; open a new terminal for other sessions to see it"
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

    warn "$running_count running Xerxes process(es) still have the previous build loaded."
    warn_note "Exit open Xerxes TUI/daemon processes, then launch xerxes again to use this install."
    warn_note "The installer leaves active sessions running so it cannot destroy in-progress work."
}

main() {
    banner

    info "checking prerequisites"
    ensure_bun
    prepare_bin_directory
    ok "launchers will be installed to $BIN_DIRECTORY"

    info "resolving source"
    source_root="$(resolve_source)"
    [ -f "$source_root/package.json" ] || die "native package manifest is missing: $source_root/package.json"
    [ -f "$source_root/bun.lock" ] || die "native lockfile is missing: $source_root/bun.lock"
    ok "$source_root"

    info "installing locked workspace dependencies"
    (
        cd "$source_root"
        bun install --frozen-lockfile
    ) || die "bun install failed in $source_root"
    ok "dependencies installed from the lockfile"

    info "building the runtime and terminal interface"
    (
        cd "$source_root"
        bun run build
    ) || die "bun run build failed in $source_root"
    [ -f "$source_root/xerxes/dist/cli.js" ] || die "runtime build is missing: $source_root/xerxes/dist/cli.js"
    [ -f "$source_root/xerxes/dist/ui/entry.js" ] || die "TUI build is missing: $source_root/xerxes/dist/ui/entry.js"
    ok "runtime and TUI built"

    info "installing launchers"
    remove_legacy_xerxes_aliases
    write_launcher "$source_root" xerxes
    write_launcher "$source_root" xerxes-acp acp
    persist_bin_path

    info "verifying the installation"
    "$BIN_DIRECTORY/xerxes" --help >/dev/null || die "the installed launcher did not run successfully"
    ok "launcher runs"
    warn_running_xerxes_processes "$source_root"

    printf '\n'
    printf '%s\n' "${C_GREEN}${C_BOLD}${G_OK} Xerxes is ready.${C_RESET}"
    printf '%s\n' "  ${C_DIM}Open a new terminal, then run${C_RESET} ${C_CYAN}xerxes${C_RESET}"
}

if [ "${XERXES_INSTALLER_SOURCE_ONLY:-0}" != "1" ]; then
    main "$@"
fi
