# Deployment guide

Xerxes is published to npm as `xerxes-bun` and requires Bun 1.3.12 or newer. Install the
package globally with either package manager:

```sh
bun add --global xerxes-bun
# or
npm install --global xerxes-bun

xerxes doctor
xerxes
```

The package provides `xerxes`, `xerxes-acp`, and the package-name alias `xerxes-bun`. To run it
without a global install, use `bunx xerxes-bun` or `npx --yes xerxes-bun`. The unscoped npm
package named `xerxes` is unrelated to this project; install `xerxes-bun`.

For source deployments, fetch the Bun workspace and install its locked dependencies:

```sh
git clone https://github.com/erfanzar/Xerxes-Agents.git
cd Xerxes-Agents
bun install --frozen-lockfile
bun run build
```

For a local launcher, run the native installer from a checkout:

```sh
sh scripts/install.sh
```

It requires Bun, installs the locked workspace, and writes `xerxes` to
`$XERXES_BIN_DIRECTORY` (default `~/.local/bin`). Set `XERXES_SOURCE_DIRECTORY` to install from a
specific existing checkout, or `XERXES_INSTALL_DIRECTORY` for the clone destination.

The sole terminal renderer is OpenTUI. After changing `xerxes/src/ui/`, rebuild only the UI
bundle with `bun run --cwd xerxes build:ui`; the generated entry is
`xerxes/dist/ui/entry.js`.

## CLI and daemon

```sh
# Interactive terminal client or one-shot turn
bun run xerxes
bun run xerxes "summarize this repository"

# Native local daemon
bun run xerxes daemon --project-dir /path/to/workspace

# Agent Client Protocol over stdio
bun run xerxes acp --project-dir /path/to/workspace
```

The local daemon speaks the v35 control protocol over a per-project channel: a Unix socket on macOS
and Linux, and a named pipe (`\\.\pipe\xerxes-<digest>`) on native Windows. `node:net` reaches both
through the same API, so nothing else about the transport differs. Start with
`bun run xerxes doctor` to verify the host and provider setup — the `platform` check names the
transport in use, and on Windows a `windows-tooling` check confirms `powershell.exe` and `cmd.exe`
are reachable (they back process-identity and clipboard access, where POSIX uses `ps`).

## Windows hosts

Native Windows is supported; WSL2 is no longer required. Install with the PowerShell script rather
than the shell one:

```powershell
# From a checkout
./scripts/install.ps1
```

It installs the locked Bun workspace, writes `xerxes.cmd` / `xerxes-acp.cmd` launchers into
`%LOCALAPPDATA%\Xerxes\bin` (a `.cmd` extension is required — Windows resolves PATH entries through
`PATHEXT`), and adds that directory to the user `PATH`. Override the locations with
`XERXES_BIN_DIRECTORY`, `XERXES_SOURCE_DIRECTORY`, and `XERXES_INSTALL_DIRECTORY` as on POSIX.

Windows-specific behaviour worth knowing:

- **Shell tools.** A PTY session launches `%COMSPEC%` (`cmd.exe`) instead of `$SHELL`, and interrupts
  are delivered as a Ctrl+C keystroke rather than SIGINT, which Windows cannot send to another
  process without killing it.
- **MCP servers.** A server launched as `npx …` resolves to a `.cmd` shim, which Windows cannot
  execute directly; Xerxes wraps those in `cmd.exe /d /s /c` automatically. An argument containing
  `%` is rejected rather than passed through, because `cmd` expands `%VAR%` even inside quotes.
- **Test suite.** `bun test` currently expects POSIX-absolute fixture paths in roughly 53 files, so
  the full suite does not yet pass on a Windows host. `bun test ./test/windowsSupport.test.ts` does,
  and CI runs it on `windows-latest` alongside a typecheck and a full build.

## Container daemon

The production image installs only runtime dependencies and starts the daemon as an unprivileged
`xerxes` user. Compose requires a control-plane token and publishes its WebSocket port only on the
host loopback interface. Generate a fresh token before resolving or starting the service:

```sh
export XERXES_DAEMON_TOKEN="$(bun -e 'console.log(crypto.randomUUID().replaceAll("-", "") + crypto.randomUUID().replaceAll("-", ""))')"

# Linux bind mounts: match the image account to the checkout owner.
export XERXES_UID="$(id -u)"

docker build -t xerxes:local .
docker run --rm xerxes:local --version
docker run --rm --entrypoint bun xerxes:local /app/xerxes/dist/ui/entry.js
docker compose build
docker compose run --rm --entrypoint sh xerxes -c \
  'probe=/workspace/.xerxes-write-probe-$$; test -w /workspace && : > "$probe" && rm "$probe"'
docker compose up --build
```

The non-TTY TUI command is a module-load smoke check and prints `xerxes-tui: no TTY`; use the
host launcher for the interactive terminal. Compose exposes the authenticated daemon WebSocket on
`127.0.0.1:11996`, persists `XERXES_HOME` in the `xerxes_home` volume, and mounts the checkout at
`/workspace`. On Linux, `XERXES_UID` must match the checkout owner so the unprivileged process can
write requested code changes. Docker Desktop users can keep the default UID. Supply provider
credentials through the environment rather than baking them into the image, and never commit the
daemon token.

## HTTP API embedding

The OpenAI-compatible HTTP handler is a native library surface, not an implicit background server.
An application supplies its `LlmClient`, advertised models, authentication policy, CORS policy, and
optional rate limiter before listening with Bun. See the [HTTP API reference](api-reference.md).

## Channels and remote services

Use an explicit daemon configuration and provider/channel credentials stored outside source
control. A configured channel only receives the adapters and credentials deliberately supplied to
the host. Browser automation, media APIs, hardware training, and persistent remote gateways remain
explicit integration boundaries; deployment must provide a real adapter rather than assuming one.

## Release checks

Before packaging a deployment artifact, run:

```sh
bun run verify
RELEASE_ROOT="$(mktemp -d)"
PACKAGE_DIR="$RELEASE_ROOT/package"
ARCHIVE="$RELEASE_ROOT/xerxes-bun-$(bun -p 'require("./package.json").version').tgz"
bun run release:prepare -- --output "$PACKAGE_DIR"
(
  cd "$PACKAGE_DIR"
  bun pm pack --filename "$ARCHIVE" --ignore-scripts
)
bun run release:check -- --package "$PACKAGE_DIR" --archive "$ARCHIVE"
bun run release:smoke -- "$ARCHIVE"
```

The native release helpers stage built artifacts, validate source/archive metadata and integrity,
install the packed artifact in an empty project, and smoke both the CLI and OpenTUI module. They do
not publish to a registry or alter a GitHub release.
