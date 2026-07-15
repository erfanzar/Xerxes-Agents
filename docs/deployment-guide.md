# Deployment guide

Xerxes ships as a Bun workspace. Install Bun, fetch the repository, and install the locked
dependencies:

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

The local daemon uses the v35 Unix-socket path on supported hosts. On native Windows, use WSL2 or
configure the WebSocket control transport. Start with `bun run xerxes doctor` to verify the host
and provider setup.

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
docker run --rm --entrypoint bun xerxes:local /app/src/typescript/dist/ui/entry.js
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
