// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { prepareManagedRuntime } from './managedRuntime.js'

/** Executed by SSH on the selected host. Installs only in a dedicated user-owned directory. */
export function remoteBootstrapScript(workspacePath: string, mode: 'tui' | 'daemon' = 'tui'): string {
  const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`
  const prepare = `
const { dirname, resolve } = await import('node:path');
const { pathToFileURL } = await import('node:url');
const cli = process.argv.at(-1);
const dist = dirname(cli);
const { GatewayClient, daemonPaths, resolveProjectDir } = await import(pathToFileURL(resolve(dist, '../src/ui/gatewayClient.ts')).href);
const projectDir = resolveProjectDir(process.cwd());
const expectedDaemonBuildId = (await Bun.file(resolve(dist, 'build-id')).text()).trim();
delete process.env.XERXES_EXPECTED_DAEMON_BUILD_ID;
const prepareManagedRuntime = ${prepareManagedRuntime.toString()};
const result = await prepareManagedRuntime(
  verify => new GatewayClient({ projectDir, bunBinary: process.execPath, bunDaemonPath: cli, ...(verify ? { expectedDaemonBuildId } : {}) }),
  expectedDaemonBuildId,
  async pid => {
    const deadline = Date.now() + 15000;
    for (;;) {
      try { process.kill(pid, 0); } catch (error) { if (error.code === 'ESRCH') return; throw error; }
      if (Date.now() >= deadline) throw new Error('Remote runtime did not finish stopping.');
      await Bun.sleep(50);
    }
  },
  {
    // An old runtime may not report its pid. Its pid file is used only when
    // that process is alive and is a Xerxes daemon — never a stale number.
    pidFallback: async () => {
      try {
        const pid = Number((await Bun.file(daemonPaths(projectDir).pidPath).text()).trim());
        if (!Number.isSafeInteger(pid) || pid <= 0) return undefined;
        const command = Bun.spawnSync(['ps', '-o', 'command=', '-p', String(pid)]).stdout.toString();
        return /\bdaemon\b/.test(command) && /xerxes|cli\.js/i.test(command) ? pid : undefined;
      } catch { return undefined; }
    },
  },
);
console.log('XERXES_REMOTE_READY ' + JSON.stringify({ projectDir, socketPath: daemonPaths(projectDir).socketPath, expectedBuildId: expectedDaemonBuildId, busy: result.busy }));
process.exit(0);
`
  // An older bootstrap may publish concurrently during the one-time lock migration.
  // Accept its complete release without merging or replacing populated releases.
  const publish = `
const { renameSync, existsSync } = require('node:fs');
const [stage, release] = process.argv.slice(-2);
try { renameSync(stage, release); }
catch (error) {
  if (!['EEXIST', 'ENOTEMPTY'].includes(error.code) || !existsSync(release + '/.ready')) process.exit(1);
}
`
  const setup = `set -eu
umask 077
root="$1"
repo="$2"
revision="$3"
release="$root/$revision"
log="$root/setup.log"
fail() { printf '\\nXerxes remote setup failed: %s\\n' "$*" >&2; printf '%s\\n' "$*" > "$log"; chmod 600 "$log"; exit 1; }
bun_ready() { command -v bun >/dev/null 2>&1 && bun -e 'process.exit(Bun.semver.satisfies(Bun.version, ">=1.3.0") ? 0 : 1)' >/dev/null 2>&1; }
stage=''
cleanup() { [ -z "$stage" ] || rm -rf "$stage"; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf '%s\\n' 'Preparing remote runtime' > "$log"
chmod 600 "$log"
if ! bun_ready; then
  printf 'Xerxes · installing/updating Bun in ~/.bun…\\n'
  for tool in curl bash unzip; do command -v "$tool" >/dev/null 2>&1 || fail "Install $tool on this host, then reconnect."; done
  stage=$(mktemp -d "$root/setup.XXXXXX")
  curl --connect-timeout 15 --max-time 120 -fsSL https://bun.sh/install -o "$stage/bun-install.sh" >/dev/null 2>&1 || fail 'Could not download Bun.'
  BUN_INSTALL="$HOME/.bun" bash "$stage/bun-install.sh" >/dev/null 2>&1 || fail 'Bun installation failed.'
  rm -rf "$stage"; stage=''
  bun_ready || fail 'Bun 1.3+ is still unavailable after installation.'
fi
if [ ! -f "$release/.ready" ]; then
  printf 'Xerxes · installing/updating remote build %s…\\n' "$revision"
  stage=$(mktemp -d "$root/setup.XXXXXX")
  git -C "$stage" init -q >/dev/null 2>&1 || fail 'Could not initialize managed checkout.'
  git -C "$stage" fetch --depth 1 "$repo" "$revision" >/dev/null 2>&1 || fail 'Could not download Xerxes.'
  git -C "$stage" checkout --detach FETCH_HEAD >/dev/null 2>&1 || fail 'Could not check out Xerxes.'
  printf 'Xerxes · installing dependencies and building for this machine…\\n'
  (cd "$stage" && bun install --frozen-lockfile && bun run build) >/dev/null 2>&1 || fail 'Dependency installation or build failed.'
  [ -f "$stage/xerxes/dist/cli.js" ] && [ -f "$stage/xerxes/dist/ui/entry.js" ] || fail 'Build artifacts are missing.'
  bun "$stage/xerxes/dist/cli.js" --help >/dev/null 2>&1 || fail 'The installed runtime did not start.'
  touch "$stage/.ready"
  # rename is atomic and must not move a stage *inside* a winning release.
  # An older client may still be building under the retired directory lock.
  bun -e ${quote(publish)} "$stage" "$release" >/dev/null 2>&1 || fail 'Could not publish the build. Check for an incomplete release, permissions, or insufficient space in ~/.xerxes/remote-runtime, then reconnect.'
  rm -rf "$stage"; stage=''
fi
cleanup
trap - EXIT INT TERM HUP
`
  return `set -eu
umask 077
cd ${quote(workspacePath)}
PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"
export PATH
root="$HOME/.xerxes/remote-runtime"
repo='https://github.com/erfanzar/Xerxes-Agents.git'
mkdir -p "$root"
log="$root/setup.log"
fail() { printf '\\nXerxes remote setup failed: %s\\n' "$*" >&2; printf '%s\\n' "$*" > "$log"; chmod 600 "$log"; exit 1; }
printf 'Xerxes · checking remote installation…\\n'
command -v git >/dev/null 2>&1 || fail 'Install Git on this host, then reconnect.'
export GIT_TERMINAL_PROMPT=0
revision=$(git ls-remote "$repo" refs/heads/main 2>/dev/null) || fail 'Cannot check GitHub for updates. Check network access and reconnect.'
revision=\${revision%%[[:space:]]*}
case "$revision" in ''|*[!0-9a-f]*) fail 'GitHub returned an invalid revision.' ;; esac
[ \${#revision} -eq 40 ] || fail 'GitHub returned an invalid revision length.'
release="$root/$revision"
bun_ready() { command -v bun >/dev/null 2>&1 && bun -e 'process.exit(Bun.semver.satisfies(Bun.version, ">=1.3.0") ? 0 : 1)' >/dev/null 2>&1; }
if [ ! -f "$release/.ready" ] || ! bun_ready; then
  # Kernel-owned locks survive concurrent clients but never leave stale ownership.
  # Keep this file: unlinking it could let clients lock different inodes. The old
  # setup.lock directory is deliberately not touched; an older client may own it.
  printf 'Xerxes · waiting for remote setup, then checking the build…\\n'
  setup_status=0
  if command -v flock >/dev/null 2>&1; then
    flock -w 240 -E 75 "$root/setup.guard" sh -c ${quote(setup)} sh "$root" "$repo" "$revision" || setup_status=$?
  elif command -v lockf >/dev/null 2>&1; then
    lockf -k -s -t 240 "$root/setup.guard" sh -c ${quote(setup)} sh "$root" "$repo" "$revision" || setup_status=$?
  else
    fail 'Safe remote setup requires flock (Linux util-linux) or lockf (macOS/BSD). Install one on this host, then reconnect.'
  fi
  [ "$setup_status" -ne 75 ] || fail 'Another setup is still working. Reconnect to wait for it; active setup and running tasks were left untouched.'
  [ "$setup_status" -eq 0 ] || exit "$setup_status"
fi
printf '%s\\n' 'Remote runtime ready' > "$log"
chmod 600 "$log"
printf 'Xerxes · ready. Opening remote workspace…\\n'
${mode === 'daemon' ? `exec bun -e ${quote(prepare)} "$release/xerxes/dist/cli.js"` : 'exec bun "$release/xerxes/dist/cli.js"'}`
}
