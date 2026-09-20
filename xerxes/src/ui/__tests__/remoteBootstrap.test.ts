// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it } from 'vitest'
import { mkdtemp, mkdir, chmod, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { remoteBootstrapScript } from '../lib/remoteBootstrap.js'

it('installs missing remote Xerxes, skips unchanged builds, updates and preserves previous releases', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bootstrap-'))
  const bin = join(root, 'bin'), home = join(root, 'home'), project = join(root, "project's $(touch nope)")
  await Promise.all([mkdir(bin), mkdir(home), mkdir(project)])
  const revision = join(root, 'revision'), calls = join(root, 'calls')
  const first = 'a'.repeat(40), second = 'b'.repeat(40)
  await Bun.write(revision, first)
  const executable = async (name: string, content: string) => { const file = join(bin, name); await Bun.write(file, '#!/bin/sh\nset -eu\n' + content); await chmod(file, 0o755) }
  await executable('git', 'if [ "$1" = ls-remote ]; then printf "%s\\trefs/heads/main\\n" "$(cat "$REVISION")"; fi\n')
  await executable('bun', `echo "$*" >> "$CALLS"
if [ "$1" = -e ]; then case "$2" in *renameSync*) exec "$REAL_BUN" "$@" ;; esac; fi
if [ "$1" = install ]; then if [ "\${FAIL_BUILD:-0}" = 1 ]; then printf 'private-sentinel-credential' >&2; exit 1; fi; exit; fi
if [ "$1" = run ]; then mkdir -p xerxes/dist/ui; touch xerxes/dist/cli.js xerxes/dist/ui/entry.js; fi
if [ "$1" != install ] && [ "$1" != run ] && [ "\${2:-}" != --help ]; then printf '%s' "$PWD" > "$HOME/opened"; fi
`)
  const run = async (extra: Record<string, string> = {}) => {
    const child = Bun.spawn(['sh', '-c', remoteBootstrapScript(project)], { env: { ...process.env, HOME: home, PATH: bin + ':/usr/bin:/bin', REVISION: revision, CALLS: calls, REAL_BUN: process.execPath, ...extra }, stdout: 'pipe', stderr: 'pipe' })
    return { code: await child.exited, output: await new Response(child.stdout).text(), error: await new Response(child.stderr).text() }
  }
  try {
    expect((await run()).code).toBe(0)
    expect(await Bun.file(join(home, 'opened')).text()).toBe(project)
    expect(await Bun.file(join(home, '.xerxes/remote-runtime', first, '.ready')).exists()).toBe(true)
    expect((await run()).output).not.toContain('installing/updating')
    expect((await Bun.file(calls).text()).match(/install --frozen/g)).toHaveLength(1)
    await Bun.write(revision, second)
    const failed = await run({ FAIL_BUILD: '1' })
    expect(failed.code).not.toBe(0)
    expect(failed.error).toContain('Dependency installation or build failed')
    expect(failed.error + failed.output).not.toContain('private-sentinel-credential')
    const log = join(home, '.xerxes/remote-runtime/setup.log')
    expect(await Bun.file(log).text()).toBe('Dependency installation or build failed.\n')
    expect((await stat(log)).mode & 0o777).toBe(0o600)
    expect(await Bun.file(join(home, '.xerxes/remote-runtime', second, '.ready')).exists()).toBe(false)
    expect((await run()).code).toBe(0)
    expect(await Bun.file(join(home, '.xerxes/remote-runtime', first, '.ready')).exists()).toBe(true)
    expect(await Bun.file(join(home, '.xerxes/remote-runtime', second, '.ready')).exists()).toBe(true)
    // Simulate a host without Bun; the bootstrap must fetch an installer and
    // explicitly add its user-owned binary to PATH in the same SSH session.
    await Bun.write(join(root, 'bun-fixture'), await Bun.file(join(bin, 'bun')).text())
    await rm(join(bin, 'bun'))
    await executable('curl', 'while [ "$1" != -o ]; do shift; done; cp "$INSTALLER_FIXTURE" "$2"\n')
    await executable('unzip', 'exit 0\n')
    await Bun.write(join(root, 'installer'), 'mkdir -p "$BUN_INSTALL/bin"\ncp "$BUN_FIXTURE" "$BUN_INSTALL/bin/bun"\nchmod +x "$BUN_INSTALL/bin/bun"\n')
    const installed = await run({ INSTALLER_FIXTURE: join(root, 'installer'), BUN_FIXTURE: join(root, 'bun-fixture') })
    expect(installed.code).toBe(0)
    expect(installed.output).toContain('installing/updating Bun')
    expect(await Bun.file(join(home, '.bun/bin/bun')).exists()).toBe(true)
  } finally { await rm(root, { recursive: true, force: true }) }
})

// Exercise the actual OS lock and shell bootstrap, including crash boundaries.
async function setupFixture() {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-setup-lock-'))
  const bin = join(root, 'bin'), home = join(root, 'home'), project = join(root, 'workspace')
  const runtime = join(home, '.xerxes/remote-runtime'), revision = 'c'.repeat(40)
  await Promise.all([mkdir(bin), mkdir(runtime, { recursive: true }), mkdir(project)])
  const executable = async (name: string, content: string) => {
    const path = join(bin, name)
    await Bun.write(path, '#!/bin/sh\nset -eu\n' + content)
    await chmod(path, 0o755)
  }
  await executable('git', `if [ "$1" = ls-remote ]; then printf '${revision}\\trefs/heads/main\\n'; fi\n`)
  await executable('bun', `
if [ "$1" = -e ]; then case "$2" in *renameSync*) exec "$REAL_BUN" "$@" ;; esac; exit 0; fi
if [ "$1" = install ]; then
  echo install >> "$HOME/calls"
  touch "$HOME/entered"
  if [ "\${HOLD:-0}" = 1 ]; then while [ ! -f "$HOME/release" ]; do sleep 0.05; done; fi
fi
if [ "$1" = run ]; then
  mkdir -p xerxes/dist/ui; touch xerxes/dist/cli.js xerxes/dist/ui/entry.js
  if [ "\${LEGACY_PUBLISH:-0}" = 1 ]; then
    mkdir -p "$HOME/.xerxes/remote-runtime/${revision}"
    touch "$HOME/.xerxes/remote-runtime/${revision}/.ready" "$HOME/.xerxes/remote-runtime/${revision}/legacy-winner"
  fi
fi
`)
  const children: ReturnType<typeof Bun.spawn>[] = []
  const start = (extra: Record<string, string> = {}, shortWait = false) => {
    // Shorten only the lock deadline for the timeout regression.
    const script = remoteBootstrapScript(project)
    const child = Bun.spawn(['sh', '-c', shortWait ? script.replaceAll(' 240 ', ' 1 ') : script], {
      env: { ...process.env, HOME: home, PATH: bin + ':/usr/bin:/bin', REAL_BUN: process.execPath, ...extra },
      stdout: 'pipe', stderr: 'pipe', detached: true,
    })
    children.push(child)
    const result = Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()])
      .then(([code, output, error]) => ({ code, output, error }))
    return { child, result, stop: (signal: NodeJS.Signals = 'SIGTERM') => process.kill(-child.pid, signal) }
  }
  const entered = async () => {
    const deadline = Date.now() + 5000
    while (!await Bun.file(join(home, 'entered')).exists()) {
      if (Date.now() >= deadline) throw new Error('Fixture did not enter setup')
      await Bun.sleep(10)
    }
  }
  const finish = () => Bun.write(join(home, 'release'), '')
  const cleanup = async () => {
    for (const child of children) if (child.exitCode === null) {
      try { process.kill(-child.pid, 'SIGKILL') } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error }
    }
    await Promise.all(children.map(child => child.exited))
    await rm(root, { recursive: true, force: true })
  }
  return { home, runtime, revision, start, entered, finish, cleanup }
}

it('automatically bypasses a stale legacy lock without deleting another client’s marker', async () => {
  const f = await setupFixture()
  try {
    await mkdir(join(f.runtime, 'setup.lock'))
    await Bun.write(join(f.runtime, 'setup.lock', 'untouched'), 'legacy owner data')
    expect((await f.start().result).code).toBe(0)
    expect(await Bun.file(join(f.runtime, f.revision, '.ready')).exists()).toBe(true)
    expect(await Bun.file(join(f.runtime, 'setup.lock', 'untouched')).text()).toBe('legacy owner data')
    expect((await stat(join(f.runtime, 'setup.guard'))).mode & 0o777).toBe(0o600)
  } finally { await f.cleanup() }
})

it('waits for a concurrent setup and reuses its build instead of building twice', async () => {
  const f = await setupFixture()
  try {
    const first = f.start({ HOLD: '1' })
    await f.entered()
    const second = f.start()
    await Bun.sleep(150)
    expect(second.child.exitCode).toBeNull()
    expect(await Bun.file(join(f.home, 'calls')).text()).toBe('install\n')
    await f.finish()
    expect((await first.result).code).toBe(0)
    expect((await second.result).code).toBe(0)
    expect(await Bun.file(join(f.home, 'calls')).text()).toBe('install\n')
  } finally { await f.cleanup() }
})

it('recovers automatically after abrupt setup termination with the guard file still present', async () => {
  const f = await setupFixture()
  try {
    const first = f.start({ HOLD: '1' })
    await f.entered()
    first.stop('SIGKILL')
    expect((await first.result).code).not.toBe(0)
    expect(await Bun.file(join(f.runtime, 'setup.guard')).exists()).toBe(true)
    expect((await f.start().result).code).toBe(0)
    expect(await Bun.file(join(f.runtime, f.revision, '.ready')).exists()).toBe(true)
  } finally { await f.cleanup() }
})

it('cancelling a waiting connection leaves the active installer intact and reconnect succeeds', async () => {
  const f = await setupFixture()
  try {
    const first = f.start({ HOLD: '1' })
    await f.entered()
    const waiting = f.start()
    await Bun.sleep(150)
    waiting.stop()
    expect((await waiting.result).code).not.toBe(0)
    expect(first.child.exitCode).toBeNull()
    await f.finish()
    expect((await first.result).code).toBe(0)
    expect((await f.start().result).code).toBe(0)
    expect(await Bun.file(join(f.home, 'calls')).text()).toBe('install\n')
  } finally { await f.cleanup() }
})

it('keeps a complete release published concurrently by a legacy client', async () => {
  const f = await setupFixture()
  try {
    const result = await f.start({ LEGACY_PUBLISH: '1' }).result
    expect(result.code, result.error).toBe(0)
    expect(await Bun.file(join(f.runtime, f.revision, 'legacy-winner')).exists()).toBe(true)
  } finally { await f.cleanup() }
})

it('bounds a lock wait and offers reconnect without stealing the active setup', async () => {
  const f = await setupFixture()
  try {
    const first = f.start({ HOLD: '1' })
    await f.entered()
    const timedOut = await f.start({}, true).result
    expect(timedOut.code).not.toBe(0)
    expect(timedOut.error).toContain('Another setup is still working. Reconnect to wait for it')
    expect(first.child.exitCode).toBeNull()
    await f.finish()
    expect((await first.result).code).toBe(0)
    expect((await f.start().result).code).toBe(0)
  } finally { await f.cleanup() }
})

it('leaves a conflicting incomplete release intact and reports an actionable failure', async () => {
  const f = await setupFixture()
  try {
    await mkdir(join(f.runtime, f.revision))
    await Bun.write(join(f.runtime, f.revision, 'retained'), 'incomplete build')
    const result = await f.start().result
    expect(result.code).not.toBe(0)
    expect(result.error).toContain('Check for an incomplete release')
    expect(await Bun.file(join(f.runtime, f.revision, 'retained')).text()).toBe('incomplete build')
    expect(await Bun.file(join(f.runtime, f.revision, '.ready')).exists()).toBe(false)
  } finally { await f.cleanup() }
})
