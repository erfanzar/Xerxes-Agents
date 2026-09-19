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
if [ "$1" = install ]; then if [ "\${FAIL_BUILD:-0}" = 1 ]; then printf 'private-sentinel-credential' >&2; exit 1; fi; exit; fi
if [ "$1" = run ]; then mkdir -p xerxes/dist/ui; touch xerxes/dist/cli.js xerxes/dist/ui/entry.js; fi
if [ "$1" != install ] && [ "$1" != run ] && [ "\${2:-}" != --help ]; then printf '%s' "$PWD" > "$HOME/opened"; fi
`)
  const run = async (extra: Record<string, string> = {}) => {
    const child = Bun.spawn(['sh', '-c', remoteBootstrapScript(project)], { env: { ...process.env, HOME: home, PATH: bin + ':/usr/bin:/bin', REVISION: revision, CALLS: calls, ...extra }, stdout: 'pipe', stderr: 'pipe' })
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
