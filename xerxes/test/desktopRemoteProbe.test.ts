// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { remoteProbeCommand } from '../src/desktop/main/remote.js'

/** Run the exact command string ssh would hand the remote login shell, with a controlled HOME and PATH. */
async function runProbe(home: string, program: string): Promise<{ code: number; stdout: string }> {
  const child = Bun.spawn(['/bin/sh', '-c', remoteProbeCommand(program)], {
    env: { HOME: home, PATH: '/usr/bin:/bin' },
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const [stdout, code] = await Promise.all([new Response(child.stdout).text(), child.exited])
  return { code, stdout }
}

test('a host without Bun answers XERXES_REMOTE_MISSING instead of failing with 127', async () => {
  const home = await mkdtemp(join(tmpdir(), 'remote-probe-nobun-'))
  try {
    const result = await runProbe(home, "console.log('XERXES_REMOTE_ALIVE')")
    expect(result.code).toBe(0)
    expect(result.stdout.trim().split('\n')).toEqual(['XERXES_REMOTE_MISSING'])
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})

test('a host with Bun in ~/.bun/bin runs the probe program, including quotes in it', async () => {
  const home = await mkdtemp(join(tmpdir(), 'remote-probe-bun-'))
  try {
    await mkdir(join(home, '.bun', 'bin'), { recursive: true })
    await symlink(process.execPath, join(home, '.bun', 'bin', 'bun'))
    const result = await runProbe(home, "console.log('XERXES_REMOTE_ALIVE', \"it's\")")
    expect(result.code).toBe(0)
    expect(result.stdout.trim()).toBe("XERXES_REMOTE_ALIVE it's")
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})
