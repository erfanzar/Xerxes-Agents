// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const PACKAGE_ROOT = resolve(import.meta.dirname, '../../..')

describe('production TUI build', () => {
  it('writes the entry-point artifact instead of validating a stale bundle', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'xerxes-ui-build-'))
    const output = join(directory, 'entry.js')
    try {
      const child = Bun.spawn([process.execPath, 'scripts/buildTui.ts'], {
        cwd: PACKAGE_ROOT,
        env: { ...process.env, XERXES_TUI_BUILD_OUT: output },
        stderr: 'pipe',
        stdout: 'pipe'
      })
      const [exitCode, stdout, stderr] = await Promise.all([
        child.exited,
        new Response(child.stdout).text(),
        new Response(child.stderr).text()
      ])

      expect(exitCode, `${stdout}\n${stderr}`).toBe(0)
      const bundle = await readFile(output, 'utf8')
      expect(bundle).toContain('Ready for your next command.')
      expect(bundle).toContain('Choose a model with /provider')
      expect(bundle).toContain('createRequire as __cr')
      expect(bundle).toContain('@opentui/core')
      expect(bundle.startsWith('#!')).toBe(false)
      expect(await Bun.file(join(directory, 'entry-opentui.js')).exists()).toBe(false)
    } finally {
      await rm(directory, { force: true, recursive: true })
    }
  }, 20_000)
})
