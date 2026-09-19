// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { test, expect } from 'bun:test'
import { mkdtemp, chmod, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

test.skipIf(process.platform === 'win32')('a hung Git lookup is killed and falls back to the selected directory', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-git-lookup-')))
  const git = join(directory, 'git')
  const pidFile = join(directory, 'git.pid')
  const modulePath = join(import.meta.dir, '../src/desktop/main/spawn.ts')
  await Bun.write(git, `#!/bin/sh\necho $ > ${JSON.stringify(pidFile)}\ntrap '' TERM\nwhile :; do :; done\n`)
  await chmod(git, 0o700)
  const runner = Bun.spawn([process.execPath, '-e', `import {canonicalProjectDir} from ${JSON.stringify(modulePath)};console.log(canonicalProjectDir(${JSON.stringify(directory)}))`], {
    env: { ...process.env, PATH: directory }, stdout: 'pipe', stderr: 'pipe',
  })
  const timeout = setTimeout(() => runner.kill(), 5000)
  try {
    expect(await runner.exited).toBe(0)
    expect((await new Response(runner.stdout).text()).trim()).toBe(directory)
    const pid = Number(await Bun.file(pidFile).text())
    expect(() => process.kill(pid, 0)).toThrow()
  } finally {
    clearTimeout(timeout)
    if (runner.exitCode === null) runner.kill()
    await runner.exited
    await rm(directory, { recursive: true, force: true })
  }
}, 10000)
