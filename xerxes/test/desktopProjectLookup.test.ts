// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { test, expect } from 'bun:test'
import { mkdtemp, chmod, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

/** The lookup's own limit (spawn.ts: execFileSync timeout). */
const GIT_LOOKUP_TIMEOUT_MS = 1_000

/**
 * Race-free by construction. The fake git used to prove it ran by writing
 * its pid, but the lookup SIGKILLs it after a fixed 1s — under full-suite
 * load the shell could be killed before reaching that line, and the test
 * failed reading a pid file that was never written. Each fact is now
 * checked without depending on how far the child got:
 * - hung and abandoned: the call took at least the lookup's timeout;
 * - killed: no process is still running this unique fake-git path;
 * - fallback: the selected directory comes back.
 */
test.skipIf(process.platform === 'win32')('a hung Git lookup is killed and falls back to the selected directory', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-git-lookup-')))
  const git = join(directory, 'git')
  const modulePath = join(import.meta.dir, '../src/desktop/main/spawn.ts')
  // Ignores SIGTERM, so only the lookup's SIGKILL can end it.
  await Bun.write(git, `#!/bin/sh\ntrap '' TERM\nwhile :; do :; done\n`)
  await chmod(git, 0o700)
  const probe = `import {canonicalProjectDir} from ${JSON.stringify(modulePath)};const started=performance.now();const dir=canonicalProjectDir(${JSON.stringify(directory)});console.log(JSON.stringify({dir,ms:performance.now()-started}))`
  const runner = Bun.spawn([process.execPath, '-e', probe], {
    env: { ...process.env, PATH: directory }, stdout: 'pipe', stderr: 'pipe',
  })
  // Only a guard against a wedged runner; generous so load cannot trip it.
  const guard = setTimeout(() => runner.kill(), 15_000)
  try {
    expect(await runner.exited).toBe(0)
    const { dir, ms } = JSON.parse((await new Response(runner.stdout).text()).trim()) as { dir: string; ms: number }
    expect(dir).toBe(directory)
    expect(ms).toBeGreaterThanOrEqual(GIT_LOOKUP_TIMEOUT_MS * 0.95)
    const survivors = Bun.spawnSync(['pgrep', '-f', git]).stdout.toString().trim()
    expect(survivors).toBe('')
  } finally {
    clearTimeout(guard)
    if (runner.exitCode === null) runner.kill()
    await runner.exited
    Bun.spawnSync(['pkill', '-9', '-f', git])
    await rm(directory, { recursive: true, force: true })
  }
}, 30_000)
