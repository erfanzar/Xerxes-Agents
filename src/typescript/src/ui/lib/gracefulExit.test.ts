// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { pathToFileURL } from 'node:url'

import { describe, expect, it } from 'vitest'

const moduleUrl = pathToFileURL(new URL('./gracefulExit.ts', import.meta.url).pathname).href

async function runFatalProcess(trigger: string) {
  const script = `
    import { setupGracefulExit } from ${JSON.stringify(moduleUrl)};
    setupGracefulExit({
      failsafeMs: 500,
      cleanups: [() => process.stderr.write('cleanup\\n')],
      onError: scope => process.stderr.write(scope + '\\n')
    });
    ${trigger}
    setTimeout(() => {}, 2_000);
  `
  const child = Bun.spawn([process.execPath, '--eval', script], {
    stderr: 'pipe',
    stdout: 'pipe'
  })
  const stderr = new Response(child.stderr).text()
  const stdout = new Response(child.stdout).text()
  const exitCode = await child.exited

  return { exitCode, stderr: await stderr, stdout: await stdout }
}

describe('setupGracefulExit fatal failures', () => {
  it('cleans up and exits after an uncaught exception', async () => {
    const result = await runFatalProcess(`queueMicrotask(() => { throw new Error('fatal exception') })`)

    expect(result.exitCode).toBe(1)
    expect(result.stderr).toContain('uncaughtException')
    expect(result.stderr).toContain('cleanup')
    expect(result.stdout).toBe('')
  })

  it('cleans up and exits after an unhandled rejection', async () => {
    const result = await runFatalProcess(`void Promise.reject(new Error('fatal rejection'))`)

    expect(result.exitCode).toBe(1)
    expect(result.stderr).toContain('unhandledRejection')
    expect(result.stderr).toContain('cleanup')
    expect(result.stdout).toBe('')
  })
})
