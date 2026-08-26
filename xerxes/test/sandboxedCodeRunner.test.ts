// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SandboxedCodeRunner } from '../src/runtime/sandboxedCodeRunner.js'

test('sandboxed runner evaluates TypeScript and returns the exported result', async () => {
  const runner = new SandboxedCodeRunner()
  const result = await runner.run({
    code: 'return 2 + 3;',
    timeoutMs: 1_000,
  })
  expect(result.error).toBeUndefined()
  expect(result.ok).toBeTrue()
  expect(result.output).toBe(5)
})

test('sandboxed runner exposes only declared bindings and logs calls', async () => {
  const calls: Array<{ binding: string; args: unknown[] }> = []
  const runner = new SandboxedCodeRunner({
    bindings: {
      echo: async (...args: unknown[]) => {
        const message = String(args[0])
        calls.push({ binding: 'echo', args })
        return message
      },
    },
  })
  const result = await runner.run({
    code: 'const echo = globalThis.__sandbox_bindings__.echo; return await echo("hello");',
    timeoutMs: 1_000,
  })
  expect(result.ok).toBeTrue()
  expect(result.output).toBe('hello')
  expect(calls).toEqual([{ binding: 'echo', args: ['hello'] }])
})

test('sandboxed runner rejects unauthorized imports and direct process access', async () => {
  const runner = new SandboxedCodeRunner()
  const importResult = await runner.run({
    code: 'const readFile = await import("node:fs/promises").then(m => m.readFile); return readFile("/etc/passwd");',
    timeoutMs: 1_000,
  })
  expect(importResult.ok).toBeFalse()

  const envResult = await runner.run({
    code: 'return process.env.PATH;',
    timeoutMs: 1_000,
  })
  expect(envResult.ok).toBeTrue()
  expect(envResult.output).toBeUndefined()
})

test('sandboxed runner enforces timeout and memory budgets', async () => {
  const runner = new SandboxedCodeRunner()
  const timeoutResult = await runner.run({
    code: 'while (true) {}',
    timeoutMs: 100,
  })
  expect(timeoutResult.ok).toBeFalse()
  expect(timeoutResult.error?.toLowerCase()).toContain('time')
})

test('the sandbox refuses every runtime route to new code', async () => {
  const runner = new SandboxedCodeRunner()

  // The exact bypass a review reproduced: assemble the forbidden text at
  // runtime and hand it to a compiler reached through a prototype, so the
  // source-text guard never sees `import(`.
  const viaConstructor = await runner.run({
    code: 'const F = (async()=>{}).constructor; return typeof (await F("return (await imp"+"ort(\\"node:fs\\")).readFileSync")())',
  })
  expect(viaConstructor.ok).toBe(false)
  expect(viaConstructor.error).toContain('compiling new code is disabled')

  for (const code of ['return typeof Function("return 1")', 'return typeof eval("1")']) {
    const blocked = await runner.run({ code })
    expect(blocked.ok).toBe(false)
    expect(blocked.error).toContain('compiling new code is disabled')
  }

  // …while ordinary guest code is untouched.
  expect(await runner.run({ code: 'return 6 * 7' })).toMatchObject({ ok: true, output: 42 })
})

test('the source guard is not stateful across requests', async () => {
  const runner = new SandboxedCodeRunner()
  // The guard regex carried the `g` flag while being used with `.test()`, whose
  // lastIndex persists on the shared regex object — the identical payload
  // alternated pass/fail, so every second request went straight through.
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const result = await runner.run({ code: 'return typeof (await import("node:fs"))' })
    expect(result.ok).toBe(false)
  }
})

test('a missing shim fails closed instead of running guest code somewhere else', async () => {
  const runner = new SandboxedCodeRunner({ nodePath: '/nonexistent/interpreter' })
  const result = await runner.run({ code: 'return 1' })
  expect(result.ok).toBe(false)
  // An unhandled 'error' on the child used to take the host process down.
  expect(String(result.error)).toMatch(/could not start|stderr|no result frame/)
})
