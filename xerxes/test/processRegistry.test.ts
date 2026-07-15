// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ProcessRegistry,
  getDefaultProcessRegistry,
  getDefaultRegistry,
  type BunSubprocessLike,
  type ProcessSignal,
} from '../src/runtime/processRegistry.js'

class FakeProcess implements BunSubprocessLike {
  readonly exited: Promise<number>
  readonly pid: number
  exitCode: number | null = null
  readonly signals: ProcessSignal[] = []
  throwOnKill = false
  private resolveExit: (code: number) => void = () => {}

  constructor(pid: number) {
    this.pid = pid
    this.exited = new Promise(resolve => { this.resolveExit = resolve })
  }

  exit(code: number): void {
    this.exitCode = code
    this.resolveExit(code)
  }

  kill(signal?: ProcessSignal): void {
    if (this.throwOnKill) {
      throw new Error('process is gone')
    }
    this.signals.push(signal ?? 0)
  }
}

test('process registry records Bun-like handles with deterministic process metadata', () => {
  const ids = ['proc-a', 'proc-b']
  const registry = new ProcessRegistry({
    idFactory: () => ids.shift() ?? 'overflow',
    now: () => 123.5,
  })
  const metadata = { owner: 'daemon' }
  const first = new FakeProcess(4001)
  const second = new FakeProcess(4002)

  expect(registry.register(first, {
    name: 'watcher',
    command: 'bun --watch src/cli.ts',
    cwd: '/workspace',
    metadata,
  })).toBe('proc-a')
  expect(registry.register(second)).toBe('proc-b')
  metadata.owner = 'mutated after registration'

  expect(registry.record('proc-a')).toEqual({
    procId: 'proc-a',
    pid: 4001,
    name: 'watcher',
    command: 'bun --watch src/cli.ts',
    cwd: '/workspace',
    metadata: { owner: 'daemon' },
    startedAt: 123.5,
  })
  expect(registry.list().map(record => record.procId)).toEqual(['proc-a', 'proc-b'])
  expect(registry.get('proc-a')).toBe(first)
  expect(Object.isFrozen(registry.record('proc-a'))).toBeTrue()
})

test('process registry polls and waits asynchronously without making unknown IDs look running', async () => {
  const registry = new ProcessRegistry({ idFactory: () => 'waiter' })
  const process = new FakeProcess(5001)
  const procId = registry.register(process)

  expect(registry.poll(procId)).toBeNull()
  expect(registry.poll('missing')).toBeUndefined()
  expect(await registry.wait('missing')).toBeUndefined()

  const waiting = registry.wait(procId)
  process.exit(17)
  expect(await waiting).toBe(17)
  expect(registry.poll(procId)).toBe(17)

  const timeoutRegistry = new ProcessRegistry({ idFactory: () => 'timeout' })
  const pending = new FakeProcess(5002)
  const timeoutId = timeoutRegistry.register(pending)
  expect(await timeoutRegistry.wait(timeoutId, 0)).toBeNull()
})

test('process registry signals, removes, clears, and safely owns its singleton', () => {
  const ids = ['signal', 'gone', 'other']
  const registry = new ProcessRegistry({ idFactory: () => ids.shift() ?? 'next' })
  const process = new FakeProcess(6001)
  const procId = registry.register(process)

  expect(registry.terminate(procId)).toBeTrue()
  expect(registry.kill(procId)).toBeTrue()
  expect(process.signals).toEqual(['SIGTERM', 'SIGKILL'])
  process.exit(0)
  expect(registry.terminate(procId)).toBeFalse()

  const gone = new FakeProcess(6002)
  gone.throwOnKill = true
  const goneId = registry.register(gone)
  expect(registry.kill(goneId)).toBeFalse()
  expect(registry.remove(procId)).toBeTrue()
  expect(registry.record(procId)).toBeUndefined()
  expect(registry.remove(procId)).toBeFalse()
  expect(registry.clear()).toBe(1)
  expect(registry.size).toBe(0)

  const first = getDefaultProcessRegistry()
  first.clear()
  expect(getDefaultRegistry()).toBe(first)
  first.clear()
})

test('a real Bun subprocess is accepted and yields its real exit code', async () => {
  const registry = new ProcessRegistry({ idFactory: () => 'bun-child' })
  const child = Bun.spawn([process.execPath, '-e', 'process.exit(7)'], {
    stdin: 'ignore',
    stdout: 'ignore',
    stderr: 'ignore',
  })
  const procId = registry.register(child, { command: 'bun -e process.exit(7)' })
  expect(await registry.wait(procId, 5)).toBe(7)
  expect(registry.poll(procId)).toBe(7)
})
