// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  PROCESS_TREE_GRACE_MS,
  ProcessRegistry,
  getDefaultProcessRegistry,
  getDefaultRegistry,
  terminateProcessSubtree,
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

test('group-leader registration still signals the direct child when no group exists', () => {
  // FakeProcess pids have no real process group, so the group delivery fails
  // with ESRCH and must fall back to the direct-child kill instead of lying
  // about a successful signal.
  const registry = new ProcessRegistry({ idFactory: () => 'leader' })
  const process = new FakeProcess(7001)
  const procId = registry.register(process, { processGroupLeader: true })

  expect(registry.signal(procId, 'SIGTERM')).toBeTrue()
  expect(process.signals).toEqual(['SIGTERM'])
})

test.skipIf(process.platform === 'win32')('terminateProcessSubtree escalates to SIGKILL past a trapped SIGTERM', async () => {
  const child = Bun.spawn(['/bin/sh', '-c', "trap '' TERM\nwhile :; do :; done"], {
    stdin: 'ignore',
    stdout: 'ignore',
    stderr: 'ignore',
    detached: true,
  })
  // Let the shell install its TERM trap and reach the loop, like a genuinely
  // wedged command would have.
  await Bun.sleep(300)
  const started = Date.now()
  await terminateProcessSubtree(child, { processGroupLeader: true })
  const exitCode = await child.exited

  // 137 = 128 + SIGKILL: only the escalation can have ended a TERM-trapped shell.
  expect(exitCode).toBe(137)
  const elapsed = Date.now() - started
  expect(elapsed).toBeGreaterThan(PROCESS_TREE_GRACE_MS - 500)
  expect(elapsed).toBeLessThan(15_000)
})

test.skipIf(process.platform === 'win32')('termination sweeps up a helper forked during the kill window', async () => {
  // The bug this pins: the leader trapped SIGTERM, forked a writer from its trap
  // handler, and exited within the grace window — so the escalation was skipped
  // (the direct child looked finished) and the newcomer was orphaned.
  const root = await mkdtemp(join(tmpdir(), 'xerxes-late-fork-'))
  const logPath = join(root, 'ticks.log')
  try {
    const scriptPath = join(root, 'late-fork.sh')
    await Bun.write(
      scriptPath,
      '#!/bin/sh\n'
        + `trap 'sh -c "(while :; do echo tick >> ${logPath}; sleep 0.05; done) &"; exit 0' TERM\n`
        + 'while :; do :; done\n',
    )
    await chmodSync(scriptPath, 0o755)
    const child = Bun.spawn(['/bin/sh', scriptPath], {
      stdin: 'ignore',
      stdout: 'ignore',
      stderr: 'ignore',
      detached: true,
    })
    await Bun.sleep(300)

    const started = Date.now()
    await terminateProcessSubtree(child, { processGroupLeader: true })
    // The leader exits on its trapped TERM well inside the grace; only the
    // post-exit group sweep can account for the helper it just forked.
    expect(Date.now() - started).toBeLessThan(PROCESS_TREE_GRACE_MS + 5_000)

    await Bun.sleep(400)
    const file = Bun.file(logPath)
    const before = (await file.exists()) ? (await file.text()).length : -1
    await Bun.sleep(800)
    const after = (await file.exists()) ? (await file.text()).length : -1
    expect(after).toBe(before)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test.skipIf(process.platform === 'win32')('a signal-killed child polls as terminated instead of running forever', async () => {
  // Bun leaves exitCode null when a child dies by signal and reports the death
  // via signalCode; polling that read "running" for the rest of eternity.
  const registry = new ProcessRegistry({ idFactory: () => 'signalled' })
  const child = Bun.spawn(['/bin/sh', '-c', 'sleep 30'], {
    stdin: 'ignore',
    stdout: 'ignore',
    stderr: 'ignore',
  })
  const procId = registry.register(child)
  child.kill('SIGKILL')
  const code = await child.exited

  expect(code).toBeGreaterThan(128)
  expect(registry.poll(procId)).toBe(code)
  expect(await registry.wait(procId)).toBe(code)
})
