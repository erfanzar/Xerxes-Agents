// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { BoundedOutputBuffer } from '../src/tools/processOutput.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { executeCommand, registerProcessTools } from '../src/tools/processTools.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

async function inTemporaryWorkspace(body: (root: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bg-'))
  try {
    await body(root, new WorkspacePathResolver(root))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('a command that backgrounds a child still returns, instead of waiting for a pipe nobody will close', async () => {
  // The bug this pins: awaiting stdout to EOF requires *every* holder of the
  // write end to close it, and `sleep 30 &` hands a copy to a process that
  // outlives the shell. The timeout fired, killed the shell, and the call then
  // sat forever on a read that could not finish — a stray `&` stalled the whole
  // turn. Observed at 74 minutes on one report.
  await inTemporaryWorkspace(async (_root, paths) => {
    const started = Date.now()
    const result = await executeCommand(
      { cmd: '/bin/sh', args: ['-c', 'sleep 30 & echo started'], timeout_ms: 2_000 },
      paths,
    )
    const elapsed = Date.now() - started

    expect(elapsed).toBeLessThan(2_000)
    expect(result).toMatchObject({ exitCode: 0, timedOut: false })
    // The output still arrives: returning early must not mean returning blind.
    expect('stdout' in result ? result.stdout : '').toContain('started')
  })
})

test('a slow foreground command is still bounded by its own timeout', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const started = Date.now()
    const result = await executeCommand({ cmd: 'sleep', args: ['30'], timeout_ms: 1_000 }, paths)

    expect(Date.now() - started).toBeLessThan(5_000)
    expect(result).toMatchObject({ timedOut: true })
  })
})

test('run_in_background returns a handle at once rather than waiting out the command', async () => {
  await inTemporaryWorkspace(async (root, paths) => {
    const background = new BackgroundCommandManager()
    try {
      const started = Date.now()
      const handle = await executeCommand(
        { cmd: '/bin/sh', args: ['-c', 'echo first; sleep 5; echo never-waited-for'], run_in_background: true },
        paths,
        undefined,
        background,
      )
      // The whole point: a five-second command does not cost the turn five seconds.
      expect(Date.now() - started).toBeLessThan(2_000)
      expect(handle).toMatchObject({ running: true })
      const procId = 'procId' in handle ? handle.procId : ''
      expect(procId).not.toBe('')
      expect(background.list().some(record => record.procId === procId)).toBe(true)

      // Output is readable while it runs, without waiting for exit.
      const early = await background.check(procId, 1_000, 500)
      expect(early.stdout).toContain('first')
      expect(early.running).toBe(true)
      expect(early.exitCode).toBeNull()

      // A second poll shows only new output, not the same line again.
      const second = await background.check(procId, 1_000, 0)
      expect(second.stdout).not.toContain('first')

      const killed = await background.kill(procId, 'SIGKILL')
      expect(killed.signalled).toBe(true)
      expect(background.list().some(record => record.procId === procId)).toBe(false)
      void root
    } finally {
      await background.disposeAll()
    }
  })
})

test('a background command that finishes reports its exit code and final output', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const background = new BackgroundCommandManager()
    try {
      const handle = await executeCommand(
        { cmd: '/bin/sh', args: ['-c', 'echo done; exit 3'], run_in_background: true },
        paths,
        undefined,
        background,
      )
      const procId = 'procId' in handle ? handle.procId : ''
      // wait_ms lets a nearly-finished command settle rather than reporting
      // running:true and being asked again immediately.
      const checked = await background.check(procId, 1_000, 5_000)
      expect(checked.running).toBe(false)
      expect(checked.exitCode).toBe(3)
      expect(checked.stdout).toContain('done')
    } finally {
      await background.disposeAll()
    }
  })
})

test('checking or killing an unknown process is a clear validation error, not a crash', async () => {
  const background = new BackgroundCommandManager()
  await expect(background.check('nope', 100, 0)).rejects.toThrow(/proc_id/)
  await expect(background.kill('nope')).rejects.toThrow(/proc_id/)
})

test('killing reports honestly when the process had already exited', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const background = new BackgroundCommandManager()
    try {
      const handle = await executeCommand(
        { cmd: '/bin/sh', args: ['-c', 'exit 0'], run_in_background: true },
        paths,
        undefined,
        background,
      )
      const procId = 'procId' in handle ? handle.procId : ''
      await background.check(procId, 100, 5_000)
      // Claiming to have killed something already dead would misreport what happened.
      expect((await background.kill(procId)).signalled).toBe(false)
    } finally {
      await background.disposeAll()
    }
  })
})

test('run_in_background is refused when the host did not enable it', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    await expect(executeCommand({ cmd: 'echo', args: ['hi'], run_in_background: true }, paths))
      .rejects.toThrow(/not enabled by this host/)
  })
})

test('process tools isolate background commands by trusted session context', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const background = new BackgroundCommandManager()
    const registry = new ToolRegistry()
    registerProcessTools(registry, paths, background)
    const ownerA = { metadata: {}, sessionId: 'owner-a' }
    const ownerB = { metadata: {}, sessionId: 'owner-b' }

    try {
      const started = JSON.parse(await registry.execute(call('exec_command', {
        cmd: '/bin/sh',
        args: ['-c', 'echo OWNER_A_SECRET; sleep 30'],
        run_in_background: true,
      }), ownerA)) as { procId: string }

      const aList = JSON.parse(await registry.execute(call('list_commands', {}), ownerA)) as {
        processes: Array<{ procId: string }>
      }
      const bList = JSON.parse(await registry.execute(call('list_commands', {}), ownerB)) as {
        processes: Array<{ procId: string }>
      }
      expect(aList.processes.map(process => process.procId)).toContain(started.procId)
      expect(bList.processes).toEqual([])

      await expect(registry.execute(call('check_command', {
        proc_id: started.procId,
        wait_ms: 500,
      }), ownerB)).rejects.toThrow(/proc_id/)
      await expect(registry.execute(call('kill_command', { proc_id: started.procId }), ownerB))
        .rejects.toThrow(/proc_id/)

      const checked = JSON.parse(await registry.execute(call('check_command', {
        proc_id: started.procId,
        wait_ms: 500,
      }), ownerA)) as { running: boolean; stdout: string }
      expect(checked.stdout).toContain('OWNER_A_SECRET')
      expect(checked.running).toBeTrue()

      const killed = JSON.parse(await registry.execute(call('kill_command', {
        proc_id: started.procId,
        signal: 'SIGKILL',
      }), ownerA)) as { signalled: boolean }
      expect(killed.signalled).toBeTrue()
    } finally {
      await background.disposeAll()
    }
  })
})

test('background process tools fail closed without trusted session context', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const background = new BackgroundCommandManager()
    const registry = new ToolRegistry()
    registerProcessTools(registry, paths, background)
    const missingContext = { metadata: {} }

    try {
      await expect(registry.execute(call('exec_command', {
        cmd: 'sleep',
        args: ['30'],
        run_in_background: true,
      }), missingContext)).rejects.toThrow(/sessionId/)
      await expect(registry.execute(call('list_commands', {}), missingContext)).rejects.toThrow(/sessionId/)
      await expect(registry.execute(call('check_command', { proc_id: 'unknown' }), missingContext))
        .rejects.toThrow(/sessionId/)
      await expect(registry.execute(call('kill_command', { proc_id: 'unknown' }), missingContext))
        .rejects.toThrow(/sessionId/)
      expect(background.list()).toEqual([])
    } finally {
      await background.disposeAll()
    }
  })
})

test('owner disposal leaves other owners running', async () => {
  await inTemporaryWorkspace(async (root) => {
    const background = new BackgroundCommandManager()
    try {
      const a = background.startForOwner('owner-a', { command: 'sleep', args: ['30'], cwd: root })
      const b = background.startForOwner('owner-b', { command: 'sleep', args: ['30'], cwd: root })

      await background.disposeOwner('owner-b')
      expect(background.listForOwner('owner-b')).toEqual([])
      expect(background.listForOwner('owner-a').map(record => record.procId)).toEqual([a.procId])
      expect((await background.checkForOwner('owner-a', a.procId, 100)).running).toBeTrue()
      await expect(background.checkForOwner('owner-b', a.procId, 100)).rejects.toThrow(/proc_id/)
      expect(b.procId).not.toBe(a.procId)
    } finally {
      await background.disposeAll()
    }
  })
})

test('the output buffer keeps the recent tail and reports that it dropped the rest', () => {
  // Dropping the oldest is deliberate: refusing to read once full would block the
  // child on a full pipe, which is the failure this whole path exists to avoid.
  const buffer = new BoundedOutputBuffer(10)
  buffer.append('0123456789')
  expect(buffer.dropped).toBe(false)
  buffer.append('abcde')
  expect(buffer.dropped).toBe(true)
  expect(buffer.take(100).text).toBe('56789abcde')
  // Consumed, so a second read sees nothing new.
  expect(buffer.take(100).text).toBe('')
})

test('a capped read keeps the remainder for the next poll instead of discarding it', () => {
  const buffer = new BoundedOutputBuffer(100)
  buffer.append('abcdefghij')
  const first = buffer.take(4)
  expect(first).toEqual({ text: 'abcd', truncated: true })
  // Paging through a chatty process must not lose the pages not yet read.
  expect(buffer.take(100)).toEqual({ text: 'efghij', truncated: false })
})
