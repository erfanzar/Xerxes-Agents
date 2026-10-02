// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { BoundedOutputBuffer } from '../src/tools/processOutput.js'
import { ProcessRegistry } from '../src/runtime/processRegistry.js'
import { TerminalRegistry, type TerminalHandle } from '../src/runtime/terminalRegistry.js'
import { RunHistory, type RunRecord } from '../src/runtime/runHistory.js'
import { ValidationError } from '../src/core/errors.js'
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

test('a foreground command that overflows its cap keeps the true start and end and counts the gap', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    // Far past the retained window (8x the cap), so the oldest output is dropped
    // while it streams: the result must still open on line 1 and close on the
    // last line, not on a slice from the middle.
    const overflowing = await executeCommand({ cmd: 'seq', args: ['1', '200000'], max_output_chars: 1_000 }, paths)
    const stdout = 'stdout' in overflowing ? overflowing.stdout : ''
    expect(stdout.startsWith('1\n2\n3\n')).toBe(true)
    expect(stdout.endsWith('199999\n200000\n')).toBe(true)
    const omitted = /…\[(\d+) chars omitted\]…/.exec(stdout)
    expect(omitted).not.toBeNull()
    // Every printed character is either shown or counted as omitted.
    const total = Array.from({ length: 200_000 }, (_value, index) => String(index + 1).length + 1).reduce((a, b) => a + b, 0)
    expect(stdout.length - (omitted?.[0].length ?? 0) - 2 + Number(omitted?.[1])).toBe(total)
    expect(overflowing).toMatchObject({ truncated: true })

    // Inside the retained window the summary and final lines are kept too.
    const moderate = await executeCommand({ cmd: 'seq', args: ['1', '1000'], max_output_chars: 1_000 }, paths)
    const moderateOut = 'stdout' in moderate ? moderate.stdout : ''
    expect(moderateOut.startsWith('1\n2\n')).toBe(true)
    expect(moderateOut.endsWith('999\n1000\n')).toBe(true)

    const fits = await executeCommand({ cmd: 'seq', args: ['1', '3'], max_output_chars: 1_000 }, paths)
    expect(fits).toMatchObject({ stdout: '1\n2\n3\n', truncated: false })
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

test('a foreground command that hits its timeout is adopted into the background instead of killed', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const background = new BackgroundCommandManager()
    try {
      const started = Date.now()
      const result = await executeCommand(
        { cmd: '/bin/sh', args: ['-c', 'echo before-ceiling; sleep 30'], timeout_ms: 500 },
        paths,
        undefined,
        background,
        undefined,
        'owner-session',
      )
      // The call answered at the ceiling, not after 30s.
      expect(Date.now() - started).toBeLessThan(5_000)
      expect(result).toMatchObject({ backgrounded: true, running: true, timedOut: true })
      const procId = 'procId' in result ? result.procId : ''
      expect(procId).not.toBe('')

      // The process survived the ceiling: output from before the handoff is
      // still readable and the process is genuinely still running.
      const check = await background.checkForOwner('owner-session', procId, 1_000, 0)
      expect(check.stdout).toContain('before-ceiling')
      expect(check.running).toBe(true)

      const killed = await background.killForOwner('owner-session', procId, 'SIGKILL')
      expect(killed.signalled).toBe(true)
    } finally {
      await background.disposeAll()
    }
  })
}, 15_000)

test('timeout adoption hands the foreground terminal over instead of reporting it interrupted', async () => {
  await inTemporaryWorkspace(async (root, paths) => {
    const history = new RunHistory(join(root, 'runs.sqlite'))
    const notified: RunRecord[] = []
    history.subscribe(run => notified.push(run))
    const terminals = new TerminalRegistry({ runHistory: history })
    const background = new BackgroundCommandManager(undefined, terminals)
    try {
      const result = await executeCommand(
        // max_output_chars:1 gave the adopted job an 8-character buffer, so
        // everything it printed after the handoff was reduced to its last 8.
        { cmd: '/bin/sh', args: ['-c', 'echo before-ceiling; sleep 1; echo after-the-handoff-output; sleep 30'], timeout_ms: 300, max_output_chars: 1 },
        paths,
        undefined,
        background,
        terminals,
        'owner-session',
      )
      expect(result).toMatchObject({ backgrounded: true, running: true })
      const procId = 'procId' in result ? result.procId : ''
      await Bun.sleep(100)

      // The command is alive: no "Run finished — terminal interrupted" notice,
      // and no interrupted foreground run left behind in history.
      expect(notified).toEqual([])
      expect(history.list('owner-session').filter(run => run.state !== 'running')).toEqual([])

      // One terminal, the background job, carrying what printed before the handoff.
      const listed = terminals.list('owner-session')
      expect(listed.map(entry => entry.id)).toEqual([procId])
      expect(terminals.inspect('owner-session', procId)?.output).toContain('before-ceiling')

      const deadline = Date.now() + 5_000
      let stdout = ''
      while (!stdout.includes('after-the-handoff-output') && Date.now() < deadline) {
        stdout += (await background.checkForOwner('owner-session', procId, 1_000, 0)).stdout
        await Bun.sleep(50)
      }
      expect(stdout).toContain('after-the-handoff-output')
      expect(terminals.inspect('owner-session', procId)?.output).toContain('before-ceiling\nafter-the-handoff-output')

      await background.killForOwner('owner-session', procId, 'SIGKILL')
    } finally {
      await background.disposeAll()
      history.close()
    }
  })
}, 15_000)

test('without a background host the timeout still kills, exactly as before', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const result = await executeCommand(
      { cmd: 'sleep', args: ['30'], timeout_ms: 500 },
      paths,
    )
    expect(result).toMatchObject({ timedOut: true })
    expect('backgrounded' in result).toBe(false)
  })
}, 15_000)

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

/**
 * A shell whose backgrounded grandchild appends to `logPath` forever while the
 * shell itself sleeps: the exact shape that used to survive a timeout or kill,
 * because only the direct child was ever signalled.
 */
async function writeTreeScript(root: string, logPath: string): Promise<string> {
  const scriptPath = join(root, 'tree.sh')
  await Bun.write(
    scriptPath,
    '#!/bin/sh\n'
      + `(while :; do echo tick >> "${logPath}"; sleep 0.05; done) &\n`
      + 'exec sleep 30\n',
  )
  await chmod(scriptPath, 0o755)
  return scriptPath
}

/** Whether the log stopped growing across `windowMs` — i.e. the writer is dead. */
async function logIsStill(logPath: string, windowMs: number): Promise<boolean> {
  const file = Bun.file(logPath)
  const before = (await file.exists()) ? (await file.text()).length : -1
  await Bun.sleep(windowMs)
  const after = (await file.exists()) ? (await file.text()).length : -1
  return after === before
}

// Windows has neither /bin/sh nor POSIX process groups; these kills are
// best-effort child.kill() calls there.
test.skipIf(process.platform === 'win32')('a timed out command takes its whole process group down', async () => {
  await inTemporaryWorkspace(async (root, paths) => {
    const logPath = join(root, 'ticks.log')
    const script = await writeTreeScript(root, logPath)

    const started = Date.now()
    const result = await executeCommand(
      { cmd: '/bin/sh', args: [script], timeout_ms: 1_000 },
      paths,
    )
    expect(result).toMatchObject({ timedOut: true })
    expect(Date.now() - started).toBeLessThan(15_000)

    // The tool returned; the grandchild must not still be appending.
    await Bun.sleep(300)
    expect(await logIsStill(logPath, 800)).toBe(true)
  })
})

test.skipIf(process.platform === 'win32')('a caller cancel takes the whole process group down', async () => {
  await inTemporaryWorkspace(async (root, paths) => {
    const logPath = join(root, 'ticks.log')
    const script = await writeTreeScript(root, logPath)
    const controller = new AbortController()
    setTimeout(() => controller.abort(new Error('cancelled by test')), 500)

    let observedError: unknown
    try {
      await executeCommand({ cmd: '/bin/sh', args: [script], timeout_ms: 30_000 }, paths, controller.signal)
    } catch (error) {
      observedError = error
    }
    expect(observedError).toBeInstanceOf(DOMException)
    expect((observedError as Error).name).toBe('AbortError')
    expect((observedError as Error).message).toContain('Command interrupted before completion')

    await Bun.sleep(300)
    expect(await logIsStill(logPath, 800)).toBe(true)
  })
})

test('interrupted exec retains partial output through the tool registry without claiming invalid arguments', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const controller = new AbortController()
    const terminals = new TerminalRegistry()
    const registry = new ToolRegistry()
    registerProcessTools(registry, paths, undefined, terminals)
    let unobserve = () => {}
    const unsubscribe = terminals.activityChanges.subscribe(() => {
      const terminal = terminals.list('cancel-owner').find(entry => entry.running)
      if (!terminal) return
      unobserve = terminals.subscribe('cancel-owner', terminal.id, event => {
        if (event.text.includes('partial-test-output')) controller.abort(new Error('Test interrupt'))
      })
    })
    try {
      const result = registry.execute(call('exec_command', {
        cmd: process.execPath,
        args: ['-e', 'console.log("partial-test-output"); await Bun.sleep(30000)'],
        timeout_ms: 60000,
      }), { sessionId: 'cancel-owner', metadata: {} }, controller.signal)
      await expect(result).rejects.toThrow('Command interrupted before completion: Test interrupt')
      await expect(result).rejects.toThrow('partial-test-output')
      await expect(result).rejects.not.toThrow('Validation error')
      expect(terminals.list('cancel-owner')[0]?.running).toBe(false)
    } finally { unsubscribe(); unobserve() }
  })
})

test.skipIf(process.platform === 'win32')('kill_command stops background commands including their grandchildren', async () => {
  await inTemporaryWorkspace(async (root) => {
    const logPath = join(root, 'ticks.log')
    const script = await writeTreeScript(root, logPath)
    const background = new BackgroundCommandManager()
    try {
      const handle = background.startForOwner('owner-kill', { command: '/bin/sh', args: [script], cwd: root })
      // Let the tree spin up and produce output first.
      await Bun.sleep(400)
      const killed = await background.killForOwner('owner-kill', handle.procId, 'SIGTERM')
      expect(killed.signalled).toBeTrue()

      await Bun.sleep(300)
      expect(await logIsStill(logPath, 800)).toBe(true)
    } finally {
      await background.disposeAll()
    }
  })
})

test.skipIf(process.platform === 'win32')('a command that forks a helper while dying leaves no survivors', async () => {
  // Auditor repro at tool level: the shell traps TERM and forks its helper from
  // inside the trap handler — the fork happens DURING the kill window, after
  // every signal the timeout path sends has already been aimed.
  await inTemporaryWorkspace(async (root, paths) => {
    const logPath = join(root, 'ticks.log')
    const scriptPath = join(root, 'late-fork.sh')
    await Bun.write(
      scriptPath,
      '#!/bin/sh\n'
        + `trap '(while :; do echo tick >> "${logPath}"; sleep 0.05; done) &' TERM\n`
        + 'echo ready\n'
        + 'while :; do :; done\n',
    )
    await chmod(scriptPath, 0o755)

    const result = await executeCommand(
      { cmd: '/bin/sh', args: [scriptPath], timeout_ms: 1_000 },
      paths,
    )
    expect(result).toMatchObject({ timedOut: true })

    await Bun.sleep(400)
    expect(await logIsStill(logPath, 800)).toBe(true)
  })
})

test('startForOwner kills the child when the terminal mirror cannot be opened', async () => {
  // A detached child outlives this process; if mirror registration throws, the
  // child must be killed and unregistered rather than leaked ownerless.
  const failingTerminals = {
    open: () => {
      throw new Error('mirror boom')
    },
  } as unknown as TerminalRegistry
  await inTemporaryWorkspace(async (root) => {
    const logPath = join(root, 'ticks.log')
    const script = await writeTreeScript(root, logPath)
    const background = new BackgroundCommandManager(new ProcessRegistry(), failingTerminals)

    await expect(() =>
      background.startForOwner('owner-leak', { command: '/bin/sh', args: [script], cwd: root }),
    ).toThrow('mirror boom')
    // Nothing is left registered under the failed start.
    expect(background.list()).toEqual([])

    // And the spawned tree was killed, not left running detached forever.
    await Bun.sleep(300)
    expect(await logIsStill(logPath, 800)).toBeTrue()
  })
})

test.each([false, true])('completion waits for adopted output drains (pipe remains open: %s)', async heldOpen => {
  await inTemporaryWorkspace(async root => {
    const history = new RunHistory(join(root, 'runs.sqlite'))
    const completed: RunRecord[] = []
    const terminals = new TerminalRegistry({ runHistory: history, onRunComplete: run => completed.push(run) })
    const manager = new BackgroundCommandManager(undefined, terminals)
    const child = Bun.spawn([process.execPath, '-e', ''], { cwd: root, stdout: 'ignore', stderr: 'ignore' })
    let finishDrain!: () => void
    let cancelled = false
    let mirror: TerminalHandle | undefined
    const done = new Promise<void>(resolve => { finishDrain = resolve })
    try {
      manager.adoptForOwner('owner', {
        child, command: ['fixture'], cwd: root,
        stdout: new BoundedOutputBuffer(), stderr: new BoundedOutputBuffer(),
        drains: [{ done, cancel: () => { cancelled = true; finishDrain() } }],
      }, terminal => { mirror = terminal })
      await child.exited
      await Bun.sleep(20)
      expect(completed).toHaveLength(0)
      mirror?.append('final stdout\nfinal stderr\n')
      if (!heldOpen) finishDrain()
      const deadline = Date.now() + 2_000
      while (completed.length === 0 && Date.now() < deadline) await Bun.sleep(10)
      expect(completed).toHaveLength(1)
      expect(completed[0]?.output).toContain('final stdout\nfinal stderr\n')
      expect(completed[0]?.state).toBe('succeeded')
      expect(cancelled).toBe(heldOpen)
      expect(completed[0]?.output.includes('output pipes remained open')).toBe(heldOpen)
      const restored = new RunHistory(join(root, 'runs.sqlite'))
      try { expect(restored.inspect('owner', completed[0]!.id)?.output).toBe(completed[0]!.output) }
      finally { restored.close() }
      await manager.disposeAll()
      expect(completed).toHaveLength(1)
    } finally {
      finishDrain()
      await manager.disposeAll()
      history.close()
    }
  })
})

test('incremental output tool preserves independent readers and requires the owning session', async () => {
  await inTemporaryWorkspace(async (root, paths) => {
    const terminals = new TerminalRegistry()
    const handle = terminals.open({ ownerSessionId: 'owner', id: 'shell', cwd: root, kind: 'background', command: 'test' })
    handle.append('one two')
    handle.close(0)
    const tools = new ToolRegistry()
    registerProcessTools(tools, paths, undefined, terminals)
    const request = call('read_terminal_output', { terminal_id: 'shell', max_output_chars: 3 })
    const first = JSON.parse(await tools.execute(request, { sessionId: 'owner', metadata: {} }))
    expect(first.text).toBe('one')
    const next = JSON.parse(await tools.execute(call('read_terminal_output', { terminal_id: 'shell', cursor: first.cursor }), { sessionId: 'owner', metadata: {} }))
    expect(next.text).toBe(' two')
    expect(JSON.parse(await tools.execute(request, { sessionId: 'owner', metadata: {} })).text).toBe('one')
    await expect(tools.execute(request, { sessionId: 'other', metadata: {} })).rejects.toThrow('Unknown terminal')
  })
})

test.skipIf(process.platform === 'win32')('background stop reaps helpers forked by a TERM handler before the leader exits', async () => {
  await inTemporaryWorkspace(async root => {
    const logPath = join(root, 'late-ticks.log')
    const scriptPath = join(root, 'late-stop.sh')
    await Bun.write(scriptPath,
      `trap '(trap "" TERM; while :; do echo tick >> "${logPath}"; sleep 0.05; done) & exit 0' TERM\n` +
      'echo ready\nwhile :; do :; done\n')
    const background = new BackgroundCommandManager()
    const started = background.startForOwner('owner', { command: '/bin/sh', args: [scriptPath], cwd: root })
    try {
      let ready = false
      for (let attempt = 0; attempt < 50 && !ready; attempt++) {
        ready = (await background.checkForOwner('owner', started.procId, 1000, 10)).stdout.includes('ready')
      }
      expect(ready).toBe(true)
      expect((await background.killForOwner('owner', started.procId)).signalled).toBe(true)
      await Bun.sleep(100)
      expect(await logIsStill(logPath, 300)).toBe(true)
    } finally { await background.disposeAll() }
  })
})

test('an unconfirmed stop retains the process and terminal for a later retry', async () => {
  class DeniedSignalRegistry extends ProcessRegistry {
    deny = true
    override signal(id: string, signal: Parameters<ProcessRegistry['signal']>[1]): boolean {
      return this.deny ? false : super.signal(id, signal)
    }
  }
  await inTemporaryWorkspace(async root => {
    const registry = new DeniedSignalRegistry()
    const terminals = new TerminalRegistry()
    const manager = new BackgroundCommandManager(registry, terminals)
    const started = manager.startForOwner('owner', { command: process.execPath, args: ['-e', 'setInterval(() => {}, 1000)'], cwd: root })
    try {
      await expect(manager.killForOwner('owner', started.procId)).rejects.toThrow('handle is retained')
      expect(manager.listForOwner('owner')).toHaveLength(1)
      expect(terminals.inspect('owner', started.procId)).toMatchObject({ running: true, canKill: true })
      registry.deny = false
      await manager.killForOwner('owner', started.procId, 'SIGKILL')
      expect(manager.listForOwner('owner')).toHaveLength(0)
      expect(terminals.inspect('owner', started.procId)?.running).toBe(false)
    } finally { registry.deny = false; await manager.disposeAll() }
  })
})

test('a stopped turn ends a waiting check instead of sitting out the wait', async () => {
  await inTemporaryWorkspace(async root => {
    const background = new BackgroundCommandManager()
    try {
      const { procId } = background.startForOwner('owner-session', { command: 'sleep', args: ['30'], cwd: root, name: 'sleep 30' })
      const controller = new AbortController()
      setTimeout(() => controller.abort(), 100)
      const started = Date.now()
      const check = background.checkForOwner('owner-session', procId, 1_000, 60_000, controller.signal)
      await expect(check).rejects.toThrow()
      expect(Date.now() - started).toBeLessThan(5_000)
      // The stop ends the wait, not the command: it is still the model's to inspect or kill.
      const after = await background.checkForOwner('owner-session', procId, 1_000, 0)
      expect(after.running).toBe(true)
    } finally {
      await background.disposeAll()
    }
  })
}, 15_000)
