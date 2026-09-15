// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { executeCommand } from '../src/tools/processTools.js'

const OWNER = 'session-owner'

async function inTemporaryWorkspace(body: (root: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-term-'))
  try {
    await body(root, new WorkspacePathResolver(root))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

async function eventually(predicate: () => boolean, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (predicate()) return
    await Bun.sleep(20)
  }
  throw new Error('condition never became true')
}

// Spawning `/bin/sh` needs a POSIX host; Windows has no /bin/sh to spawn.
test.skipIf(process.platform === 'win32')('inspecting a background command does not consume the output the model has not read', async () => {
  // The whole reason the mirror exists. `check_command` drains its buffer so
  // successive polls show progress; a viewer reading from that same buffer
  // would silently eat the lines the model was about to receive.
  await inTemporaryWorkspace(async root => {
    const terminals = new TerminalRegistry()
    const background = new BackgroundCommandManager(undefined, terminals)
    const started = background.start({ command: '/bin/sh', args: ['-c', 'echo hello-from-the-shell'], cwd: root })

    await eventually(() => (terminals.inspect('legacy-private-background-commands', started.procId)?.output ?? '').includes('hello-from-the-shell'))
    // Inspected twice: a peek that consumed would come back empty the second time.
    expect(terminals.inspect('legacy-private-background-commands', started.procId)?.output).toContain('hello-from-the-shell')
    expect(terminals.inspect('legacy-private-background-commands', started.procId)?.output).toContain('hello-from-the-shell')

    const checked = await background.check(started.procId, 10_000, 1_000)
    expect(checked.stdout).toContain('hello-from-the-shell')
    await background.kill(started.procId, 'SIGKILL')
  })
})

test.skipIf(process.platform === 'win32')('a background command is listed while it runs and keeps its exit code afterwards', async () => {
  await inTemporaryWorkspace(async root => {
    const terminals = new TerminalRegistry()
    const background = new BackgroundCommandManager(undefined, terminals)
    const started = background.start({ command: '/bin/sh', args: ['-c', 'exit 3'], cwd: root })

    const live = terminals.list('legacy-private-background-commands').find(entry => entry.id === started.procId)
    expect(live).toMatchObject({ kind: 'background', canKill: true })
    expect(live?.pid).toBeGreaterThan(0)

    await eventually(() => terminals.list('legacy-private-background-commands').find(entry => entry.id === started.procId)?.running === false)
    const finished = terminals.list('legacy-private-background-commands').find(entry => entry.id === started.procId)
    expect(finished).toMatchObject({ running: false, exitCode: 3, canKill: false })
    expect(finished?.endedAt).toBeGreaterThanOrEqual(finished!.startedAt)
  })
})

test.skipIf(process.platform === 'win32')('killing through the registry stops a live background process', async () => {
  await inTemporaryWorkspace(async root => {
    const terminals = new TerminalRegistry()
    const background = new BackgroundCommandManager(undefined, terminals)
    const started = background.start({ command: '/bin/sh', args: ['-c', 'sleep 30'], cwd: root })

    await terminals.kill('legacy-private-background-commands', started.procId, 'SIGKILL')
    await eventually(() => terminals.list('legacy-private-background-commands').find(entry => entry.id === started.procId)?.running === false)
    // Already dead: the second attempt reports why rather than pretending.
    await expect(terminals.kill('legacy-private-background-commands', started.procId)).rejects.toThrow(/already exited/)
  })
})

test.skipIf(process.platform === 'win32')('a foreground command is recorded with its output once it finishes', async () => {
  await inTemporaryWorkspace(async (_root, paths) => {
    const terminals = new TerminalRegistry()
    await executeCommand({ cmd: '/bin/sh', args: ['-c', 'echo one; echo two'] }, paths, undefined, undefined, terminals, OWNER)

    const entry = terminals.list(OWNER).at(-1)
    expect(entry).toMatchObject({ kind: 'foreground', running: false, exitCode: 0 })
    expect(terminals.inspect(OWNER, entry!.id)?.output).toContain('two')
  })
})

test('finished terminals age out of the history but running ones never do', async () => {
  const terminals = new TerminalRegistry({ historyLimit: 2 })
  const live = terminals.open({ id: 'live', kind: 'pty', command: 'bash', cwd: '/tmp', ownerSessionId: OWNER })

  for (let index = 0; index < 5; index += 1) {
    terminals.record({ id: `done-${index}`, kind: 'foreground', command: `echo ${index}`, cwd: '/tmp', ownerSessionId: OWNER, exitCode: 0, output: '' })
  }

  const ids = terminals.list(OWNER).map(entry => entry.id)
  expect(ids).toEqual(['live', 'done-3', 'done-4'])
  live.close(0)
})

test('the mirror keeps the tail of a chatty process, not its opening lines', async () => {
  const terminals = new TerminalRegistry({ mirrorCapacity: 64 })
  const handle = terminals.open({ id: 'noisy', kind: 'background', command: 'yes', cwd: '/tmp', ownerSessionId: OWNER })
  for (let index = 0; index < 100; index += 1) handle.append(`line-${index}\n`)

  const inspected = terminals.inspect(OWNER, 'noisy')
  expect(inspected?.output).toContain('line-99')
  expect(inspected?.output).not.toContain('line-0\n')
  expect(inspected?.outputTruncated).toBe(true)
  // Total observed survives the dropping, so the UI can say how much was lost.
  expect(inspected?.outputChars).toBeGreaterThan(64)
})

test('terminal entries are isolated by owner without exposing owner ids in response rows', async () => {
  const terminals = new TerminalRegistry()
  terminals.record({ id: 'owned', kind: 'foreground', command: 'echo secret', cwd: '/tmp', ownerSessionId: OWNER, exitCode: 0, output: 'secret' })

  expect(terminals.list('other-session')).toEqual([])
  expect(terminals.inspect('other-session', 'owned')).toBeUndefined()
  expect(terminals.list(OWNER)[0]).not.toHaveProperty('ownerSessionId')
  expect(terminals.inspect(OWNER, 'owned')).not.toHaveProperty('ownerSessionId')
})

test('writing to a terminal that has no input channel fails with the reason', async () => {
  const terminals = new TerminalRegistry()
  terminals.open({ id: 'no-stdin', kind: 'background', command: 'sleep 1', cwd: '/tmp', ownerSessionId: OWNER })

  await expect(terminals.write(OWNER, 'no-stdin', 'hi')).rejects.toThrow(/does not accept input/)
  await expect(terminals.write(OWNER, 'nope', 'hi')).rejects.toThrow(/unknown terminal/)
})

test('terminal run history preserves output without consuming the live mirror and restores F8 inspection', async () => {
  const { RunHistory } = await import('../src/runtime/runHistory.js')
  await inTemporaryWorkspace(async root => {
    const path = join(root, 'runs.sqlite')
    const history = new RunHistory(path)
    try {
      const terminals = new TerminalRegistry({ runHistory: history, mirrorCapacity: 64 })
      const handle = terminals.open({ id: 'build', kind: 'background', command: 'bun test', cwd: root, ownerSessionId: OWNER })
      handle.append('old'.repeat(30) + 'final output')
      await Bun.sleep(300)
      const active = history.list(OWNER)[0]!
      expect(active.state).toBe('running')
      expect(active.output).toContain('final output')
      expect(terminals.inspect(OWNER, 'build')?.output).toContain('final output')
      handle.close(1)
      expect(history.inspect(OWNER, active.id)).toMatchObject({ state: 'failed', unread: true, outputTruncated: true })
      handle.close(0)
      expect(history.inspect(OWNER, active.id)?.state).toBe('failed')
      const reopened = new TerminalRegistry({ runHistory: history })
      const archived = reopened.list(OWNER)[0]!
      expect(archived.canKill).toBe(false)
      expect(reopened.inspect(OWNER, archived.id)?.output).toContain('final output')
      expect(reopened.inspect('another-owner', archived.id)).toBeUndefined()
      await expect(reopened.kill(OWNER, archived.id)).rejects.toThrow('unknown terminal')
      const foreground = terminals.open({ id: 'quick', kind: 'foreground', command: 'pwd', cwd: root, ownerSessionId: OWNER })
      foreground.close(0)
      expect(history.list(OWNER, { unreadOnly: true })).toHaveLength(1)
    } finally { history.close() }
  })
})

test('independent output cursors survive archive/reopen and report durable retention gaps', async () => {
  await inTemporaryWorkspace(async root => {
    const path = join(root, 'runs.sqlite')
    const history = new RunHistory(path)
    const terminals = new TerminalRegistry({ runHistory: history })
    const terminal = terminals.open({ ownerSessionId: OWNER, id: 'process', cwd: root, kind: 'background', command: 'test' })
    terminal.append('a'.repeat(70_000))
    const first = terminals.readOutput(OWNER, 'process', undefined, 10)
    expect(first.text).toBe('a'.repeat(10))
    expect(terminals.readOutput(OWNER, 'process', undefined, 10)).toEqual(first)
    expect(terminals.readOutput(OWNER, 'process', first.cursor, 10).cursor.offset).toBe(20)
    expect(() => terminals.readOutput('another', 'process', first.cursor)).toThrow('Unknown terminal')
    terminal.close(0)
    const runId = history.list(OWNER)[0]!.id
    expect(first.cursor.streamId).toBe(runId)
    history.close()
    const reopened = new RunHistory(path)
    try {
      const restored = new TerminalRegistry({ runHistory: reopened })
      const page = restored.readOutput(OWNER, `run:${runId}`, first.cursor, 10)
      expect(page).toMatchObject({ text: 'a'.repeat(10), droppedChars: 5990, hasMore: true, running: false })
      expect(page.cursor.offset).toBe(6010)
      expect(() => restored.readOutput('another', `run:${runId}`, first.cursor)).toThrow('Unknown terminal')
      expect(() => restored.readOutput(OWNER, `run:${runId}`, { streamId: runId, offset: 70001 })).toThrow('Invalid output cursor')
      const end = restored.readOutput(OWNER, `run:${runId}`, { streamId: runId, offset: 70000 })
      expect(end).toMatchObject({ text: '', droppedChars: 0, hasMore: false })
    } finally { reopened.close() }
  })
})

test('terminal reuse rejects prior cursors and live overflow exposes the lost character count', () => {
  const terminals = new TerminalRegistry({ mirrorCapacity: 4 })
  const options = { ownerSessionId: OWNER, id: 'reused', cwd: '/repo', kind: 'background' as const, command: 'test' }
  const handle = terminals.open(options)
  handle.append('abcdef')
  const page = terminals.readOutput(OWNER, 'reused', undefined, 2)
  expect(page).toMatchObject({ text: 'cd', droppedChars: 2, hasMore: true })
  handle.close(0)
  terminals.open(options)
  expect(() => terminals.readOutput(OWNER, 'reused', page.cursor)).toThrow('Invalid output cursor')
  expect(() => terminals.readOutput(OWNER, 'reused', undefined, -1)).toThrow('Invalid output page limit')
})

test('an acknowledged live cursor survives abrupt owner death before the periodic checkpoint', async () => {
  await inTemporaryWorkspace(async root => {
    const path = join(root, 'runs.sqlite')
    const script = join(root, 'cursor-owner.ts')
    await Bun.write(script, `
      import { RunHistory } from ${JSON.stringify(join(import.meta.dir, '../src/runtime/runHistory.ts'))};
      import { TerminalRegistry } from ${JSON.stringify(join(import.meta.dir, '../src/runtime/terminalRegistry.ts'))};
      const history = new RunHistory(${JSON.stringify(path)});
      const terminals = new TerminalRegistry({ runHistory: history });
      const terminal = terminals.open({ ownerSessionId: 'owner', id: 'process', cwd: ${JSON.stringify(root)}, kind: 'background', command: 'test' });
      terminal.append('observed output');
      console.log(JSON.stringify(terminals.readOutput('owner', 'process')));
      // Block only this disposable fixture to prevent its periodic checkpoint.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5000);
    `)
    const child = Bun.spawn([process.execPath, script], { stdout: 'pipe', stderr: 'pipe', timeout: 6000 })
    try {
      const reader = child.stdout.getReader()
      const chunk = await reader.read()
      reader.releaseLock()
      const page = JSON.parse(new TextDecoder().decode(chunk.value))
      expect(page.text).toBe('observed output')
      child.kill('SIGKILL')
      await child.exited
      const recovered = new RunHistory(path)
      try {
        const after = recovered.terminalOutput('owner', page.cursor.streamId, page.cursor)
        expect(after).toMatchObject({ text: '', running: false, droppedChars: 0 })
        expect(recovered.inspect('owner', page.cursor.streamId)).toMatchObject({ state: 'interrupted', output: 'observed output' })
      } finally { recovered.close() }
    } finally { child.kill('SIGKILL'); await child.exited }
  })
})

test('incremental reads fail rather than acknowledge an unpersisted cursor when storage fails', () => {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry({ runHistory: history, onPersistenceError: () => {} })
  const handle = terminals.open({ ownerSessionId: OWNER, id: 'failed-store', kind: 'background', command: 'test', cwd: '/repo' })
  handle.append('pending')
  history.close()
  expect(() => terminals.readOutput(OWNER, 'failed-store')).toThrow()
  handle.close(null)
})

test('confirmed terminal cancellation is persisted only after exit, including exit during signalling', async () => {
  await inTemporaryWorkspace(async root => {
    const history = new RunHistory(join(root, 'cancel.sqlite'))
    try {
      const terminals = new TerminalRegistry({ runHistory: history })
      const handle = terminals.open({ id: 'cancel', kind: 'background', command: 'worker', cwd: root, ownerSessionId: OWNER, control: { kill: async () => {} } })
      await terminals.kill(OWNER, handle.id)
      expect(terminals.inspect(OWNER, handle.id)?.running).toBe(true)
      expect(history.list(OWNER)[0]?.state).toBe('running')
      handle.append('last output')
      handle.close(143)
      expect(history.list(OWNER)[0]).toMatchObject({ state: 'cancelled', exitCode: 143, output: 'last output', error: null })
      expect(terminals.wasCancelled(OWNER, handle.id)).toBe(true)
      expect(terminals.wasCancelled('another-session', handle.id)).toBe(false)
      const restored = new TerminalRegistry({ runHistory: history })
      expect(restored.wasCancelled(OWNER, restored.list(OWNER)[0]!.id)).toBe(true)
      const fast = terminals.open({ id: 'fast', kind: 'background', command: 'fast worker', cwd: root, ownerSessionId: OWNER, control: { kill: async () => { fast.close(143) } } })
      await terminals.kill(OWNER, fast.id)
      expect(history.list(OWNER).find(run => run.sourceId === 'fast')?.state).toBe('cancelled')
    } finally { history.close() }
  })
})

test('refused terminal cancellation propagates and does not disguise a process failure', async () => {
  await inTemporaryWorkspace(async root => {
    const history = new RunHistory(join(root, 'refused.sqlite'))
    try {
      const terminals = new TerminalRegistry({ runHistory: history })
      const handle = terminals.open({ id: 'refused', kind: 'background', command: 'worker', cwd: root, ownerSessionId: OWNER, control: { kill: async () => { handle.close(1); throw new Error('Signal denied') } } })
      await expect(terminals.kill(OWNER, handle.id)).rejects.toThrow('Signal denied')
      expect(history.list(OWNER)[0]).toMatchObject({ state: 'failed', exitCode: 1, error: 'Process exited with code 1' })
    } finally { history.close() }
  })
})

test('confirmed Ctrl+C records interruption while unrelated failures and denied interrupts remain failures', async () => {
  await inTemporaryWorkspace(async root => {
    const history = new RunHistory(join(root, 'interrupt.sqlite'))
    try {
      const terminals = new TerminalRegistry({ runHistory: history })
      const interrupted = terminals.open({ id: 'interrupted', kind: 'pty', command: 'cat', cwd: root, ownerSessionId: OWNER, control: { interrupt: async () => { interrupted.close(130) } } })
      await terminals.interrupt(OWNER, interrupted.id)
      await Promise.resolve()
      expect(history.list(OWNER).find(run => run.sourceId === interrupted.id)).toMatchObject({ state: 'interrupted', exitCode: 130, error: null })
      expect(terminals.wasInterrupted(OWNER, interrupted.id)).toBe(true)
      expect(terminals.wasInterrupted('foreign', interrupted.id)).toBe(false)
      const restored = new TerminalRegistry({ runHistory: history })
      expect(restored.wasInterrupted(OWNER, restored.list(OWNER)[0]!.id)).toBe(true)
      const continued = terminals.open({ id: 'continued', kind: 'pty', command: 'shell', cwd: root, ownerSessionId: OWNER, control: { interrupt: async () => {} } })
      await terminals.interrupt(OWNER, continued.id)
      expect(terminals.inspect(OWNER, continued.id)?.running).toBe(true)
      continued.close(1)
      expect(history.list(OWNER).find(run => run.sourceId === continued.id)?.state).toBe('failed')
      const denied = terminals.open({ id: 'denied', kind: 'pty', command: 'shell', cwd: root, ownerSessionId: OWNER, control: { interrupt: async () => { denied.close(130); throw new Error('Interrupt denied') } } })
      await expect(terminals.interrupt(OWNER, denied.id)).rejects.toThrow('Interrupt denied')
      await Promise.resolve()
      expect(history.list(OWNER).find(run => run.sourceId === denied.id)?.state).toBe('failed')
    } finally { history.close() }
  })
})
