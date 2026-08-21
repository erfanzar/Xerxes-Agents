// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

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
