// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { ReactionDispatcher } from '../src/runtime/reactionDispatcher.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerMonitorTools } from '../src/tools/monitorTools.js'

function fixture() {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry()
  const events: string[] = []
  const monitors = new TerminalMonitors(terminals, history, (_monitor, event) => events.push(event.text))
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'source', cwd: '/repo', kind: 'background', command: 'build' })
  return { history, terminals, monitors, terminal, events, close: () => { monitors.close(); history.close() } }
}

test('monitor framing deduplicates matches without model polling or consuming terminal output', () => {
  const f = fixture()
  try {
    const watch = f.monitors.start('owner', { terminalId: 'source', match: 'error', maxEvents: 2 })
    f.terminal.append('healthy\n'.repeat(3000))
    expect(f.events).toEqual([])
    f.terminal.append('ERR')
    f.terminal.append('OR one\nERROR one\n')
    expect(f.events).toEqual(['ERROR one'])
    f.terminal.append('error two\nerror three\n')
    expect(f.events).toEqual(['ERROR one', 'error two'])
    expect(f.monitors.list('owner')[0]?.state).toBe('limit-reached')
    expect(f.history.inspect('owner', watch.id)?.state).toBe('succeeded')
    expect(f.terminals.inspect('owner', 'source')?.output).toContain('error three')
    expect(f.terminals.inspect('owner', 'source')?.running).toBe(true)
  } finally { f.close() }
})

test('monitor expiry and explicit stop detach observers without killing the source', async () => {
  const f = fixture()
  try {
    const watch = f.monitors.start('owner', { terminalId: 'source', match: 'error', durationMs: 100 })
    await Bun.sleep(150)
    expect(f.monitors.list('owner')[0]?.state).toBe('expired')
    expect(f.history.inspect('owner', watch.id)?.unread).toBe(false)
    f.terminal.append('error after expiry\n')
    expect(f.events).toEqual([])
    const second = f.monitors.start('owner', { terminalId: 'source', match: 'error' })
    expect(() => f.monitors.stop('other', second.id)).toThrow('Unknown monitor')
    f.monitors.stop('owner', second.id)
    expect(f.terminals.inspect('owner', 'source')?.running).toBe(true)
    expect(() => f.monitors.start('other', { terminalId: 'source', match: 'error' })).toThrow('owned')
  } finally { f.close() }
})

test('source completion flushes the final unterminated line and closes the watch', () => {
  const f = fixture()
  try {
    f.monitors.start('owner', { terminalId: 'source', match: 'failed' })
    f.terminal.append('build failed')
    f.terminal.close(1)
    expect(f.events).toEqual(['build failed'])
    expect(f.monitors.list('owner')[0]?.state).toBe('source-ended')
  } finally { f.close() }
})

test('registered monitor tools enforce trusted session ownership and validated bounds', async () => {
  const f = fixture()
  const tools = new ToolRegistry()
  registerMonitorTools(tools, f.monitors)
  const call = { id: 'watch', type: 'function' as const, function: { name: 'monitor_terminal', arguments: { terminal_id: 'source', match: 'error', duration_seconds: 60 } } }
  try {
    await expect(tools.execute(call, { metadata: {} })).rejects.toThrow()
    const created = JSON.parse(await tools.execute(call, { sessionId: 'owner', metadata: {} }))
    expect(created.state).toBe('watching')
    f.monitors.disposeOwner('owner')
    expect(f.monitors.list('owner')[0]?.state).toBe('stopped')
    expect(() => f.monitors.start('owner', { terminalId: 'source', match: 'error', durationMs: Infinity })).toThrow()
  } finally { f.close() }
})

test('matches survive oversized unterminated lines and split boundaries with bounded evidence', () => {
  const f = fixture()
  try {
    f.monitors.start('owner', { terminalId: 'source', match: 'error' })
    f.terminal.append('ERROR early ')
    f.terminal.append('x'.repeat(100_000))
    expect(f.events).toHaveLength(0)
    f.terminal.append('\n')
    expect(f.events).toHaveLength(1)
    expect(f.events[0]).toStartWith('ERROR early ')
    expect(f.events[0]).toEndWith('[suffix omitted]')
    expect(f.events[0]!.length).toBeLessThan(8300)
    f.terminal.append('x'.repeat(100_000) + 'ERR')
    f.terminal.append('OR late\r\nhealthy\n')
    expect(f.events).toHaveLength(2)
    expect(f.events[1]).toContain('ERROR late')
    expect(f.events[1]).toStartWith('[prefix omitted]')
    expect(f.events[1]).not.toContain('\r')
    f.terminal.append('error final' + 'y'.repeat(100_000))
    f.terminal.close(1)
    expect(f.events).toHaveLength(3)
    expect(f.events[2]).toContain('error final')
    expect(f.monitors.list('owner')[0]?.state).toBe('source-ended')
  } finally { f.close() }
})

test('monitor stops scanning a large burst at its event limit', () => {
  const f = fixture()
  try {
    f.monitors.start('owner', { terminalId: 'source', match: 'error', maxEvents: 1 })
    f.terminal.append('error first\n' + 'error more\n'.repeat(50_000))
    expect(f.events).toEqual(['error first'])
    expect(f.monitors.list('owner')[0]?.state).toBe('limit-reached')
  } finally { f.close() }
})

test('storage failure fails and detaches a monitor without losing other watches or killing commands', () => {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry()
  const errors: unknown[] = []
  const events: string[] = []
  const monitors = new TerminalMonitors(terminals, history, (_watch, event) => events.push(event.text), error => errors.push(error))
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'source', cwd: '/repo', kind: 'background', command: 'build' })
  const checkpoint = history.checkpointOutput.bind(history)
  try {
    const broken = monitors.start('owner', { terminalId: 'source', match: 'error' })
    const healthy = monitors.start('owner', { terminalId: 'source', match: 'ready' })
    history.checkpointOutput = (owner, id, output, truncated) => {
      if (id === broken.id) throw new Error('disk full')
      checkpoint(owner, id, output, truncated)
    }
    terminal.append('error failed\nready now\n')
    const failed = monitors.list('owner').find(watch => watch.id === broken.id)
    expect(failed?.state).toBe('failed')
    expect(failed?.error).toBe('disk full')
    expect(history.inspect('owner', broken.id)?.state).toBe('failed')
    expect(events).toEqual(['ready now'])
    terminal.append('error again\n')
    expect(errors).toHaveLength(1)
    expect(monitors.list('owner').find(watch => watch.id === healthy.id)?.state).toBe('watching')
    expect(terminals.inspect('owner', 'source')?.running).toBe(true)
  } finally { monitors.close(); history.close() }
})

test('an unavailable history store leaves failed health visible and shutdown detaches every watch', () => {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry()
  const errors: unknown[] = []
  const events: string[] = []
  const monitors = new TerminalMonitors(terminals, history, (_watch, event) => events.push(event.text), error => errors.push(error))
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'source', cwd: '/repo', kind: 'background', command: 'build' })
  monitors.start('owner', { terminalId: 'source', match: 'error' })
  monitors.start('owner', { terminalId: 'source', match: 'ready' })
  history.close()
  expect(() => monitors.close()).not.toThrow()
  expect(monitors.list('owner').map(watch => watch.state)).toEqual(['failed', 'failed'])
  expect(monitors.list('owner').every(watch => Boolean(watch.error))).toBe(true)
  terminal.append('error later\nready later\n')
  expect(events).toEqual([])
  expect(errors.length).toBeGreaterThan(0)
  expect(terminals.inspect('owner', 'source')?.running).toBe(true)
})

test('stopping a reactive watch aborts its reaction while leaving the source alive', async () => {
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry()
  let started!: () => void
  const running = new Promise<void>(resolve => { started = resolve })
  let signal: AbortSignal | undefined
  let work: Promise<void> | undefined
  const dispatcher = new ReactionDispatcher(mailbox, { admit: async (_owner, execute) => execute(), run: async (_claim, abort) => {
    signal = abort; started()
    await new Promise<void>(resolve => abort.addEventListener('abort', () => resolve(), { once: true }))
  } })
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => {
    mailbox.offer(watch.owner, watch.id, event.sequence)
    work = dispatcher.dispatch(watch.owner)
  }, undefined, mailbox)
  const tools = new ToolRegistry()
  registerMonitorTools(tools, monitors)
  try {
    const terminal = terminals.open({ ownerSessionId: 'owner', id: 'source', cwd: '/repo', kind: 'background', command: 'build' })
    const watch = monitors.start('owner', { terminalId: 'source', match: 'error', reaction: { maxReactions: 2, maxDurationMs: 5000 } })
    terminal.append('error now\n')
    await running
    await tools.execute({ id: 'stop', type: 'function', function: { name: 'stop_monitor', arguments: { monitor_id: watch.id } } }, { sessionId: 'owner', metadata: {} })
    expect(signal?.aborted).toBe(true)
    await work
    expect(mailbox.unresolved('owner')).toEqual([])
    expect(mailbox.offer('owner', watch.id, 2)).toBe(false)
    expect(terminals.inspect('owner', 'source')?.running).toBe(true)
  } finally { await dispatcher.close(); monitors.close(); mailbox.close(); history.close() }
})

test.each([false, true])('completion watch records one durable reaction when attached after exit: %s', alreadyExited => {
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry()
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'build', cwd: '/repo', command: 'build', kind: 'background' })
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => { mailbox.offer(watch.owner, watch.id, event.sequence) }, undefined, mailbox)
  try {
    terminal.append('final diagnostic')
    if (alreadyExited) terminal.close(7)
    expect(() => monitors.start('other', { terminalId: 'build', trigger: 'completion' })).toThrow('owned')
    const watch = monitors.start('owner', { terminalId: 'build', trigger: 'completion', reaction: { maxReactions: 1, maxDurationMs: 1000 } })
    if (!alreadyExited) {
      expect(mailbox.claim('owner')).toBeUndefined()
      terminal.close(7)
    }
    terminal.close(7)
    expect(history.events('owner', watch.id, 0, 20).events).toHaveLength(1)
    const event = history.events('owner', watch.id, 0, 20).events[0]!
    expect(JSON.parse(event.text)).toMatchObject({ exit_code: 7, output: 'final diagnostic' })
    expect(monitors.inspect('owner', watch.id).state).toBe('source-ended')
    const claim = mailbox.claim('owner')!
    expect(claim.throughSequence).toBe(1)
    mailbox.settle(claim, 'completed')
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { monitors.close(); mailbox.close(); history.close() }
})

test('a reaction without a timeout may run until its watch expires, not 60 seconds', () => {
  // A reaction is a turn. A fixed 60s default cancelled every reaction on a
  // large-context model ("Reaction deadline exceeded") before it finished.
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry()
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'build', cwd: '/repo', command: 'build', kind: 'background' })
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => { mailbox.offer(watch.owner, watch.id, event.sequence) }, undefined, mailbox)
  try {
    const startedAt = Date.now()
    const watch = monitors.start('owner', { terminalId: 'build', trigger: 'completion', durationMs: 86_400_000, reaction: { maxReactions: 1 } })
    terminal.close(0)
    const claim = mailbox.claim('owner')!
    expect(claim.deadline).toBeGreaterThan(startedAt + 86_400_000 - 5_000)
    expect(claim.deadline).toBeLessThanOrEqual(watch.expiresAt)
    mailbox.settle(claim, 'completed')
    // An explicit timeout is still honoured, and may exceed the old 10-minute ceiling.
    const second = terminals.open({ ownerSessionId: 'owner', id: 'second', cwd: '/repo', command: 'test', kind: 'background' })
    monitors.start('owner', { terminalId: 'second', trigger: 'completion', durationMs: 86_400_000, reaction: { maxReactions: 1, maxDurationMs: 3_600_000 } })
    second.close(0)
    const timed = mailbox.claim('owner')!
    expect(timed.deadline - Date.now()).toBeGreaterThan(3_600_000 - 5_000)
    expect(timed.deadline - Date.now()).toBeLessThanOrEqual(3_600_000)
    mailbox.settle(timed, 'completed')
  } finally { monitors.close(); mailbox.close(); history.close() }
})

test('stopping a completion watch revokes follow-up and captures no exit event', () => {
  const f = fixture()
  try {
    const watch = f.monitors.start('owner', { terminalId: 'source', trigger: 'completion' })
    f.monitors.stop('owner', watch.id)
    f.terminal.close(0)
    expect(f.events).toHaveLength(0)
    expect(f.history.events('owner', watch.id, 0, 20).events).toHaveLength(0)
  } finally { f.close() }
})

test('completion evidence fits the durable event limit after JSON escaping', () => {
  const f = fixture()
  try {
    const watch = f.monitors.start('owner', { terminalId: 'source', trigger: 'completion' })
    f.terminal.append('\u0001'.repeat(20_000))
    f.terminal.close(0)
    const event = f.history.events('owner', watch.id, 0, 20).events[0]!
    expect(event.text.length).toBeLessThanOrEqual(8300)
    expect(JSON.parse(event.text)).toMatchObject({ output_truncated: true, exit_code: 0 })
    expect(f.monitors.inspect('owner', watch.id).state).toBe('source-ended')
  } finally { f.close() }
})

test.each([false, true])('completion delivery failure preserves durable success and the accepted reaction (already exited: %s)', alreadyExited => {
  const history = new RunHistory(':memory:')
  const mailbox = new ReactionMailbox(':memory:')
  const terminals = new TerminalRegistry()
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'build', cwd: '/repo', command: 'build', kind: 'background' })
  const errors: unknown[] = []
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => {
    mailbox.offer(watch.owner, watch.id, event.sequence)
    throw new Error('UI disconnected')
  }, error => errors.push(error), mailbox)
  try {
    if (alreadyExited) terminal.close(0)
    const watch = monitors.start('owner', { terminalId: 'build', trigger: 'completion', reaction: { maxReactions: 1, maxDurationMs: 1000 } })
    if (!alreadyExited) terminal.close(0)
    expect(monitors.inspect('owner', watch.id)).toMatchObject({ state: 'source-ended', deliveryError: 'UI disconnected' })
    expect(history.inspect('owner', watch.id)?.state).toBe('succeeded')
    expect(history.events('owner', watch.id, 0, 10).events).toHaveLength(1)
    expect(errors).toHaveLength(1)
    const claim = mailbox.claim('owner')!
    expect(claim.throughSequence).toBe(1)
    mailbox.settle(claim, 'completed')
    expect(mailbox.claim('owner')).toBeUndefined()
  } finally { monitors.close(); mailbox.close(); history.close() }
})

test('output watch reports notification failure and clears it after later successful delivery', () => {
  const history = new RunHistory(':memory:')
  const terminals = new TerminalRegistry()
  const terminal = terminals.open({ ownerSessionId: 'owner', id: 'build', cwd: '/repo', command: 'build', kind: 'background' })
  let fail = true
  const errors: unknown[] = []
  const monitors = new TerminalMonitors(terminals, history, () => { if (fail) throw new Error('delivery failed') }, error => errors.push(error))
  try {
    const watch = monitors.start('owner', { terminalId: 'build', match: 'error' })
    terminal.append('error first\n')
    expect(monitors.inspect('owner', watch.id)).toMatchObject({ state: 'watching', deliveryError: 'delivery failed' })
    fail = false
    terminal.append('error second\n')
    expect(monitors.inspect('owner', watch.id).deliveryError).toBeUndefined()
    expect(history.events('owner', watch.id, 0, 10).events).toHaveLength(2)
    expect(errors).toHaveLength(1)
  } finally { monitors.close(); history.close() }
})
