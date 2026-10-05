// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { PanelHost, type PanelDaemon, type PanelPorts } from '../src/vscode/panelHost.js'
import { webviewTransport } from '../src/vscode/webviewTransport.js'
import type { PushFrame, ResultFrame } from '../src/vscode/messages.js'

function fakeDaemon(answer: (method: string, params: Record<string, unknown>) => unknown = () => ({ ok: true })) {
  const handlers = new Set<(type: string, payload: Record<string, unknown>) => void>()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  let disposed = false
  const daemon: PanelDaemon = {
    async call(method, params = {}) { calls.push({ method, params }); return answer(method, params) as Record<string, unknown> },
    onEvent: handler => { handlers.add(handler) },
    offEvent: handler => { handlers.delete(handler) },
    dispose: () => { disposed = true },
  }
  return { daemon, calls, emit: (type: string, payload: Record<string, unknown> = {}) => { for (const handler of handlers) handler(type, payload) }, disposed: () => disposed }
}

function ports(overrides: Partial<PanelPorts> = {}) {
  const memory = new Map<string, string>()
  const opened: string[] = []
  const panels: Array<{ sessionId?: string; fresh?: boolean }> = []
  const notes: string[] = []
  const editor: string[] = []
  let shown = true
  const value: PanelPorts = {
    workspace: '/work/repo',
    workspaceFolders: () => ['/work/repo'],
    remoteName: undefined,
    chooseFolder: async () => '/work/other',
    openFolder: async dir => { opened.push(dir) },
    openPanel: options => { panels.push(options) },
    revealPath: async () => true,
    backgroundData: async name => `data:${name}`,
    memory: { get: key => memory.get(key), set: (key, next) => { if (next === undefined) memory.delete(key); else memory.set(key, next) } },
    notify: message => { notes.push(message) },
    visible: () => shown,
    openFile: async (path, line) => { editor.push(`open:${path}:${line ?? ''}`); return true },
    openDiff: async path => { editor.push(`diff:${path}`); return true },
    pickFiles: async () => { editor.push('pick'); return ['src/a.ts'] },
    showSourceControl: async () => { editor.push('scm'); return true },
    ...overrides,
  }
  return { value, memory, opened, panels, notes, editor, hide: () => { shown = false } }
}

/** A webview and a host wired together through the real frames, as VS Code does. */
function wire(host: (post: (frame: ResultFrame | PushFrame) => void) => PanelHost) {
  const listeners: Array<(event: Event) => void> = []
  const window = { addEventListener: (_type: string, listener: (event: Event) => void) => { listeners.push(listener) } } as unknown as Pick<Window, 'addEventListener'>
  const panel = host(frame => { for (const listener of listeners) listener({ data: structuredClone(frame) } as unknown as Event) })
  const bridge = createBridge(webviewTransport({ postMessage: message => { void panel.receive(structuredClone(message)) } }, window, () => {}))
  return { bridge, panel }
}

test('the desktop renderer\'s bridge runs over the webview channel: calls reach the runtime, events come back', async () => {
  const runtime = fakeDaemon((method) => method === 'initialize' ? { ok: true, session: { id: 'a1b2c3' } } : { ok: true, echoed: method })
  const env = ports()
  let connectedTo = ''
  const { bridge, panel } = wire(post => new PanelHost(env.value, post, dir => { connectedTo = dir; return runtime.daemon }))
  const status: Record<string, unknown> = await bridge.call('runtime.status')
  expect(status).toEqual({ ok: true, echoed: 'runtime.status' })
  expect(connectedTo).toBe('/work/repo')
  const events: unknown[] = []
  bridge.onEvent((event: unknown) => { events.push(event) })
  runtime.emit('status_update', { model: 'gpt-6.1-sol' })
  await Bun.sleep(0)
  expect(events).toEqual([{ type: 'status_update', payload: { model: 'gpt-6.1-sol' } }])
  // The session shown is remembered for the next time VS Code opens this folder.
  await bridge.call('initialize', {})
  expect(env.memory.get('xerxes.resumeSession')).toBe('a1b2c3')
  expect(await bridge.getResumeSession()).toBe('a1b2c3')
  panel.dispose()
  expect(runtime.disposed()).toBe(true)
})

test('the workspace is the VS Code folder; other folders open in a new VS Code window', async () => {
  const env = ports()
  const { bridge } = wire(post => new PanelHost(env.value, post, () => fakeDaemon().daemon))
  expect(await bridge.getWorkspace()).toBe('/work/repo')
  expect(await bridge.getContextScope()).toBe('local')
  expect(await bridge.useWorkspace('/work/repo', 'abc')).toBe('/work/repo')
  expect(env.opened).toEqual([])
  expect(await bridge.useWorkspace('/elsewhere')).toBeNull()
  expect(await bridge.chooseWorkspace()).toBe('/work/other')
  expect(env.opened).toEqual(['/elsewhere', '/work/other'])
  // Another task of this folder opens in another Xerxes panel.
  await bridge.openWorkspaceWindow('/work/repo', 'task-2')
  expect(env.panels).toEqual([{ sessionId: 'task-2' }])
  expect(await bridge.openWorkspaceWindow('/work/repo', 'task-3', { existingOnly: true })).toBeNull()
})

test('a window with no folder says how to start, instead of failing silently', async () => {
  const env = ports({ workspace: null })
  const { bridge } = wire(post => new PanelHost(env.value, post, () => fakeDaemon().daemon))
  expect(await bridge.getWorkspace()).toBeNull()
  await expect(bridge.call('runtime.status')).rejects.toThrow('Open a folder in VS Code to use Xerxes.')
})

test('desktop-only features answer plainly: updates come from VS Code, SSH goes through Remote-SSH, dictation is not here yet', async () => {
  const env = ports()
  const { bridge } = wire(post => new PanelHost(env.value, post, () => fakeDaemon().daemon))
  expect(await bridge.appUpdate('check')).toEqual({ phase: 'idle' })
  expect(await bridge.remote('status')).toMatchObject({ ok: true, machine: null })
  await expect(bridge.remote('connect', {})).rejects.toThrow('Remote-SSH')
  await expect(bridge.voice('check')).rejects.toThrow('not available in VS Code')
  expect(await bridge.getLoginItem()).toBe(false)
  expect(await bridge.isOccluded()).toBe(false)
})

test('needs-input and finished tasks notify only while the panel is hidden, and only when notifications are on', async () => {
  const runtime = fakeDaemon()
  const env = ports()
  const { bridge } = wire(post => new PanelHost(env.value, post, () => runtime.daemon))
  await bridge.call('runtime.status')
  runtime.emit('turn_end', { status: 'completed' })
  expect(env.notes).toEqual([])
  env.hide()
  runtime.emit('turn_end', { status: 'completed' })
  runtime.emit('approval_request', { description: 'Run tests' })
  expect(env.notes).toHaveLength(2)
  expect(env.notes[0]).toContain('Task finished')
  expect(env.notes[1]).toContain('Approval needed')
  await bridge.setNotifications(false)
  runtime.emit('turn_end', { status: 'completed' })
  expect(env.notes).toHaveLength(2)
})

test('a malformed frame is ignored and an unknown channel is answered with an error', async () => {
  const posted: Array<ResultFrame | PushFrame> = []
  const host = new PanelHost(ports().value, frame => { posted.push(frame) }, () => fakeDaemon().daemon)
  await host.receive({ kind: 'invoke', id: 'x', channel: 'desktop:workspace', args: [] })
  await host.receive('not a frame')
  expect(posted).toEqual([])
  await host.receive({ kind: 'invoke', id: 7, channel: 'desktop:nope', args: [] })
  expect(posted).toEqual([{ kind: 'result', id: 7, ok: false, error: 'Unknown channel desktop:nope' }])
})

test('files, diffs, the file picker and Git go to VS Code through the host channels', async () => {
  const env = ports()
  const { bridge } = wire(post => new PanelHost(env.value, post, () => fakeDaemon().daemon))
  expect(await bridge.openInEditor('src/a.ts', 12)).toBe(true)
  expect(await bridge.openInEditor('src/b.ts')).toBe(true)
  expect(await bridge.openDiff('src/a.ts')).toBe(true)
  expect(await bridge.pickFiles()).toEqual(['src/a.ts'])
  expect(await bridge.showSourceControl()).toBe(true)
  expect(env.editor).toEqual(['open:src/a.ts:12', 'open:src/b.ts:', 'diff:src/a.ts', 'pick', 'scm'])
})

test('a failure opening a file in VS Code reaches the view as an error', async () => {
  const env = ports({ openFile: async () => { throw new Error('File not found: gone.ts') } })
  const { bridge } = wire(post => new PanelHost(env.value, post, () => fakeDaemon().daemon))
  await expect(bridge.openInEditor('gone.ts')).rejects.toThrow('File not found: gone.ts')
})
