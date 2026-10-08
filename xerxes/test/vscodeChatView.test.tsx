// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { SettingsModal } from '../src/desktop/renderer/Overlays.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import type { SessionRow } from '../src/desktop/renderer/types.js'
import { historyRows, navigateInVscode, VscodeChat } from '../src/vscode/ui/VscodeApp.js'
import { composerImages } from '../src/desktop/renderer/App.js'

/**
 * The VS Code chat view: one conversation column with VS Code's own homes
 * for files, diffs, Git and pickers, instead of the desktop window's
 * sidebar and rails.
 */

const snapshot = (overrides: Partial<Snapshot>): Snapshot => ({
  connection: 'online', cwd: '/repo', model: 'kimi', currentAgentPreset: 'default', agentPresets: [], models: [],
  reasoningLevels: [], contextTokens: null, contextMax: null, ttftMs: null, tokensPerSecond: null, llmDurationMs: 0,
  llmSteps: 0, toolDurationMs: 0, toolSteps: 0, inputTokens: 0, outputTokens: 0, metricPhase: null,
  metricPhaseStartedAt: null, cacheHitRate: null, sessions: [], live: [], fleet: [], skillSuggestions: [],
  creatorTrace: [], backgroundJobs: [], currentId: '', currentTitle: '', sessionKey: '', branch: '', daemonWarning: null,
  costUsd: null, mcpStatus: {}, goal: '', approval: null, question: null, interactionHidden: false, planMode: false,
  turnActive: false, turnFailed: false, turnSeconds: 0, blocks: [], error: null, tab: 'activity', turnCount: 0, queue: [],
  changes: [], plan: null, log: [], failed: null, settingsOpen: false, settingsTab: 'general', paletteOpen: false,
  commands: [], pickerOpen: false, wsMenuOpen: false, sessionMenu: null, streamThinking: true, taskModalOpen: false,
  providers: [], providerModels: {}, providerModelLoading: [], providerModelWarnings: {}, providerTypes: [],
  permissionMode: '', snippets: {},
  ...overrides,
})

const row = (id: string, overrides: Partial<SessionRow> = {}): SessionRow => ({
  id, key: id, title: id, status: 'idle', age: '1d', current: false, kind: 'main', turns: 1, messages: 2,
  cwd: '/repo', untitled: false, ...overrides,
})

const realWindow = globalThis.window
const realDocument = globalThis.document
afterEach(() => {
  ;(globalThis as { window?: unknown }).window = realWindow
  ;(globalThis as { document?: unknown }).document = realDocument
})

test('renders one conversation column with a history title, not the desktop sidebar or rails', () => {
  const html = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({
    currentId: 's1', currentTitle: 'Fix the parser', changes: [{ path: 'a.ts', adds: 3, dels: 1 }] as unknown as Snapshot['changes'],
  }) }))
  expect(html).toContain('class="app atelier xv"')
  expect(html).toContain('Fix the parser')
  expect(html).toContain('aria-label="Message"')
  expect(html).toContain('+3')
  expect(html).not.toContain('session-sidebar')
  expect(html).not.toContain('desktop-rail')
})

test('an older shared runtime is visible, with what its update waits for and a restart', () => {
  // The runtime is shared, so a reloaded window reconnects to the one an older
  // install launched. An update that waited on running work was invisible here,
  // and every fix in the new extension silently never ran.
  const waiting = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({
    connection: 'online', daemonWarning: 'Daemon is older than the app — restart it.',
    runtimeUpdate: 'waiting', runtimeUpdateMessage: 'Update queued. It will install automatically when all running work finishes.',
    runtimeBlockers: ['Work in “Fix the parser”'],
  }) }))
  expect(waiting).toContain('class="xv-runtime"')
  expect(waiting).toContain('Update queued.')
  expect(waiting).toContain('Waiting for: Work in “Fix the parser”')
  expect(waiting).toContain('Restart now')

  const newer = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({
    connection: 'online', daemonWarning: 'The app is older than the daemon — update and restart Xerxes.',
  }) }))
  expect(newer).toContain('Update the extension')
  expect(newer).not.toContain('Restart now')

  const current = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({ connection: 'online', daemonWarning: null }) }))
  expect(current).not.toContain('xv-runtime')
})

test('in VS Code, settings is one page of providers, agent intelligence and approvals with no section tabs', () => {
  ;(globalThis as { document?: unknown }).document = { documentElement: { dataset: { host: 'vscode' } } }
  const html = renderToStaticMarkup(createElement(SettingsModal, { snap: snapshot({ settingsOpen: true }) }))
  expect(html).toContain('modal--single')
  expect(html).toContain('Models')
  expect(html).toContain('Permissions')
  expect(html.indexOf('Agent intelligence')).toBeGreaterThan(html.indexOf('Models'))
  expect(html.indexOf('Agent intelligence')).toBeLessThan(html.lastIndexOf('Permissions'))
  expect(html).not.toContain('mtab')
  expect(html).not.toContain('Channels')
  expect(html).not.toContain('Language servers')
})

test('a folder without VS Code open asks to open one instead of showing a conversation', () => {
  const html = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({ noWorkspace: true }) }))
  expect(html).toContain('Open Folder')
  expect(html).not.toContain('aria-label="Message"')
})

test('links go to VS Code: editor, diff, Source Control and the file quick pick; the rest open as sheets', async () => {
  const calls: string[] = []
  const events: string[] = []
  ;(globalThis as { window?: unknown }).window = { dispatchEvent: (event: CustomEvent<string>) => { events.push(`${event.type}:${event.detail}`); return true } }
  const bridge = {
    openInEditor: async (path: string) => { calls.push(`open:${path}`); return true },
    openDiff: async (path: string) => { calls.push(`diff:${path}`); return true },
    pickFiles: async () => { calls.push('pick'); return ['src/a.ts', 'docs/b c.md'] },
    showSourceControl: async () => { calls.push('scm'); return true },
  }
  const sheets: Array<string | null> = []
  const open = (sheet: string | null) => { sheets.push(sheet) }
  await navigateInVscode(bridge, open, 'files', 'src/a.ts')
  await navigateInVscode(bridge, open, 'review', 'src/a.ts')
  await navigateInVscode(bridge, open, 'review')
  await navigateInVscode(bridge, open, 'files')
  await navigateInVscode(bridge, open, 'agents')
  await navigateInVscode(bridge, open, null)
  expect(calls).toEqual(['open:src/a.ts', 'diff:src/a.ts', 'scm', 'pick'])
  expect(events).toEqual(['xerxes:add-context:@"src/a.ts" @"docs/b c.md"'])
  expect(sheets).toEqual(['agents', null])
})

test('a cancelled file pick adds nothing to the message', async () => {
  const events: unknown[] = []
  ;(globalThis as { window?: unknown }).window = { dispatchEvent: (event: unknown) => { events.push(event); return true } }
  await navigateInVscode({ pickFiles: async () => [] }, () => {}, 'files')
  expect(events).toEqual([])
})

test('history lists main tasks in this folder, the open one included, newest first, filtered by title', () => {
  const snap = snapshot({
    currentId: 'open', currentTitle: 'Open task', cwd: '/repo',
    sessions: [
      row('old', { title: 'Old parser work', activeAt: 1_000 }),
      row('new', { title: 'New lexer work', activeAt: 3_000 }),
      row('elsewhere', { cwd: '/other', activeAt: 4_000 }),
      row('child', { kind: 'subagent', activeAt: 5_000 }),
      row('open', { title: 'stale title', activeAt: 2_000 }),
    ],
  })
  expect(historyRows(snap, '').map(entry => entry.id)).toEqual(['new', 'open', 'old'])
  expect(historyRows(snap, '').find(entry => entry.id === 'open')?.title).toBe('Open task')
  expect(historyRows(snap, 'PARSER').map(entry => entry.id)).toEqual(['old'])
})

test('a pasted screenshot becomes base64 for the runtime, and other files are ignored', async () => {
  const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'shot.png', { type: 'image/png' })
  const text = new File(['hello'], 'notes.txt', { type: 'text/plain' })
  const images = await composerImages([png, text])
  expect(images).toHaveLength(1)
  expect(images[0]).toMatchObject({ mediaType: 'image/png', data: 'iVBORw==', name: 'shot.png' })
})

test('an attached image shows whole above the message bubble, not inside it', () => {
  const png = 'data:image/png;base64,iVBORw0KGgo='
  const html = renderToStaticMarkup(createElement(VscodeChat, { snap: snapshot({
    blocks: [{ kind: 'user', id: 1, text: 'trash tbh', images: [png] }] as unknown as Snapshot['blocks'],
  }) }))
  const image = html.indexOf('class="msg__images"'), bubble = html.indexOf('class="msg msg--user"')
  expect(image).toBeGreaterThan(-1)
  expect(bubble).toBeGreaterThan(image)
  expect(html.slice(bubble)).not.toContain('<img')
})
