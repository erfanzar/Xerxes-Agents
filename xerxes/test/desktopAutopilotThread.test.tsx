// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { parseAutopilotRow } from '../src/client/hubParsers.js'
import { autopilotPendingCount, autopilotThreadAvailable, initialHubState, initialSupervisorConfig, type AutopilotState, type AutopilotThreadState, type HubState } from '../src/client/hubStore.js'
import type { AutopilotRow } from '../src/client/hubTypes.js'
import { SupervisorView, type SupervisorViewProps } from '../src/desktop/renderer/SupervisorPage.js'

const noop = (): void => {}

function row(patch: Record<string, unknown> = {}): AutopilotRow {
  const parsed = parseAutopilotRow({
    workspace_id: 'w1', label: 'One · app', enabled: true, instruction: '', expires_at: '2099-01-01T00:00:00.000Z',
    enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', active: true, ...patch,
  })
  if (!parsed) throw new Error('fixture row is malformed')
  return parsed
}

const PENDING = { id: 'p1', tool: 'task.start', summary: 'Start a task', targets: [] as string[], state: 'pending' as const }
const AUTO = { id: 'a1', tool: 'task.steer', summary: 'Steer it', targets: [] as string[], state: 'confirmed' as const, auto: true, confirmedBy: 'Ann' }

function hub(opts: { canEdit?: boolean; rows?: AutopilotRow[]; thread?: Partial<AutopilotThreadState>; self?: string | undefined } = {}): HubState {
  const autopilot: AutopilotState = { phase: 'ready', canEdit: opts.canEdit ?? true, rows: opts.rows ?? [row()], unsupported: false }
  return {
    ...initialHubState(),
    connection: { state: 'live' },
    ...('self' in opts ? (opts.self === undefined ? {} : { selfUserId: opts.self }) : { selfUserId: 'u1' }),
    supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
    supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', status: 'configured', canEdit: true, config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'model-a', effort: 'high' } },
    autopilot,
    autopilotThread: { phase: 'ready', messages: [], confirmations: [PENDING, AUTO], ...opts.thread },
  }
}

function view(state: HubState, extra: Partial<SupervisorViewProps> = {}): string {
  return renderToStaticMarkup(createElement(SupervisorView, {
    state, hubLabel: 'hub.example.ts.net', now: Date.now(), onSend: async () => true, onRespond: noop, onReopen: noop,
    onOpenSettings: noop, chooserOpen: false, onChooserChange: noop, onChooserShown: noop, onOptions: noop, onSaveModel: async () => true,
    autopilotOpen: false, onAutopilotChange: noop, onSaveAutopilot: async () => true, onRetryAutopilot: noop,
    autopilotThreadOpen: false, onAutopilotThreadChange: noop, onRespondAutopilot: noop,
    ...extra,
  }))
}

test('the owner with an active workspace gets an Autopilot thread switch with the pending-card badge', () => {
  const html = view(hub())
  expect(html).toContain('Autopilot thread')
  // One pending card; the auto card is a record and is not counted.
  expect(html).toContain('aria-label="1 waiting for you"')
  expect(autopilotPendingCount(hub())).toBe(1)
  // The shared chat is still showing.
  expect(html).toContain('aria-label="Message to XERXES"')
})

test('open, the thread shows pending cards with Confirm/Cancel and auto cards as done, in place of the shared chat', () => {
  const html = view(hub({ thread: { messages: [{ id: 'm1', role: 'system', text: 'Autopilot: woke for a question in “Fix tests”.', at: new Date().toISOString(), source: 'text' }] } }), { autopilotThreadOpen: true })
  expect(html).toContain('aria-label="Autopilot thread"')
  expect(html).toContain('Autopilot: woke for a question')
  expect(html).toContain('>Confirm<')
  expect(html).toContain('>Cancel<')
  expect(html).toContain('Done automatically on behalf of Ann.')
  expect(html).toContain('Back to chat')
  expect(html).not.toContain('aria-label="Message to XERXES"')
})

test('a non-owner never gets the view', () => {
  const state = hub({ canEdit: false })
  expect(autopilotThreadAvailable(state)).toBe(false)
  const html = view(state, { autopilotThreadOpen: true })
  expect(html).not.toContain('Autopilot thread')
  expect(html).not.toContain('waiting for you')
  expect(html).toContain('aria-label="Message to XERXES"')
})

test('no view without a known signed-in user, without an active workspace, or after the Hub said not_found', () => {
  expect(autopilotThreadAvailable(hub({ self: undefined }))).toBe(false)
  expect(autopilotThreadAvailable(hub({ rows: [row({ active: false })], thread: { confirmations: [] } }))).toBe(false)
  expect(autopilotThreadAvailable(hub({ thread: { phase: 'hidden' } }))).toBe(false)
  expect(view(hub({ thread: { phase: 'hidden' } }), { autopilotThreadOpen: true })).not.toContain('Autopilot thread')
  // Cards still waiting for the owner keep the view reachable after Autopilot turned off.
  expect(autopilotThreadAvailable(hub({ rows: [row({ active: false })] }))).toBe(true)
})

test('an expiry that has passed is not offered as "Keep current expiry" and says so', () => {
  const lapsed = row({ expires_at: '2020-01-01T00:00:00.000Z', active: false })
  const html = view(hub({ rows: [lapsed] }), { autopilotOpen: true })
  expect(html).not.toContain('>Keep current expiry<')
  expect(html).toContain('The previous expiry has passed')
  for (const label of ['No expiry', '1 hour', '7 days']) expect(html).toContain(`>${label}<`)
  const live = view(hub({ rows: [row()] }), { autopilotOpen: true })
  expect(live).toContain('>Keep current expiry<')
  expect(live).not.toContain('The previous expiry has passed')
})
