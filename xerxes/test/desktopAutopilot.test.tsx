// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AUTOPILOT_COMPOSER_HINT, AUTOPILOT_EXPLANATION, AUTOPILOT_READ_ONLY_NOTE } from '../src/client/autopilot.js'
import { parseAutopilotRow } from '../src/client/hubParsers.js'
import { initialHubState, initialSupervisorConfig, type AutopilotState, type HubState } from '../src/client/hubStore.js'
import type { AutopilotRow } from '../src/client/hubTypes.js'
import { ActionConfirmationCard } from '../src/desktop/renderer/ActionConfirmationCard.js'
import { SupervisorView, type SupervisorViewProps } from '../src/desktop/renderer/SupervisorPage.js'

const noop = (): void => {}

function row(patch: Record<string, unknown> = {}): AutopilotRow {
  const parsed = parseAutopilotRow({
    workspace_id: 'w1', label: 'One · app', enabled: true, instruction: 'Keep going', expires_at: '2026-10-25T00:00:00.000Z',
    enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', active: true, ...patch,
  })
  if (!parsed) throw new Error('fixture row is malformed')
  return parsed
}

function state(autopilot: Partial<AutopilotState> = {}, patch: Partial<HubState> = {}): HubState {
  return {
    ...initialHubState(),
    connection: { state: 'live' },
    supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
    supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', status: 'configured', canEdit: true, config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'model-a', effort: 'high' } },
    autopilot: { phase: 'ready', canEdit: true, rows: [], unsupported: false, ...autopilot },
    ...patch,
  }
}

function view(hub: HubState, extra: Partial<SupervisorViewProps> = {}): string {
  return renderToStaticMarkup(createElement(SupervisorView, {
    state: hub, hubLabel: 'hub.example.ts.net', now: Date.now(), onSend: async () => true, onRespond: noop, onReopen: noop,
    onOpenSettings: noop, chooserOpen: false, onChooserChange: noop, onChooserShown: noop, onOptions: noop, onSaveModel: async () => true,
    autopilotOpen: false, onAutopilotChange: noop, onSaveAutopilot: async () => true, onRetryAutopilot: noop,
    ...extra,
  }))
}

test('the header has an Autopilot button beside Model, and no chip while nothing is active', () => {
  const html = view(state({ rows: [row({ active: false })] }))
  expect(html).toMatch(/<button[^>]*aria-expanded="false"[^>]*>Autopilot</)
  expect(html).toContain('>Model<')
  expect(html).not.toContain('Autopilot on ·')
  expect(html).toContain('XERXES proposes; nothing runs until you confirm.')
  expect(html).not.toContain(AUTOPILOT_COMPOSER_HINT)
})

test('the chip counts only workspaces the Hub reports as active', () => {
  const one = view(state({ rows: [row(), row({ workspace_id: 'w2', active: false })] }))
  expect(one).toMatch(/class="supervisor__autopilot"[^>]*>Autopilot on · 1 workspace</)
  const two = view(state({ rows: [row(), row({ workspace_id: 'w2' })] }))
  expect(two).toContain('Autopilot on · 2 workspaces')
})

test('while Autopilot is on the composer hint says only some actions run unconfirmed, and not while another person is in the thread', () => {
  const html = view(state({ rows: [row()] }))
  expect(html).toContain(AUTOPILOT_COMPOSER_HINT)
  expect(html).toContain('most actions wait for your Confirm')
  expect(AUTOPILOT_COMPOSER_HINT).toContain('Some actions')
  expect(AUTOPILOT_COMPOSER_HINT).toContain('another person')
})

test('the owner gets a toggle, a 2000-character instruction box and the expiry choices per workspace', () => {
  const html = view(state({ rows: [row(), row({ workspace_id: 'w2', label: 'Two · api', enabled: false, active: false, instruction: '', expires_at: null, enabled_by: null, updated_at: null })] }), { autopilotOpen: true })
  expect(html).toContain('aria-label="Autopilot"')
  expect(html).toContain(AUTOPILOT_EXPLANATION)
  expect(html).toContain('Permission requests (commands, edits) still come to you')
  expect(html.match(/type="checkbox"/g)).toHaveLength(2)
  expect(html.match(/<textarea[^>]*maxLength="2000"/g)).toHaveLength(2)
  expect(html).toContain('placeholder="Keep going until the goal is done; take the recommended option"')
  for (const label of ['No expiry', '1 hour', '8 hours', '24 hours', '7 days']) expect(html).toContain(`>${label}<`)
  expect(html).toContain('One · app')
  expect(html).toContain('Two · api')
  expect(html).toContain('aria-expanded="true"')
  expect(html).not.toContain(AUTOPILOT_READ_ONLY_NOTE)
  expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Save</)
})

test('a non-owner sees the rows and the reason, with no inputs and no Save', () => {
  const html = view(state({ canEdit: false, rows: [row()] }), { autopilotOpen: true })
  expect(html).toContain(AUTOPILOT_READ_ONLY_NOTE)
  expect(html).toContain('Turned on by Ann.')
  expect(html).toContain('Keep going')
  // The composer has its own textarea, so look only inside the Autopilot panel.
  const panel = /<section class="autopilot"[\s\S]*?<\/section>/.exec(html)?.[0] ?? ''
  expect(panel).toContain('Turned on by Ann.')
  expect(panel).not.toContain('<textarea')
  expect(panel).not.toContain('type="checkbox"')
  expect(panel).not.toContain('<select')
  expect(panel).not.toContain('>Save<')
})

test('a row that is enabled but not active reads as Off, never On', () => {
  const html = view(state({ canEdit: false, rows: [row({ active: false })] }), { autopilotOpen: true })
  expect(html).toContain('Off (expired or the owner changed)')
  expect(html).not.toContain('Autopilot on ·')
})

test('Save is disabled while offline or while that workspace saves', () => {
  const offline = view(state({ rows: [row()] }, { connection: { state: 'reconnecting', attempt: 1 } as HubState['connection'] }), { autopilotOpen: true })
  expect(offline).toMatch(/<button[^>]*disabled=""[^>]*>Save</)
  const saving = view(state({ rows: [row()], savingId: 'w1' }), { autopilotOpen: true })
  expect(saving).toMatch(/<button[^>]*disabled=""[^>]*>Saving…</)
})

test('loading, a retryable read error, a save error and an empty list are all stated', () => {
  expect(view(state({ phase: 'loading', rows: [] }), { autopilotOpen: true })).toContain('Reading Autopilot settings')
  const error = view(state({ phase: 'error', error: 'timed out' }), { autopilotOpen: true })
  expect(error).toContain('Could not read Autopilot: timed out')
  expect(error).toContain('>Try again<')
  expect(view(state({ rows: [row()], saveError: 'Only the Hub owner can do this.' }), { autopilotOpen: true })).toContain('Only the Hub owner can do this.')
  expect(view(state({ rows: [] }), { autopilotOpen: true })).toContain('No workspace is linked to this Hub yet.')
})

test('a Hub without Autopilot has no Autopilot control at all', () => {
  const html = view(state({ unsupported: true }), { autopilotOpen: true })
  expect(html).not.toContain('>Autopilot<')
  expect(html).not.toContain('aria-label="Autopilot"')
})

test('an auto card is a record: done on behalf of the owner, no Confirm or Cancel', () => {
  const item = { id: 'a1', tool: 'task.steer', summary: 'Steer the task: focus on tests', targets: ['One / First task'], state: 'confirmed' as const, auto: true, confirmedBy: 'Ann' }
  const html = renderToStaticMarkup(createElement(ActionConfirmationCard, { item, canAct: true, onRespond: noop }))
  expect(html).toContain('XERXES did: task.steer')
  expect(html).toContain('Done automatically on behalf of Ann.')
  expect(html).toContain('data-auto="true"')
  expect(html).not.toContain('>Confirm<')
  expect(html).not.toContain('>Cancel<')
  expect(html).not.toContain('Nothing runs until you confirm')
  // Without a name it still does not pretend a person confirmed.
  const anonymous = renderToStaticMarkup(createElement(ActionConfirmationCard, { item: { ...item, confirmedBy: undefined } as never, canAct: true, onRespond: noop }))
  expect(anonymous).toContain('Done automatically by Autopilot.')
})

test('an auto card that failed shows the error, not "done"', () => {
  const item = { id: 'a2', tool: 'task.stop', summary: 'Stop it', targets: [] as string[], state: 'failed' as const, auto: true, confirmedBy: 'Ann', message: 'The task was already gone.' }
  const html = renderToStaticMarkup(createElement(ActionConfirmationCard, { item, canAct: true, onRespond: noop }))
  expect(html).toContain('role="alert"')
  expect(html).toContain('The task was already gone.')
  expect(html).not.toContain('Done automatically')
})

test('an ordinary pending card still asks for Confirm, even while Autopilot is on', () => {
  const hub = state({ rows: [row()] }, {
    supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [{ id: 'p1', tool: 'task.stop', summary: 'Stop it', targets: [], state: 'pending' }] },
  })
  const html = view(hub)
  expect(html).toContain('>Confirm<')
  expect(html).toContain('>Cancel<')
})
