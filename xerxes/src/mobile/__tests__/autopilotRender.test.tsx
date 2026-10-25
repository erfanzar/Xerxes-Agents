// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AUTOPILOT_COMPOSER_HINT, AUTOPILOT_EXPLANATION, AUTOPILOT_READ_ONLY_NOTE } from '../../client/autopilot.js'
import { parseAutopilotRow } from '../../client/hubParsers.js'
import type { AutopilotState, HubState } from '../../client/hubStore.js'
import type { AutopilotRow } from '../../client/hubTypes.js'
import { AutopilotSheet } from '../components/AutopilotSheet.js'
import { ServicesProvider } from '../context.js'
import { liveState, withSupervisorConfig } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")

function row(patch: Record<string, unknown> = {}): AutopilotRow {
  return parseAutopilotRow({
    workspace_id: 'w1', label: 'Mac · app', enabled: true, instruction: 'Keep going', expires_at: '2099-01-01T00:00:00.000Z',
    enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', active: true, ...patch,
  })!
}

function withAutopilot(patch: Partial<AutopilotState>, base: HubState = withSupervisorConfig(liveState(), { status: 'configured', canEdit: true })): HubState {
  return { ...base, autopilot: { phase: 'ready', canEdit: true, rows: [], unsupported: false, ...patch } }
}

function sheet(state: HubState): string {
  return plain(renderToStaticMarkup(<ServicesProvider services={makeServices(state, '/xerxes')}><AutopilotSheet onClose={() => undefined} /></ServicesProvider>))
}

describe('Xerxes tab Autopilot', () => {
  it('offers an Autopilot button and no chip while nothing is active', () => {
    const html = plain(render(withAutopilot({ rows: [row({ active: false })] }), '/xerxes'))
    expect(html).toContain('>Autopilot<')
    expect(html).not.toContain('Autopilot on ·')
    expect(html).not.toContain(AUTOPILOT_COMPOSER_HINT)
  })

  it('shows "Autopilot on · N workspaces" and the no-confirmation note when any workspace is active', () => {
    const html = plain(render(withAutopilot({ rows: [row(), row({ workspace_id: 'w2', label: 'Server' }), row({ workspace_id: 'w3', active: false })] }), '/xerxes'))
    expect(html).toContain('Autopilot on · 2 workspaces')
    expect(html).toContain(AUTOPILOT_COMPOSER_HINT)
    expect(plain(render(withAutopilot({ rows: [row()] }), '/xerxes'))).toContain('Autopilot on · 1 workspace<')
  })

  it('labels the button "(view)" for a non-owner and hides it on a Hub without Autopilot', () => {
    expect(plain(render(withAutopilot({ canEdit: false, rows: [row()] }), '/xerxes'))).toContain('Autopilot (view)')
    const old = plain(render(withAutopilot({ unsupported: true }), '/xerxes'))
    expect(old).not.toContain('>Autopilot<')
    expect(old).not.toContain('Autopilot (view)')
  })

  it('renders an auto card as done, with who it acted for and no Confirm or Cancel', () => {
    const state = withAutopilot({ rows: [row()] })
    const withCard: HubState = {
      ...state,
      supervisor: { ...state.supervisor, confirmations: [{ id: 'a1', tool: 'task.steer', summary: 'Steer the task: focus on tests', targets: ['One / First task'], state: 'confirmed', auto: true, confirmedBy: 'Ann' }] },
    }
    const html = plain(render(withCard, '/xerxes'))
    expect(html).toContain('XERXES did: task.steer')
    expect(html).toContain('Steer the task: focus on tests')
    expect(html).toContain('Done automatically on behalf of Ann.')
    expect(html).not.toContain('>Confirm<')
    expect(html).not.toContain('>Cancel<')
    expect(html).not.toContain('Nothing runs until you confirm')
    // An ordinary pending card still asks.
    const ordinary: HubState = { ...state, supervisor: { ...state.supervisor, confirmations: [{ id: 'p1', tool: 'task.stop', summary: 'Stop it', targets: [], state: 'pending' }] } }
    expect(plain(render(ordinary, '/xerxes'))).toContain('>Confirm<')
  })

  it('an auto card that failed says so instead of "done"', () => {
    const state = withAutopilot({ rows: [row()] })
    const failed: HubState = { ...state, supervisor: { ...state.supervisor, confirmations: [{ id: 'a2', tool: 'task.stop', summary: 'Stop it', targets: [], state: 'failed', auto: true, confirmedBy: 'Ann', message: 'The task was already gone.' }] } }
    const html = plain(render(failed, '/xerxes'))
    expect(html).toContain('role="alert"')
    expect(html).toContain('The task was already gone.')
    expect(html).not.toContain('Done automatically')
  })
})

describe('Autopilot sheet', () => {
  it('lets the owner switch each workspace, write an instruction (max 2000) and pick an expiry', () => {
    const html = sheet(withAutopilot({ rows: [row(), row({ workspace_id: 'w2', label: 'Server', enabled: false, active: false, instruction: '', expires_at: null, enabled_by: null, updated_at: null })] }))
    expect(html).toContain('role="dialog"')
    expect(html).toContain(AUTOPILOT_EXPLANATION)
    expect(html).toContain('Permission requests (commands, edits) still come to you')
    expect(html.match(/type="checkbox"/g)).toHaveLength(2)
    expect(html.match(/<textarea/g)).toHaveLength(2)
    expect(html).toContain('maxLength="2000"')
    expect(html).toContain('placeholder="Keep going until the goal is done; take the recommended option"')
    for (const label of ['No expiry', '1 hour', '8 hours', '24 hours', '7 days']) expect(html).toContain(`>${label}<`)
    expect(html).toContain('>Keep current expiry<')
    expect(html).toContain('Mac · app · On')
    expect(html).toContain('Server · Off')
    expect(html).not.toContain(AUTOPILOT_READ_ONLY_NOTE)
    expect(html).not.toMatch(/<button type="submit"[^>]*disabled/)
  })

  it('is read-only for a non-owner: the rows and the reason, no inputs, no Save', () => {
    const html = sheet(withAutopilot({ canEdit: false, rows: [row()] }))
    expect(html).toContain(AUTOPILOT_READ_ONLY_NOTE)
    expect(html).toContain('Mac · app · On')
    expect(html).toContain('Turned on by Ann.')
    expect(html).toContain('Keep going')
    expect(html).not.toContain('<textarea')
    expect(html).not.toContain('type="checkbox"')
    expect(html).not.toContain('<select')
    expect(html).not.toContain('>Save<')
  })

  it('says "Off" for a row that is enabled but no longer active', () => {
    const html = sheet(withAutopilot({ canEdit: false, rows: [row({ active: false })] }))
    expect(html).toContain('Off (expired or the owner changed)')
  })

  it('disables Save while offline and while that workspace is saving', () => {
    const offline = sheet({ ...withAutopilot({ rows: [row()] }), connection: { state: 'reconnecting' } as HubState['connection'] })
    expect(offline).toMatch(/<button type="submit"[^>]*disabled/)
    const saving = sheet(withAutopilot({ rows: [row()], savingId: 'w1' }))
    expect(saving).toContain('Saving…')
    expect(saving).toMatch(/<button type="submit"[^>]*disabled/)
  })

  it('shows loading, a retryable error and a save error', () => {
    expect(sheet(withAutopilot({ phase: 'loading' }))).toContain('Reading Autopilot settings')
    const error = sheet(withAutopilot({ phase: 'error', error: 'timed out' }))
    expect(error).toContain('Could not read Autopilot: timed out')
    expect(error).toContain('Try again')
    expect(sheet(withAutopilot({ rows: [row()], saveError: 'Only the Hub owner can do this.' }))).toContain('Only the Hub owner can do this.')
  })

  it('says so when no workspace is linked', () => {
    expect(sheet(withAutopilot({ rows: [] }))).toContain('No workspace is linked to this Hub yet.')
  })
})
