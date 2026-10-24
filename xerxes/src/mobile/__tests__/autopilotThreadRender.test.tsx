// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { parseAutopilotRow } from '../../client/hubParsers.js'
import type { AutopilotThreadState, HubState } from '../../client/hubStore.js'
import type { AutopilotRow } from '../../client/hubTypes.js'
import { AutopilotSheet } from '../components/AutopilotSheet.js'
import { AutopilotThreadBody } from '../components/XerxesTab.js'
import { ServicesProvider } from '../context.js'
import { liveState, withSupervisorConfig } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")

function row(patch: Record<string, unknown> = {}): AutopilotRow {
  return parseAutopilotRow({
    workspace_id: 'w1', label: 'Mac · app', enabled: true, instruction: '', expires_at: '2099-01-01T00:00:00.000Z',
    enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', active: true, ...patch,
  })!
}

const PENDING = { id: 'p1', tool: 'task.start', summary: 'Start a task', targets: [] as string[], state: 'pending' as const }
const AUTO = { id: 'a1', tool: 'task.steer', summary: 'Steer it', targets: [] as string[], state: 'confirmed' as const, auto: true, confirmedBy: 'Ann' }

function state(opts: { canEdit?: boolean; rows?: AutopilotRow[]; thread?: Partial<AutopilotThreadState> } = {}): HubState {
  return {
    ...withSupervisorConfig(liveState(), { status: 'configured', canEdit: true }),
    selfUserId: 'u1',
    autopilot: { phase: 'ready', canEdit: opts.canEdit ?? true, rows: opts.rows ?? [row()], unsupported: false },
    autopilotThread: { phase: 'ready', messages: [], confirmations: [PENDING, AUTO], ...opts.thread },
  }
}

describe('Xerxes tab Autopilot thread', () => {
  it('shows the owner a thread switch with the pending-card badge', () => {
    const html = plain(render(state(), '/xerxes'))
    expect(html).toContain('Autopilot thread')
    expect(html).toContain('aria-label="1 waiting for you"')
  })

  it('never shows a non-owner the switch or the badge', () => {
    const html = plain(render(state({ canEdit: false }), '/xerxes'))
    expect(html).not.toContain('Autopilot thread')
    expect(html).not.toContain('waiting for you')
  })

  it('hides the switch once the Hub said not_found', () => {
    expect(plain(render(state({ thread: { phase: 'hidden' } }), '/xerxes'))).not.toContain('Autopilot thread')
  })

  it('the thread body keeps Confirm/Cancel on pending cards and renders auto cards as done', () => {
    const html = plain(renderToStaticMarkup(
      <ServicesProvider services={makeServices(state({ thread: { messages: [{ id: 'm1', role: 'system', text: 'Autopilot: woke for a question.', at: '2026-10-21T11:00:00.000Z', source: 'text' }] } }), '/xerxes')}>
        <AutopilotThreadBody />
      </ServicesProvider>,
    ))
    expect(html).toContain('Autopilot: woke for a question.')
    expect(html).toContain('XERXES proposes: task.start')
    expect(html).toContain('>Confirm<')
    expect(html).toContain('>Cancel<')
    expect(html).toContain('XERXES did: task.steer')
    expect(html).toContain('Done automatically on behalf of Ann.')
    // Only the pending card asks.
    expect(html.match(/>Confirm</g)).toHaveLength(1)
  })
})

describe('Autopilot sheet expiry', () => {
  const sheet = (rows: AutopilotRow[]): string =>
    plain(renderToStaticMarkup(<ServicesProvider services={makeServices(state({ rows }), '/xerxes')}><AutopilotSheet onClose={() => undefined} /></ServicesProvider>))

  it('offers "Keep current expiry" for a live expiry', () => {
    const html = sheet([row()])
    expect(html).toContain('>Keep current expiry<')
    expect(html).not.toContain('The previous expiry has passed')
  })

  it('does not offer it once the expiry has passed, and says to choose a new one', () => {
    const html = sheet([row({ expires_at: '2020-01-01T00:00:00.000Z', active: false })])
    expect(html).not.toContain('>Keep current expiry<')
    expect(html).toContain('The previous expiry has passed')
    expect(html).toContain('>No expiry<')
  })
})
