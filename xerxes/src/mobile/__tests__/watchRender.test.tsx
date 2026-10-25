// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { parseWatchRow } from '../../client/hubParsers.js'
import type { HubState, WatchState } from '../../client/hubStore.js'
import { watchKey } from '../../client/hubStore.js'
import type { WatchRow } from '../../client/hubTypes.js'
import { WATCH_EXPLANATION, WATCH_READ_ONLY_NOTE } from '../../client/watch.js'
import { ServicesProvider } from '../context.js'
import { WatchSheet } from '../components/WatchSheet.js'
import { liveState, summary, withSupervisorConfig } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")

const STATS = {
  since: '2026-10-21T11:00:00.000Z',
  last_activity_at: '2026-10-21T11:59:20.000Z',
  status: 'working',
  turns: 3,
  tool_calls: 40,
  tool_failures: 2,
  errors: 0,
  edits: 4,
  files_touched: 3,
  tests: { runs: 3, passed: 10, failed: 2, last: 'fail', last_failing: ['test_autoscale'] },
  repeated_call: { name: 'exec_command', count: 4 },
  context_tokens: 90_000,
  max_context: 100_000,
  total_tokens: 200_000,
  agents: { running: 2, total: 3 },
  anomalies: [{ kind: 'loop', since: '2026-10-21T11:55:00.000Z', detail: 'exec_command ×5 in the last 20 calls' }],
  reviews: { wakes_this_hour: 1, last_wake_at: null },
  summary_line: '4 edits · tests failing (test_autoscale) · 2 agents · active 40 s ago',
}

function row(patch: Record<string, unknown> = {}): WatchRow {
  const parsed = parseWatchRow({
    workspace_id: 'w1', task_id: 't1', title: 'Fix tests', workspace_label: 'Mac · app', enabled: true, active: true, review_interval_minutes: 15,
    expires_at: '2099-01-01T00:00:00.000Z', enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-21T11:00:00.000Z', stats: STATS, ...patch,
  })
  if (!parsed) throw new Error('fixture row is malformed')
  return parsed
}

function withWatch(patch: Partial<WatchState>, base: HubState = withSupervisorConfig(liveState(), { status: 'configured', canEdit: true })): HubState {
  return { ...base, watch: { phase: 'ready', canEdit: true, rows: [], unsupported: false, ...patch } }
}

function sheet(state: HubState): string {
  return plain(renderToStaticMarkup(<ServicesProvider services={makeServices(state, '/xerxes')}><WatchSheet onClose={() => undefined} /></ServicesProvider>))
}

describe('Xerxes tab Watch cards', () => {
  it('shows a compact live card with status, summary, counters, anomaly chip and review interval', () => {
    const html = plain(render(withWatch({ rows: [row()] }), '/xerxes'))
    expect(html).toContain('aria-label="Watching"')
    expect(html).toContain('Fix tests')
    expect(html).toContain('Mac · app')
    expect(html).toContain('working')
    expect(html).toContain('40 s ago')
    expect(html).toContain('4 edits · tests failing (test_autoscale) · 2 agents · active 40 s ago')
    expect(html).toContain('>4 edits<')
    expect(html).toContain('>40 tool calls<')
    expect(html).toContain('tests: 3 runs, 10 passed, 2 failed')
    expect(html).toContain('2/3 agents running')
    expect(html).toContain('context 90%')
    expect(html).toContain('Failing: test_autoscale')
    expect(html).toContain('chip chip-anomaly')
    expect(html).toContain('data-kind="loop"')
    expect(html).toContain('exec_command ×5 in the last 20 calls')
    expect(html).toContain('card card-attention')
    expect(html).toContain('Review every 15 min')
    expect(html).toContain('Watching · 1 task<')
  })

  it('has a Watch button, and "(view)" for a non-owner', () => {
    expect(plain(render(withWatch({ rows: [row()] }), '/xerxes'))).toContain('>Watch<')
    expect(plain(render(withWatch({ canEdit: false, rows: [row()] }), '/xerxes'))).toContain('Watch (view)')
  })

  it('renders text from a run as text, never as HTML', () => {
    const html = plain(render(withWatch({
      rows: [row({
        title: '<img src=x onerror=alert(1)>',
        stats: { ...STATS, summary_line: '<script>alert(2)</script>', tests: { ...STATS.tests, last_failing: ['<i>t</i>'] }, anomalies: [{ kind: 'stall', since: STATS.since, detail: '<u>stuck</u>' }] },
      })],
    }), '/xerxes'))
    expect(html).not.toContain('<img')
    expect(html).not.toContain('<script')
    expect(html).not.toContain('<i>t')
    expect(html).not.toContain('<u>stuck')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
    expect(html).toContain('&lt;script&gt;alert(2)&lt;/script&gt;')
  })

  it('shows nothing for a disabled row, and nothing at all on a Hub without Watch', () => {
    const off = plain(render(withWatch({ rows: [row({ enabled: false, active: false })] }), '/xerxes'))
    expect(off).not.toContain('aria-label="Watching"')
    expect(off).not.toContain('Watching ·')
    const old = plain(render(withWatch({ unsupported: true, rows: [row()] }), '/xerxes'))
    expect(old).not.toContain('aria-label="Watching"')
    expect(old).not.toContain('>Watch<')
    expect(old).not.toContain('Watching ·')
  })

  it('says a card is waiting when the Hub has no stats yet', () => {
    expect(plain(render(withWatch({ rows: [row({ stats: null })] }), '/xerxes'))).toContain('Waiting for the first step of the run')
  })

  it('offers the Autopilot thread to the owner when only a watch is active', () => {
    const owner: HubState = { ...withWatch({ rows: [row()] }), selfUserId: 'u1' }
    const html = plain(render({ ...owner, autopilot: { phase: 'ready', canEdit: true, rows: [], unsupported: false } }, '/xerxes'))
    expect(html).toContain('Autopilot thread')
    const viewer = plain(render({ ...owner, autopilot: { phase: 'ready', canEdit: false, rows: [], unsupported: false } }, '/xerxes'))
    expect(viewer).not.toContain('Autopilot thread')
  })
})

describe('Watch sheet', () => {
  const tasks = [summary({ id: 't1', title: 'Fix tests' }), summary({ id: 't2', title: 'Write docs' })]
  const base = withSupervisorConfig(liveState(tasks), { status: 'configured', canEdit: true })

  it('lets the owner tune each watch and start another from the known tasks', () => {
    const html = sheet(withWatch({ rows: [row()] }, base))
    expect(html).toContain('role="dialog"')
    expect(html).toContain(WATCH_EXPLANATION)
    expect(html).not.toContain(WATCH_READ_ONLY_NOTE)
    for (const label of ['Off', 'Every 5 min', 'Every 15 min', 'Every 30 min', 'Every 60 min']) expect(html).toContain(`>${label}<`)
    for (const label of ['Keep current expiry', 'No expiry', '1 hour', '8 hours', '24 hours', '7 days']) expect(html).toContain(`>${label}<`)
    expect(html).toContain('>Save<')
    expect(html).toContain('>Stop watching<')
    expect(html).toContain('>Start watching<')
    // The watched task is not offered again.
    expect(html).toContain('Write docs — Mac')
    const pick = /<form[^>]*aria-label="Watch another task"[\s\S]*?<\/form>/.exec(html)?.[0] ?? ''
    expect(pick).not.toContain('Fix tests')
    expect(watchKey('w1', 't2')).toBe('w1\u001ft2')
  })

  it('is read-only for a non-owner', () => {
    const html = sheet(withWatch({ canEdit: false, rows: [row()] }, base))
    expect(html).toContain(WATCH_READ_ONLY_NOTE)
    expect(html).toContain('Turned on by Ann.')
    expect(html).not.toContain('<select')
    expect(html).not.toContain('>Save<')
    expect(html).not.toContain('>Stop watching<')
    expect(html).not.toContain('Start watching')
  })

  it('disables Save while offline and while that task saves', () => {
    const offline = sheet({ ...withWatch({ rows: [row()] }, base), connection: { state: 'reconnecting' } as HubState['connection'] })
    expect(offline).toMatch(/<button type="submit"[^>]*disabled/)
    const saving = sheet(withWatch({ rows: [row()], savingKey: watchKey('w1', 't1') }, base))
    expect(saving).toContain('Saving…')
    expect(saving).toMatch(/<button type="submit"[^>]*disabled/)
  })

  it('does not offer Keep for an expiry that has passed', () => {
    const html = sheet(withWatch({ rows: [row({ expires_at: '2020-01-01T00:00:00.000Z', active: false })] }, base))
    expect(html).not.toContain('>Keep current expiry<')
    expect(html).toContain('The previous expiry has passed')
  })

  it('states loading, a retryable error, a save error and an empty list', () => {
    expect(sheet(withWatch({ phase: 'loading' }, base))).toContain('Reading Watch settings')
    const error = sheet(withWatch({ phase: 'error', error: 'timed out' }, base))
    expect(error).toContain('Could not read Watch: timed out')
    expect(error).toContain('Try again')
    expect(sheet(withWatch({ rows: [row()], saveError: 'At most 50 tasks can be watched.' }, base))).toContain('At most 50 tasks can be watched.')
    expect(sheet(withWatch({ rows: [] }, base))).toContain('No task is being watched.')
    expect(sheet(withWatch({ rows: [] }, withSupervisorConfig(liveState([]), { status: 'configured', canEdit: true })))).toContain('No task to watch yet.')
  })
})
