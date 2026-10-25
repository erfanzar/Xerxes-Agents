// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { parseAutopilotRow, parseWatchRow } from '../src/client/hubParsers.js'
import {
  autopilotThreadAvailable, initialHubState, initialSupervisorConfig, watchKey, type AutopilotState, type HubState, type WatchState,
} from '../src/client/hubStore.js'
import type { TaskSummary, WatchRow } from '../src/client/hubTypes.js'
import { NO_ACTIVITY } from '../src/client/taskTypes.js'
import { WATCH_EXPLANATION, WATCH_READ_ONLY_NOTE } from '../src/client/watch.js'
import { SupervisorView, type SupervisorViewProps } from '../src/desktop/renderer/SupervisorPage.js'

const noop = (): void => {}
const plain = (html: string): string => html.replaceAll('<!-- -->', '')

const STATS = {
  since: '2026-10-24T09:00:00.000Z',
  last_activity_at: '2026-10-24T10:00:00.000Z',
  status: 'working',
  turns: 3,
  tool_calls: 40,
  tool_failures: 2,
  errors: 0,
  edits: 4,
  files_touched: 3,
  tests: { runs: 3, passed: 10, failed: 2, last: 'fail', last_failing: ['test_autoscale', 'test_scale_down'] },
  repeated_call: { name: 'exec_command', count: 4 },
  context_tokens: 90_000,
  max_context: 100_000,
  total_tokens: 200_000,
  agents: { running: 2, total: 3 },
  anomalies: [{ kind: 'loop', since: '2026-10-24T09:55:00.000Z', detail: 'exec_command ×5 in the last 20 calls' }],
  reviews: { wakes_this_hour: 1, last_wake_at: null },
  summary_line: '4 edits · tests failing (test_autoscale) · 2 agents · active 40 s ago',
}

function watchRow(patch: Record<string, unknown> = {}): WatchRow {
  const parsed = parseWatchRow({
    workspace_id: 'w1', task_id: 't1', title: 'Fix tests', workspace_label: 'One · app', enabled: true, active: true, review_interval_minutes: 15,
    expires_at: '2099-01-01T00:00:00.000Z', enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', stats: STATS, ...patch,
  })
  if (!parsed) throw new Error('fixture row is malformed')
  return parsed
}

function task(id: string, title: string, workspaceId = 'w1'): TaskSummary {
  return {
    workspaceId, id, title, status: 'working', model: '', goalSummary: '', pendingCount: 0, interactionIds: [], agents: { total: 0, active: 0 },
    updatedAt: '2026-10-24T10:00:00.000Z', stale: false, interrupted: false, activity: NO_ACTIVITY,
  }
}

function hub(opts: { canEdit?: boolean; rows?: WatchRow[]; unsupported?: boolean; tasks?: TaskSummary[]; watch?: Partial<WatchState>; autopilotRows?: boolean } = {}): HubState {
  const canEdit = opts.canEdit ?? true
  const autopilot: AutopilotState = {
    phase: 'ready', canEdit, unsupported: false,
    rows: opts.autopilotRows === true ? [parseAutopilotRow({ workspace_id: 'w1', label: 'One', enabled: true, instruction: '', expires_at: null, enabled_by: { id: 'u1', name: 'Ann' }, updated_at: null, active: true })!] : [],
  }
  const tasks: Record<string, TaskSummary> = {}
  for (const t of opts.tasks ?? []) tasks[`${t.workspaceId}\u001f${t.id}`] = t
  return {
    ...initialHubState(),
    connection: { state: 'live' },
    selfUserId: 'u1',
    tasks,
    workspaces: { w1: { id: 'w1', label: 'One', health: 'online', models: [], efforts: [] } },
    workspaceOrder: ['w1'],
    supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
    supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', status: 'configured', canEdit: true, config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'model-a', effort: 'high' } },
    autopilot,
    watch: { phase: 'ready', canEdit, rows: opts.rows ?? [], unsupported: opts.unsupported ?? false, ...opts.watch },
  }
}

function view(state: HubState, extra: Partial<SupervisorViewProps> = {}): string {
  return plain(renderToStaticMarkup(createElement(SupervisorView, {
    state, hubLabel: 'hub.example.ts.net', now: Date.parse('2026-10-24T10:01:00.000Z'), onSend: async () => true, onRespond: noop, onReopen: noop,
    onOpenSettings: noop, chooserOpen: false, onChooserChange: noop, onChooserShown: noop, onOptions: noop, onSaveModel: async () => true,
    autopilotOpen: false, onAutopilotChange: noop, onSaveAutopilot: async () => true, onRetryAutopilot: noop,
    autopilotThreadOpen: false, onAutopilotThreadChange: noop, onRespondAutopilot: noop,
    watchOpen: false, onWatchChange: noop, onSaveWatch: async () => true, onRetryWatch: noop,
    ...extra,
  })))
}

test('a watched task shows as a live card: title, workspace, status, summary, counters, review interval', () => {
  const html = view(hub({ rows: [watchRow()] }))
  expect(html).toContain('aria-label="Watching"')
  expect(html).toContain('Fix tests')
  expect(html).toContain('One · app')
  expect(html).toContain('>working<')
  expect(html).toContain('last activity')
  expect(html).toContain('1 min ago')
  expect(html).toContain('4 edits · tests failing (test_autoscale) · 2 agents · active 40 s ago')
  expect(html).toContain('>4 edits<')
  expect(html).toContain('>40 tool calls<')
  expect(html).toContain('>2 failed<')
  expect(html).toContain('tests: 3 runs, 10 passed, 2 failed')
  expect(html).toContain('2/3 agents running')
  expect(html).toContain('context 90%')
  expect(html).toContain('repeated exec_command ×4')
  expect(html).toContain('Failing: test_autoscale, test_scale_down')
  expect(html).toContain('Review every 15 min')
  expect(html).toContain('Watching · 1 task<')
})

test('anomalies are highlighted chips with their kind and detail', () => {
  const html = view(hub({ rows: [watchRow()] }))
  expect(html).toContain('has-anomaly')
  expect(html).toMatch(/class="watchcard__chip is-anomaly" data-kind="loop"/)
  expect(html).toContain('Looping')
  expect(html).toContain('exec_command ×5 in the last 20 calls')
  const calm = view(hub({ rows: [watchRow({ stats: { ...STATS, anomalies: [] } })] }))
  expect(calm).not.toContain('has-anomaly')
  expect(calm).not.toContain('is-anomaly')
})

test('a watch with no stats yet says it is waiting', () => {
  const html = view(hub({ rows: [watchRow({ stats: null })] }))
  expect(html).toContain('Waiting for the first step of the run')
})

test('text that came from a run is rendered as text, never as HTML', () => {
  const evil = '<img src=x onerror=alert(1)>'
  const html = view(hub({
    rows: [watchRow({
      title: evil, workspace_label: '<b>ws</b>',
      stats: {
        ...STATS, summary_line: '<script>alert(2)</script>',
        tests: { ...STATS.tests, last_failing: ['<i>test</i>'] },
        anomalies: [{ kind: 'stall', since: STATS.since, detail: '<u>stuck</u>' }],
      },
    })],
  }))
  expect(html).not.toContain('<img')
  expect(html).not.toContain('<script')
  expect(html).not.toContain('<b>ws')
  expect(html).not.toContain('<i>test')
  expect(html).not.toContain('<u>stuck')
  expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
  expect(html).toContain('&lt;script&gt;alert(2)&lt;/script&gt;')
  expect(html).toContain('&lt;i&gt;test&lt;/i&gt;')
  expect(html).toContain('&lt;u&gt;stuck&lt;/u&gt;')
})

test('a disabled row is not a card, and nothing shows while no watch is on', () => {
  const html = view(hub({ rows: [watchRow({ enabled: false, active: false })] }))
  expect(html).not.toContain('aria-label="Watching"')
  expect(html).not.toContain('Watching ·')
  expect(view(hub())).not.toContain('class="watchstrip"')
})

test('an enabled watch that is no longer active reads as Off, not Watching', () => {
  const html = view(hub({ canEdit: false, rows: [watchRow({ active: false })] }), { watchOpen: true })
  expect(html).toContain('Off (expired or the owner changed)')
  expect(html).not.toContain('Watching · ')
})

test('the owner gets a Watch button and a panel with review, expiry, save, stop and a task picker', () => {
  const state = hub({ rows: [watchRow()], tasks: [task('t1', 'Fix tests'), task('t2', 'Write docs'), task('t3', 'Refactor')] })
  const closed = view(state)
  expect(closed).toMatch(/<button[^>]*aria-expanded="false"[^>]*>Watch</)
  const html = view(state, { watchOpen: true })
  expect(html).toContain('aria-label="Watch"')
  expect(html).toContain(WATCH_EXPLANATION)
  expect(html).toContain('no model cost')
  expect(html).not.toContain(WATCH_READ_ONLY_NOTE)
  for (const label of ['Off', 'Every 5 min', 'Every 15 min', 'Every 30 min', 'Every 60 min']) expect(html).toContain(`>${label}<`)
  for (const label of ['Keep current expiry', 'No expiry', '1 hour', '8 hours', '24 hours', '7 days']) expect(html).toContain(`>${label}<`)
  expect(html).toContain('>Save<')
  expect(html).toContain('>Stop watching<')
  expect(html).toContain('>Start watching<')
  // Already watched tasks are not offered again.
  const picker = /<select id="watch-pick"[\s\S]*?<\/select>/.exec(html)?.[0] ?? ''
  expect(picker).toContain('Write docs — One')
  expect(picker).toContain('Refactor — One')
  expect(picker).not.toContain('Fix tests')
  expect(watchKey('w1', 't2')).toBe('w1\u001ft2')
})

test('a non-owner sees the cards and the rows read-only, with no inputs, no Save, no picker', () => {
  const html = view(hub({ canEdit: false, rows: [watchRow()], tasks: [task('t2', 'Write docs')] }), { watchOpen: true })
  expect(html).toContain(WATCH_READ_ONLY_NOTE)
  expect(html).toContain('aria-label="Watching"')
  expect(html).toContain('Turned on by Ann.')
  const panel = /<section class="autopilot watchpanel"[\s\S]*?<\/section>/.exec(html)?.[0] ?? ''
  expect(panel).toContain('Turned on by Ann.')
  expect(panel).not.toContain('<select')
  expect(panel).not.toContain('<form')
  expect(panel).not.toContain('>Save<')
  expect(panel).not.toContain('>Stop watching<')
  expect(panel).not.toContain('Start watching')
})

test('Save is disabled while offline or while that task saves', () => {
  const offline = view({ ...hub({ rows: [watchRow()] }), connection: { state: 'reconnecting', attempt: 1 } as HubState['connection'] }, { watchOpen: true })
  expect(offline).toMatch(/<button[^>]*disabled=""[^>]*>Save</)
  const saving = view(hub({ rows: [watchRow()], watch: { savingKey: watchKey('w1', 't1') } }), { watchOpen: true })
  expect(saving).toMatch(/<button[^>]*disabled=""[^>]*>Saving…</)
})

test('an expiry that has passed is not offered as Keep and says so', () => {
  const html = view(hub({ rows: [watchRow({ expires_at: '2020-01-01T00:00:00.000Z', active: false })] }), { watchOpen: true })
  expect(html).not.toContain('>Keep current expiry<')
  expect(html).toContain('The previous expiry has passed')
})

test('loading, a retryable read error, a save error and an empty list are all stated', () => {
  expect(view(hub({ watch: { phase: 'loading' } }), { watchOpen: true })).toContain('Reading Watch settings')
  const error = view(hub({ watch: { phase: 'error', error: 'timed out' } }), { watchOpen: true })
  expect(error).toContain('Could not read Watch: timed out')
  expect(error).toContain('>Try again<')
  expect(view(hub({ rows: [watchRow()], watch: { saveError: 'At most 50 tasks can be watched.' } }), { watchOpen: true })).toContain('At most 50 tasks can be watched.')
  expect(view(hub(), { watchOpen: true })).toContain('No task is being watched.')
  expect(view(hub({ tasks: [] }), { watchOpen: true })).toContain('No task to watch yet.')
})

test('a Hub without Watch has no Watch control, strip or chip', () => {
  const html = view(hub({ unsupported: true, rows: [watchRow()] }), { watchOpen: true })
  expect(html).not.toMatch(/>Watch</)
  expect(html).not.toContain('aria-label="Watch"')
  expect(html).not.toContain('aria-label="Watching"')
  expect(html).not.toContain('Watching ·')
})

test('the owner gets the Autopilot thread switch when only a watch is active', () => {
  const onlyWatch = hub({ rows: [watchRow()] })
  expect(autopilotThreadAvailable(onlyWatch)).toBe(true)
  expect(view(onlyWatch)).toContain('Autopilot thread')
  // Neither Autopilot nor a watch is active: no switch.
  expect(autopilotThreadAvailable(hub({ rows: [watchRow({ active: false })] }))).toBe(false)
  expect(view(hub({ rows: [watchRow({ active: false })] }))).not.toContain('Autopilot thread')
  // A non-owner never gets it.
  expect(autopilotThreadAvailable(hub({ canEdit: false, rows: [watchRow()] }))).toBe(false)
  expect(view(hub({ canEdit: false, rows: [watchRow()] }))).not.toContain('Autopilot thread')
})
