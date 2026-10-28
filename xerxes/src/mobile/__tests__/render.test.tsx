// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { reduceHub, type HubState } from '../../client/hubStore.js'
import type { VoiceControl } from '../../client/voiceControl.js'
import { ApprovalPanel, approvalStep } from '../components/ApprovalSheet.js'
import { PairView } from '../components/PairPage.js'
import { TalkButton } from '../components/TalkButton.js'
import { parseRoute, routeHref } from '../routes.js'
import { hubEvent, liveState, minutesAgo, summary, workspace } from './helpers.js'
import { render } from './renderHelpers.js'

/** Adjacent text nodes are separated by comment markers in server markup. */
const plain = (html: string): string => html.replaceAll('<!-- -->', '')

function taskState(events: ReturnType<typeof hubEvent>[], extra: (state: HubState) => HubState = state => state): HubState {
  let state = liveState([summary({ id: 's1', title: 'Fix bug', status: 'working' })])
  state = reduceHub(state, { type: 'select', workspaceId: 'w1', sessionId: 's1' })
  state = reduceHub(state, { type: 'events', events })
  state = reduceHub(state, { type: 'sessionReady', workspaceId: 'w1', sessionId: 's1' })
  return extra(state)
}

describe('app shell', () => {
  it('has landmarks, a labelled main navigation with four tabs and a polite status region', () => {
    const html = plain(render(liveState(), '/'))
    expect(html).toContain('role="banner"')
    expect(html).toContain('<main')
    expect(html).toContain('<nav aria-label="Main"')
    for (const tab of ['Inbox', 'Task', 'Xerxes', 'Usage']) expect(html).toContain(`>${tab}`)
    expect(html).toContain('role="status" aria-live="polite"')
    expect(html).toContain('aria-current="page"')
  })

  it('announces a reconnecting link and a revoked one without reconnect promises', () => {
    const reconnecting = plain(render({ ...liveState(), connection: { state: 'reconnecting', attempt: 2 } }, '/'))
    expect(reconnecting).toContain('Reconnecting — status may be outdated')
    const revoked = plain(render({ ...liveState(), connection: { state: 'revoked', reason: 'Signed out. Pair this device again.' } }, '/'))
    expect(revoked).toContain('Signed out')
    expect(revoked).toContain('will not reconnect on its own')
  })

  it('parses deep links and builds them back', () => {
    expect(parseRoute('/w/w1/s/s%201')).toEqual({ name: 'task', workspaceId: 'w1', sessionId: 's 1' })
    expect(routeHref({ name: 'task', workspaceId: 'w1', sessionId: 's 1' })).toBe('/w/w1/s/s%201')
    expect(parseRoute('/w/only')).toEqual({ name: 'inbox' })
    expect(parseRoute('/w/%E0%A4%A/s/x')).toEqual({ name: 'inbox' })
    expect(parseRoute('/pair')).toEqual({ name: 'pair' })
    expect(parseRoute('/nope')).toEqual({ name: 'inbox' })
  })
})

describe('Inbox tab', () => {
  it('shows workspace, title, status text, badge, and marks offline workspaces unavailable', () => {
    const state = liveState(
      [
        summary({ id: 'a', title: 'Needs approval', status: 'needs_you', pendingCount: 1, updatedAt: minutesAgo(3) }),
        summary({ id: 'b', title: 'Remote job', workspaceId: 'w2', status: 'working' }),
      ],
      [workspace('w1', 'Mac'), workspace('w2', 'Build server', 'offline')],
    )
    const html = plain(render(state, '/'))
    expect(html).toContain('Mac')
    expect(html).toContain('Needs approval')
    expect(html).toContain('Needs you')
    expect(html).toContain('3m ago')
    expect(html).toContain('1 task needs you')
    expect(html).toContain('Build server is unavailable')
    expect(html).toContain('Unavailable. This workspace is offline')
    // An unavailable task is not a link and is never shown as working or idle.
    expect(html).not.toContain('href="/w/w2/s/b"')
    expect(html).toContain('href="/w/w1/s/a"')
  })

  it('has filters and a workspace selector only when there is more than one workspace', () => {
    const one = plain(render(liveState(), '/'))
    expect(one).not.toContain('All workspaces')
    const two = plain(render(liveState([], [workspace('w1', 'Mac'), workspace('w2', 'Server')]), '/'))
    expect(two).toContain('All workspaces')
    expect(two).toContain('aria-pressed="true"')
  })

  it('says so when there is nothing to show, and shows load errors with a retry', () => {
    expect(plain(render(liveState(), '/'))).toContain('No tasks yet')
    const failed = plain(render({ ...liveState(), inboxError: 'boom' }, '/'))
    expect(failed).toContain('Could not load tasks: boom')
    expect(failed).toContain('Try again')
  })
})

describe('Task tab', () => {
  it('shows the tail with collapsible thinking and tools, steer and stop while running', () => {
    const state = taskState([
      hubEvent('turn_begin', { user_input: 'fix it' }),
      hubEvent('think_part', { think: 'hmm' }),
      hubEvent('tool_call', { id: 't1', name: 'exec_command', arguments: '{"command":"bun test"}' }),
    ])
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('Fix bug')
    expect(html).toContain('<details')
    expect(html).toContain('Running a command')
    expect(html).toContain('>Steer<')
    expect(html).toContain('>Stop<')
    expect(html).toContain('Camera')
    expect(html).toContain('Photos')
  })

  it('offers Send, not Steer or Stop, for an idle task', () => {
    const state = taskState([hubEvent('turn_begin', { user_input: 'hi' }), hubEvent('turn_end', {})])
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('>Send<')
    expect(html).not.toContain('>Stop<')
  })

  it('disables sending while the link is down and keeps the page readable', () => {
    const state = taskState([hubEvent('text_part', { text: 'partial answer' })], s => ({ ...s, connection: { state: 'reconnecting', attempt: 1 } }))
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('partial answer')
    expect(html).toContain('may be outdated')
    expect(html).toMatch(/<button type="submit" class="btn btn-primary" disabled/)
  })

  it('lists a pending approval, a failed decision that stays visible, and who resolved another', () => {
    const state = taskState(
      [hubEvent('approval_request', { id: 'r1', tool_name: 'exec_command', arguments: { command: 'rm -rf build' } })],
      s => reduceHub(s, { type: 'decision', interactionId: 'r1', state: { phase: 'failed', decision: 'once', message: 'Hub unavailable' } }),
    )
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('Review and decide')
    expect(html).toContain('Not applied: Hub unavailable')
    expect(html).toContain('Needs you')

    const resolved = taskState([], s => reduceHub(s, { type: 'decision', interactionId: 'r9', state: { phase: 'resolved', decision: 'once', already: true, by: 'Bob' } }))
    expect(plain(render(resolved, '/w/w1/s/s1'))).toContain('Already resolved by Bob.')
  })

  it('shows requested and effective model apart on the model button state', () => {
    const state = taskState([hubEvent('status_update', { model: 'model-a' })], s => reduceHub(s, { type: 'config', field: 'model', value: { requested: 'model-b', phase: 'requested' } }))
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('Model: model-a')
    expect(html).not.toContain('Model: model-b')
  })

  it('asks for a choice when no task is open', () => {
    expect(plain(render(liveState(), '/task'))).toContain('Choose a task from the Inbox')
  })
})

describe('approval sheet', () => {
  const item = {
    id: 'r1', kind: 'permission' as const, toolName: 'exec_command', title: 'Run this command?', description: '',
    command: 'rm -rf build', inputs: [['cwd', '/proj'] as const], workspace: '/proj', detailsHidden: false, questions: [], requestedBy: 'XERXES for Ann',
  }
  const noop = (): void => {}

  it('shows the real command, inputs, workspace and requester, with Allow once and Deny before session-wide', () => {
    const html = plain(renderToStaticMarkup(<ApprovalPanel item={item} canRespond onDecide={noop} onAnswer={noop} />))
    expect(html).toContain('rm -rf build')
    expect(html).toContain('/proj')
    expect(html).toContain('requested by XERXES for Ann')
    const once = html.indexOf('Allow once')
    const deny = html.indexOf('>Deny<')
    const session = html.indexOf('Allow for this session')
    expect(once).toBeGreaterThan(-1)
    expect(deny).toBeGreaterThan(once)
    expect(session).toBeGreaterThan(deny)
  })

  it('needs a second confirmation for a session-wide approval', () => {
    expect(approvalStep('choose', 'session')).toEqual({ step: 'confirm-session' })
    expect(approvalStep('choose', 'once')).toEqual({ step: 'choose', decide: 'once' })
    expect(approvalStep('choose', 'deny')).toEqual({ step: 'choose', decide: 'deny' })
    // The confirm intent does nothing from the first step.
    expect(approvalStep('choose', 'confirm-session')).toEqual({ step: 'choose' })
    expect(approvalStep('confirm-session', 'confirm-session')).toEqual({ step: 'choose', decide: 'session' })
    expect(approvalStep('confirm-session', 'back')).toEqual({ step: 'choose' })
    const html = plain(renderToStaticMarkup(<ApprovalPanel item={item} canRespond onDecide={noop} onAnswer={noop} initialStep="confirm-session" />))
    expect(html).toContain('Yes, allow for this session')
    expect(html).toContain('Go back')
    expect(html).not.toContain('>Allow once<')
  })

  it('shows a failed decision and keeps the buttons usable, and locks after it is resolved', () => {
    const failed = plain(renderToStaticMarkup(<ApprovalPanel item={item} canRespond decision={{ phase: 'failed', decision: 'once', message: 'Hub unavailable' }} onDecide={noop} onAnswer={noop} />))
    expect(failed).toContain('Not applied: Hub unavailable')
    expect(failed).not.toMatch(/<button[^>]*disabled[^>]*>Allow once/)
    const resolved = plain(renderToStaticMarkup(<ApprovalPanel item={item} canRespond decision={{ phase: 'resolved', decision: 'once', already: true, by: 'Bob' }} onDecide={noop} onAnswer={noop} />))
    expect(resolved).toContain('Already resolved by Bob.')
    expect(resolved).toMatch(/<button[^>]*disabled[^>]*>Allow once/)
  })

  it('says when the Hub withheld the details and disables deciding offline', () => {
    const hidden = plain(renderToStaticMarkup(<ApprovalPanel item={{ ...item, command: undefined as unknown as string, inputs: [], detailsHidden: true }} canRespond={false} disabledReason="Decisions need a live connection." onDecide={noop} onAnswer={noop} />))
    expect(hidden).toContain('hidden for your role')
    expect(hidden).toContain('Decisions need a live connection.')
    expect(hidden).toMatch(/<button[^>]*disabled[^>]*>Deny/)
  })
})

describe('Xerxes tab', () => {
  it('says plainly that the supervisor is not available on this hub', () => {
    const state = { ...liveState(), supervisor: { phase: 'unsupported' as const, messages: [], confirmations: [], busy: false } }
    const html = plain(render(state, '/xerxes'))
    expect(html).toContain('Supervisor not available on this hub')
    expect(html).not.toContain('Ask XERXES…')
  })

  it('renders the thread with confirmation cards that need Confirm or Cancel', () => {
    const state = {
      ...liveState(),
      supervisor: {
        phase: 'ready' as const, busy: false,
        messages: [{ id: 'm1', role: 'xerxes' as const, text: 'Task B is blocked.', at: minutesAgo(1), source: 'text' as const }],
        confirmations: [
          { id: 'c1', tool: 'task.stop', summary: 'Stop task B', targets: ['Mac / Task B'], requestedBy: 'Ann', state: 'pending' as const },
          { id: 'c2', tool: 'task.steer', summary: 'Steer task C', targets: [], state: 'confirmed' as const },
        ],
      },
    }
    const html = plain(render(state, '/xerxes'))
    expect(html).toContain('Task B is blocked.')
    expect(html).toContain('XERXES proposes: task.stop')
    expect(html).toContain('>Confirm<')
    expect(html).toContain('>Cancel<')
    expect(html).toContain('Nothing runs until you confirm')
    expect(html).toContain('Confirmed.')
  })

  it('hides the talk button without a voice control and shows it with one', () => {
    const ready = { ...liveState(), supervisor: { phase: 'ready' as const, messages: [], confirmations: [], busy: false } }
    expect(plain(render(ready, '/xerxes'))).not.toContain('Hold to talk')
    const voice: VoiceControl = { start: async () => {}, stop: async () => 'hello', cancel: () => {} }
    expect(plain(render(ready, '/xerxes', { voice }))).toContain('Hold to talk')
    expect(renderToStaticMarkup(<TalkButton voice={undefined} onTranscript={() => {}} />)).toBe('')
  })
})

describe('Usage tab', () => {
  it('labels plan-paid usage "Included in plan" and unpriced usage "price unknown", never an invented $0', () => {
    const base = { calls: 1, input: 1000, output: 10, cacheRead: 0, cacheWrite: 0 }
    const state: HubState = {
      ...liveState(),
      usage: {
        phase: 'ready',
        report: {
          fetchedAt: 1,
          profiles: [{
            profile: 'p', label: 'Claude Max', provider: 'anthropic', model: 'x', active: true, source: 'claude', status: 'ok', plan: 'Max',
            windows: [{ label: '5-hour', usedPercent: 42.4 }, { label: 'weekly', usedPercent: 10 }],
          }],
          models: [
            { ...base, model: 'plan-model', costUsd: 3, costBasis: 'subscription' },
            { ...base, model: 'free-model' },
            { ...base, model: 'paid-model', costUsd: 1.5 },
          ],
        },
        extras: { inputTokens: 3000, outputTokens: 30, billableCostUsd: 1.5, subscriptionLabel: 'Included in plan', unavailable: ['w2'] },
      },
    }
    const html = plain(render(state, '/usage'))
    expect(html).toContain('Included in plan')
    expect(html).toContain('price unknown')
    expect(html).toContain('$1.50')
    expect(html).not.toContain('$3.00')
    expect(html).toContain('42% used')
    // A workspace this phone cannot see is counted, never shown as a raw id.
    expect(html).toContain('Usage is not available yet for 1 other workspace.')
    expect(html).not.toContain('w2')
    expect(html).toContain('listed once')
  })

  it('names unavailable workspaces by label and repeats the profile only for duplicate model rows', () => {
    const base = { calls: 1, input: 1000, output: 10, cacheRead: 0, cacheWrite: 0 }
    const state: HubState = {
      ...liveState([], [workspace('w1', 'Mac'), workspace('m-d12a2c7d-3e7b31b91354', 'TPU box')]),
      usage: {
        phase: 'ready',
        report: {
          fetchedAt: 1,
          profiles: [],
          models: [
            { ...base, model: 'claude-code/opus', profile: 'cc' },
            { ...base, model: 'claude-code/opus', profile: 'cc-work' },
            { ...base, model: 'gpt-6.1-sol' },
          ],
        },
        extras: { inputTokens: 3000, outputTokens: 30, unavailable: ['m-d12a2c7d-3e7b31b91354'] },
      },
    }
    const html = plain(render(state, '/usage'))
    expect(html).toContain('Usage is not available yet for TPU box.')
    expect(html).not.toContain('m-d12a2c7d')
    expect(html).toContain('Profile: cc<')
    expect(html).toContain('Profile: cc-work<')
  })

  it('shows an error with the last result kept', () => {
    const state: HubState = { ...liveState(), usage: { phase: 'error', message: 'offline', report: { fetchedAt: 1, profiles: [], models: [{ model: 'm', calls: 1, input: 1, output: 1, cacheRead: 0, cacheWrite: 0 }] } } }
    const html = plain(render(state, '/usage'))
    expect(html).toContain('showing the last result')
    expect(html).toContain('price unknown')
  })
})

describe('pair view', () => {
  const retry = (): void => {}
  it('shows the comparison code and waits for confirmation on the desktop', () => {
    const html = plain(renderToStaticMarkup(<PairView state={{ phase: 'waiting', comparison: '4829' }} onRetry={retry} />))
    expect(html).toContain('Waiting for confirmation on the desktop')
    expect(html).toContain('4829')
    expect(html).toContain('Comparison code')
  })

  it('states each failure without leaking a secret', () => {
    expect(plain(renderToStaticMarkup(<PairView state={{ phase: 'missing' }} onRetry={retry} />))).toContain('needs a pairing link')
    expect(plain(renderToStaticMarkup(<PairView state={{ phase: 'invalid' }} onRetry={retry} />))).toContain('incomplete')
    expect(plain(renderToStaticMarkup(<PairView state={{ phase: 'rejected', message: 'Code expired' }} onRetry={retry} />))).toContain('Code expired')
    expect(plain(renderToStaticMarkup(<PairView state={{ phase: 'unavailable', message: 'Hub down' }} onRetry={retry} />))).toContain('Try again')
    expect(plain(renderToStaticMarkup(<PairView state={{ phase: 'timeout' }} onRetry={retry} />))).toContain('did not confirm in time')
  })
})
