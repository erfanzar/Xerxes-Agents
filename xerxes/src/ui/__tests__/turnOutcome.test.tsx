// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it } from 'vitest'
import { adaptDaemonEvent, transcriptFromStoredMessages } from '../gatewayAdapter.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { MessageLine } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import { TURN_STOP_REASONS, turnOutcomeLabel } from '../../types/turnOutcome.js'

it('retains outcomes on user, assistant and tool boundaries without fabricating assistant text', () => {
  const rows = transcriptFromStoredMessages([
    { role: 'user', content: 'No output', turn_outcome: { version: 1, reason: 'provider_failed' } },
    { role: 'user', content: 'Tool only' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'c', function: { name: 'read_file', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c', content: 'file contents', turn_outcome: { version: 1, reason: 'aborted' } },
    { role: 'user', content: 'Success' },
    { role: 'assistant', content: 'Actual answer', turn_outcome: { version: 1, reason: 'completed' } },
  ])
  const messages = toTranscriptMessages(rows)
  expect(messages.filter(m => m.kind === 'outcome').map(m => m.outcome)).toEqual(['provider_failed', 'aborted', 'completed'])
  expect(messages.filter(m => m.role === 'assistant' && !m.kind).map(m => m.text)).toEqual(['Actual answer'])
  expect(messages.findIndex(m => m.kind === 'trail')).toBe(messages.findIndex(m => m.outcome === 'aborted') - 1)
  expect(toTranscriptMessages(transcriptFromStoredMessages([{ role: 'tool', turn_outcome: { version: 1, reason: 'aborted' } }]))).toEqual([{ kind: 'outcome', role: 'assistant', text: '', outcome: 'aborted' }])
})

it('validates replay metadata instead of trusting displayed diagnostic text', () => {
  const frame = { category: 'history', type: 'replay_outcome', body: 'arbitrary provider error', payload: { version: 1, reason: 'aborted' } }
  expect(adaptDaemonEvent('notification', frame)).toEqual([{ type: 'transcript.append', payload: { role: 'assistant', text: 'interrupted', outcome: 'aborted' } }])
  expect(adaptDaemonEvent('notification', { ...frame, payload: { version: 1, reason: 'secret-value' } })).toEqual([])
  expect(transcriptFromStoredMessages([{ role: 'user', content: 'Request', turn_outcome: { version: 1, reason: 'completed', diagnostic: 'secret' } }])).toEqual([{ role: 'user', text: 'Request' }])
})

for (const width of [28, 80]) it(`renders truthful one-line receipts at ${width} columns`, async () => {
  const setup = await testRender(<box flexDirection="column">
    {TURN_STOP_REASONS.map(outcome => <MessageLine key={outcome} msg={{ kind: 'outcome', role: 'assistant', text: '', outcome }} t={themeForMode(DEFAULT_THEME, 'code')} />)}
  </box>, { width, height: 20 })
  try {
    await setup.flush()
    const frame = setup.captureCharFrame()
    expect(frame).toContain('interrupted')
    expect(frame).toContain('failed')
    expect(frame).toContain('completed')
    expect(frame).not.toContain('done')
    if (width === 80) for (const outcome of TURN_STOP_REASONS) expect(frame).toContain(turnOutcomeLabel(outcome))
  } finally { await act(async () => setup.renderer.destroy()) }
})

it('does not claim success for historical turns with no outcome', async () => {
  const setup = await testRender(<MessageLine msg={{ role: 'assistant', text: 'Old reply' }} rail="end" t={themeForMode(DEFAULT_THEME, 'code')} />, { width: 80, height: 8 })
  try { await setup.flush(); expect(setup.captureCharFrame()).toContain('ended · outcome unknown') }
  finally { await act(async () => setup.renderer.destroy()) }
})
