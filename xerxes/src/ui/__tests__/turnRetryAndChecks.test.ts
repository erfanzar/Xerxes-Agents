// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import type { GatewayEventHandlerContext } from '../app/interfaces.js'
import { turnController } from '../app/turnController.js'
import { getTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import { adaptDaemonEvent, checkNoteEvents } from '../gatewayAdapter.js'
import type { Msg } from '../types.js'

// Drives daemon-shaped events through the real adapter, handler and turn
// controller, the path a live turn takes, and inspects what the transcript gets.
const harness = () => {
  const appended: Msg[] = []
  const ctx: GatewayEventHandlerContext = {
    composer: { setInput: vi.fn() },
    gateway: { gw: {} as GatewayClient, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: vi.fn(),
      recoverSidRef: { current: null },
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, stdout: { isTTY: false } as NodeJS.WriteStream, sys: vi.fn() },
    transcript: { appendMessage: message => appended.push(message), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn(), setVoiceTts: vi.fn() }
  }
  const handler = createGatewayEventHandler(ctx)
  const daemon = (type: string, payload: Record<string, unknown> = {}) => {
    for (const event of adaptDaemonEvent(type, payload)) handler(event)
  }
  return { appended, daemon }
}

const ragged = { code: 'csv_ragged', severity: 'error', path: 'totals.csv', message: "1 row do not have the header's 3 columns: line 3 has 2." }

afterEach(() => {
  turnController.fullReset()
  resetUiState()
})

describe('a provider attempt that fails mid-reply', () => {
  it('drops the partial reply before a retry instead of appending the next attempt to it', () => {
    const { appended, daemon } = harness()
    const partial = 'Looking at the invoices, the first thing I notice is that March has two'
    daemon('turn_begin')
    daemon('text_part', { text: partial })
    daemon('status_update', { kind: 'network_retry', text: 'Retrying connection…', attempt: 1, delay: 1000 })
    expect(turnController.bufRef).toBe('')
    expect(getTurnState().streamSegments.map(segment => segment.text)).toEqual(['Connection lost mid-reply · retry 1'])

    daemon('text_part', { text: 'March has two duplicate invoices.' })
    daemon('turn_end', { stop_reason: 'completed' })

    const texts = appended.map(message => message.text)
    expect(texts).toContain('Connection lost mid-reply · retry 1')
    expect(texts).toContain('March has two duplicate invoices.')
    expect(texts.some(text => text.includes(partial))).toBe(false)
  })

  it('keeps the last partial reply on its own line when the retries give up', () => {
    const { appended, daemon } = harness()
    daemon('turn_begin')
    daemon('text_part', { text: 'Looking at the invoices' })
    daemon('notification', {
      level: 'error',
      message: 'The socket connection was closed unexpectedly.',
      retry: { type: 'provider_retry', attempt: 4, maxAttempts: 0, delay: 0, final: true, error: 'closed' }
    })
    daemon('text_part', { text: '[Error: The model server closed the connection mid-reply.]' })
    daemon('turn_end', { stop_reason: 'provider_failed' })

    const texts = appended.filter(message => message.role === 'assistant' && message.text).map(message => message.text)
    expect(texts).toContain('Looking at the invoices')
    expect(texts).toContain('[Error: The model server closed the connection mid-reply.]')
  })
})

describe('retry notices', () => {
  const retryNotice = (final: boolean) => adaptDaemonEvent('notification', {
    level: final ? 'error' : 'warning',
    message: 'The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()',
    retry: { type: 'provider_retry', attempt: 1, maxAttempts: 3, delay: 0, final, error: 'closed' }
  })

  it('keeps the reason in a mid-retry toast without Bun’s note to developers', () => {
    expect(retryNotice(false)).toEqual([
      { type: 'turn.retry', payload: { final: false, note: 'Model request failed · retry 1 of 2' } },
      { type: 'notification.show', payload: { kind: 'ttl', level: 'warn', text: 'The socket connection was closed unexpectedly.', ttl_ms: 8000 } }
    ])
  })

  it('leaves the final failure to the transcript, which words it for the user', () => {
    expect(retryNotice(true)).toEqual([{ type: 'turn.retry', payload: { final: true, note: '' } }])
  })

  it('names a reply the budget stopped instead of calling it a failed request', () => {
    const message = 'Reply stopped: it reasoned for about 9,000 tokens over 10 min without a tool call or an answer (reply budget 10 min). Asking again with a note (1 of 2).'
    expect(adaptDaemonEvent('notification', {
      level: 'warning',
      message,
      retry: { type: 'provider_retry', cause: 'reply_budget', attempt: 1, maxAttempts: 3, delay: 0, final: false, error: message }
    })).toEqual([
      { type: 'turn.retry', payload: { final: false, note: 'Reply stopped at its budget · asking again 1 of 2' } },
      { type: 'notification.show', payload: { kind: 'ttl', level: 'warn', text: message, ttl_ms: 8000 } }
    ])
  })
})

describe('result checks between cascade rounds', () => {
  it('shows each check and retry in order and ends the turn with the failing count', () => {
    const { appended, daemon } = harness()
    daemon('turn_begin')
    daemon('text_part', { text: 'Done. I wrote totals.csv.' })
    daemon('notification', { level: 'warning', message: 'Result checks found 1 problem', result_checks: [ragged] })
    daemon('notification', { level: 'info', message: 'Escalation: checks failed, retrying locally (1 of 2).', escalation: { action: 'retry_local' } })
    daemon('text_part', { text: 'Fixed: totals.csv now has three columns.' })
    daemon('notification', { level: 'warning', message: 'Result checks found 1 problem', result_checks: [ragged] })
    daemon('notification', {
      level: 'warning',
      message: "Escalation stopped: no improvement after 3 attempts. Result checks found 1 problem:\n- [error] csv_ragged totals.csv: line 3 has 2.",
      escalation: { action: 'stop' }
    })
    daemon('turn_end', { stop_reason: 'completed' })

    expect(appended.filter(message => message.kind !== 'trail').map(message => [message.role, message.text])).toEqual([
      ['assistant', 'Done. I wrote totals.csv.'],
      ['system', "Checked the result: 1 problem. totals.csv: 1 row do not have the header's 3 columns: line 3 has 2."],
      ['system', 'Asked the model to fix it (1 of 2).'],
      ['assistant', 'Fixed: totals.csv now has three columns.'],
      ['system', "Checked the result: 1 problem. totals.csv: 1 row do not have the header's 3 columns: line 3 has 2."],
      ['system', 'Stopped retrying: no improvement after 3 attempts. The problems above are still there.'],
      ['assistant', '']
    ])
    expect(appended.at(-1)).toMatchObject({ kind: 'outcome', outcome: 'completed', checksFailing: 1 })
  })

  it('does not mark the turn when the retry fixed the result', () => {
    const { appended, daemon } = harness()
    daemon('turn_begin')
    daemon('text_part', { text: 'Done.' })
    daemon('notification', { level: 'warning', message: 'Result checks found 1 problem', result_checks: [ragged] })
    daemon('notification', { level: 'info', message: 'Escalation: checks failed, retrying locally (1 of 2).', escalation: { action: 'retry_local' } })
    daemon('text_part', { text: 'Fixed.' })
    // A clean check sends nothing.
    daemon('turn_end', { stop_reason: 'completed' })
    expect(appended.at(-1)?.kind).toBe('outcome')
    expect(appended.at(-1)?.checksFailing).toBeUndefined()
  })

  it('lists several problems on their own lines and leaves other notices as toasts', () => {
    const many = checkNoteEvents({ result_checks: [ragged, { code: 'xlsx_invalid', severity: 'error', path: 'out.xlsx', message: 'Not a workbook.' }] })
    expect(many).toEqual([{
      type: 'turn.checks',
      payload: { failing: 2, text: "Checked the result: 2 problems:\n- totals.csv: 1 row do not have the header's 3 columns: line 3 has 2.\n- out.xlsx: Not a workbook." }
    }])
    expect(checkNoteEvents({ level: 'info', message: 'Saved.' })).toBeUndefined()
  })
})

describe('the wait before the first token', () => {
  it('knows whether a request has started and how big the prompt is', () => {
    const { daemon } = harness()
    daemon('turn_begin')
    expect(getTurnState().providerStarted).toBe(false)
    daemon('status_update', { kind: 'provider_wait', text: 'Waiting for model response…', prompt_tokens: 2934 })
    expect(getTurnState()).toMatchObject({ providerPromptTokens: 2934, providerStarted: true, providerWaiting: true })
    daemon('status_update', { kind: 'provider_ready', text: '' })
    expect(getTurnState()).toMatchObject({ providerStarted: true, providerWaiting: false })
    daemon('turn_end', { stop_reason: 'completed' })
    daemon('turn_begin')
    expect(getTurnState()).toMatchObject({ providerPromptTokens: 0, providerStarted: false })
  })
})
