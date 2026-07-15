// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Interaction controller. `planSubmit` is PURE — it turns a draft + turn state
// into a decision, which makes the queue/interrupt/slash routing fully
// unit-testable. `executePlan` is the thin side-effecting half that issues
// gateway RPCs and dispatches UI state changes.

import type { AnyEvent, ApprovalResponseKind } from '../gatewayTypes.js'
import { classifyInput } from './slash.js'
import { decideSubmit } from './queue.js'

export type SubmitPlan =
  | { do: 'noop' }
  | { do: 'exit' }
  | { do: 'send'; text: string }
  | { do: 'queue'; text: string }
  | { do: 'interrupt' }
  | { do: 'drain' }
  | { do: 'shell'; command: string }
  | { do: 'remote'; command: string }
  | { do: 'clear' }
  | { do: 'copy' }
  | { do: 'logs' }
  | { do: 'details'; arg: string }
  | { do: 'queueView' }
  | { do: 'help' }

/** Pure: decide what an Enter press should do. No I/O. */
export function planSubmit(draft: string, busy: boolean, queueLen: number): SubmitPlan {
  const text = draft.trim()
  if (!text) {
    const d = decideSubmit('', busy, queueLen)
    return d.kind === 'interrupt' ? { do: 'interrupt' } : d.kind === 'drain' ? { do: 'drain' } : { do: 'noop' }
  }

  const action = classifyInput(text)
  switch (action.kind) {
    case 'message':
      return busy ? { do: 'queue', text: action.text } : { do: 'send', text: action.text }
    case 'shell':
      return { do: 'shell', command: action.command }
    case 'exit':
      return { do: 'exit' }
    case 'clear':
      return { do: 'clear' }
    case 'copy':
      return { do: 'copy' }
    case 'logs':
      return { do: 'logs' }
    case 'details':
      return { do: 'details', arg: action.arg }
    case 'queue':
      return { do: 'queueView' }
    case 'help':
      return { do: 'help' }
    case 'remote':
      return { do: 'remote', command: action.command }
    case 'noop':
      return { do: 'noop' }
  }
}

// Minimal surface the controller needs from GatewayClient — keeps executePlan
// testable with a fake.
export interface ClientLike {
  readonly sessionKey: string
  request(method: string, params?: Record<string, unknown>): Promise<unknown>
  stderrSnapshot(): string
}

export type Dispatch = (evt: AnyEvent) => void

function ev(type: string, payload: Record<string, unknown> = {}): AnyEvent {
  return { type, payload } as AnyEvent
}

/** Send a normal turn and optimistically echo it. */
function sendTurn(client: ClientLike, dispatch: Dispatch, text: string): void {
  dispatch(ev('__user', { text }))
  void client.request('prompt', { user_input: text, session_key: client.sessionKey }).catch(() => {})
}

/**
 * Perform a plan. Returns true if the host should exit. `onCopy` is supplied by
 * the view because copying needs the rendered transcript (OSC 52 lives there).
 */
export function executePlan(
  client: ClientLike,
  dispatch: Dispatch,
  plan: SubmitPlan,
  ctx: { queue: readonly string[]; onCopy?: () => void; onDetails?: (arg: string) => void }
): { exit: boolean } {
  switch (plan.do) {
    case 'exit':
      return { exit: true }

    case 'send':
      sendTurn(client, dispatch, plan.text)
      return { exit: false }

    case 'queue':
      dispatch(ev('__enqueue', { text: plan.text }))
      dispatch(ev('__notice', { text: `queued (${ctx.queue.length + 1})` }))
      return { exit: false }

    case 'drain': {
      const next = ctx.queue[0]
      if (next) {
        dispatch(ev('__dequeue'))
        sendTurn(client, dispatch, next)
      }
      return { exit: false }
    }

    case 'interrupt':
      void client.request('cancel_all').catch(() => {})
      dispatch(ev('__notice', { text: 'interrupted' }))
      return { exit: false }

    case 'shell':
      // `!cmd` is dispatched to the daemon as a slash so it runs through the
      // same approval/policy path as any tool.
      void client.request('slash', { command: `!${plan.command}` }).catch(() => {})
      return { exit: false }

    case 'remote':
      void client.request('slash', { command: plan.command }).catch(() => {})
      return { exit: false }

    case 'help':
      void client.request('slash', { command: '/help' }).catch(() => {})
      return { exit: false }

    case 'clear':
      dispatch(ev('__clear'))
      void client.request('slash', { command: '/clear' }).catch(() => {})
      return { exit: false }

    case 'logs':
      dispatch(ev('__notice', { text: client.stderrSnapshot().split('\n').slice(-1)[0] || 'no daemon logs' }))
      return { exit: false }

    case 'copy':
      ctx.onCopy?.()
      return { exit: false }

    case 'details':
      ctx.onDetails?.(plan.do === 'details' ? plan.arg : '')
      return { exit: false }

    case 'queueView':
      dispatch(ev('__notice', { text: ctx.queue.length ? `queued: ${ctx.queue.join(' | ')}` : 'queue empty' }))
      return { exit: false }

    case 'noop':
      return { exit: false }
  }
}

/** Answer a pending approval prompt. */
export function respondApproval(
  client: ClientLike,
  dispatch: Dispatch,
  requestId: string,
  response: ApprovalResponseKind
): void {
  void client.request('permission_response', { request_id: requestId, response }).catch(() => {})
  dispatch(ev('__approval_done'))
}

/** Answer a pending clarify/question prompt. */
export function respondQuestion(
  client: ClientLike,
  dispatch: Dispatch,
  requestId: string,
  answers: Record<string, string>
): void {
  void client.request('question_response', { request_id: requestId, answers }).catch(() => {})
  dispatch(ev('__question_done'))
}
