// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The data boundary's outbound gate on model calls (M1.3).
 *
 * Wraps a turn's LlmClient so every request is checked before it leaves the
 * machine: private mode refuses cloud requests, `approve_outbound` asks the
 * user, and every decision lands in the disclosure log (sizes and hashes,
 * never the text). A refusal throws DataBoundaryError, which the turn loop
 * treats as terminal: it is not retried and never silently switches model.
 */

import type { CompletionRequest, LlmClient, LlmCompletion } from '../llms/client.js'
import {
  DataBoundaryError,
  type DataTrust,
  type OutboundGate,
  type OutboundPurpose,
  type OutboundRequestSummary,
} from '../security/boundaryTypes.js'

export interface OutboundGateContext {
  readonly gate: OutboundGate
  readonly trust: DataTrust
  /** Provider profile (or provider) name shown in the log and the approval card. */
  readonly provider: string
  readonly purpose?: OutboundPurpose
  readonly sessionId?: string
  readonly agentId?: string
}

/** The exact text a request would send: messages and tool schemas, serialized. */
export function outboundText(request: CompletionRequest): string {
  return JSON.stringify({ messages: request.messages, tools: request.tools ?? [] })
}

export function outboundSummary(request: CompletionRequest, context: OutboundGateContext): OutboundRequestSummary {
  const text = outboundText(request)
  return {
    provider: context.provider,
    model: request.model,
    trust: context.trust,
    purpose: context.purpose ?? 'turn',
    ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
    ...(context.agentId === undefined ? {} : { agentId: context.agentId }),
    text,
    bytes: new TextEncoder().encode(text).length,
  }
}

async function admit(request: CompletionRequest, context: OutboundGateContext, signal?: AbortSignal): Promise<void> {
  const verdict = await context.gate.check(outboundSummary(request, context), signal)
  if (!verdict.allow) throw new DataBoundaryError('DATA_BOUNDARY_DENIED', `Data boundary: ${verdict.reason}`)
}

/** Gate every stream and completion of `llm`. With no context, `llm` is returned unchanged. */
export function gateLlmClient(llm: LlmClient, context: OutboundGateContext | undefined): LlmClient {
  if (context === undefined) return llm
  const gated: LlmClient = {
    async *stream(request, signal) {
      await admit(request, context, signal)
      yield* llm.stream(request, signal)
    },
    ...(llm.complete ? {
      complete: async (request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> => {
        await admit(request, context, signal)
        return llm.complete!(request, signal)
      },
    } : {}),
    ...(llm.close ? { close: () => llm.close!() } : {}),
    ...(llm.authFingerprint ? { authFingerprint: (signal?: AbortSignal) => llm.authFingerprint!(signal) } : {}),
  }
  return gated
}

export const OUTBOUND_SEND = 'Send'
export const OUTBOUND_DENY = 'Deny'

/** Question port the approval card goes through (the daemon interaction board's `ask`). */
export type AskPort = (
  sessionId: string,
  question: { readonly question: string; readonly options: readonly string[]; readonly allowFreeform: boolean },
  signal?: AbortSignal,
) => Promise<string>

/**
 * Approval card for one outbound request: Send or Deny. Anything other than
 * an explicit Send (a dismissed card, an aborted turn) is a denial.
 */
export function outboundApprovalPort(ask: AskPort): (summary: OutboundRequestSummary, signal?: AbortSignal) => Promise<boolean> {
  return async (summary, signal) => {
    if (!summary.sessionId) return false
    const kib = (summary.bytes / 1024).toFixed(1)
    const answer = await ask(summary.sessionId, {
      question: `Send ${kib} KiB of this conversation (${summary.purpose}) to ${summary.provider}/${summary.model}? It is a cloud provider; the data boundary asks before anything leaves this machine.`,
      options: [OUTBOUND_SEND, OUTBOUND_DENY],
      allowFreeform: false,
    }, signal)
    return answer.trim() === OUTBOUND_SEND
  }
}
