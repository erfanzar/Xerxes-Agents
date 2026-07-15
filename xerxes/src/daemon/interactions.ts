// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { PermissionRequest } from '../streaming/events.js'
import type { PermissionBroker, PermissionDecision } from '../streaming/permissions.js'
import {
  ApprovalRecord,
  ApprovalScope,
  ApprovalStore,
  approvalArgumentsHash,
} from '../security/approvals.js'
import type { DaemonEvent } from './runtime.js'

export interface DaemonQuestion {
  readonly allowFreeform?: boolean
  readonly options?: readonly string[]
  readonly placeholder?: string
  readonly question: string
  readonly questionId?: string
  readonly toolCallId?: string
}

interface PendingPermission {
  readonly abort: () => void
  readonly argsHash: string
  readonly resolve: (decision: PermissionDecision) => void
  readonly sessionId: string
  readonly signal: AbortSignal | undefined
  readonly toolName: string
}

interface PendingQuestion {
  readonly allowFreeform: boolean
  readonly abort: () => void
  readonly options: readonly string[]
  readonly questionId: string
  readonly resolve: (answer: string) => void
  readonly sessionId: string
  readonly signal: AbortSignal | undefined
}

interface SessionBinding {
  readonly emit: (event: DaemonEvent) => void
}

export interface DaemonInteractionBoardOptions {
  /** Optional explicit store for remembered approval decisions. */
  readonly approvalStore?: ApprovalStore
  /** Receives persistence failures after the live permission decision is safely resolved. */
  readonly onApprovalStoreError?: (error: unknown) => void
}

/**
 * Owns human-in-the-loop waits for one daemon process.
 *
 * The streaming loop emits an approval event before awaiting its permission
 * broker. Question-producing tools use this board directly, which emits a
 * native `question_request` event then blocks until the same connection sends
 * `question_response`.
 */
export class DaemonInteractionBoard {
  private readonly approvalStore: ApprovalStore | undefined
  private readonly approvedTools = new Map<string, Set<string>>()
  private readonly bindings = new Map<string, SessionBinding>()
  private readonly onApprovalStoreError: (error: unknown) => void
  private readonly permissions = new Map<string, PendingPermission>()
  private readonly questions = new Map<string, PendingQuestion>()

  constructor(options: DaemonInteractionBoardOptions = {}) {
    this.approvalStore = options.approvalStore
    this.onApprovalStoreError = options.onApprovalStoreError ?? (() => undefined)
  }

  bind(sessionId: string, emit: (event: DaemonEvent) => void): () => void {
    const binding: SessionBinding = { emit }
    this.bindings.set(sessionId, binding)
    return () => {
      if (this.bindings.get(sessionId) === binding) {
        this.bindings.delete(sessionId)
      }
      this.endTurn(sessionId)
    }
  }

  permissionBroker(sessionId: string): PermissionBroker {
    return {
      request: (request, signal) => this.requestPermission(sessionId, request, signal),
    }
  }

  async ask(sessionId: string, request: DaemonQuestion, signal?: AbortSignal): Promise<string> {
    const binding = this.bindings.get(sessionId)
    if (!binding) {
      throw new Error('User question requested outside an active daemon turn')
    }
    const question = request.question.trim()
    if (!question) {
      throw new TypeError('Question must not be empty')
    }
    if (signal?.aborted) {
      return ''
    }

    const requestId = crypto.randomUUID().replaceAll('-', '')
    const questionId = request.questionId?.trim() || 'answer'
    return new Promise(resolve => {
      const abort = () => this.finishQuestion(requestId, '')
      this.questions.set(requestId, {
        abort,
        allowFreeform: request.allowFreeform ?? true,
        options: [...(request.options ?? [])],
        questionId,
        resolve,
        sessionId,
        signal,
      })
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) {
        abort()
        return
      }
      binding.emit({
        type: 'question_request',
        payload: {
          id: requestId,
          tool_call_id: request.toolCallId ?? '',
          questions: [{
            id: questionId,
            question,
            options: [...(request.options ?? [])],
            allow_free_form: request.allowFreeform ?? true,
            ...(request.placeholder?.trim() ? { placeholder: request.placeholder.trim() } : {}),
          }],
        },
      })
    })
  }

  pendingPermissionIds(): readonly string[] {
    return [...this.permissions.keys()]
  }

  pendingQuestionIds(): readonly string[] {
    return [...this.questions.keys()]
  }

  respondPermission(requestId: string, response: string): boolean {
    const pending = this.permissions.get(requestId)
    if (!pending) {
      return false
    }
    const resolution = permissionResolution(response)
    this.permissions.delete(requestId)
    pending.signal?.removeEventListener('abort', pending.abort)
    if (resolution.decision === 'approve_for_session') {
      const tools = this.approvedTools.get(pending.sessionId) ?? new Set<string>()
      tools.add(pending.toolName)
      this.approvedTools.set(pending.sessionId, tools)
    }
    pending.resolve(resolution.decision)
    if (resolution.scope !== undefined && this.approvalStore !== undefined) {
      try {
        this.approvalStore.add(new ApprovalRecord({
          toolName: pending.toolName,
          scope: resolution.scope,
          granted: resolution.decision !== 'reject',
          sessionId: pending.sessionId,
          argsHash: pending.argsHash,
        }))
      } catch (error) {
        this.onApprovalStoreError(error)
      }
    }
    return true
  }

  respondQuestion(requestId: string, answers: Readonly<Record<string, string>>): boolean {
    const pending = this.questions.get(requestId)
    if (!pending) {
      return false
    }
    const answer = answers[pending.questionId] ?? Object.values(answers).find(value => typeof value === 'string')
    if (answer === undefined) {
      return false
    }
    if (!pending.allowFreeform && !pending.options.includes(answer)) {
      return false
    }
    this.finishQuestion(requestId, answer)
    return true
  }

  cancelSession(sessionId: string): void {
    this.approvedTools.delete(sessionId)
    this.approvalStore?.clearSession(sessionId)
    this.endTurn(sessionId)
  }

  private endTurn(sessionId: string): void {
    for (const [requestId, pending] of this.permissions) {
      if (pending.sessionId !== sessionId) continue
      this.permissions.delete(requestId)
      pending.signal?.removeEventListener('abort', pending.abort)
      pending.resolve('reject')
    }
    for (const [requestId, pending] of this.questions) {
      if (pending.sessionId === sessionId) {
        this.finishQuestion(requestId, '')
      }
    }
  }

  private requestPermission(
    sessionId: string,
    request: PermissionRequest,
    signal?: AbortSignal,
  ): Promise<PermissionDecision> {
    const argsHash = this.approvalStore === undefined ? '' : approvalArgumentsHash(request.toolCall.function.arguments)
    const remembered = this.approvalStore?.check(request.toolCall.function.name, sessionId, argsHash)
    if (remembered !== undefined) {
      return Promise.resolve(remembered ? 'approve' : 'reject')
    }
    if (this.approvedTools.get(sessionId)?.has(request.toolCall.function.name)) {
      return Promise.resolve('approve')
    }
    if (signal?.aborted) {
      return Promise.resolve('reject')
    }
    return new Promise(resolve => {
      const abort = () => this.finishPermission(request.requestId, 'reject')
      this.permissions.set(request.requestId, {
        abort,
        argsHash,
        resolve,
        sessionId,
        signal,
        toolName: request.toolCall.function.name,
      })
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) {
        abort()
      }
    })
  }

  private finishPermission(requestId: string, decision: PermissionDecision): void {
    const pending = this.permissions.get(requestId)
    if (!pending) {
      return
    }
    this.permissions.delete(requestId)
    pending.signal?.removeEventListener('abort', pending.abort)
    pending.resolve(decision)
  }

  private finishQuestion(requestId: string, answer: string): void {
    const pending = this.questions.get(requestId)
    if (!pending) {
      return
    }
    this.questions.delete(requestId)
    pending.signal?.removeEventListener('abort', pending.abort)
    pending.resolve(answer)
  }
}

function permissionResolution(response: string): { readonly decision: PermissionDecision; readonly scope?: ApprovalScope } {
  const value = response.trim().toLowerCase()
  if (value === 'approve_for_session' || value === 'always') {
    return {
      decision: 'approve_for_session',
      ...(value === 'always' ? { scope: ApprovalScope.ALWAYS } : { scope: ApprovalScope.SESSION }),
    }
  }
  if (value === 'approve' || value === 'allow') {
    return { decision: 'approve' }
  }
  return { decision: 'reject' }
}
