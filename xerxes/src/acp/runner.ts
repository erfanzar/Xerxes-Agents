// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage, type ToolExecutor } from '../executors/toolRegistry.js'
import { mergePersistedSubagentSnapshots } from '../agents/subagentPersistence.js'
import type { SubagentTurnCoordinator } from '../daemon/subagentCoordinator.js'
import { formatSubagentResults } from '../daemon/turnRunner.js'
import type { LlmClient } from '../llms/client.js'
import { createAgentState, type AgentState, type PermissionRequest, type StreamEvent } from '../streaming/events.js'
import { runTurn } from '../streaming/loop.js'
import type { PermissionBroker, PermissionDecision, PermissionMode, ToolPolicy } from '../streaming/permissions.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { routePermission, type AcpPermissionBoard } from './permissions.js'
import type { AcpSession } from './session.js'
import type { AcpEventEmitter, AcpModelInfo, AcpPromptRequest, AcpWireEvent } from './types.js'
import { toAcpEvent } from './events.js'

export interface AcpAgentRunnerOptions {
  readonly agentId?: string
  readonly defaultPermissionMode?: PermissionMode
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly model: string
  readonly modelListProvider?: () => readonly AcpModelInfo[]
  readonly policy?: ToolPolicy
  readonly systemPrompt?: string
  /** Joins detached child work back into the ACP prompt that created it. */
  readonly subagentCoordinator?: SubagentTurnCoordinator
  readonly temperature?: number
  readonly toolExecutor?: ToolExecutor
  readonly tools?: readonly ToolDefinition[]
  readonly topP?: number
}

interface ActivePrompt {
  readonly controller: AbortController
  readonly emit: AcpEventEmitter
  readonly session: AcpSession
}

interface PendingQuestion {
  readonly abortHandler: () => void
  readonly question: string
  readonly resolve: (answer: string) => void
  readonly sessionId: string
  readonly signal: AbortSignal
}

/**
 * Drive the portable TypeScript agent loop for ACP sessions.
 *
 * State is retained independently for each ACP session. The runner itself is
 * provider-neutral: callers inject an `LlmClient`, tool executor, schemas, and
 * policy just as they do for the daemon's `AgentTurnRunner`.
 */
export class AcpAgentRunner {
  private readonly activePrompts = new Map<string, ActivePrompt>()
  private readonly options: AcpAgentRunnerOptions
  private permissionBoard: AcpPermissionBoard | undefined
  private readonly questions = new Map<string, PendingQuestion>()
  private readonly states = new Map<string, AgentState>()

  constructor(options: AcpAgentRunnerOptions) {
    this.options = options
  }

  cancel(sessionId: string): boolean {
    const active = this.activePrompts.get(sessionId)
    if (!active) {
      return false
    }
    active.session.cancelled = true
    active.controller.abort(new Error('ACP prompt cancelled'))
    this.resolveQuestionsForSession(sessionId, '')
    return true
  }

  listModels(): readonly AcpModelInfo[] {
    return [...(this.options.modelListProvider?.() ?? [{ id: this.options.model, name: this.options.model }])]
  }

  listTools(): readonly ToolDefinition[] {
    return [...(this.options.tools ?? [])]
  }

  pendingQuestions(): readonly Record<string, unknown>[] {
    return [...this.questions.entries()].map(([inputId, question]) => ({
      input_id: inputId,
      session_id: question.sessionId,
      question: question.question,
    }))
  }

  resetSession(sessionId: string): void {
    this.states.delete(sessionId)
    this.resolveQuestionsForSession(sessionId, '')
  }

  respondQuestion(inputId: string, answer: string): Record<string, boolean> {
    const question = this.questions.get(inputId)
    if (!question) {
      return { ok: false }
    }
    this.questions.delete(inputId)
    question.signal.removeEventListener('abort', question.abortHandler)
    question.resolve(answer)
    return { ok: true }
  }

  setPermissionBoard(board: AcpPermissionBoard): void {
    this.permissionBoard = board
  }

  stateFor(sessionId: string): AgentState | undefined {
    return this.states.get(sessionId)
  }

  /** Run one streamed ACP prompt and return the final token/model summary. */
  async runPrompt(request: AcpPromptRequest): Promise<Record<string, unknown>> {
    const { session } = request
    if (this.activePrompts.has(session.sessionId)) {
      return { ok: false, error: `prompt already active for session: ${session.sessionId}` }
    }

    const controller = new AbortController()
    if (session.cancelled) {
      controller.abort(new Error('ACP session is cancelled'))
    }
    const emit = request.emit ?? (() => undefined)
    const active: ActivePrompt = { session, controller, emit }
    this.activePrompts.set(session.sessionId, active)
    const state = this.states.get(session.sessionId) ?? createAgentState()
    this.states.set(session.sessionId, state)
    const model = session.modelOverride ?? this.options.model
    const permissionMode = permissionModeFor(session, this.options.defaultPermissionMode ?? 'accept-all')
    state.metadata.permission_mode = permissionMode
    const summary: Record<string, unknown> = { ok: true, cancelled: false }
    const subagentCohort = this.options.subagentCoordinator?.begin(session.sessionId)

    try {
      const permissionBroker: PermissionBroker = {
        request: (permission, signal) => this.resolvePermission(session, permission, emit, signal),
      }
      for await (const event of runTurn({
        model,
        state,
        userMessage: request.text,
        sessionId: session.sessionId,
        ...(this.options.agentId ? { agentId: this.options.agentId } : {}),
        ...(this.options.maxTokens !== undefined ? { maxTokens: this.options.maxTokens } : {}),
        ...(this.options.systemPrompt ? { systemPrompt: this.options.systemPrompt } : {}),
        ...(this.options.temperature !== undefined ? { temperature: this.options.temperature } : {}),
        ...(this.options.tools ? { tools: this.options.tools } : {}),
        ...(this.options.topP !== undefined ? { topP: this.options.topP } : {}),
        permissionMode,
      }, {
        ...(subagentCohort ? {
          awaitAgentEvents: async signal => {
            const snapshots = await subagentCohort.waitForResults(signal)
            mergePersistedSubagentSnapshots(state.metadata, snapshots)
            return formatSubagentResults(snapshots)
          },
        } : {}),
        llm: this.options.llm,
        permissionBroker,
        ...(this.options.policy ? { policy: this.options.policy } : {}),
        ...(this.options.toolExecutor ? { toolExecutor: this.options.toolExecutor } : {}),
      }, controller.signal)) {
        if (event.type === 'permission_request') {
          continue
        }
        await emit(toAcpEvent(event).toWire())
        updateSummary(summary, event)
      }
    } catch (error) {
      return {
        ok: false,
        error: errorMessage(error),
        cancelled: session.cancelled || controller.signal.aborted,
      }
    } finally {
      subagentCohort?.close()
      this.activePrompts.delete(session.sessionId)
      this.resolveQuestionsForSession(session.sessionId, '')
    }

    summary.cancelled = session.cancelled || controller.signal.aborted
    return summary
  }

  /**
   * Surface a tool's question to the ACP client and await its answer.
   *
   * Tool adapters that need interactive input can call this method with their
   * current ACP session id; it deliberately has no process-global callback.
   */
  async askUserQuestion(sessionId: string, question: string): Promise<string> {
    const active = this.activePrompts.get(sessionId)
    if (!active) {
      throw new Error('AskUserQuestion called outside an active ACP turn')
    }
    if (active.controller.signal.aborted || active.session.cancelled) {
      return ''
    }

    const inputId = crypto.randomUUID().replaceAll('-', '')
    return new Promise<string>((resolve, reject) => {
      const abortHandler = () => {
        this.questions.delete(inputId)
        resolve('')
      }
      const pending: PendingQuestion = {
        sessionId,
        question,
        resolve,
        abortHandler,
        signal: active.controller.signal,
      }
      this.questions.set(inputId, pending)
      active.controller.signal.addEventListener('abort', abortHandler, { once: true })
      if (active.controller.signal.aborted) {
        abortHandler()
        return
      }
      void Promise.resolve(active.emit({
        kind: 'input_request',
        input_id: inputId,
        session_id: sessionId,
        question,
      })).catch(error => {
        this.questions.delete(inputId)
        active.controller.signal.removeEventListener('abort', abortHandler)
        reject(error)
      })
    })
  }

  private async resolvePermission(
    session: AcpSession,
    permission: PermissionRequest,
    emit: AcpEventEmitter,
    signal?: AbortSignal,
  ): Promise<PermissionDecision> {
    const request = routePermission({
      sessionId: session.sessionId,
      toolName: permission.toolCall.function.name,
      description: permission.description,
      inputs: permission.inputs,
    })
    const board = this.permissionBoard
    if (!board) {
      await emit({
        kind: 'permission_request',
        permission_id: request.id,
        session_id: session.sessionId,
        tool_name: request.toolName,
        description: request.description,
        inputs: request.inputs,
      })
      return 'reject'
    }
    // Register before notifying the client so a fast `permission/respond`
    // cannot race ahead of the pending-request board.
    board.submit(request)
    try {
      await emit({
        kind: 'permission_request',
        permission_id: request.id,
        session_id: session.sessionId,
        tool_name: request.toolName,
        description: request.description,
        inputs: request.inputs,
      })
      const allowed = await board.awaitDecision(request, signal)
      return allowed && !session.cancelled && !signal?.aborted ? 'approve' : 'reject'
    } finally {
      board.drop(request.id)
    }
  }

  private resolveQuestionsForSession(sessionId: string, answer: string): void {
    for (const [inputId, question] of this.questions) {
      if (question.sessionId !== sessionId) {
        continue
      }
      this.questions.delete(inputId)
      question.signal.removeEventListener('abort', question.abortHandler)
      question.resolve(answer)
    }
  }
}

function permissionModeFor(session: AcpSession, fallback: PermissionMode): PermissionMode {
  const raw = session.metadata.permission_mode ?? session.metadata.permissionMode
  return raw === 'accept-all' || raw === 'auto' || raw === 'manual' || raw === 'plan' ? raw : fallback
}

function updateSummary(summary: Record<string, unknown>, event: StreamEvent): void {
  if (event.type !== 'turn_done') {
    return
  }
  summary.input_tokens = event.usage.inputTokens
  summary.output_tokens = event.usage.outputTokens
  summary.tool_calls_count = event.toolCallsCount
  summary.model = event.model
}
