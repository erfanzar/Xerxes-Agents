// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage, type ToolExecutor } from '../executors/toolRegistry.js'
import { mergePersistedSubagentSnapshots } from '../agents/subagentPersistence.js'
import type { SubagentTurnCoordinator } from '../daemon/subagentCoordinator.js'
import { formatSubagentResults } from '../daemon/turnRunner.js'
import type { LlmClient } from '../llms/client.js'
import type { SpawnedAgentSnapshot } from '../operators/subagents.js'
import { createAgentState, type AgentState, type PermissionRequest, type StreamEvent } from '../streaming/events.js'
import type { HookRunner } from '../extensions/hooks.js'
import { EditFeedback } from '../runtime/editFeedback.js'
import { withActiveSession } from '../runtime/sessionContext.js'
import { runCascadingTurn, type CascadeStep, type CascadingTurnOptions } from '../runtime/cascade.js'
import { renderFindings } from '../runtime/resultChecks.js'
import type { RunSafety } from '../runtime/runSafety.js'
import { runTurn } from '../streaming/loop.js'
import { DEFAULT_PERMISSION_MODE, type PermissionBroker, type PermissionDecision, type PermissionMode, type ToolPolicy } from '../streaming/permissions.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import type { RecoverEngine } from '../local/engineRecovery.js'
import { routePermission, type AcpPermissionBoard } from './permissions.js'
import type { AcpSession } from './session.js'
import type { AcpEventEmitter, AcpModelInfo, AcpPromptRequest, AcpWireEvent } from './types.js'
import { toAcpEvent } from './events.js'

export interface AcpAgentRunnerOptions {
  readonly createEditFeedback?: (cwd: string) => EditFeedback
  readonly agentId?: string
  readonly defaultPermissionMode?: PermissionMode
  /**
   * The user's configured hooks for one session working directory, the same
   * loader the daemon uses (`workspaceShellHooks`). Absent, no hooks run.
   */
  readonly hookRunnerForCwd?: (cwd: string) => HookRunner | undefined
  readonly llm: LlmClient
  /** Network retries before a turn fails; see `defaultMaxNetworkRetries`. */
  readonly maxNetworkRetries?: number
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
  readonly topK?: number
  readonly topP?: number
  /**
   * The cascade for one session (checks, escalation, advisor), as the daemon
   * runs it. Absent: one attempt per prompt.
   */
  readonly cascadeFor?: (sessionId: string, cwd: string) => Pick<CascadingTurnOptions, 'hooks'> | undefined
  /** The session's safety floor (`runtime/runSafety.ts`), used in `auto` mode only. */
  readonly runSafetyFor?: (sessionId: string, cwd: string) => Promise<RunSafety | undefined>
  /** Reloads a local engine Xerxes started when it loses the model (`local/engineRecovery.ts`). */
  readonly recoverEngineFor?: (model: string) => RecoverEngine | undefined
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

    if (session.cancelled) {
      // A cancellation that arrived before this prompt started (when
      // runner.cancel found no active prompt to abort) must not be silently
      // dropped: consume the flag and fail the prompt instead of running.
      session.cancelled = false
      return {
        ok: false,
        cancelled: true,
        error: `prompt cancelled before it started for session: ${session.sessionId}`,
      }
    }
    const controller = new AbortController()
    const emit = request.emit ?? (() => undefined)
    const active: ActivePrompt = { session, controller, emit }
    this.activePrompts.set(session.sessionId, active)
    const state = this.states.get(session.sessionId) ?? createAgentState()
    this.states.set(session.sessionId, state)
    const model = session.modelOverride ?? this.options.model
    const permissionMode = permissionModeFor(session, this.options.defaultPermissionMode ?? DEFAULT_PERMISSION_MODE)
    state.metadata.permission_mode = permissionMode
    const recoverEngine = this.options.recoverEngineFor?.(model)
    const summary: Record<string, unknown> = { ok: true, cancelled: false }
    const subagentCohort = this.options.subagentCoordinator?.begin(session.sessionId)
    let pendingAgentEventSnapshots: readonly SpawnedAgentSnapshot[] = []
    const feedback = this.options.createEditFeedback?.(session.cwd)
    const executor = this.options.toolExecutor && feedback ? feedback.wrap(this.options.toolExecutor) : this.options.toolExecutor
    const hookRunner = this.options.hookRunnerForCwd?.(session.cwd)

    try {
      const permissionBroker: PermissionBroker = {
        request: (permission, signal) => this.resolvePermission(session, permission, emit, signal),
      }
      let runSafety: RunSafety | undefined
      let runSafetyNote: string | undefined
      try {
        runSafety = await this.options.runSafetyFor?.(session.sessionId, session.cwd)
      } catch (error) {
        runSafetyNote = `The safety floor did not start, so writes and commands ask: ${errorMessage(error)}`
      }
      const configured = this.options.cascadeFor?.(session.sessionId, session.cwd)
      // Under accept-all the user chose to let the run change anything.
      const cascade = configured && runSafety && permissionMode !== 'accept-all'
        ? { hooks: { ...configured.hooks, inputFindings: () => runSafety.inputFindings() } }
        : configured
      // A checked attempt is shown as plain text, as the edit-diagnostics report is.
      const reportStep = (step: CascadeStep): void => {
        const lines = [
          ...(step.findings.length ? [renderFindings(step.findings)] : []),
          ...(step.escalation ? [step.escalation.message] : []),
          ...(step.advisor?.answered === true ? ['Escalation: the advisor answered; the model retries with its advice.'] : []),
          ...(step.advisor?.answered === false ? [`Advisor consult failed: ${step.advisor.error}`] : []),
        ]
        if (lines.length) cascadeReports.push(`\n\n${lines.join('\n')}\n\n`)
      }
      const cascadeReports: string[] = runSafetyNote ? [`${runSafetyNote}\n\n`] : []
      const turn = (turnRequest: Parameters<typeof runTurn>[0], dependencies: Parameters<typeof runTurn>[1], turnSignal: AbortSignal) => cascade
        ? runCascadingTurn(turnRequest, dependencies, { hooks: cascade.hooks, projectRoot: session.cwd, onStep: reportStep }, turnSignal)
        : runTurn(turnRequest, dependencies, turnSignal)
      for await (const event of withActiveSession(session, turn({
        model,
        state,
        userMessage: request.text,
        sessionId: session.sessionId,
        ...(this.options.agentId ? { agentId: this.options.agentId } : {}),
        ...(this.options.maxTokens !== undefined ? { maxTokens: this.options.maxTokens } : {}),
        ...(this.options.systemPrompt ? { systemPrompt: this.options.systemPrompt } : {}),
        ...(this.options.temperature !== undefined ? { temperature: this.options.temperature } : {}),
        ...(this.options.tools ? { tools: this.options.tools } : {}),
        ...(this.options.topK !== undefined ? { topK: this.options.topK } : {}),
        ...(this.options.topP !== undefined ? { topP: this.options.topP } : {}),
        permissionMode,
      }, {
        ...(subagentCohort ? {
          awaitAgentEvents: async signal => {
            pendingAgentEventSnapshots = await subagentCohort.waitForResults(signal)
            mergePersistedSubagentSnapshots(state.metadata, pendingAgentEventSnapshots)
            return formatSubagentResults(pendingAgentEventSnapshots)
          },
          acknowledgeAgentEvents: () => {
            if (!pendingAgentEventSnapshots.length) return
            this.options.subagentCoordinator?.consume(pendingAgentEventSnapshots)
            pendingAgentEventSnapshots = []
          },
        } : {}),
        llm: this.options.llm,
        ...(hookRunner ? { hookRunner } : {}),
        ...(this.options.maxNetworkRetries === undefined ? {} : { maxNetworkRetries: this.options.maxNetworkRetries }),
        permissionBroker,
        ...(runSafety ? { runSafety } : {}),
        ...(recoverEngine ? { recoverEngine } : {}),
        ...(this.options.policy ? { policy: this.options.policy } : {}),
        ...(executor ? { toolExecutor: executor } : {}),
      }, controller.signal))) {
        for (const text of cascadeReports.splice(0)) await emit(toAcpEvent({ type: 'text', text }).toWire())
        if (event.type === 'permission_request' || event.type === 'provider_wait') {
          continue
        }
        if (event.type === 'turn_done' && feedback && !controller.signal.aborted) {
          const report = await feedback.report(controller.signal)
          if (report) {
            state.messages.push({ role: 'user', content: report })
            await emit(toAcpEvent({ type: 'text', text: '\n\n' + report }).toWire())
          }
        }
        await emit(toAcpEvent(event).toWire())
        updateSummary(summary, event)
      }
      // The last attempt's check runs after its final event.
      for (const text of cascadeReports.splice(0)) await emit(toAcpEvent({ type: 'text', text }).toWire())
    } catch (error) {
      return {
        ok: false,
        error: errorMessage(error),
        cancelled: session.cancelled || controller.signal.aborted,
      }
    } finally {
      // A mid-turn cancellation was honored by aborting this turn; clear the
      // flag so it does not leak into the next prompt for the session.
      session.cancelled = false
      subagentCohort?.close()
      this.activePrompts.delete(session.sessionId)
      this.resolveQuestionsForSession(session.sessionId, '')
    }

    summary.cancelled = controller.signal.aborted
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
