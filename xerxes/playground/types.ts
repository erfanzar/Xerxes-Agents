// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A text fragment produced while an evaluation turn is running. */
export interface EvaluationTextEvent {
  readonly text: string
  readonly type: 'text'
}

/** A tool invocation observed during an evaluation turn. */
export interface EvaluationToolCallEvent {
  readonly name: string
  readonly type: 'tool_call'
}

/** A runtime request for an already-approved evaluation permission. */
export interface EvaluationApprovalEvent {
  readonly id: string
  readonly type: 'approval_request'
}

/** A progress update emitted by the hosted runtime. */
export interface EvaluationStatusEvent {
  readonly contextTokens: number | undefined
  readonly type: 'status'
}

/** A runtime notification. Only error notifications affect the final turn result. */
export interface EvaluationNotificationEvent {
  readonly body: string | undefined
  readonly severity: 'debug' | 'error' | 'info' | 'warning'
  readonly title: string | undefined
  readonly type: 'notification'
}

/** Terminal marker for one evaluation turn. */
export interface EvaluationTurnEndEvent {
  readonly type: 'turn_end'
}

/** Events the native harness accepts from an embedding runtime. */
export type EvaluationEvent =
  | EvaluationApprovalEvent
  | EvaluationNotificationEvent
  | EvaluationStatusEvent
  | EvaluationTextEvent
  | EvaluationToolCallEvent
  | EvaluationTurnEndEvent

/** Runtime settings supplied to an explicitly embedded evaluation session. */
export interface EvaluationStartRequest {
  readonly homeDirectory: string
  readonly permissionMode: 'accept-all'
  readonly workspaceDirectory: string
}

/** Result returned after an evaluation runtime has initialized. */
export interface EvaluationStartResult {
  readonly model?: string
}

/** Private filesystem paths made available to an explicitly loaded evaluation transport. */
export interface EvaluationTransportContext {
  readonly homeDirectory: string
  readonly runDirectory: string
  readonly workspaceDirectory: string
}

/** Request for a single evaluation turn. */
export interface EvaluationSubmitRequest {
  readonly prompt: string
  readonly signal: AbortSignal
  readonly timeoutMs: number
}

/**
 * Explicit host boundary for real runtime/provider execution.
 *
 * The playground deliberately has no default transport and does not inspect
 * environment credentials. A host that wants to evaluate a real model must
 * construct this port and elect to call the suite runner.
 */
export interface EvaluationSessionPort {
  approve(requestId: string): Promise<void>
  close(): Promise<void>
  reset(): Promise<void>
  start(request: EvaluationStartRequest): Promise<EvaluationStartResult>
  submit(request: EvaluationSubmitRequest): AsyncIterable<EvaluationEvent> | Promise<AsyncIterable<EvaluationEvent>>
}

/** Factory contract used by the standalone native warm-up CLI. */
export type EvaluationSessionPortFactory = (
  context: EvaluationTransportContext,
) => EvaluationSessionPort | Promise<EvaluationSessionPort>

/** One normalized completed turn used by every scorer. */
export interface EvaluationTurnResult {
  readonly contextTokens: number
  readonly error: string | undefined
  readonly latencyMs: number
  readonly retries: number
  readonly text: string
  readonly tools: readonly string[]
}

/** Common agent boundary used by warm-up and hard suite runners. */
export interface EvaluationAgent {
  readonly model: string
  close(): Promise<void>
  freshSession(): Promise<void>
  start(): Promise<void>
  turn(prompt: string, options?: EvaluationTurnOptions): Promise<EvaluationTurnResult>
}

/** Per-turn limits. Retry behavior belongs to the agent implementation. */
export interface EvaluationTurnOptions {
  readonly retries?: number
  readonly timeoutMs?: number
}

/** A pass/fail outcome returned by a task-specific behavioral grader. */
export interface EvaluationCheck {
  readonly detail: string
  readonly ok: boolean
}

/** A scored row rendered in a warm-up or hard-suite report. */
export interface EvaluationScoreRow {
  readonly category: string
  readonly detail: string
  readonly diagnosis: string | undefined
  readonly difficulty: number | undefined
  readonly latencyMs: number
  readonly name: string
  readonly ok: boolean
}

/** A complete native evaluation scorecard. */
export interface EvaluationReport {
  readonly kind: 'hard' | 'warmup'
  readonly model: string
  readonly rows: readonly EvaluationScoreRow[]
  readonly sandboxDirectory: string
  readonly totalLatencyMs: number
}

/** Inputs sent to an optional failure-diagnosis port. */
export interface EvaluationDiagnosisRequest {
  readonly graderDetail: string
  readonly reply: string
  readonly taskPrompt: string
  readonly tools: readonly string[]
  readonly whyHard: string
}

/**
 * Optional external judge boundary.
 *
 * A caller may implement this with a provider, but the harness never creates
 * a provider client, reads credentials, or performs a network request itself.
 */
export interface EvaluationJudgePort {
  diagnose(request: EvaluationDiagnosisRequest): Promise<string>
}

/** Native file fixture used to construct a hard-task workspace. */
export interface EvaluationTaskFile {
  readonly content: string
  readonly path: string
}

/** A prompt step in an evaluation task. */
export interface EvaluationTaskStep {
  readonly freshSession: boolean
  readonly prompt: string
}
