// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The cascade: after an attempt, check what it produced in code, and let the
 * escalation ladder decide by rule whether the model gets another attempt in
 * the same turn, with the findings (or the advisor's answer) as feedback.
 *
 * One implementation for every entry point. The daemon runner calls
 * {@link cascadeStep} inside its own attempt loop; one-shot runs and the SDK
 * call {@link runCascadingTurn}, which wraps the shared streaming loop.
 */

import type { AgentState, StreamEvent } from '../streaming/events.js'
import { runTurn, type TurnDependencies, type TurnRequest } from '../streaming/loop.js'
import type { MessageContent } from '../types/messages.js'
import { renderFindings, runResultChecks, type CheckFinding, type ToolRunSummary } from './resultChecks.js'

/** Tools whose success means a file on disk changed and a checker could disagree. */
export const MUTATING_TOOL_NAMES: ReadonlySet<string> = new Set([
  'AppendFile', 'Edit', 'FileEditTool', 'NotebookEditTool', 'Write', 'WriteFile',
  'append_file', 'edit_file', 'write_file',
])

/** Tools whose output file the result checks inspect, on top of {@link MUTATING_TOOL_NAMES}. */
const CHECKED_OUTPUT_TOOL_NAMES: ReadonlySet<string> = new Set(['XlsxWriter'])

/**
 * Backstop on re-runs within one turn. The escalation ladder (local retries,
 * then advisor calls, stopping when nothing improves) ends well before this.
 */
export const MAX_CASCADE_ROUNDS = 8

export type ResultChecksMode = 'off' | 'warn' | 'enforce'

/** One escalation decision, as `daemon/turnEscalation.ts` produces it. */
export interface CascadeEscalation {
  readonly action: 'retry_local' | 'change_model' | 'ask_advisor' | 'stop'
  readonly level: 'info' | 'warning'
  /** Shown to the user. */
  readonly message: string
  /** The next harness message; absent for `stop`. */
  readonly feedback?: string
  /** For `ask_advisor`: what the host sends the advisor itself. */
  readonly advisorRequest?: { readonly question: string; readonly context: string }
}

export interface AdvisorRequest {
  readonly question: string
  readonly context: string
}

/** What the host supplies. Each piece is optional and off when absent. */
export interface CascadeHooks {
  /** Result checks: `warn` reports, `enforce` also tells the model. Default off. */
  checks(): ResultChecksMode
  /** The session's escalation ladder; undefined means escalation is off. */
  escalate?(findings: readonly CheckFinding[]): CascadeEscalation | undefined
  /** Consult the cloud advisor from code; returns its answer or why it was not consulted. */
  consultAdvisor?(request: AdvisorRequest, signal: AbortSignal): Promise<string>
  /** Inputs the run changed or deleted (the safety floor's input check), each reported once. */
  inputFindings?(): Promise<readonly CheckFinding[]>
}

/** What the checks and the escalation ladder said about one attempt. */
export interface CascadeCheck {
  readonly checksMode: ResultChecksMode
  readonly findings: readonly CheckFinding[]
  /** The checks could not run; the attempt stands. */
  readonly checkError?: string
  readonly escalation?: CascadeEscalation
  readonly escalationError?: string
}

/** What happens next, once the advisor (if the ladder reached it) has answered. */
export interface CascadeNext {
  /** Set when the host consulted the advisor for this step. */
  readonly advisor?: { readonly answered: true } | { readonly answered: false; readonly error: string }
  /** Run the model again in this turn with this harness message. */
  readonly feedback?: string
  /** No re-run, but the model reads this at its next turn (enforce mode without escalation). */
  readonly enforceMessage?: string
}

/** The outcome of checking one attempt. */
export interface CascadeStep extends CascadeCheck, CascadeNext {}

/** The latest assistant reply, as plain text. */
export function latestAssistantContent(state: AgentState): string {
  const message = state.messages.slice().reverse().find(candidate => candidate.role === 'assistant')
  if (!message) return ''
  return typeof message.content === 'string' ? message.content : JSON.stringify(message.content)
}

/**
 * Run the result checks over the executions since `executionsBefore`: files
 * written, commands run, and what the final reply claims. A cascade keeps
 * `executionsBefore` at the start of the user turn, so a fix in a later round
 * is checked against everything the turn touched.
 */
export async function turnResultFindings(
  state: AgentState,
  executionsBefore: number,
  projectRoot: string,
): Promise<CheckFinding[]> {
  const executions = state.toolExecutions.slice(executionsBefore)
  const changedFiles: string[] = []
  const toolRuns: ToolRunSummary[] = []
  for (const execution of executions) {
    const failed = !execution.permitted || execution.result.startsWith('Tool execution failed')
    if (!failed && (MUTATING_TOOL_NAMES.has(execution.name) || CHECKED_OUTPUT_TOOL_NAMES.has(execution.name))) {
      const path = execution.inputs.file_path ?? execution.inputs.path
      if (typeof path === 'string' && path) changedFiles.push(path)
    }
    let ok = !failed
    let command: string | undefined
    if (execution.name === 'exec_command') {
      const cmd = typeof execution.inputs.cmd === 'string' ? execution.inputs.cmd : ''
      const args = Array.isArray(execution.inputs.args) ? execution.inputs.args.filter(arg => typeof arg === 'string') : []
      command = [cmd, ...args].join(' ').trim()
      try {
        const parsed: unknown = JSON.parse(execution.result)
        if (typeof parsed === 'object' && parsed !== null && 'exitCode' in parsed) ok = ok && parsed.exitCode === 0
      } catch { /* A non-JSON result keeps the permitted/failed verdict. */ }
    }
    toolRuns.push({ name: execution.name, ok, ...(command ? { command } : {}) })
  }
  if (!changedFiles.length && !toolRuns.length) return []
  return runResultChecks({ workspaceRoot: projectRoot, changedFiles, finalText: latestAssistantContent(state), toolRuns })
}

/** The next harness message after the advisor answered a failed check. */
export function advisorAnswerFeedback(findings: readonly CheckFinding[], answer: string): string {
  return [
    'Your result still fails these checks after local retries:',
    renderFindings(findings),
    '',
    'A stronger model (the advisor) reviewed the failure and answered:',
    answer.trim(),
    '',
    'Apply this advice. Fix exactly these problems, then run a check that proves the fix before you report the task as done.',
  ].join('\n')
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Check one finished attempt and ask the escalation ladder. Never throws: a
 * check or escalation that cannot run leaves the attempt standing and says why.
 */
export async function cascadeCheck(
  state: AgentState,
  executionsBefore: number,
  projectRoot: string,
  hooks: CascadeHooks,
): Promise<CascadeCheck> {
  let checksMode: ResultChecksMode = 'off'
  let findings: readonly CheckFinding[] = []
  let checkError: string | undefined
  try {
    checksMode = hooks.checks()
    if (checksMode !== 'off') {
      findings = [
        ...await turnResultFindings(state, executionsBefore, projectRoot),
        ...await hooks.inputFindings?.() ?? [],
      ]
    }
  } catch (error) {
    checkError = message(error)
  }
  let escalation: CascadeEscalation | undefined
  let escalationError: string | undefined
  if (checksMode !== 'off' && hooks.escalate) {
    try {
      escalation = hooks.escalate(findings)
    } catch (error) {
      escalationError = message(error)
    }
  }
  return {
    checksMode,
    findings,
    ...(checkError === undefined ? {} : { checkError }),
    ...(escalation === undefined ? {} : { escalation }),
    ...(escalationError === undefined ? {} : { escalationError }),
  }
}

/**
 * Decide what follows a checked attempt: another attempt with feedback, a
 * message for the next turn, or nothing. When the ladder reached the advisor
 * and the host can consult it, this consults it. A cancellation during that
 * call propagates; any other advisor failure becomes feedback to fix locally.
 */
export async function cascadeNext(check: CascadeCheck, hooks: CascadeHooks, signal: AbortSignal): Promise<CascadeNext> {
  const { escalation, findings } = check
  const feedback = findings.length ? escalation?.feedback : undefined
  if (feedback !== undefined && escalation?.action === 'ask_advisor' && escalation.advisorRequest && hooks.consultAdvisor) {
    // Escalation by rule: the host consults the advisor itself rather than
    // trusting a small model to make the call.
    try {
      const answer = await hooks.consultAdvisor(escalation.advisorRequest, signal)
      return { advisor: { answered: true }, feedback: advisorAnswerFeedback(findings, answer) }
    } catch (error) {
      if (signal.aborted) throw error
      return {
        advisor: { answered: false, error: message(error) },
        feedback: `${renderFindings(findings)}\nThe advisor could not be reached. Fix these problems yourself, then run a check that proves the fix before you report the task as done.`,
      }
    }
  }
  if (feedback !== undefined) return { feedback }
  if (check.checksMode === 'enforce' && escalation === undefined && findings.some(finding => finding.severity === 'error')) {
    return { enforceMessage: `${renderFindings(findings)}\nFix these before you report the task as done.` }
  }
  return {}
}

/** {@link cascadeCheck} then {@link cascadeNext}, for hosts that report the outcome once. */
export async function cascadeStep(
  state: AgentState,
  executionsBefore: number,
  projectRoot: string,
  hooks: CascadeHooks,
  signal: AbortSignal,
): Promise<CascadeStep> {
  const check = await cascadeCheck(state, executionsBefore, projectRoot, hooks)
  return { ...check, ...await cascadeNext(check, hooks, signal) }
}

export interface CascadingTurnOptions {
  readonly hooks: CascadeHooks
  /** Where the checks read the files the turn wrote. */
  readonly projectRoot: string
  /** Called after every checked attempt, for the host to report. */
  readonly onStep?: (step: CascadeStep) => void
}

/**
 * Run one user turn through the shared loop with the cascade around it: after
 * each attempt the checks run, and while escalation hands back feedback the
 * model runs again in the same turn with it as a harness message. An attempt
 * that ended in a provider failure is not re-run.
 */
export async function* runCascadingTurn(
  request: TurnRequest,
  dependencies: TurnDependencies,
  options: CascadingTurnOptions,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const state = request.state
  const executionsBefore = state.toolExecutions.length
  const stepSignal = signal ?? new AbortController().signal
  const { userDisplayText: _display, userActor: _actor, ...followUp } = request
  let attempt: TurnRequest = request
  for (let round = 0; ; round += 1) {
    let failed = false
    for await (const event of runTurn(attempt, dependencies, signal)) {
      if (event.type === 'turn_done' && (event.reason === 'provider_failed' || event.reason === 'turn_failed')) failed = true
      yield event
    }
    if (stepSignal.aborted || failed) return
    const step = await cascadeStep(state, executionsBefore, options.projectRoot, options.hooks, stepSignal)
    options.onStep?.(step)
    if (step.feedback === undefined) {
      if (step.enforceMessage !== undefined) state.messages.push({ role: 'user', content: step.enforceMessage, origin: 'harness' })
      return
    }
    if (round + 1 >= MAX_CASCADE_ROUNDS) {
      // Past the backstop: the model reads it next turn.
      state.messages.push({ role: 'user', content: step.feedback, origin: 'harness' })
      return
    }
    const next: MessageContent = step.feedback
    attempt = { ...followUp, userMessage: next, userOrigin: 'harness' }
  }
}
