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

import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { AgentState, StreamEvent } from '../streaming/events.js'
import { runTurn, type TurnDependencies, type TurnRequest } from '../streaming/loop.js'
import { messageText, type MessageContent } from '../types/messages.js'
import {
  extractMutatedToolPaths,
  formatCascadeRollbackNotice,
  isReadOnlyInspectionTool,
  READ_ONLY_INSPECTION_TOOL_NAMES,
  WorkspaceCheckpointManager,
  type CascadeAttemptSnapshot,
  type CascadeRollbackResult,
  type FileStateSnapshot,
  type WorkspaceCheckpointOptions,
} from './cascadeCheckpoints.js'
import {
  renderFindings,
  runResultChecks,
  type CheckFinding,
  type ToolRunSummary,
} from './resultChecks.js'

export {
  extractMutatedToolPaths,
  formatCascadeRollbackNotice,
  isReadOnlyInspectionTool,
  READ_ONLY_INSPECTION_TOOL_NAMES,
  WorkspaceCheckpointManager,
  type CascadeAttemptSnapshot,
  type CascadeRollbackResult,
  type FileStateSnapshot,
  type WorkspaceCheckpointOptions,
}

/** Tools whose success means a file on disk changed and a checker could disagree. */
export const MUTATING_TOOL_NAMES: ReadonlySet<string> = new Set([
  'AppendFile', 'ApplyPatchTool', 'Edit', 'FileEditTool', 'MultiEdit', 'NotebookEdit', 'NotebookEditTool', 'Write', 'WriteFile',
  'append_file', 'apply_patch', 'apply_patch_tool', 'edit_file', 'file_edit_tool', 'notebook_edit_tool', 'replace_in_file', 'write_file',
])

/** Tools whose output file the result checks inspect, on top of {@link MUTATING_TOOL_NAMES}. */
const CHECKED_OUTPUT_TOOL_NAMES: ReadonlySet<string> = new Set(['XlsxWriter', 'XlsxConvert'])
const CSV_TOOL_NAMES: ReadonlySet<string> = new Set(['CSVProcessor', 'csv_processor'])
const SHELL_TOOL_NAMES: ReadonlySet<string> = new Set(['Bash', 'bash'])
const PROCESS_POLL_TOOL_NAMES: ReadonlySet<string> = new Set(['check_command', 'write_stdin'])

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
  /** Pre-turn code-check baseline fingerprints so pre-existing issues are not blamed on this turn. */
  readonly baselineFindings?: ReadonlySet<string>
  /** Per-turn checkpoint tracker used across cascade rounds. */
  readonly checkpoints?: WorkspaceCheckpointManager
  /** Restore touched files to the best prior attempt when a retry regresses. Default true when checkpoints are supplied. */
  readonly rollbackOnRegression?: boolean
}

/** What the checks and the escalation ladder said about one attempt. */
export interface CascadeCheck {
  readonly checksMode: ResultChecksMode
  readonly findings: readonly CheckFinding[]
  /** Paths the checked turn mutated across its tool executions. */
  readonly changedFiles?: readonly string[]
  /** The checks could not run; the attempt stands. */
  readonly checkError?: string
  readonly escalation?: CascadeEscalation
  readonly escalationError?: string
}

/** What happens next, once the advisor (if the ladder reached it) has answered. */
export interface CascadeNext {
  /** Set when the host consulted the advisor for this step. */
  readonly advisor?: { readonly answered: true } | { readonly answered: false; readonly error: string }
  /** Set when touched workspace files were restored to an earlier better-scoring attempt. */
  readonly rollback?: CascadeRollbackResult
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
  if (typeof message.content === 'string' || Array.isArray(message.content)) return messageText(message)
  return JSON.stringify(message.content)
}

function structuredExecutionVerdict(raw: string): { readonly ok: boolean; readonly command?: string } | undefined {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return undefined
    const record = parsed as Record<string, unknown>
    const command = typeof record.command === 'string' && record.command.trim() ? record.command.trim() : undefined
    if (
      record.running === true
      || record.timedOut === true
      || record.timed_out === true
      || record.ok === false
      || record.success === false
      || record.passed === false
      || record.isError === true
      || record.is_error === true
    ) {
      return { ok: false, ...(command === undefined ? {} : { command }) }
    }
    const exitCode = 'exitCode' in record ? record.exitCode : record.exit_code
    if (typeof exitCode === 'number') return { ok: exitCode === 0, ...(command === undefined ? {} : { command }) }
    if (exitCode === null) return { ok: false, ...(command === undefined ? {} : { command }) }
    return { ok: true, ...(command === undefined ? {} : { command }) }
  } catch {
    return undefined
  }
}

export interface TurnResultFindingsOptions {
  readonly baselineFindings?: ReadonlySet<string>
}

interface ParsedTurnExecutions {
  readonly changedFiles: string[]
  readonly toolRuns: ToolRunSummary[]
}

function collectTurnExecutions(state: AgentState, executionsBefore: number): ParsedTurnExecutions {
  const executions = state.toolExecutions.slice(executionsBefore)
  const changedFiles: string[] = []
  const toolRuns: ToolRunSummary[] = []
  for (const execution of executions) {
    const trimmedResult = execution.result.trimStart()
    const failed = !execution.permitted
      || trimmedResult.startsWith('Tool execution failed')
      || trimmedResult.startsWith('Cancelled before execution')
      || trimmedResult.startsWith('Denied by permission')
    const csvOperation = CSV_TOOL_NAMES.has(execution.name) && typeof execution.inputs.operation === 'string'
      ? execution.inputs.operation.trim()
      : ''
    const csvMutates = /^(?:write|convert)$/iu.test(csvOperation)
    if (
      !failed
      && !READ_ONLY_INSPECTION_TOOL_NAMES.has(execution.name)
      && (MUTATING_TOOL_NAMES.has(execution.name) || CHECKED_OUTPUT_TOOL_NAMES.has(execution.name) || csvMutates)
    ) {
      const extracted = extractMutatedToolPaths(execution.name, execution.inputs)
      for (const target of extracted) {
        changedFiles.push(target)
      }
    }
    let ok = !failed
    let command: string | undefined = csvOperation || undefined
    if (execution.name === 'exec_command') {
      const cmd = typeof execution.inputs.cmd === 'string' ? execution.inputs.cmd : ''
      const args = Array.isArray(execution.inputs.args) ? execution.inputs.args.filter((arg): arg is string => typeof arg === 'string') : []
      command = [cmd, ...args].join(' ').trim()
    } else if (SHELL_TOOL_NAMES.has(execution.name)) {
      const shellCommand = typeof execution.inputs.command === 'string' ? execution.inputs.command.trim() : ''
      if (shellCommand) command = shellCommand
    }
    if (execution.name === 'exec_command' || SHELL_TOOL_NAMES.has(execution.name) || PROCESS_POLL_TOOL_NAMES.has(execution.name)) {
      const structured = structuredExecutionVerdict(execution.result)
      if (structured !== undefined) {
        ok = ok && structured.ok
        if (!command && structured.command) command = structured.command
      }
    }
    toolRuns.push({ name: execution.name, ok, ...(command ? { command } : {}) })
  }
  return { changedFiles, toolRuns }
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
  options: TurnResultFindingsOptions = {},
): Promise<CheckFinding[]> {
  const { changedFiles, toolRuns } = collectTurnExecutions(state, executionsBefore)
  if (!changedFiles.length && !toolRuns.length) return []
  return runResultChecks({
    workspaceRoot: projectRoot,
    changedFiles,
    finalText: latestAssistantContent(state),
    toolRuns,
    ...(options.baselineFindings === undefined ? {} : { baselineFindings: options.baselineFindings }),
  })
}

/** The next harness message after the advisor answered a failed check. */
export function advisorAnswerFeedback(findings: readonly CheckFinding[], answer: string, rollbackNotice?: string): string {
  const lines = [
    ...(rollbackNotice === undefined ? [] : [rollbackNotice, '']),
    'Your result still fails these checks after local retries:',
    renderFindings(findings),
    '',
    'A stronger model (the advisor) reviewed the failure and answered:',
    answer.trim(),
    '',
    'Apply this advice. Fix exactly these problems, then run a check that proves the fix before you report the task as done.',
  ]
  return lines.join('\n')
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function mergeBaselines(
  first: ReadonlySet<string> | undefined,
  second: ReadonlySet<string> | undefined,
): ReadonlySet<string> | undefined {
  if (first === undefined || first.size === 0) return second
  if (second === undefined || second.size === 0) return first
  const merged = new Set<string>(first)
  for (const item of second) merged.add(item)
  return merged
}

function appendRollbackNotice(text: string, rollback: CascadeRollbackResult | undefined): string {
  if (rollback === undefined) return text
  const bestSummary = rollback.bestFindings.length > 0
    ? `\nRemaining check failure(s) on the restored workspace:\n${renderFindings(rollback.bestFindings)}`
    : ''
  return `${text}\n\n${rollback.notice}${bestSummary}`
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
  let changedFiles: readonly string[] | undefined
  let checkError: string | undefined
  try {
    checksMode = hooks.checks()
    if (checksMode !== 'off') {
      const collected = collectTurnExecutions(state, executionsBefore)
      changedFiles = collected.changedFiles
      const baselineFindings = mergeBaselines(hooks.checkpoints?.baselineFindings, hooks.baselineFindings)
      const turnFindings = (!collected.changedFiles.length && !collected.toolRuns.length)
        ? []
        : await runResultChecks({
          workspaceRoot: projectRoot,
          changedFiles: collected.changedFiles,
          finalText: latestAssistantContent(state),
          toolRuns: collected.toolRuns,
          ...(baselineFindings === undefined ? {} : { baselineFindings }),
        })
      findings = [
        ...turnFindings,
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
    ...(changedFiles === undefined || changedFiles.length === 0 ? {} : { changedFiles }),
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
  let rollback: CascadeRollbackResult | undefined
  if (hooks.checkpoints !== undefined && findings.some(finding => finding.severity === 'error')) {
    try {
      rollback = await hooks.checkpoints.recordAttemptAndMaybeRollback(findings, {
        ...(check.changedFiles === undefined ? {} : { additionalTouchedPaths: check.changedFiles }),
        ...(escalation?.action === undefined ? {} : { escalationAction: escalation.action }),
        rollbackEnabled: hooks.rollbackOnRegression ?? true,
      })
    } catch {
      // Checkpoint I/O failure must never abort a turn's escalation step.
    }
  }

  const rollbackSlice = rollback === undefined ? {} : { rollback }

  if (findings.length > 0 && escalation?.action === 'ask_advisor' && escalation.advisorRequest && hooks.consultAdvisor) {
    // Escalation by rule: the host consults the advisor itself rather than
    // trusting a small model to make the call.
    const advisorRequest: AdvisorRequest = rollback === undefined
      ? escalation.advisorRequest
      : {
        question: escalation.advisorRequest.question,
        context: `${escalation.advisorRequest.context}\n\n${rollback.notice}`,
      }
    try {
      const answer = await hooks.consultAdvisor(advisorRequest, signal)
      return {
        ...rollbackSlice,
        advisor: { answered: true },
        feedback: advisorAnswerFeedback(findings, answer, rollback?.notice),
      }
    } catch (error) {
      if (signal.aborted) throw error
      return {
        ...rollbackSlice,
        advisor: { answered: false, error: message(error) },
        feedback: appendRollbackNotice(
          `${renderFindings(findings)}\nThe advisor could not be reached. Fix these problems yourself, then run a check that proves the fix before you report the task as done.`,
          rollback,
        ),
      }
    }
  }
  const feedback = findings.length ? escalation?.feedback : undefined
  if (feedback !== undefined) return { ...rollbackSlice, feedback: appendRollbackNotice(feedback, rollback) }
  if (check.checksMode === 'enforce' && escalation === undefined && findings.some(finding => finding.severity === 'error')) {
    return {
      ...rollbackSlice,
      enforceMessage: appendRollbackNotice(`${renderFindings(findings)}\nFix these before you report the task as done.`, rollback),
    }
  }
  return { ...rollbackSlice }
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
  /** Snapshot touched files and restore the best prior attempt on regression. Default true. */
  readonly rollbackOnRegression?: boolean
  /** Optional external checkpoint manager; when omitted, one is created per cascading turn. */
  readonly checkpoints?: WorkspaceCheckpointManager
  /** Optional pre-turn baseline fingerprints merged with captured baselines. */
  readonly baselineFindings?: ReadonlySet<string>
}

function wrapCheckpointToolExecutor(
  executor: ToolExecutor | undefined,
  checkpoints: WorkspaceCheckpointManager,
): ToolExecutor | undefined {
  if (executor === undefined) return undefined
  return {
    async execute(call, context, signal) {
      try {
        await checkpoints.noteBeforeToolExecution(call.function.name, call.function.arguments)
      } catch {
        // Pre-mutation snapshotting is best-effort and never blocks tool execution.
      }
      return executor.execute(call, context, signal)
    },
  }
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
  const rollbackOnRegression = options.rollbackOnRegression ?? options.hooks.rollbackOnRegression ?? true
  const checkpoints = options.checkpoints
    ?? options.hooks.checkpoints
    ?? new WorkspaceCheckpointManager(options.projectRoot, {
      ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }),
    })
  try {
    await checkpoints.beginTurn()
  } catch {
    // Non-existent or unreadable workspace roots stay non-fatal.
  }

  const wrappedExecutor = wrapCheckpointToolExecutor(dependencies.toolExecutor, checkpoints)
  const effectiveDependencies: TurnDependencies = wrappedExecutor === undefined
    ? dependencies
    : { ...dependencies, toolExecutor: wrappedExecutor }
  const mergedExplicitBaseline = mergeBaselines(options.baselineFindings, options.hooks.baselineFindings)
  const effectiveHooks: CascadeHooks = {
    ...options.hooks,
    checkpoints,
    rollbackOnRegression,
    ...(mergedExplicitBaseline === undefined ? {} : { baselineFindings: mergedExplicitBaseline }),
  }

  const { userDisplayText: _display, userActor: _actor, ...followUp } = request
  let attempt: TurnRequest = request
  for (let round = 0; ; round += 1) {
    let failed = false
    for await (const event of runTurn(attempt, effectiveDependencies, signal)) {
      if (event.type === 'turn_done' && (event.reason === 'provider_failed' || event.reason === 'turn_failed')) failed = true
      yield event
    }
    if (stepSignal.aborted || failed) return
    let step: CascadeStep
    try {
      step = await cascadeStep(state, executionsBefore, options.projectRoot, effectiveHooks, stepSignal)
    } catch (error) {
      if (stepSignal.aborted) return
      throw error
    }
    options.onStep?.(step)
    if (step.feedback === undefined) {
      if (!stepSignal.aborted && step.enforceMessage !== undefined) {
        state.messages.push({ role: 'user', content: step.enforceMessage, origin: 'harness' })
      }
      return
    }
    if (stepSignal.aborted || round + 1 >= MAX_CASCADE_ROUNDS) {
      // Cancelled during checks or past the backstop: the model reads it next turn.
      state.messages.push({ role: 'user', content: step.feedback, origin: 'harness' })
      return
    }
    const next: MessageContent = step.feedback
    attempt = { ...followUp, userMessage: next, userOrigin: 'harness' }
  }
}
