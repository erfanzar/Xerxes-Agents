// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { normalizeInteractionMode } from './interactionModes.js'

export { normalizeInteractionMode } from './interactionModes.js'

const DEFAULT_RETRY_LIMIT = 6

const VERIFIED_SUCCESS_MARKERS = [
  'acceptance criteria pass',
  'all acceptance criteria pass',
  'objective met',
  'verified complete',
  'verified completion',
  'all tests pass',
  'all benchmarks pass',
  'all checks pass',
] as const

const UNRESOLVED_MARKERS = [
  '❌',
  'not met',
  'unmet',
  'not yet',
  'still fails',
  'still failing',
  'still fail',
  'still loses',
  'still losing',
  'still lose',
  'cannot beat',
  "can't beat",
  'does not pass',
  'do not pass',
  'failing benchmark',
  'failing test',
  'remaining failure',
  'remaining issue',
  'losses',
  'loses by',
] as const

const RUNAWAY_FINAL_MARKERS = [
  'want me to',
  'should i',
  'would you like',
  'do you want me',
  'honest final',
  'final state',
  'where we stand',
  'where we are',
  'path forward is',
  'next step is',
] as const

const BLOCKER_MARKERS = [
  'blocked:',
  'concrete blocker',
  'externally blocked',
  'cannot proceed because',
] as const

const BLOCKER_EVIDENCE_MARKERS = [
  'evidence:',
  'command:',
  'stderr',
  'traceback',
  'permission denied',
  'not installed',
  'missing dependency',
  'requires user',
] as const

const EXPLICIT_VERIFICATION_TOOL_NAMES = new Set([
  'bench',
  'benchmark',
  'benchmark_tool',
  'build',
  'build_tool',
  'compile',
  'compile_tool',
  'lint',
  'lint_tool',
  'run_test',
  'run_test_tool',
  'run_tests',
  'run_tests_tool',
  'test',
  'test_tool',
  'typecheck',
  'typecheck_tool',
  'verification',
  'verification_tool',
  'verify',
  'verify_tool',
])

const MUTATING_TOOL_NAMES = new Set([
  'agent_memory_append',
  'agent_memory_journal',
  'agent_memory_learn',
  'agent_memory_write',
  'agent_tool',
  'append_file',
  'apply_patch',
  'copy_file',
  'delete_file',
  'enter_worktree_tool',
  'exit_worktree_tool',
  'file_edit_tool',
  'find_and_replace',
  'git_add',
  'git_apply_patch',
  'handoff_tool',
  'move_file',
  'notebook_edit_tool',
  'remote_trigger_tool',
  'reset_agent',
  'schedule_cron_tool',
  'send_input',
  'send_message_tool',
  'spawn_agent',
  'spawn_agents',
  'task_create_tool',
  'task_stop_tool',
  'task_update_tool',
  'web_click',
  'write',
  'write_file',
])

const VERIFICATION_SCRIPT_NAMES = /^(?:bench(?:mark)?|build|check|compile|lint|test|tests|typecheck|verify)(?::[\w.-]+)?$/
const DIRECT_VERIFICATION_COMMANDS = new Set([
  'ava',
  'biome',
  'eslint',
  'jest',
  'mypy',
  'nextest',
  'playwright',
  'pytest',
  'pyright',
  'ruff',
  'tsc',
  'vitest',
])

export interface ObjectiveGuardDecision {
  readonly reason: string
  readonly reminder: string
  readonly shouldContinue: boolean
}

export interface ObjectiveToolExecutionEvidence {
  readonly inputs: Readonly<Record<string, unknown>>
  readonly name: string
  readonly permitted: boolean
  readonly result: string
}

export interface ObjectiveGuardEvidence {
  /** Explicit verification supplied by the current turn's trusted runtime. */
  readonly verificationSignals?: readonly string[]
  /** Tool executions from the current turn only; prior session history must not be passed here. */
  readonly toolExecutions?: readonly ObjectiveToolExecutionEvidence[]
}

export interface ObjectiveGuardRetryOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
}

/** Return the configured objective retry ceiling with injectable environment lookup. */
export function objectiveGuardRetryLimit(
  config: Readonly<Record<string, unknown>>,
  options: ObjectiveGuardRetryOptions = {},
): number {
  const environment = options.environment ?? process.env
  const configured = config.objective_guard_max_retries
  const raw = configured ? configured : environment.XERXES_OBJECTIVE_GUARD_MAX_RETRIES
  const parsed = integerValue(raw)
  return parsed === undefined ? DEFAULT_RETRY_LIMIT : Math.max(1, parsed)
}

/**
 * Decide whether a no-tool answer must continue objective-mode execution.
 *
 * Objective mode may end only with an uncontradicted verified-success marker,
 * or a concrete blocker accompanied by observable evidence.
 */
export function inspectObjectiveResponse(
  text: string,
  options: {
    readonly evidence?: ObjectiveGuardEvidence
    readonly mode: unknown
    readonly planMode?: boolean
  },
): ObjectiveGuardDecision {
  if (normalizeInteractionMode(options.mode, options.planMode ?? false) !== 'objective') return allowStop()
  const stripped = text.trim()
  if (!stripped) return continueObjective('empty assistant response')

  const lowered = stripped.toLowerCase()
  const success = firstMarker(lowered, VERIFIED_SUCCESS_MARKERS)
  const unresolved = firstMarker(lowered, UNRESOLVED_MARKERS)
  if (success && !unresolved) {
    if (hasCurrentTurnVerification(options.evidence)) return allowStop()
    return continueObjective(
      'unsupported success claim ' + quote(success) + ' without current-turn verification evidence',
    )
  }

  const blocker = firstMarker(lowered, BLOCKER_MARKERS)
  const evidence = firstMarker(lowered, BLOCKER_EVIDENCE_MARKERS)
  if (blocker && evidence) {
    if (hasCurrentTurnBlockerEvidence(options.evidence)) return allowStop()
    return continueObjective(
      'unsupported blocker claim ' + quote(blocker) + ' without current-turn runtime failure evidence',
    )
  }

  const runaway = firstMarker(lowered, RUNAWAY_FINAL_MARKERS)
  const reason = unresolved
    ? 'unresolved acceptance marker ' + quote(unresolved)
    : runaway
      ? 'premature stopping marker ' + quote(runaway)
      : 'no verified completion or concrete blocker evidence'
  return continueObjective(reason)
}

function allowStop(): ObjectiveGuardDecision {
  return Object.freeze({ shouldContinue: false, reason: '', reminder: '' })
}

function continueObjective(reason: string): ObjectiveGuardDecision {
  return Object.freeze({
    shouldContinue: true,
    reason,
    reminder: objectiveReminder(reason),
  })
}

function hasCurrentTurnVerification(evidence: ObjectiveGuardEvidence | undefined): boolean {
  if (evidence?.verificationSignals?.some(signal => signal.trim().length > 0)) return true
  let verifiedAfterLastMutation = false
  for (const execution of evidence?.toolExecutions ?? []) {
    const outcome = toolExecutionOutcome(execution)
    if (execution.permitted && isMutatingExecution(execution)) verifiedAfterLastMutation = false
    if (outcome === 'success' && isVerificationExecution(execution)) verifiedAfterLastMutation = true
  }
  return verifiedAfterLastMutation
}

function hasCurrentTurnBlockerEvidence(evidence: ObjectiveGuardEvidence | undefined): boolean {
  return evidence?.toolExecutions?.some(execution => toolExecutionOutcome(execution) === 'failure') ?? false
}

function toolExecutionOutcome(execution: ObjectiveToolExecutionEvidence): 'failure' | 'pending' | 'success' {
  if (!execution.permitted) return 'failure'
  const lowered = execution.result.trim().toLowerCase()
  if (
    lowered.startsWith('tool execution failed:')
    || lowered.startsWith('cancelled before execution')
    || lowered.startsWith('denied by permission')
    || /^(?:error|fail(?:ed|ure)?)(?:\b|:)/.test(lowered)
    || /\b(?:tests?|checks?|verification)\s+fail(?:ed|ure)?\b/.test(lowered)
  ) return 'failure'

  const structured = parsedRecord(execution.result)
  if (!structured) return 'success'
  if (structured.running === true) return 'pending'
  const status = stringValue(structured.status).toLowerCase()
  if (
    structured.timedOut === true
    || structured.timed_out === true
    || structured.isError === true
    || structured.is_error === true
    || structured.ok === false
    || structured.passed === false
    || structured.success === false
    || ['cancelled', 'denied', 'error', 'failed', 'failure'].includes(status)
  ) return 'failure'
  const exitCode = structured.exitCode ?? structured.exit_code
  return typeof exitCode !== 'number' || exitCode === 0 ? 'success' : 'failure'
}

function isVerificationExecution(execution: ObjectiveToolExecutionEvidence): boolean {
  const name = normalizedToolName(execution.name)
  if (EXPLICIT_VERIFICATION_TOOL_NAMES.has(name)) return true
  if (name === 'exec_command') return isVerificationCommandInput(execution.inputs)
  if (name === 'bash') {
    return isVerificationShellCommand(stringValue(execution.inputs.command))
  }
  if (name !== 'write_stdin') return false

  const structured = parsedRecord(execution.result)
  if (!structured || structured.running === true) return false
  const exitCode = structured.exitCode ?? structured.exit_code
  return exitCode === 0 && isVerificationShellCommand(stringValue(structured.command))
}

function isMutatingExecution(execution: ObjectiveToolExecutionEvidence): boolean {
  const name = normalizedToolName(execution.name)
  if (MUTATING_TOOL_NAMES.has(name)) return true
  if (name === 'exec_command' || name === 'bash' || name === 'write_stdin') return true
  if (name === 'json_processor') return execution.inputs.operation === 'save'
  if (name === 'csv_processor') {
    return execution.inputs.operation === 'write' || execution.inputs.operation === 'convert'
  }
  if (name === 'skill_manage') {
    return ['create', 'delete', 'edit'].includes(stringValue(execution.inputs.intent))
  }
  return false
}

function isVerificationCommandInput(inputs: Readonly<Record<string, unknown>>): boolean {
  const command = stringValue(inputs.cmd)
  if (!command) return false
  const args = inputs.args
  if (Array.isArray(args) && args.every(argument => typeof argument === 'string')) {
    return isVerificationArgv(command, args)
  }
  return isVerificationShellCommand(command)
}

function isVerificationShellCommand(command: string): boolean {
  if (!command.trim()) return false
  return command
    .split(/&&|\|\||[;|\n]/)
    .map(segment => segment.trim())
    .filter(Boolean)
    .some(segment => {
      const words = segment.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)?.map(unquote) ?? []
      const executable = words[0]
      return executable !== undefined && isVerificationArgv(executable, words.slice(1))
    })
}

function isVerificationArgv(command: string, args: readonly string[]): boolean {
  const executable = command.replaceAll('\\', '/').split('/').at(-1)?.toLowerCase() ?? ''
  const normalizedArgs = args.map(argument => argument.trim()).filter(Boolean)
  if (DIRECT_VERIFICATION_COMMANDS.has(executable)) return true
  if (executable === 'go') return firstCommandArgument(normalizedArgs) === 'test'
  if (executable === 'cargo') {
    return ['bench', 'build', 'check', 'clippy', 'test'].includes(firstCommandArgument(normalizedArgs))
  }
  if (executable === 'dotnet') {
    return ['build', 'test'].includes(firstCommandArgument(normalizedArgs))
  }
  if (executable === 'mvn' || executable === 'mvnw' || executable === 'gradle' || executable === 'gradlew') {
    return normalizedArgs.some(argument => VERIFICATION_SCRIPT_NAMES.test(argument.toLowerCase()))
  }
  if (executable === 'make' || executable === 'ninja') {
    return normalizedArgs.some(argument => VERIFICATION_SCRIPT_NAMES.test(argument.toLowerCase()))
  }
  if (executable === 'deno') {
    const subcommand = firstCommandArgument(normalizedArgs)
    if (['bench', 'check', 'lint', 'test'].includes(subcommand)) return true
    return subcommand === 'task' && VERIFICATION_SCRIPT_NAMES.test(argumentAfter(normalizedArgs, 'task'))
  }
  if (executable === 'bun') {
    const subcommand = firstCommandArgument(normalizedArgs)
    if (subcommand === 'test') return true
    if (subcommand === 'x') return DIRECT_VERIFICATION_COMMANDS.has(argumentAfter(normalizedArgs, 'x'))
    return subcommand === 'run' && VERIFICATION_SCRIPT_NAMES.test(argumentAfter(normalizedArgs, 'run'))
  }
  if (executable === 'npm' || executable === 'pnpm' || executable === 'yarn') {
    const subcommand = firstCommandArgument(normalizedArgs)
    if (VERIFICATION_SCRIPT_NAMES.test(subcommand)) return true
    return subcommand === 'run' && VERIFICATION_SCRIPT_NAMES.test(argumentAfter(normalizedArgs, 'run'))
  }
  return false
}

function firstCommandArgument(args: readonly string[]): string {
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index] ?? ''
    if (argument === '--cwd' || argument === '--config' || argument === '--filter') {
      index += 1
      continue
    }
    if (!argument.startsWith('-')) return argument.toLowerCase()
  }
  return ''
}

function argumentAfter(args: readonly string[], marker: string): string {
  const index = args.findIndex(argument => argument.toLowerCase() === marker)
  if (index < 0) return ''
  return firstCommandArgument(args.slice(index + 1))
}

function normalizedToolName(name: string): string {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[.\s-]+/g, '_')
    .toLowerCase()
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function unquote(value: string): string {
  if (value.length < 2) return value
  const first = value[0]
  const last = value.at(-1)
  return (first === '"' && last === '"') || (first === "'" && last === "'")
    ? value.slice(1, -1)
    : value
}

function parsedRecord(text: string): Record<string, unknown> | undefined {
  try {
    const value: unknown = JSON.parse(text)
    return typeof value === 'object' && value !== null && !Array.isArray(value)
      ? value as Record<string, unknown>
      : undefined
  } catch {
    return undefined
  }
}

function firstMarker(text: string, markers: readonly string[]): string {
  for (const marker of markers) {
    if (text.includes(marker)) return marker
  }
  return ''
}

function integerValue(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isSafeInteger(value)) return value
  if (typeof value === 'string' && /^[-+]?\d+$/.test(value.trim())) {
    const parsed = Number(value)
    return Number.isSafeInteger(parsed) ? parsed : undefined
  }
  return undefined
}

function quote(value: string): string {
  return String.fromCharCode(96) + value + String.fromCharCode(96)
}

function objectiveReminder(reason: string): string {
  return '[Objective gate]\n'
    + 'The previous assistant response tried to stop, but objective mode is still active: ' + reason + '.\n'
    + 'Continue the hard-goal loop. Do not final-answer with a narrative status. Update the ledger, '
    + 'choose the next concrete hypothesis, use tools to edit or verify, and only end after all acceptance '
    + 'criteria pass or after you report BLOCKED: with exact evidence.'
}
