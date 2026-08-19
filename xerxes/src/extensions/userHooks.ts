// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import {
  TOOL_PERMISSION_HOOK,
  type HookPayload,
  type HookPoint,
  type HookRunner,
  type ToolPermissionVerdict,
} from './hooks.js'

/**
 * User-defined external hooks, configured through daemon settings rather than
 * compiled into an extension. Each definition names a host command that Xerxes
 * runs at a fixed lifecycle point. The command receives one JSON document on
 * stdin describing the event and answers through its exit code, optionally
 * refined by a JSON verdict on stdout:
 *
 * - exit 0 allows the operation (observer events stop here);
 * - exit 2 denies it, with stderr (or stdout) text as the reason;
 * - any other exit, a spawn failure, or a timeout is a hook error.
 *
 * PreToolUse errors fail closed: a guard that cannot answer has expressed no
 * opinion, and silence must never read as consent. Observer events
 * (PostToolUse, TurnStart, TurnEnd, OnError) only log the failure.
 *
 * Stdout verdict (all fields optional):
 *
 * ```json
 * { "decision": "allow" | "deny", "reason": "...", "updated_arguments": { } }
 * ```
 *
 * Commands run with the user's own privileges; they are a policy surface, not
 * a sandbox. Keep guarding hooks fast — every PreToolUse definition runs
 * sequentially inside the turn's permission phase.
 */
export const USER_HOOK_EVENTS = ['PreToolUse', 'PostToolUse', 'TurnStart', 'TurnEnd', 'OnError'] as const

export type UserHookEvent = (typeof USER_HOOK_EVENTS)[number]

export interface UserHookDefinition {
  /** Shell command line, or the executable when `args` is provided. */
  readonly command: string
  /** Argv tail; when present the command is spawned directly without a shell. */
  readonly args?: readonly string[]
  /** Regular expression matched against the tool name; absent matches everything. */
  readonly matcher?: string
  /** Per-invocation ceiling in milliseconds; defaults to {@link DEFAULT_HOOK_TIMEOUT_MS}. */
  readonly timeoutMs?: number
}

export interface UserHooksConfig {
  readonly PreToolUse?: readonly UserHookDefinition[]
  readonly PostToolUse?: readonly UserHookDefinition[]
  readonly TurnStart?: readonly UserHookDefinition[]
  readonly TurnEnd?: readonly UserHookDefinition[]
  readonly OnError?: readonly UserHookDefinition[]
}

export const DEFAULT_HOOK_TIMEOUT_MS = 5_000
export const MAX_HOOK_TIMEOUT_MS = 60_000
/** Source tag attached to verdicts so a denial can be traced to this surface. */
export const USER_HOOK_SOURCE = 'user-hook'

/** Resolved verdict of one PreToolUse definition. */
export type UserHookOutcome =
  | { readonly kind: 'allow'; readonly updatedArguments?: HookPayload }
  | { readonly kind: 'deny'; readonly reason: string }

export interface UserHookExecutionOptions {
  /** Working directory for the spawned command; also exported as XERXES_PROJECT_DIR. */
  readonly cwd?: string
  /** Base environment; defaults to the daemon's own process environment. */
  readonly environment?: Readonly<Record<string, string | undefined>>
  /** Diagnostic sink for observer-event failures; defaults to console.error. */
  readonly onError?: (message: string) => void
}

export interface UserHookEventPayload {
  readonly event: UserHookEvent
  readonly cwd?: string
  readonly timestamp: string
  readonly [key: string]: unknown
}

/** Validate an untrusted settings value into a hook configuration. */
export function parseUserHooksConfig(input: unknown): UserHooksConfig {
  if (input === undefined || input === null) return {}
  if (typeof input !== 'object' || Array.isArray(input)) {
    throw new ValidationError('hooks', 'must be an object keyed by hook event', input)
  }
  const record = input as Record<string, unknown>
  const config: Record<string, readonly UserHookDefinition[]> = {}
  for (const [key, value] of Object.entries(record)) {
    if (!(USER_HOOK_EVENTS as readonly string[]).includes(key)) {
      throw new ValidationError('hooks', `unknown hook event '${key}'; valid: ${USER_HOOK_EVENTS.join(', ')}`, value)
    }
    config[key] = parseDefinitionList(key, value)
  }
  return config as UserHooksConfig
}

/** Whether any event has at least one definition; an empty config registers nothing. */
export function hasUserHooks(config: UserHooksConfig): boolean {
  return USER_HOOK_EVENTS.some(event => (config[event]?.length ?? 0) > 0)
}

/**
 * Run one definition against one event payload.
 *
 * Observer events discard the outcome kind's distinction; callers that only
 * observe should still await this so a hanging hook cannot outlive its turn.
 */
export async function executeUserHook(
  definition: UserHookDefinition,
  event: UserHookEvent,
  payload: UserHookEventPayload,
  options: UserHookExecutionOptions = {},
): Promise<UserHookOutcome> {
  const timeoutMs = clampTimeout(definition.timeoutMs)
  const cwd = options.cwd ?? payload.cwd
  const argv = definition.args === undefined
    ? shellArgv(definition.command)
    : [definition.command, ...definition.args]
  let child: Bun.PipedSubprocess
  try {
    child = Bun.spawn([...argv], {
      ...(cwd === undefined ? {} : { cwd }),
      env: {
        ...(options.environment ?? process.env),
        XERXES_HOOK_EVENT: event,
        ...(cwd === undefined ? {} : { XERXES_PROJECT_DIR: cwd }),
      },
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
    }) as Bun.PipedSubprocess
  } catch (error) {
    return hookError(`could not start: ${detail(error)}`)
  }
  child.stdin.write(JSON.stringify(payload))
  child.stdin.end()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill()
  }, timeoutMs)
  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ])
    if (timedOut) {
      return hookError(`timed out after ${timeoutMs}ms`)
    }
    return verdictFromOutput(exitCode, stdout, stderr)
  } catch (error) {
    return hookError(detail(error))
  } finally {
    clearTimeout(timer)
  }
}

/**
 * Register one HookRunner callback per configured definition.
 *
 * PreToolUse definitions become `tool_permission_check` verdicts; every other
 * event maps onto the matching observer point. Returns the per-point count so
 * the composing host can report what was installed.
 */
export function registerUserHooks(
  hookRunner: HookRunner,
  config: UserHooksConfig,
  options: UserHookExecutionOptions = {},
): Readonly<Partial<Record<HookPoint, number>>> {
  const counts: Partial<Record<HookPoint, number>> = {}
  for (const definition of config.PreToolUse ?? []) {
    hookRunner.register(TOOL_PERMISSION_HOOK, async payload => {
      // A matcher miss is "no opinion", but the fail-closed runner counts an
      // undefined return at this point as a malformed verdict — which denies.
      // Answer explicitly instead.
      if (!matches(definition, payload.toolName)) {
        return { allow: true, source: USER_HOOK_SOURCE }
      }
      const outcome = await executeUserHook(definition, 'PreToolUse', {
        event: 'PreToolUse',
        tool_name: payload.toolName,
        arguments: payload.arguments ?? {},
        timestamp: new Date().toISOString(),
        ...(typeof payload.cwd === 'string' ? { cwd: payload.cwd } : {}),
      }, options)
      return verdictFromOutcome(outcome)
    })
    counts[TOOL_PERMISSION_HOOK] = (counts[TOOL_PERMISSION_HOOK] ?? 0) + 1
  }
  const observers: ReadonlyArray<readonly [UserHookEvent, HookPoint]> = [
    ['PostToolUse', 'after_tool_call'],
    ['TurnStart', 'on_turn_start'],
    ['TurnEnd', 'on_turn_end'],
    ['OnError', 'on_error'],
  ]
  for (const [event, point] of observers) {
    for (const definition of config[event] ?? []) {
      hookRunner.register(point, async payload => {
        // Matchers scope tool events by tool name; lifecycle events have no
        // tool to match, so a matcher there must not silence the hook.
        if (event === 'PostToolUse' && !matches(definition, payload.name ?? payload.toolName)) {
          return undefined
        }
        try {
          await executeUserHook(definition, event, {
            ...payload,
            event,
            timestamp: new Date().toISOString(),
          }, options)
        } catch (error) {
          ;(options.onError ?? console.error)(`User hook '${event}' failed: ${detail(error)}`)
        }
        return undefined
      })
      counts[point] = (counts[point] ?? 0) + 1
    }
  }
  return Object.freeze(counts)
}

function verdictFromOutcome(outcome: UserHookOutcome): ToolPermissionVerdict {
  if (outcome.kind === 'deny') {
    return { allow: false, reason: outcome.reason, source: USER_HOOK_SOURCE }
  }
  return {
    allow: true,
    source: USER_HOOK_SOURCE,
    ...(outcome.updatedArguments === undefined ? {} : { updatedArguments: outcome.updatedArguments }),
  }
}

function matches(definition: UserHookDefinition, toolName: unknown): boolean {
  if (definition.matcher === undefined) return true
  if (typeof toolName !== 'string') return false
  return new RegExp(definition.matcher).test(toolName)
}

function verdictFromOutput(exitCode: number, stdout: string, stderr: string): UserHookOutcome {
  if (exitCode === 2) {
    return { kind: 'deny', reason: nonEmpty(stderr) ?? nonEmpty(stdout) ?? 'denied by user hook (exit 2)' }
  }
  if (exitCode !== 0) {
    return hookError(`exited with code ${exitCode}${nonEmpty(stderr) ? `: ${stderr.trim()}` : ''}`)
  }
  const verdict = parseStdoutVerdict(stdout)
  if (verdict === undefined) return { kind: 'allow' }
  if (verdict.decision === 'deny') {
    return { kind: 'deny', reason: verdict.reason ?? 'denied by user hook verdict' }
  }
  return {
    kind: 'allow',
    ...(verdict.updatedArguments === undefined ? {} : { updatedArguments: verdict.updatedArguments }),
  }
}

interface StdoutVerdict {
  readonly decision?: 'allow' | 'deny'
  readonly reason?: string
  readonly updatedArguments?: HookPayload
}

/** A stdout document is a verdict only when it is a JSON object with a known field. */
function parseStdoutVerdict(stdout: string): StdoutVerdict | undefined {
  const text = stdout.trim()
  if (!text.startsWith('{')) return undefined
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    return undefined
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const record = value as Record<string, unknown>
  const decision = record.decision === 'allow' || record.decision === 'deny' ? record.decision : undefined
  const reason = typeof record.reason === 'string' && record.reason.trim() ? record.reason.trim() : undefined
  const updatedArguments = isPlainObject(record.updated_arguments)
    ? record.updated_arguments as HookPayload
    : undefined
  if (decision === undefined && reason === undefined && updatedArguments === undefined) return undefined
  return {
    ...(decision === undefined ? {} : { decision }),
    ...(reason === undefined ? {} : { reason }),
    ...(updatedArguments === undefined ? {} : { updatedArguments }),
  }
}

/** PreToolUse never confuses a broken guard for consent; observers discard this outcome. */
function hookError(reason: string): UserHookOutcome {
  return { kind: 'deny', reason: `user hook failed: ${reason}` }
}

function parseDefinitionList(event: string, value: unknown): readonly UserHookDefinition[] {
  if (!Array.isArray(value)) {
    throw new ValidationError('hooks', `event '${event}' must be an array of hook definitions`, value)
  }
  return value.map((entry, index) => parseDefinition(`${event}[${index}]`, entry))
}

function parseDefinition(path: string, value: unknown): UserHookDefinition {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ValidationError('hooks', `${path} must be an object with a command`, value)
  }
  const record = value as Record<string, unknown>
  const command = record.command
  if (typeof command !== 'string' || !command.trim()) {
    throw new ValidationError('hooks', `${path}.command must be a non-empty string`, command)
  }
  const args = record.args
  if (args !== undefined && (!Array.isArray(args) || args.some(argument => typeof argument !== 'string'))) {
    throw new ValidationError('hooks', `${path}.args must be an array of strings`, args)
  }
  const matcher = record.matcher
  if (matcher !== undefined) {
    if (typeof matcher !== 'string' || !matcher.trim()) {
      throw new ValidationError('hooks', `${path}.matcher must be a non-empty regular expression`, matcher)
    }
    try {
      new RegExp(matcher)
    } catch (error) {
      throw new ValidationError('hooks', `${path}.matcher is not a valid regular expression: ${detail(error)}`, matcher)
    }
  }
  const timeoutMs = record.timeoutMs ?? record.timeout_ms
  if (timeoutMs !== undefined && (typeof timeoutMs !== 'number' || !Number.isFinite(timeoutMs) || timeoutMs <= 0)) {
    throw new ValidationError('hooks', `${path}.timeoutMs must be a positive number of milliseconds`, timeoutMs)
  }
  return Object.freeze({
    command: command.trim(),
    ...(args === undefined ? {} : { args: Object.freeze([...args as string[]]) }),
    ...(matcher === undefined ? {} : { matcher: matcher.trim() }),
    ...(timeoutMs === undefined ? {} : { timeoutMs: clampTimeout(timeoutMs as number) }),
  })
}

function shellArgv(command: string): readonly string[] {
  return process.platform === 'win32'
    ? ['cmd.exe', '/d', '/s', '/c', command]
    : ['sh', '-c', command]
}

function clampTimeout(value: number | undefined): number {
  if (value === undefined) return DEFAULT_HOOK_TIMEOUT_MS
  return Math.min(Math.max(Math.floor(value), 1), MAX_HOOK_TIMEOUT_MS)
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nonEmpty(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

function detail(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
