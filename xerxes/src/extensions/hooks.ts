// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const HOOK_POINTS = [
  'before_tool_call',
  'tool_permission_check',
  'after_tool_call',
  'tool_result_persist',
  'bootstrap_files',
  'on_turn_start',
  'on_turn_end',
  'on_loop_warning',
  'on_error',
] as const

export type HookPoint = (typeof HOOK_POINTS)[number]
export type HookPayload = Record<string, unknown>
export type HookCallback = (payload: HookPayload) => unknown | Promise<unknown>

/**
 * Hook point at which an extension may refuse a tool call outright.
 *
 * `before_tool_call` cannot do this: it is a mutation hook, so its return value
 * is threaded back as the tool's arguments and a throw is swallowed while the
 * loop proceeds — an extension had no way to say no. This point is deliberately
 * kept out of {@link MUTATION_HOOKS} so the collect-results branch returns the
 * full ordered list of verdicts instead of one threaded value.
 */
export const TOOL_PERMISSION_HOOK = 'tool_permission_check'

const MUTATION_HOOKS = new Set<HookPoint>(['before_tool_call', 'after_tool_call', 'tool_result_persist'])

/**
 * Points where a callback failure must produce a denial rather than silence.
 *
 * Everywhere else an isolated hook failure is the safe outcome. Here it is the
 * dangerous one: a permission hook that throws has expressed no opinion, and
 * treating "no opinion" as consent lets a crashing guard wave a call through.
 */
const FAIL_CLOSED_HOOKS = new Set<HookPoint>([TOOL_PERMISSION_HOOK])

/** Safe hook dispatcher: observer failures are isolated from agent execution. */
export class HookRunner {
  private readonly hooks = new Map<HookPoint, HookCallback[]>(HOOK_POINTS.map(point => [point, []]))

  clear(point?: HookPoint): void {
    if (point) {
      this.hooks.set(point, [])
      return
    }
    for (const hookPoint of HOOK_POINTS) this.hooks.set(hookPoint, [])
  }

  hasHooks(point: HookPoint): boolean {
    return (this.hooks.get(point)?.length ?? 0) > 0
  }

  register(point: HookPoint, callback: HookCallback): void {
    const callbacks = this.hooks.get(point)
    if (!callbacks) throw new Error(`Unknown hook point '${point}'. Valid: ${HOOK_POINTS.join(', ')}`)
    callbacks.push(callback)
  }

  /**
   * Run every callback for a point, awaiting asynchronous hooks before resolving.
   * Mutation hooks thread their returned value through the payload; observer hooks
   * resolve to the complete ordered list of non-empty results.
   */
  async run(point: HookPoint, payload: HookPayload = {}): Promise<unknown> {
    const callbacks = this.hooks.get(point) ?? []
    if (MUTATION_HOOKS.has(point)) {
      const key = point === 'before_tool_call' ? 'arguments' : 'result'
      let value = payload[key]
      for (const callback of callbacks) {
        try {
          const result = await callback(payload)
          if (result !== undefined && result !== null) {
            value = result
            payload[key] = value
          }
        } catch (error) {
          reportHookFailure(point, error)
        }
      }
      return value
    }
    const results: unknown[] = []
    for (const callback of callbacks) {
      try {
        const result = await callback(payload)
        if (result !== undefined && result !== null) {
          results.push(result)
        } else if (FAIL_CLOSED_HOOKS.has(point)) {
          results.push(undefined)
        }
      } catch (error) {
        reportHookFailure(point, error)
        if (FAIL_CLOSED_HOOKS.has(point)) results.push(denialFromFailure(point, error))
      }
    }
    return results
  }

  /** Alias of {@link run} retained for existing asynchronous dispatch sites. */
  async runAsync(point: HookPoint, payload: HookPayload = {}): Promise<unknown> {
    return this.run(point, payload)
  }

  unregister(point: HookPoint, callback: HookCallback): boolean {
    const callbacks = this.hooks.get(point)
    if (!callbacks) return false
    const index = callbacks.indexOf(callback)
    if (index < 0) return false
    callbacks.splice(index, 1)
    return true
  }
}

/** One extension's opinion on whether a tool call may proceed. */
export interface ToolPermissionVerdict {
  readonly allow: boolean
  /** Surfaced to the model and the user on a denial; ignored when the verdict allows. */
  readonly reason?: string
  /** Optional extension or rule identifier, so a denial can be traced to its author. */
  readonly source?: string
  /**
   * Replacement tool arguments proposed by an allowing verdict. A verdict may
   * tighten what runs, never loosen it: denials ignore this field entirely.
   */
  readonly updatedArguments?: HookPayload
}

/** Collapsed result of every registered permission hook for one tool call. */
export interface ToolPermissionDecision {
  readonly allowed: boolean
  /** Every denial in registration order; a host may log all of them, not just the first. */
  readonly denials: readonly ToolPermissionVerdict[]
  /** Empty when allowed, otherwise the first denial's reason. */
  readonly reason: string
  /** First allowing verdict's replacement arguments, when any verdict proposed them. */
  readonly updatedArguments?: HookPayload
}

/**
 * Ask every registered permission hook whether one tool call may run.
 *
 * Resolution rule: no verdicts means allow (the point is opt-in, and an agent
 * with no guards installed must keep working), any deny wins regardless of
 * position, and a hook that throws or returns something that is not a verdict
 * is counted as a denial. A guard that cannot answer is not a guard that
 * consents — an exception thrown by a policy extension used to vanish into
 * `reportHookFailure` and the call ran anyway.
 */
export async function resolveToolPermission(
  hooks: HookRunner,
  input: { readonly arguments?: HookPayload; readonly toolName: string },
): Promise<ToolPermissionDecision> {
  const payload: HookPayload = { toolName: input.toolName, arguments: input.arguments ?? {} }
  const results = await hooks.run(TOOL_PERMISSION_HOOK, payload)
  const denials: ToolPermissionVerdict[] = []
  for (const result of Array.isArray(results) ? results : [results]) {
    const verdict = asPermissionVerdict(result)
    if (verdict === undefined) {
      denials.push({
        allow: false,
        source: TOOL_PERMISSION_HOOK,
        reason: `a ${TOOL_PERMISSION_HOOK} hook returned a value that is not a permission verdict`,
      })
      continue
    }
    if (!verdict.allow) denials.push(verdict)
  }
  const first = denials[0]
  const allowed = first === undefined
  const updatedArguments = allowed
    ? firstAllowingUpdate(Array.isArray(results) ? results : [results])
    : undefined
  return {
    denials,
    allowed,
    reason: first === undefined ? '' : first.reason?.trim() || `tool '${input.toolName}' denied by a permission hook`,
    ...(updatedArguments === undefined ? {} : { updatedArguments }),
  }
}

/** The first well-formed allowing verdict carrying replacement arguments wins. */
function firstAllowingUpdate(results: readonly unknown[]): HookPayload | undefined {
  for (const result of results) {
    const verdict = asPermissionVerdict(result)
    if (verdict?.allow === true && verdict.updatedArguments !== undefined) {
      return verdict.updatedArguments
    }
  }
  return undefined
}

function asPermissionVerdict(value: unknown): ToolPermissionVerdict | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  const candidate = value as { allow?: unknown; reason?: unknown; source?: unknown; updatedArguments?: unknown }
  if (typeof candidate.allow !== 'boolean') return undefined
  const updatedArguments = typeof candidate.updatedArguments === 'object'
    && candidate.updatedArguments !== null
    && !Array.isArray(candidate.updatedArguments)
    ? candidate.updatedArguments as HookPayload
    : undefined
  return {
    allow: candidate.allow,
    ...(typeof candidate.reason === 'string' ? { reason: candidate.reason } : {}),
    ...(typeof candidate.source === 'string' ? { source: candidate.source } : {}),
    ...(updatedArguments === undefined ? {} : { updatedArguments }),
  }
}

function denialFromFailure(point: HookPoint, error: unknown): ToolPermissionVerdict {
  const detail = error instanceof Error ? error.message : String(error)
  return { allow: false, source: point, reason: `permission hook failed: ${detail}` }
}

/** Keep a hook failure observable without letting it prevent a tool call or persistence operation. */
function reportHookFailure(point: HookPoint, error: unknown): void {
  const detail = error instanceof Error ? error.message : String(error)
  console.error(`Hook '${point}' callback failed: ${detail}`)
}
