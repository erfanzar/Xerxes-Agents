// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const HOOK_POINTS = [
  'before_tool_call',
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

const MUTATION_HOOKS = new Set<HookPoint>(['before_tool_call', 'after_tool_call', 'tool_result_persist'])

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
        if (result !== undefined && result !== null) results.push(result)
      } catch (error) {
        reportHookFailure(point, error)
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

/** Keep a hook failure observable without letting it prevent a tool call or persistence operation. */
function reportHookFailure(point: HookPoint, error: unknown): void {
  const detail = error instanceof Error ? error.message : String(error)
  console.error(`Hook '${point}' callback failed: ${detail}`)
}
