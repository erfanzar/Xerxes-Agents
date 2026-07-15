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

  run(point: HookPoint, payload: HookPayload = {}): unknown {
    const callbacks = this.hooks.get(point) ?? []
    if (MUTATION_HOOKS.has(point)) {
      const key = point === 'before_tool_call' ? 'arguments' : 'result'
      let value = payload[key]
      for (const callback of callbacks) {
        try {
          const result = callback(payload)
          if (isPromiseLike(result)) {
            void result.catch(() => undefined)
            continue
          }
          if (result !== undefined && result !== null) {
            value = result
            payload[key] = value
          }
        } catch {
          // Plugin hooks cannot prevent a tool call or persistence operation.
        }
      }
      return value
    }
    const results: unknown[] = []
    for (const callback of callbacks) {
      try {
        const result = callback(payload)
        if (isPromiseLike(result)) {
          void result.then(value => { if (value !== undefined && value !== null) results.push(value) }).catch(() => undefined)
        } else if (result !== undefined && result !== null) {
          results.push(result)
        }
      } catch {
        // Observer failure is intentionally isolated.
      }
    }
    return results
  }

  async runAsync(point: HookPoint, payload: HookPayload = {}): Promise<unknown> {
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
        } catch {
          // See synchronous path above.
        }
      }
      return value
    }
    const results: unknown[] = []
    for (const callback of callbacks) {
      try {
        const result = await callback(payload)
        if (result !== undefined && result !== null) results.push(result)
      } catch {
        // See synchronous path above.
      }
    }
    return results
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

function isPromiseLike(value: unknown): value is Promise<unknown> {
  return typeof value === 'object' && value !== null && typeof (value as { then?: unknown }).then === 'function'
}
