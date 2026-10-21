// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Preflight never starts agents or discovers credentials; the host supplies observations. */
export interface WorkflowPreflightRequest {
  readonly isolation?: 'worktree'
  readonly profile?: string
  readonly model?: string
}

export interface WorkflowPreflightPlan {
  readonly expected?: number
  readonly requests: readonly WorkflowPreflightRequest[]
}

type Observation<T> = T | Promise<T>

export interface WorkflowPreflightPorts {
  readonly worktreeAvailable?: () => Observation<boolean>
  /** Return "ok", or a useful error describing how to configure the selection. */
  readonly resolveModel?: (profile?: string, model?: string) => Observation<'ok' | string>
  readonly quota?: () => Observation<{ readonly exhausted: boolean; readonly detail: string } | undefined>
}

export interface WorkflowPreflightResult {
  readonly ok: boolean
  readonly problems: readonly string[]
  /** Missing ports are skipped explicitly, not represented as verified successes. */
  readonly notes: readonly string[]
  /** How many checks actually ran. 0 means `ok` verified nothing. */
  readonly checked: number
}

/** Check every available boundary before a high-fan-out workflow can spend money. */
export async function preflightWorkflow(plan: WorkflowPreflightPlan, ports: WorkflowPreflightPorts): Promise<WorkflowPreflightResult> {
  const problems: string[] = []
  const notes: string[] = []
  let checked = 0
  if (plan.expected !== undefined && (!Number.isSafeInteger(plan.expected) || plan.expected < 1)) problems.push('expect_agents must be a positive integer before starting a workflow.')
  if (plan.requests.some(request => request.isolation === 'worktree')) {
    if (!ports.worktreeAvailable) notes.push('Skipped worktree adapter check: the host did not supply worktreeAvailable; configure this preflight port to verify isolation before starting.')
    else {
      checked += 1
      try {
        if (!await ports.worktreeAvailable()) problems.push('Worktree isolation was requested, but no native worktree adapter is configured. Configure a worktree adapter or remove isolation: "worktree" before starting any agents.')
      } catch (error) { problems.push(`Cannot check the worktree adapter: ${errorText(error)}. Restore the adapter before starting agents.`) }
    }
  }
  if (!ports.resolveModel) notes.push('Skipped model/profile credential resolution: the host did not supply resolveModel; configure this preflight port to verify provider availability.')
  else {
    const selections = new Map<string, WorkflowPreflightRequest>()
    // An empty plan still uses the host's inherited model/profile selection.
    for (const request of plan.requests.length ? plan.requests : [{}]) selections.set(JSON.stringify([request.profile ?? null, request.model ?? null]), request)
    for (const { profile, model } of selections.values()) {
      checked += 1
      const selection = `profile ${JSON.stringify(profile ?? '(inherited)')}, model ${JSON.stringify(model ?? '(inherited)')}`
      try {
        const resolution = await ports.resolveModel(profile, model)
        if (resolution !== 'ok') problems.push(`Cannot resolve ${selection}: ${resolution || 'no resolution was returned'}. Configure the provider profile/model and its credentials before starting agents.`)
      } catch (error) { problems.push(`Cannot resolve ${selection}: ${errorText(error)}. Fix provider configuration before starting agents.`) }
    }
  }
  if (!ports.quota) notes.push('Skipped quota check: the host did not supply a quota reporter; remaining quota is unverified.')
  else {
    try {
      const quota = await ports.quota()
      if (quota) checked += 1
      if (!quota) notes.push('Skipped quota check: the usage reporter returned no quota window; remaining quota is unverified.')
      else if (quota.exhausted) problems.push(`Provider quota is exhausted: ${quota.detail}. Wait for the quota window to reset or select an available provider before starting agents.`)
    } catch (error) { problems.push(`Cannot check provider quota: ${errorText(error)}. Restore the usage reporter before starting agents.`) }
  }
  if (checked === 0 && problems.length === 0) notes.push('No preflight check ran: nothing was verified before starting agents.')
  return { ok: problems.length === 0, problems, notes, checked }
}

/**
 * Conservative static hints only: inspect JSON args and literal option objects,
 * never execute the script to discover a plan. Dynamic selections are still
 * validated by the normal spawn path, not guessed here.
 */
export function workflowPreflightPlan(input: { readonly script: string; readonly args?: unknown; readonly expected?: number }): WorkflowPreflightPlan {
  const requests: WorkflowPreflightRequest[] = []
  const seen = new Set<object>()
  const visit = (value: unknown): void => {
    if (!value || typeof value !== 'object' || seen.has(value)) return
    seen.add(value)
    if (Array.isArray(value)) { value.forEach(visit); return }
    const record = value as Record<string, unknown>
    const request = requestFromOptions(record)
    if (request) requests.push(request)
    Object.values(record).forEach(visit)
  }
  visit(input.args)
  const objects: Record<string, unknown>[] = []
  const tokens = scriptTokens(input.script)
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!
    if (token.text === '{') objects.push({})
    else if (token.text === '}') {
      const object = objects.pop()
      const request = object && requestFromOptions(object)
      if (request) requests.push(request)
    } else if (objects.length && tokens[index + 1]?.text === ':') {
      const key = token.literal ?? token.text
      if (!['isolation', 'model', 'profile', 'provider_profile'].includes(key)) continue
      const value = tokens[index + 2]
      if (value?.literal !== undefined && [',', '}'].includes(tokens[index + 3]?.text ?? '')) objects[objects.length - 1]![key] = value.literal
      else if (value?.text === 'args') {
        let resolved: unknown = input.args
        let cursor = index + 3
        while (tokens[cursor]?.text === '.' && tokens[cursor + 1]) {
          const field = tokens[cursor + 1]!.text
          resolved = resolved && typeof resolved === 'object' ? (resolved as Record<string, unknown>)[field] : undefined
          cursor += 2
        }
        if (typeof resolved === 'string' && [',', '}'].includes(tokens[cursor]?.text ?? '')) objects[objects.length - 1]![key] = resolved
      }
    }
  }
  return { ...(input.expected === undefined ? {} : { expected: input.expected }), requests }
}

function requestFromOptions(options: Record<string, unknown>): WorkflowPreflightRequest | undefined {
  const model = typeof options.model === 'string' ? options.model.trim() : ''
  const profileValue = options.profile ?? options.provider_profile
  const profile = typeof profileValue === 'string' ? profileValue.trim() : ''
  const isolation = options.isolation === 'worktree'
  if (!model && !profile && !isolation) return undefined
  return { ...(model ? { model } : {}), ...(profile ? { profile } : {}), ...(isolation ? { isolation: 'worktree' } : {}) }
}

interface ScriptToken { readonly text: string; readonly literal?: string }

/** Tokenize strings/comments separately so prompt prose cannot invent requests. */
function scriptTokens(script: string): ScriptToken[] {
  const tokens: ScriptToken[] = []
  const pattern = /\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`|[A-Za-z_$][\w$]*|[^\s]/g
  for (const match of script.matchAll(pattern)) {
    const text = match[0]
    if (text.startsWith('//') || text.startsWith('/*')) continue
    const quote = text[0]
    if (quote === '"' || quote === "'" || quote === '`') {
      let literal: string | undefined
      if (quote === '"') { try { literal = JSON.parse(text) as string } catch { /* Not a statically readable JSON string. */ } }
      else if (!text.includes('\\') && !(quote === '`' && text.includes('${'))) literal = text.slice(1, -1)
      tokens.push({ text, ...(literal === undefined ? {} : { literal }) })
    } else tokens.push({ text })
  }
  return tokens
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
