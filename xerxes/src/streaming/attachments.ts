// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { messageText, type ChatMessage } from '../types/messages.js'

/**
 * Everything the runtime is allowed to push into a live transcript on its own
 * behalf, as one closed union.
 *
 * Before this existed the loop had exactly one hardcoded injector (sub-agent
 * events) that was unmarked, unbudgeted and undeduplicated. Each further
 * feature that wants to speak into the transcript — external file changes,
 * post-compaction rehydration, todo reminders, deferred-tool deltas — would
 * otherwise invent its own push site and none of them would be bounded. Adding
 * a variant here is the only way to get a new injection, which is what keeps
 * the budget below authoritative.
 */
export type ContextInjection =
  | { readonly events: readonly string[]; readonly kind: 'agent_events' }
  | { readonly kind: 'compaction_rehydration'; readonly text: string }
  | { readonly kind: 'deferred_tools'; readonly names: readonly string[] }
  | { readonly kind: 'external_file_changes'; readonly paths: readonly string[] }
  | { readonly kind: 'todo_reminder'; readonly text: string }

export type ContextInjectionKind = ContextInjection['kind']

/**
 * Ceiling on one injected body. Sub-agent results are the largest thing that
 * travels this path and they are already bounded upstream; this is the backstop
 * for a producer that forgets to bound its own output.
 */
export const MAX_INJECTION_CHARACTERS = 16_000
/**
 * Ceiling on everything injected since the last real user message. Without it a
 * turn that keeps draining events grows the request by an injection per round
 * until the context window is the thing that fails.
 */
export const MAX_TURN_INJECTION_CHARACTERS = 48_000
/**
 * Below this many characters of remaining turn budget an injection is dropped
 * rather than shaved to a stub: a two-line fragment of a sub-agent report costs
 * a round trip and tells the model nothing it can act on.
 */
export const MIN_INJECTION_CHARACTERS = 512

interface InjectionSpec {
  /** First body line; also the scan signature for unwrapped kinds. */
  readonly heading: string
  /** Repeat throttle: occurrences of this kind tolerated since the last user message. */
  readonly maxPerTurn: number
  /**
   * Whether the rendered message is wrapped in `<system-reminder>`.
   *
   * `agent_events` is deliberately unwrapped. Its bare `[sub-agent events]`
   * marker is already persisted in saved sessions and asserted byte-for-byte by
   * resume-parity checks, so re-tagging it would invalidate history that is
   * already on disk. New kinds all wrap.
   */
  readonly wrapInReminder: boolean
}

const INJECTION_SPECS: Readonly<Record<ContextInjectionKind, InjectionSpec>> = {
  agent_events: { heading: '[sub-agent events]', maxPerTurn: 24, wrapInReminder: false },
  compaction_rehydration: { heading: '[restored context]', maxPerTurn: 2, wrapInReminder: true },
  deferred_tools: { heading: '[tools now available]', maxPerTurn: 8, wrapInReminder: true },
  external_file_changes: {
    heading: '[files changed outside this session]',
    maxPerTurn: 12,
    wrapInReminder: true,
  },
  todo_reminder: { heading: '[todo list]', maxPerTurn: 6, wrapInReminder: true },
}

export type InjectionSkipReason = 'duplicate' | 'empty' | 'kind_throttled' | 'turn_budget'

export type InjectionPlan =
  | { readonly message: ChatMessage; readonly status: 'ready'; readonly text: string; readonly truncated: boolean }
  | { readonly reason: InjectionSkipReason; readonly status: 'skipped' }

/** What the transcript already carries from this seam since the last real user message. */
export interface InjectionUsage {
  readonly characters: number
  readonly counts: ReadonlyMap<ContextInjectionKind, number>
  /** Exact rendered texts, so an identical re-injection can be recognized. */
  readonly rendered: ReadonlySet<string>
}

/** Literal prefix every rendered message of this kind starts with. */
export function injectionSignature(kind: ContextInjectionKind): string {
  const spec = INJECTION_SPECS[kind]
  return spec.wrapInReminder ? reminderOpenTag(kind) : spec.heading
}

/**
 * Recover injection usage by reading the transcript backwards to the last
 * message this seam did not write.
 *
 * WHY a scan instead of a counter on the turn: a counter survives compaction and
 * resume, and would then refuse to inject into a history from which its own
 * earlier injections have already been summarized away — the budget would decay
 * to zero over a long session with no way to notice. Reading the messages makes
 * the budget a function of what the provider will actually see, so compaction,
 * `/clear`, branch and resume all restore it without anyone remembering to.
 */
export function scanInjections(messages: readonly ChatMessage[]): InjectionUsage {
  const counts = new Map<ContextInjectionKind, number>()
  const rendered = new Set<string>()
  let characters = 0
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message === undefined) continue
    if (message.role !== 'user') continue
    const text = messageText(message)
    const kind = injectionKindOf(text)
    // The first user message that is not ours is the turn's own prompt (or a
    // steer): everything older belongs to a previous exchange and is not this
    // turn's budget to spend.
    if (kind === undefined) break
    counts.set(kind, (counts.get(kind) ?? 0) + 1)
    rendered.add(text)
    characters += text.length
  }
  return { characters, counts, rendered }
}

/** Decide whether an injection may enter the transcript, and in what form. */
export function planInjection(
  messages: readonly ChatMessage[],
  injection: ContextInjection,
): InjectionPlan {
  const body = injectionBody(injection)
  if (!body) return { reason: 'empty', status: 'skipped' }

  const spec = INJECTION_SPECS[injection.kind]
  const usage = scanInjections(messages)
  if ((usage.counts.get(injection.kind) ?? 0) >= spec.maxPerTurn) {
    return { reason: 'kind_throttled', status: 'skipped' }
  }

  const remaining = MAX_TURN_INJECTION_CHARACTERS - usage.characters
  if (remaining < MIN_INJECTION_CHARACTERS) return { reason: 'turn_budget', status: 'skipped' }

  const limit = Math.min(MAX_INJECTION_CHARACTERS, remaining)
  const bounded = clampBody(body, limit)
  const text = render(injection.kind, spec, bounded.text)
  // An identical block already standing in this turn's window carries no new
  // information; re-sending it only teaches the model that the reminder channel
  // repeats itself.
  if (usage.rendered.has(text)) return { reason: 'duplicate', status: 'skipped' }
  return {
    message: { role: 'user', content: text },
    status: 'ready',
    text,
    truncated: bounded.truncated,
  }
}

/** Plan an injection and, when it is allowed, append it to the live transcript. */
export function appendInjection(
  messages: ChatMessage[],
  injection: ContextInjection,
): InjectionPlan {
  const plan = planInjection(messages, injection)
  if (plan.status === 'ready') messages.push(plan.message)
  return plan
}

function render(kind: ContextInjectionKind, spec: InjectionSpec, body: string): string {
  const inner = `${spec.heading}\n${body}`
  if (!spec.wrapInReminder) return inner
  // This seam is the one legitimate writer of the tag; inbound tool output is
  // defanged by neutralizeSystemReminders precisely so that stays true.
  return `${reminderOpenTag(kind)}\n${inner}\n</system-reminder>`
}

function reminderOpenTag(kind: ContextInjectionKind): string {
  return `<system-reminder kind="${kind}">`
}

function injectionKindOf(text: string): ContextInjectionKind | undefined {
  for (const kind of Object.keys(INJECTION_SPECS) as ContextInjectionKind[]) {
    if (text.startsWith(injectionSignature(kind))) return kind
  }
  return undefined
}

function injectionBody(injection: ContextInjection): string {
  switch (injection.kind) {
    case 'agent_events':
      return joinLines(injection.events)
    case 'compaction_rehydration':
      return injection.text.trim()
    case 'deferred_tools':
      return joinLines(injection.names)
    case 'external_file_changes':
      return joinLines(injection.paths)
    case 'todo_reminder':
      return injection.text.trim()
  }
}

function joinLines(values: readonly string[]): string {
  const lines: string[] = []
  for (const value of values) {
    const line = value.trim()
    if (line) lines.push(line)
  }
  return lines.join('\n')
}

function clampBody(body: string, limit: number): { readonly text: string; readonly truncated: boolean } {
  if (body.length <= limit) return { text: body, truncated: false }
  const dropped = body.length - limit
  const notice = `\n[injection truncated: ${dropped} characters dropped]`
  const keep = Math.max(0, limit - notice.length)
  return { text: body.slice(0, keep) + notice, truncated: true }
}
