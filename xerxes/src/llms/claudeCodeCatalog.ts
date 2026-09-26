// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What the signed-in Claude Code offers, asked of Claude Code itself.
 *
 * The CLI's stream-json `initialize` control request answers with the same
 * model list its own `/model` picker shows for the user's plan: each entry's
 * alias, the model it resolves to, display name and description, and its
 * thinking controls (effort levels, adaptive thinking). Nothing here is a
 * built-in list — a new model (or one the plan gains) appears on the next
 * refresh. No model call is made, so asking costs nothing.
 */

import { ConfigurationError } from '../core/errors.js'
import { claudeCodeEnvironment, claudeCodeExecutable } from './claudeCode.js'
import { REASONING_ON, type ReasoningLevelSet } from './reasoningLevels.js'

export interface ClaudeCodeModelInfo {
  /** What Claude Code accepts for `--model`, e.g. `opus[1m]` or `claude-fable-5-1[1m]`. */
  readonly value: string
  /** The model id it currently resolves to, e.g. `claude-opus-5-5[1m]`. */
  readonly resolvedModel?: string
  readonly displayName: string
  readonly description?: string
  /** Effort levels `--effort` accepts for this model; empty when it has no effort control. */
  readonly effortLevels: readonly string[]
  /** Adaptive thinking decides for itself and cannot be switched off. */
  readonly adaptiveThinking: boolean
  /** Context window, when the entry states it (`[1m]` = one million tokens). */
  readonly contextLimit?: number
}

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const text = (value: unknown): string | undefined => (typeof value === 'string' && value.trim() ? value.trim() : undefined)

/** A `[1m]` suffix on the alias or the resolved id is Claude Code's own marker for the 1M window. */
function statedContextLimit(...names: Array<string | undefined>): number | undefined {
  return names.some(name => name && /\[1m\]/i.test(name)) ? 1_000_000 : undefined
}

/** Parse the `models` array of an initialize response; malformed entries are skipped. */
export function parseClaudeCodeModels(value: unknown): ClaudeCodeModelInfo[] {
  if (!Array.isArray(value)) return []
  const models: ClaudeCodeModelInfo[] = []
  for (const item of value) {
    const entry = record(item)
    const alias = text(entry.value)
    if (!alias) continue
    const resolvedModel = text(entry.resolvedModel)
    const description = text(entry.description)
    const levels = Array.isArray(entry.supportedEffortLevels)
      ? entry.supportedEffortLevels.filter((level): level is string => typeof level === 'string' && level.trim() !== '')
      : []
    const contextLimit = statedContextLimit(alias, resolvedModel)
    models.push({
      value: alias,
      ...(resolvedModel ? { resolvedModel } : {}),
      displayName: text(entry.displayName) ?? alias,
      ...(description ? { description } : {}),
      effortLevels: entry.supportsEffort === true ? levels : [],
      adaptiveThinking: entry.supportsAdaptiveThinking === true,
      ...(contextLimit ? { contextLimit } : {}),
    })
  }
  return models
}

export interface ClaudeCodeCatalogOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly executable?: string
  readonly timeoutMs?: number
  /** Swap the CLI round trip (tests): given the argv/env, return the initialize response object. */
  readonly initialize?: (argv: readonly string[], env: Record<string, string>) => Promise<unknown>
}

async function initializeThroughCli(argv: readonly string[], env: Record<string, string>, timeoutMs: number): Promise<unknown> {
  const child = Bun.spawn([...argv], { env, stdin: 'pipe', stdout: 'pipe', stderr: 'ignore' })
  const timer = setTimeout(() => child.kill(), timeoutMs)
  try {
    child.stdin.write(JSON.stringify({ type: 'control_request', request_id: 'xerxes-models', request: { subtype: 'initialize' } }) + '\n')
    child.stdin.flush()
    const reader = child.stdout.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        let event: Record<string, unknown>
        try { event = record(JSON.parse(line)) } catch { continue }
        if (event.type !== 'control_response') continue
        const response = record(event.response)
        if (response.subtype === 'error') throw new Error(text(response.error) ?? 'Claude Code refused the model list request.')
        return response.response ?? response
      }
    }
    throw new Error('Claude Code closed before listing its models.')
  } finally {
    clearTimeout(timer)
    child.kill()
  }
}

/** Ask the local Claude Code which models this sign-in can use. */
export async function discoverClaudeCodeModels(options: ClaudeCodeCatalogOptions = {}): Promise<ClaudeCodeModelInfo[]> {
  const environment = options.environment ?? process.env
  const executable = options.executable ?? claudeCodeExecutable(environment)
  const argv = [executable, '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
    '--no-session-persistence', '--tools', '', '--setting-sources', '', '--strict-mcp-config']
  const env = claudeCodeEnvironment(environment)
  const response = await (options.initialize ?? ((a, e) => initializeThroughCli(a, e, options.timeoutMs ?? 20_000)))(argv, env)
  const models = parseClaudeCodeModels(record(response).models)
  if (!models.length) throw new ConfigurationError('claude-code', 'Claude Code reported no models for this sign-in. Run \'claude\' and check /model.')
  return models
}

/**
 * The last answer, shared by the model picker, the reasoning picker, context
 * windows and the adapter. Refreshed on demand (the picker's refresh, or when
 * older than `ttlMs`); a failed refresh keeps serving the previous answer.
 */
export class ClaudeCodeCatalog {
  private models: readonly ClaudeCodeModelInfo[] = []
  private fetchedAt = 0
  private inflight: Promise<readonly ClaudeCodeModelInfo[]> | undefined

  constructor(
    private readonly discover: () => Promise<ClaudeCodeModelInfo[]> = () => discoverClaudeCodeModels(),
    private readonly ttlMs = 30 * 60_000,
    private readonly now: () => number = Date.now,
  ) {}

  /** Synchronous view of the last answer (empty until the first discovery). */
  peek(): readonly ClaudeCodeModelInfo[] { return this.models }

  /** The entry for a Xerxes model id (`claude-code/opus[1m]`) or a bare alias. */
  find(model: string): ClaudeCodeModelInfo | undefined {
    const alias = model.trim().replace(/^claude[-_]code\//i, '')
    // A session saved on a plain alias (`opus`) still finds its variant (`opus[1m]`).
    const base = (name: string) => name.replace(/\[[^\]]*\]$/, '')
    return this.models.find(entry => entry.value === alias)
      ?? this.models.find(entry => entry.resolvedModel === alias)
      ?? this.models.find(entry => base(entry.value) === alias)
  }

  async load(refresh = false): Promise<readonly ClaudeCodeModelInfo[]> {
    if (!refresh && this.models.length && this.now() - this.fetchedAt < this.ttlMs) return this.models
    this.inflight ??= this.discover().then(models => {
      this.models = models
      this.fetchedAt = this.now()
      return models
    }).finally(() => { this.inflight = undefined })
    try {
      return await this.inflight
    } catch (error) {
      if (this.models.length) return this.models
      throw error
    }
  }
}

/** One catalog per process: the CLI answers the same for every session. */
export const claudeCodeCatalog = new ClaudeCodeCatalog()

/**
 * The reasoning picker for one Claude Code model, straight from what the CLI
 * reported: its effort levels (no invented descriptions or default), `off`
 * only where thinking can be switched off, and a plain on/off switch for a
 * model that thinks on a budget without effort levels. An unknown model gets
 * nothing to pick rather than a guessed ladder.
 */
export function claudeCodeReasoningLevels(entry: ClaudeCodeModelInfo | undefined): ReasoningLevelSet {
  if (!entry) return { defaultEffort: undefined, levels: [], shape: 'inherent', source: 'fallback', provenance: 'provider_fallback', canDisable: false }
  if (entry.effortLevels.length) {
    return {
      defaultEffort: undefined,
      levels: entry.effortLevels.map(effort => ({ effort })),
      shape: 'effort',
      source: 'provider',
      provenance: 'provider_reported',
      canDisable: !entry.adaptiveThinking,
    }
  }
  if (entry.adaptiveThinking) return { defaultEffort: undefined, levels: [], shape: 'inherent', source: 'provider', provenance: 'provider_reported', canDisable: false }
  return { defaultEffort: undefined, levels: [{ effort: REASONING_ON }], shape: 'toggle', source: 'provider', provenance: 'provider_reported', canDisable: true }
}
