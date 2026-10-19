// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolCall } from '../types/toolCalls.js'

/** Minimum number of same-shaped calls in one turn before a plugin is suggested. */
export const PLUGIN_SUGGESTION_THRESHOLD = 3

/** A repeated command shape worth turning into a plugin tool. */
export interface PluginSuggestion {
  /** Program (exec/shell) or tool name (fetch-like tools). */
  readonly program: string
  /** Generalized first argument, e.g. `https://api.example.com/…`. */
  readonly pattern: string
  readonly count: number
  /** One concrete invocation, for the message. */
  readonly example: string
}

const SHELL_TOOLS = new Set(['Bash', 'ExecuteShell', 'shell'])
const FETCH_TOOLS = /^(?:web_?fetch|fetch|http_?request|WebFetch|WebScraper|web_extract)$/i
const MAX_EXAMPLE = 160

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function programName(value: string): string {
  const parts = value.split(/[\\/]/)
  return parts[parts.length - 1] || value
}

/** Generalize an argument so calls that differ only in ids, numbers or query strings group together. */
export function argumentShape(argument: string): string {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(argument)) {
    try {
      const url = new URL(argument)
      const first = url.pathname.split('/').filter(Boolean)[0]
      return `${url.protocol}//${url.host}/${first ? `${first.replace(/\d+/g, 'N')}/` : ''}…`
    } catch { /* not a URL after all */ }
  }
  if (argument.includes('/')) {
    const segments = argument.split('/')
    return `${segments.slice(0, -1).join('/').replace(/\d+/g, 'N')}/…`
  }
  return argument.replace(/\d+/g, 'N')
}

/** `[program, args]` for one command call, or undefined when it is not a command-like call. */
function commandOf(call: Pick<ToolCall, 'function'>): { program: string; args: string[]; example: string } | undefined {
  const name = call.function.name
  const inputs = call.function.arguments ?? {}
  if (name === 'exec_command') {
    const cmd = text(inputs.cmd)
    if (!cmd) return undefined
    const args = Array.isArray(inputs.args) ? inputs.args.filter((item): item is string => typeof item === 'string') : []
    // A shell-string cmd (PTY form) is split naively on whitespace.
    const words = args.length ? [cmd, ...args] : cmd.split(/\s+/)
    return { program: programName(words[0]!), args: words.slice(1), example: words.join(' ') }
  }
  if (SHELL_TOOLS.has(name)) {
    const command = text(inputs.command)
    if (!command) return undefined
    const words = command.split(/\s+/)
    return { program: programName(words[0]!), args: words.slice(1), example: command }
  }
  if (FETCH_TOOLS.test(name)) {
    const url = text(inputs.url)
    return url ? { program: name, args: [url], example: `${name} ${url}` } : undefined
  }
  return undefined
}

/**
 * Pure detector over one turn's tool calls: when the same program was run with
 * the same first-argument shape at least {@link PLUGIN_SUGGESTION_THRESHOLD}
 * times, return the most frequent such shape. It never creates or installs
 * anything; the caller decides whether to show a suggestion.
 */
export function detectRepeatedCommand(calls: readonly Pick<ToolCall, 'function'>[], threshold = PLUGIN_SUGGESTION_THRESHOLD): PluginSuggestion | undefined {
  const groups = new Map<string, { program: string; pattern: string; count: number; example: string }>()
  for (const call of calls) {
    const command = commandOf(call)
    if (!command) continue
    // The first non-flag argument carries the intent (`curl -s URL`, `gh api PATH`).
    const first = command.args.find(argument => !argument.startsWith('-')) ?? command.args[0] ?? ''
    const pattern = argumentShape(first)
    const key = `${command.program}\0${pattern}`
    const group = groups.get(key) ?? { program: command.program, pattern, count: 0, example: command.example }
    group.count += 1
    groups.set(key, group)
  }
  let best: PluginSuggestion | undefined
  for (const group of groups.values()) {
    if (group.count < threshold) continue
    if (!best || group.count > best.count) {
      best = { ...group, example: group.example.length > MAX_EXAMPLE ? `${group.example.slice(0, MAX_EXAMPLE - 1)}…` : group.example }
    }
  }
  return best
}

/** The one-line, non-blocking suggestion shown at the end of a turn. */
export function pluginSuggestionMessage(suggestion: PluginSuggestion): string {
  const shape = `${suggestion.program}${suggestion.pattern ? ` ${suggestion.pattern}` : ''}`
  return `This turn ran \`${shape}\` ${suggestion.count} times (e.g. \`${suggestion.example}\`). A plugin tool could do it in one call: run /plugin-creator and describe it. Nothing was created.`
}
