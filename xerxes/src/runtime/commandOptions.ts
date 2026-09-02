// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Parse options which all require a value. Values are consumed before the next
 * argument is examined, preventing valid option values from being mistaken for
 * unsupported positional arguments.
 */
export function parseValueOptions(
  args: readonly string[],
  command: string,
  valueFlags: readonly string[],
): ReadonlyMap<string, string> {
  const known = new Set(valueFlags)
  const values = new Map<string, string>()
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (!argument.startsWith('-')) {
      throw new Error(`Unexpected ${command} argument: ${argument}`)
    }
    if (!known.has(argument)) {
      throw new Error(`Unknown ${command} option: ${argument}`)
    }
    const value = args[index + 1]
    if (value === undefined || value.startsWith('-')) {
      throw new Error(`${command} option ${argument} requires a value`)
    }
    values.set(argument, value)
    index += 1
  }
  return values
}

/**
 * Extract the global `--agent <name|path>` option from raw CLI arguments.
 *
 * The option is global so it can appear before or after the prompt words; the
 * command dispatcher decides whether the resolved reference is honored. Both
 * `--agent value` and `--agent=value` spellings are accepted, matching how
 * users reach for CLI tools.
 */
export function extractAgentOption(
  args: readonly string[],
): { readonly agent: string | undefined; readonly rest: string[] } {
  let agent: string | undefined
  const rest: string[] = []
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (argument === '--agent') {
      const value = args[index + 1]
      if (value === undefined || value.startsWith('-')) {
        throw new Error('The --agent option requires an agent name or file path')
      }
      agent = value
      index += 1
      continue
    }
    if (argument.startsWith('--agent=')) {
      const value = argument.slice('--agent='.length)
      if (!value) {
        throw new Error('The --agent option requires an agent name or file path')
      }
      agent = value
      continue
    }
    rest.push(argument)
  }
  return { agent, rest }
}

export type OutputFormat = 'json' | 'stream-json' | 'text'

/**
 * Extract the global `--output-format <text|json|stream-json>` option
 * (Claude Code `-p --output-format` parity). `text` is the default and keeps
 * the historical plain-text stream; `json` buffers the reply and prints one
 * result object; `stream-json` prints one NDJSON event per line as the turn
 * runs, for scripts and CI.
 */
export function extractOutputFormatOption(
  args: readonly string[],
): { readonly format: OutputFormat; readonly rest: string[] } {
  let format: OutputFormat = 'text'
  const rest: string[] = []
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    const value = argument === '--output-format'
      ? args[index + 1]
      : argument.startsWith('--output-format=')
        ? argument.slice('--output-format='.length)
        : undefined
    if (value === undefined) {
      if (argument === '--output-format') {
        throw new Error('The --output-format option requires a value (text, json, or stream-json)')
      }
      rest.push(argument)
      continue
    }
    if (argument === '--output-format') index += 1
    if (value !== 'text' && value !== 'json' && value !== 'stream-json') {
      throw new Error(`Unknown --output-format '${value}' (expected text, json, or stream-json)`)
    }
    format = value
  }
  return { format, rest }
}
