// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  CATEGORIES,
  COMMAND_REGISTRY,
  resolveCommand,
  type CommandCategory,
  type CommandDefinition,
} from '../bridge/commands.js'

const PLUGIN_COMMAND_NAME = /^[a-z0-9][a-z0-9_-]*$/

/** A plugin-owned command callback. Invocation remains host-owned. */
export type SlashHandler = (...arguments_: never[]) => unknown | Promise<unknown>

/** Immutable pairing of plugin command metadata and its callback. */
export interface SlashPlugin {
  readonly command: CommandDefinition
  readonly handler: SlashHandler
}

export interface SlashPluginOptions {
  readonly aliases?: readonly string[]
  readonly argsHint?: string
  readonly category?: CommandCategory
  readonly cliOnly?: boolean
  readonly description?: string
  readonly gatewayOnly?: boolean
}

/** Raised when a plugin tries to claim a built-in or already-owned slash token. */
export class SlashPluginConflictError extends Error {
  constructor(
    readonly token: string,
    readonly owner: 'built-in' | 'plugin',
  ) {
    super(`Slash command token '${token}' is already owned by a ${owner} command`)
    this.name = 'SlashPluginConflictError'
  }
}

/**
 * Plugin-contributed slash commands, isolated from the built-in bridge registry.
 *
 * Registration is synchronous and validates every name and alias before mutating
 * state, so a failed registration cannot leave aliases partially installed.
 */
export class SlashPluginRegistry {
  private readonly aliases = new Map<string, string>()
  private readonly plugins = new Map<string, SlashPlugin>()

  /**
   * Register a plugin command without allowing it to shadow built-in names,
   * built-in aliases, or tokens owned by another plugin.
   */
  register(name: string, handler: SlashHandler, options: SlashPluginOptions = {}): SlashPlugin {
    if (typeof handler !== 'function') {
      throw new TypeError('slash command handler must be a function')
    }
    const cleanName = normalizeRegistrationToken(name, 'slash command name')
    const normalized = normalizeOptions(cleanName, options)
    const tokens = [cleanName, ...normalized.aliases]
    if (new Set(tokens).size !== tokens.length) {
      throw new TypeError('slash command names and aliases must be unique')
    }
    for (const token of tokens) this.assertTokenAvailable(token)

    const command: CommandDefinition = Object.freeze({
      name: cleanName,
      description: normalized.description,
      category: normalized.category,
      aliases: Object.freeze([...normalized.aliases]),
      argsHint: normalized.argsHint,
      cliOnly: normalized.cliOnly,
      gatewayOnly: normalized.gatewayOnly,
      deprecated: false,
      examples: Object.freeze([]),
    })
    const plugin: SlashPlugin = Object.freeze({ command, handler })
    this.plugins.set(cleanName, plugin)
    for (const alias of normalized.aliases) this.aliases.set(alias, cleanName)
    return plugin
  }

  /** Remove an exact canonical plugin command name and return whether it existed. */
  unregister(name: string): boolean {
    const cleanName = normalizedUnregisterName(name)
    if (cleanName === undefined) return false
    const plugin = this.plugins.get(cleanName)
    if (plugin === undefined) return false
    this.plugins.delete(cleanName)
    for (const alias of plugin.command.aliases) this.aliases.delete(alias)
    return true
  }

  /** Resolve slash-prefixed input, arguments, aliases, and bot suffixes to a plugin. */
  resolve(text: string): SlashPlugin | undefined {
    const token = resolutionToken(text)
    if (token === undefined) return undefined
    return this.plugins.get(this.aliases.get(token) ?? token)
  }

  /** Return immutable plugin records sorted by canonical command name. */
  list(): SlashPlugin[] {
    return [...this.plugins.values()].sort((left, right) => compareNames(left.command.name, right.command.name))
  }

  /** Return built-in commands followed by lexically sorted plugin commands. */
  allCommands(): CommandDefinition[] {
    return [...COMMAND_REGISTRY, ...this.list().map(plugin => plugin.command)]
  }

  private assertTokenAvailable(token: string): void {
    if (resolveCommand(token) !== undefined) {
      throw new SlashPluginConflictError(token, 'built-in')
    }
    if (this.plugins.has(token) || this.aliases.has(token)) {
      throw new SlashPluginConflictError(token, 'plugin')
    }
  }
}

/** Module-level default registry for plugins that do not need dependency injection. */
export const defaultSlashPluginRegistry = new SlashPluginRegistry()

/** Return the shared default slash-plugin registry. */
export function getDefaultSlashPluginRegistry(): SlashPluginRegistry {
  return defaultSlashPluginRegistry
}

/** Register a command on the shared default slash-plugin registry. */
export function registerSlash(
  name: string,
  handler: SlashHandler,
  options: SlashPluginOptions = {},
): SlashPlugin {
  return defaultSlashPluginRegistry.register(name, handler, options)
}

/** Remove an exact canonical command name from the shared slash-plugin registry. */
export function unregisterSlash(name: string): boolean {
  return defaultSlashPluginRegistry.unregister(name)
}

/** Resolve input against the shared slash-plugin registry. */
export function resolveSlash(text: string): SlashPlugin | undefined {
  return defaultSlashPluginRegistry.resolve(text)
}

/** List the shared registry's plugin commands in deterministic order. */
export function registeredSlashes(): SlashPlugin[] {
  return defaultSlashPluginRegistry.list()
}

function normalizeOptions(name: string, options: SlashPluginOptions): {
  readonly aliases: readonly string[]
  readonly argsHint: string
  readonly category: CommandCategory
  readonly cliOnly: boolean
  readonly description: string
  readonly gatewayOnly: boolean
} {
  const aliases = options.aliases ?? []
  if (!Array.isArray(aliases)) {
    throw new TypeError('slash command aliases must be an array')
  }
  const category = options.category ?? 'tools'
  if (!(CATEGORIES as readonly string[]).includes(category)) {
    throw new TypeError(`unknown slash command category: ${String(category)}`)
  }
  const description = stringOption(options.description, 'slash command description')
  const argsHint = stringOption(options.argsHint, 'slash command args hint')
  return {
    aliases: aliases.map(alias => normalizeRegistrationToken(alias, 'slash command alias')),
    argsHint: argsHint ?? '',
    category,
    cliOnly: booleanOption(options.cliOnly, 'slash command cliOnly'),
    description: description?.trim() || `plugin-registered command /${name}`,
    gatewayOnly: booleanOption(options.gatewayOnly, 'slash command gatewayOnly'),
  }
}

function normalizeRegistrationToken(value: string, label: string): string {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be a string`)
  }
  const token = value.trim().replace(/^\/+/, '').toLowerCase()
  if (!PLUGIN_COMMAND_NAME.test(token)) {
    throw new TypeError(`${label} may contain only letters, numbers, hyphens, or underscores`)
  }
  return token
}

function normalizedUnregisterName(name: string): string | undefined {
  try {
    return normalizeRegistrationToken(name, 'slash command name')
  } catch {
    return undefined
  }
}

function resolutionToken(text: string): string | undefined {
  if (typeof text !== 'string') return undefined
  const slashless = text.trim().replace(/^\/+/, '')
  const raw = slashless.split(/\s+/, 1)[0] ?? ''
  const token = (raw.split('@', 1)[0] ?? '').toLowerCase()
  return token || undefined
}

function stringOption(value: string | undefined, label: string): string | undefined {
  if (value !== undefined && typeof value !== 'string') {
    throw new TypeError(`${label} must be a string`)
  }
  return value
}

function booleanOption(value: boolean | undefined, label: string): boolean {
  if (value !== undefined && typeof value !== 'boolean') {
    throw new TypeError(`${label} must be a boolean`)
  }
  return value ?? false
}

function compareNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}
