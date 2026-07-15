// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const CATEGORIES = [
  'session',
  'config',
  'tools',
  'skills',
  'info',
  'feedback',
  'memory',
  'voice',
  'snapshots',
  'messaging',
  'exit',
] as const

export type CommandCategory = (typeof CATEGORIES)[number]
export type CommandSurface = 'all' | 'cli' | 'gateway'

export interface CommandDefinition {
  readonly aliases: readonly string[]
  readonly argsHint: string
  readonly category: CommandCategory
  readonly cliOnly: boolean
  readonly deprecated: boolean
  readonly description: string
  readonly examples: readonly string[]
  readonly gatewayOnly: boolean
  readonly name: string
}

export type CommandDef = CommandDefinition

interface CommandOptions {
  readonly aliases?: readonly string[]
  readonly argsHint?: string
  readonly cliOnly?: boolean
  readonly deprecated?: boolean
  readonly examples?: readonly string[]
  readonly gatewayOnly?: boolean
}

const COMMAND_NAME = /^[a-z0-9][a-z0-9-]*$/
const TELEGRAM_COMMAND_NAME = /^[a-z0-9_]{1,32}$/

/**
 * Single canonical source for TUI, daemon, and gateway slash-command metadata.
 *
 * Definitions are data only. Dispatch remains host-owned so listing a command
 * never implies that a runtime handler is available.
 */
export const COMMAND_REGISTRY: readonly CommandDefinition[] = Object.freeze([
  command('new', 'Start a fresh conversation', 'session', { aliases: ['reset'] }),
  command('clear', 'Clear the visible scrollback', 'session'),
  command('history', 'Show or search conversation history', 'session'),
  command('save', 'Save the session by name', 'session', { argsHint: '<name>' }),
  command('retry', 'Re-run the last turn', 'session'),
  command('retry-connection', 'Retry the last failed provider connection', 'session'),
  command('undo', 'Undo the last turn', 'session'),
  command('title', 'Generate or set the session title', 'session', { argsHint: '[title]' }),
  command('branch', 'Fork this session', 'session', { argsHint: '[label]' }),
  command('branches', 'List branches of this session', 'session'),
  command('compact', 'Compress the conversation', 'session', { aliases: ['compress'] }),
  command('rollback', 'Restore filesystem to a snapshot', 'snapshots', { argsHint: '<id|label>' }),
  command('snapshot', 'Take a filesystem snapshot', 'snapshots', { argsHint: '[label]' }),
  command('snapshots', 'List filesystem snapshots', 'snapshots'),
  command('stop', 'Cancel the in-flight tool call', 'session', { aliases: ['cancel'] }),
  command('cancel-all', 'Cancel every queued action', 'session'),
  command('background', 'Detach the session into the daemon', 'session', { cliOnly: true }),
  command('btw', 'Inject side-channel context', 'session'),
  command('queue', 'Show or manage queued prompts', 'session', { cliOnly: true }),
  command('status', 'Show platform / session status', 'info'),
  command('resume', 'Resume a saved session', 'session', { argsHint: '<id|name>' }),
  command('steer', 'Course-correct mid-stream', 'session', { cliOnly: true }),

  command('config', 'Inspect or edit runtime config', 'config'),
  command('model', 'List provider models or switch model', 'config', { argsHint: '[provider:model]' }),
  command('provider', 'Setup or switch provider profile', 'config'),
  command('sampling', 'Get or set sampling params', 'config'),
  command('personality', 'Choose a SOUL.md preset', 'config', { argsHint: '[name]' }),
  command('statusbar', 'Toggle the status bar', 'config', { cliOnly: true }),
  command('verbose', 'Toggle verbose event logging', 'config', { cliOnly: true }),
  command('yolo', 'Toggle accept-all permissions mode', 'config'),
  command('reasoning', 'Show/set thinking effort (off|low|medium|high)', 'config', {
    aliases: ['thinking'],
    argsHint: '[level]',
  }),
  command('fast', 'Use the aux model for the next turn', 'config'),
  command('skin', 'Change theme/skin', 'config', { argsHint: '[name]', cliOnly: true }),
  command('voice', 'Toggle voice mode', 'voice', { argsHint: 'on|off|play', cliOnly: true }),
  command('permissions', 'View / set permission mode', 'config'),
  command('debug', 'Toggle debug output', 'config', { cliOnly: true }),

  command('tools', 'List available tools', 'tools'),
  command('toolsets', 'List configured toolsets', 'tools'),
  command('skills', 'List available skills', 'skills'),
  command('skill', 'Invoke a skill by name', 'skills', { argsHint: '<name>' }),
  command('skill-create', 'Scaffold a new skill directory', 'skills', { argsHint: '<name>' }),
  command('cron', 'Manage scheduled tasks', 'tools', { argsHint: 'list|add|remove|run' }),
  command('reload', 'Reload skills + tools', 'tools'),
  command('reload-mcp', 'Reload MCP servers', 'tools'),
  command('browser', 'Manage browser sessions', 'tools'),
  command('plugins', 'List loaded plugins', 'tools'),
  command('init', 'Ask the agent to inspect and initialize project context', 'tools'),
  command('workspace', 'Inspect or initialize project .agents workspace', 'tools', { argsHint: '[status|init]' }),
  command('soul', 'Show or edit SOUL.md', 'tools'),
  command('agents', 'List or select sub-agents', 'tools'),

  command('help', 'Show help', 'info', { aliases: ['?'] }),
  command('commands', 'List every available command', 'info'),
  command('restart', 'Restart the agent process', 'info'),
  command('usage', 'Show token & cost usage', 'info'),
  command('insights', 'Show usage analytics', 'info', { argsHint: '[--days N]' }),
  command('platforms', 'List configured messaging platforms', 'info'),
  command('paste', 'Paste from clipboard', 'info', { cliOnly: true }),
  command('image', 'Attach a clipboard image to the next message', 'info', { cliOnly: true }),
  command('update', 'Update Xerxes', 'info'),
  command('cost', 'Show running cost', 'info'),
  command('context', 'Show session info', 'info'),
  command('doctor', 'Diagnose configuration', 'info'),

  command('memory', 'Manage memory backends', 'memory', { argsHint: 'backend list|use NAME|status' }),
  command('feedback', 'Give compaction / behavior feedback', 'feedback'),
  command('nudge', 'Toggle proactive nudges', 'feedback', { argsHint: 'on|off' }),
  command('budget', 'Set or show per-session budget', 'feedback', { argsHint: 'set <amount>' }),

  command('exit', 'Quit', 'exit', { aliases: ['quit', 'q'] }),
])

const commandMaps = commandMapsFor(COMMAND_REGISTRY)

export interface CommandListOptions {
  readonly category?: CommandCategory
  readonly includeDeprecated?: boolean
  readonly surface?: CommandSurface
}

export interface CommandResolveOptions {
  readonly surface?: CommandSurface
}

/** Return a registry-order snapshot filtered by category and presentation surface. */
export function listCommands(options: CommandCategory | CommandListOptions = {}): CommandDefinition[] {
  const resolved = typeof options === 'string' ? { category: options } : options
  const surface = resolved.surface ?? 'all'
  return COMMAND_REGISTRY.filter(commandDefinition => (
    (resolved.category === undefined || commandDefinition.category === resolved.category)
    && ((resolved.includeDeprecated ?? true) || !commandDefinition.deprecated)
    && commandAvailableOnSurface(commandDefinition, surface)
  ))
}

/** Resolve canonical names and aliases from slash-prefixed input with optional arguments. */
export function resolveCommand(text: string, options: CommandResolveOptions = {}): CommandDefinition | undefined {
  const token = commandToken(text)
  if (!token) {
    return undefined
  }
  const commandDefinition = commandMaps.names.get(token) ?? commandMaps.aliases.get(token)
  if (!commandDefinition || !commandAvailableOnSurface(commandDefinition, options.surface ?? 'all')) {
    return undefined
  }
  return commandDefinition
}

/** Return whether a command belongs on a concrete presentation surface. */
export function commandAvailableOnSurface(commandDefinition: CommandDefinition, surface: CommandSurface): boolean {
  switch (surface) {
    case 'all':
      return true
    case 'cli':
      return !commandDefinition.gatewayOnly
    case 'gateway':
      return !commandDefinition.cliOnly
  }
}

/**
 * Convert a canonical command name into Telegram BotCommand syntax.
 *
 * Hyphens become underscores, slash and bot suffix notation are accepted for
 * convenience, and invalid or overlong values are rejected rather than
 * silently changed into a different command.
 */
export function normalizeTelegramCommandName(name: string): string | undefined {
  const slashless = name.trim().replace(/^\/+/, '')
  const base = slashless.split('@', 1)[0] ?? ''
  const normalized = base.toLowerCase().replaceAll('-', '_')
  return TELEGRAM_COMMAND_NAME.test(normalized) ? normalized : undefined
}

/** Render non-CLI command definitions in the JSON shape accepted by Telegram setMyCommands. */
export function telegramBotCommands(commands: Iterable<CommandDefinition> = COMMAND_REGISTRY): Array<{
  readonly command: string
  readonly description: string
}> {
  const result: Array<{ readonly command: string; readonly description: string }> = []
  const seen = new Set<string>()
  for (const commandDefinition of commands) {
    if (commandDefinition.cliOnly) {
      continue
    }
    const name = normalizeTelegramCommandName(commandDefinition.name)
    if (!name || seen.has(name)) {
      continue
    }
    seen.add(name)
    result.push({ command: name, description: commandDefinition.description.slice(0, 256) })
  }
  return result
}

function command(
  name: string,
  description: string,
  category: CommandCategory,
  options: CommandOptions = {},
): CommandDefinition {
  const normalizedName = name.trim().toLowerCase()
  if (!COMMAND_NAME.test(normalizedName)) {
    throw new TypeError('invalid command name: ' + name)
  }
  if (!description.trim()) {
    throw new TypeError('command description must not be empty')
  }
  const aliases = [...(options.aliases ?? [])].map(alias => alias.trim().toLowerCase())
  if (aliases.some(alias => !alias || /\s/.test(alias))) {
    throw new TypeError('command aliases must be non-empty single tokens')
  }
  return Object.freeze({
    name: normalizedName,
    description,
    category,
    aliases: Object.freeze(aliases),
    argsHint: options.argsHint ?? '',
    cliOnly: options.cliOnly ?? false,
    gatewayOnly: options.gatewayOnly ?? false,
    deprecated: options.deprecated ?? false,
    examples: Object.freeze([...(options.examples ?? [])]),
  })
}

function commandMapsFor(commands: readonly CommandDefinition[]): {
  readonly aliases: ReadonlyMap<string, CommandDefinition>
  readonly names: ReadonlyMap<string, CommandDefinition>
} {
  const aliases = new Map<string, CommandDefinition>()
  const names = new Map<string, CommandDefinition>()
  for (const commandDefinition of commands) {
    if (names.has(commandDefinition.name) || aliases.has(commandDefinition.name)) {
      throw new TypeError('duplicate command name: ' + commandDefinition.name)
    }
    names.set(commandDefinition.name, commandDefinition)
    for (const alias of commandDefinition.aliases) {
      if (names.has(alias) || aliases.has(alias)) {
        throw new TypeError('duplicate command alias: ' + alias)
      }
      aliases.set(alias, commandDefinition)
    }
  }
  return { names, aliases }
}

function commandToken(text: string): string {
  const withoutSlash = text.trim().replace(/^\/+/, '')
  const token = withoutSlash.split(/\s+/, 1)[0] ?? ''
  return (token.split('@', 1)[0] ?? '').toLowerCase()
}
