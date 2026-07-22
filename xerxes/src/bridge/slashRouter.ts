// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { COMMAND_REGISTRY, resolveCommand, type CommandDefinition } from './commands.js'
import { calcCost, resolveProvider } from '../llms/providerRegistry.js'
import { skillMatchesPlatform, type Skill, type SkillRegistry } from '../extensions/skills.js'
import type { CostTracker } from '../runtime/costTracker.js'
import type { AgentState } from '../streaming/events.js'

const SAMPLING_PARAMS = new Set([
  'temperature', 'top_p', 'top_k', 'max_tokens', 'frequency_penalty', 'presence_penalty', 'repetition_penalty', 'min_p',
  'thinking', 'reasoning_effort', 'thinking_budget',
])

const THINKING_LEVELS = new Set(['off', 'low', 'medium', 'high'])
const SENSITIVE_CONFIG_NAME = /api[_-]?key|token|secret|password/iu
const BOOLEAN_SAMPLING_VALUES: ReadonlyMap<string, boolean> = new Map([
  ['1', true],
  ['0', false],
  ['true', true],
  ['false', false],
  ['on', true],
  ['off', false],
  ['yes', true],
  ['no', false],
])
const PERMISSION_MODES = ['auto', 'accept-all', 'manual'] as const
const SKILL_NAME = /^[A-Za-z0-9_-]+$/
const ROUTED_COMMANDS = new Set([
  'agents', 'clear', 'commands', 'compact', 'config', 'context', 'cost', 'debug', 'exit', 'help', 'history',
  'model', 'permissions', 'plan', 'provider', 'reasoning', 'sampling', 'skill', 'skill-create', 'skills',
  'tools', 'usage', 'verbose', 'yolo',
])

export type BridgeSlashConfig = Record<string, unknown>
export type BridgeSlashStatus = 'handled' | 'invalid' | 'unavailable' | 'unknown'

export interface ParsedBridgeSlashCommand {
  readonly args: string
  readonly command?: CommandDefinition
  readonly name: string
  readonly raw: string
}

export interface InvalidBridgeSlashCommand {
  readonly error: string
  readonly raw: string
}

export interface BridgeSlashResult {
  readonly command: string
  readonly output: string
  readonly status: BridgeSlashStatus
}

export interface BridgeSlashModelCatalog {
  list(input: {
    readonly apiKey: string
    readonly baseUrl: string
    readonly currentModel: string
  }): Promise<readonly string[]> | readonly string[]
  switchModel(model: string): Promise<void> | void
}

export interface BridgeProviderProfile {
  readonly api_key: string
  readonly base_url: string
  readonly model: string
  readonly name: string
  readonly provider: string
  readonly sampling: Readonly<Record<string, unknown>>
  readonly active?: boolean
}

/**
 * A provider host owns persistence and runtime reconfiguration atomically.
 *
 * `select` must only return a profile after the selected connection is live;
 * this router deliberately cannot make a persisted profile look selected when
 * the caller has not reconfigured its actual provider client.
 */
export interface BridgeSlashProviderPort {
  active(): Promise<BridgeProviderProfile | undefined> | BridgeProviderProfile | undefined
  list(): Promise<readonly BridgeProviderProfile[]> | readonly BridgeProviderProfile[]
  saveSampling(
    name: string,
    sampling: Readonly<Record<string, unknown>>,
  ): Promise<BridgeProviderProfile | undefined> | BridgeProviderProfile | undefined
  select(name: string): Promise<BridgeProviderProfile | undefined> | BridgeProviderProfile | undefined
}

export interface BridgeSlashCompactionResult {
  readonly compacted: boolean
  readonly error?: string
  readonly keptCount?: number
  readonly reason?: string
  readonly summarizedCount?: number
}

export interface BridgeSlashTool {
  readonly description: string
  readonly name: string
  readonly safe?: boolean
}

export interface BridgeSlashAgent {
  readonly description: string
  readonly name: string
  readonly source?: string
}

export interface BridgeSlashSkillsPort {
  /** Re-discover skills through a host-owned source before listing them. */
  discover?(): Promise<void> | void
  /** Run an activated skill using the host's real agent/session execution path. */
  invoke(input: {
    readonly args: string
    readonly name: string
    readonly skill: Skill
  }): Promise<string | void> | string | void
  readonly registry: Pick<SkillRegistry, 'all' | 'get' | 'search'>
  /** Begin the existing native skill-authoring flow; it must not merely acknowledge the request. */
  startCreate?(name: string): Promise<string | void> | string | void
}

export interface BridgeSlashRouterHost {
  /** Performs real conversation compaction against the caller-owned state and returns its measured result. */
  compact?(state: AgentState): Promise<BridgeSlashCompactionResult> | BridgeSlashCompactionResult
  /** Propagates an accepted in-memory config update to the owning runtime. */
  configChanged?(config: Readonly<BridgeSlashConfig>): Promise<void> | void
  readonly costTracker?: Pick<CostTracker, 'summary'>
  /** Stops the host process/session. It is never inferred from this router. */
  exit?(): Promise<void> | void
  agents?(): Promise<readonly BridgeSlashAgent[]> | readonly BridgeSlashAgent[]
  readonly models?: BridgeSlashModelCatalog
  plan?(objective: string): Promise<string> | string
  readonly providers?: BridgeSlashProviderPort
  readonly skills?: BridgeSlashSkillsPort
  tools?(): Promise<readonly BridgeSlashTool[]> | readonly BridgeSlashTool[]
}

export interface BridgeSlashRouterOptions {
  /** Mutable session configuration owned by the caller. */
  readonly config: BridgeSlashConfig
  /** Captured session working directory; no ambient cwd lookup is performed. */
  readonly cwd: string
  readonly host?: BridgeSlashRouterHost
  /** Captured platform used for skill compatibility checks. */
  readonly platform?: NodeJS.Platform
  readonly state: AgentState
}

/**
 * Parse a user slash line without performing I/O or mutating a session.
 *
 * Built-in aliases are canonicalized with the shared command registry. The
 * Python bridge's historical `/h` alias remains accepted even though the
 * metadata registry calls it `/help`.
 */
export function parseBridgeSlashCommand(raw: string): ParsedBridgeSlashCommand | InvalidBridgeSlashCommand {
  if (typeof raw !== 'string') {
    throw new TypeError('slash command must be a string')
  }
  const input = raw.trim()
  if (!input.startsWith('/')) {
    return { raw, error: 'Slash commands must start with `/`.' }
  }
  const body = input.slice(1).trim()
  if (!body) {
    return { raw, error: 'Slash command name is required.' }
  }
  const separator = body.search(/\s/u)
  const rawToken = separator < 0 ? body : body.slice(0, separator)
  const args = separator < 0 ? '' : body.slice(separator).trim()
  const token = (rawToken.split('@', 1)[0] ?? '').trim().toLowerCase()
  if (!token) {
    return { raw, error: 'Slash command name is required.' }
  }
  const normalized = token === 'h' ? 'help' : token
  const command = resolveCommand(`/${normalized}`)
  return {
    raw,
    args,
    name: command?.name ?? normalized,
    ...(command === undefined ? {} : { command }),
  }
}

/**
 * Native bridge slash router with no Python fallback, shell subprocess, or
 * ambient runtime state. Any action that crosses a session/provider/process
 * boundary is supplied by an explicit host port.
 */
export class BridgeSlashRouter {
  private readonly config: BridgeSlashConfig
  private readonly cwd: string
  private readonly host: BridgeSlashRouterHost
  private readonly platform: NodeJS.Platform | undefined
  private readonly state: AgentState

  constructor(options: BridgeSlashRouterOptions) {
    if (!options.cwd.trim()) {
      throw new TypeError('cwd must be a non-empty string')
    }
    this.config = options.config
    this.cwd = options.cwd
    this.host = options.host ?? {}
    this.platform = options.platform
    this.state = options.state
  }

  async dispatch(raw: string): Promise<BridgeSlashResult> {
    const parsed = parseBridgeSlashCommand(raw)
    if ('error' in parsed) {
      return result('', parsed.error, 'invalid')
    }

    switch (parsed.name) {
      case 'help':
      case 'commands':
        return result(parsed.name, helpText(), 'handled')
      case 'model':
        return this.model(parsed)
      case 'provider':
        return this.provider(parsed)
      case 'cost':
        return this.cost(parsed)
      case 'history':
        return result(parsed.name, `${this.state.messages.length} messages, ${this.state.turnCount} turns`, 'handled')
      case 'verbose':
      case 'debug':
        return this.toggle(parsed.name)
      case 'thinking':
      case 'reasoning':
        return this.thinking(parsed)
      case 'sampling':
        return this.sampling(parsed)
      case 'compact':
        return this.compact(parsed)
      case 'skills':
        return this.skills(parsed)
      case 'skill':
        return this.invokeSkill(parsed, parsed.args)
      case 'skill-create':
        return this.skillCreate(parsed)
      case 'clear':
        return this.clear(parsed)
      case 'context':
      case 'usage':
        return this.context(parsed)
      case 'config':
        return result(parsed.name, configText(this.config), 'handled')
      case 'permissions':
        return this.permissions(parsed)
      case 'yolo':
        return this.yolo(parsed)
      case 'tools':
        return this.tools(parsed)
      case 'plan':
        return this.plan(parsed)
      case 'agents':
        return this.agents(parsed)
      case 'exit':
        return this.exit(parsed)
      default:
        return this.unknownOrSkill(parsed)
    }
  }

  private async model(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const requested = parsed.args.trim()
    if (requested) {
      const models = this.host.models
      if (models === undefined) {
        return unavailable(parsed.name, 'Model switching requires a configured model host.')
      }
      const previousModel = configString(this.config, 'model')
      await models.switchModel(requested)
      const updateError = await this.updateConfig(
        config => { config.model = requested },
        // The host switch already happened; keep the host and the local
        // config from diverging when the runtime rejects the commit.
        previousModel
          ? async () => { await models.switchModel(previousModel) }
          : undefined,
      )
      return updateError === undefined
        ? result(parsed.name, `Model set to: ${requested}`, 'handled')
        : result(parsed.name, updateError, 'unavailable')
    }

    const current = configString(this.config, 'model') || '(none)'
    const models = this.host.models
    if (models === undefined) {
      return result(parsed.name, `Current model: ${current}`, 'handled')
    }
    const available = await models.list({
      currentModel: current === '(none)' ? '' : current,
      baseUrl: configString(this.config, 'base_url') || configString(this.config, 'custom_base_url'),
      apiKey: configString(this.config, 'api_key'),
    })
    if (!available.length) {
      return result(
        parsed.name,
        `Current model: ${current}\nNo models returned by the configured model host.`,
        'handled',
      )
    }
    const lines = [`Current model: ${current}`, '', `Available models (${available.length}):`]
    for (const name of available) lines.push(`  ${name}${name === current ? ' (active)' : ''}`)
    lines.push('', 'Use /model <name> to switch')
    return result(parsed.name, lines.join('\n'), 'handled')
  }

  private async provider(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const providers = this.host.providers
    if (providers === undefined) {
      return unavailable(parsed.name, 'Provider profile management requires a configured provider host.')
    }
    const requested = parsed.args.trim()
    if (requested) {
      const previousProfile = await providers.active()
      const profile = await providers.select(requested)
      if (profile === undefined) {
        return result(parsed.name, `Could not switch to '${requested}'.`, 'unknown')
      }
      const updateError = await this.updateConfig(
        config => applyProfile(config, profile),
        // Re-select the previously active host profile when the runtime
        // rejects the commit so host and router state cannot diverge.
        previousProfile !== undefined && previousProfile.name !== profile.name
          ? async () => { await providers.select(previousProfile.name) }
          : undefined,
      )
      return updateError === undefined
        ? result(parsed.name, `Switched to '${profile.name}'  (model: ${profile.model})`, 'handled')
        : result(parsed.name, updateError, 'unavailable')
    }

    const profiles = await providers.list()
    if (!profiles.length) {
      return result(parsed.name, [
        'No provider profiles configured.',
        'Configure a provider profile before selecting one.',
      ].join('\n'), 'handled')
    }
    const active = await providers.active()
    const activeName = active?.name ?? profiles.find(profile => profile.active)?.name ?? ''
    const lines = ['Provider profiles:']
    for (const profile of profiles) {
      const marker = profile.name === activeName ? '*' : ' '
      lines.push(`  ${marker} ${profile.name.padEnd(20)}  ${profile.model}  (${profile.base_url})`)
    }
    lines.push('', '* = active. Pass a profile name to switch: /provider NAME')
    return result(parsed.name, lines.join('\n'), 'handled')
  }

  private cost(parsed: ParsedBridgeSlashCommand): BridgeSlashResult {
    const tracker = this.host.costTracker
    if (tracker === undefined) {
      return unavailable(parsed.name, 'Cost reporting requires a session cost tracker.')
    }
    return result(parsed.name, tracker.summary(), 'handled')
  }

  private async toggle(name: 'debug' | 'verbose'): Promise<BridgeSlashResult> {
    const next = !Boolean(this.config[name])
    const updateError = await this.updateConfig(config => { config[name] = next })
    return updateError === undefined
      ? result(name, `${capitalize(name)}: ${next}`, 'handled')
      : result(name, updateError, 'unavailable')
  }

  private async thinking(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const target = parsed.args.trim().toLowerCase()
    if (!target) {
      const effort = configString(this.config, 'reasoning_effort') || 'off'
      return result(parsed.name, `Thinking: ${effort}  (levels: off | low | medium | high)`, 'handled')
    }
    if (!THINKING_LEVELS.has(target)) {
      return result(parsed.name, `Unknown thinking level: ${target}. Use off|low|medium|high.`, 'invalid')
    }
    const updateError = await this.updateConfig(config => {
      config.reasoning_effort = target
      config.thinking = target !== 'off'
    })
    if (updateError !== undefined) {
      return result(parsed.name, updateError, 'unavailable')
    }
    const profile = await this.host.providers?.active()
    if (profile !== undefined) {
      const saved = await this.host.providers?.saveSampling(profile.name, {
        thinking: target !== 'off',
        reasoning_effort: target === 'off' ? null : target,
        thinking_budget: target === 'off' ? 0 : this.config.thinking_budget ?? 0,
      })
      if (saved === undefined) {
        return result(
          parsed.name,
          `Thinking effort set to: ${target}\nCould not persist the active profile.`,
          'unavailable',
        )
      }
    }
    return result(parsed.name, `Thinking effort set to: ${target}`, 'handled')
  }

  private async sampling(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const input = parsed.args.trim()
    if (!input) {
      const lines = ['Sampling parameters (current session):']
      for (const parameter of [...SAMPLING_PARAMS].sort()) {
        const current = this.config[parameter]
        lines.push(`  ${parameter}: ${current === undefined ? '(default)' : String(current)}`)
      }
      lines.push(
        '',
        'Usage: /sampling <param> <value>',
        '       /sampling reset',
        '       /sampling save  (persist to active profile)',
      )
      return result(parsed.name, lines.join('\n'), 'handled')
    }
    const [subcommand, ...rest] = input.split(/\s+/u)
    const name = (subcommand ?? '').toLowerCase()
    if (name === 'reset') {
      const updateError = await this.updateConfig(config => {
        for (const parameter of SAMPLING_PARAMS) delete config[parameter]
      })
      return updateError === undefined
        ? result(parsed.name, 'Sampling parameters reset to defaults.', 'handled')
        : result(parsed.name, updateError, 'unavailable')
    }
    if (name === 'save') {
      const providers = this.host.providers
      if (providers === undefined) {
        return unavailable(parsed.name, 'Saving sampling requires a configured provider host.')
      }
      const profile = await providers.active()
      if (profile === undefined) {
        return result(parsed.name, 'No active profile. Run /provider first.', 'unavailable')
      }
      const sampling: Record<string, unknown> = {}
      for (const parameter of SAMPLING_PARAMS) {
        if (this.config[parameter] !== undefined) sampling[parameter] = this.config[parameter]
      }
      const saved = await providers.saveSampling(profile.name, sampling)
      return saved === undefined
        ? result(parsed.name, `Could not save sampling parameters to '${profile.name}'.`, 'unavailable')
        : result(parsed.name, `Sampling parameters saved to profile '${profile.name}'.`, 'handled')
    }
    const value = rest.join(' ')
    if (!value) {
      return result(parsed.name, `Usage: /sampling <param> <value>\nValid params: ${samplingNames()}`, 'invalid')
    }
    if (!SAMPLING_PARAMS.has(name)) {
      return result(parsed.name, `Unknown param: ${name}\nValid: ${samplingNames()}`, 'invalid')
    }
    const parsedValue = parseSamplingValue(name, value)
    if (parsedValue === undefined) {
      return result(parsed.name, `Invalid value: ${value}`, 'invalid')
    }
    const updateError = await this.updateConfig(config => { config[name] = parsedValue })
    return updateError === undefined
      ? result(parsed.name, `${name} = ${parsedValue}`, 'handled')
      : result(parsed.name, updateError, 'unavailable')
  }

  private async compact(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const compact = this.host.compact
    if (compact === undefined) {
      return unavailable(parsed.name, 'Conversation compaction requires a configured session host.')
    }
    if (this.state.messages.length < 4) {
      return result(parsed.name, 'Nothing to compact (fewer than 4 messages).', 'handled')
    }
    if (!configString(this.config, 'model')) {
      return result(parsed.name, 'No model configured. Run /provider first.', 'handled')
    }
    const originalCount = this.state.messages.length
    const compacted = await compact(this.state)
    if (!compacted.compacted) {
      const detail = compacted.error ? ` (${compacted.error})` : ''
      return result(parsed.name, `Compaction skipped: ${compacted.reason ?? 'nothing_to_compact'}${detail}.`, 'handled')
    }
    const kept = compacted.keptCount ?? this.state.messages.length
    const summarized = compacted.summarizedCount ?? Math.max(0, originalCount - kept)
    return result(parsed.name, [
      `Compacted ${originalCount} messages -> ${this.state.messages.length} messages.`,
      `Summarized ${summarized} older messages, kept ${kept} live messages.`,
    ].join('\n'), 'handled')
  }

  private async skills(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const skills = this.host.skills
    if (skills === undefined) {
      return unavailable(parsed.name, 'Skill discovery requires a configured skills host.')
    }
    await skills.discover?.()
    const all = skills.registry.all()
    if (!all.length) {
      return result(parsed.name, 'No skills found. Create one with /skill-create.', 'handled')
    }
    const lines = [`Skills (${all.length}):`]
    for (const skill of all) {
      const tags = skill.metadata.tags.length ? ` [${skill.metadata.tags.join(', ')}]` : ''
      lines.push(`  ${skill.metadata.name}${tags} — ${skill.metadata.description || 'No description'}`)
    }
    lines.push('', 'Use /skill <name> to invoke a skill')
    return result(parsed.name, lines.join('\n'), 'handled')
  }

  private async skillCreate(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const name = parsed.args.trim()
    if (!name) {
      return result(parsed.name, 'Usage: /skill-create <name>', 'invalid')
    }
    if (!SKILL_NAME.test(name)) {
      return result(
        parsed.name,
        `Invalid skill name '${name}'. Use only letters, numbers, hyphens, and underscores.`,
        'invalid',
      )
    }
    const startCreate = this.host.skills?.startCreate
    if (startCreate === undefined) {
      return unavailable(parsed.name, 'Skill creation requires a configured native skill-authoring flow.')
    }
    const output = await startCreate(name)
    return result(parsed.name, output || `Creating skill '${name}'.`, 'handled')
  }

  private clear(parsed: ParsedBridgeSlashCommand): BridgeSlashResult {
    this.state.messages.length = 0
    this.state.thinkingContent.length = 0
    this.state.toolExecutions.length = 0
    this.state.turnCount = 0
    // Usage counters belong to the cleared conversation too; leaving them
    // would make /usage and /context report stale cumulative cost.
    this.state.totalApiCalls = 0
    this.state.totalCacheCreationTokens = 0
    this.state.totalCacheReadTokens = 0
    this.state.totalInputTokens = 0
    this.state.totalOutputTokens = 0
    this.state.usageComplete = true
    return result(parsed.name, 'Conversation cleared.', 'handled')
  }

  private context(parsed: ParsedBridgeSlashCommand): BridgeSlashResult {
    const model = configString(this.config, 'model')
    const provider = resolveProvider(model, this.config)
    const cost = calcCost(model, this.state.totalInputTokens, this.state.totalOutputTokens)
    return result(parsed.name, [
      `CWD: ${this.cwd}`,
      `Model: ${model}`,
      `Provider: ${provider}`,
      `Turns: ${this.state.turnCount}`,
      `Messages: ${this.state.messages.length}`,
      `Tokens: ${this.state.totalInputTokens} in / ${this.state.totalOutputTokens} out`,
      `Cost: $${cost.toFixed(4)}`,
    ].join('\n'), 'handled')
  }

  private async permissions(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const current = configString(this.config, 'permission_mode') || 'accept-all'
    const index = PERMISSION_MODES.indexOf(current as typeof PERMISSION_MODES[number])
    const next = PERMISSION_MODES[index < 0 ? 0 : (index + 1) % PERMISSION_MODES.length] ?? 'auto'
    const updateError = await this.updateConfig(config => { config.permission_mode = next })
    return updateError === undefined
      ? result(parsed.name, `Permission mode: ${next}`, 'handled')
      : result(parsed.name, updateError, 'unavailable')
  }

  private async yolo(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const current = configString(this.config, 'permission_mode') || 'accept-all'
    const next = current === 'accept-all' ? 'auto' : 'accept-all'
    const updateError = await this.updateConfig(config => { config.permission_mode = next })
    return updateError === undefined
      ? result(parsed.name, `YOLO mode ${next === 'accept-all' ? 'ON (accept-all)' : 'OFF (auto)'}`, 'handled')
      : result(parsed.name, updateError, 'unavailable')
  }

  private async tools(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const tools = this.host.tools
    if (tools === undefined) {
      return unavailable(parsed.name, 'Tool listing requires a configured execution registry host.')
    }
    const entries = await tools()
    const lines = entries.map(entry => (
      `  ${entry.name}${entry.safe ? ' [safe]' : ''} -- ${entry.description.slice(0, 60)}`
    ))
    lines.push(`  (${entries.length} total)`)
    return result(parsed.name, lines.join('\n'), 'handled')
  }

  private async plan(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const objective = parsed.args.trim()
    if (!objective) {
      return result(
        parsed.name,
        'Usage: /plan <objective>\n\nExample: /plan refactor the auth module into separate files',
        'invalid',
      )
    }
    const plan = this.host.plan
    if (plan === undefined) {
      return unavailable(parsed.name, 'Planning requires a configured native plan host.')
    }
    return result(parsed.name, await plan(objective), 'handled')
  }

  private async agents(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const agents = this.host.agents
    if (agents === undefined) {
      return unavailable(parsed.name, 'Agent listing requires a configured agent host.')
    }
    const entries = await agents()
    if (!entries.length) {
      return result(parsed.name, 'No agent types registered.', 'handled')
    }
    const lines = [`Agent types (${entries.length}):`]
    for (const agent of entries) {
      const source = agent.source && agent.source !== 'built-in' ? ` [${agent.source}]` : ''
      lines.push(`  ${agent.name}${source} — ${agent.description}`)
    }
    return result(parsed.name, lines.join('\n'), 'handled')
  }

  private async exit(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    const exit = this.host.exit
    if (exit === undefined) {
      return unavailable(parsed.name, 'Exit requires a host-owned shutdown handler.')
    }
    await exit()
    return result(parsed.name, 'Exit requested.', 'handled')
  }

  private async unknownOrSkill(parsed: ParsedBridgeSlashCommand): Promise<BridgeSlashResult> {
    if (parsed.command !== undefined) {
      return unavailable(parsed.name, `/${parsed.name} needs a host-owned implementation that is not configured here.`)
    }
    const skills = this.host.skills
    const skill = skills?.registry.get(parsed.name)
    if (skill === undefined) {
      return result(parsed.name, `Unknown command: /${parsed.name} (type /help)`, 'unknown')
    }
    return this.invokeSkill(parsed, `${parsed.name}${parsed.args ? `:${parsed.args}` : ''}`)
  }

  private async invokeSkill(parsed: ParsedBridgeSlashCommand, raw: string): Promise<BridgeSlashResult> {
    const skills = this.host.skills
    if (skills === undefined) {
      return unavailable(parsed.name, 'Skill invocation requires a configured skills host.')
    }
    const invocation = parseSkillInvocation(raw)
    if (invocation === undefined) {
      return result(parsed.name, 'Usage: /skill <name>[:<args>]\nUse /skills to list available skills.', 'invalid')
    }
    const skill = skills.registry.get(invocation.name)
    if (skill === undefined) {
      const suggestions = skills.registry.search(invocation.name).slice(0, 5).map(match => match.metadata.name)
      const output = suggestions.length
        ? `Skill '${invocation.name}' not found. Did you mean: ${suggestions.join(', ')}`
        : `Skill '${invocation.name}' not found. Use /skills to list available skills.`
      return result(parsed.name, output, 'unknown')
    }
    if (this.platform !== undefined && !skillMatchesPlatform(skill, this.platform)) {
      return result(
        parsed.name,
        `Skill '${invocation.name}' is not compatible with this platform (${this.platform}).`,
        'unavailable',
      )
    }
    const output = await skills.invoke({ name: invocation.name, args: invocation.args, skill })
    return result(parsed.name, output || `Running skill '${invocation.name}'...`, 'handled')
  }

  /**
   * Apply a config mutation and commit it through the host port. When the
   * commit fails the local config is restored, and an already-applied host
   * transition (model switch, profile select) is reverted through
   * `revertHost` so the router and runtime cannot stay permanently diverged.
   */
  private async updateConfig(
    mutator: (config: BridgeSlashConfig) => void,
    revertHost?: () => Promise<void> | void,
  ): Promise<string | undefined> {
    const previous = { ...this.config }
    mutator(this.config)
    const changed = this.host.configChanged
    if (changed === undefined) return undefined
    try {
      await changed(Object.freeze({ ...this.config }))
      return undefined
    } catch (error) {
      replaceConfig(this.config, previous)
      let message = `Could not apply configuration: ${errorMessage(error)}`
      if (revertHost !== undefined) {
        try {
          await revertHost()
        } catch (revertError) {
          message += ` Host state could not be restored: ${errorMessage(revertError)}`
        }
      }
      return message
    }
  }
}

function helpText(): string {
  return ['Commands:', ...COMMAND_REGISTRY.filter(command => ROUTED_COMMANDS.has(command.name)).map(command => {
    const args = command.argsHint ? ` ${command.argsHint}` : ''
    return `  /${command.name}${args}  ${command.description}`
  })].join('\n')
}

function applyProfile(config: BridgeSlashConfig, profile: BridgeProviderProfile): void {
  config.model = profile.model
  config.provider = profile.provider
  config.base_url = profile.base_url
  if (profile.api_key) {
    config.api_key = profile.api_key
  } else {
    delete config.api_key
  }
  for (const [name, value] of Object.entries(profile.sampling)) config[name] = value
}

function configText(config: BridgeSlashConfig): string {
  const entries = Object.entries(config)
    .filter(([name]) => !name.startsWith('_'))
    .sort(([left], [right]) => left.localeCompare(right))
  if (!entries.length) return '(empty config)'
  return entries
    .map(([name, value]) => `  ${name}: ${SENSITIVE_CONFIG_NAME.test(name) ? redactSecret(value) : String(value)}`)
    .join('\n')
}

function redactSecret(value: unknown): string {
  const text = typeof value === 'string' ? value : String(value ?? '')
  return text ? '********' : '(not set)'
}

function configString(config: BridgeSlashConfig, name: string): string {
  const value = config[name]
  return typeof value === 'string' ? value : ''
}

function parseSamplingValue(parameter: string, raw: string): unknown {
  switch (parameter) {
    case 'reasoning_effort': {
      const level = raw.trim().toLowerCase()
      return THINKING_LEVELS.has(level) ? level : undefined
    }
    case 'thinking':
      return BOOLEAN_SAMPLING_VALUES.get(raw.trim().toLowerCase())
    case 'thinking_budget':
      return numericSamplingValue(raw, { integer: true, min: 0 })
    case 'max_tokens':
      return numericSamplingValue(raw, { integer: true, min: 1 })
    case 'top_k':
      return numericSamplingValue(raw, { integer: true, min: 0 })
    case 'temperature':
      return numericSamplingValue(raw, { max: 2, min: 0 })
    case 'top_p':
      return numericSamplingValue(raw, { exclusiveMin: true, max: 1, min: 0 })
    case 'min_p':
      return numericSamplingValue(raw, { max: 1, min: 0 })
    case 'frequency_penalty':
    case 'presence_penalty':
      return numericSamplingValue(raw, { max: 2, min: -2 })
    case 'repetition_penalty':
      return numericSamplingValue(raw, { exclusiveMin: true, min: 0 })
    default:
      return undefined
  }
}

function numericSamplingValue(
  raw: string,
  options: {
    readonly exclusiveMin?: boolean
    readonly integer?: boolean
    readonly max?: number
    readonly min?: number
  },
): number | undefined {
  if (options.integer === true && !/^[+-]?\d+$/u.test(raw)) {
    return undefined
  }
  const value = Number(raw)
  if (!Number.isFinite(value) || (options.integer === true && !Number.isSafeInteger(value))) {
    return undefined
  }
  if (options.min !== undefined && (options.exclusiveMin === true ? value <= options.min : value < options.min)) {
    return undefined
  }
  if (options.max !== undefined && value > options.max) {
    return undefined
  }
  return value
}

function parseSkillInvocation(raw: string): { readonly args: string; readonly name: string } | undefined {
  const input = raw.trim()
  if (!input) return undefined
  const separator = input.indexOf(':')
  const name = (separator < 0 ? input : input.slice(0, separator)).trim()
  if (!name) return undefined
  return { name, args: separator < 0 ? '' : input.slice(separator + 1).trim() }
}

function replaceConfig(target: BridgeSlashConfig, source: BridgeSlashConfig): void {
  for (const name of Object.keys(target)) delete target[name]
  Object.assign(target, source)
}

function samplingNames(): string {
  return [...SAMPLING_PARAMS].sort().join(', ')
}

function capitalize(value: string): string {
  return value.slice(0, 1).toUpperCase() + value.slice(1)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function result(command: string, output: string, status: BridgeSlashStatus): BridgeSlashResult {
  return Object.freeze({ command, output, status })
}

function unavailable(command: string, output: string): BridgeSlashResult {
  return result(command, output, 'unavailable')
}
