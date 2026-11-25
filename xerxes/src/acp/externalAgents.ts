// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { accessSync, constants as fsConstants, readdirSync, readFileSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, delimiter, isAbsolute, join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { errorMessage } from '../executors/toolRegistry.js'
import type {
  AcpAvailableCommand,
  AcpClientEvent,
  AcpContentBlock,
  AcpElicitationField,
  AcpElicitationFieldType,
  AcpElicitationOption,
  AcpElicitationRequest,
  AcpElicitationResponse,
  AcpPlanEntry,
  AcpUsageCost,
} from './client.js'
import { isAcpRecord } from './types.js'

export type { AcpAvailableCommand, AcpContentBlock, AcpUsageCost } from './client.js'

export type AcpExternalAgentProvider = 'xerxes' | 'claude-code' | 'antigravity' | 'codex' | 'acp'
export type AcpExternalAgentTransport = 'native' | 'acp' | 'claude-stream-json'

export interface AcpExternalAgentInfo {
  readonly id: string
  readonly name: string
  readonly provider: AcpExternalAgentProvider
  readonly transport: AcpExternalAgentTransport
  readonly available: boolean
  readonly command?: string
  readonly args?: readonly string[]
  readonly binaryPath?: string
  readonly version?: string
  readonly installHint?: string
  readonly description: string
}

export interface AcpExternalModelOption {
  readonly id: string
  readonly name: string
  readonly description?: string
}

export interface AcpExternalModeOption {
  readonly id: string
  readonly name: string
  readonly description?: string
}

export interface AcpExternalConfigOptionChoice {
  readonly value: string
  readonly name: string
  readonly description?: string
  readonly group?: string
}

export interface AcpExternalConfigOption {
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly category?: string
  readonly type: 'select' | 'boolean' | string
  readonly current_value: string | boolean
  readonly options: readonly AcpExternalConfigOptionChoice[]
}

export type AcpExternalCommandOption = AcpAvailableCommand

export interface AcpRuntimePromptCapabilities {
  readonly image: boolean
  readonly audio: boolean
  readonly embedded_context: boolean
}

export interface AcpRuntimeSessionCapabilities {
  readonly load: boolean
  readonly resume: boolean
  readonly fork: boolean
  readonly list: boolean
  readonly close: boolean
  readonly additional_directories: boolean
}

export interface AcpRuntimeMcpCapabilities {
  readonly http: boolean
  readonly sse: boolean
}

export interface AcpRuntimeAuthMethod {
  readonly args?: readonly string[]
  readonly description?: string
  readonly env?: Readonly<Record<string, string>>
  readonly id: string
  readonly meta?: Readonly<Record<string, unknown>>
  readonly name: string
  readonly type?: 'agent' | 'terminal' | string
}

export interface AcpRuntimeCapabilities {
  readonly prompt: AcpRuntimePromptCapabilities
  readonly session: AcpRuntimeSessionCapabilities
  readonly mcp: AcpRuntimeMcpCapabilities
  readonly auth_methods: readonly AcpRuntimeAuthMethod[]
  readonly meta?: Readonly<Record<string, unknown>>
}

export interface AcpDiscoveredSkill {
  readonly name: string
  readonly description?: string
  readonly source?: string
  readonly path?: string
  readonly slash_command?: string
}

export interface AcpBridgeDiff {
  readonly path: string
  readonly old_text?: string
  readonly new_text?: string
}

export interface AcpBridgeLocation {
  readonly path: string
  readonly line?: number
}

export const DEFAULT_CLAUDE_CODE_MODELS: readonly AcpExternalModelOption[] = Object.freeze([
  {
    id: 'claude-opus-4-6',
    name: 'Claude Opus 4.6',
    description: 'Most capable Claude model for complex reasoning and engineering',
  },
  {
    id: 'claude-sonnet-4-6',
    name: 'Claude Sonnet 4.6',
    description: 'Balanced speed and intelligence for everyday coding',
  },
  {
    id: 'claude-haiku-4-5',
    name: 'Claude Haiku 4.5',
    description: 'Fastest, lightest Claude model for quick tasks',
  },
  {
    id: 'opus',
    name: 'Opus',
    description: 'Claude Code CLI alias for the latest Opus model',
  },
  {
    id: 'sonnet',
    name: 'Sonnet',
    description: 'Claude Code CLI alias for the latest Sonnet model',
  },
  {
    id: 'haiku',
    name: 'Haiku',
    description: 'Claude Code CLI alias for the latest Haiku model',
  },
])

export const ACP_TURBO_MODE_ID = 'turbo'

export const ACP_TURBO_MODE_OPTION: AcpExternalModeOption = Object.freeze({
  id: ACP_TURBO_MODE_ID,
  name: 'Turbo',
  description: 'Run all commands without restrictions.',
})

const NATIVE_UNRESTRICTED_ACP_MODE_KEYS: ReadonlySet<string> = new Set([
  'turbo',
  'bypasspermissions',
  'bypass',
  'yolo',
  'unrestricted',
  'acceptall',
  'dangerouslyskippermissions',
])

/** Return true if a mode ID represents an unrestricted / bypass-permissions execution mode. */
export function isUnrestrictedAcpModeId(modeId: string | undefined): boolean {
  if (!modeId) return false
  const normalized = modeId.trim().toLowerCase().replace(/[\s_-]+/g, '')
  return NATIVE_UNRESTRICTED_ACP_MODE_KEYS.has(normalized)
}

/** Return true when a connected stdio ACP binding is currently operating in host-enforced or native Turbo mode. */
export function isAcpTurboMode(binding: AcpSessionBinding | null | undefined): boolean {
  if (!binding || !binding.connected || binding.transport !== 'acp') {
    return false
  }
  if (binding.current_mode?.trim().toLowerCase() === ACP_TURBO_MODE_ID) {
    return true
  }
  const modeCfgId = binding.mode_config_id?.toLowerCase()
  const modeOption = binding.config_options?.find(
    opt =>
      (modeCfgId !== undefined && opt.id.toLowerCase() === modeCfgId) ||
      opt.category?.toLowerCase() === 'mode' ||
      opt.id.toLowerCase() === 'mode',
  )
  return (
    typeof modeOption?.current_value === 'string' &&
    modeOption.current_value.trim().toLowerCase() === ACP_TURBO_MODE_ID
  )
}

export interface AcpSessionBinding {
  readonly session_id: string
  readonly agent_id: string
  readonly name: string
  readonly provider: AcpExternalAgentProvider
  readonly transport: AcpExternalAgentTransport
  readonly cwd: string
  readonly external_session_id?: string
  readonly connected: boolean
  readonly command?: string
  readonly error?: string
  readonly agent_version?: string
  readonly agent_title?: string
  readonly capabilities?: AcpRuntimeCapabilities
  readonly current_model?: string
  readonly current_model_name?: string
  readonly models?: readonly AcpExternalModelOption[]
  readonly current_mode?: string
  readonly current_mode_name?: string
  readonly modes?: readonly AcpExternalModeOption[]
  readonly model_config_id?: string
  readonly mode_config_id?: string
  readonly config_options?: readonly AcpExternalConfigOption[]
  readonly thought_level?: string
  readonly thought_levels?: readonly AcpExternalModeOption[]
  readonly thought_level_config_id?: string
  readonly available_commands?: readonly AcpExternalCommandOption[]
  readonly discovered_skills?: readonly AcpDiscoveredSkill[]
  readonly additional_directories?: readonly string[]
  readonly sandbox?: boolean
  readonly agent_override?: string
  readonly mcp_server_names?: readonly string[]
}

/** Unified streaming event union shared by ACP stdio agents and Claude stream-json. */
export type AcpBridgeStreamEvent =
  | { readonly externalSessionId: string; readonly model?: string; readonly type: 'session_init' }
  | {
    readonly contentBlocks?: readonly AcpContentBlock[]
    readonly meta?: Readonly<Record<string, unknown>>
    readonly text: string
    readonly type: 'text_delta'
  }
  | {
    readonly contentBlocks?: readonly AcpContentBlock[]
    readonly text: string
    readonly type: 'user_message_chunk'
  }
  | { readonly text: string; readonly type: 'thought_delta' }
  | {
    readonly argumentsText: string
    readonly content?: string
    readonly contentBlocks?: readonly AcpContentBlock[]
    readonly diff?: AcpBridgeDiff
    readonly diffs?: readonly AcpBridgeDiff[]
    readonly kind?: string
    readonly locations?: readonly AcpBridgeLocation[]
    readonly meta?: Readonly<Record<string, unknown>>
    readonly name: string
    readonly rawInput?: unknown
    readonly status?: string
    readonly title?: string
    readonly toolCallId: string
    readonly type: 'tool_call'
  }
  | {
    readonly argumentsText?: string
    readonly contentBlocks?: readonly AcpContentBlock[]
    readonly diff?: AcpBridgeDiff
    readonly diffs?: readonly AcpBridgeDiff[]
    readonly kind?: string
    readonly locations?: readonly AcpBridgeLocation[]
    readonly meta?: Readonly<Record<string, unknown>>
    readonly name: string
    readonly output?: string
    readonly rawInput?: unknown
    readonly status?: string
    readonly title?: string
    readonly toolCallId: string
    readonly type: 'tool_update'
  }
  | {
    readonly contentBlocks?: readonly AcpContentBlock[]
    readonly diff?: AcpBridgeDiff
    readonly diffs?: readonly AcpBridgeDiff[]
    readonly durationMs?: number
    readonly isError?: boolean
    readonly kind?: string
    readonly locations?: readonly AcpBridgeLocation[]
    readonly meta?: Readonly<Record<string, unknown>>
    readonly name: string
    readonly output: string
    readonly permissionDenied?: boolean
    readonly rawOutput?: unknown
    readonly title?: string
    readonly toolCallId: string
    readonly type: 'tool_result'
  }
  | { readonly entries: readonly AcpPlanEntry[]; readonly type: 'plan' }
  | { readonly focus: string; readonly type: 'focus' }
  | { readonly modeId: string; readonly modeName?: string; readonly type: 'mode_update' }
  | { readonly modelId: string; readonly modelName?: string; readonly type: 'model_update' }
  | {
    readonly configId?: string
    readonly configOptions?: readonly unknown[]
    readonly raw: Readonly<Record<string, unknown>>
    readonly type: 'config_update'
    readonly value?: string | boolean
  }
  | { readonly commands: readonly AcpExternalCommandOption[]; readonly type: 'available_commands' }
  | { readonly title?: string; readonly type: 'session_info'; readonly updatedAt?: string }
  | { readonly description?: string; readonly severity: string; readonly title: string; readonly type: 'notice' }
  | { readonly elicitationId: string; readonly type: 'elicitation_complete' }
  | { readonly events: readonly AcpBridgeStreamEvent[]; readonly type: 'session_replay' }
  | { readonly raw: Readonly<Record<string, unknown>>; readonly text?: string; readonly type: 'continuation' }
  | {
    readonly cachedReadTokens?: number
    readonly cachedWriteTokens?: number
    readonly contextSize?: number
    readonly contextUsed?: number
    readonly cost?: AcpUsageCost
    readonly costUsd?: number
    readonly inputTokens?: number
    readonly outputTokens?: number
    readonly thoughtTokens?: number
    readonly totalTokens?: number
    readonly type: 'usage'
  }
  | { readonly message: string; readonly type: 'error' }

export interface CustomAcpAgentConfig {
  readonly args?: readonly string[]
  readonly command: string
  readonly description?: string
  readonly env?: Readonly<Record<string, string>>
  readonly id: string
  readonly name: string
  readonly provider?: AcpExternalAgentProvider
}

export interface DiscoverExternalAcpAgentsOptions {
  /** Override filesystem executable check (deterministic tests). */
  readonly binaryExists?: (filePath: string) => boolean
  /** Override path to `~/.xerxes/acp-agents.json`. */
  readonly configPath?: string
  /** Environment variables map (defaults to `process.env`). */
  readonly env?: Readonly<Record<string, string | undefined>>
  /** Override home directory (defaults to `env.HOME || homedir()`). */
  readonly homeDir?: string
  /** Optional custom config reader (defaults to `readFileSync`). */
  readonly readConfigText?: (filePath: string) => string | undefined
  /** Override binary resolution on PATH and standard install dirs. */
  readonly resolveBinary?: (name: string, extraDirs: readonly string[]) => string | undefined
}

const VALID_PROVIDERS: ReadonlySet<string> = new Set([
  'xerxes',
  'claude-code',
  'antigravity',
  'codex',
  'acp',
])

const CLAUDE_INSTALL_HINT = 'Install Claude Code (claude) or @zed-industries/claude-agent-acp'
const ANTIGRAVITY_INSTALL_HINT =
  'Install Antigravity CLI (agy) so `agy acp` is on PATH, or set XERXES_ANTIGRAVITY_ACP_CMD'

/** Split a shell-like command override (`agy acp --flag`) into command + args. */
export function splitCommandString(commandLine: string): readonly string[] {
  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false
  for (const ch of commandLine.trim()) {
    if (escaped) {
      current += ch
      escaped = false
      continue
    }
    if (ch === '\\' && quote !== "'") {
      escaped = true
      continue
    }
    if (quote) {
      if (ch === quote) {
        quote = null
      } else {
        current += ch
      }
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (/\s/.test(ch)) {
      if (current.length > 0) {
        tokens.push(current)
        current = ''
      }
      continue
    }
    current += ch
  }
  if (current.length > 0) {
    tokens.push(current)
  }
  return tokens
}

function defaultBinaryExists(filePath: string): boolean {
  try {
    accessSync(filePath, process.platform === 'win32' ? fsConstants.F_OK : fsConstants.X_OK)
    return true
  } catch {
    return false
  }
}

/** Resolve a command name or explicit path against PATH and standard install locations. */
export function resolveExecutablePath(
  command: string,
  options: {
    readonly binaryExists?: (filePath: string) => boolean
    readonly env?: Readonly<Record<string, string | undefined>>
    readonly extraDirs?: readonly string[]
  } = {},
): string | undefined {
  const trimmed = command.trim()
  if (!trimmed) {
    return undefined
  }
  const exists = options.binaryExists ?? defaultBinaryExists
  const env = options.env ?? process.env
  const home = env.HOME?.trim() || homedir()

  const expanded = trimmed.startsWith('~/') || trimmed.startsWith('~\\')
    ? resolve(home, trimmed.slice(2))
    : trimmed

  if (isAbsolute(expanded) || expanded.includes('/') || expanded.includes('\\')) {
    return exists(expanded) ? expanded : undefined
  }

  const pathDirs = (env.PATH ?? '')
    .split(delimiter)
    .map(entry => entry.trim())
    .filter(Boolean)
  const searchDirs = [...new Set([...pathDirs, ...(options.extraDirs ?? [])])]
  const extensions = process.platform === 'win32' ? ['', '.exe', '.cmd', '.bat'] : ['']

  for (const dir of searchDirs) {
    for (const ext of extensions) {
      const candidate = join(dir, `${expanded}${ext}`)
      if (exists(candidate)) {
        return candidate
      }
    }
  }
  return undefined
}

/** Parse and validate custom agent declarations from `~/.xerxes/acp-agents.json`. */
export function parseCustomAcpAgentsConfig(rawJson: string): readonly CustomAcpAgentConfig[] {
  let parsed: unknown
  try {
    parsed = JSON.parse(rawJson) as unknown
  } catch {
    return []
  }
  const entries = Array.isArray(parsed)
    ? parsed
    : isAcpRecord(parsed) && Array.isArray(parsed.agents)
      ? parsed.agents
      : []
  const validated: CustomAcpAgentConfig[] = []
  const seen = new Set<string>()
  for (const item of entries) {
    if (!isAcpRecord(item)) continue
    const id = typeof item.id === 'string' ? item.id.trim() : ''
    const name = typeof item.name === 'string' ? item.name.trim() : ''
    const command = typeof item.command === 'string' ? item.command.trim() : ''
    if (!id || !name || !command || id === 'xerxes' || seen.has(id)) {
      continue
    }
    const args = Array.isArray(item.args)
      ? item.args.filter((arg): arg is string => typeof arg === 'string')
      : undefined
    let envMap: Record<string, string> | undefined
    if (isAcpRecord(item.env)) {
      const cleanEnv: Record<string, string> = {}
      for (const [key, value] of Object.entries(item.env)) {
        if (typeof value === 'string') {
          cleanEnv[key] = value
        }
      }
      if (Object.keys(cleanEnv).length > 0) {
        envMap = cleanEnv
      }
    }
    const rawProvider = typeof item.provider === 'string' ? item.provider.trim().toLowerCase() : ''
    const provider: AcpExternalAgentProvider = VALID_PROVIDERS.has(rawProvider)
      ? (rawProvider as AcpExternalAgentProvider)
      : 'acp'
    const description =
      typeof item.description === 'string' && item.description.trim()
        ? item.description.trim()
        : undefined
    seen.add(id)
    validated.push({
      id,
      name,
      command,
      ...(args && args.length > 0 ? { args } : {}),
      ...(envMap ? { env: envMap } : {}),
      provider,
      ...(description ? { description } : {}),
    })
  }
  return validated
}

/**
 * Discover available agent runtimes: built-in Xerxes, Claude Code (ACP or
 * official `claude` stream-json), Google Antigravity (`antigravity-acp`,
 * `agy acp`, or `antigravity acp`), and custom agents from
 * `~/.xerxes/acp-agents.json`.
 */
export function discoverExternalAcpAgents(
  options: DiscoverExternalAcpAgentsOptions = {},
): readonly AcpExternalAgentInfo[] {
  const env = options.env ?? process.env
  const home = options.homeDir ?? (env.HOME?.trim() || homedir())
  const bunBin = join(env.BUN_INSTALL?.trim() || join(home, '.bun'), 'bin')
  const claudeExtraDirs = [
    join(home, '.local', 'bin'),
    join(home, '.claude', 'local'),
    bunBin,
    '/opt/homebrew/bin',
    '/usr/local/bin',
  ]
  const antigravityExtraDirs = [
    join(home, '.local', 'bin'),
    join(home, '.gemini', 'antigravity', 'bin'),
    bunBin,
    '/opt/homebrew/bin',
    '/usr/local/bin',
  ]

  const resolveBin = (name: string, extraDirs: readonly string[]): string | undefined =>
    options.resolveBinary
      ? options.resolveBinary(name, extraDirs)
      : resolveExecutablePath(name, {
          ...(options.binaryExists ? { binaryExists: options.binaryExists } : {}),
          env,
          extraDirs,
        })

  const agents: AcpExternalAgentInfo[] = [
    {
      id: 'xerxes',
      name: 'Xerxes',
      provider: 'xerxes',
      transport: 'native',
      available: true,
      description: 'Native Bun multi-agent runtime & Swarm',
    },
    discoverClaudeCodeAgent(env, claudeExtraDirs, resolveBin),
    discoverAntigravityAgent(env, antigravityExtraDirs, resolveBin),
  ]

  const configPath = options.configPath ?? join(resolveXerxesHome(env, home), 'acp-agents.json')
  const rawConfig = readOptionalText(configPath, options.readConfigText)
  if (rawConfig) {
    const customConfigs = parseCustomAcpAgentsConfig(rawConfig)
    for (const cfg of customConfigs) {
      const binaryPath = resolveBin(cfg.command, claudeExtraDirs)
      const existingIdx = agents.findIndex(item => item.id === cfg.id)
      const info: AcpExternalAgentInfo = {
        id: cfg.id,
        name: cfg.name,
        provider: cfg.provider ?? 'acp',
        transport: 'acp',
        available: Boolean(binaryPath),
        command: binaryPath ?? cfg.command,
        ...(cfg.args ? { args: cfg.args } : { args: [] }),
        ...(binaryPath ? { binaryPath } : {}),
        ...(binaryPath
          ? {}
          : { installHint: `Ensure "${cfg.command}" is installed and executable on PATH.` }),
        description: cfg.description ?? `Custom stdio ACP agent (${cfg.command})`,
      }
      if (existingIdx >= 0) {
        agents[existingIdx] = info
      } else {
        agents.push(info)
      }
    }
  }

  return agents
}

function discoverClaudeCodeAgent(
  env: Readonly<Record<string, string | undefined>>,
  extraDirs: readonly string[],
  resolveBin: (name: string, extraDirs: readonly string[]) => string | undefined,
): AcpExternalAgentInfo {
  const overrideRaw = env.XERXES_CLAUDE_ACP_CMD?.trim()
  if (overrideRaw) {
    const tokens = splitCommandString(overrideRaw)
    const cmd = tokens[0] ?? ''
    const args = tokens.slice(1)
    const binaryPath = cmd ? resolveBin(cmd, extraDirs) : undefined
    return {
      id: 'claude-code',
      name: 'Claude Code',
      provider: 'claude-code',
      transport: 'acp',
      available: Boolean(binaryPath),
      ...(binaryPath ? { command: binaryPath, binaryPath } : cmd ? { command: cmd } : {}),
      args,
      ...(binaryPath
        ? {}
        : {
            installHint: `XERXES_CLAUDE_ACP_CMD binary "${cmd || overrideRaw}" was not found. ${CLAUDE_INSTALL_HINT}`,
          }),
      description: 'Claude Code via configured ACP bridge (XERXES_CLAUDE_ACP_CMD)',
    }
  }

  for (const acpBin of ['claude-agent-acp', 'claude-code-acp']) {
    const found = resolveBin(acpBin, extraDirs)
    if (found) {
      return {
        id: 'claude-code',
        name: 'Claude Code',
        provider: 'claude-code',
        transport: 'acp',
        available: true,
        command: found,
        args: [],
        binaryPath: found,
        description: `Claude Code via native ACP stdio bridge (${acpBin})`,
      }
    }
  }

  const claudeBin = resolveBin('claude', extraDirs)
  if (claudeBin) {
    return {
      id: 'claude-code',
      name: 'Claude Code',
      provider: 'claude-code',
      transport: 'claude-stream-json',
      available: true,
      command: claudeBin,
      args: ['-p', '--output-format', 'stream-json', '--verbose'],
      binaryPath: claudeBin,
      description: 'Claude Code CLI via stream-json bidirectional runner',
    }
  }

  return {
    id: 'claude-code',
    name: 'Claude Code',
    provider: 'claude-code',
    transport: 'acp',
    available: false,
    installHint: CLAUDE_INSTALL_HINT,
    description: 'Anthropic Claude Code (ACP or stream-json)',
  }
}

function discoverAntigravityAgent(
  env: Readonly<Record<string, string | undefined>>,
  extraDirs: readonly string[],
  resolveBin: (name: string, extraDirs: readonly string[]) => string | undefined,
): AcpExternalAgentInfo {
  const overrideRaw = env.XERXES_ANTIGRAVITY_ACP_CMD?.trim()
  if (overrideRaw) {
    const tokens = splitCommandString(overrideRaw)
    const cmd = tokens[0] ?? ''
    const args = tokens.slice(1)
    const binaryPath = cmd ? resolveBin(cmd, extraDirs) : undefined
    return {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: Boolean(binaryPath),
      ...(binaryPath ? { command: binaryPath, binaryPath } : cmd ? { command: cmd } : {}),
      args,
      ...(binaryPath
        ? {}
        : {
            installHint: `XERXES_ANTIGRAVITY_ACP_CMD binary "${cmd || overrideRaw}" was not found. ${ANTIGRAVITY_INSTALL_HINT}`,
          }),
      description: 'Google Antigravity via configured ACP command (XERXES_ANTIGRAVITY_ACP_CMD)',
    }
  }

  const directAcp = resolveBin('antigravity-acp', extraDirs)
  if (directAcp) {
    return {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: true,
      command: directAcp,
      args: [],
      binaryPath: directAcp,
      description: 'Google Antigravity via native ACP stdio server (antigravity-acp)',
    }
  }

  const agyBin = resolveBin('agy', extraDirs)
  if (agyBin) {
    return {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: true,
      command: agyBin,
      args: ['acp'],
      binaryPath: agyBin,
      description: 'Google Antigravity CLI via `agy acp` stdio bridge',
    }
  }

  const antigravityBin = resolveBin('antigravity', extraDirs)
  if (antigravityBin) {
    return {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: true,
      command: antigravityBin,
      args: ['acp'],
      binaryPath: antigravityBin,
      description: 'Google Antigravity CLI via `antigravity acp` stdio bridge',
    }
  }

  return {
    id: 'antigravity',
    name: 'Antigravity',
    provider: 'antigravity',
    transport: 'acp',
    available: false,
    installHint: ANTIGRAVITY_INSTALL_HINT,
    description: 'Google Antigravity ACP agent',
  }
}

function resolveXerxesHome(env: Readonly<Record<string, string | undefined>>, home: string): string {
  const override = env.XERXES_HOME?.trim()
  if (override === '~') return home
  if (override?.startsWith('~/') || override?.startsWith('~\\')) {
    return resolve(home, override.slice(2))
  }
  if (override) return resolve(override)
  return env === process.env ? xerxesHome() : join(home, '.xerxes')
}

function readOptionalText(
  filePath: string,
  customReader?: (path: string) => string | undefined,
): string | undefined {
  if (customReader) {
    try {
      return customReader(filePath)
    } catch {
      return undefined
    }
  }
  try {
    return readFileSync(filePath, 'utf8')
  } catch {
    return undefined
  }
}

export interface AcpToolMetaSnapshot {
  readonly contentBlocks?: readonly AcpContentBlock[]
  readonly diff?: AcpBridgeDiff
  readonly diffs?: readonly AcpBridgeDiff[]
  readonly kind?: string
  readonly locations?: readonly AcpBridgeLocation[]
  readonly settledOutput?: string
  readonly settledRawInput?: unknown
  readonly title?: string
}

function wireDiffOf(diff: { readonly newText?: string; readonly oldText?: string; readonly path: string } | undefined): AcpBridgeDiff | undefined {
  if (!diff) return undefined
  return {
    path: diff.path,
    ...(diff.oldText !== undefined ? { old_text: diff.oldText } : {}),
    ...(diff.newText !== undefined ? { new_text: diff.newText } : {}),
  }
}

function wireDiffsOf(
  diffs: readonly { readonly newText?: string; readonly oldText?: string; readonly path: string }[] | undefined,
  singleDiff: { readonly newText?: string; readonly oldText?: string; readonly path: string } | undefined,
): readonly AcpBridgeDiff[] | undefined {
  const source = diffs && diffs.length > 0 ? diffs : singleDiff ? [singleDiff] : []
  if (source.length === 0) return undefined
  const out: AcpBridgeDiff[] = []
  for (const item of source) {
    const mapped = wireDiffOf(item)
    if (mapped) out.push(mapped)
  }
  return out.length > 0 ? out : undefined
}

/** Convert an {@link AcpClientEvent} into a unified {@link AcpBridgeStreamEvent}. */
export function bridgeEventFromAcpClientEvent(
  event: AcpClientEvent,
  toolNamesById: Map<string, string>,
  toolMetaById?: Map<string, AcpToolMetaSnapshot>,
  settledToolCallIds?: Set<string>,
): AcpBridgeStreamEvent | undefined {
  switch (event.type) {
    case 'text_delta':
      return {
        type: 'text_delta',
        text: event.text,
        ...(event.contentBlocks?.length ? { contentBlocks: event.contentBlocks } : {}),
        ...(event.meta ? { meta: event.meta } : {}),
      }
    case 'user_message_chunk':
      return {
        type: 'user_message_chunk',
        text: event.text,
        ...(event.contentBlocks?.length ? { contentBlocks: event.contentBlocks } : {}),
      }
    case 'thought_delta':
      return { type: 'thought_delta', text: event.text }
    case 'tool_call': {
      toolNamesById.set(event.toolCallId, event.name)
      const diffs = wireDiffsOf(event.diffs, event.diff)
      const diff = diffs?.[0]
      const contentBlocks = event.contentBlocks && event.contentBlocks.length > 0 ? event.contentBlocks : undefined
      const locations = event.locations && event.locations.length > 0 ? event.locations : undefined
      if (toolMetaById && (diff || diffs || contentBlocks || locations || event.kind || event.title)) {
        const prev = toolMetaById.get(event.toolCallId)
        const mergedDiffs = diffs ?? prev?.diffs
        const mergedDiff = diff ?? mergedDiffs?.[0] ?? prev?.diff
        const mergedBlocks = contentBlocks ?? prev?.contentBlocks
        const mergedLocations = locations ?? prev?.locations
        const mergedKind = event.kind ?? prev?.kind
        const mergedTitle = event.title ?? prev?.title
        toolMetaById.set(event.toolCallId, {
          ...(mergedDiff ? { diff: mergedDiff } : {}),
          ...(mergedDiffs ? { diffs: mergedDiffs } : {}),
          ...(mergedBlocks ? { contentBlocks: mergedBlocks } : {}),
          ...(mergedLocations ? { locations: mergedLocations } : {}),
          ...(mergedKind ? { kind: mergedKind } : {}),
          ...(mergedTitle ? { title: mergedTitle } : {}),
        })
      }
      const argumentsText =
        event.rawInput === undefined
          ? ''
          : typeof event.rawInput === 'string'
            ? event.rawInput
            : safeJsonStringify(event.rawInput)
      return {
        type: 'tool_call',
        toolCallId: event.toolCallId,
        name: event.name,
        argumentsText,
        ...(event.rawInput !== undefined ? { rawInput: event.rawInput } : {}),
        ...(event.title ? { title: event.title } : {}),
        ...(event.kind ? { kind: event.kind } : {}),
        ...(event.status ? { status: event.status } : {}),
        ...(event.content !== undefined ? { content: event.content } : {}),
        ...(diff ? { diff } : {}),
        ...(diffs ? { diffs } : {}),
        ...(contentBlocks ? { contentBlocks } : {}),
        ...(locations ? { locations } : {}),
        ...(event.meta ? { meta: event.meta } : {}),
      }
    }
    case 'tool_call_update': {
      const name = event.name ?? toolNamesById.get(event.toolCallId) ?? 'tool'
      if (event.name) {
        toolNamesById.set(event.toolCallId, event.name)
      }
      const diffs = wireDiffsOf(event.diffs, event.diff)
      const diff = diffs?.[0]
      const contentBlocks = event.contentBlocks && event.contentBlocks.length > 0 ? event.contentBlocks : undefined
      const locations = event.locations && event.locations.length > 0 ? event.locations : undefined
      const prevMetaBeforeUpdate = toolMetaById?.get(event.toolCallId)
      if (toolMetaById && (diff || diffs || contentBlocks || locations || event.kind || event.title)) {
        const prev = prevMetaBeforeUpdate
        const mergedDiffs = diffs ?? prev?.diffs
        const mergedDiff = diff ?? mergedDiffs?.[0] ?? prev?.diff
        const mergedBlocks = contentBlocks ?? prev?.contentBlocks
        const mergedLocations = locations ?? prev?.locations
        const mergedKind = event.kind ?? prev?.kind
        const mergedTitle = event.title ?? prev?.title
        toolMetaById.set(event.toolCallId, {
          ...(mergedDiff ? { diff: mergedDiff } : {}),
          ...(mergedDiffs ? { diffs: mergedDiffs } : {}),
          ...(mergedBlocks ? { contentBlocks: mergedBlocks } : {}),
          ...(mergedLocations ? { locations: mergedLocations } : {}),
          ...(mergedKind ? { kind: mergedKind } : {}),
          ...(mergedTitle ? { title: mergedTitle } : {}),
          ...(prev?.settledOutput !== undefined ? { settledOutput: prev.settledOutput } : {}),
          ...(prev?.settledRawInput !== undefined ? { settledRawInput: prev.settledRawInput } : {}),
        })
      }

      const normalizedStatus = event.status?.trim().toLowerCase()
      const isNonTerminalStatus =
        normalizedStatus === 'pending' ||
        normalizedStatus === 'in_progress' ||
        normalizedStatus === 'running' ||
        normalizedStatus === 'queued' ||
        normalizedStatus === 'waiting'
      const isTerminalStatus =
        normalizedStatus === 'completed' ||
        normalizedStatus === 'failed' ||
        normalizedStatus === 'error' ||
        normalizedStatus === 'cancelled' ||
        normalizedStatus === 'canceled' ||
        normalizedStatus === 'rejected' ||
        normalizedStatus === 'denied'
      const settled =
        isTerminalStatus ||
        (!isNonTerminalStatus && (event.isError === true || event.rawOutput !== undefined || event.content !== undefined))
      const hasLatePayload =
        event.content !== undefined ||
        event.rawOutput !== undefined ||
        Boolean(diff) ||
        Boolean(diffs?.length) ||
        Boolean(contentBlocks?.length) ||
        Boolean(locations?.length) ||
        event.rawInput !== undefined ||
        Boolean(event.title)
      const buildToolUpdate = (): AcpBridgeStreamEvent => {
        const cachedMeta = toolMetaById?.get(event.toolCallId)
        const resolvedDiffs = diffs ?? cachedMeta?.diffs
        const resolvedDiff = diff ?? resolvedDiffs?.[0] ?? cachedMeta?.diff
        const resolvedBlocks = contentBlocks ?? cachedMeta?.contentBlocks
        const resolvedLocations = locations ?? cachedMeta?.locations
        const resolvedKind = event.kind ?? cachedMeta?.kind
        const resolvedTitle = event.title ?? cachedMeta?.title
        const argumentsText =
          event.rawInput === undefined
            ? undefined
            : typeof event.rawInput === 'string'
              ? event.rawInput
              : safeJsonStringify(event.rawInput)
        const updateOutput =
          event.content !== undefined
            ? event.content
            : event.rawOutput !== undefined
              ? typeof event.rawOutput === 'string'
                ? event.rawOutput
                : safeJsonStringify(event.rawOutput)
              : undefined
        return {
          type: 'tool_update',
          toolCallId: event.toolCallId,
          name,
          ...(resolvedTitle ? { title: resolvedTitle } : {}),
          ...(resolvedKind ? { kind: resolvedKind } : {}),
          ...(event.status ? { status: event.status } : {}),
          ...(event.rawInput !== undefined ? { rawInput: event.rawInput } : {}),
          ...(argumentsText !== undefined ? { argumentsText } : {}),
          ...(updateOutput !== undefined ? { output: updateOutput } : {}),
          ...(resolvedDiff ? { diff: resolvedDiff } : {}),
          ...(resolvedDiffs ? { diffs: resolvedDiffs } : {}),
          ...(resolvedBlocks ? { contentBlocks: resolvedBlocks } : {}),
          ...(resolvedLocations ? { locations: resolvedLocations } : {}),
          ...(event.meta ? { meta: event.meta } : {}),
        }
      }
      const resolvedOutputText =
        event.content !== undefined
          ? event.content
          : event.rawOutput !== undefined
            ? typeof event.rawOutput === 'string'
              ? event.rawOutput
              : safeJsonStringify(event.rawOutput)
            : ''
      const shouldEmitLateUpdate = (): boolean => {
        if (!hasLatePayload) return false
        const prev = prevMetaBeforeUpdate
        const hasNewOutput =
          resolvedOutputText.length > 0 && resolvedOutputText !== (prev?.settledOutput ?? '')
        const hasNewDiffs =
          Boolean(diffs?.length) && safeJsonStringify(diffs) !== safeJsonStringify(prev?.diffs ?? null)
        const hasNewBlocks =
          Boolean(contentBlocks?.length) &&
          safeJsonStringify(contentBlocks) !== safeJsonStringify(prev?.contentBlocks ?? null)
        const hasNewLocations =
          Boolean(locations?.length) &&
          safeJsonStringify(locations) !== safeJsonStringify(prev?.locations ?? null)
        const hasNewInput =
          event.rawInput !== undefined &&
          safeJsonStringify(event.rawInput) !== safeJsonStringify(prev?.settledRawInput ?? null)
        const hasNewTitle = Boolean(event.title) && event.title !== prev?.title
        if (!hasNewOutput && !hasNewDiffs && !hasNewBlocks && !hasNewLocations && !hasNewInput && !hasNewTitle) {
          return false
        }
        if (toolMetaById) {
          const cur = toolMetaById.get(event.toolCallId)
          toolMetaById.set(event.toolCallId, {
            ...(cur ?? {}),
            ...(hasNewOutput ? { settledOutput: resolvedOutputText } : {}),
            ...(hasNewInput ? { settledRawInput: event.rawInput } : {}),
          })
        }
        return true
      }
      if (!settled) {
        if (settledToolCallIds?.has(event.toolCallId)) {
          return shouldEmitLateUpdate() ? buildToolUpdate() : undefined
        }
        return buildToolUpdate()
      }
      if (settledToolCallIds) {
        if (settledToolCallIds.has(event.toolCallId)) {
          return shouldEmitLateUpdate() ? buildToolUpdate() : undefined
        }
        settledToolCallIds.add(event.toolCallId)
      }
      const cachedMeta = toolMetaById?.get(event.toolCallId)
      const resolvedDiffs = diffs ?? cachedMeta?.diffs
      const resolvedDiff = diff ?? resolvedDiffs?.[0] ?? cachedMeta?.diff
      const resolvedBlocks = contentBlocks ?? cachedMeta?.contentBlocks
      const resolvedLocations = locations ?? cachedMeta?.locations
      const resolvedKind = event.kind ?? cachedMeta?.kind
      const resolvedTitle = event.title ?? cachedMeta?.title
      const output = resolvedOutputText
      if (toolMetaById) {
        toolMetaById.set(event.toolCallId, {
          ...(resolvedDiff ? { diff: resolvedDiff } : {}),
          ...(resolvedDiffs ? { diffs: resolvedDiffs } : {}),
          ...(resolvedBlocks ? { contentBlocks: resolvedBlocks } : {}),
          ...(resolvedLocations ? { locations: resolvedLocations } : {}),
          ...(resolvedKind ? { kind: resolvedKind } : {}),
          ...(resolvedTitle ? { title: resolvedTitle } : {}),
          settledOutput: output,
          ...(event.rawInput !== undefined ? { settledRawInput: event.rawInput } : {}),
        })
      }
      return {
        type: 'tool_result',
        toolCallId: event.toolCallId,
        name,
        output,
        ...(resolvedTitle ? { title: resolvedTitle } : {}),
        ...(resolvedKind ? { kind: resolvedKind } : {}),
        ...(event.rawOutput !== undefined ? { rawOutput: event.rawOutput } : {}),
        ...(event.isError ? { isError: true } : {}),
        ...(event.permissionDenied ? { permissionDenied: true } : {}),
        ...(resolvedDiff ? { diff: resolvedDiff } : {}),
        ...(resolvedDiffs ? { diffs: resolvedDiffs } : {}),
        ...(resolvedBlocks ? { contentBlocks: resolvedBlocks } : {}),
        ...(resolvedLocations ? { locations: resolvedLocations } : {}),
        ...(event.meta ? { meta: event.meta } : {}),
      }
    }
    case 'plan':
      return { type: 'plan', entries: event.entries }
    case 'focus':
      return { type: 'focus', focus: event.focus }
    case 'mode_update':
      return {
        type: 'mode_update',
        modeId: event.modeId,
        ...(event.modeName ? { modeName: event.modeName } : {}),
      }
    case 'model_update':
      return {
        type: 'model_update',
        modelId: event.modelId,
        ...(event.modelName ? { modelName: event.modelName } : {}),
      }
    case 'config_update':
      return {
        type: 'config_update',
        ...(event.configOptions ? { configOptions: event.configOptions } : {}),
        ...(event.configId ? { configId: event.configId } : {}),
        ...(event.value !== undefined ? { value: event.value } : {}),
        raw: event.rawUpdate,
      }
    case 'usage_update':
      return {
        type: 'usage',
        ...(event.contextUsed !== undefined ? { contextUsed: event.contextUsed } : {}),
        ...(event.contextSize !== undefined ? { contextSize: event.contextSize } : {}),
        ...(event.inputTokens !== undefined ? { inputTokens: event.inputTokens } : {}),
        ...(event.outputTokens !== undefined ? { outputTokens: event.outputTokens } : {}),
        ...(event.cachedReadTokens !== undefined ? { cachedReadTokens: event.cachedReadTokens } : {}),
        ...(event.cachedWriteTokens !== undefined ? { cachedWriteTokens: event.cachedWriteTokens } : {}),
        ...(event.thoughtTokens !== undefined ? { thoughtTokens: event.thoughtTokens } : {}),
        ...(event.totalTokens !== undefined ? { totalTokens: event.totalTokens } : {}),
        ...(event.cost ? { cost: event.cost } : {}),
        ...(event.costUsd !== undefined ? { costUsd: event.costUsd } : {}),
      }
    case 'session_info':
      return {
        type: 'session_info',
        ...(event.title ? { title: event.title } : {}),
        ...(event.updatedAt ? { updatedAt: event.updatedAt } : {}),
      }
    case 'available_commands':
      return {
        type: 'available_commands',
        commands: event.commands,
      }
    case 'notice':
      return {
        type: 'notice',
        severity: event.severity,
        title: event.title,
        ...(event.description ? { description: event.description } : {}),
      }
    case 'elicitation_complete':
      return {
        type: 'elicitation_complete',
        elicitationId: event.elicitationId,
      }
    case 'continuation':
      return {
        type: 'continuation',
        ...(event.text ? { text: event.text } : {}),
        raw: event.rawUpdate,
      }
    case 'error':
      return { type: 'error', message: event.message }
    case 'stderr':
    case 'exit':
      return undefined
  }
}

// ── Claude Code stream-json runner ────────────────────────────────────────

export interface ClaudeStreamJsonProcess {
  readonly exited: Promise<number | null>
  readonly kill: () => void
  readonly stderr?: ReadableStream<Uint8Array> | null
  readonly stdout: ReadableStream<Uint8Array>
}

export type ClaudeStreamJsonSpawner = (
  binaryPath: string,
  args: readonly string[],
  options: {
    readonly cwd: string
    readonly env?: Readonly<Record<string, string | undefined>>
    readonly input: string
  },
) => ClaudeStreamJsonProcess

export interface ClaudeStreamJsonRunnerOptions {
  readonly binaryPath: string
  readonly cwd: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly model?: string
  readonly permissionMode?: string
  readonly resumeSessionId?: string
  readonly spawn?: ClaudeStreamJsonSpawner
}

export interface ClaudeStreamJsonPromptParams {
  readonly onEvent: (event: AcpBridgeStreamEvent) => void
  readonly permissionMode?: string
  readonly signal?: AbortSignal
  readonly text: string
}

export interface ClaudeStreamJsonPromptResult {
  readonly cancelled: boolean
  readonly externalSessionId?: string
  readonly stopReason: string
}

/**
 * Streaming driver for the official Anthropic `claude` CLI using
 * `claude -p --output-format stream-json --verbose [--resume <session_id>]`.
 * Translates Claude NDJSON events into the exact {@link AcpBridgeStreamEvent}
 * union shared with {@link AcpStdioClient}.
 */
export class ClaudeStreamJsonRunner {
  private readonly binaryPath: string
  private readonly cwd: string
  private readonly env: Readonly<Record<string, string | undefined>> | undefined
  private readonly defaultPermissionMode: string | undefined
  private readonly spawner: ClaudeStreamJsonSpawner
  private externalSessionId: string | undefined
  private selectedModel: string | undefined
  private activeProcess: ClaudeStreamJsonProcess | undefined
  private activeReader: ReturnType<ReadableStream<Uint8Array>['getReader']> | undefined
  private activeCancelled = false

  constructor(options: ClaudeStreamJsonRunnerOptions) {
    this.binaryPath = options.binaryPath
    this.cwd = options.cwd
    this.env = options.env
    this.defaultPermissionMode = options.permissionMode
    this.externalSessionId = options.resumeSessionId
    this.selectedModel = options.model?.trim() || undefined
    this.spawner = options.spawn ?? spawnBunClaudeStreamProcess
  }

  get sessionId(): string | undefined {
    return this.externalSessionId
  }

  getModel(): string | undefined {
    return this.selectedModel
  }

  setModel(modelId?: string): void {
    const trimmed = modelId?.trim()
    this.selectedModel = trimmed ? trimmed : undefined
  }

  cancel(): boolean {
    if (!this.activeProcess) {
      return false
    }
    this.activeCancelled = true
    try {
      this.activeProcess.kill()
    } catch {
      // Ignore kill errors on already-exited subprocesses.
    }
    if (this.activeReader) {
      void this.activeReader.cancel().catch(() => undefined)
    }
    return true
  }

  async dispose(): Promise<void> {
    this.cancel()
    const active = this.activeProcess
    if (active) {
      await active.exited.catch(() => null)
    }
  }

  async prompt(params: ClaudeStreamJsonPromptParams): Promise<ClaudeStreamJsonPromptResult> {
    if (params.signal?.aborted) {
      return {
        cancelled: true,
        ...(this.externalSessionId ? { externalSessionId: this.externalSessionId } : {}),
        stopReason: 'cancelled',
      }
    }

    const args: string[] = [
      '-p',
      '--output-format',
      'stream-json',
      '--verbose',
      '--include-partial-messages',
    ]
    if (this.externalSessionId) {
      args.push('--resume', this.externalSessionId)
    }
    if (this.selectedModel) {
      args.push('--model', this.selectedModel)
    }
    const mappedMode = mapXerxesPermissionModeToClaude(params.permissionMode ?? this.defaultPermissionMode)
    if (mappedMode) {
      args.push('--permission-mode', mappedMode)
    }

    this.activeCancelled = false
    const proc = this.spawner(this.binaryPath, args, {
      cwd: this.cwd,
      ...(this.env ? { env: this.env } : {}),
      input: params.text,
    })
    this.activeProcess = proc

    const reader = proc.stdout.getReader()
    this.activeReader = reader
    const onAbort = () => {
      this.activeCancelled = true
      try {
        proc.kill()
      } catch {
        // Ignore kill error during abort.
      }
      void reader.cancel().catch(() => undefined)
    }
    params.signal?.addEventListener('abort', onAbort, { once: true })

    const stderrChunks: string[] = []
    const stderrTask = proc.stderr ? collectStreamText(proc.stderr, stderrChunks) : Promise.resolve()

    const state: ClaudeStreamParseState = {
      emittedAnyText: false,
      emittedDeltaInCurrentAssistant: false,
      emittedThinkingInCurrentAssistant: false,
      seenToolCallIds: new Set<string>(),
      toolNamesById: new Map<string, string>(),
      hadErrorEvent: false,
      lastErrorMessage: undefined,
    }

    const decoder = new TextDecoder()
    let buffer = ''
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        let newline = buffer.indexOf('\n')
        while (newline >= 0) {
          const line = buffer.slice(0, newline).trim()
          buffer = buffer.slice(newline + 1)
          if (line) {
            this.handleNdjsonLine(line, state, params.onEvent)
          }
          newline = buffer.indexOf('\n')
        }
      }
      buffer += decoder.decode()
      const tail = buffer.trim()
      if (tail) {
        this.handleNdjsonLine(tail, state, params.onEvent)
      }
    } catch (error) {
      if (!this.activeCancelled && !params.signal?.aborted) {
        throw error
      }
    } finally {
      if (this.activeReader === reader) {
        this.activeReader = undefined
      }
      try {
        reader.releaseLock()
      } catch {
        // Already closed.
      }
      await stderrTask.catch(() => undefined)
      params.signal?.removeEventListener('abort', onAbort)
    }

    const exitCode = await proc.exited.catch(() => null)
    if (this.activeProcess === proc) {
      this.activeProcess = undefined
    }

    const cancelled = this.activeCancelled || params.signal?.aborted === true
    if (cancelled) {
      return {
        cancelled: true,
        ...(this.externalSessionId ? { externalSessionId: this.externalSessionId } : {}),
        stopReason: 'cancelled',
      }
    }

    if (exitCode !== 0 && exitCode !== null) {
      const stderrText = stderrChunks.join('').trim()
      const detail = state.lastErrorMessage || stderrText || `claude exited with code ${exitCode}`
      if (!state.hadErrorEvent) {
        params.onEvent({ type: 'error', message: detail })
      }
      throw new Error(`Claude Code CLI failed (exit ${exitCode}): ${detail}`)
    }

    return {
      cancelled: false,
      ...(this.externalSessionId ? { externalSessionId: this.externalSessionId } : {}),
      stopReason: state.hadErrorEvent ? 'error' : 'end_turn',
    }
  }

  private handleNdjsonLine(
    line: string,
    state: ClaudeStreamParseState,
    emit: (event: AcpBridgeStreamEvent) => void,
  ): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line) as unknown
    } catch {
      return
    }
    if (!isAcpRecord(parsed)) {
      return
    }

    const type = typeof parsed.type === 'string' ? parsed.type : ''
    const subtype = typeof parsed.subtype === 'string' ? parsed.subtype : ''
    const frameSessionId = typeof parsed.session_id === 'string' && parsed.session_id.trim()
      ? parsed.session_id.trim()
      : undefined
    const frameModel = typeof parsed.model === 'string' && parsed.model.trim()
      ? parsed.model.trim()
      : undefined
    const isSystemInit = type === 'system' && subtype === 'init'

    if (isSystemInit) {
      const sid = frameSessionId ?? this.externalSessionId ?? ''
      if (frameSessionId) {
        this.externalSessionId = frameSessionId
      }
      if (sid || frameModel) {
        emit({
          type: 'session_init',
          externalSessionId: sid,
          ...(frameModel ? { model: frameModel } : {}),
        })
      }
      const initCommands = parseClaudeInitCommands(parsed)
      if (initCommands.length > 0) {
        emit({
          type: 'available_commands',
          commands: initCommands,
        })
      }
    } else if (frameSessionId && frameSessionId !== this.externalSessionId) {
      this.externalSessionId = frameSessionId
      emit({
        type: 'session_init',
        externalSessionId: frameSessionId,
        ...(frameModel ? { model: frameModel } : {}),
      })
    }

    if (type === 'stream_event' && isAcpRecord(parsed.event)) {
      this.handleStreamDeltaEvent(parsed.event, state, emit)
      return
    }

    if (type === 'content_block_delta') {
      this.handleStreamDeltaEvent(parsed, state, emit)
      return
    }

    if (type === 'thinking') {
      const text = typeof parsed.thinking === 'string'
        ? parsed.thinking
        : typeof parsed.text === 'string'
          ? parsed.text
          : ''
      if (text) {
        state.emittedThinkingInCurrentAssistant = true
        emit({ type: 'thought_delta', text })
      }
      return
    }

    if (type === 'assistant') {
      const message = isAcpRecord(parsed.message) ? parsed.message : parsed
      const content = Array.isArray(message.content) ? message.content : []
      for (const block of content) {
        if (!isAcpRecord(block)) continue
        const blockType = typeof block.type === 'string' ? block.type : ''
        if (blockType === 'thinking') {
          const thinkingText = typeof block.thinking === 'string'
            ? block.thinking
            : typeof block.text === 'string'
              ? block.text
              : ''
          if (thinkingText && !state.emittedThinkingInCurrentAssistant) {
            emit({ type: 'thought_delta', text: thinkingText })
          }
        } else if (blockType === 'text') {
          const text = typeof block.text === 'string' ? block.text : ''
          if (text && !state.emittedDeltaInCurrentAssistant) {
            state.emittedAnyText = true
            emit({ type: 'text_delta', text })
          }
        } else if (blockType === 'tool_use') {
          this.emitClaudeToolUse(block, state, emit)
        }
      }
      // Reset per-assistant-step streaming flags for subsequent multi-step rounds.
      state.emittedDeltaInCurrentAssistant = false
      state.emittedThinkingInCurrentAssistant = false
      return
    }

    if (type === 'tool_use') {
      this.emitClaudeToolUse(parsed, state, emit)
      return
    }

    if (type === 'user') {
      const message = isAcpRecord(parsed.message) ? parsed.message : parsed
      const content = Array.isArray(message.content) ? message.content : []
      for (const block of content) {
        if (isAcpRecord(block) && block.type === 'tool_result') {
          this.emitClaudeToolResult(block, state, emit)
        }
      }
      return
    }

    if (type === 'tool_result') {
      this.emitClaudeToolResult(parsed, state, emit)
      return
    }

    if (type === 'result') {
      const usageRecord = isAcpRecord(parsed.usage) ? parsed.usage : undefined
      const inputTokens =
        typeof usageRecord?.input_tokens === 'number' ? usageRecord.input_tokens : undefined
      const outputTokens =
        typeof usageRecord?.output_tokens === 'number' ? usageRecord.output_tokens : undefined
      const costUsd = typeof parsed.total_cost_usd === 'number' ? parsed.total_cost_usd : undefined
      if (inputTokens !== undefined || outputTokens !== undefined || costUsd !== undefined) {
        emit({
          type: 'usage',
          ...(inputTokens !== undefined ? { inputTokens } : {}),
          ...(outputTokens !== undefined ? { outputTokens } : {}),
          ...(costUsd !== undefined ? { costUsd } : {}),
        })
      }
      const isError =
        parsed.is_error === true ||
        parsed.subtype === 'error' ||
        parsed.subtype === 'error_during_execution'
      if (isError) {
        const msg =
          typeof parsed.result === 'string' && parsed.result.trim()
            ? parsed.result.trim()
            : typeof parsed.error === 'string' && parsed.error.trim()
              ? parsed.error.trim()
              : 'Claude Code reported an error.'
        state.hadErrorEvent = true
        state.lastErrorMessage = msg
        emit({ type: 'error', message: msg })
      } else if (!state.emittedAnyText && typeof parsed.result === 'string' && parsed.result) {
        state.emittedAnyText = true
        emit({ type: 'text_delta', text: parsed.result })
      }
    }
  }

  private handleStreamDeltaEvent(
    event: Readonly<Record<string, unknown>>,
    state: ClaudeStreamParseState,
    emit: (event: AcpBridgeStreamEvent) => void,
  ): void {
    const eventType = typeof event.type === 'string' ? event.type : ''
    if (eventType !== 'content_block_delta') {
      return
    }
    const delta = isAcpRecord(event.delta) ? event.delta : undefined
    if (!delta) return
    const deltaType = typeof delta.type === 'string' ? delta.type : ''
    if (deltaType === 'text_delta' && typeof delta.text === 'string' && delta.text) {
      state.emittedAnyText = true
      state.emittedDeltaInCurrentAssistant = true
      emit({ type: 'text_delta', text: delta.text })
    } else if (deltaType === 'thinking_delta') {
      const thinking = typeof delta.thinking === 'string'
        ? delta.thinking
        : typeof delta.text === 'string'
          ? delta.text
          : ''
      if (thinking) {
        state.emittedThinkingInCurrentAssistant = true
        emit({ type: 'thought_delta', text: thinking })
      }
    }
  }

  private emitClaudeToolUse(
    block: Readonly<Record<string, unknown>>,
    state: ClaudeStreamParseState,
    emit: (event: AcpBridgeStreamEvent) => void,
  ): void {
    const toolCallId =
      typeof block.id === 'string' && block.id.trim() ? block.id.trim() : crypto.randomUUID()
    if (state.seenToolCallIds.has(toolCallId)) {
      return
    }
    state.seenToolCallIds.add(toolCallId)
    const name = typeof block.name === 'string' && block.name.trim() ? block.name.trim() : 'tool'
    state.toolNamesById.set(toolCallId, name)
    const argumentsText =
      block.input === undefined
        ? ''
        : typeof block.input === 'string'
          ? block.input
          : safeJsonStringify(block.input)
    emit({
      type: 'tool_call',
      toolCallId,
      name,
      argumentsText,
    })
  }

  private emitClaudeToolResult(
    block: Readonly<Record<string, unknown>>,
    state: ClaudeStreamParseState,
    emit: (event: AcpBridgeStreamEvent) => void,
  ): void {
    const toolCallId =
      typeof block.tool_use_id === 'string' && block.tool_use_id.trim()
        ? block.tool_use_id.trim()
        : typeof block.id === 'string' && block.id.trim()
          ? block.id.trim()
          : 'tool'
    const name =
      (typeof block.name === 'string' && block.name.trim() ? block.name.trim() : undefined) ??
      state.toolNamesById.get(toolCallId) ??
      'tool'
    const output = flattenClaudeToolResultContent(block.content)
    const isError = block.is_error === true
    emit({
      type: 'tool_result',
      toolCallId,
      name,
      output,
      ...(isError ? { isError: true } : {}),
    })
  }
}

interface ClaudeStreamParseState {
  emittedAnyText: boolean
  emittedDeltaInCurrentAssistant: boolean
  emittedThinkingInCurrentAssistant: boolean
  hadErrorEvent: boolean
  lastErrorMessage: string | undefined
  readonly seenToolCallIds: Set<string>
  readonly toolNamesById: Map<string, string>
}

function mapXerxesPermissionModeToClaude(mode: string | undefined): string | undefined {
  if (!mode) return undefined
  switch (mode.trim().toLowerCase()) {
    case 'accept-all':
    case 'bypasspermissions':
      return 'bypassPermissions'
    case 'plan':
      return 'plan'
    case 'auto':
    case 'acceptedits':
      return 'acceptEdits'
    case 'manual':
      return 'manual'
    default:
      return undefined
  }
}

function flattenClaudeToolResultContent(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content
      .map(item => {
        if (typeof item === 'string') return item
        if (isAcpRecord(item) && typeof item.text === 'string') return item.text
        return safeJsonStringify(item)
      })
      .filter(Boolean)
      .join('\n')
  }
  if (content === undefined || content === null) {
    return ''
  }
  return safeJsonStringify(content)
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return String(value)
  }
}

function spawnBunClaudeStreamProcess(
  binaryPath: string,
  args: readonly string[],
  options: {
    readonly cwd: string
    readonly env?: Readonly<Record<string, string | undefined>>
    readonly input: string
  },
): ClaudeStreamJsonProcess {
  const mergedEnv: Record<string, string> = {}
  for (const [key, value] of Object.entries({ ...process.env, ...(options.env ?? {}) })) {
    if (typeof value === 'string') {
      mergedEnv[key] = value
    }
  }
  const child = Bun.spawn([binaryPath, ...args], {
    cwd: options.cwd,
    env: mergedEnv,
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'pipe',
  })
  if (child.stdin && typeof child.stdin !== 'number') {
    child.stdin.write(options.input)
    void child.stdin.end?.()
  }
  return {
    stdout: child.stdout,
    stderr: child.stderr,
    exited: child.exited.then(code => (typeof code === 'number' ? code : null)),
    kill: () => {
      try {
        child.kill()
      } catch {
        // Ignore already-exited subprocess.
      }
    },
  }
}

async function collectStreamText(
  stream: ReadableStream<Uint8Array>,
  sink: string[],
): Promise<void> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      sink.push(decoder.decode(value, { stream: true }))
    }
    const tail = decoder.decode()
    if (tail) sink.push(tail)
  } catch {
    // Ignore cancellation on process kill.
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // Already released.
    }
  }
}

export function formatAcpError(error: unknown): string {
  return errorMessage(error)
}

// ── Automatic ACP Runtime Feature & Skill Discovery ───────────────────────

const ANTIGRAVITY_BUILTIN_COMMANDS: ReadonlySet<string> = new Set([
  'goal',
  'compact',
  'schedule',
  'browser',
  'plan',
  'grill-me',
  'teamwork',
  'learn',
  'boost',
  'swarm',
])

const MAX_SKILL_FRONTMATTER_BYTES = 262_144

export interface RuntimeDiscoveredEntry {
  readonly description?: string
  readonly dirName?: string
  readonly disableSlashCommand?: boolean
  readonly kind: 'command' | 'skill'
  readonly name: string
  readonly path: string
  readonly source: string
}

export interface DiscoverRuntimeSkillEntriesOptions {
  readonly cwd: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
  readonly provider: AcpExternalAgentProvider
}

export interface ClassifyAcpRuntimeFeaturesInput {
  readonly agentName: string
  readonly cwd: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
  readonly includeFilesystemUnlisted?: boolean
  readonly provider: AcpExternalAgentProvider
  readonly rawCommands?: readonly AcpAvailableCommand[]
}

export interface ClassifiedAcpRuntimeFeatures {
  readonly available_commands: readonly AcpExternalCommandOption[]
  readonly discovered_skills: readonly AcpDiscoveredSkill[]
}

/**
 * Parse YAML frontmatter (`name`, `description`, `disable-slash-command`) from
 * a `SKILL.md` or `.claude/commands/*.md` document without executing anything.
 */
export function parseMarkdownSkillFrontmatter(markdown: string): {
  readonly description?: string
  readonly disableSlashCommand?: boolean
  readonly name?: string
} {
  const normalized = markdown.replace(/\r\n/g, '\n')
  const match = /^---\n([\s\S]*?)\n---(?:\n|$)/.exec(normalized)
  if (!match) {
    const fallbackDesc = extractLeadingMarkdownDescription(normalized)
    return fallbackDesc ? { description: fallbackDesc } : {}
  }

  const lines = match[1]!.split('\n')
  let name: string | undefined
  let description: string | undefined
  let disableSlashCommand = false

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!
    const field = /^([A-Za-z0-9_-]+)\s*:\s*(.*)$/.exec(line)
    if (!field) continue
    const key = field[1]!.toLowerCase()
    const rawVal = field[2]!.trim()

    let value = rawVal
    if (rawVal === '>' || rawVal === '>-' || rawVal === '|' || rawVal === '|-') {
      const folded = rawVal.startsWith('>')
      const blockLines: string[] = []
      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1]!
        if (nextLine.trim() === '') {
          blockLines.push('')
          i += 1
          continue
        }
        if (/^[ \t]+/.test(nextLine)) {
          blockLines.push(nextLine.trim())
          i += 1
          continue
        }
        break
      }
      value = folded
        ? blockLines.filter(Boolean).join(' ').replace(/\s+/g, ' ').trim()
        : blockLines.join('\n').trim()
    } else {
      value = stripYamlQuotes(rawVal)
    }

    if (key === 'name' && value) {
      name = value
    } else if (key === 'description' && value) {
      description = value
    } else if (
      (key === 'disable-slash-command' || key === 'disable_slash_command') &&
      value.toLowerCase() === 'true'
    ) {
      disableSlashCommand = true
    }
  }

  if (!description) {
    const body = normalized.slice(match[0].length)
    description = extractLeadingMarkdownDescription(body)
  }

  return {
    ...(name ? { name } : {}),
    ...(description ? { description } : {}),
    ...(disableSlashCommand ? { disableSlashCommand: true } : {}),
  }
}

/**
 * Inspect ONLY the external runtime's own workspace/user skill directories to
 * enrich and classify runtime-native skills and commands. Never reads Xerxes's
 * bundled `xerxes/skills/` or `~/.xerxes/skills/` directories.
 */
export function discoverRuntimeSkillEntries(
  options: DiscoverRuntimeSkillEntriesOptions,
): readonly RuntimeDiscoveredEntry[] {
  const env = options.env ?? process.env
  const home = options.homeDir ?? (env.HOME?.trim() || homedir())
  const cwd = resolve(options.cwd)
  const entries: RuntimeDiscoveredEntry[] = []

  if (options.provider === 'antigravity') {
    entries.push(
      ...scanSkillMdDirectory(join(cwd, '.agents', 'skills'), 'Workspace'),
      ...scanSkillMdDirectory(join(home, '.gemini', 'antigravity', 'skills'), 'Antigravity'),
      ...scanSkillMdDirectory(join(home, '.gemini', 'config', 'skills'), 'Antigravity'),
      ...scanSkillMdDirectory(join(home, '.gemini', 'antigravity', 'builtin', 'skills'), 'Antigravity'),
    )
    return entries
  }

  if (options.provider === 'claude-code') {
    entries.push(
      ...scanSkillMdDirectory(join(cwd, '.claude', 'skills'), 'Workspace'),
      ...scanMarkdownCommandsDirectory(join(cwd, '.claude', 'commands'), 'Workspace'),
      ...scanSkillMdDirectory(join(home, '.claude', 'skills'), 'Claude Code'),
      ...scanMarkdownCommandsDirectory(join(home, '.claude', 'commands'), 'Claude Code'),
    )
    return entries
  }

  // Generic / custom ACP agents rely exclusively on runtime-advertised
  // `available_commands_update`, `configOptions`, and `agentCapabilities`.
  return entries
}

/**
 * Classify and enrich an external ACP runtime's advertised slash commands and
 * skills without leaking any Xerxes bundled skills or native commands.
 */
export function classifyAcpRuntimeFeatures(
  input: ClassifyAcpRuntimeFeaturesInput,
): ClassifiedAcpRuntimeFeatures {
  const defaultSource =
    input.provider === 'antigravity'
      ? 'Antigravity'
      : input.provider === 'claude-code'
        ? 'Claude Code'
        : input.agentName.trim() || 'ACP'

  const fsEntries = discoverRuntimeSkillEntries({
    provider: input.provider,
    cwd: input.cwd,
    ...(input.env ? { env: input.env } : {}),
    ...(input.homeDir ? { homeDir: input.homeDir } : {}),
  })

  // Index discovered runtime entries (Workspace first, then user/builtin).
  const byExactName = new Map<string, RuntimeDiscoveredEntry>()
  const byNormalizedName = new Map<string, RuntimeDiscoveredEntry>()
  for (const entry of fsEntries) {
    const lower = entry.name.toLowerCase()
    if (!byExactName.has(lower)) {
      byExactName.set(lower, entry)
    }
    const normalized = normalizeRuntimeFeatureKey(entry.name)
    if (normalized && !byNormalizedName.has(normalized)) {
      byNormalizedName.set(normalized, entry)
    }
    if (entry.dirName) {
      const dirLower = entry.dirName.toLowerCase()
      if (!byExactName.has(dirLower)) {
        byExactName.set(dirLower, entry)
      }
      const dirNorm = normalizeRuntimeFeatureKey(entry.dirName)
      if (dirNorm && !byNormalizedName.has(dirNorm)) {
        byNormalizedName.set(dirNorm, entry)
      }
    }
  }

  const availableCommands: AcpExternalCommandOption[] = []
  const discoveredSkills: AcpDiscoveredSkill[] = []
  const seenCommands = new Set<string>()
  const seenSkills = new Set<string>()

  const lookupFsEntry = (name: string): RuntimeDiscoveredEntry | undefined => {
    const lower = name.toLowerCase()
    return byExactName.get(lower) ?? byNormalizedName.get(normalizeRuntimeFeatureKey(name))
  }

  for (const raw of input.rawCommands ?? []) {
    const name = raw.name.trim().replace(/^\/+/, '')
    if (!name) continue
    const lower = name.toLowerCase()
    if (seenCommands.has(lower)) continue
    seenCommands.add(lower)

    if (input.provider === 'antigravity' && ANTIGRAVITY_BUILTIN_COMMANDS.has(lower)) {
      availableCommands.push({
        name,
        ...(raw.description ? { description: raw.description } : {}),
        ...(raw.input_hint ? { input_hint: raw.input_hint } : {}),
        kind: 'command',
        source: raw.source ?? 'Antigravity',
      })
      continue
    }

    const matched = lookupFsEntry(name)
    if (matched && matched.kind === 'skill') {
      const description = raw.description || matched.description
      const source = raw.source ?? matched.source
      availableCommands.push({
        name,
        ...(description ? { description } : {}),
        ...(raw.input_hint ? { input_hint: raw.input_hint } : {}),
        kind: 'skill',
        source,
      })
      if (!seenSkills.has(lower)) {
        seenSkills.add(lower)
        discoveredSkills.push({
          name,
          ...(description ? { description } : {}),
          source,
          path: matched.path,
          slash_command: `/${name}`,
        })
      }
      continue
    }

    if (matched && matched.kind === 'command') {
      const description = raw.description || matched.description
      const source = raw.source ?? matched.source
      availableCommands.push({
        name,
        ...(description ? { description } : {}),
        ...(raw.input_hint ? { input_hint: raw.input_hint } : {}),
        kind: 'command',
        source,
      })
      continue
    }

    if (raw.kind === 'skill') {
      const source = raw.source ?? defaultSource
      availableCommands.push({
        name,
        ...(raw.description ? { description: raw.description } : {}),
        ...(raw.input_hint ? { input_hint: raw.input_hint } : {}),
        kind: 'skill',
        source,
      })
      if (!seenSkills.has(lower)) {
        seenSkills.add(lower)
        discoveredSkills.push({
          name,
          ...(raw.description ? { description: raw.description } : {}),
          source,
          slash_command: `/${name}`,
        })
      }
      continue
    }

    availableCommands.push({
      name,
      ...(raw.description ? { description: raw.description } : {}),
      ...(raw.input_hint ? { input_hint: raw.input_hint } : {}),
      kind: 'command',
      source: raw.source ?? defaultSource,
    })
  }

  if (input.includeFilesystemUnlisted) {
    for (const entry of fsEntries) {
      if (entry.disableSlashCommand) continue
      const lower = entry.name.toLowerCase()
      if (input.provider !== 'antigravity') {
        if (seenCommands.has(lower)) continue
        seenCommands.add(lower)
        availableCommands.push({
          name: entry.name,
          ...(entry.description ? { description: entry.description } : {}),
          kind: entry.kind,
          source: entry.source,
        })
      }
      if (entry.kind === 'skill' && !seenSkills.has(lower)) {
        seenSkills.add(lower)
        discoveredSkills.push({
          name: entry.name,
          ...(entry.description ? { description: entry.description } : {}),
          source: entry.source,
          path: entry.path,
          slash_command: `/${entry.name}`,
        })
      }
    }
  }

  return {
    available_commands: availableCommands,
    discovered_skills: discoveredSkills,
  }
}

function parseClaudeInitCommands(record: Readonly<Record<string, unknown>>): AcpAvailableCommand[] {
  const rawSlash = Array.isArray(record.slash_commands)
    ? record.slash_commands
    : Array.isArray(record.slashCommands)
      ? record.slashCommands
      : []
  const rawSkills = Array.isArray(record.skills) ? record.skills : []
  const skillNames = new Set<string>()
  const commands: AcpAvailableCommand[] = []
  const seen = new Set<string>()

  for (const item of rawSkills) {
    const rawName = typeof item === 'string'
      ? item
      : isAcpRecord(item) && typeof item.name === 'string'
        ? item.name
        : ''
    const name = rawName.trim().replace(/^\/+/, '')
    if (!name) continue
    const lower = name.toLowerCase()
    skillNames.add(lower)
    if (seen.has(lower)) continue
    seen.add(lower)
    const description =
      isAcpRecord(item) && typeof item.description === 'string' && item.description.trim()
        ? item.description.trim()
        : undefined
    commands.push({
      name,
      ...(description ? { description } : {}),
      kind: 'skill',
      source: 'Claude Code',
    })
  }

  for (const item of rawSlash) {
    const rawName = typeof item === 'string'
      ? item
      : isAcpRecord(item) && typeof item.name === 'string'
        ? item.name
        : ''
    const name = rawName.trim().replace(/^\/+/, '')
    if (!name) continue
    const lower = name.toLowerCase()
    if (seen.has(lower)) continue
    seen.add(lower)
    const description =
      isAcpRecord(item) && typeof item.description === 'string' && item.description.trim()
        ? item.description.trim()
        : undefined
    commands.push({
      name,
      ...(description ? { description } : {}),
      kind: skillNames.has(lower) ? 'skill' : 'command',
      source: 'Claude Code',
    })
  }

  return commands
}

function scanSkillMdDirectory(rootDir: string, source: string): RuntimeDiscoveredEntry[] {
  let dirEntries: string[]
  try {
    dirEntries = readdirSync(rootDir).sort()
  } catch {
    return []
  }

  const results: RuntimeDiscoveredEntry[] = []
  for (const entryName of dirEntries) {
    if (!entryName || entryName.startsWith('.') || entryName === 'node_modules') {
      continue
    }
    const dirPath = join(rootDir, entryName)
    let isDir = false
    try {
      isDir = statSync(dirPath).isDirectory()
    } catch {
      continue
    }
    if (!isDir) continue
    const skillPath = join(dirPath, 'SKILL.md')
    const raw = readBoundedUtf8(skillPath)
    if (raw === undefined) continue
    const meta = parseMarkdownSkillFrontmatter(raw)
    const name = (meta.name ?? entryName).trim().replace(/^\/+/, '')
    if (!name) continue
    results.push({
      name,
      dirName: entryName,
      ...(meta.description ? { description: meta.description } : {}),
      ...(meta.disableSlashCommand ? { disableSlashCommand: true } : {}),
      kind: 'skill',
      source,
      path: skillPath,
    })
  }
  return results
}

function scanMarkdownCommandsDirectory(
  rootDir: string,
  source: string,
  prefix = '',
  depth = 0,
): RuntimeDiscoveredEntry[] {
  if (depth > 2) return []
  let dirEntries: string[]
  try {
    dirEntries = readdirSync(rootDir).sort()
  } catch {
    return []
  }

  const results: RuntimeDiscoveredEntry[] = []
  for (const entryName of dirEntries) {
    if (!entryName || entryName.startsWith('.') || entryName === 'node_modules') {
      continue
    }
    const fullPath = join(rootDir, entryName)
    let stats: ReturnType<typeof statSync>
    try {
      stats = statSync(fullPath)
    } catch {
      continue
    }
    if (stats.isDirectory()) {
      const nextPrefix = prefix ? `${prefix}:${entryName}` : entryName
      results.push(...scanMarkdownCommandsDirectory(fullPath, source, nextPrefix, depth + 1))
      continue
    }
    if (!stats.isFile() || !entryName.toLowerCase().endsWith('.md')) {
      continue
    }
    const raw = readBoundedUtf8(fullPath)
    if (raw === undefined) continue
    const meta = parseMarkdownSkillFrontmatter(raw)
    const stem = basename(entryName, entryName.slice(-3))
    const fallbackName = prefix ? `${prefix}:${stem}` : stem
    const name = (meta.name ?? fallbackName).trim().replace(/^\/+/, '')
    if (!name) continue
    results.push({
      name,
      ...(meta.description ? { description: meta.description } : {}),
      ...(meta.disableSlashCommand ? { disableSlashCommand: true } : {}),
      kind: 'command',
      source,
      path: fullPath,
    })
  }
  return results
}

function readBoundedUtf8(filePath: string): string | undefined {
  try {
    const stats = statSync(filePath)
    if (!stats.isFile() || stats.size > MAX_SKILL_FRONTMATTER_BYTES) {
      return undefined
    }
    return readFileSync(filePath, 'utf8')
  } catch {
    return undefined
  }
}

function stripYamlQuotes(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"') && value.length >= 2) ||
    (value.startsWith("'") && value.endsWith("'") && value.length >= 2)
  ) {
    return value.slice(1, -1).trim()
  }
  return value.trim()
}

function extractLeadingMarkdownDescription(markdown: string): string | undefined {
  for (const rawLine of markdown.split('\n')) {
    const line = rawLine.trim()
    if (!line || line === '---') continue
    if (line.startsWith('#')) {
      const heading = line.replace(/^#+\s*/, '').trim()
      if (heading) return heading.slice(0, 240)
      continue
    }
    return line.slice(0, 240)
  }
  return undefined
}

function normalizeRuntimeFeatureKey(name: string): string {
  return name.trim().toLowerCase().replaceAll('_', '-')
}

/** Normalized question descriptor derived from an ACP `elicitation/create` form schema. */
export interface AcpElicitationQuestionSpec {
  readonly allowFreeform: boolean
  readonly choiceMap: readonly AcpElicitationOption[]
  readonly fieldType: AcpElicitationFieldType
  readonly id: string
  readonly multiSelect: boolean
  readonly options: readonly string[]
  readonly otherFieldKey?: string
  readonly placeholder?: string
  readonly question: string
}

/**
 * Convert an ACP `elicitation/create` form request into Xerxes `question_request`
 * specs, collapsing companion `<key>_other` string fields (such as Antigravity's
 * `q1` + `q1_other`) into `allowFreeform: true` on the primary question.
 */
export function elicitationToQuestionItems(
  request: AcpElicitationRequest,
): readonly AcpElicitationQuestionSpec[] {
  if (request.mode === 'url' || request.fields.length === 0) {
    return []
  }

  const byKey = new Map<string, AcpElicitationField>()
  for (const field of request.fields) {
    byKey.set(field.key, field)
  }

  const consumedKeys = new Set<string>()
  const specs: AcpElicitationQuestionSpec[] = []

  for (const field of request.fields) {
    if (consumedKeys.has(field.key)) {
      continue
    }

    const choiceMap = extractFieldChoices(field)
    const multiSelect = field.type === 'array'
    const otherCandidate = (choiceMap.length > 0 || multiSelect)
      ? byKey.get(`${field.key}_other`)
      : undefined
    const hasCompanionOther =
      otherCandidate !== undefined &&
      otherCandidate.type === 'string' &&
      !otherCandidate.oneOf?.length &&
      !otherCandidate.enum?.length
    if (hasCompanionOther && otherCandidate) {
      consumedKeys.add(otherCandidate.key)
    }

    const allowFreeform = hasCompanionOther
      ? true
      : choiceMap.length > 0 || field.type === 'boolean'
        ? false
        : true

    const fallbackQuestion =
      request.fields.length === 1 && request.message.trim()
        ? request.message.trim()
        : field.key
    const question = field.title?.trim() || field.description?.trim() || fallbackQuestion
    const placeholder = hasCompanionOther
      ? otherCandidate?.title?.trim() || 'Other'
      : field.type === 'integer'
        ? field.description?.trim() || 'Enter an integer'
        : field.type === 'number'
          ? field.description?.trim() || 'Enter a number'
          : !choiceMap.length && field.description?.trim()
            ? field.description.trim()
            : undefined

    specs.push({
      id: field.key,
      question,
      options: choiceMap.map(choice => choice.title),
      choiceMap,
      fieldType: field.type,
      allowFreeform,
      multiSelect,
      ...(hasCompanionOther && otherCandidate ? { otherFieldKey: otherCandidate.key } : {}),
      ...(placeholder ? { placeholder } : {}),
    })
  }

  return specs
}

/**
 * Translate Xerxes `question_response` answers back into an ACP
 * `elicitation/create` response (`accept` with typed `content`, or `decline`
 * when the user skipped the card).
 */
export function buildElicitationResponseFromAnswers(
  specs: readonly AcpElicitationQuestionSpec[],
  answers: Readonly<Record<string, string | readonly string[]>>,
): AcpElicitationResponse {
  if (specs.length === 0) {
    return { action: 'accept', content: {} }
  }

  let explicitSkipCount = 0
  let answeredCount = 0
  for (const spec of specs) {
    const raw = answers[spec.id]
    if (raw === undefined) continue
    if (isSkippedElicitationAnswer(spec, raw)) {
      explicitSkipCount += 1
    } else if (hasNonEmptyElicitationAnswer(raw)) {
      answeredCount += 1
    }
  }
  if (answeredCount === 0 && explicitSkipCount > 0) {
    return { action: 'decline' }
  }

  const content: Record<string, unknown> = {}

  for (const spec of specs) {
    const raw = answers[spec.id]
    if (raw === undefined || isSkippedElicitationAnswer(spec, raw)) {
      continue
    }

    if (spec.fieldType === 'boolean') {
      const text = firstAnswerText(raw).toLowerCase()
      if (text === 'yes' || text === 'true' || text === '1') {
        content[spec.id] = true
      } else if (text === 'no' || text === 'false' || text === '0') {
        content[spec.id] = false
      }
      continue
    }

    if (spec.fieldType === 'number' || spec.fieldType === 'integer') {
      const text = firstAnswerText(raw)
      if (!text) continue
      const parsed = Number(text)
      if (Number.isFinite(parsed)) {
        content[spec.id] = spec.fieldType === 'integer' ? Math.trunc(parsed) : parsed
      }
      continue
    }

    if (spec.multiSelect) {
      const tokens = parseMultiSelectTokens(spec, raw)
      if (tokens.length === 0) continue
      if (spec.choiceMap.length > 0) {
        const matchedConsts: string[] = []
        const seenConsts = new Set<string>()
        const unmatched: string[] = []
        for (const token of tokens) {
          const matched = matchChoiceConst(spec.choiceMap, token)
          if (matched !== undefined) {
            if (!seenConsts.has(matched)) {
              seenConsts.add(matched)
              matchedConsts.push(matched)
            }
          } else {
            unmatched.push(token)
          }
        }
        if (matchedConsts.length > 0) {
          content[spec.id] = matchedConsts
        }
        if (unmatched.length > 0) {
          if (spec.otherFieldKey) {
            content[spec.otherFieldKey] = unmatched.join(', ')
          } else if (spec.allowFreeform) {
            content[spec.id] = [...matchedConsts, ...unmatched]
          }
        }
      } else {
        content[spec.id] = tokens
      }
      continue
    }

    const text = Array.isArray(raw)
      ? raw.map(item => String(item).trim()).filter(Boolean).join(', ')
      : String(raw).trim()
    if (!text) continue
    if (spec.choiceMap.length > 0) {
      const matched = matchChoiceConst(spec.choiceMap, text)
      if (matched !== undefined) {
        content[spec.id] = matched
      } else if (spec.otherFieldKey) {
        content[spec.otherFieldKey] = text
      } else if (spec.allowFreeform) {
        content[spec.id] = text
      }
    } else {
      content[spec.id] = text
    }
  }

  return { action: 'accept', content }
}

/** Human-readable summary of a resolved ACP elicitation for tool-result display. */
export function formatElicitationAnswersSummary(
  specs: readonly AcpElicitationQuestionSpec[],
  response: AcpElicitationResponse,
): string {
  if (response.action === 'decline') {
    return 'User skipped.'
  }
  if (response.action === 'cancel') {
    return 'Cancelled.'
  }
  const content = response.content ?? {}
  const lines: string[] = []
  for (const spec of specs) {
    const primary = content[spec.id]
    const other = spec.otherFieldKey ? content[spec.otherFieldKey] : undefined
    const parts: string[] = []
    if (Array.isArray(primary)) {
      for (const entry of primary) {
        const key = String(entry)
        parts.push(spec.choiceMap.find(c => c.const === key)?.title ?? key)
      }
    } else if (typeof primary === 'boolean') {
      parts.push(primary ? 'Yes' : 'No')
    } else if (primary !== undefined && primary !== null && String(primary).trim() !== '') {
      const key = String(primary)
      parts.push(spec.choiceMap.find(c => c.const === key)?.title ?? key)
    }
    if (typeof other === 'string' && other.trim()) {
      parts.push(other.trim())
    }
    if (parts.length > 0) {
      lines.push(`${spec.question} -> ${parts.join(', ')}`)
    }
  }
  return lines.join('\n')
}

function extractFieldChoices(field: AcpElicitationField): readonly AcpElicitationOption[] {
  if (field.type === 'boolean') {
    return [
      { const: 'true', title: 'Yes' },
      { const: 'false', title: 'No' },
    ]
  }
  if (field.type === 'array') {
    if (field.items?.anyOf?.length) return field.items.anyOf
    if (field.items?.oneOf?.length) return field.items.oneOf
    if (field.items?.enum?.length) {
      return field.items.enum.map(value => ({ const: value, title: value }))
    }
    return []
  }
  if (field.oneOf?.length) {
    return field.oneOf
  }
  if (field.enum?.length) {
    return field.enum.map((value, index) => ({
      const: value,
      title: field.enumNames?.[index] ?? value,
    }))
  }
  return []
}

function matchChoiceConst(
  choices: readonly AcpElicitationOption[],
  token: string,
): string | undefined {
  const exactTitle = choices.find(choice => choice.title === token)
  if (exactTitle) return exactTitle.const
  const lower = token.trim().toLowerCase()
  const caseInsensitiveTitle = choices.find(choice => choice.title.trim().toLowerCase() === lower)
  if (caseInsensitiveTitle) return caseInsensitiveTitle.const
  const exactConst = choices.find(choice => choice.const === token)
  if (exactConst) return exactConst.const
  return undefined
}

function parseMultiSelectTokens(
  spec: AcpElicitationQuestionSpec,
  raw: string | readonly string[],
): string[] {
  if (Array.isArray(raw)) {
    return raw.map(item => String(item).trim()).filter(Boolean)
  }
  const trimmed = String(raw).trim()
  if (!trimmed) {
    return []
  }
  if (spec.choiceMap.length > 0 && matchChoiceConst(spec.choiceMap, trimmed) !== undefined) {
    return [trimmed]
  }
  if (trimmed.includes('\n')) {
    return trimmed.split('\n').map(item => item.trim()).filter(Boolean)
  }
  if (trimmed.includes(',') && spec.choiceMap.length > 0) {
    const parts = trimmed.split(',').map(item => item.trim()).filter(Boolean)
    if (parts.length >= 2 && parts.every(part => matchChoiceConst(spec.choiceMap, part) !== undefined)) {
      return parts
    }
  }
  return [trimmed]
}

function isSkippedElicitationAnswer(
  spec: AcpElicitationQuestionSpec,
  raw: string | readonly string[],
): boolean {
  const skipIsOption = spec.choiceMap.some(
    choice => /^skipped$/i.test(choice.title.trim()) || /^skipped$/i.test(choice.const.trim()),
  )
  if (skipIsOption) {
    return false
  }
  if (Array.isArray(raw)) {
    return raw.length === 1 && /^skipped$/i.test(String(raw[0] ?? '').trim())
  }
  return /^skipped$/i.test(String(raw).trim())
}

function hasNonEmptyElicitationAnswer(raw: string | readonly string[]): boolean {
  if (Array.isArray(raw)) {
    return raw.some(item => String(item).trim().length > 0)
  }
  return String(raw).trim().length > 0
}

function firstAnswerText(raw: string | readonly string[]): string {
  if (Array.isArray(raw)) {
    return String(raw[0] ?? '').trim()
  }
  return String(raw).trim()
}
