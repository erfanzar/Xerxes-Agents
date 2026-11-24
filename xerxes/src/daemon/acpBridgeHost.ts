// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, readFile, realpath, stat, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path'

import {
  AcpRpcError,
  AcpStdioClient,
  isAcpAuthRequiredError,
  type AcpAvailableCommand,
  type AcpClientEvent,
  type AcpClientHostHandlers,
  type AcpClientPermissionRequest,
  type AcpClientPermissionResponse,
  type AcpElicitationRequest,
  type AcpElicitationResponse,
  type AcpInitializeResult,
  type AcpMcpCapabilities,
  type AcpPromptCapabilities,
  type AcpPromptContentBlock,
  type AcpPromptUsage,
  type AcpSessionCapabilities,
  type AcpStdioClientOptions,
} from '../acp/client.js'
import {
  ACP_TURBO_MODE_ID,
  ACP_TURBO_MODE_OPTION,
  bridgeEventFromAcpClientEvent,
  classifyAcpRuntimeFeatures,
  ClaudeStreamJsonRunner,
  DEFAULT_CLAUDE_CODE_MODELS,
  discoverExternalAcpAgents,
  isAcpTurboMode,
  isUnrestrictedAcpModeId,
  resolveExecutablePath,
  splitCommandString,
  type AcpBridgeStreamEvent,
  type AcpExternalAgentInfo,
  type AcpExternalAgentProvider,
  type AcpExternalConfigOption,
  type AcpExternalConfigOptionChoice,
  type AcpExternalModeOption,
  type AcpExternalModelOption,
  type AcpRuntimeAuthMethod,
  type AcpRuntimeCapabilities,
  type AcpSessionBinding,
  type AcpToolMetaSnapshot,
  type ClaudeStreamJsonRunnerOptions,
  type DiscoverExternalAcpAgentsOptions,
} from '../acp/externalAgents.js'
import { isAcpRecord } from '../acp/types.js'
import { buildAgySubprocessEnv, loadAgyRuntimeEnv } from '../auth/agyRuntimeEnv.js'
import { errorMessage } from '../executors/toolRegistry.js'
import {
  companionCliSearchDirectories,
  ensureAgyCli,
  ensureClaudeCode,
  ensureCodexCli,
  type EnsureAgyCliHost,
  type EnsureAgyCliResult,
  type EnsureClaudeCodeHost,
  type EnsureClaudeCodeResult,
  type EnsureCodexCliHost,
  type EnsureCodexCliResult,
} from '../runtime/companionInstall.js'

export type AcpDefaultCreateElicitationHandler = (
  sessionId: string,
  request: AcpElicitationRequest,
  signal?: AbortSignal,
) => Promise<AcpElicitationResponse> | AcpElicitationResponse

export type AcpDefaultRequestPermissionHandler = (
  sessionId: string,
  request: AcpClientPermissionRequest,
  signal?: AbortSignal,
) => Promise<AcpClientPermissionResponse> | AcpClientPermissionResponse

export type AcpTerminalClientHandlers = Pick<
  AcpClientHostHandlers,
  'createTerminal' | 'getTerminalOutput' | 'waitForTerminalExit' | 'killTerminal' | 'releaseTerminal'
>

export interface DaemonAcpBridgeHostOptions {
  /** Test seam to override external ACP client construction. */
  readonly createAcpClient?: (options: AcpStdioClientOptions) => AcpStdioClient
  /** Test seam to override Claude stream-json runner construction. */
  readonly createClaudeRunner?: (options: ClaudeStreamJsonRunnerOptions) => ClaudeStreamJsonRunner
  /** Session-level fallback elicitation handler for out-of-turn ACP wakeups. */
  readonly defaultCreateElicitation?: AcpDefaultCreateElicitationHandler
  /** Session-level fallback permission handler for out-of-turn ACP wakeups. */
  readonly defaultRequestPermission?: AcpDefaultRequestPermissionHandler
  /** Test seam to override agent discovery. */
  readonly discoverAgents?: (options?: DiscoverExternalAcpAgentsOptions) => readonly AcpExternalAgentInfo[]
  /** Test seam to override automatic `agy` installation when Antigravity is selected. */
  readonly ensureAgy?: (host: EnsureAgyCliHost) => Promise<EnsureAgyCliResult>
  /** Test seam to override automatic `claude` installation when Claude Code is selected. */
  readonly ensureClaudeCode?: (host: EnsureClaudeCodeHost) => Promise<EnsureClaudeCodeResult>
  /** Test seam to override automatic `codex` installation when Codex is selected. */
  readonly ensureCodex?: (host: EnsureCodexCliHost) => Promise<EnsureCodexCliResult>
  /** Environment variables forwarded during discovery/resolution. */
  readonly env?: Readonly<Record<string, string | undefined>>
  /** Override home directory used for external runtime skill discovery and agy env lookup (deterministic tests). */
  readonly homeDir?: string
  /** Test seam for custom binary validation. */
  readonly resolveBinary?: (command: string) => string | undefined
  /** Optional ACP `terminal/*` client handlers; `clientCapabilities.terminal` is true only when provided. */
  readonly terminalHandlers?: AcpTerminalClientHandlers
}

export interface AcpConnectMcpServerInput {
  readonly args?: readonly string[]
  readonly command?: string
  readonly enabled?: boolean
  readonly env?: Readonly<Record<string, string>>
  readonly headers?: Readonly<Record<string, string>>
  readonly name: string
  readonly transport?: 'stdio' | 'sse' | 'streamable_http'
  readonly url?: string
}

export interface AcpConnectRequest {
  readonly additionalDirectories?: readonly string[]
  readonly agent?: string
  readonly agentId: string
  readonly args?: readonly string[]
  readonly command?: string
  readonly cwd: string
  readonly ideMcpPort?: number
  readonly initialModeId?: string
  readonly mcpServers?: readonly AcpConnectMcpServerInput[]
  readonly permissionMode?: string
  readonly resumeSessionId?: string
  readonly sandbox?: boolean
  readonly sessionId: string
}

export type AcpConnectResult =
  | { readonly binding: AcpSessionBinding; readonly ok: true }
  | {
      readonly auth_methods?: readonly AcpRuntimeAuthMethod[]
      readonly auth_required?: boolean
      readonly error: string
      readonly install_hint?: string
      readonly ok: false
    }

export type AcpBindingMutationResult =
  | { readonly binding: AcpSessionBinding; readonly ok: true }
  | { readonly error: string; readonly ok: false }

export interface AcpAuthenticateRequest {
  readonly agentId?: string
  readonly cwd?: string
  readonly methodId: string
  readonly sessionId: string
}

export type AcpAuthenticateResult =
  | { readonly mode: 'agent'; readonly ok: true }
  | {
      readonly args: readonly string[]
      readonly command: string
      readonly env: Readonly<Record<string, string>>
      readonly label: string
      readonly mode: 'terminal'
      readonly ok: true
    }
  | { readonly error: string; readonly ok: false }

export interface AcpListedSessionWire {
  readonly cwd?: string
  readonly meta?: Readonly<Record<string, unknown>>
  readonly session_id: string
  readonly title?: string
  readonly updated_at?: string
}

export type AcpListSessionsHostResult =
  | {
      readonly next_cursor?: string
      readonly ok: true
      readonly sessions: readonly AcpListedSessionWire[]
    }
  | { readonly error: string; readonly ok: false }

export interface ExtractedAcpModelAndModeState {
  readonly config_options?: readonly AcpExternalConfigOption[]
  readonly current_mode?: string
  readonly current_mode_name?: string
  readonly current_model?: string
  readonly current_model_name?: string
  readonly mode_config_id?: string
  readonly model_config_id?: string
  readonly models?: readonly AcpExternalModelOption[]
  readonly modes?: readonly AcpExternalModeOption[]
  readonly thought_level?: string
  readonly thought_level_config_id?: string
  readonly thought_levels?: readonly AcpExternalModeOption[]
}

export interface AcpPromptImageInput {
  readonly data: string
  readonly mediaType: string
}

export interface AcpPromptAudioInput {
  readonly data: string
  readonly mediaType: string
}

export interface AcpPromptResourceInput {
  readonly blob?: string
  readonly description?: string
  readonly mimeType?: string
  readonly name?: string
  readonly path?: string
  readonly size?: number
  readonly text?: string
  readonly title?: string
  readonly type?: 'resource' | 'resource_link'
  readonly uri: string
}

export interface AcpRunPromptRequest {
  readonly audio?: readonly AcpPromptAudioInput[]
  readonly createElicitation?: (
    request: AcpElicitationRequest,
    signal?: AbortSignal,
  ) => Promise<AcpElicitationResponse> | AcpElicitationResponse
  readonly fallbackInlineText?: string
  readonly images?: readonly AcpPromptImageInput[]
  readonly onEvent: (event: AcpBridgeStreamEvent) => void
  readonly permissionMode?: string
  readonly requestPermission?: (
    request: AcpClientPermissionRequest,
  ) => Promise<AcpClientPermissionResponse> | AcpClientPermissionResponse
  readonly resources?: readonly AcpPromptResourceInput[]
  readonly sessionId: string
  readonly signal?: AbortSignal
  readonly text: string
}

export interface AcpSteerPromptRequest {
  readonly acceptProbeMs?: number
  readonly audio?: readonly AcpPromptAudioInput[]
  readonly images?: readonly AcpPromptImageInput[]
  readonly onFallbackQueue?: () => Promise<void> | void
  readonly resources?: readonly AcpPromptResourceInput[]
  readonly sessionId: string
  readonly signal?: AbortSignal
  readonly text: string
}

export type AcpSteerPromptResult =
  | { readonly ok: true; readonly settled: boolean; readonly stopReason?: string }
  | { readonly error: string; readonly ok: false; readonly rejectedConcurrent: boolean }

export interface AcpRunPromptResult {
  readonly assistantText: string
  readonly cancelled: boolean
  readonly error?: string
  readonly externalSessionId?: string
  readonly stopReason: string
  readonly thinkingText: string
  readonly usage?: AcpPromptUsage
}

export interface AcpSideQuestionToolEvent {
  readonly name: string
  readonly status: 'done' | 'error' | 'running'
  readonly target: string
}

export interface AcpSideQuestionRequest {
  /** Compact prompt used when `session/fork` succeeds and the forked session already holds native history. */
  readonly forkedPrompt?: string
  readonly model?: string
  readonly onDelta?: (delta: string) => void
  readonly onTool?: (tool: AcpSideQuestionToolEvent) => void
  readonly prompt: string
  readonly sessionId: string
  readonly signal?: AbortSignal
  /** Whether to prefer `session/fork` when advertised by the agent (defaults to true). */
  readonly useFork?: boolean
}

export interface AcpSideQuestionResult {
  readonly answer: string
  readonly error?: string
  readonly model: string
  readonly ok: boolean
}

function isReadOnlySideQuestionPermission(request: AcpClientPermissionRequest): boolean {
  const kind = (request.kind ?? '').trim().toLowerCase()
  if (kind === 'edit' || kind === 'write' || kind === 'delete' || kind === 'execute' || kind === 'move') {
    return false
  }
  const name = `${request.toolName} ${request.title}`.toLowerCase()
  if (/\b(edit|write|delete|remove|rm|run_command|command|bash|shell|exec|patch|apply|create|mkdir|mv|cp|schedule|kill)\b/.test(name)) {
    return false
  }
  if (kind === 'read' || kind === 'search' || kind === 'fetch' || kind === 'think') {
    return true
  }
  return /\b(read|view|list|glob|grep|find|search|outline|ls|cat|head|tail|stat)\b/.test(name)
}

function acpSideQuestionToolTarget(name: string, input: Record<string, unknown>, cwd: string): string {
  const rawPath =
    (typeof input.Path === 'string' && input.Path) ||
    (typeof input.path === 'string' && input.path) ||
    (typeof input.file_path === 'string' && input.file_path) ||
    (typeof input.SearchPath === 'string' && input.SearchPath) ||
    (typeof input.DirectoryPath === 'string' && input.DirectoryPath) ||
    (typeof input.query === 'string' && input.query) ||
    (typeof input.Query === 'string' && input.Query) ||
    (typeof input.pattern === 'string' && input.pattern) ||
    ''
  if (!rawPath) return name
  if (isAbsolute(rawPath)) {
    const rel = relative(cwd, rawPath)
    if (rel && !rel.startsWith('..') && !isAbsolute(rel)) return rel
  }
  return rawPath.slice(0, 120)
}

const MAX_ACP_TOOL_NAMES = 256
const SESSION_CLOSE_TIMEOUT_MS = 500
const ANTIGRAVITY_AUTH_ERROR =
  'Antigravity is not signed in. Run `agy acp --login` (or `agy`) in your terminal to sign in with Google, then click Retry.'

interface ActiveAcpSession {
  activeAbort: AbortController | undefined
  activeElicitationAbort: AbortController | undefined
  activeElicitationHandler:
    | ((request: AcpElicitationRequest) => Promise<AcpElicitationResponse> | AcpElicitationResponse)
    | undefined
  activePermissionHandler:
    | ((request: AcpClientPermissionRequest) => Promise<AcpClientPermissionResponse> | AcpClientPermissionResponse)
    | undefined
  activeSyntheticUpdate?: ((event: AcpClientEvent) => void) | undefined
  readonly additionalDirectories: readonly string[]
  binding: AcpSessionBinding
  readonly client: AcpStdioClient
  externalSessionId: string
  hostTurboActive: boolean
  hostTurboUnderlyingMode: string | undefined
  readonly kind: 'acp'
  readonly liveSteerTasks: Set<Promise<void>>
  readonly mcpServers: readonly Readonly<Record<string, unknown>>[]
  nativeModes: readonly AcpExternalModeOption[]
  outOfTurnInteractionAbort: AbortController
  readonly outOfTurnSettledToolCallIds: Set<string>
  readonly promptCapabilities: AcpPromptCapabilities
  readonly readRepair?: ReturnType<typeof makeAntigravityReadRepair>
  readonly sessionCapabilities: AcpSessionCapabilities
  readonly sideSessionIds: Set<string>
  readonly toolMetaById: Map<string, AcpToolMetaSnapshot>
  readonly toolNamesById: Map<string, string>
  readonly updateStepsPlans?: ReturnType<typeof makeAntigravityUpdateStepsPlans>
}

interface ActiveClaudeSession {
  activeAbort: AbortController | undefined
  binding: AcpSessionBinding
  readonly kind: 'claude-stream-json'
  readonly runner: ClaudeStreamJsonRunner
}

interface LastAcpAuthChallenge {
  readonly agentId: string
  readonly args: readonly string[]
  readonly authMethods: readonly AcpRuntimeAuthMethod[]
  readonly command: string
  readonly cwd: string
  readonly provider: AcpExternalAgentProvider
}

type ActiveExternalSession = ActiveAcpSession | ActiveClaudeSession

/**
 * Daemon-owned host managing per-session external ACP & Claude Code bridges.
 *
 * Keeps external agent subprocess lifecycles bound to their Xerxes session so
 * `/new`, session eviction, turn cancellation, and daemon shutdown never leak
 * child processes or orphan permissions.
 */
export class DaemonAcpBridgeHost {
  private readonly active = new Map<string, ActiveExternalSession>()
  private readonly lastAuthChallengeBySession = new Map<string, LastAcpAuthChallenge>()
  private readonly bindingListeners = new Set<(sessionId: string, binding: AcpSessionBinding) => void>()
  private readonly sessionEventListeners = new Set<(sessionId: string, event: AcpBridgeStreamEvent) => void>()
  private readonly options: DaemonAcpBridgeHostOptions
  private defaultCreateElicitation: AcpDefaultCreateElicitationHandler | undefined
  private defaultRequestPermission: AcpDefaultRequestPermissionHandler | undefined

  constructor(options: DaemonAcpBridgeHostOptions = {}) {
    this.options = options
    this.defaultCreateElicitation = options.defaultCreateElicitation
    this.defaultRequestPermission = options.defaultRequestPermission
  }

  /** Attach or replace persistent session-level fallback handlers for out-of-turn ACP requests. */
  setDefaultHandlers(handlers: {
    readonly createElicitation?: AcpDefaultCreateElicitationHandler
    readonly requestPermission?: AcpDefaultRequestPermissionHandler
  }): void {
    this.defaultCreateElicitation = handlers.createElicitation
    this.defaultRequestPermission = handlers.requestPermission
  }

  /** Last known auth methods for a session (active binding or most recent auth-required connect failure). */
  getAuthMethods(sessionId: string): readonly AcpRuntimeAuthMethod[] {
    const active = this.getBinding(sessionId)
    if (active?.capabilities?.auth_methods && active.capabilities.auth_methods.length > 0) {
      return active.capabilities.auth_methods
    }
    return this.lastAuthChallengeBySession.get(sessionId)?.authMethods ?? []
  }

  /** Resolved home directory for external ACP workspace & credential discovery. */
  getHomeDir(): string {
    return this.options.homeDir ?? (this.options.env?.HOME?.trim() || homedir())
  }

  /** Subscribe to asynchronous binding updates (e.g., out-of-turn ACP `session/update` frames or unexpected exits). */
  onBindingChange(listener: (sessionId: string, binding: AcpSessionBinding) => void): () => void {
    this.bindingListeners.add(listener)
    return () => {
      this.bindingListeners.delete(listener)
    }
  }

  /** Subscribe to asynchronous out-of-turn session updates (e.g. initial `usage_update` or `session_info_update`). */
  onSessionEvent(listener: (sessionId: string, event: AcpBridgeStreamEvent) => void): () => void {
    this.sessionEventListeners.add(listener)
    return () => {
      this.sessionEventListeners.delete(listener)
    }
  }

  listAgents(sessionId?: string): {
    readonly active: AcpSessionBinding | null
    readonly agents: readonly AcpExternalAgentInfo[]
  } {
    const agents = this.options.discoverAgents
      ? this.options.discoverAgents(this.options.env ? { env: this.options.env } : undefined)
      : discoverExternalAcpAgents(this.options.env ? { env: this.options.env } : undefined)
    return {
      agents,
      active: sessionId ? this.getBinding(sessionId) : null,
    }
  }

  getBinding(sessionId: string): AcpSessionBinding | null {
    const entry = this.active.get(sessionId)
    return entry && entry.binding.connected ? entry.binding : null
  }

  hasActiveBridge(sessionId: string): boolean {
    return this.getBinding(sessionId) !== null
  }

  hasActiveTurn(sessionId: string): boolean {
    return this.active.get(sessionId)?.activeAbort !== undefined
  }

  isTurboMode(sessionId: string): boolean {
    const entry = this.active.get(sessionId)
    if (!entry || !entry.binding.connected || entry.kind !== 'acp') {
      return false
    }
    return entry.hostTurboActive || isAcpTurboMode(entry.binding)
  }

  async connect(request: AcpConnectRequest): Promise<AcpConnectResult> {
    const sessionId = request.sessionId.trim()
    const agentId = request.agentId.trim() || (request.command?.trim() ? 'custom' : '')
    const cwd = resolve(request.cwd)
    if (!sessionId) {
      return { ok: false, error: 'session_id is required to connect an ACP agent.' }
    }
    if (!agentId) {
      return { ok: false, error: 'agent_id is required.' }
    }

    // Switching back to native Xerxes disconnects any active external subprocess.
    if (agentId === 'xerxes') {
      await this.disconnect(sessionId)
      return {
        ok: true,
        binding: {
          session_id: sessionId,
          agent_id: 'xerxes',
          name: 'Xerxes',
          provider: 'xerxes',
          transport: 'native',
          cwd,
          connected: true,
        },
      }
    }

    const { agents } = this.listAgents(sessionId)
    const discovered = agents.find(item => item.id === agentId)
    const customCommandRaw = request.command?.trim()

    if (customCommandRaw) {
      let command = customCommandRaw
      let args: readonly string[] = request.args ? [...request.args] : []
      if (args.length === 0 && /\s/.test(customCommandRaw)) {
        const split = splitCommandString(customCommandRaw)
        command = split[0] ?? customCommandRaw
        args = split.slice(1)
      }
      const resolvedBinary = this.resolveCommandBinary(command)
      if (!resolvedBinary && !this.options.createAcpClient) {
        return {
          ok: false,
          error: `ACP command not found or not executable: "${command}".`,
          install_hint: 'Provide a binary available on PATH or an absolute path to an ACP stdio executable.',
        }
      }
      const provider: AcpExternalAgentProvider =
        discovered?.provider ??
        (agentId === 'antigravity'
          ? 'antigravity'
          : agentId === 'claude-code'
            ? 'claude-code'
            : agentId === 'codex'
              ? 'codex'
              : 'acp')
      const displayName = discovered?.name ?? (agentId === 'custom' ? command : agentId)
      return this.connectStdioAcp({
        sessionId,
        agentId,
        name: displayName,
        provider,
        cwd,
        command: resolvedBinary ?? command,
        args,
        ...(request.additionalDirectories?.length ? { additionalDirectories: request.additionalDirectories } : {}),
        ...(request.agent ? { agent: request.agent } : {}),
        ...(request.ideMcpPort !== undefined ? { ideMcpPort: request.ideMcpPort } : {}),
        ...(request.initialModeId ? { initialModeId: request.initialModeId } : {}),
        ...(request.mcpServers?.length ? { mcpServers: request.mcpServers } : {}),
        ...(request.resumeSessionId ? { resumeSessionId: request.resumeSessionId } : {}),
        ...(request.sandbox ? { sandbox: true } : {}),
        ...(discovered?.installHint ? { installHint: discovered.installHint } : {}),
      })
    }

    if (!discovered) {
      return {
        ok: false,
        error: `Unknown ACP agent "${agentId}".`,
        install_hint: 'Use "xerxes", "claude-code", "antigravity", or specify a custom command.',
      }
    }

    let resolvedDiscovered = discovered
    if (!resolvedDiscovered.available) {
      const envOpt = this.options.env ? { environment: this.options.env } : {}
      const autoInstallTask =
        resolvedDiscovered.provider === 'antigravity' && (!this.options.discoverAgents || this.options.ensureAgy)
          ? () => (this.options.ensureAgy ?? ensureAgyCli)({
              ...envOpt,
              find: () => this.resolveCommandBinary('agy'),
            })
          : resolvedDiscovered.provider === 'claude-code' && (!this.options.discoverAgents || this.options.ensureClaudeCode)
            ? () => (this.options.ensureClaudeCode ?? ensureClaudeCode)({
                ...envOpt,
                find: () => this.resolveCommandBinary('claude'),
              })
            : resolvedDiscovered.provider === 'codex' && (!this.options.discoverAgents || this.options.ensureCodex)
              ? () => (this.options.ensureCodex ?? ensureCodexCli)({
                  ...envOpt,
                  find: () => this.resolveCommandBinary('codex'),
                })
              : undefined

      if (autoInstallTask) {
        try {
          await autoInstallTask()
          const refreshed = this.listAgents(sessionId).agents.find(item => item.id === agentId)
          if (refreshed) {
            resolvedDiscovered = refreshed
          }
        } catch (error) {
          return {
            ok: false,
            error: error instanceof Error ? error.message : `${resolvedDiscovered.name} is not installed or unavailable on PATH.`,
            ...(resolvedDiscovered.installHint ? { install_hint: resolvedDiscovered.installHint } : {}),
          }
        }
      }
    }

    if (!resolvedDiscovered.available) {
      return {
        ok: false,
        error: `${resolvedDiscovered.name} is not installed or unavailable on PATH.`,
        ...(resolvedDiscovered.installHint ? { install_hint: resolvedDiscovered.installHint } : {}),
      }
    }

    if (resolvedDiscovered.transport === 'claude-stream-json') {
      const binaryPath = resolvedDiscovered.binaryPath ?? resolvedDiscovered.command ?? ''
      if (!binaryPath) {
        return {
          ok: false,
          error: `${resolvedDiscovered.name} binary path could not be resolved.`,
          ...(resolvedDiscovered.installHint ? { install_hint: resolvedDiscovered.installHint } : {}),
        }
      }
      if (!this.options.createClaudeRunner && !this.resolveCommandBinary(binaryPath)) {
        return {
          ok: false,
          error: `${resolvedDiscovered.name} binary is no longer executable at "${binaryPath}".`,
          ...(resolvedDiscovered.installHint ? { install_hint: resolvedDiscovered.installHint } : {}),
        }
      }
      await this.disconnect(sessionId)
      const runnerOptions: ClaudeStreamJsonRunnerOptions = {
        binaryPath,
        cwd,
        ...(this.options.env ? { env: this.options.env } : {}),
        ...(request.permissionMode ? { permissionMode: request.permissionMode } : {}),
        ...(request.resumeSessionId ? { resumeSessionId: request.resumeSessionId } : {}),
      }
      const runner = this.options.createClaudeRunner
        ? this.options.createClaudeRunner(runnerOptions)
        : new ClaudeStreamJsonRunner(runnerOptions)
      const initialModel = runner.getModel()
      const initialModelName = initialModel
        ? (DEFAULT_CLAUDE_CODE_MODELS.find(m => m.id === initialModel)?.name ?? initialModel)
        : undefined
      const claudeFeatures = classifyAcpRuntimeFeatures({
        provider: resolvedDiscovered.provider,
        agentName: resolvedDiscovered.name,
        cwd,
        includeFilesystemUnlisted: true,
        ...(this.options.env ? { env: this.options.env } : {}),
        ...(this.options.homeDir ? { homeDir: this.options.homeDir } : {}),
      })
      const claudeCapabilities: AcpRuntimeCapabilities = {
        prompt: {
          image: false,
          audio: false,
          embedded_context: false,
        },
        session: {
          load: false,
          resume: true,
          fork: false,
          list: false,
          close: false,
          additional_directories: false,
        },
        mcp: {
          http: false,
          sse: false,
        },
        auth_methods: [],
      }
      const binding: AcpSessionBinding = {
        session_id: sessionId,
        agent_id: resolvedDiscovered.id,
        name: resolvedDiscovered.name,
        provider: resolvedDiscovered.provider,
        transport: 'claude-stream-json',
        cwd,
        ...(request.resumeSessionId ? { external_session_id: request.resumeSessionId } : {}),
        connected: true,
        command: binaryPath,
        agent_title: resolvedDiscovered.name,
        ...(resolvedDiscovered.version ? { agent_version: resolvedDiscovered.version } : {}),
        capabilities: claudeCapabilities,
        models: DEFAULT_CLAUDE_CODE_MODELS,
        ...(initialModel ? { current_model: initialModel } : {}),
        ...(initialModelName ? { current_model_name: initialModelName } : {}),
        ...(claudeFeatures.available_commands.length > 0
          ? { available_commands: claudeFeatures.available_commands }
          : {}),
        ...(claudeFeatures.discovered_skills.length > 0
          ? { discovered_skills: claudeFeatures.discovered_skills }
          : {}),
      }
      this.active.set(sessionId, {
        kind: 'claude-stream-json',
        runner,
        binding,
        activeAbort: undefined,
      })
      return { ok: true, binding }
    }

    // Standard ACP stdio agent (claude-agent-acp, agy acp, antigravity-acp, custom config).
    const command = resolvedDiscovered.binaryPath ?? resolvedDiscovered.command ?? ''
    if (!command) {
      return {
        ok: false,
        error: `${resolvedDiscovered.name} has no configured ACP command.`,
        ...(resolvedDiscovered.installHint ? { install_hint: resolvedDiscovered.installHint } : {}),
      }
    }
    return this.connectStdioAcp({
      sessionId,
      agentId: resolvedDiscovered.id,
      name: resolvedDiscovered.name,
      provider: resolvedDiscovered.provider,
      cwd,
      command,
      args: resolvedDiscovered.args ?? [],
      ...(request.additionalDirectories?.length ? { additionalDirectories: request.additionalDirectories } : {}),
      ...(request.agent ? { agent: request.agent } : {}),
      ...(request.ideMcpPort !== undefined ? { ideMcpPort: request.ideMcpPort } : {}),
      ...(request.initialModeId ? { initialModeId: request.initialModeId } : {}),
      ...(request.mcpServers?.length ? { mcpServers: request.mcpServers } : {}),
      ...(request.resumeSessionId ? { resumeSessionId: request.resumeSessionId } : {}),
      ...(request.sandbox ? { sandbox: true } : {}),
      ...(resolvedDiscovered.installHint ? { installHint: resolvedDiscovered.installHint } : {}),
    })
  }

  async disconnect(sessionId: string): Promise<boolean> {
    this.lastAuthChallengeBySession.delete(sessionId)
    const existing = this.active.get(sessionId)
    if (!existing) {
      return false
    }
    this.active.delete(sessionId)
    existing.activeAbort?.abort(new Error('External ACP agent disconnected'))
    existing.activeAbort = undefined
    if (existing.kind === 'acp') {
      existing.activePermissionHandler = undefined
      existing.activeElicitationAbort?.abort(new Error('External ACP agent disconnected'))
      existing.activeElicitationAbort = undefined
      existing.activeElicitationHandler = undefined
      existing.outOfTurnInteractionAbort.abort(new Error('External ACP agent disconnected'))
      // Yield a macrotask tick so aborted permission/elicitation handler chains finish
      // flushing `{ outcome: 'cancelled' }` / `{ action: 'cancel' }` replies before stdin closes.
      await new Promise<void>(resolve => setTimeout(resolve, 0))
      if (
        existing.sessionCapabilities.close &&
        !existing.client.isDisposed &&
        !existing.client.hasExited
      ) {
        await existing.client
          .closeSession(existing.externalSessionId, SESSION_CLOSE_TIMEOUT_MS)
          .catch(() => undefined)
      }
      await existing.client.dispose().catch(() => undefined)
    } else {
      await existing.runner.dispose().catch(() => undefined)
    }
    return true
  }

  async cancel(sessionId: string): Promise<boolean> {
    const existing = this.active.get(sessionId)
    if (!existing) {
      return false
    }
    let cancelled = false
    if (existing.activeAbort && !existing.activeAbort.signal.aborted) {
      existing.activeAbort.abort(new Error('External ACP turn cancelled'))
      cancelled = true
    }
    if (existing.kind === 'acp') {
      if (!existing.outOfTurnInteractionAbort.signal.aborted) {
        existing.outOfTurnInteractionAbort.abort(new Error('External ACP interaction cancelled'))
      }
      existing.outOfTurnInteractionAbort = new AbortController()
      await existing.client.cancel(existing.externalSessionId).catch(() => undefined)
      return cancelled || true
    }
    return existing.runner.cancel() || cancelled
  }

  /** Cancel every currently active external ACP turn across all sessions. */
  cancelAll(): number {
    let cancelled = 0
    for (const [sessionId, existing] of this.active) {
      if (!existing.activeAbort || existing.activeAbort.signal.aborted) {
        continue
      }
      cancelled += 1
      void this.cancel(sessionId).catch(() => undefined)
    }
    return cancelled
  }

  async setModel(sessionId: string, modelId: string): Promise<AcpBindingMutationResult> {
    const sid = sessionId.trim()
    const targetModelId = modelId.trim()
    if (!sid) {
      return { ok: false, error: 'session_id is required to set an ACP model.' }
    }
    if (!targetModelId) {
      return { ok: false, error: 'model_id is required.' }
    }
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }

    if (existing.kind === 'claude-stream-json') {
      existing.runner.setModel(targetModelId)
      const models = existing.binding.models ?? DEFAULT_CLAUDE_CODE_MODELS
      const matched = models.find(item => item.id === targetModelId)
      existing.binding = {
        ...existing.binding,
        models,
        current_model: targetModelId,
        current_model_name: matched?.name ?? targetModelId,
      }
      return { ok: true, binding: existing.binding }
    }

    try {
      const rpcResult = await existing.client.setModel({
        sessionId: existing.externalSessionId,
        modelId: targetModelId,
        ...(existing.binding.model_config_id ? { configId: existing.binding.model_config_id } : {}),
      })
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: rpcResult.configOptions ?? rpcResult.config_options,
        models: rpcResult.models,
        modes: rpcResult.modes,
        raw: rpcResult,
      })
      if (extracted.modes && extracted.modes.length > 0) {
        existing.nativeModes = extracted.modes
      }
      const merged = mergeBindingModelAndModeState(existing.binding, {
        ...extracted,
        current_model: extracted.current_model ?? targetModelId,
      })
      existing.binding = projectAcpBindingWithHostTurbo(existing, merged)
      return { ok: true, binding: existing.binding }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to switch ${existing.binding.name} model to "${targetModelId}": ${errorMessage(error)}`,
      }
    }
  }

  async setMode(
    sessionId: string,
    modeId: string,
    explicitConfigId?: string,
  ): Promise<AcpBindingMutationResult> {
    const sid = sessionId.trim()
    const targetModeId = modeId.trim()
    if (!sid) {
      return { ok: false, error: 'session_id is required to set an ACP mode.' }
    }
    if (!targetModeId) {
      return { ok: false, error: 'mode_id is required.' }
    }
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }
    if (existing.kind !== 'acp') {
      return {
        ok: false,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support ACP mode switching.`,
      }
    }

    const modeConfigId = explicitConfigId?.trim() || resolveModeConfigId(existing.binding)
    const isSyntheticTurboRequest =
      targetModeId.toLowerCase() === ACP_TURBO_MODE_ID &&
      !existing.nativeModes.some(item => item.id.trim().toLowerCase() === ACP_TURBO_MODE_ID)

    if (isSyntheticTurboRequest) {
      const underlyingMode = pickNativeUnderlyingModeForTurbo(existing.nativeModes)
      const prevHostTurbo = existing.hostTurboActive
      const prevUnderlying = existing.hostTurboUnderlyingMode
      existing.hostTurboActive = true
      existing.hostTurboUnderlyingMode = underlyingMode
      try {
        let merged = existing.binding
        if (underlyingMode) {
          const rpcResult = await existing.client.setMode({
            sessionId: existing.externalSessionId,
            modeId: underlyingMode,
            ...(modeConfigId ? { configId: modeConfigId } : {}),
          })
          const extracted = extractAcpSessionModelAndModeState({
            configOptions: rpcResult.configOptions ?? rpcResult.config_options,
            models: rpcResult.models,
            modes: rpcResult.modes,
            raw: rpcResult,
          })
          if (extracted.modes && extracted.modes.length > 0) {
            existing.nativeModes = extracted.modes
          }
          merged = mergeBindingModelAndModeState(existing.binding, {
            ...extracted,
            ...(modeConfigId ? { mode_config_id: extracted.mode_config_id ?? modeConfigId } : {}),
          })
        }
        existing.binding = projectAcpBindingWithHostTurbo(existing, merged)
        return { ok: true, binding: existing.binding }
      } catch (error) {
        existing.hostTurboActive = prevHostTurbo
        existing.hostTurboUnderlyingMode = prevUnderlying
        return {
          ok: false,
          error: `Failed to switch ${existing.binding.name} mode to "${ACP_TURBO_MODE_ID}": ${errorMessage(error)}`,
        }
      }
    }

    try {
      const rpcResult = await existing.client.setMode({
        sessionId: existing.externalSessionId,
        modeId: targetModeId,
        ...(modeConfigId ? { configId: modeConfigId } : {}),
      })
      existing.hostTurboActive = false
      existing.hostTurboUnderlyingMode = undefined
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: rpcResult.configOptions ?? rpcResult.config_options,
        models: rpcResult.models,
        modes: rpcResult.modes,
        raw: rpcResult,
      })
      if (extracted.modes && extracted.modes.length > 0) {
        existing.nativeModes = extracted.modes
      }
      const merged = mergeBindingModelAndModeState(existing.binding, {
        ...extracted,
        ...(modeConfigId ? { mode_config_id: extracted.mode_config_id ?? modeConfigId } : {}),
        current_mode: extracted.current_mode ?? targetModeId,
      })
      existing.binding = projectAcpBindingWithHostTurbo(existing, merged)
      return { ok: true, binding: existing.binding }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to switch ${existing.binding.name} mode to "${targetModeId}": ${errorMessage(error)}`,
      }
    }
  }

  async setConfigOption(
    sessionId: string,
    configId: string,
    value: string | boolean,
    explicitType?: string,
  ): Promise<AcpBindingMutationResult> {
    const sid = sessionId.trim()
    const targetConfigId = configId.trim()
    if (!sid) {
      return { ok: false, error: 'session_id is required to set an ACP config option.' }
    }
    if (!targetConfigId) {
      return { ok: false, error: 'config_id is required.' }
    }
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }

    const existingOpt = existing.binding.config_options?.find(
      item => item.id.toLowerCase() === targetConfigId.toLowerCase(),
    )
    const isBooleanOption =
      typeof value === 'boolean' ||
      explicitType?.trim().toLowerCase() === 'boolean' ||
      existingOpt?.type === 'boolean'

    let normalizedValue: string | boolean
    if (isBooleanOption) {
      if (typeof value === 'boolean') {
        normalizedValue = value
      } else {
        const lower = value.trim().toLowerCase()
        if (lower === 'true' || lower === '1' || lower === 'on') {
          normalizedValue = true
        } else if (lower === 'false' || lower === '0' || lower === 'off') {
          normalizedValue = false
        } else {
          return { ok: false, error: `Boolean config option "${targetConfigId}" requires true or false.` }
        }
      }
    } else {
      const trimmed = String(value).trim()
      if (!trimmed) {
        return { ok: false, error: 'value is required.' }
      }
      normalizedValue = trimmed
    }

    if (existing.kind === 'claude-stream-json') {
      if (targetConfigId.toLowerCase() === 'model' && typeof normalizedValue === 'string') {
        return this.setModel(sid, normalizedValue)
      }
      return {
        ok: false,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support ACP config option "${targetConfigId}".`,
      }
    }

    if (typeof normalizedValue === 'string' && isModeConfigOptionId(existing.binding, targetConfigId)) {
      return this.setMode(sid, normalizedValue, targetConfigId)
    }

    try {
      const rpcResult = await existing.client.setConfigOption({
        sessionId: existing.externalSessionId,
        configId: targetConfigId,
        ...(isBooleanOption ? { type: 'boolean' as const } : {}),
        value: normalizedValue,
      })
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: rpcResult.configOptions ?? rpcResult.config_options,
        models: rpcResult.models,
        modes: rpcResult.modes,
        raw: rpcResult,
      })
      if (extracted.modes && extracted.modes.length > 0) {
        existing.nativeModes = extracted.modes
      }
      const nextBinding =
        extracted.config_options && extracted.config_options.length > 0
          ? mergeBindingModelAndModeState(existing.binding, extracted)
          : applySingleConfigOptionPatch(existing.binding, targetConfigId, normalizedValue, extracted)
      existing.binding = projectAcpBindingWithHostTurbo(existing, nextBinding)
      return { ok: true, binding: existing.binding }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to set ${existing.binding.name} option "${targetConfigId}" to "${String(normalizedValue)}": ${errorMessage(error)}`,
      }
    }
  }

  /**
   * Perform ACP authentication for an advertised `authMethod` (`agent` or `terminal`).
   * Never extracts, reads, or transfers external OAuth tokens.
   */
  async authenticate(request: AcpAuthenticateRequest): Promise<AcpAuthenticateResult> {
    const sid = request.sessionId.trim()
    const methodId = request.methodId.trim()
    if (!methodId) {
      return { ok: false, error: 'method_id is required.' }
    }
    const existing = sid ? this.active.get(sid) : undefined
    const activeAcp = existing && existing.kind === 'acp' && existing.binding.connected ? existing : undefined
    const cached = sid ? this.lastAuthChallengeBySession.get(sid) : undefined

    const knownMethods =
      activeAcp?.binding.capabilities?.auth_methods ??
      cached?.authMethods ??
      []
    const matchedMethod = knownMethods.find(item => item.id === methodId)

    const resolveCommandAndArgs = (): {
      readonly args: readonly string[]
      readonly command: string
      readonly cwd: string
      readonly provider: AcpExternalAgentProvider
    } | undefined => {
      if (activeAcp) {
        const discovered = this.listAgents(sid).agents.find(a => a.id === activeAcp.binding.agent_id)
        const command = discovered?.binaryPath ?? discovered?.command ?? splitCommandString(activeAcp.binding.command ?? '')[0] ?? ''
        const args = discovered?.args ?? splitCommandString(activeAcp.binding.command ?? '').slice(1)
        if (command) {
          return { command, args, cwd: activeAcp.binding.cwd, provider: activeAcp.binding.provider }
        }
      }
      if (cached && (!request.agentId || request.agentId === cached.agentId)) {
        return {
          command: cached.command,
          args: cached.args,
          cwd: cached.cwd,
          provider: cached.provider,
        }
      }
      if (request.agentId) {
        const discovered = this.listAgents(sid).agents.find(a => a.id === request.agentId)
        if (discovered?.available && (discovered.binaryPath ?? discovered.command)) {
          return {
            command: discovered.binaryPath ?? discovered.command!,
            args: discovered.args ?? [],
            cwd: resolve(request.cwd ?? process.cwd()),
            provider: discovered.provider,
          }
        }
      }
      return undefined
    }

    const terminalAuthMeta = isAcpRecord(matchedMethod?.meta?.['terminal-auth'])
      ? matchedMethod.meta['terminal-auth']
      : undefined
    const isTerminalMethod =
      matchedMethod?.type === 'terminal' || terminalAuthMeta !== undefined
    if (isTerminalMethod && matchedMethod) {
      const target = resolveCommandAndArgs()
      const metaCommand =
        typeof terminalAuthMeta?.command === 'string' && terminalAuthMeta.command.trim()
          ? terminalAuthMeta.command.trim()
          : undefined
      const command = metaCommand ?? target?.command ?? (target?.provider === 'antigravity' ? 'agy' : '')
      const defaultArgs =
        matchedMethod.args && matchedMethod.args.length > 0
          ? matchedMethod.args
          : target?.provider === 'antigravity'
            ? ['acp', '--login']
            : [...(target?.args ?? [])]
      const env = matchedMethod.env ?? {}
      const label =
        (typeof terminalAuthMeta?.label === 'string' && terminalAuthMeta.label.trim()) ||
        matchedMethod.name
      return {
        ok: true,
        mode: 'terminal',
        command,
        args: defaultArgs,
        env,
        label,
      }
    }

    if (activeAcp) {
      try {
        await activeAcp.client.authenticate(methodId)
        return { ok: true, mode: 'agent' }
      } catch (error) {
        return { ok: false, error: `ACP authentication failed: ${errorMessage(error)}` }
      }
    }

    const target = resolveCommandAndArgs()
    if (!target) {
      return {
        ok: false,
        error: 'No connected ACP agent or pending auth challenge found for this session.',
      }
    }
    const isAntigravity = target.provider === 'antigravity'
    const transferredAgyEnv = isAntigravity
      ? await loadAgyRuntimeEnv(this.options.homeDir ?? homedir())
      : {}
    const mergedEnv = isAntigravity
      ? buildAgySubprocessEnv(
          this.options.env
            ? { ...transferredAgyEnv, ...this.options.env }
            : { ...process.env, ...transferredAgyEnv },
        )
      : this.options.env
    const clientOptions: AcpStdioClientOptions = {
      command: target.command,
      args: target.args,
      cwd: target.cwd,
      ...(mergedEnv ? { env: mergedEnv } : {}),
    }
    const ephemeralClient = this.options.createAcpClient
      ? this.options.createAcpClient(clientOptions)
      : new AcpStdioClient(clientOptions)
    try {
      const initResult = await ephemeralClient.initialize()
      const discoveredMethod = initResult.authMethods.find(item => item.id === methodId)
      const discoveredTermMeta = isAcpRecord(discoveredMethod?.meta?.['terminal-auth'])
        ? discoveredMethod.meta['terminal-auth']
        : undefined
      if (discoveredMethod?.type === 'terminal' || discoveredTermMeta !== undefined) {
        const metaCommand =
          typeof discoveredTermMeta?.command === 'string' && discoveredTermMeta.command.trim()
            ? discoveredTermMeta.command.trim()
            : undefined
        const label =
          (typeof discoveredTermMeta?.label === 'string' && discoveredTermMeta.label.trim()) ||
          discoveredMethod?.name ||
          methodId
        return {
          ok: true,
          mode: 'terminal',
          command: metaCommand ?? target.command,
          args: discoveredMethod?.args ?? [],
          env: discoveredMethod?.env ?? {},
          label,
        }
      }
      await ephemeralClient.authenticate(methodId)
      return { ok: true, mode: 'agent' }
    } catch (error) {
      return { ok: false, error: `ACP authentication failed: ${errorMessage(error)}` }
    } finally {
      await ephemeralClient.dispose().catch(() => undefined)
    }
  }

  /** List sessions known to the connected external ACP agent (`session/list`). */
  async listExternalSessions(
    sessionId: string,
    params: { readonly cursor?: string; readonly cwd?: string } = {},
  ): Promise<AcpListSessionsHostResult> {
    const sid = sessionId.trim()
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }
    if (existing.kind !== 'acp') {
      return {
        ok: false,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support ACP session/list.`,
      }
    }
    try {
      const res = await existing.client.listSessions({
        cwd: params.cwd ? resolve(params.cwd) : existing.binding.cwd,
        ...(params.cursor ? { cursor: params.cursor } : {}),
      })
      const sessions: AcpListedSessionWire[] = res.sessions.map(item => ({
        session_id: item.sessionId,
        ...(item.cwd ? { cwd: item.cwd } : {}),
        ...(item.title ? { title: item.title } : {}),
        ...(item.updatedAt ? { updated_at: item.updatedAt } : {}),
        ...(item.meta ? { meta: item.meta } : {}),
      }))
      return {
        ok: true,
        sessions,
        ...(res.nextCursor ? { next_cursor: res.nextCursor } : {}),
      }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to list ${existing.binding.name} sessions: ${errorMessage(error)}`,
      }
    }
  }

  /** Fork the active external ACP session (`session/fork`) and switch this session binding to the fork. */
  async forkSession(
    sessionId: string,
    params: { readonly cwd?: string } = {},
  ): Promise<AcpBindingMutationResult> {
    const sid = sessionId.trim()
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }
    if (existing.kind !== 'acp') {
      return {
        ok: false,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support ACP session/fork.`,
      }
    }
    const cwd = params.cwd ? resolve(params.cwd) : existing.binding.cwd
    try {
      const forked = await existing.client.forkSession({
        sessionId: existing.externalSessionId,
        cwd,
        mcpServers: existing.mcpServers,
        ...(existing.sessionCapabilities.additionalDirectories && existing.additionalDirectories.length > 0
          ? { additionalDirectories: existing.additionalDirectories }
          : {}),
      })
      existing.externalSessionId = forked.sessionId
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: forked.configOptions ?? forked.raw.configOptions ?? forked.raw.config_options,
        models: forked.models,
        modes: forked.modes,
        raw: forked.raw,
      })
      if (extracted.modes && extracted.modes.length > 0) {
        existing.nativeModes = extracted.modes
      }
      const merged = mergeBindingModelAndModeState(
        {
          ...existing.binding,
          cwd,
          external_session_id: forked.sessionId,
        },
        extracted,
      )
      existing.binding = projectAcpBindingWithHostTurbo(existing, merged)
      return { ok: true, binding: existing.binding }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to fork ${existing.binding.name} session: ${errorMessage(error)}`,
      }
    }
  }

  /** True when the active ACP bridge can attempt native concurrent mid-turn steering. */
  supportsLiveSteer(sessionId: string): boolean {
    const existing = this.active.get(sessionId)
    if (!existing || !existing.binding.connected || existing.kind !== 'acp' || !existing.activeAbort) {
      return false
    }
    if (existing.binding.provider === 'antigravity') {
      return isAtLeastSemver(existing.binding.agent_version, 1, 5, 0)
    }
    const meta = existing.binding.capabilities?.meta
    return meta?.steering === true || meta?.midTurnSteering === true
  }

  /**
   * Dispatch a concurrent mid-turn `session/prompt` steer on an active ACP turn
   * (e.g. `agy acp 1.5.0+`). If the agent rejects concurrent prompts, returns
   * `{ ok: false, rejectedConcurrent: true }` so callers cleanly fall back to
   * queued steer continuation.
   */
  async steerPrompt(request: AcpSteerPromptRequest): Promise<AcpSteerPromptResult> {
    const existing = this.active.get(request.sessionId)
    if (!existing || !existing.binding.connected || existing.kind !== 'acp' || !existing.activeAbort) {
      return {
        ok: false,
        rejectedConcurrent: true,
        error: 'No active ACP turn available for live steering.',
      }
    }
    if (
      existing.binding.provider === 'antigravity' &&
      !antigravitySteersWithPrompt(existing.binding.agent_version)
    ) {
      return {
        ok: false,
        rejectedConcurrent: true,
        error: `Antigravity ${existing.binding.agent_version ?? ''} queues follow-ups instead of mid-turn prompt steering.`,
      }
    }
    const promptBlocks = buildAcpPromptContentBlocks({
      agentName: existing.binding.name,
      capabilities: existing.promptCapabilities,
      provider: existing.binding.provider,
      text: request.text,
      ...(request.audio?.length ? { audio: request.audio } : {}),
      ...(request.images?.length ? { images: request.images } : {}),
      ...(request.resources?.length ? { resources: request.resources } : {}),
    })
    const combinedSignal = request.signal
      ? AbortSignal.any([request.signal, existing.activeAbort.signal])
      : existing.activeAbort.signal

    // Launch concurrent prompt without registering a duplicate onEvent listener:
    // the active turn's `promptListeners` subscription already receives every
    // `session/update` frame for `existing.externalSessionId`!
    const steerRpc = existing.client.prompt({
      sessionId: existing.externalSessionId,
      text: request.text,
      content: promptBlocks,
      signal: combinedSignal,
    })

    const probeMs = Math.max(0, request.acceptProbeMs ?? 120)
    let probeTimer: ReturnType<typeof setTimeout> | undefined
    type ProbeOutcome =
      | { readonly kind: 'settled'; readonly stopReason: string }
      | { readonly kind: 'error'; readonly error: unknown }
      | { readonly kind: 'accepted_inflight' }

    const probeResult = await Promise.race<ProbeOutcome>([
      steerRpc.then(
        res => ({ kind: 'settled', stopReason: res.stopReason }),
        error => ({ kind: 'error', error }),
      ),
      new Promise<ProbeOutcome>(resolveProbe => {
        probeTimer = setTimeout(() => resolveProbe({ kind: 'accepted_inflight' }), probeMs)
        probeTimer.unref?.()
      }),
    ])
    if (probeTimer !== undefined) {
      clearTimeout(probeTimer)
    }

    if (probeResult.kind === 'settled') {
      return { ok: true, settled: true, stopReason: probeResult.stopReason }
    }
    if (probeResult.kind === 'error') {
      return {
        ok: false,
        rejectedConcurrent: isConcurrentPromptRejection(probeResult.error),
        error: errorMessage(probeResult.error),
      }
    }

    // Accepted and still streaming alongside the active turn: track so `runPrompt`
    // drains it before emitting `turn_end`, and invoke fallback if it rejects later.
    const tracked = steerRpc.then(
      () => undefined,
      async error => {
        if (!combinedSignal.aborted && isConcurrentPromptRejection(error) && request.onFallbackQueue) {
          await request.onFallbackQueue()
        }
      },
    )
    existing.liveSteerTasks.add(tracked)
    void tracked.finally(() => {
      existing.liveSteerTasks.delete(tracked)
    })
    return { ok: true, settled: false }
  }

  /**
   * Run an ephemeral, read-only side question (`/btw`) on a concurrent ACP
   * session (`session/new` -> `session/prompt` -> `session/close`) without
   * interrupting an active main turn on `existing.externalSessionId`.
   */
  async runSideQuestion(request: AcpSideQuestionRequest): Promise<AcpSideQuestionResult> {
    const existing = this.active.get(request.sessionId)
    if (!existing || !existing.binding.connected) {
      return {
        ok: false,
        answer: '',
        model: request.model?.trim() || 'default',
        error: `No external ACP agent is connected for session "${request.sessionId}".`,
      }
    }
    const desiredModel = request.model?.trim() || existing.binding.current_model?.trim() || ''
    const effectiveModel = desiredModel || existing.binding.name
    if (existing.kind !== 'acp') {
      return {
        ok: false,
        answer: '',
        model: effectiveModel,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support concurrent ACP side sessions.`,
      }
    }

    let sideSessionId = ''
    let answer = ''
    let lastError: string | undefined
    const toolNamesById = new Map<string, string>()
    const toolMetaById = new Map<string, AcpToolMetaSnapshot>()
    const settledToolCallIds = new Set<string>()
    const toolTargetsById = new Map<string, { readonly name: string; readonly target: string }>()

    try {
      let usedFork = false
      const created =
        request.useFork !== false && existing.sessionCapabilities.fork
          ? await existing.client
              .forkSession({
                sessionId: existing.externalSessionId,
                cwd: existing.binding.cwd,
                mcpServers: existing.mcpServers,
                ...(existing.sessionCapabilities.additionalDirectories && existing.additionalDirectories.length > 0
                  ? { additionalDirectories: existing.additionalDirectories }
                  : {}),
              })
              .then(res => {
                usedFork = true
                return res
              })
              .catch(() => existing.client.newSession({ cwd: existing.binding.cwd }))
          : await existing.client.newSession({ cwd: existing.binding.cwd })
      sideSessionId = created.sessionId
      existing.sideSessionIds.add(sideSessionId)

      if (desiredModel) {
        const setRes = await existing.client
          .setModel({
            sessionId: sideSessionId,
            modelId: desiredModel,
            ...(existing.binding.model_config_id ? { configId: existing.binding.model_config_id } : {}),
          })
          .catch(() => undefined)
        void setRes
      }

      const promptText = usedFork && request.forkedPrompt ? request.forkedPrompt : request.prompt
      const result = await existing.client.prompt({
        sessionId: sideSessionId,
        text: promptText,
        ...(request.signal ? { signal: request.signal } : {}),
        onEvent: acpEvent => {
          const bridged = bridgeEventFromAcpClientEvent(
            acpEvent,
            toolNamesById,
            toolMetaById,
            settledToolCallIds,
          )
          if (!bridged) return
          if (bridged.type === 'text_delta' && bridged.text) {
            answer += bridged.text
            request.onDelta?.(bridged.text)
          } else if (bridged.type === 'tool_call') {
            const rawInput = isAcpRecord(bridged.rawInput) ? bridged.rawInput : {}
            const locationPath = bridged.locations?.[0]?.path
            const target = acpSideQuestionToolTarget(
              bridged.name,
              locationPath ? { path: locationPath, ...rawInput } : rawInput,
              existing.binding.cwd,
            )
            toolTargetsById.set(bridged.toolCallId, { name: bridged.name, target })
            request.onTool?.({ name: bridged.name, target, status: 'running' })
          } else if (bridged.type === 'tool_result') {
            const meta = toolTargetsById.get(bridged.toolCallId)
            request.onTool?.({
              name: meta?.name ?? bridged.name,
              target: meta?.target ?? bridged.name,
              status: bridged.permissionDenied || bridged.isError ? 'error' : 'done',
            })
          } else if (bridged.type === 'error') {
            lastError = bridged.message
          }
        },
      })

      const trimmedAnswer = answer.trim()
      if (lastError && !trimmedAnswer) {
        return { ok: false, answer: '', model: effectiveModel, error: lastError }
      }
      if (result.stopReason === 'cancelled' && !trimmedAnswer) {
        return { ok: false, answer: '', model: effectiveModel, error: 'Side question was cancelled.' }
      }
      return { ok: true, answer: trimmedAnswer, model: effectiveModel }
    } catch (error) {
      return {
        ok: false,
        answer: answer.trim(),
        model: effectiveModel,
        error: errorMessage(error),
      }
    } finally {
      if (sideSessionId) {
        await existing.client.closeSession(sideSessionId, SESSION_CLOSE_TIMEOUT_MS).catch(() => undefined)
        existing.sideSessionIds.delete(sideSessionId)
      }
    }
  }

  async runPrompt(request: AcpRunPromptRequest): Promise<AcpRunPromptResult> {
    const existing = this.active.get(request.sessionId)
    if (!existing || !existing.binding.connected) {
      throw new Error(`No external ACP agent is connected for session "${request.sessionId}".`)
    }
    if (existing.activeAbort) {
      throw new Error('An external ACP turn is already in progress for this session.')
    }

    const turnAbort = new AbortController()
    existing.activeAbort = turnAbort
    const combinedSignal = request.signal
      ? AbortSignal.any([request.signal, turnAbort.signal])
      : turnAbort.signal

    let assistantText = ''
    let thinkingText = ''
    let lastError: string | undefined

    const handleBridgeEvent = (event: AcpBridgeStreamEvent): void => {
      if (event.type === 'text_delta') {
        assistantText += event.text
      } else if (event.type === 'thought_delta') {
        thinkingText += event.text
      } else if (event.type === 'error') {
        lastError = event.message
      } else if (
        event.type === 'session_init' ||
        event.type === 'model_update' ||
        event.type === 'mode_update' ||
        event.type === 'config_update' ||
        event.type === 'available_commands'
      ) {
        applyLiveBindingEvent(existing, event, this.options)
      }
      request.onEvent(event)
    }

    try {
      if (existing.kind === 'acp') {
        const elicitAbort = new AbortController()
        const elicitSignal = AbortSignal.any([combinedSignal, elicitAbort.signal])
        existing.activeElicitationAbort = elicitAbort
        existing.activePermissionHandler = request.requestPermission
        existing.activeElicitationHandler = request.createElicitation
          ? elicitReq => request.createElicitation!(elicitReq, elicitSignal)
          : undefined
        const settledToolCallIds = new Set<string>()
        existing.outOfTurnSettledToolCallIds.clear()
        existing.readRepair?.reset()
        const dispatchTurnEvent = (acpEvent: AcpClientEvent): void => {
          const transformed = transformAntigravityClientEvent(existing, acpEvent)
          if (!transformed) return
          const bridged = bridgeEventFromAcpClientEvent(
            transformed,
            existing.toolNamesById,
            existing.toolMetaById,
            settledToolCallIds,
          )
          if (existing.toolNamesById.size > MAX_ACP_TOOL_NAMES) {
            trimToolNamesMap(existing.toolNamesById, MAX_ACP_TOOL_NAMES)
          }
          if (existing.toolMetaById.size > MAX_ACP_TOOL_NAMES) {
            trimToolNamesMap(existing.toolMetaById, MAX_ACP_TOOL_NAMES)
          }
          if (bridged) {
            handleBridgeEvent(bridged)
          }
        }
        existing.activeSyntheticUpdate = dispatchTurnEvent
        const promptBlocks = buildAcpPromptContentBlocks({
          agentName: existing.binding.name,
          capabilities: existing.promptCapabilities,
          provider: existing.binding.provider,
          text: request.text,
          ...(request.audio?.length ? { audio: request.audio } : {}),
          ...(request.fallbackInlineText ? { fallbackInlineText: request.fallbackInlineText } : {}),
          ...(request.images?.length ? { images: request.images } : {}),
          ...(request.resources?.length ? { resources: request.resources } : {}),
        })
        try {
          const result = await existing.client.prompt({
            sessionId: existing.externalSessionId,
            text: request.text,
            content: promptBlocks,
            signal: combinedSignal,
            onEvent: dispatchTurnEvent,
          })
          if (!combinedSignal.aborted && existing.liveSteerTasks.size > 0) {
            await Promise.allSettled([...existing.liveSteerTasks])
          }
          if (result.usage) {
            handleBridgeEvent({
              type: 'usage',
              ...(result.usage.inputTokens !== undefined ? { inputTokens: result.usage.inputTokens } : {}),
              ...(result.usage.outputTokens !== undefined ? { outputTokens: result.usage.outputTokens } : {}),
              ...(result.usage.cachedReadTokens !== undefined ? { cachedReadTokens: result.usage.cachedReadTokens } : {}),
              ...(result.usage.cachedWriteTokens !== undefined ? { cachedWriteTokens: result.usage.cachedWriteTokens } : {}),
              ...(result.usage.thoughtTokens !== undefined ? { thoughtTokens: result.usage.thoughtTokens } : {}),
              ...(result.usage.totalTokens !== undefined ? { totalTokens: result.usage.totalTokens } : {}),
              ...(result.usage.cost ? { cost: result.usage.cost } : {}),
              ...(result.usage.costUsd !== undefined ? { costUsd: result.usage.costUsd } : {}),
            })
          }
          return {
            assistantText,
            thinkingText,
            externalSessionId: existing.externalSessionId,
            stopReason: result.stopReason,
            cancelled: combinedSignal.aborted || result.stopReason === 'cancelled',
            ...(result.usage ? { usage: result.usage } : {}),
            ...(lastError ? { error: lastError } : {}),
          }
        } catch (error) {
          if (combinedSignal.aborted) {
            return {
              assistantText,
              thinkingText,
              externalSessionId: existing.externalSessionId,
              stopReason: 'cancelled',
              cancelled: true,
            }
          }
          const message = errorMessage(error)
          request.onEvent({ type: 'error', message })
          return {
            assistantText,
            thinkingText,
            externalSessionId: existing.externalSessionId,
            stopReason: 'error',
            cancelled: false,
            error: message,
          }
        } finally {
          existing.activePermissionHandler = undefined
          existing.activeSyntheticUpdate = undefined
          existing.activeElicitationAbort?.abort(new Error('External ACP turn finished'))
          existing.activeElicitationAbort = undefined
          existing.activeElicitationHandler = undefined
          trimToolNamesMap(existing.toolNamesById, MAX_ACP_TOOL_NAMES)
          trimToolNamesMap(existing.toolMetaById, MAX_ACP_TOOL_NAMES)
        }
      }

      // Claude Code stream-json runner
      try {
        const result = await existing.runner.prompt({
          text: request.fallbackInlineText ?? request.text,
          signal: combinedSignal,
          ...(request.permissionMode ? { permissionMode: request.permissionMode } : {}),
          onEvent: handleBridgeEvent,
        })
        if (result.externalSessionId) {
          existing.binding = {
            ...existing.binding,
            external_session_id: result.externalSessionId,
          }
        }
        return {
          assistantText,
          thinkingText,
          ...(result.externalSessionId ? { externalSessionId: result.externalSessionId } : {}),
          stopReason: result.stopReason,
          cancelled: result.cancelled,
          ...(lastError ? { error: lastError } : {}),
        }
      } catch (error) {
        if (combinedSignal.aborted) {
          return {
            assistantText,
            thinkingText,
            ...(existing.runner.sessionId ? { externalSessionId: existing.runner.sessionId } : {}),
            stopReason: 'cancelled',
            cancelled: true,
          }
        }
        const message = errorMessage(error)
        return {
          assistantText,
          thinkingText,
          ...(existing.runner.sessionId ? { externalSessionId: existing.runner.sessionId } : {}),
          stopReason: 'error',
          cancelled: false,
          error: message,
        }
      }
    } finally {
      if (existing.activeAbort === turnAbort) {
        existing.activeAbort = undefined
      }
    }
  }

  async disposeAll(): Promise<void> {
    const ids = [...this.active.keys()]
    await Promise.allSettled(ids.map(id => this.disconnect(id)))
  }

  private resolveCommandBinary(command: string): string | undefined {
    if (this.options.resolveBinary) {
      return this.options.resolveBinary(command)
    }
    const env = this.options.env ?? process.env
    const home = this.options.homeDir ?? (env.HOME?.trim() || homedir())
    return resolveExecutablePath(command, {
      ...(this.options.env ? { env: this.options.env } : {}),
      extraDirs: companionCliSearchDirectories(env, home),
    })
  }

  private async connectStdioAcp(
    params: {
      readonly additionalDirectories?: readonly string[]
      readonly agent?: string
      readonly agentId: string
      readonly args: readonly string[]
      readonly command: string
      readonly cwd: string
      readonly ideMcpPort?: number
      readonly initialModeId?: string
      readonly installHint?: string
      readonly mcpServers?: readonly AcpConnectMcpServerInput[]
      readonly name: string
      readonly provider: AcpExternalAgentProvider
      readonly resumeSessionId?: string
      readonly sandbox?: boolean
      readonly sessionId: string
    },
  ): Promise<AcpConnectResult> {
    await this.disconnect(params.sessionId)

    let entryRef: ActiveAcpSession | undefined
    let initResult: AcpInitializeResult | undefined
    let initialAvailableCommands: readonly AcpAvailableCommand[] | undefined
    let initialUsageEvent: AcpClientEvent | undefined
    let initialSessionInfoEvent: AcpClientEvent | undefined
    const initialReplayEvents: AcpClientEvent[] = []

    const normalizedAdditionalDirs = normalizeAdditionalDirectories(params.cwd, params.additionalDirectories)
    const isAntigravity = params.provider === 'antigravity' || params.agentId === 'antigravity'
    const launchArgs = isAntigravity
      ? buildAgyLaunchArgs(params.command, params.args, {
          additionalDirectories: normalizedAdditionalDirs,
          ...(params.agent ? { agent: params.agent } : {}),
          ...(params.ideMcpPort !== undefined ? { ideMcpPort: params.ideMcpPort } : {}),
          ...(params.mcpServers?.length ? { mcpServers: params.mcpServers } : {}),
          ...(params.sandbox ? { sandbox: true } : {}),
        })
      : params.args

    const transferredAgyEnv = isAntigravity
      ? await loadAgyRuntimeEnv(this.options.homeDir ?? homedir())
      : {}
    const mergedEnv = isAntigravity
      ? buildAgySubprocessEnv(
          this.options.env
            ? { ...transferredAgyEnv, ...this.options.env }
            : { ...process.env, ...transferredAgyEnv },
        )
      : this.options.env

    const readRepair = isAntigravity ? makeAntigravityReadRepair() : undefined
    const updateStepsPlans = isAntigravity ? makeAntigravityUpdateStepsPlans() : undefined

    const clientOptions: AcpStdioClientOptions = {
      command: params.command,
      args: launchArgs,
      cwd: params.cwd,
      ...(mergedEnv ? { env: mergedEnv } : {}),
      handlers: {
        createElicitation: async elicitReq => {
          if (entryRef && elicitReq.sessionId && elicitReq.sessionId !== entryRef.externalSessionId) {
            return { action: 'cancel' }
          }
          const activeHandler = entryRef?.activeElicitationHandler
          if (activeHandler) {
            return activeHandler(elicitReq)
          }
          if (entryRef && this.defaultCreateElicitation) {
            return this.defaultCreateElicitation(
              params.sessionId,
              elicitReq,
              entryRef.outOfTurnInteractionAbort.signal,
            )
          }
          return { action: 'cancel' }
        },
        requestPermission: async permReq => {
          if (entryRef && permReq.sessionId && permReq.sessionId !== entryRef.externalSessionId) {
            if (isReadOnlySideQuestionPermission(permReq)) {
              const allowOption =
                permReq.options.find(opt => opt.kind === 'allow_once') ??
                permReq.options.find(opt => opt.kind === 'allow_always') ??
                permReq.options[0]
              if (allowOption) {
                return { outcome: 'selected', optionId: allowOption.optionId }
              }
            }
            const rejectOption =
              permReq.options.find(opt => opt.kind === 'reject_once') ??
              permReq.options.find(opt => opt.kind === 'reject_always')
            return rejectOption
              ? { outcome: 'selected', optionId: rejectOption.optionId }
              : { outcome: 'cancelled' }
          }
          const activeHandler = entryRef?.activePermissionHandler
          if (activeHandler) {
            return activeHandler(permReq)
          }
          const isUserInputQuestion = isAntigravity && isAntigravityUserInputRequest(permReq)
          if (
            !isUserInputQuestion &&
            entryRef &&
            (entryRef.hostTurboActive || isAcpTurboMode(entryRef.binding))
          ) {
            const allowOption =
              permReq.options.find(opt => opt.kind === 'allow_once') ??
              permReq.options.find(opt => opt.kind === 'allow_always') ??
              permReq.options[0]
            if (allowOption) {
              return { outcome: 'selected', optionId: allowOption.optionId }
            }
          }
          if (entryRef && this.defaultRequestPermission) {
            return this.defaultRequestPermission(
              params.sessionId,
              permReq,
              entryRef.outOfTurnInteractionAbort.signal,
            )
          }
          return { outcome: 'cancelled' }
        },
        readTextFile: async fileReq => {
          try {
            const result = await readWorkspaceScopedTextFile(
              params.cwd,
              fileReq,
              normalizedAdditionalDirs,
            )
            if (readRepair) {
              const completed = readRepair.attempted(fileReq.path, true)
              if (completed && entryRef) {
                const syntheticUpdate: AcpClientEvent = {
                  type: 'tool_call_update',
                  sessionId: completed.sessionId,
                  toolCallId: completed.toolCallId,
                  status: 'completed',
                }
                if (entryRef.activeSyntheticUpdate) {
                  entryRef.activeSyntheticUpdate(syntheticUpdate)
                } else if (!entryRef.activeAbort) {
                  const bridged = bridgeEventFromAcpClientEvent(
                    syntheticUpdate,
                    entryRef.toolNamesById,
                    entryRef.toolMetaById,
                    entryRef.outOfTurnSettledToolCallIds,
                  )
                  if (bridged) {
                    this.notifySessionEvent(params.sessionId, bridged)
                  }
                }
              }
            }
            return result
          } catch (error) {
            readRepair?.attempted(fileReq.path, false)
            throw error
          }
        },
        // Advertising `fs.writeTextFile` to `agy acp` swaps its native diff edits for whole-file
        // `client_edit_file` (matching T3 Code's AntigravityAcpSupport). Omit it for Antigravity.
        ...(!isAntigravity
          ? {
              writeTextFile: async (fileReq: {
                readonly content: string
                readonly path: string
                readonly sessionId: string
              }) => {
                if (entryRef && fileReq.sessionId && fileReq.sessionId !== entryRef.externalSessionId) {
                  throw new Error('File writes are not permitted in a read-only side question.')
                }
                const safePath = await resolveWorkspaceScopedPathReal(
                  params.cwd,
                  fileReq.path,
                  normalizedAdditionalDirs,
                )
                await writeFile(safePath, fileReq.content, 'utf8')
              },
            }
          : {}),
        ...(this.options.terminalHandlers ? this.options.terminalHandlers : {}),
      },
    }

    const client = this.options.createAcpClient
      ? this.options.createAcpClient(clientOptions)
      : new AcpStdioClient(clientOptions)

    // Subscribe BEFORE initialize/session startup so initial notifications
    // (`available_commands_update`, `usage_update`, `session_info_update`,
    // `session/load` replay chunks) and unexpected subprocess exits are never missed.
    client.onEvent(acpEvent => {
      if (!entryRef) {
        if (acpEvent.type === 'available_commands') {
          initialAvailableCommands = acpEvent.commands
        } else if (acpEvent.type === 'usage_update') {
          initialUsageEvent = acpEvent
        } else if (acpEvent.type === 'session_info') {
          initialSessionInfoEvent = acpEvent
        } else if (
          acpEvent.type === 'user_message_chunk' ||
          acpEvent.type === 'text_delta' ||
          acpEvent.type === 'thought_delta' ||
          acpEvent.type === 'tool_call' ||
          acpEvent.type === 'tool_call_update' ||
          acpEvent.type === 'plan'
        ) {
          initialReplayEvents.push(acpEvent)
        }
        return
      }

      if (acpEvent.type === 'exit') {
        if (client.isDisposed || this.active.get(params.sessionId) !== entryRef) {
          return
        }
        this.active.delete(params.sessionId)
        entryRef.activeAbort = undefined
        entryRef.activePermissionHandler = undefined
        entryRef.activeElicitationAbort?.abort(new Error('External ACP process exited unexpectedly'))
        entryRef.activeElicitationAbort = undefined
        entryRef.activeElicitationHandler = undefined
        entryRef.outOfTurnInteractionAbort.abort(new Error('External ACP process exited unexpectedly'))
        const codeLabel =
          acpEvent.code !== null
            ? `code ${acpEvent.code}`
            : acpEvent.signal
              ? `signal ${acpEvent.signal}`
              : 'unknown exit'
        const stderrDetail = client.recentStderr ? `: ${client.recentStderr}` : ''
        entryRef.binding = {
          ...entryRef.binding,
          connected: false,
          error: `${entryRef.binding.name} ACP process exited unexpectedly (${codeLabel})${stderrDetail}`,
        }
        this.notifyBindingChange(params.sessionId, entryRef.binding)
        return
      }

      if (acpEvent.type === 'error') {
        if (client.isDisposed || this.active.get(params.sessionId) !== entryRef) {
          return
        }
        const stderrDetail = client.recentStderr ? `: ${client.recentStderr}` : ''
        if (client.hasExited) {
          this.active.delete(params.sessionId)
          entryRef.activeAbort = undefined
          entryRef.activePermissionHandler = undefined
          entryRef.activeElicitationAbort?.abort(new Error('External ACP process failed'))
          entryRef.activeElicitationAbort = undefined
          entryRef.activeElicitationHandler = undefined
          entryRef.outOfTurnInteractionAbort.abort(new Error('External ACP process failed'))
          entryRef.binding = {
            ...entryRef.binding,
            connected: false,
            error: `${entryRef.binding.name} ACP process failed: ${acpEvent.message}${stderrDetail}`,
          }
          this.notifyBindingChange(params.sessionId, entryRef.binding)
        } else {
          entryRef.binding = {
            ...entryRef.binding,
            error: acpEvent.message,
          }
          this.notifyBindingChange(params.sessionId, entryRef.binding)
        }
        return
      }

      if (acpEvent.type === 'elicitation_complete') {
        if (!acpEvent.sessionId || acpEvent.sessionId === entryRef.externalSessionId) {
          this.notifySessionEvent(params.sessionId, {
            type: 'elicitation_complete',
            elicitationId: acpEvent.elicitationId,
          })
        }
        return
      }

      if (
        'sessionId' in acpEvent &&
        acpEvent.sessionId &&
        acpEvent.sessionId !== entryRef.externalSessionId
      ) {
        // Ephemeral side-question sessions (`session/new` / `session/fork`) run concurrently on the
        // same ACP stdio client; never leak their updates into the main session.
        return
      }

      if (
        acpEvent.type === 'mode_update' ||
        acpEvent.type === 'model_update' ||
        acpEvent.type === 'config_update' ||
        acpEvent.type === 'available_commands'
      ) {
        const bridged = bridgeEventFromAcpClientEvent(acpEvent, entryRef.toolNamesById)
        if (!bridged) return
        if (applyLiveBindingEvent(entryRef, bridged, this.options)) {
          this.notifyBindingChange(params.sessionId, entryRef.binding)
        }
        return
      }

      if (
        !entryRef.activeAbort &&
        (acpEvent.type === 'usage_update' ||
          acpEvent.type === 'session_info' ||
          acpEvent.type === 'text_delta' ||
          acpEvent.type === 'thought_delta' ||
          acpEvent.type === 'tool_call' ||
          acpEvent.type === 'tool_call_update' ||
          acpEvent.type === 'plan' ||
          acpEvent.type === 'notice' ||
          acpEvent.type === 'continuation')
      ) {
        const transformed = transformAntigravityClientEvent(entryRef, acpEvent)
        if (!transformed) return
        const bridged = bridgeEventFromAcpClientEvent(
          transformed,
          entryRef.toolNamesById,
          entryRef.toolMetaById,
          entryRef.outOfTurnSettledToolCallIds,
        )
        if (entryRef.toolNamesById.size > MAX_ACP_TOOL_NAMES) {
          trimToolNamesMap(entryRef.toolNamesById, MAX_ACP_TOOL_NAMES)
        }
        if (entryRef.toolMetaById.size > MAX_ACP_TOOL_NAMES) {
          trimToolNamesMap(entryRef.toolMetaById, MAX_ACP_TOOL_NAMES)
        }
        if (entryRef.outOfTurnSettledToolCallIds.size > MAX_ACP_TOOL_NAMES) {
          trimStringSet(entryRef.outOfTurnSettledToolCallIds, MAX_ACP_TOOL_NAMES)
        }
        if (bridged) {
          this.notifySessionEvent(params.sessionId, bridged)
        }
      }
    })

    try {
      initResult = await client.initialize()
      const wireMcpServers = buildAcpWireMcpServers(params.mcpServers, initResult.mcpCapabilities)
      const sessionAdditionalDirs =
        initResult.sessionCapabilities.additionalDirectories && normalizedAdditionalDirs.length > 0
          ? normalizedAdditionalDirs
          : undefined
      const openSession = () =>
        openOrResumeAcpSession(
          client,
          initResult!,
          params.cwd,
          params.resumeSessionId,
          wireMcpServers,
          sessionAdditionalDirs,
        )
      let opened: Awaited<ReturnType<typeof openOrResumeAcpSession>>
      try {
        opened = await openSession()
      } catch (openError) {
        const nonInteractiveAuthMethod =
          isAntigravity && isAcpAuthRequiredError(openError)
            ? selectNonInteractiveAgyAuthMethod(mergedEnv)
            : undefined
        if (!nonInteractiveAuthMethod) {
          throw openError
        }
        await client.authenticate(nonInteractiveAuthMethod)
        opened = await openSession()
      }
      const sessionResult = opened.result
      const commandDisplay = launchArgs.length > 0
        ? `${params.command} ${launchArgs.join(' ')}`
        : params.command
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: sessionResult.configOptions ?? sessionResult.raw.configOptions ?? sessionResult.raw.config_options,
        models: sessionResult.models,
        modes: sessionResult.modes,
        raw: sessionResult.raw,
      })
      const capabilities = buildAcpRuntimeCapabilities(initResult)
      const agentTitle = initResult.agentInfo?.title?.trim() || initResult.agentInfo?.name?.trim() || undefined
      const agentVersion = initResult.agentInfo?.version?.trim() || undefined
      const classified = initialAvailableCommands && initialAvailableCommands.length > 0
        ? classifyAcpRuntimeFeatures({
            provider: params.provider,
            agentName: params.name,
            cwd: params.cwd,
            rawCommands: initialAvailableCommands,
            ...(this.options.env ? { env: this.options.env } : {}),
            ...(this.options.homeDir ? { homeDir: this.options.homeDir } : {}),
          })
        : undefined
      const rawBinding: AcpSessionBinding = {
        session_id: params.sessionId,
        agent_id: params.agentId,
        name: params.name,
        provider: params.provider,
        transport: 'acp',
        cwd: params.cwd,
        external_session_id: sessionResult.sessionId,
        connected: true,
        command: commandDisplay,
        ...(agentVersion ? { agent_version: agentVersion } : {}),
        ...(agentTitle ? { agent_title: agentTitle } : {}),
        capabilities,
        ...extracted,
        ...(classified && classified.available_commands.length > 0
          ? { available_commands: classified.available_commands }
          : {}),
        ...(classified && classified.discovered_skills.length > 0
          ? { discovered_skills: classified.discovered_skills }
          : {}),
        ...(normalizedAdditionalDirs.length > 0
          ? { additional_directories: normalizedAdditionalDirs }
          : {}),
        ...(params.sandbox ? { sandbox: true } : {}),
        ...(params.agent?.trim() ? { agent_override: params.agent.trim() } : {}),
        ...(wireMcpServers.length > 0
          ? {
              mcp_server_names: wireMcpServers
                .map(item => (typeof item.name === 'string' ? item.name.trim() : ''))
                .filter(Boolean),
            }
          : {}),
      }
      const entry: ActiveAcpSession = {
        kind: 'acp',
        client,
        externalSessionId: sessionResult.sessionId,
        additionalDirectories: normalizedAdditionalDirs,
        mcpServers: wireMcpServers,
        binding: rawBinding,
        hostTurboActive: false,
        hostTurboUnderlyingMode: undefined,
        liveSteerTasks: new Set<Promise<void>>(),
        nativeModes: extracted.modes ?? [],
        outOfTurnInteractionAbort: new AbortController(),
        outOfTurnSettledToolCallIds: new Set<string>(),
        activeAbort: undefined,
        activeElicitationAbort: undefined,
        activeElicitationHandler: undefined,
        activePermissionHandler: undefined,
        promptCapabilities: initResult.promptCapabilities,
        ...(readRepair ? { readRepair } : {}),
        sessionCapabilities: initResult.sessionCapabilities,
        sideSessionIds: new Set<string>(),
        toolMetaById: new Map<string, AcpToolMetaSnapshot>(),
        toolNamesById: new Map<string, string>(),
        ...(updateStepsPlans ? { updateStepsPlans } : {}),
      }
      entry.binding = projectAcpBindingWithHostTurbo(entry, rawBinding)
      entryRef = entry
      this.active.set(params.sessionId, entry)
      this.lastAuthChallengeBySession.delete(params.sessionId)

      const desiredMode =
        params.initialModeId?.trim() ||
        (isAntigravity && entry.binding.modes?.some(m => m.id === ACP_TURBO_MODE_ID)
          ? ACP_TURBO_MODE_ID
          : undefined)
      if (
        desiredMode &&
        desiredMode.toLowerCase() !== (entry.binding.current_mode ?? '').trim().toLowerCase()
      ) {
        // Default Antigravity (`agy acp`) to Turbo on fresh connect, or restore a
        // persisted/explicit mode (e.g. `auto`, `turbo`, `accept-edits`, `plan`) after
        // `session/resume` / reconnect because agents like `agy acp` reset mode to `default`.
        await this.setMode(params.sessionId, desiredMode).catch(() => undefined)
      }

      if (opened.via === 'load' && initialReplayEvents.length > 0) {
        const replayToolNames = new Map<string, string>()
        const replayToolMeta = new Map<string, AcpToolMetaSnapshot>()
        const replaySettled = new Set<string>()
        const bridgedReplay: AcpBridgeStreamEvent[] = []
        for (const replayEvent of initialReplayEvents) {
          if (
            'sessionId' in replayEvent &&
            replayEvent.sessionId &&
            replayEvent.sessionId !== entry.externalSessionId
          ) {
            continue
          }
          const bridged = bridgeEventFromAcpClientEvent(
            replayEvent,
            replayToolNames,
            replayToolMeta,
            replaySettled,
          )
          if (bridged) {
            bridgedReplay.push(bridged)
          }
        }
        if (bridgedReplay.length > 0) {
          this.notifySessionEvent(params.sessionId, {
            type: 'session_replay',
            events: bridgedReplay,
          })
        }
      }
      if (initialUsageEvent) {
        const bridged = bridgeEventFromAcpClientEvent(initialUsageEvent, entry.toolNamesById)
        if (bridged) {
          this.notifySessionEvent(params.sessionId, bridged)
        }
      }
      if (initialSessionInfoEvent) {
        const bridged = bridgeEventFromAcpClientEvent(initialSessionInfoEvent, entry.toolNamesById)
        if (bridged) {
          this.notifySessionEvent(params.sessionId, bridged)
        }
      }
      return { ok: true, binding: entry.binding }
    } catch (error) {
      await client.dispose().catch(() => undefined)
      if (isAcpAuthRequiredError(error)) {
        const authMethods = mapAuthMethodsToWire(initResult?.authMethods ?? [])
        this.lastAuthChallengeBySession.set(params.sessionId, {
          agentId: params.agentId,
          command: params.command,
          args: launchArgs,
          cwd: params.cwd,
          provider: params.provider,
          authMethods,
        })
        if (isAntigravity) {
          return {
            ok: false,
            error: ANTIGRAVITY_AUTH_ERROR,
            auth_required: true,
            ...(authMethods.length > 0 ? { auth_methods: authMethods } : {}),
          }
        }
        const authNames = authMethods.map(method => method.name).filter(Boolean)
        const authDetail = authNames.length > 0 ? ` (${authNames.join(', ')})` : ''
        return {
          ok: false,
          error: `${params.name} requires authentication${authDetail}. Sign in via the ${params.name} CLI, then click Retry.`,
          auth_required: true,
          ...(authMethods.length > 0 ? { auth_methods: authMethods } : {}),
          ...(params.installHint ? { install_hint: params.installHint } : {}),
        }
      }
      return {
        ok: false,
        error: `Failed to connect to ${params.name} ACP agent: ${errorMessage(error)}`,
        ...(params.installHint ? { install_hint: params.installHint } : {}),
      }
    }
  }

  private notifyBindingChange(sessionId: string, binding: AcpSessionBinding): void {
    for (const listener of this.bindingListeners) {
      try {
        listener(sessionId, binding)
      } catch {
        // Keep ACP stream healthy if an observer callback throws.
      }
    }
  }

  private notifySessionEvent(sessionId: string, event: AcpBridgeStreamEvent): void {
    for (const listener of this.sessionEventListeners) {
      try {
        listener(sessionId, event)
      } catch {
        // Keep ACP stream healthy if an observer callback throws.
      }
    }
  }
}

/**
 * Extract model and mode options, active selections, and ACP config option IDs
 * from a `session/new`, `session/load`, `session/set_config_option`, or
 * `config_option_update` payload.
 */
export function extractAcpSessionModelAndModeState(input: {
  readonly configOptions?: unknown
  readonly models?: unknown
  readonly modes?: unknown
  readonly raw?: Readonly<Record<string, unknown>>
}): ExtractedAcpModelAndModeState {
  const raw = input.raw

  // 1. Parse legacy/explicit `models` field (`{ currentModelId, availableModels }` or array).
  const modelsField = input.models ?? raw?.models
  let modelsFromField: AcpExternalModelOption[] = []
  let currentModelFromField: string | undefined
  let currentModelNameFromField: string | undefined
  if (Array.isArray(modelsField)) {
    modelsFromField = parseOptionList(modelsField, ['modelId', 'model_id', 'id', 'value'])
    currentModelFromField = raw
      ? readNonEmptyString(raw, 'currentModelId', 'current_model_id', 'modelId', 'model_id')
      : undefined
  } else if (isAcpRecord(modelsField)) {
    currentModelFromField = readNonEmptyString(
      modelsField,
      'currentModelId',
      'current_model_id',
      'current',
      'selectedModelId',
      'modelId',
      'model_id',
    )
    currentModelNameFromField = readNonEmptyString(modelsField, 'currentModelName', 'current_model_name')
    modelsFromField = parseOptionList(
      modelsField.availableModels ?? modelsField.available_models ?? modelsField.models ?? modelsField.options,
      ['modelId', 'model_id', 'id', 'value'],
    )
  } else if (raw) {
    currentModelFromField = readNonEmptyString(raw, 'currentModelId', 'current_model_id')
  }

  // 2. Parse legacy/explicit `modes` field (`{ currentModeId, availableModes }` or array).
  const modesField = input.modes ?? raw?.modes
  let modesFromField: AcpExternalModeOption[] = []
  let currentModeFromField: string | undefined
  let currentModeNameFromField: string | undefined
  if (Array.isArray(modesField)) {
    modesFromField = parseOptionList(modesField, ['id', 'modeId', 'mode_id', 'value'])
    currentModeFromField = raw
      ? readNonEmptyString(raw, 'currentModeId', 'current_mode_id', 'modeId', 'mode_id')
      : undefined
  } else if (isAcpRecord(modesField)) {
    currentModeFromField = readNonEmptyString(
      modesField,
      'currentModeId',
      'current_mode_id',
      'current',
      'selectedModeId',
      'modeId',
      'mode_id',
    )
    currentModeNameFromField = readNonEmptyString(modesField, 'currentModeName', 'current_mode_name')
    modesFromField = parseOptionList(
      modesField.availableModes ?? modesField.available_modes ?? modesField.modes ?? modesField.options,
      ['id', 'modeId', 'mode_id', 'value'],
    )
  } else if (raw) {
    currentModeFromField = readNonEmptyString(raw, 'currentModeId', 'current_mode_id')
  }

  // 3. Parse modern ACP `configOptions` array (flat or grouped select options).
  const rawConfigOptions = input.configOptions ?? raw?.configOptions ?? raw?.config_options
  let parsedConfigOptions: readonly AcpExternalConfigOption[] | undefined
  let modelsFromConfig: AcpExternalModelOption[] = []
  let currentModelFromConfig: string | undefined
  let modelConfigId: string | undefined
  let modesFromConfig: AcpExternalModeOption[] = []
  let currentModeFromConfig: string | undefined
  let modeConfigId: string | undefined
  let thoughtLevelsFromConfig: AcpExternalModeOption[] = []
  let thoughtLevelFromConfig: string | undefined
  let thoughtLevelConfigId: string | undefined

  if (Array.isArray(rawConfigOptions)) {
    parsedConfigOptions = parseConfigOptions(rawConfigOptions)
    const modelEntry = findConfigOptionEntry(rawConfigOptions, 'model')
    if (modelEntry) {
      modelConfigId = readNonEmptyString(modelEntry, 'id') ?? 'model'
      currentModelFromConfig = readNonEmptyString(modelEntry, 'currentValue', 'current_value', 'value')
      modelsFromConfig = parseOptionList(modelEntry.options, ['value', 'id', 'modelId', 'model_id'])
    }
    const modeEntry = findConfigOptionEntry(rawConfigOptions, 'mode')
    if (modeEntry) {
      modeConfigId = readNonEmptyString(modeEntry, 'id') ?? 'mode'
      currentModeFromConfig = readNonEmptyString(modeEntry, 'currentValue', 'current_value', 'value')
      modesFromConfig = parseOptionList(modeEntry.options, ['value', 'id', 'modeId', 'mode_id'])
    }
    const thoughtEntry = findConfigOptionEntry(rawConfigOptions, 'thought_level')
    if (thoughtEntry) {
      thoughtLevelConfigId = readNonEmptyString(thoughtEntry, 'id') ?? 'thought_level'
      thoughtLevelFromConfig = readNonEmptyString(thoughtEntry, 'currentValue', 'current_value', 'value')
      thoughtLevelsFromConfig = parseOptionList(thoughtEntry.options, ['value', 'id'])
    }
  }

  const models = modelsFromConfig.length > 0 ? modelsFromConfig : modelsFromField
  const currentModel = currentModelFromConfig ?? currentModelFromField
  const currentModelName = currentModel
    ? (models.find(item => item.id === currentModel)?.name ?? currentModelNameFromField ?? currentModel)
    : undefined

  const modes = modesFromConfig.length > 0 ? modesFromConfig : modesFromField
  const currentMode = currentModeFromConfig ?? currentModeFromField
  const currentModeName = currentMode
    ? (modes.find(item => item.id === currentMode)?.name ?? currentModeNameFromField ?? currentMode)
    : undefined

  return {
    ...(currentModel ? { current_model: currentModel } : {}),
    ...(currentModelName ? { current_model_name: currentModelName } : {}),
    ...(models.length > 0 ? { models } : {}),
    ...(currentMode ? { current_mode: currentMode } : {}),
    ...(currentModeName ? { current_mode_name: currentModeName } : {}),
    ...(modes.length > 0 ? { modes } : {}),
    ...(modelConfigId ? { model_config_id: modelConfigId } : {}),
    ...(modeConfigId ? { mode_config_id: modeConfigId } : {}),
    ...(parsedConfigOptions && parsedConfigOptions.length > 0 ? { config_options: parsedConfigOptions } : {}),
    ...(thoughtLevelFromConfig ? { thought_level: thoughtLevelFromConfig } : {}),
    ...(thoughtLevelsFromConfig.length > 0 ? { thought_levels: thoughtLevelsFromConfig } : {}),
    ...(thoughtLevelConfigId ? { thought_level_config_id: thoughtLevelConfigId } : {}),
  }
}

async function openOrResumeAcpSession(
  client: AcpStdioClient,
  initResult: AcpInitializeResult,
  cwd: string,
  resumeSessionId?: string,
  mcpServers: readonly Readonly<Record<string, unknown>>[] = [],
  additionalDirectories?: readonly string[],
): Promise<{
  readonly result: Awaited<ReturnType<AcpStdioClient['newSession']>>
  readonly via: 'load' | 'new' | 'resume'
}> {
  const sharedOpts = {
    cwd,
    mcpServers,
    ...(additionalDirectories && additionalDirectories.length > 0 ? { additionalDirectories } : {}),
  }
  const trimmedResume = resumeSessionId?.trim()
  if (trimmedResume) {
    if (initResult.sessionCapabilities.resume) {
      try {
        return {
          result: await client.resumeSession({ sessionId: trimmedResume, ...sharedOpts }),
          via: 'resume',
        }
      } catch (resumeError) {
        if (isAcpAuthRequiredError(resumeError)) {
          throw resumeError
        }
        if (initResult.sessionCapabilities.load || initResult.supportsLoadSession) {
          try {
            return {
              result: await client.loadSession({ sessionId: trimmedResume, ...sharedOpts }),
              via: 'load',
            }
          } catch (loadError) {
            if (isAcpAuthRequiredError(loadError)) {
              throw loadError
            }
          }
        }
      }
    } else if (initResult.sessionCapabilities.load || initResult.supportsLoadSession) {
      try {
        return {
          result: await client.loadSession({ sessionId: trimmedResume, ...sharedOpts }),
          via: 'load',
        }
      } catch (loadError) {
        if (isAcpAuthRequiredError(loadError)) {
          throw loadError
        }
      }
    }
  }
  return {
    result: await client.newSession(sharedOpts),
    via: 'new',
  }
}

export function buildAcpPromptContentBlocks(input: {
  readonly agentName: string
  readonly audio?: readonly AcpPromptAudioInput[]
  readonly capabilities: AcpPromptCapabilities
  readonly fallbackInlineText?: string
  readonly images?: readonly AcpPromptImageInput[]
  readonly provider?: AcpExternalAgentProvider
  readonly resources?: readonly AcpPromptResourceInput[]
  readonly text: string
}): readonly AcpPromptContentBlock[] {
  const blocks: AcpPromptContentBlock[] = []
  const hasResources = Boolean(input.resources && input.resources.length > 0)
  const isAntigravity = input.provider === 'antigravity'
  let textBody = hasResources && !input.capabilities.embeddedContext
    ? (input.fallbackInlineText || buildInlineResourcesFallback(input.text, input.resources!))
    : input.text

  if (!isAntigravity && input.images && input.images.length > 0 && !input.capabilities.image) {
    const count = input.images.length
    textBody = `${textBody}\n\n[Note: ${count} image attachment${count === 1 ? '' : 's'} omitted because ${input.agentName} does not advertise ACP image input support.]`
  }

  blocks.push({ type: 'text', text: textBody })

  if (hasResources) {
    for (const resource of input.resources!) {
      if (
        resource.type === 'resource_link' ||
        (isAntigravity && resource.mimeType?.split(';', 1)[0]?.trim().toLowerCase() === 'application/pdf')
      ) {
        blocks.push({
          type: 'resource_link',
          uri: resource.uri,
          name: resource.name || basename(resource.path || resource.uri) || resource.uri,
          ...(resource.mimeType ? { mimeType: resource.mimeType } : {}),
          ...(resource.title ? { title: resource.title } : {}),
          ...(resource.description ? { description: resource.description } : {}),
          ...(resource.size !== undefined ? { size: resource.size } : {}),
        })
        continue
      }
      if (!input.capabilities.embeddedContext) continue
      if (resource.blob !== undefined) {
        blocks.push({
          type: 'resource',
          resource: {
            uri: resource.uri,
            ...(resource.mimeType ? { mimeType: resource.mimeType } : {}),
            blob: resource.blob,
          },
        })
      } else {
        blocks.push({
          type: 'resource',
          resource: {
            uri: resource.uri,
            ...(resource.mimeType ? { mimeType: resource.mimeType } : { mimeType: 'text/plain' }),
            text: resource.text ?? '',
          },
        })
      }
    }
  }

  if (input.images && input.images.length > 0 && input.capabilities.image) {
    for (const image of input.images) {
      blocks.push({
        type: 'image',
        data: image.data,
        mimeType: image.mediaType,
      })
    }
  }

  if (input.audio && input.audio.length > 0 && input.capabilities.audio) {
    for (const clip of input.audio) {
      blocks.push({
        type: 'audio',
        data: clip.data,
        mimeType: clip.mediaType,
      })
    }
  }

  return blocks
}

function buildInlineResourcesFallback(
  text: string,
  resources: readonly AcpPromptResourceInput[],
): string {
  const parts = resources
    .filter(res => res.type !== 'resource_link' && res.text !== undefined)
    .map(res => {
      const label = res.path || res.uri
      return `<file path="${label}">\n${res.text ?? ''}\n</file>`
    })
  if (parts.length === 0) return text
  return `<attached_files>\n${parts.join('\n\n')}\n</attached_files>\n\n${text}`
}

function mapAuthMethodsToWire(
  methods: AcpInitializeResult['authMethods'],
): readonly AcpRuntimeAuthMethod[] {
  return methods.map(method => ({
    id: method.id,
    name: method.name,
    ...(method.description ? { description: method.description } : {}),
    ...(method.type ? { type: method.type } : {}),
    ...(method.args && method.args.length > 0 ? { args: method.args } : {}),
    ...(method.env ? { env: method.env } : {}),
    ...(method.meta ? { meta: method.meta } : {}),
  }))
}

function buildAcpRuntimeCapabilities(initResult: AcpInitializeResult): AcpRuntimeCapabilities {
  return {
    prompt: {
      image: initResult.promptCapabilities.image,
      audio: initResult.promptCapabilities.audio,
      embedded_context: initResult.promptCapabilities.embeddedContext,
    },
    session: {
      load: initResult.sessionCapabilities.load,
      resume: initResult.sessionCapabilities.resume,
      fork: initResult.sessionCapabilities.fork,
      list: initResult.sessionCapabilities.list,
      close: initResult.sessionCapabilities.close,
      additional_directories: initResult.sessionCapabilities.additionalDirectories,
    },
    mcp: {
      http: initResult.mcpCapabilities.http,
      sse: initResult.mcpCapabilities.sse,
    },
    auth_methods: mapAuthMethodsToWire(initResult.authMethods),
    ...(initResult.capabilitiesMeta ? { meta: initResult.capabilitiesMeta } : {}),
  }
}

export function parseConfigOptions(configOptions: readonly unknown[]): AcpExternalConfigOption[] {
  const parsed: AcpExternalConfigOption[] = []
  const seen = new Set<string>()
  for (const item of configOptions) {
    if (!isAcpRecord(item)) continue
    const id = readNonEmptyString(item, 'id')
    if (!id || seen.has(id)) continue
    seen.add(id)
    const name = readNonEmptyString(item, 'name', 'label', 'title') ?? id
    const description = readNonEmptyString(item, 'description')
    const category = readNonEmptyString(item, 'category')
    const inferredType =
      readNonEmptyString(item, 'type') ??
      (typeof item.currentValue === 'boolean' || typeof item.current_value === 'boolean' || typeof item.value === 'boolean'
        ? 'boolean'
        : 'select')
    const rawValue = readConfigOptionValue(item, inferredType, 'currentValue', 'current_value', 'value')
    const currentValue = rawValue !== undefined ? rawValue : (inferredType === 'boolean' ? false : '')
    const options = parseConfigOptionChoices(item.options, ['value', 'id', 'modelId', 'modeId'])
    parsed.push({
      id,
      name,
      ...(description ? { description } : {}),
      ...(category ? { category } : {}),
      type: inferredType,
      current_value: currentValue,
      options,
    })
  }
  return parsed
}

function readConfigOptionValue(
  record: Readonly<Record<string, unknown>>,
  optionType: string,
  ...keys: readonly string[]
): string | boolean | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'boolean') {
      return value
    }
    if (typeof value === 'string') {
      const trimmed = value.trim()
      if (trimmed.length > 0) {
        if (optionType === 'boolean') {
          const lower = trimmed.toLowerCase()
          if (lower === 'true') return true
          if (lower === 'false') return false
        }
        return trimmed
      }
    } else if (typeof value === 'number' && Number.isFinite(value)) {
      return String(value)
    }
  }
  return undefined
}

function parseConfigOptionChoices(
  rawOptions: unknown,
  idKeys: readonly string[],
): AcpExternalConfigOptionChoice[] {
  if (!Array.isArray(rawOptions)) {
    return []
  }
  const results: AcpExternalConfigOptionChoice[] = []
  const seen = new Set<string>()
  const visit = (items: readonly unknown[], groupLabel?: string): void => {
    for (const item of items) {
      if (typeof item === 'string') {
        const value = item.trim()
        if (value && !seen.has(value)) {
          seen.add(value)
          results.push({
            value,
            name: value,
            ...(groupLabel ? { group: groupLabel } : {}),
          })
        }
        continue
      }
      if (!isAcpRecord(item)) continue
      if (Array.isArray(item.options)) {
        const nextGroup =
          readNonEmptyString(item, 'name', 'group', 'label', 'title') ?? groupLabel
        visit(item.options, nextGroup)
        continue
      }
      const value = readNonEmptyString(item, ...idKeys)
      if (!value || seen.has(value)) continue
      seen.add(value)
      const name = readNonEmptyString(item, 'name', 'displayName', 'display_name', 'label', 'title') ?? value
      const description = readNonEmptyString(item, 'description')
      const explicitGroup = readNonEmptyString(item, 'group') ?? groupLabel
      results.push({
        value,
        name,
        ...(description ? { description } : {}),
        ...(explicitGroup ? { group: explicitGroup } : {}),
      })
    }
  }
  visit(rawOptions)
  return results
}

function findConfigOptionEntry(
  configOptions: readonly unknown[],
  kind: 'model' | 'mode' | 'thought_level',
): Record<string, unknown> | undefined {
  let byIdFallback: Record<string, unknown> | undefined
  const matchesKind = (value: string | undefined): boolean => {
    if (!value) return false
    const lower = value.toLowerCase()
    if (lower === kind) return true
    if (kind === 'thought_level' && (lower === 'thinking_level' || lower === 'reasoning_effort' || lower === 'thinking')) {
      return true
    }
    return false
  }
  for (const item of configOptions) {
    if (!isAcpRecord(item)) continue
    if (matchesKind(readNonEmptyString(item, 'category'))) {
      return item
    }
    if (!byIdFallback && matchesKind(readNonEmptyString(item, 'id'))) {
      byIdFallback = item
    }
  }
  return byIdFallback
}

function parseOptionList(
  rawOptions: unknown,
  idKeys: readonly string[],
): AcpExternalModelOption[] {
  if (!Array.isArray(rawOptions)) {
    return []
  }
  const results: AcpExternalModelOption[] = []
  const seen = new Set<string>()
  const visit = (items: readonly unknown[]): void => {
    for (const item of items) {
      if (typeof item === 'string') {
        const id = item.trim()
        if (id && !seen.has(id)) {
          seen.add(id)
          results.push({ id, name: id })
        }
        continue
      }
      if (!isAcpRecord(item)) continue
      if (Array.isArray(item.options)) {
        visit(item.options)
        continue
      }
      const id = readNonEmptyString(item, ...idKeys)
      if (!id || seen.has(id)) continue
      seen.add(id)
      const name = readNonEmptyString(item, 'name', 'displayName', 'display_name', 'label', 'title') ?? id
      const description = readNonEmptyString(item, 'description')
      results.push({
        id,
        name,
        ...(description ? { description } : {}),
      })
    }
  }
  visit(rawOptions)
  return results
}

function mergeBindingModelAndModeState(
  binding: AcpSessionBinding,
  patch: ExtractedAcpModelAndModeState,
): AcpSessionBinding {
  const models = patch.models && patch.models.length > 0 ? patch.models : binding.models
  const currentModel = patch.current_model ?? binding.current_model
  const currentModelName = currentModel
    ? (models?.find(item => item.id === currentModel)?.name
      ?? (patch.current_model !== undefined ? patch.current_model_name : binding.current_model_name)
      ?? currentModel)
    : undefined
  const modelConfigId = patch.model_config_id ?? binding.model_config_id

  const modes = patch.modes && patch.modes.length > 0 ? patch.modes : binding.modes
  const currentMode = patch.current_mode ?? binding.current_mode
  const currentModeName = currentMode
    ? (modes?.find(item => item.id === currentMode)?.name
      ?? (patch.current_mode !== undefined ? patch.current_mode_name : binding.current_mode_name)
      ?? currentMode)
    : undefined
  const modeConfigId = patch.mode_config_id ?? binding.mode_config_id

  const hasFullConfigOptions = patch.config_options !== undefined
  const configOptions = hasFullConfigOptions
    ? patch.config_options
    : updateConfigOptionsValues(binding.config_options, {
        ...(patch.current_model ? { [modelConfigId ?? 'model']: patch.current_model } : {}),
        ...(patch.current_mode ? { [modeConfigId ?? 'mode']: patch.current_mode } : {}),
        ...(patch.thought_level
          ? { [patch.thought_level_config_id ?? binding.thought_level_config_id ?? 'thought_level']: patch.thought_level }
          : {}),
      })

  const thoughtLevel = hasFullConfigOptions ? patch.thought_level : (patch.thought_level ?? binding.thought_level)
  const thoughtLevels = hasFullConfigOptions ? patch.thought_levels : (patch.thought_levels ?? binding.thought_levels)
  const thoughtLevelConfigId = hasFullConfigOptions
    ? patch.thought_level_config_id
    : (patch.thought_level_config_id ?? binding.thought_level_config_id)

  const {
    config_options: _prevConfigOptions,
    thought_level: _prevThoughtLevel,
    thought_levels: _prevThoughtLevels,
    thought_level_config_id: _prevThoughtLevelConfigId,
    ...restBinding
  } = binding
  void _prevConfigOptions
  void _prevThoughtLevel
  void _prevThoughtLevels
  void _prevThoughtLevelConfigId

  return {
    ...restBinding,
    ...(currentModel ? { current_model: currentModel } : {}),
    ...(currentModelName ? { current_model_name: currentModelName } : {}),
    ...(models && models.length > 0 ? { models } : {}),
    ...(currentMode ? { current_mode: currentMode } : {}),
    ...(currentModeName ? { current_mode_name: currentModeName } : {}),
    ...(modes && modes.length > 0 ? { modes } : {}),
    ...(modelConfigId ? { model_config_id: modelConfigId } : {}),
    ...(modeConfigId ? { mode_config_id: modeConfigId } : {}),
    ...(configOptions && configOptions.length > 0 ? { config_options: configOptions } : {}),
    ...(thoughtLevel ? { thought_level: thoughtLevel } : {}),
    ...(thoughtLevels && thoughtLevels.length > 0 ? { thought_levels: thoughtLevels } : {}),
    ...(thoughtLevelConfigId ? { thought_level_config_id: thoughtLevelConfigId } : {}),
  }
}

function updateConfigOptionsValues(
  existing: readonly AcpExternalConfigOption[] | undefined,
  updates: Readonly<Record<string, string | boolean>>,
): readonly AcpExternalConfigOption[] | undefined {
  if (!existing || existing.length === 0) return existing
  const keys = Object.keys(updates)
  if (keys.length === 0) return existing
  let changed = false
  const next = existing.map(option => {
    const direct = updates[option.id]
    const byCategory = option.category ? updates[option.category] : undefined
    const targetValue = direct !== undefined ? direct : byCategory
    if (targetValue === undefined || option.current_value === targetValue) {
      return option
    }
    changed = true
    return {
      ...option,
      current_value: targetValue,
    }
  })
  return changed ? next : existing
}

function applySingleConfigOptionPatch(
  binding: AcpSessionBinding,
  configId: string,
  value: string | boolean,
  extracted: ExtractedAcpModelAndModeState = {},
): AcpSessionBinding {
  const lowerId = configId.toLowerCase()
  const modelCfgId = binding.model_config_id?.toLowerCase()
  const modeCfgId = binding.mode_config_id?.toLowerCase()
  const thoughtCfgId = binding.thought_level_config_id?.toLowerCase()
  const existingOpt = binding.config_options?.find(
    item => item.id.toLowerCase() === lowerId,
  )
  const category = existingOpt?.category?.toLowerCase()

  let next = mergeBindingModelAndModeState(binding, extracted)
  if (typeof value === 'string') {
    if (lowerId === modelCfgId || lowerId === 'model' || category === 'model') {
      next = mergeBindingModelAndModeState(next, {
        current_model: value,
        model_config_id: configId,
      })
    } else if (lowerId === modeCfgId || lowerId === 'mode' || category === 'mode') {
      next = mergeBindingModelAndModeState(next, {
        current_mode: value,
        mode_config_id: configId,
      })
    } else if (
      lowerId === thoughtCfgId ||
      lowerId === 'thought_level' ||
      lowerId === 'thinking_level' ||
      category === 'thought_level'
    ) {
      next = mergeBindingModelAndModeState(next, {
        thought_level: value,
        thought_level_config_id: configId,
      })
    }
  }

  const updatedOptions = updateConfigOptionsValues(next.config_options, { [configId]: value })
  if (updatedOptions && updatedOptions !== next.config_options) {
    next = { ...next, config_options: updatedOptions }
  }
  return next
}

function applyLiveBindingEvent(
  entry: ActiveExternalSession,
  event: AcpBridgeStreamEvent,
  options: DaemonAcpBridgeHostOptions = {},
): boolean {
  const previous = entry.binding
  let next: AcpSessionBinding = previous

  if (event.type === 'session_init') {
    const withExternal = event.externalSessionId
      ? { ...next, external_session_id: event.externalSessionId }
      : next
    next = event.model
      ? mergeBindingModelAndModeState(withExternal, { current_model: event.model })
      : withExternal
  } else if (event.type === 'model_update') {
    next = mergeBindingModelAndModeState(next, {
      current_model: event.modelId,
      ...(event.modelName ? { current_model_name: event.modelName } : {}),
    })
  } else if (event.type === 'mode_update') {
    next = mergeBindingModelAndModeState(next, {
      current_mode: event.modeId,
      ...(event.modeName ? { current_mode_name: event.modeName } : {}),
    })
  } else if (event.type === 'config_update') {
    if (event.configOptions) {
      const extracted = extractAcpSessionModelAndModeState({
        configOptions: event.configOptions,
        raw: event.raw,
      })
      if (entry.kind === 'acp' && extracted.modes && extracted.modes.length > 0) {
        entry.nativeModes = extracted.modes
      }
      next = mergeBindingModelAndModeState(next, extracted)
    } else if (event.configId && event.value !== undefined) {
      next = applySingleConfigOptionPatch(next, event.configId, event.value)
    }
  } else if (event.type === 'available_commands') {
    const classified = classifyAcpRuntimeFeatures({
      provider: next.provider,
      agentName: next.name,
      cwd: next.cwd,
      rawCommands: event.commands,
      includeFilesystemUnlisted: entry.kind === 'claude-stream-json',
      ...(options.env ? { env: options.env } : {}),
      ...(options.homeDir ? { homeDir: options.homeDir } : {}),
    })
    const {
      available_commands: _prevCommands,
      discovered_skills: _prevSkills,
      ...rest
    } = next
    void _prevCommands
    void _prevSkills
    next = {
      ...rest,
      ...(classified.available_commands.length > 0
        ? { available_commands: classified.available_commands }
        : {}),
      ...(classified.discovered_skills.length > 0
        ? { discovered_skills: classified.discovered_skills }
        : {}),
    }
  } else {
    return false
  }

  if (entry.kind === 'acp') {
    next = projectAcpBindingWithHostTurbo(entry, next)
  }

  if (JSON.stringify(previous) === JSON.stringify(next)) {
    return false
  }
  entry.binding = next
  return true
}

function resolveModeConfigId(binding: AcpSessionBinding): string | undefined {
  if (binding.mode_config_id?.trim()) {
    return binding.mode_config_id.trim()
  }
  const match = binding.config_options?.find(
    opt => opt.category?.trim().toLowerCase() === 'mode' || opt.id.trim().toLowerCase() === 'mode',
  )
  return match?.id.trim() || undefined
}

function isModeConfigOptionId(binding: AcpSessionBinding, configId: string): boolean {
  const lower = configId.trim().toLowerCase()
  if (!lower) return false
  if (binding.mode_config_id?.trim().toLowerCase() === lower) return true
  if (lower === 'mode') return true
  const opt = binding.config_options?.find(item => item.id.trim().toLowerCase() === lower)
  return opt?.category?.trim().toLowerCase() === 'mode'
}

function normalizeModeLookupKey(modeId: string): string {
  return modeId.trim().toLowerCase().replace(/[\s_-]+/g, '')
}

function pickNativeUnderlyingModeForTurbo(
  nativeModes: readonly AcpExternalModeOption[],
): string | undefined {
  const findBy = (pred: (normalized: string) => boolean): string | undefined =>
    nativeModes.find(item => pred(normalizeModeLookupKey(item.id)))?.id
  return (
    findBy(norm => norm === ACP_TURBO_MODE_ID) ??
    findBy(norm => isUnrestrictedAcpModeId(norm)) ??
    findBy(norm => norm === 'auto') ??
    findBy(norm => norm === 'acceptedits') ??
    findBy(norm => norm === 'default') ??
    nativeModes[0]?.id
  )
}

/**
 * Augment an ACP agent's advertised modes list with host-enforced `Turbo` right
 * after `Auto` (or `Accept edits`, or before `Plan`) whenever the agent exposes
 * modes but lacks a native unrestricted/bypass mode.
 */
export function augmentModesWithHostTurbo(
  nativeModes: readonly AcpExternalModeOption[],
): readonly AcpExternalModeOption[] {
  if (nativeModes.length === 0) {
    return nativeModes
  }
  if (nativeModes.some(mode => isUnrestrictedAcpModeId(mode.id))) {
    return nativeModes
  }
  const result = [...nativeModes]
  const autoIdx = result.findIndex(mode => normalizeModeLookupKey(mode.id) === 'auto')
  if (autoIdx >= 0) {
    result.splice(autoIdx + 1, 0, ACP_TURBO_MODE_OPTION)
    return result
  }
  const acceptEditsIdx = result.findIndex(mode => normalizeModeLookupKey(mode.id) === 'acceptedits')
  if (acceptEditsIdx >= 0) {
    result.splice(acceptEditsIdx + 1, 0, ACP_TURBO_MODE_OPTION)
    return result
  }
  const planIdx = result.findIndex(mode => {
    const norm = normalizeModeLookupKey(mode.id)
    return norm === 'plan' || norm === 'planning'
  })
  if (planIdx >= 0) {
    result.splice(planIdx, 0, ACP_TURBO_MODE_OPTION)
    return result
  }
  result.push(ACP_TURBO_MODE_OPTION)
  return result
}

function projectAcpBindingWithHostTurbo(
  entry: ActiveAcpSession,
  binding: AcpSessionBinding,
): AcpSessionBinding {
  const augmentedModes =
    entry.nativeModes.length > 0
      ? augmentModesWithHostTurbo(entry.nativeModes)
      : entry.hostTurboActive
        ? [ACP_TURBO_MODE_OPTION]
        : binding.modes

  const modeConfigId = resolveModeConfigId(binding)
  const effectiveCurrentMode = entry.hostTurboActive ? ACP_TURBO_MODE_ID : binding.current_mode
  const effectiveCurrentModeName = entry.hostTurboActive
    ? ACP_TURBO_MODE_OPTION.name
    : effectiveCurrentMode
      ? (augmentedModes?.find(item => item.id === effectiveCurrentMode)?.name ??
        binding.current_mode_name ??
        effectiveCurrentMode)
      : undefined

  let projectedConfigOptions = binding.config_options
  if (projectedConfigOptions && projectedConfigOptions.length > 0) {
    const targetModeIdLower = modeConfigId?.toLowerCase()
    let configChanged = false
    const updatedConfigs = projectedConfigOptions.map(option => {
      const isModeOpt =
        (targetModeIdLower !== undefined && option.id.toLowerCase() === targetModeIdLower) ||
        option.category?.toLowerCase() === 'mode' ||
        option.id.toLowerCase() === 'mode'
      if (!isModeOpt) {
        return option
      }
      const nextOptions: readonly AcpExternalConfigOptionChoice[] =
        augmentedModes && augmentedModes.length > 0
          ? augmentedModes.map(mode => ({
              value: mode.id,
              name: mode.name,
              ...(mode.description ? { description: mode.description } : {}),
            }))
          : option.options
      const nextValue = entry.hostTurboActive ? ACP_TURBO_MODE_ID : option.current_value
      if (
        nextValue === option.current_value &&
        JSON.stringify(nextOptions) === JSON.stringify(option.options)
      ) {
        return option
      }
      configChanged = true
      return {
        ...option,
        current_value: nextValue,
        options: nextOptions,
      }
    })
    if (configChanged) {
      projectedConfigOptions = updatedConfigs
    }
  }

  return {
    ...binding,
    ...(augmentedModes && augmentedModes.length > 0 ? { modes: augmentedModes } : {}),
    ...(effectiveCurrentMode ? { current_mode: effectiveCurrentMode } : {}),
    ...(effectiveCurrentModeName ? { current_mode_name: effectiveCurrentModeName } : {}),
    ...(projectedConfigOptions && projectedConfigOptions.length > 0
      ? { config_options: projectedConfigOptions }
      : {}),
  }
}

function readNonEmptyString(
  record: Readonly<Record<string, unknown>>,
  ...keys: readonly string[]
): string | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string') {
      const trimmed = value.trim()
      if (trimmed.length > 0) {
        return trimmed
      }
    }
  }
  return undefined
}

function trimToolNamesMap(map: Map<string, unknown>, maxEntries: number): void {
  while (map.size > maxEntries) {
    const oldest = map.keys().next().value
    if (oldest === undefined) break
    map.delete(oldest)
  }
}

function trimStringSet(set: Set<string>, maxEntries: number): void {
  while (set.size > maxEntries) {
    const oldest = set.values().next().value
    if (oldest === undefined) break
    set.delete(oldest)
  }
}

const CLIENT_FILE_MAX_BYTES = 8 * 1024 * 1024
const ANTIGRAVITY_CLIENT_READ_TITLES = new Set([
  'Running client_view_file',
  'Run client_view_file?',
])

/**
 * Antigravity sends native multiple-choice user questions over `session/request_permission`
 * with `toolCallId` prefixed by `interaction_`. Unlike tool approvals, these must always be
 * answered by the user (even in Turbo / Full Access mode).
 */
export function isAntigravityUserInputRequest(request: AcpClientPermissionRequest): boolean {
  return request.toolCallId.startsWith('interaction_') && request.options.length > 0
}

/**
 * `agy acp` 1.5.0+ supports mid-turn prompt steering via concurrent `session/prompt`;
 * 1.4.x harnesses queue follow-ups instead. Unversioned test/dev runtimes allow live steering
 * with automatic queued fallback.
 */
export function antigravitySteersWithPrompt(version: string | undefined): boolean {
  const trimmed = version?.trim()
  if (!trimmed) return true
  const match = /(\d+)\.(\d+)\.(\d+)/.exec(trimmed)
  if (!match) return true
  const major = Number(match[1])
  const minor = Number(match[2])
  return major > 1 || (major === 1 && minor >= 5)
}

function selectNonInteractiveAgyAuthMethod(
  env: Readonly<Record<string, string | undefined>> | undefined,
): 'gemini-api-key' | 'agent-platform' | undefined {
  if (!env) return undefined
  if (env.GEMINI_API_KEY?.trim()) return 'gemini-api-key'
  if (
    env.AGY_ADC_AUTH?.trim() ||
    (env.GOOGLE_CLOUD_QUOTA_PROJECT?.trim() && env.GOOGLE_CLOUD_LOCATION?.trim())
  ) {
    return 'agent-platform'
  }
  return undefined
}

/**
 * `agy` emits its `update_steps` checklist as tool calls (`title === "Run update_steps"`)
 * rather than ACP `plan` updates. Transform them into live `plan` events and drop redundant
 * noise updates for those call IDs (matching T3 Code's `makeAntigravityUpdateStepsPlans`).
 */
export function makeAntigravityUpdateStepsPlans() {
  const calls = new Set<string>()
  return (event: AcpClientEvent): AcpClientEvent | undefined => {
    if (event.type !== 'tool_call' && event.type !== 'tool_call_update') {
      return event
    }
    const key = `${event.sessionId}\u0000${event.toolCallId}`
    if (
      (event.title === 'Run update_steps' || event.name === 'Run update_steps') &&
      (!event.kind || event.kind === 'other')
    ) {
      calls.add(key)
    }
    if (!calls.has(key)) return event
    if (event.status === 'completed' || event.status === 'failed') {
      calls.delete(key)
    }
    if (!isAcpRecord(event.rawInput) || !Array.isArray(event.rawInput.steps)) {
      return undefined
    }
    const entries: Array<{ readonly content: string; readonly priority?: string; readonly status?: string }> = []
    for (const step of event.rawInput.steps) {
      if (!isAcpRecord(step) || typeof step.title !== 'string' || !step.title.trim()) continue
      const rawStatus = typeof step.status === 'string' ? step.status.trim() : 'pending'
      entries.push({
        content: step.title.trim(),
        priority: 'medium',
        status: rawStatus === 'done' ? 'completed' : rawStatus || 'pending',
      })
    }
    return {
      type: 'plan',
      sessionId: event.sessionId,
      entries,
    }
  }
}

function antigravityClientReadPath(
  event: Extract<AcpClientEvent, { readonly type: 'tool_call' | 'tool_call_update' }>,
): string | undefined {
  const title = event.title ?? event.name
  if (!title || !ANTIGRAVITY_CLIENT_READ_TITLES.has(title)) return undefined
  const locationPath = event.locations?.[0]?.path
  if (locationPath) return locationPath
  if (isAcpRecord(event.rawInput)) {
    const raw = event.rawInput.absolute_path ?? event.rawInput.path
    if (typeof raw === 'string' && raw.trim()) return raw.trim()
  }
  return undefined
}

/**
 * `agy` sometimes leaves subagent `client_view_file` tool calls open after `fs/read_text_file`
 * succeeds, then fails every open call with the next unrelated step error. Mark served reads
 * completed immediately and suppress spurious later `failed` overwrites (matching T3 Code).
 */
export function makeAntigravityReadRepair() {
  type Outcome = 'served' | 'refused'
  const openReads = new Map<string, { readonly path: string; readonly sessionId: string; outcome?: Outcome }>()
  const earlyOutcomes: Array<{ readonly path: string; readonly outcome: Outcome }> = []
  return {
    reset(): void {
      openReads.clear()
      earlyOutcomes.length = 0
    },
    attempted(path: string, served: boolean): { readonly sessionId: string; readonly toolCallId: string } | undefined {
      const outcome: Outcome = served ? 'served' : 'refused'
      for (const [toolCallId, read] of openReads) {
        if (read.path !== path || read.outcome !== undefined) continue
        read.outcome = outcome
        return served ? { sessionId: read.sessionId, toolCallId } : undefined
      }
      earlyOutcomes.push({ path, outcome })
      return undefined
    },
    repair(event: AcpClientEvent): AcpClientEvent {
      if (event.type !== 'tool_call' && event.type !== 'tool_call_update') {
        return event
      }
      const path = antigravityClientReadPath(event)
      if (path !== undefined && !openReads.has(event.toolCallId)) {
        const earlyIdx = earlyOutcomes.findIndex(entry => entry.path === path)
        const outcome = earlyIdx === -1 ? undefined : earlyOutcomes.splice(earlyIdx, 1)[0]?.outcome
        openReads.set(
          event.toolCallId,
          outcome === undefined ? { path, sessionId: event.sessionId } : { path, sessionId: event.sessionId, outcome },
        )
      }
      const read = openReads.get(event.toolCallId)
      if (event.status === 'pending' || event.status === 'in_progress') {
        return read?.outcome === 'served'
          ? { ...event, status: 'completed' }
          : event
      }
      if (event.status !== 'completed' && event.status !== 'failed') {
        return event
      }
      openReads.delete(event.toolCallId)
      if (event.status === 'completed' || read?.outcome !== 'served') {
        return event
      }
      if (event.type === 'tool_call_update') {
        const { rawOutput: _stepError, isError: _isError, ...rest } = event
        return { ...rest, status: 'completed' }
      }
      return { ...event, status: 'completed' }
    },
  }
}

function transformAntigravityClientEvent(
  entry: ActiveAcpSession,
  event: AcpClientEvent,
): AcpClientEvent | undefined {
  const repaired = entry.readRepair ? entry.readRepair.repair(event) : event
  return entry.updateStepsPlans ? entry.updateStepsPlans(repaired) : repaired
}

function isInsideRoot(root: string, candidate: string): boolean {
  const rel = relative(root, candidate)
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))
}

/**
 * Resolve all symlinks (canonicalizing the deepest existing ancestor) before checking
 * workspace and `additionalDirectories` containment (matching T3 Code's `AntigravityClientFiles`).
 */
export async function resolveWorkspaceScopedPathReal(
  cwd: string,
  requestedPath: string,
  additionalDirectories: readonly string[] = [],
): Promise<string> {
  const root = resolve(cwd)
  const target = isAbsolute(requestedPath) ? resolve(requestedPath) : resolve(root, requestedPath)
  const allowedRawRoots = [root, ...additionalDirectories.map(dir => resolve(root, dir))]
  const deny = () =>
    new Error(`ACP filesystem request denied: "${requestedPath}" escapes workspace "${root}"`)

  let candidate = target
  const missingSuffix: string[] = []
  let canonicalTarget: string | undefined
  while (true) {
    try {
      const real = await realpath(candidate)
      canonicalTarget = missingSuffix.length > 0 ? join(real, ...missingSuffix) : real
      break
    } catch {
      // Walk up only past truly missing paths, never past dangling/unreadable symlinks.
      let existsOrLink = true
      try {
        await lstat(candidate)
      } catch (statErr) {
        const code = (statErr as NodeJS.ErrnoException)?.code
        if (code === 'ENOENT') {
          existsOrLink = false
        }
      }
      const parent = dirname(candidate)
      if (existsOrLink || parent === candidate) {
        throw deny()
      }
      missingSuffix.unshift(basename(candidate))
      candidate = parent
    }
  }

  const canonicalRoots = await Promise.all(
    allowedRawRoots.map(async dir => {
      try {
        return await realpath(dir)
      } catch {
        return dir
      }
    }),
  )
  for (const allowedRoot of canonicalRoots) {
    if (isInsideRoot(allowedRoot, canonicalTarget)) {
      return canonicalTarget
    }
  }
  throw deny()
}

async function readWorkspaceScopedTextFile(
  cwd: string,
  fileReq: { readonly limit?: number; readonly line?: number; readonly path: string },
  additionalDirectories: readonly string[],
): Promise<{ readonly content: string }> {
  const safePath = await resolveWorkspaceScopedPathReal(cwd, fileReq.path, additionalDirectories)
  const info = await stat(safePath)
  if (!info.isFile() || info.size > CLIENT_FILE_MAX_BYTES) {
    throw new Error(
      `File "${fileReq.path}" is not a readable text file under ${CLIENT_FILE_MAX_BYTES} bytes.`,
    )
  }
  const bytes = await readFile(safePath)
  let raw: string
  try {
    raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    throw new Error(`File "${fileReq.path}" is not a UTF-8 text file.`)
  }
  if (raw.includes('\0')) {
    throw new Error(`File "${fileReq.path}" contains binary data.`)
  }
  if (fileReq.line !== undefined || fileReq.limit !== undefined) {
    const lines = raw.split('\n')
    const start = Math.max(0, (fileReq.line ?? 1) - 1)
    const end =
      fileReq.limit !== undefined && fileReq.limit > 0
        ? Math.min(lines.length, start + fileReq.limit)
        : lines.length
    return { content: lines.slice(start, end).join('\n') }
  }
  return { content: raw }
}

function normalizeAdditionalDirectories(
  cwd: string,
  directories: readonly string[] | undefined,
): readonly string[] {
  if (!directories || directories.length === 0) return []
  const root = resolve(cwd)
  const seen = new Set<string>([root])
  const out: string[] = []
  for (const raw of directories) {
    const trimmed = raw.trim()
    if (!trimmed) continue
    const resolved = isAbsolute(trimmed) ? resolve(trimmed) : resolve(root, trimmed)
    if (!seen.has(resolved)) {
      seen.add(resolved)
      out.push(resolved)
    }
  }
  return out
}

/** Convert Xerxes MCP server configs into ACP spec `mcpServers` entries filtered by agent `mcpCapabilities`. */
export function buildAcpWireMcpServers(
  servers: readonly AcpConnectMcpServerInput[] | undefined,
  capabilities: AcpMcpCapabilities,
): readonly Readonly<Record<string, unknown>>[] {
  if (!servers || servers.length === 0) return []
  const wire: Array<Readonly<Record<string, unknown>>> = []
  const seen = new Set<string>()
  for (const server of servers) {
    if (server.enabled === false) continue
    const name = server.name.trim()
    if (!name || seen.has(name)) continue
    const transport = server.transport ?? 'stdio'
    if (transport === 'stdio') {
      if (!server.command?.trim()) continue
      seen.add(name)
      wire.push({
        name,
        command: server.command.trim(),
        args: [...(server.args ?? [])],
        env: Object.entries(server.env ?? {}).map(([envName, value]) => ({
          name: envName,
          value,
        })),
      })
    } else if (transport === 'streamable_http') {
      if (!capabilities.http || !server.url?.trim()) continue
      seen.add(name)
      wire.push({
        type: 'http',
        name,
        url: server.url.trim(),
        headers: Object.entries(server.headers ?? {}).map(([headerName, value]) => ({
          name: headerName,
          value,
        })),
      })
    } else if (transport === 'sse') {
      if (!capabilities.sse || !server.url?.trim()) continue
      seen.add(name)
      wire.push({
        type: 'sse',
        name,
        url: server.url.trim(),
        headers: Object.entries(server.headers ?? {}).map(([headerName, value]) => ({
          name: headerName,
          value,
        })),
      })
    }
  }
  return wire
}

/** Append supported `agy acp` CLI flags (`--process-per-session`, `--add-dir`, `--ide-mcp-port`, `--agent`, `--sandbox`). */
export function buildAgyLaunchArgs(
  command: string,
  baseArgs: readonly string[],
  options: {
    readonly additionalDirectories?: readonly string[]
    readonly agent?: string
    readonly ideMcpPort?: number
    readonly mcpServers?: readonly AcpConnectMcpServerInput[]
    readonly sandbox?: boolean
  },
): readonly string[] {
  const cmdBase = basename(command).toLowerCase()
  const isAgySubcommand =
    (cmdBase === 'agy' || cmdBase === 'agy.exe' || cmdBase === 'antigravity' || cmdBase === 'antigravity.exe') &&
    baseArgs[0] === 'acp'
  if (!isAgySubcommand) {
    return baseArgs
  }
  const args = [...baseArgs]
  const hasFlag = (flag: string): boolean =>
    args.some(arg => arg === flag || arg.startsWith(`${flag}=`))
  const hasEnabledHttpMcp = (options.mcpServers ?? []).some(
    server => server.enabled !== false && server.transport === 'streamable_http' && Boolean(server.url?.trim()),
  )
  if (hasEnabledHttpMcp && !hasFlag('--process-per-session')) {
    args.push('--process-per-session')
  }
  for (const dir of options.additionalDirectories ?? []) {
    if (dir.trim()) {
      args.push('--add-dir', dir.trim())
    }
  }
  if (
    typeof options.ideMcpPort === 'number' &&
    Number.isInteger(options.ideMcpPort) &&
    options.ideMcpPort > 0 &&
    options.ideMcpPort <= 65535 &&
    !hasFlag('--ide-mcp-port')
  ) {
    args.push('--ide-mcp-port', String(options.ideMcpPort))
  }
  if (options.agent?.trim() && !hasFlag('--agent')) {
    args.push('--agent', options.agent.trim())
  }
  if (options.sandbox === true && !hasFlag('--sandbox')) {
    args.push('--sandbox')
  }
  return args
}

function isAtLeastSemver(
  version: string | undefined,
  minMajor: number,
  minMinor: number,
  minPatch: number,
): boolean {
  if (!version) return true
  const match = /^v?(\d+)\.(\d+)(?:\.(\d+))?/.exec(version.trim())
  if (!match) return true
  const major = Number.parseInt(match[1] ?? '0', 10)
  const minor = Number.parseInt(match[2] ?? '0', 10)
  const patch = Number.parseInt(match[3] ?? '0', 10)
  if (major !== minMajor) return major > minMajor
  if (minor !== minMinor) return minor > minMinor
  return patch >= minPatch
}

function isConcurrentPromptRejection(error: unknown): boolean {
  if (error instanceof AcpRpcError) {
    if (error.code === -32002 || error.code === -32600 || error.code === -32603 || error.code === -32000) {
      if (/busy|in\s+progress|concurrent|active\s+prompt|already\s+(?:running|active|in\s+progress)|not\s+supported/i.test(error.message)) {
        return true
      }
      if (error.code === -32002) return true
    }
  }
  const msg = errorMessage(error)
  return /busy|prompt\s+already|in\s+progress|concurrent\s+prompt|already\s+(?:running|active)/i.test(msg)
}
