// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { isAbsolute, relative, resolve } from 'node:path'

import {
  AcpStdioClient,
  isAcpAuthRequiredError,
  type AcpAvailableCommand,
  type AcpClientEvent,
  type AcpClientPermissionRequest,
  type AcpClientPermissionResponse,
  type AcpElicitationRequest,
  type AcpElicitationResponse,
  type AcpInitializeResult,
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
  type AcpRuntimeCapabilities,
  type AcpSessionBinding,
  type AcpToolMetaSnapshot,
  type ClaudeStreamJsonRunnerOptions,
  type DiscoverExternalAcpAgentsOptions,
} from '../acp/externalAgents.js'
import { isAcpRecord } from '../acp/types.js'
import { buildAgySubprocessEnv, hasSshSessionEnv, loadAgyRuntimeEnv } from '../auth/agyRuntimeEnv.js'
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

export interface DaemonAcpBridgeHostOptions {
  /** Test seam to override external ACP client construction. */
  readonly createAcpClient?: (options: AcpStdioClientOptions) => AcpStdioClient
  /** Test seam to override Claude stream-json runner construction. */
  readonly createClaudeRunner?: (options: ClaudeStreamJsonRunnerOptions) => ClaudeStreamJsonRunner
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
  /** Override host platform for deterministic tests. */
  readonly platform?: NodeJS.Platform
  /** Test seam for custom binary validation. */
  readonly resolveBinary?: (command: string) => string | undefined
}

export interface AcpConnectRequest {
  readonly agentId: string
  readonly args?: readonly string[]
  readonly command?: string
  readonly cwd: string
  readonly initialModeId?: string
  readonly permissionMode?: string
  readonly resumeSessionId?: string
  readonly sessionId: string
}

export type AcpConnectResult =
  | { readonly binding: AcpSessionBinding; readonly ok: true }
  | { readonly error: string; readonly install_hint?: string; readonly ok: false }

export type AcpBindingMutationResult =
  | { readonly binding: AcpSessionBinding; readonly ok: true }
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

export interface AcpPromptResourceInput {
  readonly mimeType?: string
  readonly path?: string
  readonly text: string
  readonly uri: string
}

export interface AcpRunPromptRequest {
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

export interface AcpRunPromptResult {
  readonly assistantText: string
  readonly cancelled: boolean
  readonly error?: string
  readonly externalSessionId?: string
  readonly stopReason: string
  readonly thinkingText: string
  readonly usage?: AcpPromptUsage
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
  binding: AcpSessionBinding
  readonly client: AcpStdioClient
  readonly externalSessionId: string
  hostTurboActive: boolean
  hostTurboUnderlyingMode: string | undefined
  readonly kind: 'acp'
  nativeModes: readonly AcpExternalModeOption[]
  readonly outOfTurnSettledToolCallIds: Set<string>
  readonly promptCapabilities: AcpPromptCapabilities
  readonly sessionCapabilities: AcpSessionCapabilities
  readonly toolMetaById: Map<string, AcpToolMetaSnapshot>
  readonly toolNamesById: Map<string, string>
}

interface ActiveClaudeSession {
  activeAbort: AbortController | undefined
  binding: AcpSessionBinding
  readonly kind: 'claude-stream-json'
  readonly runner: ClaudeStreamJsonRunner
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
  private readonly bindingListeners = new Set<(sessionId: string, binding: AcpSessionBinding) => void>()
  private readonly sessionEventListeners = new Set<(sessionId: string, event: AcpBridgeStreamEvent) => void>()
  private readonly options: DaemonAcpBridgeHostOptions

  constructor(options: DaemonAcpBridgeHostOptions = {}) {
    this.options = options
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
        ...(request.initialModeId ? { initialModeId: request.initialModeId } : {}),
        ...(request.resumeSessionId ? { resumeSessionId: request.resumeSessionId } : {}),
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
      ...(request.initialModeId ? { initialModeId: request.initialModeId } : {}),
      ...(request.resumeSessionId ? { resumeSessionId: request.resumeSessionId } : {}),
      ...(resolvedDiscovered.installHint ? { installHint: resolvedDiscovered.installHint } : {}),
    })
  }

  async disconnect(sessionId: string): Promise<boolean> {
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
    value: string,
  ): Promise<AcpBindingMutationResult> {
    const sid = sessionId.trim()
    const targetConfigId = configId.trim()
    const targetValue = value.trim()
    if (!sid) {
      return { ok: false, error: 'session_id is required to set an ACP config option.' }
    }
    if (!targetConfigId) {
      return { ok: false, error: 'config_id is required.' }
    }
    if (!targetValue) {
      return { ok: false, error: 'value is required.' }
    }
    const existing = this.active.get(sid)
    if (!existing || !existing.binding.connected) {
      return { ok: false, error: `No external ACP agent is connected for session "${sid}".` }
    }
    if (existing.kind === 'claude-stream-json') {
      if (targetConfigId.toLowerCase() === 'model') {
        return this.setModel(sid, targetValue)
      }
      return {
        ok: false,
        error: `${existing.binding.name} (${existing.binding.transport}) does not support ACP config option "${targetConfigId}".`,
      }
    }

    if (isModeConfigOptionId(existing.binding, targetConfigId)) {
      return this.setMode(sid, targetValue, targetConfigId)
    }

    try {
      const rpcResult = await existing.client.setConfigOption({
        sessionId: existing.externalSessionId,
        configId: targetConfigId,
        value: targetValue,
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
          : applySingleConfigOptionPatch(existing.binding, targetConfigId, targetValue, extracted)
      existing.binding = projectAcpBindingWithHostTurbo(existing, nextBinding)
      return { ok: true, binding: existing.binding }
    } catch (error) {
      return {
        ok: false,
        error: `Failed to set ${existing.binding.name} option "${targetConfigId}" to "${targetValue}": ${errorMessage(error)}`,
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
        const promptBlocks = buildAcpPromptContentBlocks({
          agentName: existing.binding.name,
          capabilities: existing.promptCapabilities,
          text: request.text,
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
            onEvent: acpEvent => {
              const bridged = bridgeEventFromAcpClientEvent(
                acpEvent,
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
            },
          })
          if (result.usage) {
            handleBridgeEvent({
              type: 'usage',
              ...(result.usage.inputTokens !== undefined ? { inputTokens: result.usage.inputTokens } : {}),
              ...(result.usage.outputTokens !== undefined ? { outputTokens: result.usage.outputTokens } : {}),
              ...(result.usage.cachedReadTokens !== undefined ? { cachedReadTokens: result.usage.cachedReadTokens } : {}),
              ...(result.usage.cachedWriteTokens !== undefined ? { cachedWriteTokens: result.usage.cachedWriteTokens } : {}),
              ...(result.usage.thoughtTokens !== undefined ? { thoughtTokens: result.usage.thoughtTokens } : {}),
              ...(result.usage.totalTokens !== undefined ? { totalTokens: result.usage.totalTokens } : {}),
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
      readonly agentId: string
      readonly args: readonly string[]
      readonly command: string
      readonly cwd: string
      readonly initialModeId?: string
      readonly installHint?: string
      readonly name: string
      readonly provider: AcpExternalAgentProvider
      readonly resumeSessionId?: string
      readonly sessionId: string
    },
    attemptOptions?: { readonly stripSshSession?: boolean },
  ): Promise<AcpConnectResult> {
    await this.disconnect(params.sessionId)

    let entryRef: ActiveAcpSession | undefined
    let initResult: AcpInitializeResult | undefined
    let initialAvailableCommands: readonly AcpAvailableCommand[] | undefined
    let initialUsageEvent: AcpClientEvent | undefined
    let initialSessionInfoEvent: AcpClientEvent | undefined

    const isAntigravity = params.provider === 'antigravity' || params.agentId === 'antigravity'
    const transferredAgyEnv = isAntigravity
      ? await loadAgyRuntimeEnv(this.options.homeDir ?? homedir())
      : {}
    const mergedEnv = isAntigravity
      ? buildAgySubprocessEnv(
          this.options.env
            ? { ...transferredAgyEnv, ...this.options.env }
            : { ...process.env, ...transferredAgyEnv },
          { ...(attemptOptions?.stripSshSession ? { stripSshSession: true } : {}) },
        )
      : this.options.env

    const clientOptions: AcpStdioClientOptions = {
      command: params.command,
      args: params.args,
      cwd: params.cwd,
      ...(mergedEnv ? { env: mergedEnv } : {}),
      handlers: {
        createElicitation: async elicitReq => {
          const activeHandler = entryRef?.activeElicitationHandler
          if (!activeHandler) {
            return { action: 'cancel' }
          }
          return activeHandler(elicitReq)
        },
        requestPermission: async permReq => {
          const activeHandler = entryRef?.activePermissionHandler
          if (activeHandler) {
            return activeHandler(permReq)
          }
          if (entryRef && (entryRef.hostTurboActive || isAcpTurboMode(entryRef.binding))) {
            const allowOption =
              permReq.options.find(opt => opt.kind === 'allow_once') ??
              permReq.options.find(opt => opt.kind === 'allow_always') ??
              permReq.options[0]
            if (allowOption) {
              return { outcome: 'selected', optionId: allowOption.optionId }
            }
          }
          return { outcome: 'cancelled' }
        },
        readTextFile: async fileReq => {
          const safePath = resolveWorkspaceScopedPath(params.cwd, fileReq.path)
          const raw = await readFile(safePath, 'utf8')
          if (fileReq.line !== undefined || fileReq.limit !== undefined) {
            const lines = raw.split('\n')
            const start = Math.max(0, (fileReq.line ?? 1) - 1)
            const end = fileReq.limit !== undefined && fileReq.limit > 0
              ? start + fileReq.limit
              : lines.length
            return { content: lines.slice(start, end).join('\n') }
          }
          return { content: raw }
        },
        writeTextFile: async fileReq => {
          const safePath = resolveWorkspaceScopedPath(params.cwd, fileReq.path)
          await writeFile(safePath, fileReq.content, 'utf8')
        },
      },
    }

    const client = this.options.createAcpClient
      ? this.options.createAcpClient(clientOptions)
      : new AcpStdioClient(clientOptions)

    // Subscribe BEFORE initialize/session startup so initial notifications
    // (`available_commands_update`, `usage_update`, `session_info_update`) and
    // unexpected subprocess exits are never missed.
    client.onEvent(acpEvent => {
      if (!entryRef) {
        if (acpEvent.type === 'available_commands') {
          initialAvailableCommands = acpEvent.commands
        } else if (acpEvent.type === 'usage_update') {
          initialUsageEvent = acpEvent
        } else if (acpEvent.type === 'session_info') {
          initialSessionInfoEvent = acpEvent
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
          acpEvent.type === 'continuation')
      ) {
        const bridged = bridgeEventFromAcpClientEvent(
          acpEvent,
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
      const sessionResult = await openOrResumeAcpSession(client, initResult, params.cwd, params.resumeSessionId)
      const commandDisplay = params.args.length > 0
        ? `${params.command} ${params.args.join(' ')}`
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
      }
      const entry: ActiveAcpSession = {
        kind: 'acp',
        client,
        externalSessionId: sessionResult.sessionId,
        binding: rawBinding,
        hostTurboActive: false,
        hostTurboUnderlyingMode: undefined,
        nativeModes: extracted.modes ?? [],
        outOfTurnSettledToolCallIds: new Set<string>(),
        activeAbort: undefined,
        activeElicitationAbort: undefined,
        activeElicitationHandler: undefined,
        activePermissionHandler: undefined,
        promptCapabilities: initResult.promptCapabilities,
        sessionCapabilities: initResult.sessionCapabilities,
        toolMetaById: new Map<string, AcpToolMetaSnapshot>(),
        toolNamesById: new Map<string, string>(),
      }
      entry.binding = projectAcpBindingWithHostTurbo(entry, rawBinding)
      entryRef = entry
      this.active.set(params.sessionId, entry)

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
        if (isAntigravity) {
          const hostPlatform = this.options.platform ?? process.platform
          if (
            hostPlatform === 'darwin' &&
            !attemptOptions?.stripSshSession &&
            mergedEnv &&
            hasSshSessionEnv(mergedEnv)
          ) {
            return this.connectStdioAcp(params, { stripSshSession: true })
          }
          return {
            ok: false,
            error: ANTIGRAVITY_AUTH_ERROR,
          }
        }
        const authNames = (initResult?.authMethods ?? []).map(method => method.name).filter(Boolean)
        const authDetail = authNames.length > 0 ? ` (${authNames.join(', ')})` : ''
        return {
          ok: false,
          error: `${params.name} requires authentication${authDetail}. Sign in via the ${params.name} CLI, then click Retry.`,
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
) {
  const trimmedResume = resumeSessionId?.trim()
  if (trimmedResume) {
    if (initResult.sessionCapabilities.resume) {
      try {
        return await client.resumeSession({ sessionId: trimmedResume, cwd })
      } catch (resumeError) {
        if (isAcpAuthRequiredError(resumeError)) {
          throw resumeError
        }
        if (initResult.sessionCapabilities.load || initResult.supportsLoadSession) {
          try {
            return await client.loadSession({ sessionId: trimmedResume, cwd })
          } catch (loadError) {
            if (isAcpAuthRequiredError(loadError)) {
              throw loadError
            }
          }
        }
      }
    } else if (initResult.sessionCapabilities.load || initResult.supportsLoadSession) {
      try {
        return await client.loadSession({ sessionId: trimmedResume, cwd })
      } catch (loadError) {
        if (isAcpAuthRequiredError(loadError)) {
          throw loadError
        }
      }
    }
  }
  return client.newSession({ cwd })
}

function buildAcpPromptContentBlocks(input: {
  readonly agentName: string
  readonly capabilities: AcpPromptCapabilities
  readonly fallbackInlineText?: string
  readonly images?: readonly AcpPromptImageInput[]
  readonly resources?: readonly AcpPromptResourceInput[]
  readonly text: string
}): readonly AcpPromptContentBlock[] {
  const blocks: AcpPromptContentBlock[] = []
  const hasResources = Boolean(input.resources && input.resources.length > 0)
  let textBody = hasResources && !input.capabilities.embeddedContext
    ? (input.fallbackInlineText || buildInlineResourcesFallback(input.text, input.resources!))
    : input.text

  if (input.images && input.images.length > 0 && !input.capabilities.image) {
    const count = input.images.length
    textBody = `${textBody}\n\n[Note: ${count} image attachment${count === 1 ? '' : 's'} omitted because ${input.agentName} does not advertise ACP image input support.]`
  }

  blocks.push({ type: 'text', text: textBody })

  if (hasResources && input.capabilities.embeddedContext) {
    for (const resource of input.resources!) {
      blocks.push({
        type: 'resource',
        resource: {
          uri: resource.uri,
          ...(resource.mimeType ? { mimeType: resource.mimeType } : { mimeType: 'text/plain' }),
          text: resource.text,
        },
      })
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

  return blocks
}

function buildInlineResourcesFallback(
  text: string,
  resources: readonly AcpPromptResourceInput[],
): string {
  const parts = resources.map(res => {
    const label = res.path || res.uri
    return `<file path="${label}">\n${res.text}\n</file>`
  })
  return `<attached_files>\n${parts.join('\n\n')}\n</attached_files>\n\n${text}`
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
      list: initResult.sessionCapabilities.list,
      close: initResult.sessionCapabilities.close,
      additional_directories: initResult.sessionCapabilities.additionalDirectories,
    },
    mcp: {
      http: initResult.mcpCapabilities.http,
      sse: initResult.mcpCapabilities.sse,
    },
    auth_methods: initResult.authMethods.map(method => ({
      id: method.id,
      name: method.name,
      ...(method.description ? { description: method.description } : {}),
    })),
    ...(initResult.capabilitiesMeta ? { meta: initResult.capabilitiesMeta } : {}),
  }
}

function parseConfigOptions(configOptions: readonly unknown[]): AcpExternalConfigOption[] {
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
    const type = readNonEmptyString(item, 'type') ?? 'select'
    const currentValue = readConfigOptionValue(item, 'currentValue', 'current_value', 'value') ?? ''
    const rawChoices = parseOptionList(item.options, ['value', 'id', 'modelId', 'modeId'])
    const options: AcpExternalConfigOptionChoice[] = rawChoices.map(choice => ({
      value: choice.id,
      name: choice.name,
      ...(choice.description ? { description: choice.description } : {}),
    }))
    parsed.push({
      id,
      name,
      ...(description ? { description } : {}),
      ...(category ? { category } : {}),
      type,
      current_value: currentValue,
      options,
    })
  }
  return parsed
}

function readConfigOptionValue(
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
    } else if (typeof value === 'boolean') {
      return value ? 'true' : 'false'
    } else if (typeof value === 'number' && Number.isFinite(value)) {
      return String(value)
    }
  }
  return undefined
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
  updates: Readonly<Record<string, string>>,
): readonly AcpExternalConfigOption[] | undefined {
  if (!existing || existing.length === 0) return existing
  const keys = Object.keys(updates)
  if (keys.length === 0) return existing
  let changed = false
  const next = existing.map(option => {
    const direct = updates[option.id]
    const byCategory = option.category ? updates[option.category] : undefined
    const targetValue = direct ?? byCategory
    if (!targetValue || option.current_value === targetValue) {
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
  value: string,
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
    } else if (event.configId && event.value) {
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

function resolveWorkspaceScopedPath(cwd: string, requestedPath: string): string {
  const root = resolve(cwd)
  const target = isAbsolute(requestedPath) ? resolve(requestedPath) : resolve(root, requestedPath)
  const rel = relative(root, target)
  if (rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`ACP filesystem request denied: "${requestedPath}" escapes workspace "${root}"`)
  }
  return target
}
