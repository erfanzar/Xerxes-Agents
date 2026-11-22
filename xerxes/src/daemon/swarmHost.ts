// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { cp, mkdir, realpath, rm } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import { profileSessionOptions } from '../auth/profileSessionOptions.js'
import { profileQuota, type ProfileQuota } from '../auth/profileUsage.js'
import type { UsageProfile } from '../auth/usage.js'
import type { ProfileStore, ProviderProfile } from '../bridge/profiles.js'
import { ToolRegistry, type ToolExecutor } from '../executors/toolRegistry.js'
import type { SkillRegistry } from '../extensions/skills.js'
import { createLlmClient, type LlmClient } from '../llms/client.js'
import type { LspManager } from '../lsp/manager.js'
import type { MCPManager } from '../mcp/manager.js'
import type { PtySessionManager } from '../operators/pty.js'
import { localOnlyActive } from '../runtime/localOnly.js'
import { modelInventory, type ModelInventoryPort } from '../runtime/modelInventory.js'
import { ProcessRegistry } from '../runtime/processRegistry.js'
import { runWithActiveSession } from '../runtime/sessionContext.js'
import type { TerminalMonitors } from '../runtime/terminalMonitors.js'
import type { TerminalRegistry } from '../runtime/terminalRegistry.js'
import type { PermissionRequest, StreamEvent } from '../streaming/events.js'
import {
  ALWAYS_APPROVAL_TOOLS,
  type PermissionBroker,
  type PermissionDecision,
  type PermissionMode,
} from '../streaming/permissions.js'
import {
  COORDINATOR_NAME,
  createRunTurnWorkerRunner,
  defaultSwarmStorageDirectory,
  formatBoardWakeNotice,
  SwarmManager,
  type SwarmCoordinatorWakeEvent,
  type SwarmLifecycleEvent,
  type SwarmMemberSnapshot,
  type SwarmModelCatalogPort,
  type SwarmModelInfo,
  type SwarmWorkerRunnerPort,
  type SwarmWorkspaceAllocation,
  type SwarmWorkspaceCleanupRequest,
  type SwarmWorkspaceCreateRequest,
  type SwarmWorkspacePort,
} from '../swarm/index.js'
import { BackgroundCommandManager } from '../tools/backgroundCommands.js'
import { registerClaudeNotebookTools } from '../tools/claudeTools/notebook.js'
import {
  registerClaudeSkillTool,
  registerCoreTools,
  type ModelInventoryHost,
} from '../tools/index.js'
import { LSP_RUNTIME_TOOL_NAME, registerConfiguredLspTool } from '../tools/lspTools.js'
import { registerMcpTools } from '../tools/mcpTools.js'
import { registerMonitorTools } from '../tools/monitorTools.js'
import { registerScheduleTools, type ScheduleToolHost } from '../tools/scheduleTools.js'
import { createSwarmTools } from '../tools/swarmTools.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import { resolveProjectDirectory, xerxesHome } from './paths.js'
import type { DaemonSession, DaemonTranscriptMessage } from './runtime.js'
import {
  registerSwarmWorkerScheduleTools,
  SwarmWorkerScheduleRegistry,
  type SwarmWorkerTaskSnapshot,
} from './swarmWorkerSchedules.js'
import type { SessionSwarmTurnAdapter } from './turnRunner.js'

export {
  registerSwarmWorkerScheduleTools,
  SwarmWorkerScheduleRegistry,
  type SwarmWorkerTaskSnapshot,
} from './swarmWorkerSchedules.js'

const gitWorktreeLocks = new Map<string, Promise<void>>()

async function withGitWorktreeLock<T>(lockKey: string, operation: () => Promise<T>): Promise<T> {
  const pending = (gitWorktreeLocks.get(lockKey) ?? Promise.resolve()).then(operation, operation)
  const settled = pending.then(() => undefined, () => undefined)
  gitWorktreeLocks.set(lockKey, settled)
  try {
    return await pending
  } finally {
    if (gitWorktreeLocks.get(lockKey) === settled) {
      gitWorktreeLocks.delete(lockKey)
    }
  }
}

function sanitizedGitEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {
    ...Bun.env,
    GIT_OPTIONAL_LOCKS: '0',
  }
  for (const key of [
    'GIT_DIR',
    'GIT_WORK_TREE',
    'GIT_COMMON_DIR',
    'GIT_INDEX_FILE',
    'GIT_OBJECT_DIRECTORY',
    'GIT_ALTERNATE_OBJECT_DIRECTORIES',
    'GIT_PREFIX',
  ]) {
    delete env[key]
  }
  return env
}

async function runGitCommand(
  args: readonly string[],
  cwd: string,
  signal?: AbortSignal,
): Promise<{ readonly exitCode: number; readonly stdout: string; readonly stderr: string }> {
  signal?.throwIfAborted()
  const timeoutSignal = AbortSignal.timeout(60_000)
  const combinedSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal
  const proc = Bun.spawn(['git', '-c', 'core.hooksPath=/dev/null', ...args], {
    cwd,
    env: sanitizedGitEnv(),
    stdout: 'pipe',
    stderr: 'pipe',
    signal: combinedSignal,
  })
  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  return {
    exitCode,
    stdout: stdout.trim(),
    stderr: stderr.trim(),
  }
}

export interface DaemonSwarmWorkspacePortOptions {
  readonly shareRootDir?: string
  readonly branchRootDir?: string
}

/**
 * Production SwarmWorkspacePort for Xerxes Daemon:
 * - `'inherit'`: shares the caller's working directory.
 * - `'share'`: allocates an isolated Git worktree (`git worktree add --detach`)
 *   serialized per repository common dir and removes it on `kill`.
 * - `'branch'`: clones via `git clone --local` into `<baseDir>/.xerxes/swarm-workspaces/...`
 *   (or `branchRootDir`), falling back to a filtered copy for non-Git directories,
 *   and removes the clone directory on `kill`.
 */
export function createDaemonSwarmWorkspacePort(
  options: DaemonSwarmWorkspacePortOptions = {},
): SwarmWorkspacePort {
  return {
    async createWorkspace(request: SwarmWorkspaceCreateRequest): Promise<SwarmWorkspaceAllocation> {
      const baseDir = resolve(request.baseDir)
      if (request.mode === 'inherit') {
        return {
          mode: 'inherit',
          path: baseDir,
        }
      }

      const scopeHash = createHash('sha256')
        .update(`${baseDir}\u0000${request.sessionId}`, 'utf8')
        .digest('hex')
        .slice(0, 12)
      const token = crypto.randomUUID().slice(0, 8)

      if (request.mode === 'share') {
        const [commonProbe, headProbe] = await Promise.all([
          runGitCommand(
            ['rev-parse', '--path-format=absolute', '--git-common-dir'],
            baseDir,
            request.signal,
          ),
          runGitCommand(['rev-parse', 'HEAD'], baseDir, request.signal),
        ])
        if (commonProbe.exitCode !== 0 || !commonProbe.stdout) {
          throw new Error(
            `Cannot allocate "share" git worktree for "${request.memberName}": "${baseDir}" is not a Git repository (${commonProbe.stderr || 'git rev-parse failed'})`,
          )
        }
        const canonicalCommon = await realpath(commonProbe.stdout).catch(() => commonProbe.stdout)
        const shareRoot = options.shareRootDir
          ? resolve(options.shareRootDir)
          : join(canonicalCommon, 'xerxes-swarm-worktrees')
        const targetPath = join(shareRoot, scopeHash, `share-${request.memberName}-${token}`)
        await mkdir(dirname(targetPath), { recursive: true, mode: 0o700 })

        await withGitWorktreeLock(canonicalCommon, async () => {
          const added = await runGitCommand(
            ['worktree', 'add', '--detach', targetPath, 'HEAD'],
            baseDir,
            request.signal,
          )
          if (added.exitCode !== 0) {
            throw new Error(
              `Failed to create "share" git worktree for "${request.memberName}": ${added.stderr || `exit ${added.exitCode}`}`,
            )
          }
        })
        const resolvedPath = await realpath(targetPath).catch(() => targetPath)
        const baseHead = headProbe.exitCode === 0 && headProbe.stdout ? headProbe.stdout : undefined
        return {
          mode: 'share',
          path: resolvedPath,
          handle_id: canonicalCommon,
          ...(baseHead === undefined ? {} : { base_head: baseHead }),
        }
      }

      // mode === 'branch': clone into `.xerxes/swarm-workspaces/<scope>/branch-<member>-<token>`
      const branchRoot = options.branchRootDir
        ? resolve(options.branchRootDir)
        : join(baseDir, '.xerxes', 'swarm-workspaces')
      const targetPath = join(branchRoot, scopeHash, `branch-${request.memberName}-${token}`)
      await mkdir(dirname(targetPath), { recursive: true, mode: 0o700 })

      const gitProbe = await runGitCommand(['rev-parse', '--is-inside-work-tree'], baseDir, request.signal)
      if (gitProbe.exitCode === 0 && gitProbe.stdout === 'true') {
        const headProbe = await runGitCommand(['rev-parse', 'HEAD'], baseDir, request.signal)
        const cloned = await runGitCommand(
          ['clone', '--local', '--quiet', baseDir, targetPath],
          baseDir,
          request.signal,
        )
        if (cloned.exitCode === 0) {
          const resolvedPath = await realpath(targetPath).catch(() => targetPath)
          const baseHead = headProbe.exitCode === 0 && headProbe.stdout ? headProbe.stdout : undefined
          return {
            mode: 'branch',
            path: resolvedPath,
            ...(baseHead === undefined ? {} : { base_head: baseHead }),
          }
        }
      }

      const normalizedBranchRoot = resolve(branchRoot)
      await cp(baseDir, targetPath, {
        recursive: true,
        filter: (src) => {
          const resolvedSrc = resolve(src)
          return resolvedSrc !== normalizedBranchRoot && !resolvedSrc.startsWith(`${normalizedBranchRoot}/`) && !resolvedSrc.startsWith(`${normalizedBranchRoot}\\`)
        },
      })
      const resolvedPath = await realpath(targetPath).catch(() => targetPath)
      return {
        mode: 'branch',
        path: resolvedPath,
      }
    },

    async cleanupWorkspace(request: SwarmWorkspaceCleanupRequest): Promise<void> {
      if (request.mode === 'inherit') {
        return
      }
      const baseDir = resolve(request.baseDir)
      const targetPath = resolve(request.workspacePath)
      if (!targetPath || targetPath === baseDir) {
        return
      }
      if (request.mode === 'share') {
        const lockKey = request.handleId ?? baseDir
        await withGitWorktreeLock(lockKey, async () => {
          await runGitCommand(['worktree', 'remove', '--force', targetPath], baseDir).catch(() => undefined)
          await rm(targetPath, { recursive: true, force: true }).catch(() => undefined)
          await runGitCommand(['worktree', 'prune'], baseDir).catch(() => undefined)
        })
        return
      }
      await rm(targetPath, { recursive: true, force: true }).catch(() => undefined)
    },
  }
}

export interface DaemonSwarmModelCatalogOptions {
  readonly profileStore?: Pick<ProfileStore, 'list' | 'active' | 'get'>
  readonly defaultModel?: () => string
  readonly quotaFetcher?: (profile: UsageProfile, signal?: AbortSignal) => Promise<ProfileQuota>
  readonly extraModels?: () => Promise<readonly SwarmModelInfo[]> | readonly SwarmModelInfo[]
}

function applyProfileQuotaToModel(
  base: SwarmModelInfo,
  quota: ProfileQuota | undefined,
): SwarmModelInfo {
  if (!quota || quota.status !== 'available' || !quota.windows.length) {
    return base
  }
  const maxUsed = Math.max(...quota.windows.map(window => window.used_percent))
  const remaining = Math.max(0, Math.round(100 - maxUsed))
  const exhaustedWindow = quota.windows.find(window => window.used_percent >= 100)
  const resets = exhaustedWindow?.resets_at
    ?? quota.windows.find(window => window.resets_at)?.resets_at
    ?? undefined
  const note = quota.windows
    .map(window => `${window.label}: ${Math.round(window.used_percent)}% used`)
    .join('; ')

  return {
    ...base,
    quota_remaining: remaining,
    ...(resets ? { quota_resets: resets } : {}),
    quota_exhausted: Boolean(exhaustedWindow),
    ...(note ? { quota_note: note } : {}),
  }
}

/**
 * SwarmModelCatalogPort backed by Xerxes's `modelInventory` + `profileQuota`.
 */
export function createDaemonSwarmModelCatalogPort(
  options: DaemonSwarmModelCatalogOptions = {},
): SwarmModelCatalogPort {
  const fetchQuota = options.quotaFetcher ?? ((profile, signal) => profileQuota(profile, signal ? { signal } : {}))

  return {
    getDefaultModel() {
      const explicit = options.defaultModel?.()?.trim()
      if (explicit) return explicit
      return options.profileStore?.active()?.model?.trim() || 'default'
    },

    async listModels(listOptions = {}): Promise<readonly SwarmModelInfo[]> {
      const includeAll = listOptions.all === true
      const byId = new Map<string, SwarmModelInfo>()
      const defaultModelId = (options.defaultModel?.()?.trim() || options.profileStore?.active()?.model?.trim() || '').trim()

      const profiles = options.profileStore?.list() ?? []
      const activeName = options.profileStore?.active()?.name

      if (profiles.length > 0) {
        const inventoryPort: ModelInventoryPort = {
          profiles: () => profiles.map(profile => ({
            name: profile.name,
            provider: profile.provider,
            model: profile.model,
            active: profile.name === activeName,
          })),
          quota: async (profileName, signal) => {
            const target = profiles.find(item => item.name === profileName)
            if (!target) {
              return { status: 'unknown', remaining_tokens: null, reason: 'Unknown profile' }
            }
            return fetchQuota(target, signal)
          },
          discover: async (profileName) => {
            const target = profiles.find(item => item.name === profileName)
            if (!target) return { models: [], source: 'configured' }
            const knownIds = new Set<string>()
            if (target.model.trim()) knownIds.add(target.model.trim())
            if (target.model_capabilities) {
              for (const id of Object.keys(target.model_capabilities)) {
                if (id.trim()) knownIds.add(id.trim())
              }
            }
            return {
              models: [...knownIds].map(id => ({ id })),
              source: 'configured',
            }
          },
          reasoning: async () => ({ efforts: [], source: 'unknown', shape: 'toggle' }),
        }

        const providerPage = await modelInventory(inventoryPort, { limit: 50 }).catch(() => undefined)
        const listedProfiles = providerPage?.entries ?? []

        const quotaSignal = AbortSignal.timeout(2_500)
        const quotaByProfile = new Map<string, ProfileQuota>()
        await Promise.all(profiles.map(async (profile) => {
          try {
            const q = await fetchQuota(profile, quotaSignal)
            quotaByProfile.set(profile.name, q)
          } catch {
            // Keep catalog listing responsive if a quota endpoint times out.
          }
        }))

        for (const entry of listedProfiles) {
          const profileName = typeof entry.provider_profile === 'string' ? entry.provider_profile : ''
          const configuredModel = typeof entry.configured_model === 'string' ? entry.configured_model.trim() : ''
          const provider = typeof entry.provider === 'string' ? entry.provider : ''
          const quota = quotaByProfile.get(profileName)

          if (configuredModel) {
            const baseInfo: SwarmModelInfo = {
              id: configuredModel,
              display_name: profileName ? `${configuredModel} (${profileName})` : configuredModel,
              suggested: true,
            }
            if (!byId.has(configuredModel)) {
              byId.set(configuredModel, applyProfileQuotaToModel(baseInfo, quota))
            }
          }

          if (includeAll && profileName) {
            const modelPage = await modelInventory(inventoryPort, {
              provider_profile: profileName,
              limit: 50,
            }).catch(() => undefined)
            for (const modelEntry of modelPage?.entries ?? []) {
              const modelId = typeof modelEntry.model === 'string' ? modelEntry.model.trim() : ''
              if (!modelId || byId.has(modelId)) continue
              const display = provider ? `${modelId} (${profileName})` : modelId
              byId.set(
                modelId,
                applyProfileQuotaToModel(
                  {
                    id: modelId,
                    display_name: display,
                    suggested: modelId === configuredModel || modelId === defaultModelId,
                  },
                  quota,
                ),
              )
            }
          }
        }
      }

      if (options.extraModels) {
        const extra = await options.extraModels()
        for (const item of extra) {
          byId.set(item.id, { ...item })
        }
      }

      if (defaultModelId && !byId.has(defaultModelId)) {
        byId.set(defaultModelId, {
          id: defaultModelId,
          display_name: defaultModelId,
          suggested: true,
        })
      }

      return [...byId.values()]
    },
  }
}

function isMessageContentValue(value: unknown): value is MessageContent {
  if (typeof value === 'string') return true
  return Array.isArray(value) && value.every(part => typeof part === 'object' && part !== null && 'type' in part)
}

/**
 * Convert persisted DaemonSession transcript records into provider ChatMessage[]
 * for `fork: "coordinator"`.
 */
export function transcriptMessagesToChatMessages(
  messages: readonly DaemonTranscriptMessage[],
): ChatMessage[] {
  const result: ChatMessage[] = []
  for (const item of messages) {
    const role = item.role
    const content = item.content ?? (typeof item.text === 'string' ? item.text : undefined)
    if (role === 'user' && isMessageContentValue(content)) {
      result.push({ role: 'user', content })
      continue
    }
    if (role === 'assistant' && isMessageContentValue(content)) {
      result.push({
        role: 'assistant',
        content,
        ...(typeof item.thinking === 'string' ? { thinking: item.thinking } : {}),
        ...(Array.isArray(item.tool_calls) ? { tool_calls: item.tool_calls as readonly ToolCall[] } : {}),
      })
      continue
    }
    if (role === 'tool' && typeof content === 'string' && typeof item.tool_call_id === 'string') {
      result.push({
        role: 'tool',
        content,
        tool_call_id: item.tool_call_id,
        ...(typeof item.name === 'string' ? { name: item.name } : {}),
      })
    }
  }
  return result
}

const PLAN_MODE_WORKER_SAFE_TOOLS: ReadonlySet<string> = new Set([
  'read_board',
  'post_message',
  'swarm_wait',
  'swarm_overview',
  'schedule',
  'manage_task',
  'SkillTool',
  LSP_RUNTIME_TOOL_NAME,
  'list_monitors',
  'list_monitor_sources',
  'read_terminal_output',
  'check_command',
  'list_commands',
  'pty_list',
  'list_schedules',
  'list_available_models',
])

function isGlobalApprovalGatedCall(request: PermissionRequest): boolean {
  const toolName = request.toolCall.function.name
  if (ALWAYS_APPROVAL_TOOLS.has(toolName) || toolName === 'ExitPlanModeTool') {
    return true
  }
  const action = typeof request.inputs.action === 'string' ? request.inputs.action : ''
  if (toolName === 'manage_schedule' && ['create', 'update', 'resume', 'run'].includes(action)) {
    return true
  }
  if (toolName === 'CreatorForgeTool' && (action === 'define' || action === 'undefine')) {
    return true
  }
  if (toolName === 'PluginManageTool' && action !== 'list' && action !== 'inspect') {
    return true
  }
  return false
}

function isPlanSafeWorkerCall(request: PermissionRequest): boolean {
  const toolName = request.toolCall.function.name
  if (PLAN_MODE_WORKER_SAFE_TOOLS.has(toolName)) {
    return true
  }
  if (toolName === 'swarm') {
    const action = typeof request.inputs.action === 'string' ? request.inputs.action : ''
    return action === 'status' || action === 'models'
  }
  return false
}

function normalizePermissionMode(value: unknown): PermissionMode | undefined {
  if (value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan') {
    return value
  }
  return undefined
}

export interface DaemonSwarmHostOptions {
  readonly storageRootDir?: string
  readonly shareWorkspaceRootDir?: string
  readonly branchWorkspaceRootDir?: string
  readonly workspacePort?: SwarmWorkspacePort
  readonly modelCatalog?: SwarmModelCatalogPort
  readonly profileStore?: Pick<ProfileStore, 'list' | 'active' | 'get'>
  readonly quotaFetcher?: (profile: UsageProfile, signal?: AbortSignal) => Promise<ProfileQuota>
  readonly extraModels?: () => Promise<readonly SwarmModelInfo[]> | readonly SwarmModelInfo[]
  readonly defaultModel?: () => string
  readonly fallbackLlm?: LlmClient | (() => LlmClient | undefined)
  readonly resolveWorkerLlm?: (session: DaemonSession | undefined, member: SwarmMemberSnapshot) => LlmClient | undefined
  readonly configureWorkerTools?: (params: {
    readonly registry: ToolRegistry
    readonly member: SwarmMemberSnapshot
    readonly workspacePath: string
    readonly sessionId: string
  }) => Promise<void> | void
  readonly modelInventoryHost?: ModelInventoryHost
  readonly skillRegistryForWorkspace?: (cwd: string) => SkillRegistry | undefined
  readonly scheduleHost?: ScheduleToolHost
  readonly processRegistry?: ProcessRegistry
  readonly backgroundCommands?: BackgroundCommandManager
  readonly ptySessions?: PtySessionManager | (() => PtySessionManager | undefined)
  readonly terminalRegistry?: TerminalRegistry
  readonly terminalMonitors?: TerminalMonitors
  readonly lspManager?: LspManager | ((cwd: string) => LspManager | undefined)
  readonly mcpManager?: MCPManager | ((cwd: string) => MCPManager | undefined)
  readonly permissionBroker?: PermissionBroker
  readonly permissionMode?: PermissionMode | ((session: DaemonSession | undefined) => PermissionMode)
  readonly sessionLookup?: (sessionId: string) => DaemonSession | undefined
  readonly onCoordinatorModelSwitch?: (sessionId: string, model: SwarmModelInfo) => Promise<void> | void
  readonly onEvent?: (event: SwarmLifecycleEvent, workspaceDir: string) => void
  readonly onCoordinatorWake?: (event: SwarmCoordinatorWakeEvent, workspaceDir: string) => void
  readonly onWorkerStreamEvent?: (sessionId: string, memberName: string, event: StreamEvent) => void
}

export class DaemonSwarmHost {
  private readonly managers = new Map<string, SwarmManager>()
  private readonly initPromises = new Map<string, Promise<SwarmManager>>()
  private readonly workspaceDirs = new Map<string, string>()
  private readonly eventListeners = new Set<(event: SwarmLifecycleEvent, workspaceDir: string) => void>()
  private readonly coordinatorWakeListeners = new Set<(event: SwarmCoordinatorWakeEvent, workspaceDir: string) => void>()
  private readonly workspacePort: SwarmWorkspacePort
  private readonly defaultModelCatalog: SwarmModelCatalogPort
  private readonly workerSchedules = new SwarmWorkerScheduleRegistry()
  private readonly ownedBackgroundCommands: BackgroundCommandManager | undefined
  private closed = false

  constructor(private readonly options: DaemonSwarmHostOptions = {}) {
    if (options.onEvent) {
      this.eventListeners.add(options.onEvent)
    }
    if (options.onCoordinatorWake) {
      this.coordinatorWakeListeners.add(options.onCoordinatorWake)
    }
    this.workspacePort = options.workspacePort ?? createDaemonSwarmWorkspacePort({
      ...(options.shareWorkspaceRootDir === undefined ? {} : { shareRootDir: options.shareWorkspaceRootDir }),
      ...(options.branchWorkspaceRootDir === undefined ? {} : { branchRootDir: options.branchWorkspaceRootDir }),
    })
    this.defaultModelCatalog = options.modelCatalog ?? createDaemonSwarmModelCatalogPort({
      ...(options.profileStore === undefined ? {} : { profileStore: options.profileStore }),
      ...(options.defaultModel === undefined ? {} : { defaultModel: options.defaultModel }),
      ...(options.quotaFetcher === undefined ? {} : { quotaFetcher: options.quotaFetcher }),
      ...(options.extraModels === undefined ? {} : { extraModels: options.extraModels }),
    })
    if (options.backgroundCommands === undefined) {
      this.ownedBackgroundCommands = new BackgroundCommandManager(
        options.processRegistry ?? new ProcessRegistry(),
        options.terminalRegistry,
      )
    }
  }

  onEvent(listener: (event: SwarmLifecycleEvent, workspaceDir: string) => void): () => void {
    this.eventListeners.add(listener)
    return () => {
      this.eventListeners.delete(listener)
    }
  }

  onCoordinatorWake(listener: (event: SwarmCoordinatorWakeEvent, workspaceDir: string) => void): () => void {
    this.coordinatorWakeListeners.add(listener)
    return () => {
      this.coordinatorWakeListeners.delete(listener)
    }
  }

  peekManager(sessionId: string): SwarmManager | undefined {
    return this.managers.get(sessionId)
  }

  /** True when this session has living swarm workers or active worker schedules. */
  hasActiveSwarm(sessionId: string): boolean {
    if (this.workerSchedules.activeCount(sessionId) > 0) return true
    const manager = this.managers.get(sessionId)
    return manager !== undefined && manager.hasLivingWorkers()
  }

  /** Release an in-memory swarm manager when it has no living workers or active schedules. */
  async evictIdleManager(sessionId: string): Promise<boolean> {
    if (this.hasActiveSwarm(sessionId)) return false
    const manager = this.managers.get(sessionId)
    if (!manager) return false
    await this.disposeSession(sessionId)
    return true
  }

  activeWorkerScheduleCount(sessionId?: string, memberName?: string): number {
    return this.workerSchedules.activeCount(sessionId, memberName)
  }

  listWorkerSchedules(sessionId: string, memberName: string): SwarmWorkerTaskSnapshot[] {
    return this.workerSchedules.listForMember(sessionId, memberName)
  }

  getOrCreateSync(target: {
    readonly id: string
    readonly cwd: string
    readonly model?: string
  }): SwarmManager {
    const existing = this.managers.get(target.id)
    if (existing) {
      if (target.model && target.model.trim()) {
        existing.setCoordinatorModel(target.model.trim())
      }
      return existing
    }

    const workspaceDir = resolveProjectDirectory(target.cwd || process.cwd())
    this.workspaceDirs.set(target.id, workspaceDir)
    const storageDir = this.options.storageRootDir
      ? join(resolve(this.options.storageRootDir), createHash('sha256').update(`${workspaceDir}\u0000${target.id}`, 'utf8').digest('hex').slice(0, 24))
      : defaultSwarmStorageDirectory(workspaceDir, target.id)

    const initialModel = target.model?.trim()
      || this.options.defaultModel?.()?.trim()
      || this.options.profileStore?.active()?.model?.trim()
      || 'default'

    const sessionModelCatalog: SwarmModelCatalogPort = this.options.modelCatalog ?? createDaemonSwarmModelCatalogPort({
      ...(this.options.profileStore === undefined ? {} : { profileStore: this.options.profileStore }),
      defaultModel: () => {
        const liveSession = this.options.sessionLookup?.(target.id)
        return liveSession?.model?.trim()
          || this.options.defaultModel?.()?.trim()
          || initialModel
      },
      ...(this.options.quotaFetcher === undefined ? {} : { quotaFetcher: this.options.quotaFetcher }),
      ...(this.options.extraModels === undefined ? {} : { extraModels: this.options.extraModels }),
    })

    let managerRef: SwarmManager | undefined
    const workerRunner = this.buildWorkerRunner(target.id, workspaceDir, () => managerRef)

    const manager = new SwarmManager({
      sessionId: target.id,
      workspaceDir,
      storageDir,
      coordinatorModel: initialModel,
      modelCatalog: sessionModelCatalog,
      workspacePort: this.workspacePort,
      forkMessages: (memberName) => {
        if (memberName === COORDINATOR_NAME) {
          const liveSession = this.options.sessionLookup?.(target.id)
          return liveSession ? transcriptMessagesToChatMessages(liveSession.messages) : []
        }
        return managerRef?.getMemberHistory(memberName) ?? []
      },
      ...(workerRunner === undefined ? {} : { workerRunner }),
      onCoordinatorModelSwitch: async (modelInfo) => {
        await this.options.onCoordinatorModelSwitch?.(target.id, modelInfo)
      },
      onEvent: (event) => {
        if (event.kind === 'message_posted' && event.message) {
          this.workerSchedules.onBoardMessagePosted(target.id, event.message)
        } else if (event.kind === 'member_killed' && event.member?.name) {
          this.workerSchedules.cancelMember(target.id, event.member.name, 'member_killed')
        }
        if (!this.closed) {
          for (const listener of this.eventListeners) {
            listener(event, workspaceDir)
          }
        }
      },
      onCoordinatorWake: (event) => {
        if (!this.closed) {
          for (const listener of this.coordinatorWakeListeners) {
            listener(event, workspaceDir)
          }
        }
      },
    })
    managerRef = manager
    this.managers.set(target.id, manager)

    const initPromise = manager.initialize().then(() => manager)
    this.initPromises.set(target.id, initPromise)
    return manager
  }

  async ensureSession(target: {
    readonly id: string
    readonly cwd: string
    readonly model?: string
  }): Promise<SwarmManager> {
    const manager = this.getOrCreateSync(target)
    const pending = this.initPromises.get(target.id)
    if (pending) {
      await pending
    }
    if (target.model && target.model.trim()) {
      manager.setCoordinatorModel(target.model.trim())
    }
    return manager
  }

  coordinatorSkillInstructions(cwd: string): string {
    const registry = this.options.skillRegistryForWorkspace?.(cwd)
    const skill = registry?.get('immortals-coordinator') ?? registry?.get('swarm-coordinator')
    return skill?.instructions?.trim() ?? ''
  }

  async activateCoordinator(target: {
    readonly id: string
    readonly cwd: string
    readonly model?: string
    readonly enabled?: boolean
  }): Promise<{ readonly ok: true; readonly active: boolean; readonly prompt_overlay: string }> {
    const manager = await this.ensureSession(target)
    const result = await manager.activate(target.enabled !== false)
    if (!result.active) {
      return result
    }
    const skillInstructions = this.coordinatorSkillInstructions(target.cwd)
    const combinedOverlay = skillInstructions
      ? `${result.prompt_overlay}\n\n${skillInstructions}`
      : result.prompt_overlay
    return {
      ok: true,
      active: true,
      prompt_overlay: combinedOverlay,
    }
  }

  sessionTurnAdapter(session: DaemonSession): SessionSwarmTurnAdapter {
    const manager = this.getOrCreateSync(session)
    const turnStartedActive = manager.isActive()
    const suite = createSwarmTools({ swarm: manager, caller: COORDINATOR_NAME })
    const visibleTools = suite.visibleDefinitions({
      caller: COORDINATOR_NAME,
      turnStartedActive,
    })
    const allowedToolNames = new Set(visibleTools.map(tool => tool.function.name))
    const byName = new Map(suite.allTools.map(tool => [tool.definition.function.name, tool]))

    const executor: ToolExecutor = {
      execute: async (call, context, signal) => {
        if (!allowedToolNames.has(call.function.name)) {
          throw new Error(
            `Tool "${call.function.name}" is not unlocked in this turn; spawn at least one worker with swarm{action:"add"} first so board tools unlock on the next turn.`,
          )
        }
        const bound = byName.get(call.function.name)
        if (!bound) {
          throw new Error(`Unknown swarm tool "${call.function.name}"`)
        }
        const args = parseCallArgs(call.function.arguments)
        const outcome = await bound.handler(args, context, signal)
        return typeof outcome === 'string' ? outcome : JSON.stringify(outcome)
      },
    }

    const shouldInjectOverlay = manager.isCoordinatorActivated() || manager.isActive()
    const baseOverlay = shouldInjectOverlay ? manager.buildCoordinatorPromptOverlay() : ''
    const skillInstructions = shouldInjectOverlay
      ? this.coordinatorSkillInstructions(session.cwd)
      : ''
    const promptOverlay = [baseOverlay, skillInstructions].filter(Boolean).join('\n\n') || undefined

    const unreadBoard = manager.isActive() && manager.head() > manager.coordinatorLastReadSeq()
    const contextNotice = unreadBoard
      ? formatBoardWakeNotice(manager.head(), manager.coordinatorLastReadSeq())
      : undefined

    return {
      tools: visibleTools,
      executor,
      capabilities: (toolName) => {
        const bound = byName.get(toolName)
        return {
          concurrencySafe: bound?.capabilities.concurrencySafe ?? false,
          interruptBehavior: bound?.capabilities.interruptBehavior ?? 'block',
        }
      },
      ...(promptOverlay === undefined ? {} : { promptOverlay }),
      ...(contextNotice === undefined ? {} : { contextNotice }),
      onTurnStart: () => {
        manager.setCoordinatorState('working')
      },
      onTurnEnd: () => {
        manager.setCoordinatorState('waiting')
      },
    }
  }

  async disposeSession(sessionId: string): Promise<void> {
    this.workerSchedules.cancelSession(sessionId, 'session_disposed')
    const manager = this.managers.get(sessionId)
    this.managers.delete(sessionId)
    this.initPromises.delete(sessionId)
    this.workspaceDirs.delete(sessionId)
    await Promise.allSettled([
      ...(manager ? [manager.dispose()] : []),
      ...(this.ownedBackgroundCommands ? [this.ownedBackgroundCommands.disposeOwner(sessionId)] : []),
    ])
  }

  async close(): Promise<void> {
    this.closed = true
    this.workerSchedules.close()
    const all = [...this.managers.values()]
    this.managers.clear()
    this.initPromises.clear()
    this.workspaceDirs.clear()
    await Promise.allSettled([
      ...all.map(manager => manager.dispose()),
      ...(this.ownedBackgroundCommands ? [this.ownedBackgroundCommands.disposeAll()] : []),
    ])
  }

  private resolveWorkerPermissionMode(sessionId: string): PermissionMode {
    const liveSession = this.options.sessionLookup?.(sessionId)
    if (liveSession?.interactionMode === 'plan') {
      return 'plan'
    }
    const configured = typeof this.options.permissionMode === 'function'
      ? this.options.permissionMode(liveSession)
      : this.options.permissionMode
    if (configured !== undefined) {
      return configured
    }
    return normalizePermissionMode(liveSession?.permissionMode) ?? 'accept-all'
  }

  private buildWorkerPermissionBroker(mode: PermissionMode): PermissionBroker {
    if (this.options.permissionBroker) {
      return this.options.permissionBroker
    }
    return {
      request: async (request: PermissionRequest): Promise<PermissionDecision> => {
        if (isGlobalApprovalGatedCall(request)) {
          return 'reject'
        }
        if (mode === 'plan') {
          return isPlanSafeWorkerCall(request) ? 'approve' : 'reject'
        }
        if (mode === 'auto') {
          return 'approve'
        }
        if (mode === 'manual') {
          const toolName = request.toolCall.function.name
          return toolName === 'read_board' || toolName === 'swarm_wait'
            ? 'approve'
            : 'reject'
        }
        return 'reject'
      },
    }
  }

  private resolveLlmForWorker(sessionId: string, member: SwarmMemberSnapshot): LlmClient | undefined {
    const liveSession = this.options.sessionLookup?.(sessionId)
    const custom = this.options.resolveWorkerLlm?.(liveSession, member)
    if (custom) return custom

    const profileStore = this.options.profileStore
    if (profileStore) {
      const profiles = profileStore.list()
      const matchedProfile: ProviderProfile | undefined = profiles.find(p => p.model === member.model)
        ?? profileStore.active()
      if (matchedProfile && (matchedProfile.api_key || matchedProfile.provider === 'ollama' || matchedProfile.provider === 'openai-codex' || matchedProfile.provider === 'claude-code' || matchedProfile.provider === 'github-copilot')) {
        try {
          return createLlmClient(
            member.model,
            {
              provider: matchedProfile.provider,
              api_key: matchedProfile.api_key,
              base_url: matchedProfile.base_url,
            },
            profileSessionOptions(matchedProfile),
          )
        } catch {
          // Fall back to host LlmClient when profile credentials cannot construct a standalone client.
        }
      }
    }

    const fallback = typeof this.options.fallbackLlm === 'function'
      ? this.options.fallbackLlm()
      : this.options.fallbackLlm
    return fallback
  }

  private buildWorkerRunner(
    sessionId: string,
    defaultWorkspaceDir: string,
    getManager: () => SwarmManager | undefined,
  ): SwarmWorkerRunnerPort | undefined {
    const canResolveAnyLlm = Boolean(
      this.options.resolveWorkerLlm
      || this.options.fallbackLlm
      || this.options.profileStore,
    )
    if (!canResolveAnyLlm) {
      return undefined
    }

    const effectiveBackgroundCommands = this.options.backgroundCommands ?? this.ownedBackgroundCommands

    return (input) => {
      const effectivePermissionMode = this.resolveWorkerPermissionMode(sessionId)
      const permissionBroker = this.buildWorkerPermissionBroker(effectivePermissionMode)

      const innerRunner = createRunTurnWorkerRunner({
        llm: (member) => {
          const client = this.resolveLlmForWorker(sessionId, member)
          if (!client) {
            throw new Error(`No LLM provider configured to execute swarm worker "${member.name}" (${member.model})`)
          }
          return client
        },
        configureWorkerRegistry: async ({ registry, member, workspacePath }) => {
          // Map synthetic `${sessionId}:swarm:${member.name}` turn sessionId back to the
          // owning parent sessionId so terminals, PTYs, background commands, monitors,
          // schedules, and MCP calls share full `/terminals` and `/monitors` visibility.
          const rawExecute = registry.execute.bind(registry)
          registry.execute = (call, context, signal) => {
            const scopedContext = context.sessionId && context.sessionId !== sessionId
              ? {
                  ...context,
                  sessionId,
                  metadata: {
                    ...context.metadata,
                    swarmWorkerSessionId: context.sessionId,
                  },
                }
              : context
            return rawExecute(call, scopedContext, signal)
          }

          if (this.options.terminalMonitors) {
            registerMonitorTools(registry, this.options.terminalMonitors)
          }
          const resolvedMcp = typeof this.options.mcpManager === 'function'
            ? this.options.mcpManager(workspacePath)
            : this.options.mcpManager
          if (resolvedMcp) {
            registerMcpTools(registry, resolvedMcp)
          }
          const resolvedLsp = typeof this.options.lspManager === 'function'
            ? this.options.lspManager(workspacePath)
            : this.options.lspManager
          if (resolvedLsp) {
            registerConfiguredLspTool(registry, resolvedLsp, () => workspacePath)
          }
          const resolvedPty = typeof this.options.ptySessions === 'function'
            ? this.options.ptySessions()
            : this.options.ptySessions

          const corePaths = registerCoreTools(registry, {
            workspaceRoot: workspacePath,
            activeWorkspaceRoot: () => workspacePath,
            includeCodingTools: true,
            readOnlyRoots: [join(xerxesHome(), 'tool-results')],
            ...(effectiveBackgroundCommands === undefined ? {} : { backgroundCommands: effectiveBackgroundCommands }),
            ...(resolvedPty === undefined ? {} : { ptySessions: resolvedPty }),
            ...(this.options.terminalRegistry === undefined ? {} : { terminals: this.options.terminalRegistry }),
            ...(this.options.terminalMonitors === undefined
              ? {}
              : {
                  completionWatch: (owner: string, terminalId: string) =>
                    this.options.terminalMonitors!.start(owner, {
                      terminalId,
                      trigger: 'completion',
                      durationMs: 86_400_000,
                      maxEvents: 1,
                    }),
                }),
            ...(localOnlyActive() ? { includeWebTools: false } : {}),
            ...(this.options.modelInventoryHost ? { modelInventory: this.options.modelInventoryHost } : {}),
          })
          registerClaudeNotebookTools(registry, corePaths)

          const skillRegistry = this.options.skillRegistryForWorkspace?.(workspacePath)
            ?? this.options.skillRegistryForWorkspace?.(defaultWorkspaceDir)
          if (skillRegistry) {
            registerClaudeSkillTool(registry, skillRegistry, 'default', () => workspacePath)
          }
          if (this.options.scheduleHost) {
            registerScheduleTools(registry, this.options.scheduleHost)
          }

          registerSwarmWorkerScheduleTools(registry, this.workerSchedules, {
            sessionId,
            memberName: member.name,
            wake: (memberName, wakePrompt) => {
              const liveManager = getManager() ?? this.managers.get(sessionId)
              if (!liveManager) return false
              return liveManager.wakeMember(memberName, wakePrompt, 'schedule')
            },
          })

          await this.options.configureWorkerTools?.({
            registry,
            member,
            workspacePath,
            sessionId,
          })
        },
        permissionMode: effectivePermissionMode,
        turnDependencies: {
          permissionBroker,
        },
        onStreamEvent: (memberName, event) => {
          const liveManager = getManager() ?? this.managers.get(sessionId)
          liveManager?.recordWorkerStreamEvent(memberName, event)
          this.options.onWorkerStreamEvent?.(sessionId, memberName, event)
        },
      })

      const workerSessionScope = {
        id: input.sessionId,
        workerSessionId: `${input.sessionId}:swarm:${input.member.name}`,
        cwd: input.workspacePath,
      }
      return runWithActiveSession(workerSessionScope, () => innerRunner(input))
    }
  }
}

function parseCallArgs(raw: JsonObject | string): JsonObject {
  if (typeof raw !== 'string') {
    return raw
  }
  const trimmed = raw.trim()
  if (!trimmed) return {}
  const parsed: unknown = JSON.parse(trimmed)
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('Tool arguments must be a JSON object')
  }
  return parsed as JsonObject
}
