// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { basename, dirname, resolve } from 'node:path'
import { mkdir } from 'node:fs/promises'

import { ValidationError } from '../core/errors.js'
import { errorMessage } from '../executors/toolRegistry.js'
import { triggerSoftGc } from '../runtime/memoryGuard.js'
import type { StreamEvent, ToolResult } from '../streaming/events.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import {
  defaultSwarmStorageDirectory,
  formatNarratorFeedbackText,
  parseBoardPostText,
  SwarmBoardStore,
  type PersistedSwarmState,
} from './board.js'
import {
  assertValidChannelName,
  computeChannelSummaries,
  createMainChannel,
  defaultChannelsForRole,
  memberCanSeeChannel,
  normalizeWorkerChannelList,
  roleHasWildcardChannels,
} from './channels.js'
import {
  buildSwarmCoordinatorPromptOverlay,
  buildSwarmWorkerPromptOverlay,
  buildWorkerModelSwitchPrompt,
  buildWorkerStartupPrompt,
  buildWorkerWakePrompt,
} from './prompts.js'
import {
  canManageWorker,
  expandPlannedWorkers,
  isNarratorName,
  resolveTargetMembers,
  roleForMemberName,
  toMemberSnapshot,
} from './roster.js'
import {
  MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS,
  normalizeWorkerHistory,
  sanitizeForkedMessages,
  type SwarmWorkerRunnerPort,
} from './runner.js'
import {
  COORDINATOR_NAME,
  isSwarmKillSignal,
  isSwarmParkSignal,
  MAIN_CHANNEL,
  SWARM_OPEN_BOARD_MARKDOWN,
  SwarmKillSignal,
  SwarmParkSignal,
  USER_BOARD_AUTHOR,
  WILDCARD_CHANNEL,
  type SwarmActivateResult,
  type SwarmAddInput,
  type SwarmAddResult,
  type SwarmApplyWinnerPatchOptions,
  type SwarmApplyWinnerPatchResult,
  type SwarmBoardMessage,
  type SwarmCandidateComparisonResult,
  type SwarmCandidateInput,
  type SwarmChannelActionInput,
  type SwarmChannelActionResult,
  type SwarmChannelRecord,
  type SwarmChannelSummary,
  type SwarmCompareCandidatesOptions,
  type SwarmCoordinatorWakeEvent,
  type SwarmEventListener,
  type SwarmExportedMemberPatch,
  type SwarmExportMemberPatchOptions,
  type SwarmForkMessagesPort,
  type SwarmInspectMessage,
  type SwarmKillInput,
  type SwarmKillResult,
  type SwarmLifecycleEvent,
  type SwarmLifecycleEventKind,
  type SwarmMemberActivitySnapshot,
  type SwarmMemberInspectResult,
  type SwarmMemberPatchInspection,
  type SwarmMemberRecord,
  type SwarmMemberSnapshot,
  type SwarmModelCatalogPort,
  type SwarmModelInfo,
  type SwarmOverviewSnapshot,
  type SwarmOverviewUpdateInput,
  type SwarmOverviewUpdateResult,
  type SwarmPostMessageInput,
  type SwarmPostMessageResult,
  type SwarmRankedCandidatePatch,
  type SwarmReadBoardInput,
  type SwarmReadBoardResult,
  type SwarmRosterStatusEntry,
  type SwarmStatusSnapshot,
  type SwarmSwitchModelInput,
  type SwarmSwitchModelResult,
  type SwarmTitleInput,
  type SwarmTitleResult,
  type SwarmUserPostInput,
  type SwarmWaitInput,
  type SwarmWaitResult,
  type SwarmWorkerErrorInfo,
  type SwarmWorkerToolActivity,
  type SwarmWorkerWakeReason,
  type SwarmWorkspaceAllocation,
  type SwarmWorkspacePort,
} from './types.js'
import {
  CoordinatorWaitHub,
  doesMessageReviveStoppedWorker,
  doesMessageWakeCoordinator,
  doesMessageWakeWorker,
  formatBoardWakeNotice,
  formatWorkerErrorWakeNotice,
  isQuotaOrCapacityError,
} from './wakeup.js'
import {
  applyWorkspacePatch,
  createDefaultSwarmWorkspacePort,
  inspectIsolatedWorkspacePatch,
} from './workspaces.js'

export interface SwarmManagerOptions {
  readonly sessionId: string
  readonly workspaceDir?: string
  readonly storageDir?: string
  readonly workspaceRootDir?: string
  readonly coordinatorModel?: string
  readonly coordinatorModelDisplayName?: string
  readonly modelCatalog?: SwarmModelCatalogPort
  readonly workspacePort?: SwarmWorkspacePort
  readonly forkMessages?: SwarmForkMessagesPort
  readonly workerRunner?: SwarmWorkerRunnerPort
  readonly onEvent?: SwarmEventListener
  readonly onCoordinatorWake?: (event: SwarmCoordinatorWakeEvent) => void
  readonly onCoordinatorModelSwitch?: (model: SwarmModelInfo) => Promise<void> | void
  readonly now?: () => Date
}

interface PendingWorkerJob {
  readonly reason: SwarmWorkerWakeReason
  readonly prompt: string
}

interface MutableWorkerActivity {
  phase: 'idle' | 'thinking' | 'responding' | 'tool'
  summary: string | undefined
  activeTools: SwarmWorkerToolActivity[]
  recentTools: SwarmWorkerToolActivity[]
  liveText: string
  updatedAt: string | undefined
  lastEmitMs: number
  lastEmittedPhase: 'idle' | 'thinking' | 'responding' | 'tool' | undefined
  lastEmittedSignature: string
}

const MAX_RECENT_TOOLS = 20
const MAX_LIVE_TEXT_PREVIEW_CHARS = 400
const ACTIVITY_PHASE_TRANSITION_THROTTLE_MS = 250
const ACTIVITY_SAME_PHASE_THROTTLE_MS = 500
const MAX_EVICTED_INSPECT_CACHE = 4
const MAX_WARM_IDLE_WORKER_HISTORIES = 8
const DEFAULT_INSPECT_LIMIT = 60
const MAX_INSPECT_LIMIT = 200
const MAX_INSPECT_TOOL_TEXT_CHARS = 1200
const MAX_INSPECT_MESSAGE_TEXT_CHARS = 4000
const MAX_TOOL_RESULT_PREVIEW_CHARS = 240
const MAX_TOOL_ARGS_PREVIEW_CHARS = 240

export class SwarmManager {
  readonly sessionId: string
  readonly workspaceDir: string
  readonly storageDir: string

  private readonly boardStore: SwarmBoardStore
  private readonly workspacePort: SwarmWorkspacePort
  private readonly modelCatalog: SwarmModelCatalogPort | undefined
  private readonly forkMessagesPort: SwarmForkMessagesPort | undefined
  private workerRunner: SwarmWorkerRunnerPort | undefined
  private readonly nowClock: () => Date
  private readonly eventListeners = new Set<SwarmEventListener>()
  private readonly coordinatorWakeListeners = new Set<(event: SwarmCoordinatorWakeEvent) => void>()
  private readonly onCoordinatorModelSwitchHook: ((model: SwarmModelInfo) => Promise<void> | void) | undefined
  private readonly coordinatorWaitHub = new CoordinatorWaitHub()

  private swarmTitle = ''
  private coordinatorActivated = false
  private coordinatorModelId: string
  private coordinatorModelDisplay: string
  private coordinatorState: 'working' | 'waiting' = 'waiting'
  private coordinatorParkedSince: number | undefined
  private coordinatorMentionsOnly: boolean | undefined
  private coordinatorLastRead = 0
  private coordinatorLastPostSeq: number | undefined
  private coordinatorLastPostTime: string | undefined
  private headSeq = 0
  private overviewState: SwarmOverviewSnapshot | null = null
  private readonly channels = new Map<string, SwarmChannelRecord>()
  private readonly members = new Map<string, SwarmMemberRecord>()
  private readonly messages: SwarmBoardMessage[] = []
  private readonly memberHistories = new Map<string, ChatMessage[]>()
  /** Bounded LRU cache for killed worker histories lazily loaded on `inspectMember`. */
  private readonly evictedInspectCache = new Map<string, ChatMessage[]>()
  private readonly memberActivities = new Map<string, MutableWorkerActivity>()

  private readonly activeWorkerTurns = new Map<string, AbortController>()
  private readonly runningWorkerLoops = new Map<string, Promise<void>>()
  private readonly pendingWorkerJobs = new Map<string, PendingWorkerJob[]>()
  private readonly explicitParkInTurn = new Set<string>()
  private initialized = false
  private disposed = false

  constructor(options: SwarmManagerOptions) {
    if (!options.sessionId.trim()) {
      throw new ValidationError('sessionId', 'must be a non-empty string')
    }
    this.sessionId = options.sessionId.trim()
    this.workspaceDir = resolve(options.workspaceDir ?? process.cwd())
    this.storageDir = resolve(
      options.storageDir ?? defaultSwarmStorageDirectory(this.workspaceDir, this.sessionId),
    )
    this.nowClock = options.now ?? (() => new Date())
    this.boardStore = new SwarmBoardStore(this.storageDir)
    this.workspacePort = options.workspacePort ?? createDefaultSwarmWorkspacePort(
      options.workspaceRootDir === undefined ? {} : { rootDir: options.workspaceRootDir },
    )
    this.modelCatalog = options.modelCatalog
    this.forkMessagesPort = options.forkMessages
    this.workerRunner = options.workerRunner
    this.onCoordinatorModelSwitchHook = options.onCoordinatorModelSwitch

    this.coordinatorModelId = options.coordinatorModel?.trim() || 'default'
    this.coordinatorModelDisplay = options.coordinatorModelDisplayName?.trim() || this.coordinatorModelId

    const main = createMainChannel(this.nowIso())
    this.channels.set(MAIN_CHANNEL, main)

    if (options.onEvent) {
      this.eventListeners.add(options.onEvent)
    }
    if (options.onCoordinatorWake) {
      this.coordinatorWakeListeners.add(options.onCoordinatorWake)
    }
  }

  static async open(options: SwarmManagerOptions): Promise<SwarmManager> {
    const manager = new SwarmManager(options)
    await manager.initialize()
    return manager
  }

  async initialize(): Promise<void> {
    if (this.initialized) return
    const loaded = await this.boardStore.load({
      sessionId: this.sessionId,
      workspaceDir: this.workspaceDir,
      coordinatorModel: this.coordinatorModelId,
      coordinatorModelDisplayName: this.coordinatorModelDisplay,
      nowIso: this.nowIso(),
      includeInactiveHistories: false,
    })
    if (loaded) {
      this.swarmTitle = loaded.state.title
      this.coordinatorActivated = loaded.state.coordinator_activated
      this.coordinatorModelId = loaded.state.coordinator_model
      this.coordinatorModelDisplay = loaded.state.coordinator_model_display_name
      this.coordinatorLastRead = loaded.state.coordinator_last_read_seq
      this.coordinatorLastPostSeq = loaded.state.coordinator_last_post_seq
      this.coordinatorLastPostTime = loaded.state.coordinator_last_post_time
      this.headSeq = loaded.state.head
      this.overviewState = loaded.state.overview

      this.channels.clear()
      for (const ch of loaded.state.channels) {
        this.channels.set(ch.name, ch)
      }
      if (!this.channels.has(MAIN_CHANNEL)) {
        this.channels.set(MAIN_CHANNEL, createMainChannel(this.nowIso()))
      }

      this.members.clear()
      for (const member of loaded.state.members) {
        this.members.set(member.name, { ...member, channels: [...member.channels] })
      }

      this.messages.length = 0
      this.messages.push(...loaded.messages)

      this.memberHistories.clear()
      this.evictedInspectCache.clear()
      for (const [name, history] of loaded.histories.entries()) {
        if (!this.members.get(name)?.killed) {
          this.memberHistories.set(name, history)
        }
      }
    }
    this.initialized = true
  }

  setWorkerRunner(runner: SwarmWorkerRunnerPort | undefined): void {
    this.workerRunner = runner
  }

  setCoordinatorModel(modelId: string, displayName?: string): void {
    if (!modelId.trim()) return
    this.coordinatorModelId = modelId.trim()
    this.coordinatorModelDisplay = displayName?.trim() || this.coordinatorModelId
  }

  setCoordinatorState(
    state: 'working' | 'waiting',
    park?: { readonly parked_since?: number; readonly mentions_only?: boolean },
  ): void {
    const nextParkedSince = state === 'waiting' ? park?.parked_since : undefined
    const nextMentionsOnly = state === 'waiting' ? park?.mentions_only : undefined
    const changed = this.coordinatorState !== state
      || this.coordinatorParkedSince !== nextParkedSince
      || this.coordinatorMentionsOnly !== nextMentionsOnly
    this.coordinatorState = state
    this.coordinatorParkedSince = nextParkedSince
    this.coordinatorMentionsOnly = nextMentionsOnly
    if (changed && this.isActive()) {
      this.emitLifecycleEvent('member_updated', { member: this.buildCoordinatorSnapshot() })
    }
  }

  coordinatorModel(): string {
    return this.coordinatorModelId
  }

  coordinatorLastReadSeq(): number {
    return this.coordinatorLastRead
  }

  head(): number {
    return this.headSeq
  }

  /**
   * True once the swarm has at least one worker on the roster (excluding `coordinator`).
   * Used to unlock `post_message`, `read_board`, and `swarm_wait` for `coordinator` after the first `add`.
   */
  isActive(): boolean {
    return this.members.size > 0
  }

  /** True while at least one non-killed worker remains on the roster. */
  hasLivingWorkers(): boolean {
    for (const member of this.members.values()) {
      if (!member.killed) return true
    }
    return false
  }

  /** Number of living worker conversation histories currently pinned in RAM. */
  inMemoryHistoryCount(): number {
    return this.memberHistories.size
  }

  isCoordinatorActivated(): boolean {
    return this.coordinatorActivated
  }

  getMember(name: string): SwarmMemberSnapshot | undefined {
    const record = this.members.get(name)
    return record ? this.buildMemberSnapshot(record) : undefined
  }

  getMemberHistory(name: string): ChatMessage[] {
    const history = this.resolveMemberHistorySync(name)
    return history ? structuredClone(history) : []
  }

  recordWorkerStreamEvent(memberName: string, event: StreamEvent): void {
    const record = this.members.get(memberName)
    if (!record || record.killed) return

    const activity = this.getOrCreateWorkerActivity(memberName)
    const nowDate = this.nowClock()
    const nowMs = nowDate.getTime()
    const nowIso = nowDate.toISOString()

    let emitMode: 'none' | 'throttled' | 'immediate' = 'none'

    switch (event.type) {
      case 'provider_wait': {
        if (event.active && activity.activeTools.length === 0) {
          activity.phase = 'thinking'
          activity.summary = 'Thinking…'
          activity.updatedAt = nowIso
          emitMode = 'throttled'
        }
        break
      }
      case 'compaction': {
        if (event.active && activity.activeTools.length === 0) {
          activity.phase = 'thinking'
          activity.summary = 'Compacting context…'
          activity.updatedAt = nowIso
          emitMode = 'throttled'
        }
        break
      }
      case 'thinking': {
        if (activity.activeTools.length === 0) {
          activity.phase = 'thinking'
          activity.summary = 'Thinking…'
          activity.updatedAt = nowIso
          emitMode = 'throttled'
        }
        break
      }
      case 'text': {
        if (event.text) {
          const combined = activity.liveText + event.text
          activity.liveText = combined.length > MAX_LIVE_TEXT_PREVIEW_CHARS
            ? combined.slice(-MAX_LIVE_TEXT_PREVIEW_CHARS)
            : combined
        }
        if (activity.activeTools.length === 0) {
          activity.phase = 'responding'
          activity.summary = 'Responding…'
          activity.updatedAt = nowIso
          emitMode = 'throttled'
        }
        break
      }
      case 'tool_start': {
        const info = summarizeToolCall(event.call.function.name, event.call.function.arguments)
        const runningEntry: SwarmWorkerToolActivity = {
          id: event.call.id,
          name: event.call.function.name,
          ...(info.summary === undefined ? {} : { summary: info.summary }),
          status: 'running',
          started_at: nowIso,
        }
        const existingIdx = activity.activeTools.findIndex(item => item.id === event.call.id)
        if (existingIdx >= 0) {
          activity.activeTools[existingIdx] = runningEntry
        } else {
          activity.activeTools.push(runningEntry)
        }
        activity.phase = 'tool'
        activity.summary = info.label
        activity.updatedAt = nowIso
        emitMode = 'immediate'
        break
      }
      case 'tool_end': {
        const existingIdx = activity.activeTools.findIndex(item => item.id === event.result.toolCallId)
        const previous = existingIdx >= 0 ? activity.activeTools.splice(existingIdx, 1)[0] : undefined
        const preview = formatToolResultPreview(event.result.result, MAX_TOOL_RESULT_PREVIEW_CHARS)
        const completedEntry: SwarmWorkerToolActivity = {
          id: event.result.toolCallId,
          name: event.result.name,
          ...(previous?.summary === undefined ? {} : { summary: previous.summary }),
          status: isToolResultError(event.result) ? 'error' : 'done',
          started_at: previous?.started_at ?? nowIso,
          finished_at: nowIso,
          ...(preview === undefined ? {} : { result_preview: preview }),
        }
        activity.recentTools.push(completedEntry)
        if (activity.recentTools.length > MAX_RECENT_TOOLS) {
          activity.recentTools.splice(0, activity.recentTools.length - MAX_RECENT_TOOLS)
        }
        const latestActive = activity.activeTools.at(-1)
        if (latestActive) {
          activity.phase = 'tool'
          activity.summary = formatRunningToolLabel(latestActive.name, latestActive.summary)
        } else {
          activity.phase = 'thinking'
          activity.summary = 'Thinking…'
        }
        activity.updatedAt = nowIso
        emitMode = 'immediate'
        break
      }
      case 'provider_retry': {
        activity.phase = 'thinking'
        activity.summary = event.final ? 'Provider failed' : `Retrying provider (${event.attempt})…`
        activity.updatedAt = nowIso
        emitMode = 'immediate'
        break
      }
      case 'turn_done': {
        activity.phase = 'idle'
        activity.summary = undefined
        activity.activeTools = []
        activity.liveText = ''
        activity.updatedAt = nowIso
        emitMode = 'immediate'
        break
      }
      default:
        break
    }

    if (emitMode === 'none') return
    const signature = computeActivitySignature(activity)
    if (signature === activity.lastEmittedSignature) {
      return
    }
    if (emitMode === 'throttled' && activity.lastEmitMs > 0) {
      const throttleWindow = activity.phase === activity.lastEmittedPhase
        ? ACTIVITY_SAME_PHASE_THROTTLE_MS
        : ACTIVITY_PHASE_TRANSITION_THROTTLE_MS
      if (nowMs - activity.lastEmitMs < throttleWindow) {
        return
      }
    }
    activity.lastEmitMs = nowMs
    activity.lastEmittedPhase = activity.phase
    activity.lastEmittedSignature = signature
    this.emitLifecycleEvent('member_activity', {
      member: this.buildMemberSnapshot(record),
      activity: this.snapshotMemberActivity(memberName),
    })
  }

  inspectMember(
    name: string,
    options?: { readonly limit?: number },
  ): SwarmMemberInspectResult {
    const cleanName = name.trim()
    if (!cleanName) {
      throw new ValidationError('name', 'member name must not be empty')
    }
    const requestedLimit = options?.limit
    const limit = typeof requestedLimit === 'number' && Number.isFinite(requestedLimit)
      ? Math.min(MAX_INSPECT_LIMIT, Math.max(1, Math.trunc(requestedLimit)))
      : DEFAULT_INSPECT_LIMIT

    if (cleanName === COORDINATOR_NAME) {
      const coordMember = this.buildCoordinatorSnapshot()
      return {
        session_id: this.sessionId,
        member: coordMember,
        conversation_id: this.memberConversationId(COORDINATOR_NAME),
        activity: {
          phase: this.coordinatorState === 'working' ? 'responding' : 'idle',
          ...(this.coordinatorState === 'working' ? { summary: 'Coordinating…' } : {}),
          active_tools: [],
          recent_tools: [],
        },
        messages: [],
        total_messages: 0,
      }
    }

    const record = this.members.get(cleanName)
    if (!record) {
      throw new ValidationError('name', `unknown swarm member "${cleanName}"`, cleanName)
    }

    const history = this.resolveMemberHistorySync(record.name) ?? []
    const visibleHistory = history.filter(msg => msg.role !== 'system')
    const startOffset = Math.max(0, visibleHistory.length - limit)
    const slice = visibleHistory.slice(startOffset)
    const inspectMessages: SwarmInspectMessage[] = slice.map((msg, idx) =>
      toInspectMessage(msg, startOffset + idx),
    )

    return {
      session_id: this.sessionId,
      member: this.buildMemberSnapshot(record),
      conversation_id: this.memberConversationId(record.name),
      activity: this.snapshotMemberActivity(record.name),
      messages: inspectMessages,
      total_messages: visibleHistory.length,
    }
  }

  buildWorkerPromptOverlay(name: string): string {
    const record = this.members.get(name)
    if (!record) {
      throw new ValidationError('name', `unknown swarm member "${name}"`, name)
    }
    return buildSwarmWorkerPromptOverlay(this.buildMemberSnapshot(record))
  }

  buildCoordinatorPromptOverlay(): string {
    return buildSwarmCoordinatorPromptOverlay({
      sessionId: this.sessionId,
      ...(this.swarmTitle ? { title: this.swarmTitle } : {}),
      active: this.isActive(),
      head: this.headSeq,
      lastReadSeq: this.coordinatorLastRead,
    })
  }

  status(): SwarmStatusSnapshot {
    const channelSummaries = computeChannelSummaries(this.channels.values(), this.members.values())
    const memberSnapshots: SwarmMemberSnapshot[] = [
      this.buildCoordinatorSnapshot(),
      ...[...this.members.values()].map(record => this.buildMemberSnapshot(record)),
    ]
    return {
      active: this.isActive(),
      session_id: this.sessionId,
      title: this.swarmTitle,
      board: 'local',
      open_board: SWARM_OPEN_BOARD_MARKDOWN,
      head: this.headSeq,
      tail_from_seq: 1,
      overview: this.overviewState ? { ...this.overviewState } : null,
      channels: channelSummaries,
      members: memberSnapshots,
    }
  }

  onEvent(listener: SwarmEventListener): () => void {
    this.eventListeners.add(listener)
    return () => {
      this.eventListeners.delete(listener)
    }
  }

  onCoordinatorWake(listener: (event: SwarmCoordinatorWakeEvent) => void): () => void {
    this.coordinatorWakeListeners.add(listener)
    return () => {
      this.coordinatorWakeListeners.delete(listener)
    }
  }

  async activate(enabled = true): Promise<SwarmActivateResult> {
    this.coordinatorActivated = enabled
    await this.boardStore.saveState(this.snapshotStateForPersistence())
    const overlay = this.buildCoordinatorPromptOverlay()
    this.emitLifecycleEvent('activated')
    return {
      ok: true,
      active: enabled,
      prompt_overlay: overlay,
    }
  }

  async setTitle(input: SwarmTitleInput): Promise<SwarmTitleResult> {
    const caller = this.normalizeCaller(input.caller)
    if (caller !== COORDINATOR_NAME) {
      throw new ValidationError('action', 'only coordinator may set the swarm title', caller)
    }
    const trimmed = input.title.trim()
    if (!trimmed) {
      throw new ValidationError('title', 'must be a non-empty string')
    }
    if (trimmed.length > 120) {
      throw new ValidationError('title', 'must be at most 120 characters', trimmed)
    }
    this.swarmTitle = trimmed
    await this.boardStore.saveState(this.snapshotStateForPersistence())
    this.emitLifecycleEvent('titled')
    return {
      status: 'titled',
      title: this.swarmTitle,
      open_board: SWARM_OPEN_BOARD_MARKDOWN,
    }
  }

  async listModels(options: { readonly all?: boolean } = {}): Promise<{
    readonly models: SwarmModelInfo[]
    readonly default_model: string
  }> {
    const all = options.all ?? false
    if (!this.modelCatalog) {
      return {
        models: [{ id: this.coordinatorModelId, display_name: this.coordinatorModelDisplay, suggested: true }],
        default_model: this.coordinatorModelId,
      }
    }
    const raw = await this.modelCatalog.listModels({ all })
    const filtered = !all && raw.some(item => item.suggested !== undefined)
      ? raw.filter(item => item.suggested !== false)
      : raw
    return {
      models: filtered.map(item => ({ ...item })),
      default_model: this.coordinatorModelId,
    }
  }

  async add(input: SwarmAddInput): Promise<SwarmAddResult> {
    const caller = this.normalizeCaller(input.caller)
    const callerMember = this.assertLivingCallerOrCoordinator(caller)

    const existingNames = new Set<string>()
    const existingGroups = new Set<string>()
    const killedNames = new Set<string>()
    for (const member of this.members.values()) {
      existingNames.add(member.name)
      if (member.group) existingGroups.add(member.group)
      if (member.killed) killedNames.add(member.name)
    }

    const planned = expandPlannedWorkers({
      baseName: input.name,
      brief: input.brief,
      ...(input.n === undefined ? {} : { n: input.n }),
      ...(input.tasks === undefined ? {} : { tasks: input.tasks }),
      existingNames,
      existingGroups,
      killedNames,
    })

    const defaultModelId = callerMember ? callerMember.model : this.coordinatorModelId
    const defaultModelDisplay = callerMember ? callerMember.model_display_name : this.coordinatorModelDisplay
    const resolvedModel = await this.resolveModelSelection(
      input.model ?? defaultModelId,
      input.model !== undefined,
      defaultModelDisplay,
    )

    let forkedTemplate: ChatMessage[] | undefined
    if (input.fork !== undefined) {
      const forkName = input.fork.trim()
      if (!forkName) {
        throw new ValidationError('fork', 'must name "coordinator" or an existing swarm member')
      }
      if (forkName !== COORDINATOR_NAME && !this.members.has(forkName)) {
        throw new ValidationError('fork', `unknown fork source "${forkName}"`, forkName)
      }
      const fromPort = await this.forkMessagesPort?.(forkName)
      let rawHistory: readonly ChatMessage[]
      if (fromPort !== undefined) {
        rawHistory = fromPort
      } else if (forkName === COORDINATOR_NAME) {
        rawHistory = []
      } else {
        rawHistory = this.memberHistories.get(forkName)
          ?? this.evictedInspectCache.get(forkName)
          ?? (await this.boardStore.loadMemberHistory(forkName))
          ?? []
      }
      forkedTemplate = sanitizeForkedMessages(rawHistory)
    }

    const explicitChannels = input.channels && input.channels.length > 0
      ? input.channels.map(ch => assertValidChannelName(ch, 'channels'))
      : undefined
    if (explicitChannels) {
      for (const channelName of explicitChannels) {
        const channelRecord = this.channels.get(channelName)
        if (!channelRecord) {
          throw new ValidationError('channels', `channel "#${channelName}" does not exist; create it first`, channelName)
        }
        if (channelRecord.archived) {
          throw new ValidationError('channels', `channel "#${channelName}" is archived`, channelName)
        }
      }
    }

    const inheritedChannels = callerMember
      ? callerMember.channels.filter(ch => ch !== WILDCARD_CHANNEL && !this.channels.get(ch)?.archived)
      : [MAIN_CHANNEL]
    const requestedChannels = explicitChannels ?? (inheritedChannels.length > 0 ? inheritedChannels : [MAIN_CHANNEL])
    const briefChannel = requestedChannels[0] ?? MAIN_CHANNEL
    const briefChannelRecord = this.channels.get(briefChannel)
    if (!briefChannelRecord || briefChannelRecord.archived) {
      throw new ValidationError('channels', `cannot post startup brief to unavailable channel "#${briefChannel}"`, briefChannel)
    }

    const workspaceMode = input.workspace ?? 'inherit'
    const baseWorkspaceDir = callerMember ? callerMember.workspace_path : this.workspaceDir

    const allocations: SwarmWorkspaceAllocation[] = []
    try {
      for (const slice of planned) {
        const allocation = await this.workspacePort.createWorkspace({
          mode: workspaceMode,
          baseDir: baseWorkspaceDir,
          memberName: slice.name,
          sessionId: this.sessionId,
        })
        allocations.push(allocation)
      }
    } catch (error) {
      // Roll back any already-allocated workspaces if a multi-worker batch fails midway.
      await Promise.all(allocations.map((allocation, idx) => {
        const sliceName = planned[idx]?.name ?? 'unknown'
        return this.workspacePort.cleanupWorkspace({
          mode: allocation.mode,
          baseDir: baseWorkspaceDir,
          workspacePath: allocation.path,
          memberName: sliceName,
          sessionId: this.sessionId,
          ...(allocation.handle_id === undefined ? {} : { handleId: allocation.handle_id }),
        }).catch(() => undefined)
      }))
      throw error
    }

    const createdRecords: SwarmMemberRecord[] = []
    const briefMessages: SwarmBoardMessage[] = []
    const time = this.nowIso()

    for (const [index, slice] of planned.entries()) {
      const allocation = allocations[index]!
      const role = roleForMemberName(slice.name)
      const memberChannels = roleHasWildcardChannels(role)
        ? defaultChannelsForRole(role)
        : normalizeWorkerChannelList(role, requestedChannels)

      const record: SwarmMemberRecord = {
        name: slice.name,
        ...(slice.group === undefined ? {} : { group: slice.group }),
        role,
        added_by: caller,
        model: resolvedModel.id,
        model_display_name: resolvedModel.display_name,
        workspace: workspaceMode,
        workspace_path: allocation.path,
        ...(allocation.handle_id === undefined ? {} : { workspace_handle_id: allocation.handle_id }),
        ...(allocation.base_head === undefined ? {} : { workspace_base_head: allocation.base_head }),
        channels: memberChannels,
        state: this.workerRunner ? 'working' : 'waiting',
        killed: false,
        last_read_seq: this.headSeq,
      }
      this.members.set(record.name, record)
      createdRecords.push(record)

      const initialHistory = forkedTemplate ? structuredClone(forkedTemplate) : []
      this.memberHistories.set(record.name, initialHistory)

      const briefText = `@${slice.name} ${slice.effectiveBrief}`
      this.headSeq += 1
      const briefMsg: SwarmBoardMessage = {
        seq: this.headSeq,
        agent: caller,
        channel: briefChannel,
        text: briefText,
        mentions: [slice.name],
        channel_mentions: [],
        broadcast: false,
        time,
        kind: 'brief',
      }
      this.messages.push(briefMsg)
      briefMessages.push(briefMsg)
    }

    if (callerMember) {
      callerMember.last_post_seq = this.headSeq
      callerMember.last_post_time = time
    } else if (caller === COORDINATOR_NAME) {
      this.coordinatorLastRead = Math.max(this.coordinatorLastRead, this.headSeq)
      this.coordinatorLastPostSeq = this.headSeq
      this.coordinatorLastPostTime = time
    }

    // Advance every newly added worker's cursor past the startup brief batch so
    // it receives its scoped startup brief directly without duplicate self-wakeups.
    const postBatchHead = this.headSeq
    for (const record of createdRecords) {
      record.last_read_seq = postBatchHead
      record.parked_since = postBatchHead
      record.mentions_only = record.role === 'narrator' ? false : true
    }

    const persistedState = this.snapshotStateForPersistence()
    for (const record of createdRecords) {
      await this.boardStore.saveMemberHistory(record.name, this.memberHistories.get(record.name) ?? [])
    }
    await this.boardStore.appendMessagesAndSaveState(briefMessages, persistedState)

    for (const record of createdRecords) {
      this.emitLifecycleEvent('member_added', { member: this.buildMemberSnapshot(record) })
    }
    for (const briefMsg of briefMessages) {
      this.emitLifecycleEvent('message_posted', { message: briefMsg })
      // Notify any other existing observers (e.g., narrator watching mentions_only:false)
      this.routeWakeupsForMessage(briefMsg, new Set(createdRecords.map(item => item.name)))
    }

    // Immediately dispatch each new worker's initial background turn!
    for (const [index, record] of createdRecords.entries()) {
      const slice = planned[index]!
      const briefMsg = briefMessages[index]!
      const startupPrompt = buildWorkerStartupPrompt({
        memberName: record.name,
        briefSeq: briefMsg.seq,
        channel: briefMsg.channel,
        effectiveBrief: slice.effectiveBrief,
      })
      this.scheduleWorkerJob(record, { reason: 'startup', prompt: startupPrompt })
    }

    return {
      status: 'added',
      added: createdRecords.map(record => this.buildMemberSnapshot(record)),
      brief_seqs: briefMessages.map(msg => msg.seq),
      channel: briefChannel,
      head: this.headSeq,
      open_board: SWARM_OPEN_BOARD_MARKDOWN,
      note: `Include open_board on its own line when reporting to the user: ${SWARM_OPEN_BOARD_MARKDOWN}`,
    }
  }

  async switchModel(input: SwarmSwitchModelInput): Promise<SwarmSwitchModelResult> {
    const caller = this.normalizeCaller(input.caller)
    const callerMember = this.assertLivingCallerOrCoordinator(caller)
    const targetName = input.name?.trim() || caller

    if (targetName === COORDINATOR_NAME) {
      if (caller !== COORDINATOR_NAME) {
        throw new ValidationError('name', 'workers may not switch the coordinator model', targetName)
      }
      if (!input.model?.trim()) {
        throw new ValidationError('model', 'specify a target model from swarm{action:"models"} when switching yourself')
      }
      const resolved = await this.resolveModelSelection(input.model, true)
      const previousModel = this.coordinatorModelId
      this.coordinatorModelId = resolved.id
      this.coordinatorModelDisplay = resolved.display_name
      await this.onCoordinatorModelSwitchHook?.(resolved)
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      this.emitLifecycleEvent('member_updated', { member: this.buildCoordinatorSnapshot() })
      const trimmedReason = input.reason?.trim()
      return {
        status: 'switched',
        name: COORDINATOR_NAME,
        previous_model: previousModel,
        model: resolved.id,
        model_display_name: resolved.display_name,
        resumed: false,
        ...(trimmedReason ? { reason: trimmedReason } : {}),
      }
    }

    const target = this.members.get(targetName)
    if (!target) {
      throw new ValidationError('name', `unknown swarm member "${targetName}"`, targetName)
    }
    if (target.killed) {
      throw new ValidationError('name', `worker "${target.name}" was killed and cannot be switched`, targetName)
    }
    if (targetName !== caller && !canManageWorker(caller, target)) {
      throw new ValidationError(
        'name',
        `worker "${caller}" may only switch its own model or helpers it added ("${target.name}" was added by "${target.added_by ?? COORDINATOR_NAME}")`,
        targetName,
      )
    }
    if (targetName !== caller && target.state === 'working') {
      throw new ValidationError(
        'name',
        `worker "${target.name}" is currently working; wait until it is waiting or stopped before switching its model`,
        targetName,
      )
    }

    let requestedModel = input.model?.trim()
    let explicitModel = requestedModel !== undefined && requestedModel.length > 0
    if (!explicitModel) {
      if (targetName === caller) {
        throw new ValidationError(
          'model',
          'specify a target model from swarm{action:"models"} when switching yourself',
        )
      }
      requestedModel = callerMember ? callerMember.model : this.coordinatorModelId
      explicitModel = false
    }

    const fallbackDisplay = callerMember ? callerMember.model_display_name : this.coordinatorModelDisplay
    const resolved = await this.resolveModelSelection(requestedModel!, explicitModel, fallbackDisplay)
    const previousModel = target.model
    target.model = resolved.id
    target.model_display_name = resolved.display_name
    delete target.error

    const trimmedReason = input.reason?.trim()
    const shouldResume = targetName !== caller
    if (shouldResume && this.workerRunner) {
      target.state = 'working'
    }
    await this.boardStore.saveState(this.snapshotStateForPersistence())
    this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(target) })

    if (shouldResume) {
      const resumePrompt = buildWorkerModelSwitchPrompt({
        previousModel,
        nextModel: resolved.display_name,
        lastReadSeq: target.last_read_seq,
        ...(trimmedReason ? { reason: trimmedReason } : {}),
      })
      this.scheduleWorkerJob(target, { reason: 'model_switch', prompt: resumePrompt })
    }

    return {
      status: 'switched',
      name: target.name,
      previous_model: previousModel,
      model: resolved.id,
      model_display_name: resolved.display_name,
      resumed: shouldResume && Boolean(this.workerRunner),
      ...(trimmedReason ? { reason: trimmedReason } : {}),
    }
  }

  async killFromHost(nameOrAll: string): Promise<SwarmKillResult> {
    const clean = nameOrAll.trim()
    return this.kill({
      caller: COORDINATOR_NAME,
      name: clean,
      allowAll: clean === 'all',
    })
  }

  async kill(input: SwarmKillInput): Promise<SwarmKillResult> {
    const caller = this.normalizeCaller(input.caller)
    this.assertLivingCallerOrCoordinator(caller)
    const targetKey = input.name.trim()
    if (!targetKey) {
      throw new ValidationError('name', 'must specify a worker or group name to kill')
    }
    if (targetKey === COORDINATOR_NAME || targetKey.startsWith(`${COORDINATOR_NAME}-`)) {
      throw new ValidationError('name', 'cannot kill coordinator', targetKey)
    }

    let targets: SwarmMemberRecord[]
    if (targetKey === 'all') {
      if (!input.allowAll) {
        throw new ValidationError('name', '"all" is reserved; specify the worker or group name to kill', targetKey)
      }
      targets = [...this.members.values()].filter(member => !member.killed)
    } else {
      targets = resolveTargetMembers(this.members, targetKey)
      if (targets.length === 0) {
        const allIncludingKilled = resolveTargetMembers(this.members, targetKey, { includeKilled: true })
        if (allIncludingKilled.length > 0) {
          return { status: 'killed', killed: [] }
        }
        throw new ValidationError('name', `unknown swarm member or group "${targetKey}"`, targetKey)
      }
      for (const member of targets) {
        if (!canManageWorker(caller, member)) {
          throw new ValidationError(
            'name',
            `worker "${caller}" may only kill helpers it added ("${member.name}" was added by "${member.added_by ?? COORDINATOR_NAME}")`,
            member.name,
          )
        }
      }
    }

    const killedNames: string[] = []
    for (const member of targets) {
      if (member.killed) continue
      member.killed = true
      member.state = 'killed'
      delete member.parked_since
      delete member.mentions_only
      this.pendingWorkerJobs.delete(member.name)
      this.resetWorkerActivityToIdle(member.name, false)

      const turnController = this.activeWorkerTurns.get(member.name)
      if (turnController && !turnController.signal.aborted) {
        turnController.abort(new SwarmKillSignal(member.name))
      }

      const activeLoop = this.runningWorkerLoops.get(member.name)
      if (activeLoop && caller !== member.name) {
        await activeLoop.catch(() => undefined)
      }

      // Persist any in-memory history before evicting the killed worker from RAM.
      // `inspectMember()` lazy-loads killed histories from disk into a bounded LRU.
      const remainingHistory = this.memberHistories.get(member.name)
      if (remainingHistory !== undefined) {
        await this.boardStore.saveMemberHistory(member.name, remainingHistory)
        this.memberHistories.delete(member.name)
      }
      this.evictedInspectCache.delete(member.name)
      this.memberActivities.delete(member.name)

      const parentRecord = member.added_by && member.added_by !== COORDINATOR_NAME
        ? this.members.get(member.added_by)
        : undefined
      const baseDir = parentRecord ? parentRecord.workspace_path : this.workspaceDir
      await this.workspacePort.cleanupWorkspace({
        mode: member.workspace,
        baseDir,
        workspacePath: member.workspace_path,
        memberName: member.name,
        sessionId: this.sessionId,
        ...(member.workspace_handle_id === undefined ? {} : { handleId: member.workspace_handle_id }),
      })

      killedNames.push(member.name)
    }

    if (killedNames.length > 0) {
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      for (const name of killedNames) {
        const record = this.members.get(name)
        this.emitLifecycleEvent('member_killed', record ? { member: this.buildMemberSnapshot(record) } : {})
      }
      triggerSoftGc()
    }

    return {
      status: 'killed',
      killed: killedNames,
    }
  }

  async channelAction(input: SwarmChannelActionInput): Promise<SwarmChannelActionResult> {
    const caller = this.normalizeCaller(input.caller)
    const callerMember = this.assertLivingCallerOrCoordinator(caller)

    switch (input.op) {
      case 'list': {
        return {
          channels: computeChannelSummaries(this.channels.values(), this.members.values()),
        }
      }
      case 'create': {
        if (!input.channel) {
          throw new ValidationError('channel', 'channel name is required for channel create')
        }
        const channelName = assertValidChannelName(input.channel, 'channel')
        if (this.channels.has(channelName)) {
          throw new ValidationError('channel', `channel "#${channelName}" already exists`, channelName)
        }
        const record: SwarmChannelRecord = {
          name: channelName,
          topic: input.topic?.trim() ?? '',
          created_by: caller,
          created_at: this.nowIso(),
          archived: false,
        }
        this.channels.set(channelName, record)
        if (callerMember && !roleHasWildcardChannels(callerMember.role) && !callerMember.channels.includes(channelName)) {
          callerMember.channels.push(channelName)
        }
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        const summary = this.findChannelSummary(channelName)!
        this.emitLifecycleEvent('channel_updated', { channel: summary })
        return {
          status: 'created',
          channel: summary,
        }
      }
      case 'subscribe': {
        if (!input.channel) {
          throw new ValidationError('channel', 'channel name is required for subscribe')
        }
        const channelName = assertValidChannelName(input.channel, 'channel')
        const channelRecord = this.channels.get(channelName)
        if (!channelRecord) {
          throw new ValidationError('channel', `channel "#${channelName}" does not exist`, channelName)
        }
        if (channelRecord.archived) {
          throw new ValidationError('channel', `channel "#${channelName}" is archived`, channelName)
        }

        const targetName = input.name?.trim() || caller
        if (targetName === COORDINATOR_NAME) {
          return { status: 'subscribed', channel: channelName, members: [COORDINATOR_NAME] }
        }
        const targets = resolveTargetMembers(this.members, targetName)
        if (targets.length === 0) {
          throw new ValidationError('name', `unknown living swarm member or group "${targetName}"`, targetName)
        }
        for (const target of targets) {
          if (target.killed) {
            throw new ValidationError('name', `worker "${target.name}" was killed`, target.name)
          }
          if (target.name !== caller && !canManageWorker(caller, target)) {
            throw new ValidationError(
              'name',
              `worker "${caller}" may only subscribe itself or helpers it added`,
              target.name,
            )
          }
        }
        const updatedMembers: string[] = []
        for (const target of targets) {
          if (!roleHasWildcardChannels(target.role) && !target.channels.includes(channelName)) {
            target.channels.push(channelName)
          }
          updatedMembers.push(target.name)
        }
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        const summary = this.findChannelSummary(channelName)!
        this.emitLifecycleEvent('channel_updated', { channel: summary })
        return {
          status: 'subscribed',
          channel: channelName,
          members: updatedMembers,
        }
      }
      case 'unsubscribe': {
        if (!input.channel) {
          throw new ValidationError('channel', 'channel name is required for unsubscribe')
        }
        const channelName = assertValidChannelName(input.channel, 'channel')
        if (channelName === MAIN_CHANNEL) {
          throw new ValidationError('channel', 'cannot unsubscribe from #main', channelName)
        }
        if (!this.channels.has(channelName)) {
          throw new ValidationError('channel', `channel "#${channelName}" does not exist`, channelName)
        }

        const targetName = input.name?.trim() || caller
        if (targetName === COORDINATOR_NAME) {
          throw new ValidationError('name', 'coordinator observes all channels and cannot unsubscribe', targetName)
        }
        const targets = resolveTargetMembers(this.members, targetName)
        if (targets.length === 0) {
          throw new ValidationError('name', `unknown living swarm member or group "${targetName}"`, targetName)
        }
        for (const target of targets) {
          if (target.name !== caller && !canManageWorker(caller, target)) {
            throw new ValidationError(
              'name',
              `worker "${caller}" may only unsubscribe itself or helpers it added`,
              target.name,
            )
          }
        }
        const updatedMembers: string[] = []
        for (const target of targets) {
          if (!roleHasWildcardChannels(target.role)) {
            const idx = target.channels.indexOf(channelName)
            if (idx >= 0) target.channels.splice(idx, 1)
          }
          updatedMembers.push(target.name)
        }
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        const summary = this.findChannelSummary(channelName)!
        this.emitLifecycleEvent('channel_updated', { channel: summary })
        return {
          status: 'unsubscribed',
          channel: channelName,
          members: updatedMembers,
        }
      }
      case 'archive': {
        if (!input.channel) {
          throw new ValidationError('channel', 'channel name is required for archive')
        }
        const channelName = assertValidChannelName(input.channel, 'channel')
        if (channelName === MAIN_CHANNEL) {
          throw new ValidationError('channel', 'cannot archive #main', channelName)
        }
        const existing = this.channels.get(channelName)
        if (!existing) {
          throw new ValidationError('channel', `channel "#${channelName}" does not exist`, channelName)
        }
        if (caller !== COORDINATOR_NAME && existing.created_by !== caller) {
          throw new ValidationError(
            'channel',
            `only coordinator or channel creator "${existing.created_by}" may archive "#${channelName}"`,
            channelName,
          )
        }
        const archivedRecord: SwarmChannelRecord = {
          ...existing,
          archived: true,
        }
        this.channels.set(channelName, archivedRecord)
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        const summary = this.findChannelSummary(channelName)!
        this.emitLifecycleEvent('channel_updated', { channel: summary })
        return {
          status: 'archived',
          channel: summary,
        }
      }
    }
  }

  async postFromUser(input: SwarmUserPostInput): Promise<SwarmPostMessageResult> {
    if (input.as_feedback) {
      const knownNames = new Set<string>(this.members.keys())
      for (const member of this.members.values()) {
        if (member.group) knownNames.add(member.group)
      }
      const formatted = formatNarratorFeedbackText(input.text, knownNames)
      return this.postMessage({
        caller: USER_BOARD_AUTHOR,
        text: formatted,
        ...(input.channel === undefined ? {} : { channel: input.channel }),
        kind: 'feedback',
      })
    }
    return this.postMessage({
      caller: USER_BOARD_AUTHOR,
      text: input.text,
      ...(input.channel === undefined ? {} : { channel: input.channel }),
      kind: 'message',
    })
  }

  async postMessage(input: SwarmPostMessageInput): Promise<SwarmPostMessageResult> {
    const caller = input.caller?.trim() || COORDINATOR_NAME
    let callerMember: SwarmMemberRecord | undefined
    if (caller !== COORDINATOR_NAME && caller !== USER_BOARD_AUTHOR) {
      callerMember = this.assertLivingCallerOrCoordinator(caller)
    }

    const trimmedText = input.text.trim()
    if (!trimmedText) {
      throw new ValidationError('text', 'message text must not be empty')
    }

    const channelName = input.channel ? assertValidChannelName(input.channel, 'channel') : MAIN_CHANNEL
    const channelRecord = this.channels.get(channelName)
    if (!channelRecord) {
      throw new ValidationError('channel', `channel "#${channelName}" does not exist`, channelName)
    }
    if (channelRecord.archived) {
      throw new ValidationError('channel', `channel "#${channelName}" is archived and rejects new posts`, channelName)
    }
    if (callerMember && !memberCanSeeChannel(callerMember.role, callerMember.channels, channelName)) {
      throw new ValidationError(
        'channel',
        `worker "${callerMember.name}" is not subscribed to "#${channelName}"`,
        channelName,
      )
    }

    const parsed = parseBoardPostText(trimmedText)
    for (const mentionedChannel of parsed.channel_mentions) {
      const targetChannel = this.channels.get(mentionedChannel)
      if (!targetChannel) {
        throw new ValidationError('text', `referenced channel "@#${mentionedChannel}" does not exist`, mentionedChannel)
      }
      if (targetChannel.archived) {
        throw new ValidationError('text', `referenced channel "@#${mentionedChannel}" is archived`, mentionedChannel)
      }
    }

    const time = this.nowIso()
    this.headSeq += 1
    const message: SwarmBoardMessage = {
      seq: this.headSeq,
      agent: caller,
      channel: channelName,
      text: trimmedText,
      mentions: parsed.mentions,
      channel_mentions: parsed.channel_mentions,
      broadcast: parsed.broadcast,
      time,
      ...(input.kind === undefined ? {} : { kind: input.kind }),
    }
    this.messages.push(message)

    if (callerMember) {
      callerMember.last_post_seq = message.seq
      callerMember.last_post_time = time
    } else if (caller === COORDINATOR_NAME) {
      this.coordinatorLastPostSeq = message.seq
      this.coordinatorLastPostTime = time
      // Coordinator knows about its own outgoing post when nothing earlier is unread.
      if (this.coordinatorLastRead === message.seq - 1) {
        this.coordinatorLastRead = message.seq
      }
    }

    await this.boardStore.appendMessageAndSaveState(message, this.snapshotStateForPersistence())
    this.emitLifecycleEvent('message_posted', { message })
    const woken = this.routeWakeupsForMessage(message)

    return {
      status: 'posted',
      seq: message.seq,
      channel: channelName,
      mentions: [...message.mentions],
      channel_mentions: [...message.channel_mentions],
      broadcast: message.broadcast,
      woken,
      message,
    }
  }

  async readBoard(input: SwarmReadBoardInput = {}): Promise<SwarmReadBoardResult> {
    const caller = input.caller?.trim() || COORDINATOR_NAME
    const callerMember = caller === COORDINATOR_NAME || caller === USER_BOARD_AUTHOR
      ? undefined
      : this.members.get(caller)
    if (caller !== COORDINATOR_NAME && caller !== USER_BOARD_AUTHOR && !callerMember) {
      throw new ValidationError('caller', `unknown swarm member "${caller}"`, caller)
    }

    if (input.status) {
      const bySeq = new Map(this.messages.map(msg => [msg.seq, msg]))
      const coordSnap = this.buildCoordinatorSnapshot()
      const coordLastMsg = this.coordinatorLastPostSeq !== undefined
        ? bySeq.get(this.coordinatorLastPostSeq)
        : undefined
      const canSeeCoordLast = coordLastMsg
        ? (!callerMember || memberCanSeeChannel(callerMember.role, callerMember.channels, coordLastMsg.channel))
        : false
      const coordEntry: SwarmRosterStatusEntry = {
        ...coordSnap,
        ...(coordLastMsg && canSeeCoordLast ? { last_post_preview: coordLastMsg.text.slice(0, 160) } : {}),
      }
      const workerEntries: SwarmRosterStatusEntry[] = [...this.members.values()].map((record) => {
        const snap = this.buildMemberSnapshot(record)
        const lastMsg = record.last_post_seq !== undefined ? bySeq.get(record.last_post_seq) : undefined
        const canSeeLast = lastMsg
          ? (!callerMember || memberCanSeeChannel(callerMember.role, callerMember.channels, lastMsg.channel))
          : false
        return {
          ...snap,
          ...(lastMsg && canSeeLast ? { last_post_preview: lastMsg.text.slice(0, 160) } : {}),
        }
      })
      return {
        head: this.headSeq,
        active: this.isActive(),
        title: this.swarmTitle,
        members: [coordEntry, ...workerEntries],
      }
    }

    let channelFilter: string | undefined
    if (input.channel !== undefined && input.channel !== WILDCARD_CHANNEL) {
      channelFilter = assertValidChannelName(input.channel, 'channel')
      if (!this.channels.has(channelFilter)) {
        throw new ValidationError('channel', `channel "#${channelFilter}" does not exist`, channelFilter)
      }
      if (callerMember && !memberCanSeeChannel(callerMember.role, callerMember.channels, channelFilter)) {
        throw new ValidationError(
          'channel',
          `worker "${callerMember.name}" is not subscribed to "#${channelFilter}"`,
          channelFilter,
        )
      }
    }

    const visibleMessages = this.messages.filter((msg) => {
      if (callerMember && !memberCanSeeChannel(callerMember.role, callerMember.channels, msg.channel)) {
        return false
      }
      if (channelFilter !== undefined && msg.channel !== channelFilter) {
        return false
      }
      return true
    })

    if (input.seqs !== undefined) {
      const requested = new Set(input.seqs)
      const selected = visibleMessages
        .filter(msg => requested.has(msg.seq))
        .map(cloneBoardMessage)
      return {
        head: this.headSeq,
        messages: selected,
      }
    }

    const limit = input.limit ?? 20
    if (!Number.isInteger(limit) || limit < 1 || limit > 500) {
      throw new ValidationError('limit', 'must be an integer between 1 and 500', limit)
    }

    let selected: SwarmBoardMessage[]
    if (input.since !== undefined) {
      if (!Number.isInteger(input.since) || input.since < 0) {
        throw new ValidationError('since', 'must be a non-negative integer', input.since)
      }
      selected = visibleMessages
        .filter(msg => msg.seq > input.since!)
        .slice(0, limit)
        .map(cloneBoardMessage)
    } else {
      selected = visibleMessages.slice(-limit).map(cloneBoardMessage)
    }

    const highestReturned = selected.at(-1)?.seq
    if (highestReturned !== undefined) {
      await this.advanceCallerReadSeq(caller, callerMember, highestReturned)
    } else if (input.since !== undefined && channelFilter === undefined) {
      // When no new visible messages exist and all channels visible to caller were scanned,
      // advance cursor up to head if caller had already read up to `since`.
      const currentCursor = callerMember ? callerMember.last_read_seq : this.coordinatorLastRead
      if (input.since <= currentCursor && this.headSeq > currentCursor) {
        await this.advanceCallerReadSeq(caller, callerMember, this.headSeq)
      }
    }

    return {
      head: this.headSeq,
      messages: selected,
    }
  }

  async waitForBoard(input: SwarmWaitInput): Promise<SwarmWaitResult> {
    const caller = this.normalizeCaller(input.caller)
    const since = input.since
    if (!Number.isInteger(since) || since < 0) {
      throw new ValidationError('since', 'must be a non-negative integer', since)
    }
    const mentionsOnly = input.mentions_only ?? true

    if (caller === COORDINATOR_NAME) {
      const immediate = this.messages.find(msg => doesMessageWakeCoordinator(msg, since, mentionsOnly))
      if (immediate) {
        return {
          status: 'woken',
          head: this.headSeq,
          last_read_seq: this.coordinatorLastRead,
          wake_seq: immediate.seq,
          message: formatBoardWakeNotice(this.headSeq, this.coordinatorLastRead),
        }
      }

      this.setCoordinatorState('waiting', { parked_since: since, mentions_only: mentionsOnly })
      try {
        const waited = await this.coordinatorWaitHub.waitForMessage({
          since,
          mentionsOnly,
          ...(input.signal === undefined ? {} : { signal: input.signal }),
          ...(input.wakeSignal === undefined ? {} : { wakeSignal: input.wakeSignal }),
        })
        if (waited.kind === 'worker_error') {
          return {
            status: 'worker_error',
            head: this.headSeq,
            last_read_seq: this.coordinatorLastRead,
            worker_error: waited.worker_error,
            message: formatWorkerErrorWakeNotice(waited.worker_error),
          }
        }
        if (waited.kind === 'steered') {
          return {
            status: 'interrupted_by_steer',
            head: this.headSeq,
            last_read_seq: this.coordinatorLastRead,
            message: formatBoardWakeNotice(this.headSeq, this.coordinatorLastRead),
          }
        }
        return {
          status: 'woken',
          head: this.headSeq,
          last_read_seq: this.coordinatorLastRead,
          wake_seq: waited.message.seq,
          message: formatBoardWakeNotice(this.headSeq, this.coordinatorLastRead),
        }
      } finally {
        this.setCoordinatorState('working')
      }
    }

    const member = this.assertLivingCallerOrCoordinator(caller)
    if (!member) {
      throw new ValidationError('caller', `unknown worker "${caller}"`, caller)
    }
    const allNames = new Set(this.members.keys())
    const immediate = this.messages.find(msg => doesMessageWakeWorker(member, msg, since, mentionsOnly, allNames))
    if (immediate) {
      return {
        status: 'woken',
        head: this.headSeq,
        last_read_seq: member.last_read_seq,
        wake_seq: immediate.seq,
        message: formatBoardWakeNotice(this.headSeq, member.last_read_seq),
      }
    }

    member.parked_since = since
    member.mentions_only = mentionsOnly
    this.explicitParkInTurn.add(member.name)

    const parkSignal = new SwarmParkSignal(member.name, since, mentionsOnly)
    const activeTurn = this.activeWorkerTurns.get(member.name)
    if (activeTurn) {
      if (!activeTurn.signal.aborted) {
        activeTurn.abort(parkSignal)
      }
    } else {
      member.state = 'waiting'
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
    }
    input.onWorkerPark?.(parkSignal)

    return {
      status: 'parked',
      head: this.headSeq,
      since,
      mentions_only: mentionsOnly,
    }
  }

  async updateOverview(input: SwarmOverviewUpdateInput): Promise<SwarmOverviewUpdateResult> {
    const caller = this.normalizeCaller(input.caller)
    if (!isNarratorName(caller)) {
      throw new ValidationError(
        'caller',
        'swarm_overview may only be called by a narrator member ("narrator" or "narrator-<N>")',
        caller,
      )
    }
    this.assertLivingCallerOrCoordinator(caller)
    const markdown = input.markdown.trim()
    if (!markdown) {
      throw new ValidationError('markdown', 'overview markdown must not be empty')
    }
    const summary = input.summary?.trim()
    const overview: SwarmOverviewSnapshot = {
      markdown,
      ...(summary ? { summary } : {}),
      updated_at: this.nowIso(),
      author: caller,
    }
    this.overviewState = overview
    await this.boardStore.saveState(this.snapshotStateForPersistence())
    this.emitLifecycleEvent('overview_updated')
    return {
      status: 'updated',
      overview: { ...overview },
      head: this.headSeq,
    }
  }

  /**
   * Wake a `'waiting'` or `'stopped'` non-killed worker directly (or enqueue into its
   * `PendingWorkerJob` queue when `'working'`) without posting any message to the shared board.
   */
  async wakeMember(
    memberName: string,
    prompt: string,
    reason: SwarmWorkerWakeReason = 'schedule',
  ): Promise<boolean> {
    const cleanName = memberName.trim()
    if (!cleanName || cleanName === COORDINATOR_NAME || this.disposed) {
      return false
    }
    const member = this.members.get(cleanName)
    if (!member || member.killed) {
      return false
    }
    const cleanPrompt = prompt.trim()
    if (!cleanPrompt) {
      throw new ValidationError('prompt', 'wake prompt must not be empty')
    }

    if (this.workerRunner) {
      if (member.state !== 'working') {
        member.state = 'working'
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
      }
      this.scheduleWorkerJob(member, { reason, prompt: cleanPrompt })
      return true
    }

    if (member.state === 'stopped') {
      member.state = 'waiting'
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
    }
    return true
  }

  /**
   * Inspect uncommitted and committed changes inside an isolated `'share'` or `'branch'`
   * candidate worker workspace relative to its base HEAD.
   */
  async inspectMemberPatch(
    name: string,
    options: { readonly targetDir?: string; readonly signal?: AbortSignal } = {},
  ): Promise<SwarmMemberPatchInspection> {
    const member = this.resolveCandidateMemberForPatch(name)
    const parentBaseDir = this.resolveMemberBaseDir(member)
    const targetDir = resolve(options.targetDir ?? parentBaseDir)
    const inspectFn = this.workspacePort.inspectWorkspacePatch?.bind(this.workspacePort) ?? inspectIsolatedWorkspacePatch
    const inspection = await inspectFn({
      mode: member.workspace,
      baseDir: parentBaseDir,
      workspacePath: member.workspace_path,
      memberName: member.name,
      ...(member.workspace_base_head === undefined ? {} : { baseHead: member.workspace_base_head }),
      targetDir,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })

    return {
      ...inspection,
      memberName: member.name,
      workspace: member.workspace,
      workspacePath: member.workspace_path,
    }
  }

  /**
   * Inspect and rank multiple `'share'` or `'branch'` candidate workers by verification status,
   * clean conflict-free applicability, and patch minimality (`changedFiles.length`, `insertions + deletions`).
   */
  async compareCandidatePatches(
    candidates: readonly (string | SwarmCandidateInput)[],
    options: SwarmCompareCandidatesOptions = {},
  ): Promise<SwarmCandidateComparisonResult> {
    if (candidates.length === 0) {
      throw new ValidationError('candidates', 'must provide at least one candidate worker name')
    }

    const expandedEntries: Array<{ readonly member: SwarmMemberRecord; readonly explicitVerified?: boolean }> = []
    const seenMembers = new Set<string>()
    for (const rawEntry of candidates) {
      const targetKey = (typeof rawEntry === 'string' ? rawEntry : rawEntry.name).trim()
      const explicitVerified = typeof rawEntry === 'string' ? undefined : rawEntry.verified
      if (!targetKey) {
        throw new ValidationError('candidates', 'candidate name must not be empty')
      }
      const matched = resolveTargetMembers(this.members, targetKey)
      if (matched.length === 0) {
        throw new ValidationError('candidates', `unknown swarm member or group "${targetKey}"`, targetKey)
      }
      for (const member of matched) {
        if (member.killed) {
          throw new ValidationError(
            'candidates',
            `worker "${member.name}" was killed and its isolated workspace was cleaned up`,
            member.name,
          )
        }
        if (member.workspace === 'inherit') {
          throw new ValidationError(
            'workspace',
            `cannot compare candidate "${member.name}" using "inherit" workspace; candidates must use "share" or "branch"`,
            member.name,
          )
        }
        if (!seenMembers.has(member.name)) {
          seenMembers.add(member.name)
          expandedEntries.push({
            member,
            ...(explicitVerified === undefined ? {} : { explicitVerified }),
          })
        }
      }
    }

    const resolveVerified = (memberName: string, explicitVerified: boolean | undefined): boolean => {
      if (explicitVerified !== undefined) return explicitVerified
      if (options.verified === undefined) return true
      if (options.verified instanceof Map) {
        return options.verified.get(memberName) === true
      }
      const record = options.verified as Readonly<Record<string, boolean>>
      return record[memberName] === true
    }

    const unranked = await Promise.all(
      expandedEntries.map(async ({ member, explicitVerified }) => {
        const inspected = await this.inspectMemberPatch(member.name, {
          ...(options.targetDir === undefined ? {} : { targetDir: options.targetDir }),
          ...(options.signal === undefined ? {} : { signal: options.signal }),
        })
        const verified = resolveVerified(member.name, explicitVerified)
        const nonEmpty = inspected.changedFiles.length > 0 && inspected.patchText.trim().length > 0
        const eligible = verified && nonEmpty && !inspected.hasConflictMarkers && inspected.applicable
        const totalLineChanges = inspected.insertions + inspected.deletions
        return {
          ...inspected,
          verified,
          eligible,
          totalLineChanges,
        }
      }),
    )

    unranked.sort((left, right) => {
      if (left.eligible !== right.eligible) return left.eligible ? -1 : 1
      if (left.verified !== right.verified) return left.verified ? -1 : 1
      if (left.hasConflictMarkers !== right.hasConflictMarkers) return left.hasConflictMarkers ? 1 : -1
      const leftNonEmpty = left.changedFiles.length > 0 && left.patchText.trim().length > 0
      const rightNonEmpty = right.changedFiles.length > 0 && right.patchText.trim().length > 0
      if (leftNonEmpty !== rightNonEmpty) return leftNonEmpty ? -1 : 1
      if (left.applicable !== right.applicable) return left.applicable ? -1 : 1
      if (left.changedFiles.length !== right.changedFiles.length) {
        return left.changedFiles.length - right.changedFiles.length
      }
      if (left.totalLineChanges !== right.totalLineChanges) {
        return left.totalLineChanges - right.totalLineChanges
      }
      if (left.deletions !== right.deletions) {
        return left.deletions - right.deletions
      }
      return left.memberName.localeCompare(right.memberName)
    })

    const ranked: SwarmRankedCandidatePatch[] = unranked.map((entry, index) => ({
      ...entry,
      rank: index + 1,
    }))
    const top = ranked[0]
    return {
      candidates: ranked,
      winner: top && top.eligible ? top : null,
    }
  }

  /**
   * Export a `'share'` or `'branch'` member's unified diff relative to its base HEAD,
   * optionally writing the patch file to disk.
   */
  async exportMemberPatch(
    name: string,
    options: SwarmExportMemberPatchOptions = {},
  ): Promise<SwarmExportedMemberPatch> {
    const inspected = await this.inspectMemberPatch(name, {
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    if (!options.allowConflictMarkers && inspected.hasConflictMarkers) {
      throw new ValidationError(
        'name',
        `candidate "${inspected.memberName}" contains unresolved conflict markers`,
        name,
      )
    }

    let resolvedOutputPath: string | undefined
    if (options.outputPath !== undefined) {
      const trimmedOutput = options.outputPath.trim()
      if (!trimmedOutput) {
        throw new ValidationError('outputPath', 'outputPath must be a non-empty file path')
      }
      resolvedOutputPath = resolve(trimmedOutput)
      await mkdir(dirname(resolvedOutputPath), { recursive: true })
      await Bun.write(resolvedOutputPath, inspected.patchText)
    }

    return {
      memberName: inspected.memberName,
      workspace: inspected.workspace,
      workspacePath: inspected.workspacePath,
      ...(inspected.baseHead === undefined ? {} : { baseHead: inspected.baseHead }),
      changedFiles: [...inspected.changedFiles],
      insertions: inspected.insertions,
      deletions: inspected.deletions,
      patchText: inspected.patchText,
      ...(resolvedOutputPath === undefined ? {} : { outputPath: resolvedOutputPath }),
    }
  }

  /**
   * Safely verify (`git apply --check`) and apply (`git apply`) a winning `'share'` or `'branch'`
   * candidate's patch into `targetDir` without running `git commit` or `git add`.
   */
  async applyWinnerPatchToWorkspace(
    name: string,
    targetDir: string = this.workspaceDir,
    options: SwarmApplyWinnerPatchOptions = {},
  ): Promise<SwarmApplyWinnerPatchResult> {
    const cleanTarget = targetDir.trim()
    if (!cleanTarget) {
      throw new ValidationError('targetDir', 'targetDir must be a non-empty directory path')
    }
    const resolvedTarget = resolve(cleanTarget)
    const inspected = await this.inspectMemberPatch(name, {
      targetDir: resolvedTarget,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })

    const isEmpty = inspected.changedFiles.length === 0 || !inspected.patchText.trim()
    if (isEmpty && !options.allowEmpty) {
      throw new ValidationError(
        'name',
        `candidate "${inspected.memberName}" has no patch changes to apply`,
        name,
      )
    }
    if (inspected.hasConflictMarkers && !options.allowConflictMarkers) {
      throw new ValidationError(
        'name',
        `candidate "${inspected.memberName}" contains unresolved conflict markers`,
        name,
      )
    }

    if (!isEmpty) {
      const applyFn = this.workspacePort.applyPatchToWorkspace?.bind(this.workspacePort) ?? applyWorkspacePatch
      await applyFn({
        patchText: inspected.patchText,
        targetDir: resolvedTarget,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      })
    }

    return {
      status: 'applied',
      memberName: inspected.memberName,
      targetDir: resolvedTarget,
      changedFiles: [...inspected.changedFiles],
      insertions: inspected.insertions,
      deletions: inspected.deletions,
      patchText: inspected.patchText,
    }
  }

  async waitForIdle(): Promise<void> {
    while (this.runningWorkerLoops.size > 0) {
      const current = [...this.runningWorkerLoops.values()]
      await Promise.allSettled(current)
    }
    await this.boardStore.flush()
  }

  async dispose(): Promise<void> {
    this.disposed = true
    this.coordinatorWaitHub.dispose()
    this.pendingWorkerJobs.clear()
    for (const [name, controller] of this.activeWorkerTurns.entries()) {
      if (!controller.signal.aborted) {
        controller.abort(new SwarmParkSignal(name, this.members.get(name)?.last_read_seq ?? 0, true))
      }
    }
    await Promise.allSettled([...this.runningWorkerLoops.values()])
    await this.boardStore.flush()
    this.memberHistories.clear()
    this.evictedInspectCache.clear()
    this.memberActivities.clear()
  }

  private enqueuePendingJob(memberName: string, job: PendingWorkerJob): void {
    const queue = this.pendingWorkerJobs.get(memberName)
    if (queue) {
      queue.push(job)
    } else {
      this.pendingWorkerJobs.set(memberName, [job])
    }
  }

  private dequeuePendingJob(memberName: string): PendingWorkerJob | undefined {
    const queue = this.pendingWorkerJobs.get(memberName)
    if (!queue || queue.length === 0) {
      this.pendingWorkerJobs.delete(memberName)
      return undefined
    }
    const next = queue.shift()
    if (queue.length === 0) {
      this.pendingWorkerJobs.delete(memberName)
    }
    return next
  }

  private hasPendingJobs(memberName: string): boolean {
    const queue = this.pendingWorkerJobs.get(memberName)
    return queue !== undefined && queue.length > 0
  }

  private resolveCandidateMemberForPatch(name: string): SwarmMemberRecord {
    const clean = name.trim()
    if (!clean) {
      throw new ValidationError('name', 'must specify a candidate worker name')
    }
    const member = this.members.get(clean)
    if (!member) {
      throw new ValidationError('name', `unknown swarm member "${clean}"`, clean)
    }
    if (member.killed) {
      throw new ValidationError(
        'name',
        `worker "${member.name}" was killed and its workspace is no longer available`,
        clean,
      )
    }
    if (member.workspace === 'inherit') {
      throw new ValidationError(
        'workspace',
        `cannot inspect isolated candidate patch for "${member.name}" using "inherit" workspace; use "share" or "branch"`,
        clean,
      )
    }
    return member
  }

  private resolveMemberBaseDir(member: SwarmMemberRecord): string {
    const parentRecord = member.added_by && member.added_by !== COORDINATOR_NAME
      ? this.members.get(member.added_by)
      : undefined
    return parentRecord ? parentRecord.workspace_path : this.workspaceDir
  }

  private scheduleWorkerJob(member: SwarmMemberRecord, job: PendingWorkerJob): void {
    if (!this.workerRunner || this.disposed || member.killed) {
      if (!this.workerRunner && !member.killed && member.state === 'working') {
        member.state = 'waiting'
      }
      return
    }
    this.enqueuePendingJob(member.name, job)
    if (this.runningWorkerLoops.has(member.name)) {
      return
    }

    const loopPromise = this.runWorkerLoop(member).finally(() => {
      this.runningWorkerLoops.delete(member.name)
      this.trimIdleWorkerHistories()
    })
    this.runningWorkerLoops.set(member.name, loopPromise)
  }

  private async runWorkerLoop(member: SwarmMemberRecord): Promise<void> {
    while (!this.disposed && !member.killed) {
      let nextJob = this.dequeuePendingJob(member.name)

      if (!nextJob) {
        const pendingMsg = this.findNextUnreadWakeMessage(member)
        if (!pendingMsg) break
        nextJob = {
          reason: 'wake',
          prompt: buildWorkerWakePrompt({
            head: this.headSeq,
            wakeSeq: pendingMsg.seq,
            wakeAuthor: pendingMsg.agent,
            wakeChannel: pendingMsg.channel,
            lastReadSeq: member.last_read_seq,
          }),
        }
      }

      if (!this.workerRunner) break

      if (member.state !== 'working') {
        member.state = 'working'
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
      }

      this.beginWorkerTurnActivity(member.name)
      this.explicitParkInTurn.delete(member.name)
      const turnStartHead = this.headSeq
      const turnController = new AbortController()
      this.activeWorkerTurns.set(member.name, turnController)
      let historyBefore = this.memberHistories.get(member.name)
      if (historyBefore === undefined) {
        historyBefore = this.evictedInspectCache.get(member.name)
          ?? (await this.boardStore.loadMemberHistory(member.name))
          ?? []
        this.evictedInspectCache.delete(member.name)
        this.memberHistories.set(member.name, historyBefore)
      }
      // Shallow copy isolates array pushes without deep-cloning the entire
      // multi-hundred-KB conversation on every turn.
      const workingMessages = [...historyBefore]

      let turnFailedError: string | undefined
      try {
        const snapshot = this.buildMemberSnapshot(member)
        const outcome = await this.workerRunner({
          swarm: this,
          sessionId: this.sessionId,
          member: snapshot,
          workspacePath: member.workspace_path,
          reason: nextJob.reason,
          prompt: nextJob.prompt,
          systemPromptOverlay: buildSwarmWorkerPromptOverlay(snapshot),
          messages: workingMessages,
          signal: turnController.signal,
        })
        const updatedMessages = outcome?.messages ? [...outcome.messages] : workingMessages
        const normalized = normalizeWorkerHistory(updatedMessages, {
          clone: false,
          maxToolContentChars: MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS,
        })
        await this.boardStore.saveMemberHistory(member.name, normalized)
        if (member.killed || this.disposed) {
          this.memberHistories.delete(member.name)
          this.evictedInspectCache.delete(member.name)
        } else {
          this.memberHistories.delete(member.name)
          this.memberHistories.set(member.name, normalized)
        }
        if (outcome?.error && outcome.error.trim()) {
          turnFailedError = outcome.error.trim()
        }
      } catch (error) {
        const abortReason: unknown = turnController.signal.aborted ? turnController.signal.reason : error
        if (!isSwarmParkSignal(abortReason) && !isSwarmKillSignal(abortReason) && !isSwarmParkSignal(error) && !isSwarmKillSignal(error)) {
          turnFailedError = errorMessage(error)
        }
        const normalized = normalizeWorkerHistory(workingMessages, {
          clone: false,
          maxToolContentChars: MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS,
        })
        await this.boardStore.saveMemberHistory(member.name, normalized)
        if (member.killed || this.disposed) {
          this.memberHistories.delete(member.name)
          this.evictedInspectCache.delete(member.name)
        } else {
          this.memberHistories.delete(member.name)
          this.memberHistories.set(member.name, normalized)
        }
      } finally {
        this.activeWorkerTurns.delete(member.name)
      }

      if (member.killed || this.disposed) {
        this.resetWorkerActivityToIdle(member.name, false)
        break
      }

      if (!this.explicitParkInTurn.has(member.name)) {
        member.parked_since = Math.max(member.last_read_seq, turnStartHead)
        member.mentions_only = member.role === 'narrator' ? false : true
      }
      this.explicitParkInTurn.delete(member.name)

      if (turnFailedError !== undefined) {
        member.error = turnFailedError
        member.state = 'waiting'
        this.resetWorkerActivityToIdle(member.name, true)
        await this.boardStore.saveState(this.snapshotStateForPersistence())
        this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
        const workerError: SwarmWorkerErrorInfo = {
          member_name: member.name,
          message: turnFailedError,
          quota_exhausted: isQuotaOrCapacityError(turnFailedError),
        }
        const consumedByWait = this.coordinatorWaitHub.notifyWorkerError(workerError)
        if (!consumedByWait && this.coordinatorWakeListeners.size > 0) {
          const event: SwarmCoordinatorWakeEvent = {
            session_id: this.sessionId,
            head: this.headSeq,
            last_read_seq: this.coordinatorLastRead,
            wake_seq: this.headSeq,
            reason: 'worker_error',
            worker_error: workerError,
            notice: formatWorkerErrorWakeNotice(workerError),
            status: this.status(),
          }
          for (const listener of this.coordinatorWakeListeners) {
            try {
              listener(event)
            } catch {
              // Never allow an observer callback failure to corrupt swarm execution.
            }
          }
        }
        break
      }

      delete member.error

      // Re-check whether a matching message or queued job landed while the turn was finishing!
      if (this.hasPendingJobs(member.name) || this.findNextUnreadWakeMessage(member) !== undefined) {
        continue
      }

      member.state = 'waiting'
      this.resetWorkerActivityToIdle(member.name, true)
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      this.trimIdleWorkerHistories()
      if (!member.killed && !this.disposed && (this.hasPendingJobs(member.name) || this.findNextUnreadWakeMessage(member) !== undefined)) {
        continue
      }
      this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
      break
    }
  }

  private findNextUnreadWakeMessage(member: SwarmMemberRecord): SwarmBoardMessage | undefined {
    const since = member.parked_since ?? member.last_read_seq
    const mentionsOnly = member.mentions_only ?? (member.role === 'narrator' ? false : true)
    const allNames = new Set(this.members.keys())
    return this.messages.find(msg => doesMessageWakeWorker(member, msg, since, mentionsOnly, allNames))
  }

  private routeWakeupsForMessage(
    message: SwarmBoardMessage,
    skipNames: ReadonlySet<string> = new Set(),
  ): string[] {
    const woken: string[] = []
    const allNames = new Set(this.members.keys())

    for (const member of this.members.values()) {
      if (member.killed || skipNames.has(member.name) || member.name === message.agent) {
        continue
      }
      if (member.state === 'stopped') {
        if (doesMessageReviveStoppedWorker(member, message, allNames)) {
          woken.push(member.name)
          if (this.workerRunner) {
            member.state = 'working'
            this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
            this.scheduleWorkerJob(member, {
              reason: 'revive',
              prompt: buildWorkerWakePrompt({
                head: this.headSeq,
                wakeSeq: message.seq,
                wakeAuthor: message.agent,
                wakeChannel: message.channel,
                lastReadSeq: member.last_read_seq,
                revived: true,
              }),
            })
          }
        }
        continue
      }

      if (member.state === 'waiting') {
        const since = member.parked_since ?? member.last_read_seq
        const mentionsOnly = member.mentions_only ?? (member.role === 'narrator' ? false : true)
        if (doesMessageWakeWorker(member, message, since, mentionsOnly, allNames)) {
          woken.push(member.name)
          if (this.workerRunner) {
            member.state = 'working'
            this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(member) })
            this.scheduleWorkerJob(member, {
              reason: 'wake',
              prompt: buildWorkerWakePrompt({
                head: this.headSeq,
                wakeSeq: message.seq,
                wakeAuthor: message.agent,
                wakeChannel: message.channel,
                lastReadSeq: member.last_read_seq,
              }),
            })
          }
        }
      }
    }

    const consumedByCoordinatorWait = this.coordinatorWaitHub.notifyMessage(message)
    const addressesCoordinator = message.agent !== COORDINATOR_NAME
      && (message.broadcast || message.mentions.includes(COORDINATOR_NAME))

    if (consumedByCoordinatorWait || addressesCoordinator) {
      woken.push(COORDINATOR_NAME)
    }

    if (!consumedByCoordinatorWait && addressesCoordinator && this.coordinatorWakeListeners.size > 0) {
      const event: SwarmCoordinatorWakeEvent = {
        session_id: this.sessionId,
        head: this.headSeq,
        last_read_seq: this.coordinatorLastRead,
        wake_seq: message.seq,
        reason: 'board',
        message: cloneBoardMessage(message),
        notice: formatBoardWakeNotice(this.headSeq, this.coordinatorLastRead),
        status: this.status(),
      }
      for (const listener of this.coordinatorWakeListeners) {
        try {
          listener(event)
        } catch {
          // Never allow an observer callback failure to corrupt board delivery.
        }
      }
    }

    return woken
  }

  private async advanceCallerReadSeq(
    caller: string,
    callerMember: SwarmMemberRecord | undefined,
    seq: number,
  ): Promise<void> {
    if (caller === COORDINATOR_NAME) {
      if (seq > this.coordinatorLastRead) {
        this.coordinatorLastRead = seq
        await this.boardStore.saveState(this.snapshotStateForPersistence())
      }
      return
    }
    if (callerMember && seq > callerMember.last_read_seq) {
      callerMember.last_read_seq = seq
      await this.boardStore.saveState(this.snapshotStateForPersistence())
      this.emitLifecycleEvent('member_updated', { member: this.buildMemberSnapshot(callerMember) })
    }
  }

  private async resolveModelSelection(
    requested: string,
    explicit: boolean,
    fallbackDisplayName?: string,
  ): Promise<SwarmModelInfo> {
    const clean = requested.trim()
    if (!clean) {
      throw new ValidationError('model', 'model identifier must not be empty')
    }
    if (!this.modelCatalog) {
      return {
        id: clean,
        display_name: fallbackDisplayName ?? clean,
      }
    }

    const catalog = await this.modelCatalog.listModels({ all: true })
    const lower = clean.toLowerCase()
    const match = catalog.find(item => item.id === clean || item.display_name.toLowerCase() === lower)

    if (!match) {
      if (!explicit && catalog.length === 0) {
        return {
          id: clean,
          display_name: fallbackDisplayName ?? clean,
        }
      }
      throw new ValidationError(
        'model',
        `model "${clean}" is not in the selectable swarm model catalog; call swarm{action:"models", all:true} to inspect options`,
        clean,
      )
    }

    if (match.quota_exhausted === true && match.credits_available !== true) {
      const resetNote = match.quota_resets ? ` (quota resets ${match.quota_resets})` : ''
      throw new ValidationError(
        'model',
        `model "${match.display_name}" (${match.id}) has exhausted its quota${resetNote} and is not covered by fallback credits`,
        clean,
      )
    }

    return match
  }

  private normalizeCaller(caller: string | undefined): string {
    const trimmed = caller?.trim()
    return trimmed || COORDINATOR_NAME
  }

  private assertLivingCallerOrCoordinator(caller: string): SwarmMemberRecord | undefined {
    if (caller === COORDINATOR_NAME) {
      return undefined
    }
    const member = this.members.get(caller)
    if (!member) {
      throw new ValidationError('caller', `unknown swarm member "${caller}"`, caller)
    }
    if (member.killed) {
      throw new ValidationError('caller', `worker "${caller}" has been killed`, caller)
    }
    return member
  }

  private findChannelSummary(channelName: string): SwarmChannelSummary | undefined {
    return computeChannelSummaries(this.channels.values(), this.members.values())
      .find(item => item.name === channelName)
  }

  private snapshotStateForPersistence(): PersistedSwarmState {
    return {
      version: 1,
      session_id: this.sessionId,
      workspace_dir: this.workspaceDir,
      title: this.swarmTitle,
      coordinator_activated: this.coordinatorActivated,
      coordinator_model: this.coordinatorModelId,
      coordinator_model_display_name: this.coordinatorModelDisplay,
      coordinator_last_read_seq: this.coordinatorLastRead,
      ...(this.coordinatorLastPostSeq === undefined ? {} : { coordinator_last_post_seq: this.coordinatorLastPostSeq }),
      ...(this.coordinatorLastPostTime === undefined ? {} : { coordinator_last_post_time: this.coordinatorLastPostTime }),
      head: this.headSeq,
      overview: this.overviewState ? { ...this.overviewState } : null,
      channels: [...this.channels.values()].map(ch => ({ ...ch })),
      members: [...this.members.values()].map(member => ({ ...member, channels: [...member.channels] })),
    }
  }

  private memberConversationId(name: string): string {
    return `${this.sessionId}:swarm:${name}`
  }

  private buildCoordinatorSnapshot(): SwarmMemberSnapshot {
    return {
      name: COORDINATOR_NAME,
      role: 'coordinator',
      model: this.coordinatorModelId,
      model_display_name: this.coordinatorModelDisplay,
      workspace: 'inherit',
      workspace_path: this.workspaceDir,
      channels: [WILDCARD_CHANNEL],
      state: this.coordinatorState,
      killed: false,
      conversation_id: this.memberConversationId(COORDINATOR_NAME),
      ...(this.coordinatorParkedSince === undefined ? {} : { parked_since: this.coordinatorParkedSince }),
      ...(this.coordinatorMentionsOnly === undefined ? {} : { mentions_only: this.coordinatorMentionsOnly }),
      last_read_seq: this.coordinatorLastRead,
      ...(this.coordinatorLastPostSeq === undefined ? {} : { last_post_seq: this.coordinatorLastPostSeq }),
      ...(this.coordinatorLastPostTime === undefined ? {} : { last_post_time: this.coordinatorLastPostTime }),
    }
  }

  private buildMemberSnapshot(record: SwarmMemberRecord): SwarmMemberSnapshot {
    const activity = this.memberActivities.get(record.name)
    const currentActivity = record.state === 'working' && !record.killed
      ? activity?.summary
      : undefined
    return toMemberSnapshot(record, {
      conversationId: this.memberConversationId(record.name),
      ...(currentActivity === undefined ? {} : { currentActivity }),
    })
  }

  /**
   * Evict oldest idle/waiting worker histories from `memberHistories` past the
   * warm cap once their transcripts have been saved to disk. Actively working
   * workers are never evicted, and evicted workers reload seamlessly from disk
   * if woken again or inspected via `inspectMember`.
   */
  private trimIdleWorkerHistories(): void {
    if (this.memberHistories.size <= MAX_WARM_IDLE_WORKER_HISTORIES) return
    let idleCount = 0
    for (const name of this.memberHistories.keys()) {
      const record = this.members.get(name)
      if (!record || record.killed || (record.state !== 'working' && !this.activeWorkerTurns.has(name) && !this.runningWorkerLoops.has(name))) {
        idleCount += 1
      }
    }
    if (idleCount <= MAX_WARM_IDLE_WORKER_HISTORIES) return
    let excess = idleCount - MAX_WARM_IDLE_WORKER_HISTORIES
    for (const name of [...this.memberHistories.keys()]) {
      if (excess <= 0) break
      const record = this.members.get(name)
      if (record && !record.killed && (record.state === 'working' || this.activeWorkerTurns.has(name) || this.runningWorkerLoops.has(name))) {
        continue
      }
      this.memberHistories.delete(name)
      excess -= 1
    }
  }

  private resolveMemberHistorySync(memberName: string): ChatMessage[] | undefined {
    const live = this.memberHistories.get(memberName)
    if (live !== undefined) return live
    const cached = this.evictedInspectCache.get(memberName)
    if (cached !== undefined) {
      this.evictedInspectCache.delete(memberName)
      this.evictedInspectCache.set(memberName, cached)
      return cached
    }
    if (!this.members.has(memberName)) return undefined
    const loaded = this.boardStore.loadMemberHistorySync(memberName)
    if (loaded === undefined) return undefined
    this.evictedInspectCache.set(memberName, loaded)
    while (this.evictedInspectCache.size > MAX_EVICTED_INSPECT_CACHE) {
      const oldest = this.evictedInspectCache.keys().next().value
      if (oldest === undefined) break
      this.evictedInspectCache.delete(oldest)
    }
    return loaded
  }

  private getOrCreateWorkerActivity(memberName: string): MutableWorkerActivity {
    let existing = this.memberActivities.get(memberName)
    if (!existing) {
      existing = {
        phase: 'idle',
        summary: undefined,
        activeTools: [],
        recentTools: [],
        liveText: '',
        updatedAt: undefined,
        lastEmitMs: 0,
        lastEmittedPhase: undefined,
        lastEmittedSignature: '',
      }
      this.memberActivities.set(memberName, existing)
    }
    return existing
  }

  private beginWorkerTurnActivity(memberName: string): void {
    const activity = this.getOrCreateWorkerActivity(memberName)
    activity.phase = 'thinking'
    activity.summary = 'Thinking…'
    activity.activeTools = []
    activity.liveText = ''
    activity.updatedAt = this.nowIso()
    activity.lastEmitMs = 0
    activity.lastEmittedPhase = undefined
    activity.lastEmittedSignature = ''
  }

  private resetWorkerActivityToIdle(memberName: string, emitIfChanged: boolean): void {
    const activity = this.memberActivities.get(memberName)
    if (!activity) return
    const wasActive = activity.phase !== 'idle' || activity.activeTools.length > 0 || activity.summary !== undefined || activity.liveText.length > 0
    activity.phase = 'idle'
    activity.summary = undefined
    activity.activeTools = []
    activity.liveText = ''
    activity.updatedAt = this.nowIso()
    activity.lastEmitMs = this.nowClock().getTime()
    activity.lastEmittedPhase = 'idle'
    activity.lastEmittedSignature = computeActivitySignature(activity)
    if (emitIfChanged && wasActive) {
      const record = this.members.get(memberName)
      if (record) {
        this.emitLifecycleEvent('member_activity', {
          member: this.buildMemberSnapshot(record),
          activity: this.snapshotMemberActivity(memberName),
        })
      }
    }
  }

  private snapshotMemberActivity(memberName: string): SwarmMemberActivitySnapshot {
    const activity = this.memberActivities.get(memberName)
    if (!activity) {
      return {
        phase: 'idle',
        active_tools: [],
        recent_tools: [],
      }
    }
    return {
      phase: activity.phase,
      ...(activity.summary === undefined ? {} : { summary: activity.summary }),
      active_tools: activity.activeTools.map(item => ({ ...item })),
      recent_tools: activity.recentTools.map(item => ({ ...item })),
      ...(activity.liveText ? { live_text_preview: activity.liveText } : {}),
      ...(activity.updatedAt === undefined ? {} : { updated_at: activity.updatedAt }),
    }
  }

  private emitLifecycleEvent(
    kind: SwarmLifecycleEventKind,
    extra: {
      readonly message?: SwarmBoardMessage
      readonly member?: SwarmMemberSnapshot
      readonly channel?: SwarmChannelSummary
      readonly activity?: SwarmMemberActivitySnapshot
    } = {},
  ): void {
    if (this.eventListeners.size === 0) return
    const event: SwarmLifecycleEvent = {
      session_id: this.sessionId,
      kind,
      head: this.headSeq,
      status: this.status(),
      ...(extra.message === undefined ? {} : { message: cloneBoardMessage(extra.message) }),
      ...(extra.member === undefined ? {} : { member: extra.member }),
      ...(extra.channel === undefined ? {} : { channel: extra.channel }),
      ...(extra.activity === undefined ? {} : { activity: extra.activity }),
    }
    for (const listener of this.eventListeners) {
      try {
        listener(event)
      } catch {
        // Observers never disrupt core swarm execution.
      }
    }
  }

  private nowIso(): string {
    return this.nowClock().toISOString()
  }
}

function computeActivitySignature(activity: MutableWorkerActivity): string {
  const active = activity.activeTools.map(item => `${item.id}:${item.status}`).join(',')
  const lastRecent = activity.recentTools.at(-1)
  const recent = lastRecent ? `${activity.recentTools.length}:${lastRecent.id}:${lastRecent.status}` : '0'
  return `${activity.phase}|${activity.summary ?? ''}|${activity.liveText}|${active}|${recent}`
}

function cloneBoardMessage(msg: SwarmBoardMessage): SwarmBoardMessage {
  return {
    seq: msg.seq,
    agent: msg.agent,
    channel: msg.channel,
    text: msg.text,
    mentions: [...msg.mentions],
    channel_mentions: [...msg.channel_mentions],
    broadcast: msg.broadcast,
    time: msg.time,
    ...(msg.kind === undefined ? {} : { kind: msg.kind }),
  }
}

function parseArgsObject(rawArgs: unknown): Record<string, unknown> | undefined {
  if (typeof rawArgs === 'object' && rawArgs !== null && !Array.isArray(rawArgs)) {
    return rawArgs as Record<string, unknown>
  }
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (!trimmed) return undefined
    try {
      const parsed: unknown = JSON.parse(trimmed)
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      return undefined
    }
  }
  return undefined
}

function firstNonEmptyString(record: Record<string, unknown> | undefined, keys: readonly string[]): string | undefined {
  if (!record) return undefined
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }
  return undefined
}

function collapseWhitespace(value: string, maxChars: number): string {
  const collapsed = value.replace(/\s+/gu, ' ').trim()
  if (collapsed.length <= maxChars) return collapsed
  return `${collapsed.slice(0, Math.max(1, maxChars - 1))}…`
}

function formatRunningToolLabel(toolName: string, summary?: string): string {
  const lower = toolName.toLowerCase()
  if (lower === 'read' || lower === 'readfile' || lower === 'read_file' || lower === 'view_file') {
    return summary ? `Reading ${summary}` : 'Reading file…'
  }
  if (
    lower === 'write'
    || lower === 'writefile'
    || lower === 'write_file'
    || lower === 'edit'
    || lower === 'editfile'
    || lower === 'edit_file'
    || lower === 'applypatch'
    || lower === 'apply_patch'
  ) {
    return summary ? `Editing ${summary}` : 'Editing file…'
  }
  if (lower === 'bash' || lower === 'bashtool' || lower === 'exec_command' || lower === 'run_command') {
    return summary ? `Running ${summary}` : 'Running bash…'
  }
  if (lower === 'grep' || lower === 'greptool' || lower === 'glob' || lower === 'globtool' || lower === 'grep_search') {
    return summary ? `Searching ${summary}` : 'Searching workspace…'
  }
  if (lower === 'read_board') return 'Reading swarm board…'
  if (lower === 'post_message') return 'Posting to swarm board…'
  if (lower === 'swarm_wait') return 'Parking on board…'
  if (lower === 'swarm_overview') return 'Updating swarm overview…'
  if (lower === 'swarm') return summary ? `Swarm ${summary}…` : 'Managing swarm…'
  return summary ? `Running ${toolName} (${summary})` : `Running ${toolName}…`
}

function summarizeToolCall(
  toolName: string,
  rawArgs: unknown,
): { readonly label: string; readonly summary?: string } {
  const args = parseArgsObject(rawArgs)
  const filePath = firstNonEmptyString(args, ['file_path', 'path', 'target_file', 'Path', 'FilePath'])
  const command = firstNonEmptyString(args, ['command', 'cmd', 'CommandLine'])
  const pattern = firstNonEmptyString(args, ['pattern', 'query', 'Query'])
  const action = firstNonEmptyString(args, ['action', 'Action'])
  const url = firstNonEmptyString(args, ['url', 'Url'])

  let summary: string | undefined
  if (filePath) {
    summary = basename(filePath) || collapseWhitespace(filePath, 40)
  } else if (command) {
    summary = collapseWhitespace(command, 44)
  } else if (pattern) {
    summary = collapseWhitespace(pattern, 40)
  } else if (action) {
    summary = collapseWhitespace(action, 32)
  } else if (url) {
    summary = collapseWhitespace(url, 44)
  }

  return {
    label: collapseWhitespace(formatRunningToolLabel(toolName, summary), 72),
    ...(summary === undefined ? {} : { summary }),
  }
}

function isToolResultError(result: ToolResult): boolean {
  if (!result.permitted) return true
  const trimmed = result.result.trimStart()
  return trimmed.startsWith('Error:') || trimmed.startsWith('[Error:') || trimmed.startsWith('{"error":')
}

function formatToolResultPreview(raw: string, maxChars: number): string | undefined {
  const cleaned = collapseWhitespace(raw, maxChars)
  return cleaned || undefined
}

function formatToolArgsPreview(rawArgs: unknown, maxChars: number): string {
  if (typeof rawArgs === 'string') {
    return collapseWhitespace(rawArgs, maxChars)
  }
  try {
    return collapseWhitespace(JSON.stringify(rawArgs ?? {}), maxChars)
  } catch {
    return '{}'
  }
}

function extractMessageText(content: MessageContent): string {
  if (typeof content === 'string') return content
  return content
    .map(part => (part.type === 'text' ? part.text : '[image]'))
    .join('\n')
}

function truncateInspectText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const omitted = text.length - maxChars
  return `${text.slice(0, maxChars)}… [truncated ${omitted} chars]`
}

function toInspectMessage(msg: ChatMessage, index: number): SwarmInspectMessage {
  if (msg.role === 'tool') {
    const rawText = extractMessageText(msg.content)
    return {
      index,
      role: 'tool',
      text: truncateInspectText(rawText, MAX_INSPECT_TOOL_TEXT_CHARS),
      ...(msg.name ? { tool_name: msg.name } : {}),
    }
  }
  if (msg.role === 'assistant') {
    const rawText = extractMessageText(msg.content)
    const toolCalls = msg.tool_calls && msg.tool_calls.length > 0
      ? msg.tool_calls.map(tc => ({
          id: tc.id,
          name: tc.function.name,
          args_preview: formatToolArgsPreview(tc.function.arguments, MAX_TOOL_ARGS_PREVIEW_CHARS),
        }))
      : undefined
    return {
      index,
      role: 'assistant',
      text: truncateInspectText(rawText, MAX_INSPECT_MESSAGE_TEXT_CHARS),
      ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
    }
  }
  const rawText = extractMessageText(msg.content)
  return {
    index,
    role: 'user',
    text: truncateInspectText(rawText, MAX_INSPECT_MESSAGE_TEXT_CHARS),
  }
}
