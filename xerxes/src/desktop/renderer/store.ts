// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Application store: one plain observable object, read by React through
 * `useSyncExternalStore`. All daemon interaction lives here as actions; the
 * transcript fold lives in a BlockBuilder and is exposed as part of the
 * snapshot. No React state, no effect cascades.
 *
 * Beyond the transcript, the store folds the same event stream into the
 * workspace surfaces: a per-file changes accumulator (edit-family tool calls
 * carry real old/new strings), a bounded raw-event ring for the Log tab, a
 * plan capture driven by plan-mode agent output, a steering queue mirrored
 * from `turn.steer` acceptances, and failed-turn state keyed off error
 * notifications (a provider failure arrives as text, never as silence).
 */

import { todoItemsOf, todosFromResult, type TodoItem } from './todoState.js'
import { pageIsBackground, pageIsInactive } from './pageVisibility.js'
import { hideWorkspaceInState, normalizeWorkspacePath, setThreadInboxState, unhideWorkspaceInState } from './threadInbox.js'
import type { WorkspaceContext } from '../main/contextNavigation.js'
import {
  BlockBuilder,
  blocksFromStoredMessages,
  editStatsOf,
  normalizeToolName,
  parseArgs,
  parseStructuredDiff,
  parseStructuredDiffs,
  toolFailureText,
  trimmedChangedLines,
  type StructuredDiffPayload,
} from './blocks.js'
import {
  daemonCompatibilityWarning,
  DESKTOP_DAEMON_PROTOCOL,
  DESKTOP_VERSION,
  EXPECTED_DAEMON_BUILD_ID,
} from './buildInfo.js'
import { sessionToMarkdown, type ExportSession } from './exportMarkdown.js'
import { selectedSession, rememberSession } from './sessionPreference.js'
import { historyBlocks, readHistoryPage, type HistoryPage } from './history.js'
import { connectionFailureKind } from './connectionFailure.js'
import { desktopCall, desktopError, text } from './desktopRpc.js'
import { learningCounts, readLearningList, type LearningAction, type LearningCounts, type LearningList, type LearningResult, type LearningTarget } from './learning.js'
import {
  applySwarmStatusDelta,
  mergeSwarmMessages,
  parseSwarmBoard,
  parseSwarmEvent,
  parseSwarmMemberInspect,
  parseSwarmMessage,
  parseSwarmSlashCommand,
  parseSwarmStatus,
  type SwarmBoardMessage,
  type SwarmMemberInspectResult,
  type SwarmStatusSnapshot,
} from './swarmState.js'
import {
  applyBtwEventPayload,
  parseSideQuestionSlashCommand,
  settleSideQuestionTools,
  type SideQuestionTurn,
} from './sideQuestionState.js'
import { agentGroupOf, foldAgentEvent } from './agentEvents.js'
import { parseGoal } from './goalText.js'
import { agentStatusOf, syncAgentMembers } from './agentMembers.js'
import type {
  AgentMember,
  AgentPreset,
  Approval,
  ApprovalResponse,
  BackgroundJob,
  Block,
  CachedModel,
  ChannelRow,
  ChannelStatus,
  CreatorTrace,
  DaemonEvent,
  DiffFile,
  DiffLine,
  FailedTurn,
  LogEntry,
  McpServerStatus,
  ModelChoice,
  PermissionMode,
  PlanItem,
  PlanState,
  ProviderRow,
  ProviderTypeRow,
  LocalSetupView,
  RunModeName,
  RunModeView,
  QueueItem,
  SessionRow,
  SessionSearchHit,
  SessionSearchStats,
  SettingsTab,
  SkillSuggestion,
  TerminalDetail,
  TerminalRow,
  WorkspaceTab,
} from './types.js'
import { harnessOriginOf } from './types.js'
import { localSetupJobView, localStatusView, runModeView } from './runModeView.js'
import { isLegacyHarnessPrompt, withoutTurnContext } from './transcriptContent.js'

export type Connection = 'connecting' | 'online' | 'offline'

const STREAM_THINKING_KEY = 'xerxes.streamThinking'
const SIDE_QUESTION_MODEL_KEY = 'xerxes.sideQuestion.model'

/** Persisted display preference; defaults on everywhere (SSR/test safe). */
function readStreamThinking(): boolean {
  try {
    return (typeof localStorage === 'undefined' ? null : localStorage.getItem(STREAM_THINKING_KEY)) !== '0'
  } catch {
    return true
  }
}

function writeStreamThinking(value: boolean): void {
  try {
    localStorage?.setItem(STREAM_THINKING_KEY, value ? '1' : '0')
  } catch {
    // A blocked storage area must not break the toggle — the choice just
    // stays session-scoped.
  }
}

/** Persisted Side Question model override (`''` = follow current session model). */
function readSideQuestionModel(): string {
  try {
    return (typeof localStorage === 'undefined' ? null : localStorage.getItem(SIDE_QUESTION_MODEL_KEY))?.trim() ?? ''
  } catch {
    return ''
  }
}

function writeSideQuestionModel(model: string): void {
  try {
    if (!model) localStorage?.removeItem(SIDE_QUESTION_MODEL_KEY)
    else localStorage?.setItem(SIDE_QUESTION_MODEL_KEY, model)
  } catch {
    // A blocked storage area keeps the choice session-scoped.
  }
}

export type AcpAgentProvider = 'xerxes' | 'claude-code' | 'antigravity' | 'codex' | 'acp'
export type AcpAgentTransport = 'native' | 'acp' | 'claude-stream-json'

export interface AcpExternalAgentInfo {
  readonly id: string
  readonly name: string
  readonly provider: AcpAgentProvider
  readonly transport: AcpAgentTransport
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

export interface AcpExternalThoughtLevelOption {
  readonly id: string
  readonly name: string
  readonly description?: string
}

export interface AcpConfigSelectOption {
  readonly value: string
  readonly name: string
  readonly description?: string
  readonly group?: string
}

export interface AcpConfigOption {
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly category?: string
  readonly type: string
  readonly current_value: string | boolean
  readonly options: readonly AcpConfigSelectOption[]
}

export interface AcpAuthMethodInfo {
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly type?: string
  readonly args?: readonly string[]
  readonly env?: Readonly<Record<string, string>>
  readonly meta?: Readonly<Record<string, unknown>>
}

export interface AcpExternalSessionInfo {
  readonly session_id: string
  readonly cwd?: string
  readonly title?: string
  readonly updated_at?: string
}

export interface AcpSessionsResult {
  readonly ok: boolean
  readonly sessions: readonly AcpExternalSessionInfo[]
  readonly next_cursor?: string
  readonly error?: string
}

export interface AcpAuthenticateResult {
  readonly ok: boolean
  readonly mode?: 'agent' | 'terminal'
  readonly command?: string
  readonly args?: readonly string[]
  readonly env?: Readonly<Record<string, string>>
  readonly label?: string
  readonly error?: string
}

export type AcpLoginPhase =
  | 'starting'
  | 'awaiting_code'
  | 'waiting_browser'
  | 'verifying'
  | 'done'
  | 'error'
  | 'cancelled'

export interface AcpLoginStatus {
  readonly ok: boolean
  readonly login_id: string
  readonly phase: AcpLoginPhase
  readonly needs_code: boolean
  readonly authorization_url?: string
  readonly error?: string
}

export function parseAcpLoginStatus(value: unknown): AcpLoginStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, login_id: '', phase: 'error', needs_code: false, error: 'Invalid login response' }
  }
  const row = value as Readonly<Record<string, unknown>>
  const rawPhase = typeof row.phase === 'string' ? row.phase : ''
  const phase: AcpLoginPhase =
    rawPhase === 'starting'
    || rawPhase === 'awaiting_code'
    || rawPhase === 'waiting_browser'
    || rawPhase === 'verifying'
    || rawPhase === 'done'
    || rawPhase === 'error'
    || rawPhase === 'cancelled'
      ? rawPhase
      : row.ok === false
        ? 'error'
        : 'starting'
  const loginId = typeof row.login_id === 'string' ? row.login_id.trim() : ''
  const authUrl = typeof row.authorization_url === 'string' && row.authorization_url.trim()
    ? row.authorization_url.trim()
    : undefined
  const error = typeof row.error === 'string' && row.error.trim()
    ? row.error.trim()
    : undefined
  return {
    ok: row.ok !== false && phase !== 'error',
    login_id: loginId,
    phase,
    needs_code: row.needs_code === true,
    ...(authUrl ? { authorization_url: authUrl } : {}),
    ...(error ? { error } : {}),
  }
}

export interface AcpRuntimeCapabilities {
  readonly prompt: {
    readonly image: boolean
    readonly audio: boolean
    readonly embedded_context: boolean
  }
  readonly session: {
    readonly load: boolean
    readonly resume: boolean
    readonly list: boolean
    readonly fork: boolean
    readonly close: boolean
    readonly additional_directories: boolean
  }
  readonly mcp: {
    readonly http: boolean
    readonly sse: boolean
  }
  readonly auth_methods: readonly AcpAuthMethodInfo[]
  readonly meta?: Readonly<Record<string, unknown>>
}

export interface AcpAvailableCommand {
  readonly name: string
  readonly description?: string
  readonly input_hint?: string
  readonly kind?: 'command' | 'skill'
  readonly source?: string
}

export interface AcpDiscoveredSkill {
  readonly name: string
  readonly description?: string
  readonly source?: string
  readonly path?: string
  readonly slash_command?: string
}

export interface AcpCatalogResult {
  readonly ok: boolean
  readonly connected: boolean
  readonly binding?: AcpSessionBinding | null
  readonly commands: readonly AcpAvailableCommand[]
  readonly skills: readonly AcpDiscoveredSkill[]
  readonly capabilities?: AcpRuntimeCapabilities
  readonly config_options: readonly AcpConfigOption[]
}

export interface ApprovalDiff {
  readonly path: string
  readonly oldText?: string
  readonly newText?: string
}

export interface ApprovalOption {
  readonly optionId: string
  readonly name: string
  readonly kind: string
  readonly warning?: string
}

export interface DesktopApproval extends Approval {
  readonly allowPermanent?: boolean
  readonly diff?: ApprovalDiff
  readonly diffs?: readonly ApprovalDiff[]
  readonly options?: readonly ApprovalOption[]
}

export interface CompletionHintItem {
  readonly value: string
  readonly label: string
  readonly meta: string
  readonly kind: 'command' | 'skill'
  readonly badge?: string
  readonly runtimeProvider?: string
}

export interface AcpSessionBinding {
  readonly session_id: string
  readonly agent_id: string
  readonly name: string
  readonly provider: AcpAgentProvider
  readonly transport: AcpAgentTransport
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
  readonly config_options?: readonly AcpConfigOption[]
  readonly thought_level?: string
  readonly thought_levels?: readonly AcpExternalThoughtLevelOption[]
  readonly thought_level_config_id?: string
  readonly available_commands?: readonly AcpAvailableCommand[]
  readonly discovered_skills?: readonly AcpDiscoveredSkill[]
  readonly additional_directories?: readonly string[]
  readonly sandbox?: boolean
  readonly agent_override?: string
  readonly mcp_server_names?: readonly string[]
}

function parseAcpProvider(value: unknown): AcpAgentProvider {
  return value === 'xerxes'
    || value === 'claude-code'
    || value === 'antigravity'
    || value === 'codex'
    || value === 'acp'
    ? value
    : 'acp'
}

function parseAcpTransport(value: unknown): AcpAgentTransport {
  return value === 'native' || value === 'acp' || value === 'claude-stream-json'
    ? value
    : 'acp'
}

export function parseAcpExternalAgent(value: unknown): AcpExternalAgentInfo | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Readonly<Record<string, unknown>>
  const id = typeof row.id === 'string' ? row.id.trim() : ''
  if (!id) return null
  const name = typeof row.name === 'string' && row.name.trim() ? row.name.trim() : id
  const command = typeof row.command === 'string' && row.command.trim() ? row.command.trim() : undefined
  const args = Array.isArray(row.args)
    ? row.args.filter((item): item is string => typeof item === 'string')
    : undefined
  const binaryPath = typeof row.binaryPath === 'string' && row.binaryPath.trim()
    ? row.binaryPath.trim()
    : typeof row.binary_path === 'string' && row.binary_path.trim()
      ? row.binary_path.trim()
      : undefined
  const version = typeof row.version === 'string' && row.version.trim() ? row.version.trim() : undefined
  const installHint = typeof row.installHint === 'string' && row.installHint.trim()
    ? row.installHint.trim()
    : typeof row.install_hint === 'string' && row.install_hint.trim()
      ? row.install_hint.trim()
      : undefined
  const description = typeof row.description === 'string' ? row.description.trim() : ''
  return {
    id,
    name,
    provider: parseAcpProvider(row.provider),
    transport: parseAcpTransport(row.transport),
    available: row.available === true,
    ...(command ? { command } : {}),
    ...(args ? { args } : {}),
    ...(binaryPath ? { binaryPath } : {}),
    ...(version ? { version } : {}),
    ...(installHint ? { installHint } : {}),
    description,
  }
}

export function parseAcpExternalAgents(value: unknown): readonly AcpExternalAgentInfo[] {
  if (!Array.isArray(value)) return []
  return value
    .map(parseAcpExternalAgent)
    .filter((item): item is AcpExternalAgentInfo => item !== null)
}

function parseAcpExternalOption(value: unknown): AcpExternalModelOption | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Readonly<Record<string, unknown>>
  const id =
    typeof row.id === 'string' && row.id.trim()
      ? row.id.trim()
      : typeof row.modelId === 'string' && row.modelId.trim()
        ? row.modelId.trim()
        : typeof row.value === 'string' && row.value.trim()
          ? row.value.trim()
          : ''
  if (!id) return null
  const name = typeof row.name === 'string' && row.name.trim() ? row.name.trim() : id
  const description = typeof row.description === 'string' && row.description.trim() ? row.description.trim() : undefined
  return {
    id,
    name,
    ...(description ? { description } : {}),
  }
}

function parseAcpExternalOptions(value: unknown): readonly AcpExternalModelOption[] | undefined {
  if (!Array.isArray(value)) return undefined
  const seen = new Set<string>()
  const parsed: AcpExternalModelOption[] = []
  for (const item of value) {
    const option = parseAcpExternalOption(item)
    if (!option || seen.has(option.id)) continue
    seen.add(option.id)
    parsed.push(option)
  }
  return parsed.length > 0 ? parsed : undefined
}

function parseAcpConfigOptions(value: unknown): readonly AcpConfigOption[] | undefined {
  if (!Array.isArray(value)) return undefined
  const parsed: AcpConfigOption[] = []
  const seen = new Set<string>()
  for (const item of value) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const row = item as Readonly<Record<string, unknown>>
    const id = typeof row.id === 'string' ? row.id.trim() : ''
    if (!id || seen.has(id)) continue
    seen.add(id)
    const name = typeof row.name === 'string' && row.name.trim() ? row.name.trim() : id
    const description = typeof row.description === 'string' && row.description.trim() ? row.description.trim() : undefined
    const category = typeof row.category === 'string' && row.category.trim() ? row.category.trim() : undefined
    const type = typeof row.type === 'string' && row.type.trim() ? row.type.trim() : 'select'
    const rawCurrent = row.current_value ?? row.currentValue
    const currentValue: string | boolean =
      typeof rawCurrent === 'boolean'
        ? rawCurrent
        : type === 'boolean'
          ? (typeof rawCurrent === 'string'
              ? rawCurrent.trim().toLowerCase() === 'true'
              : Boolean(rawCurrent))
          : typeof rawCurrent === 'string' && rawCurrent.trim()
            ? rawCurrent.trim()
            : typeof rawCurrent === 'number'
              ? String(rawCurrent)
              : ''
    const rawOptions = Array.isArray(row.options) ? row.options : []
    const options: AcpConfigSelectOption[] = []
    const seenOpts = new Set<string>()
    const pushOption = (opt: Readonly<Record<string, unknown>>, fallbackGroup?: string): void => {
      const optValue =
        typeof opt.value === 'string' && opt.value.trim()
          ? opt.value.trim()
          : typeof opt.id === 'string' && opt.id.trim()
            ? opt.id.trim()
            : ''
      if (!optValue || seenOpts.has(optValue)) return
      seenOpts.add(optValue)
      const optName = typeof opt.name === 'string' && opt.name.trim() ? opt.name.trim() : optValue
      const optDesc = typeof opt.description === 'string' && opt.description.trim() ? opt.description.trim() : undefined
      const optGroup =
        (typeof opt.group === 'string' && opt.group.trim() ? opt.group.trim() : undefined)
        ?? fallbackGroup
      options.push({
        value: optValue,
        name: optName,
        ...(optDesc ? { description: optDesc } : {}),
        ...(optGroup ? { group: optGroup } : {}),
      })
    }
    for (const rawOpt of rawOptions) {
      if (!rawOpt || typeof rawOpt !== 'object' || Array.isArray(rawOpt)) continue
      const opt = rawOpt as Readonly<Record<string, unknown>>
      if (Array.isArray(opt.options)) {
        const groupLabel =
          (typeof opt.name === 'string' && opt.name.trim() ? opt.name.trim() : undefined)
          ?? (typeof opt.group === 'string' && opt.group.trim() ? opt.group.trim() : undefined)
        for (const nested of opt.options) {
          if (!nested || typeof nested !== 'object' || Array.isArray(nested)) continue
          pushOption(nested as Readonly<Record<string, unknown>>, groupLabel)
        }
        continue
      }
      pushOption(opt)
    }
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
  return parsed.length > 0 ? parsed : undefined
}

function parseAcpAvailableCommands(value: unknown): readonly AcpAvailableCommand[] | undefined {
  if (!Array.isArray(value)) return undefined
  const seen = new Set<string>()
  const parsed: AcpAvailableCommand[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const row = item as Readonly<Record<string, unknown>>
    const rawName = typeof row.name === 'string' ? row.name.trim().replace(/^\/+/, '') : ''
    if (!rawName) continue
    const key = rawName.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    const description = typeof row.description === 'string' && row.description.trim() ? row.description.trim() : undefined
    const inputObj = row.input && typeof row.input === 'object' && !Array.isArray(row.input)
      ? (row.input as Readonly<Record<string, unknown>>)
      : null
    const inputHint =
      typeof row.input_hint === 'string' && row.input_hint.trim()
        ? row.input_hint.trim()
        : typeof row.inputHint === 'string' && row.inputHint.trim()
          ? row.inputHint.trim()
          : typeof inputObj?.hint === 'string' && inputObj.hint.trim()
            ? inputObj.hint.trim()
            : undefined
    const kind = row.kind === 'skill' || row.kind === 'command' ? row.kind : undefined
    const source = typeof row.source === 'string' && row.source.trim() ? row.source.trim() : undefined
    parsed.push({
      name: rawName,
      ...(description ? { description } : {}),
      ...(inputHint ? { input_hint: inputHint } : {}),
      ...(kind ? { kind } : {}),
      ...(source ? { source } : {}),
    })
  }
  return parsed.length > 0 ? parsed : undefined
}

function parseAcpDiscoveredSkills(value: unknown): readonly AcpDiscoveredSkill[] | undefined {
  if (!Array.isArray(value)) return undefined
  const seen = new Set<string>()
  const parsed: AcpDiscoveredSkill[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const row = item as Readonly<Record<string, unknown>>
    const rawName = typeof row.name === 'string' ? row.name.trim().replace(/^\/+/, '') : ''
    if (!rawName) continue
    const key = rawName.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    const description = typeof row.description === 'string' && row.description.trim() ? row.description.trim() : undefined
    const source = typeof row.source === 'string' && row.source.trim() ? row.source.trim() : undefined
    const path = typeof row.path === 'string' && row.path.trim() ? row.path.trim() : undefined
    const rawSlash =
      typeof row.slash_command === 'string' && row.slash_command.trim()
        ? row.slash_command.trim()
        : typeof row.slashCommand === 'string' && row.slashCommand.trim()
          ? row.slashCommand.trim()
          : undefined
    const slashCommand = rawSlash ? (rawSlash.startsWith('/') ? rawSlash : `/${rawSlash}`) : undefined
    parsed.push({
      name: rawName,
      ...(description ? { description } : {}),
      ...(source ? { source } : {}),
      ...(path ? { path } : {}),
      ...(slashCommand ? { slash_command: slashCommand } : {}),
    })
  }
  return parsed.length > 0 ? parsed : undefined
}

function costUsdFromPayload(payload: Readonly<Record<string, unknown>>): number | null {
  if (typeof payload.cost_usd === 'number' && Number.isFinite(payload.cost_usd) && payload.cost_usd >= 0) {
    return payload.cost_usd
  }
  if (typeof payload.costUsd === 'number' && Number.isFinite(payload.costUsd) && payload.costUsd >= 0) {
    return payload.costUsd
  }
  const rawCost = payload.cost
  if (typeof rawCost === 'number' && Number.isFinite(rawCost) && rawCost >= 0) {
    return rawCost
  }
  if (rawCost && typeof rawCost === 'object' && !Array.isArray(rawCost)) {
    const rec = rawCost as Readonly<Record<string, unknown>>
    const amount = rec.amount
    const currency = typeof rec.currency === 'string' ? rec.currency.trim().toUpperCase() : ''
    if (typeof amount === 'number' && Number.isFinite(amount) && amount >= 0 && (!currency || currency === 'USD')) {
      return amount
    }
  }
  return null
}

export function parseAcpAuthMethods(value: unknown): readonly AcpAuthMethodInfo[] {
  if (!Array.isArray(value)) return []
  const authMethods: AcpAuthMethodInfo[] = []
  const seenAuth = new Set<string>()
  for (const item of value) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const auth = item as Readonly<Record<string, unknown>>
    const id = typeof auth.id === 'string' && auth.id.trim() ? auth.id.trim() : ''
    if (!id || seenAuth.has(id)) continue
    seenAuth.add(id)
    const name = typeof auth.name === 'string' && auth.name.trim() ? auth.name.trim() : id
    const description = typeof auth.description === 'string' && auth.description.trim() ? auth.description.trim() : undefined
    const type = typeof auth.type === 'string' && auth.type.trim() ? auth.type.trim() : undefined
    const args = Array.isArray(auth.args)
      ? auth.args.filter((entry): entry is string => typeof entry === 'string')
      : undefined
    const rawEnv = auth.env && typeof auth.env === 'object' && !Array.isArray(auth.env)
      ? (auth.env as Readonly<Record<string, unknown>>)
      : undefined
    let env: Record<string, string> | undefined
    if (rawEnv) {
      const pairs = Object.entries(rawEnv).filter((pair): pair is [string, string] => typeof pair[1] === 'string')
      if (pairs.length > 0) env = Object.fromEntries(pairs)
    }
    const meta =
      auth.meta && typeof auth.meta === 'object' && !Array.isArray(auth.meta)
        ? (auth.meta as Readonly<Record<string, unknown>>)
        : auth._meta && typeof auth._meta === 'object' && !Array.isArray(auth._meta)
          ? (auth._meta as Readonly<Record<string, unknown>>)
          : undefined
    authMethods.push({
      id,
      name,
      ...(description ? { description } : {}),
      ...(type ? { type } : {}),
      ...(args && args.length > 0 ? { args } : {}),
      ...(env ? { env } : {}),
      ...(meta ? { meta } : {}),
    })
  }
  return authMethods
}

function parseAcpRuntimeCapabilities(value: unknown): AcpRuntimeCapabilities | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const row = value as Readonly<Record<string, unknown>>
  const promptObj =
    row.prompt && typeof row.prompt === 'object' && !Array.isArray(row.prompt)
      ? (row.prompt as Readonly<Record<string, unknown>>)
      : row.promptCapabilities && typeof row.promptCapabilities === 'object' && !Array.isArray(row.promptCapabilities)
        ? (row.promptCapabilities as Readonly<Record<string, unknown>>)
        : null
  const sessionObj =
    row.session && typeof row.session === 'object' && !Array.isArray(row.session)
      ? (row.session as Readonly<Record<string, unknown>>)
      : row.sessionCapabilities && typeof row.sessionCapabilities === 'object' && !Array.isArray(row.sessionCapabilities)
        ? (row.sessionCapabilities as Readonly<Record<string, unknown>>)
        : null
  const mcpObj =
    row.mcp && typeof row.mcp === 'object' && !Array.isArray(row.mcp)
      ? (row.mcp as Readonly<Record<string, unknown>>)
      : row.mcpCapabilities && typeof row.mcpCapabilities === 'object' && !Array.isArray(row.mcpCapabilities)
        ? (row.mcpCapabilities as Readonly<Record<string, unknown>>)
        : null
  const authMethods = parseAcpAuthMethods(row.auth_methods ?? row.authMethods)
  const metaObj =
    row.meta && typeof row.meta === 'object' && !Array.isArray(row.meta)
      ? (row.meta as Readonly<Record<string, unknown>>)
      : row._meta && typeof row._meta === 'object' && !Array.isArray(row._meta)
        ? (row._meta as Readonly<Record<string, unknown>>)
        : undefined
  const flag = (obj: Readonly<Record<string, unknown>> | null, ...keys: readonly string[]): boolean => {
    if (!obj) return false
    for (const key of keys) {
      const entry = obj[key]
      if (entry === true) return true
      if (entry && typeof entry === 'object' && !Array.isArray(entry)) return true
    }
    return false
  }
  return {
    prompt: {
      image: flag(promptObj, 'image'),
      audio: flag(promptObj, 'audio'),
      embedded_context: flag(promptObj, 'embedded_context', 'embeddedContext'),
    },
    session: {
      load: flag(sessionObj, 'load') || row.loadSession === true,
      resume: flag(sessionObj, 'resume'),
      list: flag(sessionObj, 'list'),
      fork: flag(sessionObj, 'fork'),
      close: flag(sessionObj, 'close'),
      additional_directories: flag(sessionObj, 'additional_directories', 'additionalDirectories'),
    },
    mcp: {
      http: flag(mcpObj, 'http'),
      sse: flag(mcpObj, 'sse'),
    },
    auth_methods: authMethods,
    ...(metaObj ? { meta: metaObj } : {}),
  }
}

/** Xerxes-bundled native skills that must NEVER leak into an external ACP runtime's catalog or completions. */
const XERXES_NATIVE_ONLY_SKILLS = new Set([
  'immortals-coordinator',
  'immortals-monitor',
  'immortals-narrator',
  'xerxes-plugin-creator',
])

export function isXerxesNativeOnlySkill(name: string, source?: string): boolean {
  const clean = name.trim().replace(/^\/+/, '').toLowerCase()
  if (!clean) return false
  if (XERXES_NATIVE_ONLY_SKILLS.has(clean)) return true
  if (clean.startsWith('immortals-')) return true
  if (source && /xerxes[-_\s]?bundled/i.test(source)) return true
  return false
}

/** Deduplicated, runtime-isolated ACP commands & skills derived from a session binding. */
export function resolveExternalAcpCatalog(
  binding: AcpSessionBinding | null | undefined,
): {
  readonly commands: readonly AcpAvailableCommand[]
  readonly skills: readonly AcpDiscoveredSkill[]
} {
  if (!binding?.connected || binding.transport === 'native') {
    return { commands: [], skills: [] }
  }
  const skillByKey = new Map<string, AcpDiscoveredSkill>()
  for (const skill of binding.discovered_skills ?? []) {
    if (isXerxesNativeOnlySkill(skill.name, skill.source)) continue
    const key = skill.name.trim().toLowerCase()
    if (!key || skillByKey.has(key)) continue
    skillByKey.set(key, {
      ...skill,
      slash_command: skill.slash_command?.startsWith('/')
        ? skill.slash_command
        : `/${skill.slash_command ?? skill.name}`,
    })
  }
  const commands: AcpAvailableCommand[] = []
  const seenCmd = new Set<string>()
  for (const cmd of binding.available_commands ?? []) {
    const cleanName = cmd.name.trim().replace(/^\/+/, '')
    const key = cleanName.toLowerCase()
    if (!key || isXerxesNativeOnlySkill(cleanName, cmd.source)) continue
    if (cmd.kind === 'skill') {
      if (!skillByKey.has(key)) {
        skillByKey.set(key, {
          name: cleanName,
          ...(cmd.description ? { description: cmd.description } : {}),
          ...(cmd.source ? { source: cmd.source } : {}),
          slash_command: `/${cleanName}`,
        })
      }
      continue
    }
    if (skillByKey.has(key) || seenCmd.has(key)) continue
    seenCmd.add(key)
    commands.push({ ...cmd, name: cleanName })
  }
  return {
    commands,
    skills: [...skillByKey.values()],
  }
}

/**
 * Whether Immortals UI entry points (FloatingRail, More menu, Rail + menu)
 * should be shown for this session. Always visible on native Xerxes, whenever
 * a swarm is active, or when connected to Antigravity / any ACP runtime that
 * advertises a `swarm` command.
 */
export function showsImmortalsForSession(
  snap: Pick<Snapshot, 'acpBinding' | 'swarm'>,
): boolean {
  const binding = snap.acpBinding
  const externalAcp = Boolean(binding?.connected && binding.transport !== 'native')
  if (!externalAcp) return true
  if (snap.swarm?.active) return true
  if (binding?.provider === 'antigravity') return true
  if (binding?.available_commands?.some(cmd => cmd.name.trim().replace(/^\/+/, '').toLowerCase() === 'swarm')) {
    return true
  }
  return false
}

/**
 * Decide whether a slash command typed while connected to an external ACP runtime
 * should bypass Xerxes's native slash handlers and forward directly to the ACP agent.
 * Keeps `/acp ...`, `/immortals ...`, and host-owned Immortals UI control commands
 * (`/swarm`, `/swarm board`, `/swarm ui`, `/swarm status`, `/swarm channels`,
 * `/swarm stop`, `/swarm kill <target>`) on the host while still forwarding
 * `/swarm <task description>` directly to ACP so starting an ACP swarm works.
 */
export function shouldForwardSlashToAcp(
  text: string,
  binding: AcpSessionBinding | null | undefined,
): boolean {
  if (!binding?.connected || binding.transport === 'native') return false
  const trimmed = text.trim()
  if (!trimmed.startsWith('/')) return false
  const [command = ''] = trimmed.split(/\s+/)
  const bare = command.replace(/^\/+/, '').toLowerCase()
  if (!bare || bare === 'acp' || bare === 'immortals') return false
  if (bare === 'swarm') {
    const swarmCmd = parseSwarmSlashCommand(trimmed)
    if (swarmCmd && swarmCmd.kind !== 'activate_task') return false
  }
  if (binding.available_commands?.some(cmd => cmd.name.trim().replace(/^\/+/, '').toLowerCase() === bare)) {
    return true
  }
  if (binding.discovered_skills?.some(skill => {
    const skillName = skill.name.trim().replace(/^\/+/, '').toLowerCase()
    const slashName = skill.slash_command?.trim().replace(/^\/+/, '').toLowerCase()
    return skillName === bare || slashName === bare
  })) {
    return true
  }
  return false
}

export function parseAcpSessionBinding(value: unknown): AcpSessionBinding | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Readonly<Record<string, unknown>>
  const agentId = typeof row.agent_id === 'string' ? row.agent_id.trim() : ''
  if (!agentId) return null
  const sessionId = typeof row.session_id === 'string' ? row.session_id.trim() : ''
  const name =
    typeof row.name === 'string' && row.name.trim()
      ? row.name.trim()
      : typeof row.agent_name === 'string' && row.agent_name.trim()
        ? row.agent_name.trim()
        : agentId
  const cwd = typeof row.cwd === 'string' ? row.cwd : ''
  const externalSessionId =
    typeof row.external_session_id === 'string' && row.external_session_id.trim()
      ? row.external_session_id.trim()
      : undefined
  const command = typeof row.command === 'string' && row.command.trim() ? row.command.trim() : undefined
  const error = typeof row.error === 'string' && row.error.trim() ? row.error.trim() : undefined
  const agentVersion =
    typeof row.agent_version === 'string' && row.agent_version.trim()
      ? row.agent_version.trim()
      : typeof row.agentVersion === 'string' && row.agentVersion.trim()
        ? row.agentVersion.trim()
        : undefined
  const agentTitle =
    typeof row.agent_title === 'string' && row.agent_title.trim()
      ? row.agent_title.trim()
      : typeof row.agentTitle === 'string' && row.agentTitle.trim()
        ? row.agentTitle.trim()
        : undefined
  const capabilities = parseAcpRuntimeCapabilities(row.capabilities)
  const configOptions = parseAcpConfigOptions(row.config_options ?? row.configOptions)
  const modelConfig = configOptions?.find(opt => opt.category === 'model' || opt.id === 'model')
  const modeConfig = configOptions?.find(opt => opt.category === 'mode' || opt.id === 'mode')
  const thoughtConfig = configOptions?.find(opt => opt.category === 'thought_level' || opt.id === 'thought_level')
  const modelConfigValue = typeof modelConfig?.current_value === 'string' && modelConfig.current_value ? modelConfig.current_value : undefined
  const modeConfigValue = typeof modeConfig?.current_value === 'string' && modeConfig.current_value ? modeConfig.current_value : undefined
  const thoughtConfigValue = typeof thoughtConfig?.current_value === 'string' && thoughtConfig.current_value ? thoughtConfig.current_value : undefined
  const currentModel =
    typeof row.current_model === 'string' && row.current_model.trim()
      ? row.current_model.trim()
      : typeof row.model_id === 'string' && row.model_id.trim()
        ? row.model_id.trim()
        : modelConfigValue
  const models =
    parseAcpExternalOptions(row.models)
    ?? (modelConfig?.options.length
      ? modelConfig.options.map(opt => ({
          id: opt.value,
          name: opt.name,
          ...(opt.description ? { description: opt.description } : {}),
        }))
      : undefined)
  const currentModelName =
    typeof row.current_model_name === 'string' && row.current_model_name.trim()
      ? row.current_model_name.trim()
      : currentModel
        ? models?.find(opt => opt.id === currentModel)?.name
        : undefined
  const currentMode =
    typeof row.current_mode === 'string' && row.current_mode.trim()
      ? row.current_mode.trim()
      : typeof row.mode_id === 'string' && row.mode_id.trim()
        ? row.mode_id.trim()
        : modeConfigValue
  const modes =
    parseAcpExternalOptions(row.modes)
    ?? (modeConfig?.options.length
      ? modeConfig.options.map(opt => ({
          id: opt.value,
          name: opt.name,
          ...(opt.description ? { description: opt.description } : {}),
        }))
      : undefined)
  const currentModeName =
    typeof row.current_mode_name === 'string' && row.current_mode_name.trim()
      ? row.current_mode_name.trim()
      : currentMode
        ? modes?.find(opt => opt.id === currentMode)?.name
        : undefined
  const modelConfigId =
    typeof row.model_config_id === 'string' && row.model_config_id.trim()
      ? row.model_config_id.trim()
      : modelConfig?.id
  const modeConfigId =
    typeof row.mode_config_id === 'string' && row.mode_config_id.trim()
      ? row.mode_config_id.trim()
      : modeConfig?.id
  const thoughtLevel =
    typeof row.thought_level === 'string' && row.thought_level.trim()
      ? row.thought_level.trim()
      : thoughtConfigValue
  const thoughtLevels =
    parseAcpExternalOptions(row.thought_levels)
    ?? (thoughtConfig?.options.length
      ? thoughtConfig.options.map(opt => ({
          id: opt.value,
          name: opt.name,
          ...(opt.description ? { description: opt.description } : {}),
        }))
      : undefined)
  const thoughtLevelConfigId =
    typeof row.thought_level_config_id === 'string' && row.thought_level_config_id.trim()
      ? row.thought_level_config_id.trim()
      : thoughtConfig?.id
  const availableCommands = parseAcpAvailableCommands(row.available_commands ?? row.availableCommands)
  const discoveredSkills = parseAcpDiscoveredSkills(row.discovered_skills ?? row.discoveredSkills)
  const additionalDirectories = Array.isArray(row.additional_directories)
    ? row.additional_directories
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        .map(item => item.trim())
    : undefined
  const sandbox = row.sandbox === true ? true : undefined
  const agentOverride =
    typeof row.agent_override === 'string' && row.agent_override.trim()
      ? row.agent_override.trim()
      : undefined
  const mcpServerNames = Array.isArray(row.mcp_server_names)
    ? row.mcp_server_names
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        .map(item => item.trim())
    : undefined
  return {
    session_id: sessionId,
    agent_id: agentId,
    name,
    provider: parseAcpProvider(row.provider),
    transport: parseAcpTransport(row.transport),
    cwd,
    ...(externalSessionId ? { external_session_id: externalSessionId } : {}),
    connected: row.connected === true,
    ...(command ? { command } : {}),
    ...(error ? { error } : {}),
    ...(agentVersion ? { agent_version: agentVersion } : {}),
    ...(agentTitle ? { agent_title: agentTitle } : {}),
    ...(capabilities ? { capabilities } : {}),
    ...(currentModel ? { current_model: currentModel } : {}),
    ...(currentModelName ? { current_model_name: currentModelName } : {}),
    ...(models ? { models } : {}),
    ...(currentMode ? { current_mode: currentMode } : {}),
    ...(currentModeName ? { current_mode_name: currentModeName } : {}),
    ...(modes ? { modes } : {}),
    ...(modelConfigId ? { model_config_id: modelConfigId } : {}),
    ...(modeConfigId ? { mode_config_id: modeConfigId } : {}),
    ...(configOptions ? { config_options: configOptions } : {}),
    ...(thoughtLevel ? { thought_level: thoughtLevel } : {}),
    ...(thoughtLevels ? { thought_levels: thoughtLevels } : {}),
    ...(thoughtLevelConfigId ? { thought_level_config_id: thoughtLevelConfigId } : {}),
    ...(availableCommands ? { available_commands: availableCommands } : {}),
    ...(discoveredSkills ? { discovered_skills: discoveredSkills } : {}),
    ...(additionalDirectories && additionalDirectories.length > 0 ? { additional_directories: additionalDirectories } : {}),
    ...(sandbox ? { sandbox: true } : {}),
    ...(agentOverride ? { agent_override: agentOverride } : {}),
    ...(mcpServerNames && mcpServerNames.length > 0 ? { mcp_server_names: mcpServerNames } : {}),
  }
}

export function parseAcpCatalogResult(
  value: unknown,
  fallbackBinding: AcpSessionBinding | null = null,
): AcpCatalogResult | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Readonly<Record<string, unknown>>
  const binding = parseAcpSessionBinding(row.binding) ?? fallbackBinding
  const connected = typeof row.connected === 'boolean' ? row.connected : Boolean(binding?.connected)
  const capabilities = parseAcpRuntimeCapabilities(row.capabilities) ?? binding?.capabilities
  const configOptions = parseAcpConfigOptions(row.config_options ?? row.configOptions) ?? binding?.config_options ?? []
  const rawCommands = parseAcpAvailableCommands(row.commands ?? row.available_commands ?? row.availableCommands)
  const rawSkills = parseAcpDiscoveredSkills(row.skills ?? row.discovered_skills ?? row.discoveredSkills)
  const mergedBinding: AcpSessionBinding | null = binding
    ? {
        ...binding,
        ...(capabilities ? { capabilities } : {}),
        ...(configOptions.length > 0 ? { config_options: configOptions } : {}),
        ...(rawCommands ? { available_commands: rawCommands } : {}),
        ...(rawSkills ? { discovered_skills: rawSkills } : {}),
      }
    : null
  const resolved = resolveExternalAcpCatalog(mergedBinding)
  return {
    ok: row.ok !== false,
    connected,
    ...(mergedBinding ? { binding: mergedBinding } : {}),
    commands: resolved.commands,
    skills: resolved.skills,
    ...(capabilities ? { capabilities } : {}),
    config_options: configOptions,
  }
}

/** Slow fleet cadence while agent-family tools run — snapshots lag the event. */
const FLEET_POLL_MS = 2_000

/**
 * Wire names whose execution can spawn, stop, or reap a subagent. The fleet
 * rail reads the parent's `subagent_snapshots`, which only move while one of
 * these runs — every other tool call leaves the panel unchanged, so only
 * these wake the poller.
 */
const AGENT_FAMILY_TOOLS = new Set([
  'agent', 'agenttool',
  'spawnagents',
  'workflow', 'run_workflow', 'runworkflow',
  'invoke_subagent', 'invokesubagent',
  'manage_subagents', 'managesubagents',
  'handoff', 'handofftool',
  'sendmessage', 'sendmessagetool',
  'taskcreate', 'taskcreatetool',
  'taskstop', 'taskstoptool',
  'taskupdate', 'taskupdatetool',
  'awaitagents', 'resetagent',
])

export function isAgentFamilyTool(name: unknown): boolean {
  if (typeof name !== 'string' || !name) return false
  const tail = (name.split(/[.:/]/).pop() ?? name).trim().toLowerCase()
  return AGENT_FAMILY_TOOLS.has(tail) || AGENT_FAMILY_TOOLS.has(tail.replace(/[_-]+/g, ''))
}

/** The spawn-capable subset — these open members on the turn's agents card. */
const SPAWN_TOOLS = new Set([
  'agent', 'agenttool',
  'spawnagents',
  'workflow', 'run_workflow', 'runworkflow',
  'invoke_subagent', 'invokesubagent',
  'taskcreate', 'taskcreatetool',
  'handoff', 'handofftool',
])

export function isSpawnTool(name: unknown): boolean {
  if (typeof name !== 'string' || !name) return false
  const tail = (name.split(/[.:/]/).pop() ?? name).trim().toLowerCase()
  return SPAWN_TOOLS.has(tail) || SPAWN_TOOLS.has(tail.replace(/[_-]+/g, ''))
}

/** Snapshot statuses fold onto the card's display vocabulary. */
/** Tools whose success changes the session goal. */
const GOAL_WRITE_TOOLS = new Set(['create_goal', 'update_goal'])


/**
 * Members a spawn call opens on the agents card, parsed from its arguments.
 * SpawnAgents carries an `agents` batch; the single-spawn tools carry one
 * title-ish field. Keys are call-local; fleet sync matches on title.
 */
/** What the update prompt shows; the main process's state, validated. */
export interface AppUpdateView {
  readonly phase: 'checking' | 'current' | 'available' | 'downloading' | 'restarting' | 'error'
  readonly version?: string
  readonly notes?: string
  readonly pageUrl?: string
  readonly publishedAt?: string
  readonly installable?: boolean
  readonly reason?: string
  readonly received?: number
  readonly total?: number
  readonly message?: string
}

export function appUpdateView(value: unknown): AppUpdateView | null {
  if (!value || typeof value !== 'object') return null
  const state = value as Record<string, unknown>
  const phase = state.phase
  if (phase !== 'checking' && phase !== 'current' && phase !== 'available' && phase !== 'downloading' && phase !== 'restarting' && phase !== 'error') return null
  const release = state.release && typeof state.release === 'object' ? state.release as Record<string, unknown> : {}
  const text = (field: unknown) => typeof field === 'string' ? field : undefined
  const count = (field: unknown) => typeof field === 'number' && Number.isFinite(field) ? field : undefined
  const view: Record<string, unknown> = { phase }
  for (const [key, field] of [['version', text(release.version) ?? text(state.version)], ['notes', text(release.notes)], ['pageUrl', text(release.pageUrl)], ['publishedAt', text(release.publishedAt)], ['reason', text(state.reason)], ['message', text(state.message)]] as const) {
    if (field !== undefined) view[key] = field
  }
  if (typeof state.installable === 'boolean') view.installable = state.installable
  if (count(state.received) !== undefined) view.received = count(state.received)
  if (count(state.total) !== undefined) view.total = count(state.total)
  return view as unknown as AppUpdateView
}

/**
 * An SSH conversation whose prompts run through this computer (the Mac's
 * ChatGPT or Claude Code sign-in) instead of on the host. Shown on the
 * composer so it is never silent; details live in the workspace panel.
 */
export interface ProviderRelayView {
  /** The host saved this conversation as running on this computer. */
  readonly bound: boolean
  /** This computer is carrying its requests now (false: access ended). */
  readonly live: boolean
  readonly destination?: string
  readonly profile?: string
  readonly model?: string
  readonly requests: number
  readonly inFlight: number
  readonly lastAt?: number
  readonly lastError?: string
  readonly expiresAt?: number
  /** Not live, and this window is binding it again (retrying until it is). */
  readonly reconnecting?: boolean
  readonly lastFailure?: string
  /** Not relayed: the host runs this task on its own sign-in (`bound` is false). */
  readonly onHost?: { readonly profile: string; readonly model: string }
}

/**
 * A profile whose login lives on each machine (the host reports `signs_in`).
 * In an SSH window it runs on this computer, through the relay, not on the host.
 */
export function runsOnThisComputer(frame: { readonly storageScope?: string }, row: { readonly signsIn: boolean; readonly ready?: boolean } | undefined): boolean {
  // A host whose own sign-in works runs it itself.
  return Boolean(frame.storageScope?.startsWith('ssh:') && row?.signsIn && row.ready !== true)
}

/**
 * How a Settings provider row reads. In an SSH task that runs through this
 * computer, the profile carrying it is the one in use ("via this Mac") and the
 * host's active profile is only its default for other tasks; without that,
 * the list said "zai active · in use" while the task ran on ChatGPT here.
 */
export function providerRowState(frame: Pick<Snapshot, 'storageScope' | 'providerRelay'>, row: { readonly name: string; readonly active: boolean; readonly signsIn: boolean }): { inUse: boolean; viaMac: boolean; chip: string | null; status: string } {
  const viaMac = runsOnThisComputer(frame, row)
  const relay = frame.providerRelay
  if (relay?.bound) {
    const carrying = relay.profile === row.name
    if (carrying) return { inUse: true, viaMac: true, chip: relay.live ? 'via this Mac' : relay.reconnecting ? 'reconnecting' : 'Mac not connected', status: relay.live ? 'in use' : relay.reconnecting ? 'reconnecting to this computer' : 'needs this computer' }
    return { inUse: false, viaMac, chip: row.active ? 'host default' : null, status: viaMac ? 'runs on this Mac' : 'saved' }
  }
  if (row.active && !viaMac) return { inUse: true, viaMac: false, chip: 'active', status: 'in use' }
  return { inUse: false, viaMac, chip: row.active ? 'host default' : null, status: viaMac ? 'runs on this Mac' : 'saved' }
}

export function providerRelayView(value: unknown): ProviderRelayView | null {
  if (!value || typeof value !== 'object') return null
  const row = value as Record<string, unknown>
  const text = (field: unknown) => typeof field === 'string' && field ? field : undefined
  const host = row.onHost && typeof row.onHost === 'object' ? row.onHost as Record<string, unknown> : undefined
  if (row.ok === true && row.bound !== true && host && text(host.profile) && text(host.model)) {
    return { bound: false, live: true, requests: 0, inFlight: 0, profile: text(host.profile)!, model: text(host.model)!,
      onHost: { profile: text(host.profile)!, model: text(host.model)! }, ...(text(row.destination) ? { destination: text(row.destination)! } : {}) }
  }
  if (row.ok !== true || row.bound !== true) return null
  const count = (field: unknown) => typeof field === 'number' && Number.isFinite(field) && field >= 0 ? field : undefined
  const view: Record<string, unknown> = { bound: true, live: row.live === true, requests: count(row.requests) ?? 0, inFlight: count(row.inFlight) ?? 0 }
  for (const [key, field] of [['destination', text(row.destination)], ['profile', text(row.profile)], ['model', text(row.model)], ['lastError', text(row.lastError)], ['lastFailure', text(row.lastFailure)]] as const) if (field) view[key] = field
  if (row.reconnecting === true && row.live !== true) view.reconnecting = true
  for (const [key, field] of [['lastAt', count(row.lastAt)], ['expiresAt', count(row.expiresAt)]] as const) if (field !== undefined) view[key] = field
  return view as unknown as ProviderRelayView
}

/** The daemon's delegation mode; an older runtime without one is auto. */
function delegationModeValue(value: unknown): 'off' | 'auto' | 'eager' {
  return value === 'off' || value === 'eager' ? value : 'auto'
}

/** How far back (in 100-action pages) opening a chat looks for the person's last message. */
const REVEAL_USER_MESSAGE_MAX_PAGES = 20

/** Minimum spacing of routine (streaming) store updates: 20 a second. */
const STREAM_FRAME_MS = 50

/** The daemon's refusal to resume a session whose turn is running under another connection. */
const RUNNING_ELSEWHERE = /still running a turn under another connection/i

export function spawnMembersOf(name: unknown, args: unknown, callId: string): AgentMember[] {
  const parsed = parseArgs(args)
  const promptOf = (value: Record<string, unknown>): string =>
    str(value.prompt) || str(value.Prompt) || str(value.task) || str(value.Task)
  const baseAgentOf = (value: Record<string, unknown>): string =>
    str(value.subagent_type)
    || str(value.subagent_name)
    || str(value.SubagentName)
    || str(value.TypeName)
    || str(value.type_name)
    || str(value.Role)
    || str(value.role)
    || str(value.target_agent)
    || str(value.agent)
  const label = (value: Record<string, unknown>, index: number): string =>
    str(value.title)
    || str(value.Title)
    || str(value.name)
    || str(value.Name)
    || str(value.subagent_name)
    || str(value.SubagentName)
    || str(value.Role)
    || str(value.role)
    || str(value.description)
    || str(value.Description)
    || str(value.TypeName)
    || str(value.type_name)
    || promptOf(value).slice(0, 60)
    || `agent ${index + 1}`
  const raw = parsed.agents
  // A native workflow's agents are unknown until its script starts them; they join
  // the card from their own events, tagged with the run and phase.
  if (typeof name === 'string' && /(^|[.:])workflow$/i.test(name)) return []
  if (Array.isArray(raw) && raw.length) {
    return raw.map((item, index) => {
      const record = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
      return {
        key: `${callId}:${index}`,
        title: label(record, index),
        status: 'working',
        baseAgent: str(record.agent) || baseAgentOf(record),
        prompt: promptOf(record),
        model: str(record.model) || str(record.Model),
        providerProfile: str(record.provider_profile),
        reasoningEffort: str(record.reasoning_effort),
      }
    })
  }
  if (!isSpawnTool(name)) return []
  return [{
    key: `${callId}:0`,
    title: label(parsed, 0),
    status: 'working',
    baseAgent: baseAgentOf(parsed),
    prompt: promptOf(parsed),
    model: str(parsed.model) || str(parsed.Model),
    providerProfile: str(parsed.provider_profile),
    reasoningEffort: str(parsed.reasoning_effort),
  }]
}

function agentReceiptRows(value: unknown): Record<string, unknown>[] {
  if (typeof value === 'string') { try {value=JSON.parse(value)} catch {return []} }
  if (!value || typeof value !== 'object') return []
  const object=value as Record<string, unknown>
  const rows=Array.isArray(value)
    ? value
    : Array.isArray(object.agents)
      ? object.agents
      : (object.id || object.conversation_id || object.conversationId)
        ? [object]
        : []
  return rows.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === 'object' && !Array.isArray(row))
}

/** The session context menu (mockup 08), anchored at pointer coordinates. */
export interface SessionMenuState {
  readonly id: string
  /** Daemon session key — what `session.title` mutates through. */
  readonly key: string
  readonly title: string
  readonly x: number
  readonly y: number
  readonly renaming?: boolean
  readonly pending?: boolean
  readonly error?: string
}

export interface ReasoningLevelRow {
  readonly effort: string
  readonly description: string
}

/** Estimated split of the next request's token budget (daemon-computed). */
export interface ContextBreakdown {
  readonly systemPromptTokens: number
  readonly toolsTokens: number
  readonly messagesTokens: number
  readonly totalTokens: number
  readonly contextLimit: number
}

export interface Snapshot {
  readonly swarm?: SwarmStatusSnapshot | null
  readonly swarmMessages?: readonly SwarmBoardMessage[]
  readonly swarmInspect?: SwarmMemberInspectResult | null
  readonly swarmInspectTarget?: string
  readonly swarmLoading?: boolean
  readonly swarmError?: string | null
  readonly swarmOpen?: boolean
  readonly swarmOpenRevision?: number
  readonly sideQuestionTurns?: readonly SideQuestionTurn[]
  readonly sideQuestionStreaming?: boolean
  readonly sideQuestionError?: string | null
  readonly sideQuestionOpen?: boolean
  readonly sideQuestionOpenRevision?: number
  /** Persisted Side Question model override (`''` = follow session model). */
  readonly sideQuestionModel?: string
  /** Discovered external ACP agent runtimes (`acp.agents`). */
  readonly acpAgents?: readonly AcpExternalAgentInfo[]
  /** Active external ACP runtime bound to the current session, or null when using native Xerxes. */
  readonly acpBinding?: AcpSessionBinding | null
  /** Persisted external ACP runtime from a prior run on a reopened session (when not connected yet). */
  readonly savedAcpBinding?: AcpSessionBinding | null
  readonly acpConnecting?: boolean
  readonly acpError?: string | null
  readonly acpAuthRequired?: boolean
  readonly acpAuthMethods?: readonly AcpAuthMethodInfo[]
  readonly learning?: LearningList | null
  readonly learningCounts?: LearningCounts
  readonly learningLoading?: boolean
  readonly learningError?: string
  readonly learningReviewBusy?: boolean
  readonly learningResults?: Readonly<Record<string, LearningResult>>
  readonly learningOpen?: boolean
  readonly learningOpenRevision?: number
  readonly learningProposalId?: string
  readonly connection: Connection
  /** True before any workspace folder is chosen — the shell shows the gate. */
  readonly noWorkspace: boolean
  readonly submissionPending?: boolean
  readonly cwd: string
  readonly model: string
  /** Agent preset fixed for the current session after its first turn. */
  readonly currentAgentPreset: string
  /** Live DSH-style preset roster used by settings and the new-task seat. */
  readonly agentPresets: readonly AgentPreset[]
  readonly agentPresetsError: string | null
  /** Messaging gateways from `channel.list`, kept fresh by `channel_status` broadcasts. */
  readonly channels: readonly ChannelRow[]
  readonly channelsAvailable: boolean
  readonly channelsConfigured: boolean
  readonly channelsLoading?: boolean
  readonly channelsError?: string
  /** Daemon-tracked terminals from `terminal.list` (owner: the bound session). */
  readonly terminals: readonly TerminalRow[]
  readonly terminalsLoading: boolean
  readonly terminalsError: string | null
  /** Transcript search (`session.search`) state for the search overlay. */
  readonly searchOpen: boolean
  readonly searchResults: readonly SessionSearchHit[]
  readonly searchStats: SessionSearchStats | null
  readonly searchSearching: boolean
  readonly searchError: string | null
  /** Git branch of the workspace, from the daemon's initialize — '' when unknown. */
  readonly branch: string
  /** Actionable initialize-handshake mismatch; null when app and daemon agree. */
  readonly daemonWarning: string | null
  readonly runtimeUpdate?: 'checking' | 'waiting' | 'restarting' | 'failed'
  readonly runtimeUpdateMessage?: string
  /** What keeps a queued runtime update waiting, as the runtime names it. */
  readonly runtimeBlockers?: readonly string[]
  /** Session cost estimate from the daemon wire (USD); null when unpriced. */
  readonly costUsd: number | null
  /** `subscription` when a plan pays for this session's calls: views say "Included in plan", not a dollar figure. */
  readonly costBasis: 'subscription' | null
  /** MCP server statuses from the daemon; empty until fetched or without servers. */
  readonly mcpStatus: Readonly<Record<string, McpServerStatus>>
  readonly models: readonly ModelChoice[]
  readonly contextTokens: number | null
  /** A task in this window tried to use the computer, so its macOS permissions are worth asking for. */
  readonly computerUseWanted: boolean
  readonly contextMax: number | null
  /** Session-average provider time to first output token. */
  readonly ttftMs: number | null
  /** Latest provider round's output decode rate. */
  readonly tokensPerSecond: number | null
  /** Cumulative provider and tool telemetry for the bound session. */
  readonly llmDurationMs: number
  readonly llmSteps: number
  readonly toolDurationMs: number
  readonly toolSteps: number
  readonly inputTokens: number
  readonly outputTokens: number
  /** Current live phase, used to advance the active timing bucket. */
  readonly metricPhase: 'llm' | 'tool' | null
  readonly metricPhaseStartedAt: number | null
  /** Latest provider-reported input-cache hit share, from 0 through 1. */
  readonly cacheHitRate: number | null
  /** Session prompt tokens served from the provider cache, and written into it (billed above plain input). */
  readonly cacheReadTokens: number | null
  readonly cacheWriteTokens: number | null
  readonly sessions: readonly SessionRow[]
  readonly live: readonly SessionRow[]
  readonly fleet: readonly SessionRow[]
  /** Repeatable workflows the runtime observed and proposed as skills. */
  readonly skillSuggestions: readonly SkillSuggestion[]
  /** Policy-gated declarative creator actions for the bound session. */
  readonly creatorTrace: readonly CreatorTrace[]
  /** Daemon-backgrounded turns (bg-* sessions) currently working. */
  readonly backgroundJobs: readonly BackgroundJob[]
  readonly currentId: string
  /** Successful explicit resumes, including reopening the current session. */
  readonly sessionOpenRevision: number
  readonly currentTitle: string
  /** The connection's bound daemon session key — what key-scoped RPCs address. */
  readonly sessionKey: string
  readonly goal: string
  readonly approval: DesktopApproval | null
  readonly question: import('./types.js').TaskQuestion | null
  /**
   * The person hid the pending approval or question card. The request is
   * still pending and still shown as "needs input"; only the card is folded
   * away until they reopen it or a new request arrives.
   */
  readonly interactionHidden: boolean
  readonly planMode: boolean
  readonly turnActive: boolean
  readonly networkRetrying?: boolean
  readonly liveFocus?: string | null
  /**
   * The runtime is compacting this task's history: its latest progress line.
   * The task is busy and takes no new turn until it finishes, but nothing
   * said so — the view showed no activity and a send went nowhere.
   */
  readonly compacting?: string
  readonly turnFailed: boolean
  readonly turnSeconds: number
  /** Epoch ms when the active turn started; drives isolated leaf `<ElapsedClock>` without global 1s re-renders. */
  readonly turnStartedAt?: number | null
  readonly blocks: readonly Block[]
  readonly historyMore?: boolean
  readonly historyLoading?: boolean
  readonly historyError?: string | null
  readonly error: string | null
  // ── workspace surfaces ──
  readonly tab: WorkspaceTab
  readonly turnCount: number
  readonly queue: readonly QueueItem[]
  readonly changes: readonly DiffFile[]
  readonly changesKept: boolean
  /** The Session edits file the rail asked to show; `at` re-triggers the same file. */
  readonly changesFocus?: { readonly path: string; readonly at: number } | null
  readonly todos: readonly TodoItem[] | null
  readonly plan: PlanState | null
  readonly log: readonly LogEntry[]
  readonly failed: FailedTurn | null
  /**
   * Escape pressed once over a running turn. A second press inside
   * {@link STOP_CONFIRM_MS} stops it; a stray one (reaching for a screenshot
   * shortcut) only shows the hint instead of cancelling every agent.
   */
  readonly stopArmed: boolean
  // ── overlays ──
  readonly settingsOpen: boolean
  readonly settingsTab: SettingsTab
  readonly paletteOpen: boolean
  readonly pickerOpen: boolean
  /** The single model/effort chip's dropdown — rows drill into the pickers. */
  readonly modelMenuOpen: boolean
  /** A newer desktop release, and where installing it has got to (null when current). */
  readonly appUpdate: AppUpdateView | null
  /** Set while an SSH conversation's prompts run through this computer. */
  readonly providerRelay: ProviderRelayView | null
  /** The shell a code block's Run just used; the Terminal tab shows it (revision re-triggers). */
  readonly terminalFocus?: { readonly id: string; readonly revision: number }
  /** How eagerly this conversation fans out to agents (the composer's Agents chip). */
  readonly delegationMode: 'off' | 'auto' | 'eager'
  readonly delegationMenuOpen: boolean
  /** Context-usage popover with the estimated token split. */
  readonly contextMenuOpen: boolean
  readonly contextBreakdown: ContextBreakdown | null
  readonly contextBreakdownLoading: boolean
  /** Reasoning-effort picker state — levels come from the daemon per model. */
  readonly reasoningPickerOpen: boolean
  readonly reasoningEffort: string
  readonly reasoningLevels: readonly ReasoningLevelRow[]
  readonly reasoningDefault: string
  readonly reasoningNote: string
  readonly reasoningLoading: boolean
  readonly wsMenuOpen: boolean
  readonly workspaceDirectories?: readonly string[]
  /**
   * Latest-message time per session id, kept across refreshes. The lists
   * above drop the open session, so this is where its sort key survives.
   */
  readonly sessionActivity?: Readonly<Record<string, number>>
  readonly contexts?: readonly WorkspaceContext[]
  readonly storageScope?: string
  /** The sidebar name of this view's host: its SSH alias; absent for this Mac. */
  readonly contextHost?: string
  readonly workspaceBusy: boolean
  readonly workspaceError: string | null
  readonly sessionMenu: SessionMenuState | null
  /** Display choice: show reasoning trails in the activity feed. */
  readonly streamThinking: boolean
  /** The ⌘N new-task modal (mockup 18). */
  readonly taskModalOpen: boolean
  // ── settings data ──
  readonly providers: readonly ProviderRow[]
  readonly providerError: string
  /**
   * A session switch or a new task is in flight. Distinct from `connection`:
   * both used to reuse 'connecting', so deliberate navigation impersonated
   * a dropped daemon.
   */
  readonly navigating: 'session' | 'fresh' | null
  /** provider_list is in flight — distinct from "there are none". */
  readonly providersLoading: boolean
  /** fetch_models is in flight — distinct from "zero models discovered". */
  readonly modelsLoading: boolean
  /** Why discovery failed, shown where the person waiting can see it. */
  readonly modelsError: string | null
  readonly providerSwitching: string | null
  readonly providerSwitchError: string | null
  /** Private, Hybrid or Cloud (`run_mode.get`); null until loaded or on a runtime without it. */
  readonly runMode: RunModeView | null
  readonly runModeError: string
  /** A run_mode.set is in flight. */
  readonly runModeSaving: boolean
  /** Bonsai on the workspace host: engine, model and the setup job. */
  readonly localSetup: LocalSetupView
  /** Model catalogs and editable capacities cached for each provider profile. */
  readonly providerModels: Readonly<Record<string, readonly CachedModel[]>>
  readonly providerModelLoading: readonly string[]
  readonly providerModelWarnings: Readonly<Record<string, string>>
  /** The daemon registry's adapter catalog — the add/edit form's dropdown. */
  readonly providerTypes: readonly ProviderTypeRow[]
  /** The daemon's slash catalog — name/description pairs from commands.catalog. */
  readonly commands: readonly { readonly name: string; readonly description: string }[]
  readonly permissionMode: string
  readonly permissionUpdating: boolean
  readonly permissionError: string | null
  readonly snippets: Readonly<Record<string, string>>
}

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

const num = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null)

function cachedModelsFromResult(result: Record<string, unknown>): CachedModel[] {
  const catalog = Array.isArray(result.catalog) ? result.catalog : []
  const byId = new Map<string, CachedModel>()
  for (const value of catalog) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) continue
    const record = value as Record<string, unknown>
    const id = str(record.id).trim()
    if (!id) continue
    const contextLimit = num(record.context_limit)
    const maxOutputTokens = num(record.max_output_tokens)
    const contextSource = capabilitySource(record.context_source)
    const outputSource = capabilitySource(record.output_source)
    byId.set(id, {
      id,
      ...(contextLimit === null ? {} : { contextLimit }),
      ...(contextSource === undefined ? {} : { contextSource }),
      ...(maxOutputTokens === null ? {} : { maxOutputTokens }),
      ...(outputSource === undefined ? {} : { outputSource }),
      overridden: record.overridden === true,
    })
  }
  const ids = Array.isArray(result.models)
    ? result.models.map(value => str(value).trim()).filter(Boolean)
    : []
  for (const id of ids) {
    if (!byId.has(id)) byId.set(id, { id, overridden: false })
  }
  return [...byId.values()]
}

function agentPresetsFromResult(result: Record<string, unknown>): AgentPreset[] {
  const rows = Array.isArray(result.presets) ? result.presets : []
  return rows.flatMap(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const row = value as Record<string, unknown>
    const id = str(row.id).trim()
    if (!id) return []
    const trustValue = str(row.trust)
    const trust: AgentPreset['trust'] = trustValue === 'system' || trustValue === 'project' ? trustValue : 'user'
    return [{
      id,
      name: str(row.name) || id,
      description: str(row.description),
      trust,
      isDefault: row.is_default === true,
      manageable: row.manageable === true,
      ...(str(row.broken) ? { broken: str(row.broken) } : {}),
    }]
  })
}

function capabilitySource(
  value: unknown,
): CachedModel['contextSource'] | undefined {
  return value === 'catalog' || value === 'override' || value === 'provider' || value === 'unknown'
    ? value
    : undefined
}

/** Fold a `channel.list` / `channel_status` payload onto the channels state. */
function channelStatusFrom(result: Record<string, unknown>): ChannelStatus {
  const rows = Array.isArray(result.channels) ? result.channels : []
  const channels: ChannelRow[] = rows.flatMap(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const row = value as Record<string, unknown>
    const name = str(row.name).trim()
    if (!name) return []
    return [{
      name,
      adapterName: str(row.adapter_name) || name,
      enabled: row.enabled === true,
      ...(str(row.last_operation) ? { lastOperation: str(row.last_operation) } : {}),
      ...(str(row.last_error) ? { lastError: str(row.last_error) } : {}),
    }]
  })
  return {
    channels,
    available: result.channels_available === true,
    configured: result.channels_configured === true,
  }
}

/** The terminal registry's rows pass the wire camelCase — parse them as-is. */
function terminalRowOf(value: unknown): TerminalRow | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  const id = str(row.id).trim()
  if (!id) return null
  const pid = num(row.pid)
  const endedAt = num(row.endedAt)
  return {
    id,
    kind: str(row.kind) || 'background',
    label: str(row.label),
    command: str(row.command),
    cwd: str(row.cwd),
    ...(pid === null ? {} : { pid }),
    running: row.running === true,
    // Required on the wire (registry rows always carry the start epoch).
    startedAt: Math.max(0, num(row.startedAt) ?? 0),
    ...(endedAt === null ? {} : { endedAt }),
    exitCode: typeof row.exitCode === 'number' ? row.exitCode : null,
    outputChars: Math.max(0, num(row.outputChars) ?? 0),
    canWrite: row.canWrite === true,
    canInterrupt: row.canInterrupt === true,
    canKill: row.canKill === true,
  }
}

function terminalsFromResult(result: Record<string, unknown>): TerminalRow[] {
  const rows = Array.isArray(result.terminals) ? result.terminals : []
  return rows.map(terminalRowOf).filter((row): row is TerminalRow => row !== null)
}

function terminalDetailOf(value: unknown): TerminalDetail | null {
  const row = terminalRowOf(value)
  if (!row) return null
  const record = (value ?? {}) as Record<string, unknown>
  return {
    ...row,
    output: str(record.output),
    outputTruncated: record.outputTruncated === true,
  }
}

function searchHitOf(value: unknown): SessionSearchHit | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  const sessionId = str(row.session_id).trim()
  if (!sessionId) return null
  return {
    sessionId,
    messageIndex: Math.max(0, num(row.message_index) ?? 0),
    role: str(row.role),
    excerpt: str(row.excerpt),
    title: str(row.title),
    updatedAt: str(row.updated_at),
  }
}

function searchStatsOf(value: unknown): SessionSearchStats | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  return {
    sessions: Math.max(0, num(row.sessions) ?? 0),
    indexedMessages: Math.max(0, num(row.indexed_messages) ?? 0),
    searchableMessages: Math.max(0, num(row.searchable_messages) ?? 0),
  }
}

/** An image pasted or dropped into the message box: base64 bytes for `turn.submit`'s `images`. */
export interface ComposerImage {
  readonly id: string
  readonly mediaType: string
  /** Base64 without the `data:` prefix, as the runtime's image validator expects. */
  readonly data: string
  readonly name: string
}

function clientHandshake(): Record<string, unknown> {
  return {
    client_version: DESKTOP_VERSION,
    client_protocol: DESKTOP_DAEMON_PROTOCOL,
    ...(EXPECTED_DAEMON_BUILD_ID
      ? { expected_daemon_build_id: EXPECTED_DAEMON_BUILD_ID }
      : {}),
  }
}

/** Epoch ms from an ISO string or epoch seconds; undefined when unreadable. */
function epochMsOf(when: unknown): number | undefined {
  if (typeof when === 'number' && Number.isFinite(when)) return when * 1000
  if (typeof when === 'string') {
    const parsed = Date.parse(when)
    if (!Number.isNaN(parsed)) return parsed
  }
  return undefined
}

function ageOf(when: unknown): string {
  // Saved rows carry updated_at as an ISO string; live rows carry
  // last_active as epoch seconds. Accept both or show nothing rather than
  // mislabel.
  let epoch = typeof when === 'number' && Number.isFinite(when) ? when : NaN
  if (typeof when === 'string') {
    const parsed = Date.parse(when)
    if (!Number.isNaN(parsed)) epoch = parsed / 1000
  }
  if (Number.isNaN(epoch)) return ''
  const minutes = Math.floor((Date.now() - epoch * 1000) / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

/**
 * Grouping for a model id when the runtime names no profile (older
 * runtimes): the id's own `vendor/` prefix, else one plain group. Never a
 * family guessed from the name.
 */
export function providerOf(id: string): string {
  const slashed = id.split('/')[0]
  return slashed && id.includes('/') ? slashed : 'Models'
}

/**
 * Picker rows from what the runtime reported: each model's own name and
 * description when the provider gives them, grouped under the serving
 * profile's label. The id-based family guess is only for older runtimes.
 */
function toChoices(ids: readonly string[], catalog?: unknown, group = ''): ModelChoice[] {
  const entries = new Map<string, Record<string, unknown>>()
  if (Array.isArray(catalog)) {
    for (const item of catalog) {
      if (item && typeof item === 'object' && typeof (item as Record<string, unknown>).id === 'string') entries.set((item as Record<string, unknown>).id as string, item as Record<string, unknown>)
    }
  }
  return ids.filter(Boolean).map(id => {
    const entry = entries.get(id)
    const label = str(entry?.display_name)
    const hint = str(entry?.description)
    return { id, provider: group || providerOf(id), ...(label ? { label } : {}), ...(hint ? { hint } : {}) }
  })
}

function skillSuggestionOf(value: unknown): SkillSuggestion | null {
  const row = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
  const skillName = str(row.skill_name ?? row.skillName)
  if (!skillName) return null
  const rawTools = row.unique_tools ?? row.uniqueTools
  return {
    skillName,
    description: str(row.description),
    version: str(row.version),
    sourcePath: str(row.source_path ?? row.sourcePath),
    toolCount: num(row.tool_count ?? row.toolCount) ?? 0,
    uniqueTools: Array.isArray(rawTools) ? rawTools.map(tool => str(tool)).filter(Boolean) : [],
  }
}

function creatorTraceOf(value: unknown): CreatorTrace | null {
  const row = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
  const action = str(row.action)
  const status = row.status === 'error' ? 'error' : row.status === 'ok' ? 'ok' : null
  if (!action || !status) return null
  return {
    action,
    status,
    name: str(row.name),
    version: str(row.version),
    detail: str(row.detail),
    at: str(row.at),
  }
}

const wireRow = (row: Record<string, unknown>, currentId: string): SessionRow | null => {
  const id = str(row.id ?? row.session_id ?? row.key)
  if (!id) return null
  const count = num(row.message_count) ?? num(row.messages) ?? 0
  const turns = num(row.turn_count) ?? 0
  const kind = str(row.kind ?? row.session_kind) === 'subagent' ? 'subagent' : 'main'
  // Untitled rows get the short id — never a wall of 'Untitled'. The lazy
  // enrichment below replaces it with a first-message snippet whenever the
  // session is still loaded in the daemon and can answer session.status.
  const hasTitle = Boolean(str(row.title))
  return {
    id,
    key: str(row.key) || id,
    title: str(row.title) || `#${id.slice(0, 6)}`,
    status: str(row.status) || '',
    age: ageOf(row.updated_at ?? row.last_active),
    ...(epochMsOf(row.updated_at ?? row.last_active) === undefined ? {} : { activeAt: epochMsOf(row.updated_at ?? row.last_active)! }),
    current: id === currentId,
    kind,
    turns,
    messages: count,
    cwd: str(row.cwd),
    untitled: !hasTitle,
  } satisfies SessionRow
}

interface SavedWire {
  id?: unknown
  session_id?: unknown
  key?: unknown
  title?: unknown
  status?: unknown
  updated_at?: unknown
  last_active?: unknown
  message_count?: unknown
  messages?: unknown
  kind?: unknown
  session_kind?: unknown
  turn_count?: unknown
  model?: unknown
  cwd?: unknown
}

function normalize(rows: unknown, currentId: string): SessionRow[] {
  if (!Array.isArray(rows)) return []
  const out: SessionRow[] = []
  for (const raw of rows) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const row = raw as SavedWire
    const mapped = wireRow(row as Record<string, unknown>, currentId)
    if (mapped) out.push(mapped)
  }
  return out
}

/** Parse `- [ ]` / `- [x]` checklist items out of plan markdown. */
function planItemsOf(markdown: string): PlanItem[] {
  const items: PlanItem[] = []
  for (const line of markdown.split('\n')) {
    const match = /^\s*[-*]\s+\[([ xX])]\s*(.+)$/.exec(line)
    if (match) items.push({ done: match[1]!.toLowerCase() === 'x', text: match[2]!.trim() })
  }
  return items
}

/** Does this batched question look like a plan review (approve + revise)? */
export function isPlanReview(question: import('./types.js').TaskQuestion): boolean {
  const haystack = [question.items.map(item => `${item.question} ${item.options.join(' ')}`).join(' ')].join(' ').toLowerCase()
  return haystack.includes('plan') && (haystack.includes('approve') || haystack.includes('accept'))
}

/** Epoch ms when an in-flight daemon turn began, or null when unreported. */
export function inflightStartedAtMs(inflight: unknown): number | null {
  if (!inflight || typeof inflight !== 'object' || Array.isArray(inflight)) return null
  const startedAtSec = num((inflight as Record<string, unknown>).started_at)
  if (startedAtSec === null || startedAtSec <= 0) return null
  return Math.round(startedAtSec * 1000)
}

/**
 * Rehydrate an in-flight turn from `session.inflight` onto `builder` when a
 * client attaches to or reconnects mid-turn. Both runner-managed native turns
 * and external ACP turns only commit `session.messages` at `turn_end`, so
 * `session.inflight` is the sole carrier of the prompt and work done so far.
 */
export function applyInflightTurn(
  builder: BlockBuilder,
  inflight: unknown,
): { readonly user?: string; readonly startedAtMs?: number } | null {
  if (!inflight || typeof inflight !== 'object' || Array.isArray(inflight)) return null
  const record = inflight as Record<string, unknown>
  const rawUser = str(record.user).trim()
  const thinking = str(record.thinking)
  const assistant = str(record.assistant)
  const tools = Array.isArray(record.tools) ? record.tools : []
  const startedAtMs = inflightStartedAtMs(record)

  let hydratedUser: string | undefined
  if (rawUser && !isLegacyHarnessPrompt(rawUser)) {
    const cleanUser = withoutTurnContext(rawUser)
    if (cleanUser) {
      hydratedUser = cleanUser
      const last = builder.snapshot(true).at(-1)
      if (!(last && last.kind === 'user' && last.text === cleanUser)) {
        builder.pushUser(cleanUser)
      }
    }
  }
  if (thinking) {
    builder.push('think_part', { think: thinking })
  }
  for (const rawTool of tools) {
    if (!rawTool || typeof rawTool !== 'object' || Array.isArray(rawTool)) continue
    const tool = rawTool as Record<string, unknown>
    const name = str(tool.name).trim()
    if (!name) continue
    const id = str(tool.id).trim() || undefined
    const rawArgs = typeof tool.arguments === 'string' ? tool.arguments.trim() : ''
    const context = str(tool.context).trim()
    let toolArgs: unknown = tool.arguments
    if (rawArgs && context) {
      try {
        const parsed = JSON.parse(rawArgs) as unknown
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          toolArgs = context
        }
      } catch {
        // Truncated JSON previews end with `…`; fall back to the concise human context.
        toolArgs = context
      }
    } else if (!rawArgs && context) {
      toolArgs = context
    }
    builder.push('tool_call', {
      ...(id ? { id, tool_call_id: id } : {}),
      name,
      ...(toolArgs !== undefined ? { arguments: toolArgs } : {}),
    })
    const settled =
      typeof tool.ok === 'boolean' ||
      typeof tool.duration_ms === 'number' ||
      typeof tool.error === 'string'
    if (settled) {
      const errorText = str(tool.error) || (tool.ok === false ? 'Tool execution failed.' : '')
      builder.push('tool_result', {
        ...(id ? { id, tool_call_id: id } : {}),
        name,
        ...(typeof tool.duration_ms === 'number' ? { duration_ms: tool.duration_ms } : {}),
        ...(typeof tool.output_preview === 'string' ? { return_value: tool.output_preview } : {}),
        ...(errorText ? { error: errorText } : {}),
      })
    }
  }
  if (assistant) {
    builder.push('text_part', { text: assistant })
  }
  const steers = Array.isArray(record.steers)
    ? record.steers.filter((value): value is string => typeof value === 'string')
    : []
  for (const rawSteer of steers) {
    const cleanSteer = withoutTurnContext(rawSteer).trim()
    if (!cleanSteer) continue
    const last = builder.snapshot(true).at(-1)
    if (!(last && last.kind === 'user' && last.text === cleanSteer)) {
      builder.pushUser(cleanSteer)
    }
  }
  return {
    ...(hydratedUser ? { user: hydratedUser } : {}),
    ...(startedAtMs !== null ? { startedAtMs } : {}),
  }
}

export const LOG_CAP = 300
const SNIPPET_CAP = 64
const SNIPPET_MAP_CAP = 128
const MAX_DIFF_LINES_PER_FILE = 1_200
const MAX_AGENT_TEXT_CHARS = 256_000
const BACKGROUND_STREAM_FRAME_MS = 250
const SWARM_INSPECT_THROTTLE_MS = 750
const HEARTBEAT_MS = 5_000
/** A visible but unfocused window refreshes its session lists this often. */
const UNFOCUSED_LIST_REFRESH_MS = 60_000
/** How long a first Escape over a running turn waits for the second that stops it. */
export const STOP_CONFIRM_MS = 2_000


/** Compact one-line summary of an event for the Log tab. */
function summarize(type: string, payload: Readonly<Record<string, unknown>>): string {
  const pieces: string[] = []
  for (const key of ['model', 'text', 'think', 'title', 'body', 'message', 'error', 'name', 'status', 'plan_mode', 'context_tokens', 'duration_ms', 'tool_name', 'action']) {
    const value = payload[key]
    if (typeof value === 'string' && value) pieces.push(`${key}=${value.slice(0, 120)}`)
    else if (typeof value === 'number' || typeof value === 'boolean') pieces.push(`${key}=${String(value)}`)
  }
  return pieces.length ? pieces.join(' ') : '{}'
}

export class Store {
  private listeners = new Set<() => void>()
  private frame: Snapshot
  private builder = new BlockBuilder()
  private tick: NodeJS.Timeout | null = null
  private heartbeat: NodeJS.Timeout | null = null
  private updatingRuntime = false
  private terminalsLoadVersion = 0
  private channelsLoadVersion = 0
  private learningLoadVersion = 0
  private learningReviewToken: object | null = null
  /** In-flight agent-family tool calls; the fleet poll lives while any do. */
  private readonly pendingAgentCalls = new Set<string>()
  private fleetPoll: NodeJS.Timeout | null = null
  /** The turn's agents-card members by local key — merged into the builder. */
  private readonly agentMembers = new Map<string, AgentMember>()
  /** Titles the current card shows, for fleet-snapshot status matching. */
  private started = false
  private sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
  private historyBefore: string | null = null
  private historyGeneration = 0
  private historySeen = new Set<string>()
  private legacyHistory: Block[] = []
  private historyActions: HistoryPage['actions'] = []
  private historyBlockIds = new Map<string, number>()
  private unsubEvents: (() => void) | null = null

  // ── workspace folds (not part of the transcript) ──
  private queue: QueueItem[] = []
  private changes = new Map<string, DiffFile>()
  /** Edit totals when the current turn began — a checkpoint marks only this turn's change. */
  private turnStartTotals = { adds: 0, dels: 0 }
  private logRing: LogEntry[] = []
  private logView: readonly LogEntry[] = []
  private logDirty = false
  private pendingPatch: Record<string, unknown> | null = null
  private frameDirty = false
  private lastSwarmInspectAt = 0
  private swarmInspectTimer: ReturnType<typeof setTimeout> | null = null
  private planState: PlanState | null = null
  private failure: FailedTurn | null = null
  private turnCount = 0
  private seq = 1
  private turnError: string | null = null
  private agentText = ''
  private lastUser = ''
  private preparingSubmissions = new Map<string, object>()
  private optimisticSubmissions = new Map<string, { text: string; id: number; acknowledged: boolean }>()
  /** `images` are the data URLs shown with the message, whichever path shows it first. */
  private pendingSteers = new Map<string, { session: string; text: string; images: readonly string[]; shown: boolean; echoed: boolean; blockId?: number }>()
  private slashResult: { sessionKey: string; text: string } | null = null
  private snippets: Record<string, string> = {}
  private enriching = new Set<string>()
  private snippetRetryAt = new Map<string, number>()
  /** Latest session-search request; stale responses must not win. */
  private searchSeq = 0
  private swarmLoadVersion = 0
  private acpLoadVersion = 0
  private lastAcpConnect: {
    readonly additionalDirectories?: readonly string[]
    readonly agent?: string
    readonly agentId: string
    readonly custom?: { readonly command: string; readonly args?: readonly string[] }
    readonly modeId?: string
    readonly resumeSessionId?: string
    readonly sandbox?: boolean
  } | null = null
  private foldedToolIds = new Set<string>()
  private ttftTotalMs = 0
  private ttftSamples = 0
  private activeMetricTools = new Set<string>()

  constructor() {
    this.frame = this.frozen({
      swarm: null,
      swarmMessages: [],
      swarmInspect: null,
      swarmInspectTarget: '',
      swarmLoading: false,
      swarmError: null,
      swarmOpen: false,
      swarmOpenRevision: 0,
      sideQuestionModel: readSideQuestionModel(),
      acpAgents: [],
      acpBinding: null,
      savedAcpBinding: null,
      acpConnecting: false,
      acpError: null,
      acpAuthRequired: false,
      acpAuthMethods: [],
      connection: 'connecting',
      noWorkspace: false,
      cwd: '',
      model: '',
      currentAgentPreset: 'default',
      agentPresets: [],
      agentPresetsError: null,
      channels: [],
      channelsAvailable: false,
      channelsConfigured: false,
      terminals: [],
      terminalsLoading: false,
      terminalsError: null,
      searchOpen: false,
      searchResults: [],
      searchStats: null,
      searchSearching: false,
      searchError: null,
      branch: '',
      daemonWarning: null,
      costUsd: null,
      costBasis: null,
      mcpStatus: {},
      models: [],
      contextTokens: null,
      computerUseWanted: false,
      contextMax: null,
      ttftMs: null,
      tokensPerSecond: null,
      llmDurationMs: 0,
      llmSteps: 0,
      toolDurationMs: 0,
      toolSteps: 0,
      inputTokens: 0,
      outputTokens: 0,
      metricPhase: null,
      metricPhaseStartedAt: null,
      cacheHitRate: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      sessions: [],
      live: [],
      fleet: [],
      skillSuggestions: [],
      creatorTrace: [],
      backgroundJobs: [],
      currentId: '',
      sessionOpenRevision: 0,
      currentTitle: '',
      sessionKey: '',
      goal: '',
      approval: null,
      question: null,
      interactionHidden: false,
      planMode: false,
      turnActive: false, networkRetrying: false,
      liveFocus: null,
      turnFailed: false,
      turnSeconds: 0,
      turnStartedAt: null,
      blocks: [],
      error: null,
      tab: 'activity',
      turnCount: 0,
      queue: [],
      changes: [],
      changesKept: false,
      plan: null,
      todos: null,
      log: [],
      failed: null,
      stopArmed: false,
      settingsOpen: false,
      settingsTab: 'general',
      paletteOpen: false,
      pickerOpen: false,
      modelMenuOpen: false,
      delegationMode: 'auto',
      delegationMenuOpen: false,
      appUpdate: null,
      providerRelay: null,
      contextMenuOpen: false,
      contextBreakdown: null,
      contextBreakdownLoading: false,
      reasoningPickerOpen: false,
      reasoningEffort: '',
      reasoningLevels: [],
      reasoningDefault: '',
      reasoningNote: '',
      reasoningLoading: false,
      wsMenuOpen: false,
      workspaceBusy: false,
      workspaceError: null,
      sessionMenu: null,
      streamThinking: readStreamThinking(),
      taskModalOpen: false,
      providers: [],
      providerError: '',
      navigating: null,
      providersLoading: false,
      modelsLoading: false,
      modelsError: null,
      providerSwitching: null,
      providerSwitchError: null,
      runMode: null,
      runModeError: '',
      runModeSaving: false,
      localSetup: { status: null, job: null, error: '', loading: false },
      providerModels: {},
      providerModelLoading: [],
      providerModelWarnings: {},
      providerTypes: [],
      commands: [],
      permissionMode: '',
      permissionUpdating: false,
      permissionError: null,
      snippets: {},
    })
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): Snapshot => (this.frameDirty ? this.materializeFrame() : this.frame)

  /** Synchronously flush any coalesced streaming update and notify subscribers. */
  flush = (): void => {
    if (this.frameDirty) this.materializeFrame()
    this.emit(true)
  }

  // ── Actions ──────────────────────────────────────────────────────────

  openLearning(proposalId = ''): void {
    this.patch({ learningOpen: true, learningProposalId: proposalId, learningOpenRevision: (this.frame.learningOpenRevision ?? 0) + 1 })
  }

  closeLearning(): void { this.patch({ learningOpen: false }) }

  openSwarmBoard(inspectMember?: string): void {
    const cleanMember = inspectMember?.trim() ?? ''
    this.patch({
      swarmOpen: true,
      swarmOpenRevision: (this.frame.swarmOpenRevision ?? 0) + 1,
      ...(cleanMember ? { swarmInspectTarget: cleanMember } : {}),
    })
    if (this.frame.connection === 'online') {
      void this.refreshSwarm()
      if (cleanMember) void this.inspectSwarmMember(cleanMember)
    }
  }

  closeSwarmBoard(): void {
    if (this.frame.swarmOpen || this.frame.swarmInspectTarget) {
      this.patch({ swarmOpen: false, swarmInspectTarget: '' })
    }
  }

  async openSwarmConversation(conversationId: string): Promise<boolean> {
    const needle = conversationId.trim().toLowerCase()
    const matchMember = () => this.frame.swarm?.members.find(
      member =>
        member.conversation_id?.trim().toLowerCase() === needle
        || member.name.trim().toLowerCase() === needle,
    )
    let matched = needle ? matchMember() : undefined
    if (!matched && needle && this.frame.connection === 'online') {
      await this.refreshSwarm({ silent: true })
      matched = matchMember()
    }
    if (matched) {
      this.openSwarmBoard(matched.name)
      return true
    }
    this.openSwarmBoard()
    return false
  }

  openSideQuestion(): void {
    this.patch({
      sideQuestionOpen: true,
      sideQuestionOpenRevision: (this.frame.sideQuestionOpenRevision ?? 0) + 1,
    })
  }

  closeSideQuestion(): void {
    if (this.frame.sideQuestionOpen) this.patch({ sideQuestionOpen: false })
  }

  toggleSideQuestion(): void {
    if (this.frame.sideQuestionOpen) this.closeSideQuestion()
    else this.openSideQuestion()
  }

  clearSideQuestion(): void {
    this.patch({
      sideQuestionTurns: [],
      sideQuestionStreaming: false,
      sideQuestionError: null,
    })
  }

  setSideQuestionModel(model: string): void {
    const clean = model.trim()
    writeSideQuestionModel(clean)
    this.patch({ sideQuestionModel: clean })
  }

  async askSideQuestion(question: string, model?: string): Promise<boolean> {
    const clean = question.trim()
    this.openSideQuestion()
    if (!clean) return true
    const { isCurrent } = this.captureSessionRequest()
    const priorTurns = this.frame.sideQuestionTurns ?? []
    const history = priorTurns
      .filter(turn => turn.status === 'done' && turn.answer.trim().length > 0)
      .flatMap(turn => [
        { role: 'user' as const, content: turn.question },
        { role: 'assistant' as const, content: turn.answer },
      ])
    const activeAcp =
      this.frame.acpBinding?.connected && this.frame.acpBinding.transport !== 'native'
        ? this.frame.acpBinding
        : null
    const savedSideModel = (this.frame.sideQuestionModel ?? '').trim()
    const savedMatchesAcp =
      !activeAcp ||
      !savedSideModel ||
      savedSideModel === activeAcp.current_model ||
      Boolean(activeAcp.models?.some(entry => entry.id === savedSideModel))
    const requestedModel = (
      model !== undefined ? model : savedMatchesAcp ? savedSideModel : ''
    ).trim()
    const requestId = crypto.randomUUID()
    const optimistic: SideQuestionTurn = {
      requestId,
      question: clean,
      answer: '',
      model: requestedModel || activeAcp?.current_model || this.frame.model,
      status: 'streaming',
    }
    this.patch({
      sideQuestionTurns: [...priorTurns, optimistic],
      sideQuestionStreaming: true,
      sideQuestionError: null,
    })
    try {
      const result = await this.bridge.call(
        'session.side_question',
        this.swarmParams({
          request_id: requestId,
          question: clean,
          ...(requestedModel ? { model: requestedModel } : {}),
          ...(history.length > 0 ? { history } : {}),
        }),
      )
      if (!isCurrent()) return result.ok !== false
      const currentTurns = this.frame.sideQuestionTurns ?? []
      const answer = str(result.answer)
      const resolvedModel = str(result.model)
      if (result.ok === false) {
        const err = str(result.error) || 'Side question failed'
        const updated = currentTurns.map(turn => {
          if (turn.requestId !== requestId) return turn
          const settledTools = settleSideQuestionTools(turn.tools, 'error')
          return {
            ...turn,
            ...(answer ? { answer } : {}),
            ...(resolvedModel ? { model: resolvedModel } : {}),
            ...(settledTools ? { tools: settledTools } : {}),
            status: 'error' as const,
            error: err,
          }
        })
        this.patch({
          sideQuestionTurns: updated,
          sideQuestionStreaming: updated.some(turn => turn.status === 'streaming'),
          sideQuestionError: err,
        })
        return false
      }
      const updated = currentTurns.map(turn => {
        if (turn.requestId !== requestId) return turn
        const settledTools = settleSideQuestionTools(turn.tools, 'done')
        return {
          ...turn,
          answer: answer || turn.answer,
          ...(resolvedModel ? { model: resolvedModel } : {}),
          ...(settledTools ? { tools: settledTools } : {}),
          status: 'done' as const,
        }
      })
      this.patch({
        sideQuestionTurns: updated,
        sideQuestionStreaming: updated.some(turn => turn.status === 'streaming'),
        sideQuestionError: null,
      })
      return true
    } catch (error) {
      if (isCurrent()) {
        const err = desktopError(error)
        const currentTurns = this.frame.sideQuestionTurns ?? []
        const updated = currentTurns.map(turn => {
          if (turn.requestId !== requestId) return turn
          const settledTools = settleSideQuestionTools(turn.tools, 'error')
          return {
            ...turn,
            ...(settledTools ? { tools: settledTools } : {}),
            status: 'error' as const,
            error: err,
          }
        })
        this.patch({
          sideQuestionTurns: updated,
          sideQuestionStreaming: updated.some(turn => turn.status === 'streaming'),
          sideQuestionError: err,
        })
      }
      return false
    }
  }

  clearAcpError(): void {
    if (this.frame.acpError || this.frame.acpAuthRequired || (this.frame.acpAuthMethods?.length ?? 0) > 0) {
      this.patch({ acpError: null, acpAuthRequired: false, acpAuthMethods: [] })
    }
  }

  dismissSavedAcpBinding(): void {
    if (this.frame.savedAcpBinding) this.patch({ savedAcpBinding: null })
  }

  hasRetryableAcpTarget(): boolean {
    return Boolean(this.lastAcpConnect || this.frame.savedAcpBinding || (this.frame.acpBinding && this.frame.acpBinding.transport !== 'native'))
  }

  async refreshAcpAgents(options: { readonly silent?: boolean; readonly preserveBinding?: boolean } = {}): Promise<void> {
    const { isCurrent } = this.captureSessionRequest()
    const version = ++this.acpLoadVersion
    const current = (): boolean => isCurrent() && version === this.acpLoadVersion
    try {
      const raw = await this.bridge.call('acp.agents', this.swarmParams())
      if (!current()) return
      if (raw.ok === false || raw.kind === 'error') {
        const message = str(raw.error) || str(raw.message) || 'acp.agents failed'
        if (options.silent && /unknown method/i.test(message)) return
        if (!options.silent) this.patch({ acpError: message })
        return
      }
      const agents = parseAcpExternalAgents(raw.agents)
      if (options.preserveBinding) {
        this.patch({ acpAgents: agents })
        return
      }
      const active = parseAcpSessionBinding(raw.active)
      this.patch({
        acpAgents: agents,
        acpBinding: active,
        ...(active?.connected ? { savedAcpBinding: null } : {}),
        acpError: null,
      })
    } catch (error) {
      if (!current()) return
      const message = desktopError(error)
      if (options.silent && /unknown method|not found/i.test(message)) return
      if (!options.silent) this.patch({ acpError: message })
    }
  }

  async connectAcpAgent(
    agentId: string,
    custom?: { readonly command: string; readonly args?: readonly string[] },
    options?: {
      readonly additionalDirectories?: readonly string[]
      readonly agent?: string
      readonly modeId?: string
      readonly resumeSessionId?: string
      readonly sandbox?: boolean
    },
  ): Promise<boolean> {
    const cleanId = agentId.trim()
    if (!cleanId) return false
    if (cleanId === 'xerxes') return this.disconnectAcpAgent()
    const resumeSessionId = options?.resumeSessionId?.trim() || undefined
    const additionalDirectories = options?.additionalDirectories !== undefined
      ? options.additionalDirectories.map(dir => dir.trim()).filter(Boolean)
      : undefined
    const sandbox = options?.sandbox
    const agentOverride = options?.agent !== undefined ? options.agent.trim() : undefined
    const savedMode =
      !options?.modeId && this.frame.savedAcpBinding?.agent_id === cleanId
        ? this.frame.savedAcpBinding.current_mode?.trim() || undefined
        : undefined
    const modeId =
      options?.modeId?.trim() ||
      savedMode ||
      (cleanId === 'antigravity' ? 'turbo' : undefined)
    this.lastAcpConnect = {
      agentId: cleanId,
      ...(custom ? { custom } : {}),
      ...(modeId ? { modeId } : {}),
      ...(resumeSessionId ? { resumeSessionId } : {}),
      ...(additionalDirectories !== undefined ? { additionalDirectories } : {}),
      ...(sandbox !== undefined ? { sandbox } : {}),
      ...(agentOverride !== undefined ? { agent: agentOverride } : {}),
    }
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpConnecting: true, acpError: null, acpAuthRequired: false, acpAuthMethods: [] })
    const isAutoInstallableBuiltIn = !custom && (cleanId === 'antigravity' || cleanId === 'claude-code')
    const cachedAgent = this.frame.acpAgents?.find(item => item.id === cleanId)
    let ensuredProviderClis = false
    const tryEnsureHostProviderClis = async (): Promise<void> => {
      if (ensuredProviderClis || !isAutoInstallableBuiltIn || !this.bridge.remote) return
      ensuredProviderClis = true
      await this.bridge.remote('provider-setup', {}).catch(() => undefined)
      await this.refreshAcpAgents({ silent: true, preserveBinding: true })
    }
    const callConnect = () => this.bridge.call(
      'acp.connect',
      this.swarmParams({
        agent_id: cleanId,
        ...(this.frame.cwd ? { cwd: this.frame.cwd } : {}),
        ...(custom?.command?.trim() ? { command: custom.command.trim() } : {}),
        ...(custom?.args && custom.args.length > 0 ? { args: [...custom.args] } : {}),
        ...(modeId ? { mode_id: modeId } : {}),
        ...(resumeSessionId ? { resume_session_id: resumeSessionId } : {}),
        ...(additionalDirectories !== undefined ? { additional_directories: [...additionalDirectories] } : {}),
        ...(sandbox !== undefined ? { sandbox } : {}),
        ...(agentOverride !== undefined ? { agent: agentOverride } : {}),
      }),
    )
    try {
      if (isAutoInstallableBuiltIn && cachedAgent && !cachedAgent.available) {
        await tryEnsureHostProviderClis()
        if (!isCurrent()) return false
      }
      let result = await callConnect()
      if (!isCurrent()) return result.ok === true
      if (
        (result.ok === false || result.kind === 'error')
        && isAutoInstallableBuiltIn
        && !ensuredProviderClis
        && /not installed|unavailable on path|not found/i.test(str(result.error) || str(result.message))
      ) {
        await tryEnsureHostProviderClis()
        if (!isCurrent()) return false
        result = await callConnect()
        if (!isCurrent()) return result.ok === true
      }
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || `Could not connect to ${cleanId}`
        const hint = str(result.install_hint) || str(result.installHint)
        const redundantHint =
          Boolean(hint)
          && (err.includes(hint) || (/agy acp --login/i.test(err) && /agy acp --login/i.test(hint)))
        const authMethods = parseAcpAuthMethods(result.auth_methods ?? result.authMethods)
        const authRequired = result.auth_required === true || result.authRequired === true || authMethods.length > 0
        this.patch({
          acpConnecting: false,
          acpError: hint && !redundantHint ? `${err} — ${hint}` : err,
          acpAuthRequired: authRequired,
          acpAuthMethods: authMethods,
        })
        return false
      }
      const binding = parseAcpSessionBinding(result.binding)
      const updatedAgents =
        binding?.connected && cachedAgent && !cachedAgent.available && this.frame.acpAgents
          ? this.frame.acpAgents.map(item => (item.id === cleanId ? { ...item, available: true } : item))
          : undefined
      this.patch({
        acpBinding: binding,
        ...(updatedAgents ? { acpAgents: updatedAgents } : {}),
        ...(binding?.connected ? { savedAcpBinding: null } : {}),
        acpConnecting: false,
        acpError: null,
        acpAuthRequired: false,
        acpAuthMethods: [],
        ...(binding?.connected && binding.transport !== 'native'
          ? {
              modelMenuOpen: false,
              pickerOpen: false,
              reasoningPickerOpen: false,
              delegationMenuOpen: false,
              skillSuggestions: [],
              swarm: null,
              swarmMessages: [],
              swarmInspect: null,
              swarmInspectTarget: null,
            }
          : {}),
      })
      if (binding?.connected && binding.transport !== 'native') {
        void this.refreshSwarm({ silent: true })
      }
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpConnecting: false, acpError: desktopError(error) })
      }
      return false
    }
  }

  async reconnectSavedAcp(): Promise<boolean> {
    const saved = this.frame.savedAcpBinding
    if (!saved || !saved.agent_id) return false
    const isBuiltIn = saved.agent_id === 'antigravity' || saved.agent_id === 'claude-code' || saved.agent_id === 'codex'
    const custom = !isBuiltIn && saved.command ? { command: saved.command } : undefined
    const connectOpts = {
      ...(saved.external_session_id ? { resumeSessionId: saved.external_session_id } : {}),
      ...(saved.current_mode ? { modeId: saved.current_mode } : {}),
      ...(saved.additional_directories?.length ? { additionalDirectories: saved.additional_directories } : {}),
      ...(saved.sandbox ? { sandbox: true } : {}),
      ...(saved.agent_override ? { agent: saved.agent_override } : {}),
    }
    return this.connectAcpAgent(
      saved.agent_id,
      custom,
      Object.keys(connectOpts).length > 0 ? connectOpts : undefined,
    )
  }

  async retryAcpConnect(): Promise<boolean> {
    if (this.lastAcpConnect) {
      const { agentId, custom, modeId, resumeSessionId, additionalDirectories, sandbox, agent } = this.lastAcpConnect
      const savedForAgent = this.frame.savedAcpBinding?.agent_id === agentId ? this.frame.savedAcpBinding : undefined
      const effectiveModeId = savedForAgent?.current_mode || modeId
      const effectiveResumeId = savedForAgent?.external_session_id || resumeSessionId
      const connectOpts = {
        ...(effectiveResumeId ? { resumeSessionId: effectiveResumeId } : {}),
        ...(effectiveModeId ? { modeId: effectiveModeId } : {}),
        ...(additionalDirectories !== undefined ? { additionalDirectories } : {}),
        ...(sandbox !== undefined ? { sandbox } : {}),
        ...(agent !== undefined ? { agent } : {}),
      }
      return this.connectAcpAgent(
        agentId,
        custom,
        Object.keys(connectOpts).length > 0 ? connectOpts : undefined,
      )
    }
    if (this.frame.savedAcpBinding) {
      return this.reconnectSavedAcp()
    }
    const active = this.frame.acpBinding
    if (active && active.transport !== 'native') {
      const connectOpts = {
        ...(active.external_session_id ? { resumeSessionId: active.external_session_id } : {}),
        ...(active.current_mode ? { modeId: active.current_mode } : {}),
      }
      return this.connectAcpAgent(
        active.agent_id,
        undefined,
        Object.keys(connectOpts).length > 0 ? connectOpts : undefined,
      )
    }
    const err = this.frame.acpError ?? ''
    if (/\b(antigravity|agy)\b/i.test(err)) return this.connectAcpAgent('antigravity')
    if (/\bclaude\b/i.test(err)) return this.connectAcpAgent('claude-code')
    return false
  }

  async setAcpConfigOption(configId: string, value: string | boolean): Promise<boolean> {
    const cleanConfigId = configId.trim()
    if (!cleanConfigId) return false
    const isBoolean = typeof value === 'boolean'
    const cleanValue: string | boolean = isBoolean ? value : value.trim()
    if (!isBoolean && !cleanValue) return false
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpError: null })
    try {
      const result = await this.bridge.call(
        'acp.set_config_option',
        this.swarmParams({
          config_id: cleanConfigId,
          value: cleanValue,
          ...(isBoolean ? { type: 'boolean' } : {}),
        }),
      )
      if (!isCurrent()) return result.ok === true
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || `Could not update ${cleanConfigId} to ${String(cleanValue)}`
        this.patch({ acpError: err })
        return false
      }
      const parsed = parseAcpSessionBinding(result.binding)
      if (parsed) {
        this.patch({ acpBinding: parsed, acpError: null })
      } else if (this.frame.acpBinding) {
        const prev = this.frame.acpBinding
        const updatedConfigs = prev.config_options?.map(opt =>
          opt.id === cleanConfigId ? { ...opt, current_value: cleanValue } : opt,
        )
        const strValue = typeof cleanValue === 'string' ? cleanValue : undefined
        const isThought =
          strValue !== undefined
          && (cleanConfigId === (prev.thought_level_config_id ?? 'thought_level')
            || prev.config_options?.some(opt => opt.id === cleanConfigId && opt.category === 'thought_level'))
        const isModel =
          strValue !== undefined
          && (cleanConfigId === (prev.model_config_id ?? 'model')
            || prev.config_options?.some(opt => opt.id === cleanConfigId && opt.category === 'model'))
        const isMode =
          strValue !== undefined
          && (cleanConfigId === (prev.mode_config_id ?? 'mode')
            || prev.config_options?.some(opt => opt.id === cleanConfigId && opt.category === 'mode'))
        const matchedModel = isModel ? prev.models?.find(m => m.id === strValue) : undefined
        const matchedMode = isMode ? prev.modes?.find(m => m.id === strValue) : undefined
        this.patch({
          acpBinding: {
            ...prev,
            ...(updatedConfigs ? { config_options: updatedConfigs } : {}),
            ...(isThought && strValue !== undefined ? { thought_level: strValue } : {}),
            ...(isModel && strValue !== undefined ? { current_model: strValue, current_model_name: matchedModel?.name ?? strValue } : {}),
            ...(isMode && strValue !== undefined ? { current_mode: strValue, current_mode_name: matchedMode?.name ?? strValue } : {}),
          },
          acpError: null,
        })
      }
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpError: desktopError(error) })
      }
      return false
    }
  }

  async setAcpThoughtLevel(level: string): Promise<boolean> {
    const configId = this.frame.acpBinding?.thought_level_config_id || 'thought_level'
    return this.setAcpConfigOption(configId, level)
  }

  async setAcpModel(modelId: string): Promise<boolean> {
    const cleanId = modelId.trim()
    if (!cleanId) return false
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpError: null })
    try {
      const result = await this.bridge.call(
        'acp.set_model',
        this.swarmParams({ model_id: cleanId }),
      )
      if (!isCurrent()) return result.ok === true
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || `Could not switch external model to ${cleanId}`
        this.patch({ acpError: err })
        return false
      }
      const parsed = parseAcpSessionBinding(result.binding)
      if (parsed) {
        this.patch({ acpBinding: parsed, acpError: null })
      } else if (this.frame.acpBinding) {
        const matched = this.frame.acpBinding.models?.find(m => m.id === cleanId)
        this.patch({
          acpBinding: {
            ...this.frame.acpBinding,
            current_model: cleanId,
            current_model_name: matched?.name ?? cleanId,
          },
          acpError: null,
        })
      }
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpError: desktopError(error) })
      }
      return false
    }
  }

  async setAcpMode(modeId: string): Promise<boolean> {
    const cleanId = modeId.trim()
    if (!cleanId) return false
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpError: null })
    try {
      const result = await this.bridge.call(
        'acp.set_mode',
        this.swarmParams({ mode_id: cleanId }),
      )
      if (!isCurrent()) return result.ok === true
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || `Could not switch external mode to ${cleanId}`
        this.patch({ acpError: err })
        return false
      }
      const parsed = parseAcpSessionBinding(result.binding)
      if (parsed) {
        this.patch({ acpBinding: parsed, acpError: null })
      } else if (this.frame.acpBinding) {
        const matched = this.frame.acpBinding.modes?.find(m => m.id === cleanId)
        this.patch({
          acpBinding: {
            ...this.frame.acpBinding,
            current_mode: cleanId,
            current_mode_name: matched?.name ?? cleanId,
          },
          acpError: null,
        })
      }
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpError: desktopError(error) })
      }
      return false
    }
  }

  async disconnectAcpAgent(): Promise<boolean> {
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpConnecting: true, acpError: null, acpAuthRequired: false, acpAuthMethods: [] })
    try {
      const result = await this.bridge.call('acp.disconnect', this.swarmParams())
      if (!isCurrent()) return result.ok !== false
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || 'Could not disconnect external agent'
        this.patch({ acpConnecting: false, acpError: err })
        return false
      }
      this.lastAcpConnect = null
      this.patch({
        acpBinding: null,
        savedAcpBinding: null,
        acpConnecting: false,
        acpError: null,
        acpAuthRequired: false,
        acpAuthMethods: [],
        swarm: null,
        swarmMessages: [],
        swarmInspect: null,
        swarmInspectTarget: null,
      })
      void this.refreshSwarm({ silent: true })
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpConnecting: false, acpError: desktopError(error) })
      }
      return false
    }
  }

  async acpSessions(
    options: { readonly cwd?: string; readonly cursor?: string; readonly agentId?: string } = {},
  ): Promise<AcpSessionsResult> {
    try {
      const result = await this.bridge.call(
        'acp.sessions',
        this.swarmParams({
          ...(options.cwd?.trim() ? { cwd: options.cwd.trim() } : {}),
          ...(options.cursor?.trim() ? { cursor: options.cursor.trim() } : {}),
          ...(options.agentId?.trim() ? { agent_id: options.agentId.trim() } : {}),
        }),
      )
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || 'Could not list external ACP sessions'
        return { ok: false, sessions: [], error: err }
      }
      const rawList = Array.isArray(result.sessions) ? result.sessions : []
      const sessions: AcpExternalSessionInfo[] = []
      for (const entry of rawList) {
        if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue
        const row = entry as Readonly<Record<string, unknown>>
        const sessionId = str(row.session_id) || str(row.sessionId) || str(row.id)
        if (!sessionId) continue
        const cwd = str(row.cwd) || undefined
        const title = str(row.title) || undefined
        const updatedAt = str(row.updated_at) || str(row.updatedAt) || undefined
        sessions.push({
          session_id: sessionId,
          ...(cwd ? { cwd } : {}),
          ...(title ? { title } : {}),
          ...(updatedAt ? { updated_at: updatedAt } : {}),
        })
      }
      const nextCursor = str(result.next_cursor) || str(result.nextCursor) || undefined
      return {
        ok: true,
        sessions,
        ...(nextCursor ? { next_cursor: nextCursor } : {}),
      }
    } catch (error) {
      return { ok: false, sessions: [], error: desktopError(error) }
    }
  }

  async forkAcpSession(
    options: { readonly cwd?: string; readonly additionalDirectories?: readonly string[] } = {},
  ): Promise<boolean> {
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ acpError: null })
    try {
      const result = await this.bridge.call(
        'acp.fork',
        this.swarmParams({
          ...(options.cwd?.trim() ? { cwd: options.cwd.trim() } : {}),
          ...(options.additionalDirectories && options.additionalDirectories.length > 0
            ? { additional_directories: [...options.additionalDirectories] }
            : {}),
        }),
      )
      if (!isCurrent()) return result.ok === true
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || 'Could not fork ACP session'
        this.patch({ acpError: err })
        return false
      }
      const binding = parseAcpSessionBinding(result.binding)
      if (binding) {
        this.patch({ acpBinding: binding, acpError: null })
      }
      return true
    } catch (error) {
      if (isCurrent()) {
        this.patch({ acpError: desktopError(error) })
      }
      return false
    }
  }

  async authenticateAcp(methodId: string, agentId?: string): Promise<AcpAuthenticateResult> {
    const cleanMethodId = methodId.trim()
    if (!cleanMethodId) return { ok: false, error: 'Authentication method ID is required' }
    const targetAgentId =
      agentId?.trim()
      || this.lastAcpConnect?.agentId
      || this.frame.acpBinding?.agent_id
      || this.frame.savedAcpBinding?.agent_id
      || undefined
    const { isCurrent } = this.captureSessionRequest()
    try {
      const result = await this.bridge.call(
        'acp.authenticate',
        this.swarmParams({
          method_id: cleanMethodId,
          ...(targetAgentId ? { agent_id: targetAgentId } : {}),
        }),
      )
      if (result.ok === false || result.kind === 'error') {
        const err = str(result.error) || str(result.message) || `Authentication failed for ${cleanMethodId}`
        if (isCurrent()) this.patch({ acpError: err })
        return { ok: false, error: err }
      }
      const mode = result.mode === 'terminal' ? 'terminal' : 'agent'
      if (mode === 'agent') {
        if (isCurrent()) {
          this.patch({ acpError: null, acpAuthRequired: false, acpAuthMethods: [] })
        }
        return { ok: true, mode: 'agent' }
      }
      const command = str(result.command) || undefined
      const args = Array.isArray(result.args)
        ? result.args.filter((entry): entry is string => typeof entry === 'string')
        : undefined
      const rawEnv = result.env && typeof result.env === 'object' && !Array.isArray(result.env)
        ? (result.env as Readonly<Record<string, unknown>>)
        : undefined
      const envEntries = rawEnv
        ? Object.entries(rawEnv).filter((pair): pair is [string, string] => typeof pair[1] === 'string')
        : []
      const label = str(result.label) || undefined
      return {
        ok: true,
        mode: 'terminal',
        ...(command ? { command } : {}),
        ...(args && args.length > 0 ? { args } : {}),
        ...(envEntries.length > 0 ? { env: Object.fromEntries(envEntries) } : {}),
        ...(label ? { label } : {}),
      }
    } catch (error) {
      const err = desktopError(error)
      if (isCurrent()) this.patch({ acpError: err })
      return { ok: false, error: err }
    }
  }

  async startAcpLogin(agentId = 'antigravity'): Promise<AcpLoginStatus> {
    try {
      const result = await this.bridge.call(
        'acp.login',
        this.swarmParams({
          action: 'start',
          agent_id: agentId,
          ...(this.frame.cwd ? { cwd: this.frame.cwd } : {}),
        }),
      )
      return parseAcpLoginStatus(result)
    } catch (error) {
      return { ok: false, login_id: '', phase: 'error', needs_code: false, error: desktopError(error) }
    }
  }

  async pollAcpLogin(loginId: string): Promise<AcpLoginStatus> {
    const cleanId = loginId.trim()
    if (!cleanId) return { ok: false, login_id: '', phase: 'error', needs_code: false, error: 'Missing login session id' }
    try {
      const result = await this.bridge.call(
        'acp.login',
        this.swarmParams({ action: 'status', login_id: cleanId }),
      )
      return parseAcpLoginStatus(result)
    } catch (error) {
      return { ok: false, login_id: cleanId, phase: 'error', needs_code: false, error: desktopError(error) }
    }
  }

  async submitAcpLoginCode(loginId: string, code: string): Promise<AcpLoginStatus> {
    const cleanId = loginId.trim()
    if (!cleanId) return { ok: false, login_id: '', phase: 'error', needs_code: false, error: 'Missing login session id' }
    try {
      const result = await this.bridge.call(
        'acp.login',
        this.swarmParams({ action: 'submit', login_id: cleanId, code }),
      )
      return parseAcpLoginStatus(result)
    } catch (error) {
      return { ok: false, login_id: cleanId, phase: 'error', needs_code: false, error: desktopError(error) }
    }
  }

  async cancelAcpLogin(loginId: string): Promise<void> {
    const cleanId = loginId.trim()
    if (!cleanId) return
    await this.bridge.call(
      'acp.login',
      this.swarmParams({ action: 'cancel', login_id: cleanId }),
    ).catch(() => undefined)
  }

  /**
   * Fetch the connected ACP runtime's automatically discovered skills, commands,
   * capabilities, and config options (`acp.catalog`), gracefully falling back to
   * metadata already present on `snap.acpBinding`.
   */
  async loadAcpCatalog(options: { readonly silent?: boolean } = {}): Promise<AcpCatalogResult | null> {
    const fallbackBinding = this.frame.acpBinding
    const fallback = (): AcpCatalogResult | null => {
      if (!fallbackBinding?.connected || fallbackBinding.transport === 'native') return null
      const resolved = resolveExternalAcpCatalog(fallbackBinding)
      return {
        ok: true,
        connected: true,
        binding: fallbackBinding,
        commands: resolved.commands,
        skills: resolved.skills,
        ...(fallbackBinding.capabilities ? { capabilities: fallbackBinding.capabilities } : {}),
        config_options: fallbackBinding.config_options ?? [],
      }
    }
    const { isCurrent } = this.captureSessionRequest()
    try {
      const raw = await this.bridge.call('acp.catalog', this.swarmParams())
      if (!isCurrent()) return null
      if (raw.ok === false || raw.kind === 'error') {
        const message = str(raw.error) || str(raw.message) || 'acp.catalog failed'
        if (/unknown method|not found/i.test(message)) return fallback()
        if (!options.silent) this.patch({ acpError: message })
        return fallback()
      }
      const parsed = parseAcpCatalogResult(raw, this.frame.acpBinding ?? null)
      if (!parsed) return fallback()
      if (parsed.connected && parsed.binding && parsed.binding.transport !== 'native') {
        this.patch({ acpBinding: parsed.binding })
      }
      return parsed
    } catch (error) {
      if (!isCurrent()) return null
      const message = desktopError(error)
      if (!options.silent && !/unknown method|not found/i.test(message)) {
        this.patch({ acpError: message })
      }
      return fallback()
    }
  }

  private applyBtwEvent(payload: Readonly<Record<string, unknown>>): void {
    const current = this.getSnapshot()
    const folded = applyBtwEventPayload(current.sideQuestionTurns ?? [], payload)
    if (!folded) return
    const update = {
      sideQuestionTurns: folded.turns,
      sideQuestionStreaming: folded.streaming,
      sideQuestionError: folded.error,
    }
    // Coalesce high-frequency streaming deltas onto frame boundaries; flush
    // immediately on terminal transitions (completion / error / start).
    const kind = str(payload.kind)
    const streamingDelta = folded.streaming && !folded.error && (kind === 'delta' || kind === 'text' || kind === 'think' || kind === 'thinking' || kind === 'text_part' || kind === 'think_part')
    if (streamingDelta) this.patchSoft(update)
    else this.patch(update)
  }

  private swarmParams(extra: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      ...(this.sessionKey ? { session_key: this.sessionKey } : {}),
      ...(this.frame.currentId ? { session_id: this.frame.currentId } : {}),
      ...extra,
    }
  }

  async refreshSwarm(options: { channel?: string; silent?: boolean } = {}): Promise<void> {
    const { isCurrent } = this.captureSessionRequest()
    const version = ++this.swarmLoadVersion
    const current = (): boolean => isCurrent() && version === this.swarmLoadVersion
    if (!options.silent) this.patch({ swarmLoading: true, swarmError: null })
    try {
      const rawStatus = await this.bridge.call('swarm.status', this.swarmParams())
      if (!current()) return
      if (rawStatus.ok === false || rawStatus.kind === 'error') {
        const message = str(rawStatus.error) || str(rawStatus.message) || 'swarm.status failed'
        if (options.silent && /unknown method/i.test(message)) return
        throw new Error(message)
      }
      const status = parseSwarmStatus(rawStatus)
      if (!status) {
        if (!options.silent) this.patch({ swarmLoading: false })
        return
      }
      const shouldReadBoard = status.active || status.head > 0 || !options.silent
      let messages = this.frame.swarmMessages ?? []
      if (shouldReadBoard) {
        const boardParams: Record<string, unknown> = {
          limit: 200,
          ...(options.channel && options.channel !== '*' ? { channel: options.channel } : {}),
        }
        const rawBoard = await this.bridge.call('swarm.board', this.swarmParams(boardParams))
        if (!current()) return
        if (rawBoard.ok === false || rawBoard.kind === 'error') {
          const message = str(rawBoard.error) || str(rawBoard.message) || 'swarm.board failed'
          if (!options.silent || !/unknown method/i.test(message)) throw new Error(message)
        } else {
          const board = parseSwarmBoard(rawBoard)
          if (board) {
            messages = mergeSwarmMessages(messages, board.messages)
          }
        }
      }
      if (current()) {
        this.patch({
          swarm: status,
          swarmMessages: messages,
          swarmLoading: false,
          swarmError: null,
        })
      }
    } catch (error) {
      if (!current()) return
      const message = desktopError(error)
      if (options.silent && /unknown method/i.test(message)) return
      this.patch({ swarmLoading: false, swarmError: message })
    }
  }

  async postToSwarm(input: {
    readonly text: string
    readonly channel?: string
    readonly asFeedback?: boolean
  }): Promise<boolean> {
    const clean = input.text.trim()
    if (!clean) return false
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ swarmError: null })
    const targetChannel = input.channel && input.channel !== '*' ? input.channel : 'main'
    try {
      const result = await this.bridge.call(
        'swarm.post',
        this.swarmParams({
          text: clean,
          channel: targetChannel,
          ...(input.asFeedback ? { as_feedback: true } : {}),
        }),
      )
      if (result.ok === false || result.kind === 'error') {
        throw new Error(str(result.error) || str(result.message) || 'Could not post to Immortals Board')
      }
      if (isCurrent()) {
        const echoed = parseSwarmMessage(result.message)
        if (echoed) {
          this.patch({
            swarmMessages: mergeSwarmMessages(this.frame.swarmMessages ?? [], [echoed]),
            ...(this.frame.swarm ? { swarm: { ...this.frame.swarm, head: Math.max(this.frame.swarm.head, echoed.seq) } } : {}),
          })
        }
        void this.refreshSwarm({ silent: true })
      }
      return true
    } catch (error) {
      if (isCurrent()) this.patch({ swarmError: desktopError(error) })
      return false
    }
  }

  async killSwarmMember(name: string): Promise<boolean> {
    const clean = name.trim()
    if (!clean) return false
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ swarmError: null })
    try {
      const result = await this.bridge.call('swarm.kill', this.swarmParams({ name: clean }))
      if (result.ok === false || result.kind === 'error') {
        throw new Error(str(result.error) || str(result.message) || `Could not stop ${clean}`)
      }
      if (isCurrent()) await this.refreshSwarm({ silent: true })
      return true
    } catch (error) {
      if (isCurrent()) this.patch({ swarmError: desktopError(error) })
      return false
    }
  }

  async switchSwarmMemberModel(name: string, model?: string): Promise<boolean> {
    const cleanName = name.trim()
    if (!cleanName) return false
    const cleanModel = model?.trim()
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ swarmError: null })
    try {
      const result = await this.bridge.call(
        'swarm.model',
        this.swarmParams({
          name: cleanName,
          ...(cleanModel ? { model: cleanModel } : {}),
        }),
      )
      if (result.ok === false || result.kind === 'error') {
        throw new Error(str(result.error) || str(result.message) || `Could not switch model for ${cleanName}`)
      }
      if (isCurrent()) await this.refreshSwarm({ silent: true })
      return true
    } catch (error) {
      if (isCurrent()) this.patch({ swarmError: desktopError(error) })
      return false
    }
  }

  async activateSwarm(enabled = true): Promise<boolean> {
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ swarmError: null })
    try {
      const result = await this.bridge.call('swarm.activate', this.swarmParams({ enabled }))
      if (result.ok === false || result.kind === 'error') {
        throw new Error(str(result.error) || str(result.message) || 'Could not activate Immortals mode')
      }
      return true
    } catch (error) {
      if (isCurrent()) this.fail(error)
      return false
    }
  }

  async inspectSwarmMember(name: string, limit = 60): Promise<SwarmMemberInspectResult | null> {
    const clean = name.trim()
    if (!clean) {
      this.patch({ swarmInspect: null })
      return null
    }
    const { isCurrent } = this.captureSessionRequest()
    try {
      const result = await this.bridge.call('swarm.inspect', this.swarmParams({ name: clean, limit }))
      if (result.ok === false || result.kind === 'error') {
        throw new Error(str(result.error) || str(result.message) || `Could not inspect @${clean}`)
      }
      const parsed = parseSwarmMemberInspect(result)
      if (isCurrent() && parsed) {
        this.patch({ swarmInspect: parsed })
      }
      return parsed
    } catch (error) {
      if (isCurrent() && !/unknown method/i.test(desktopError(error))) {
        this.patch({ swarmError: desktopError(error) })
      }
      return null
    }
  }

  closeSwarmInspect(): void {
    if (this.frame.swarmInspect || this.frame.swarmInspectTarget) {
      this.patch({ swarmInspect: null, swarmInspectTarget: '' })
    }
  }

  private scheduleSwarmInspectRefresh(memberName: string): void {
    const now = Date.now()
    const wait = SWARM_INSPECT_THROTTLE_MS - (now - this.lastSwarmInspectAt)
    if (wait <= 0) {
      this.lastSwarmInspectAt = now
      void this.inspectSwarmMember(memberName)
      return
    }
    if (this.swarmInspectTimer) return
    this.swarmInspectTimer = setTimeout(() => {
      this.swarmInspectTimer = null
      const open = this.getSnapshot().swarmInspect
      if (open && open.member.name === memberName) {
        this.lastSwarmInspectAt = Date.now()
        void this.inspectSwarmMember(memberName)
      }
    }, wait)
  }

  private applySwarmEvent(payload: Readonly<Record<string, unknown>>): void {
    const parsed = parseSwarmEvent(payload)
    if (!parsed) return
    const snap = this.getSnapshot()
    const isActivityTick = parsed.kind === 'member_activity'
    const nextStatus = parsed.status
      ? parsed.status
      : snap.swarm
        ? applySwarmStatusDelta(snap.swarm, parsed)
        : undefined
    const statusChanged = nextStatus !== undefined && nextStatus !== snap.swarm
    const nextMessages = parsed.message
      ? mergeSwarmMessages(snap.swarmMessages ?? [], [parsed.message])
      : undefined
    const openInspect = snap.swarmInspect
    const targetName = parsed.member?.name ?? parsed.member_name
    const updatedInspectMember =
      openInspect
        ? parsed.member && parsed.member.name === openInspect.member.name
          ? parsed.member
          : nextStatus?.members.find(m => m.name === openInspect.member.name)
        : undefined
    const inspectMemberChanged = Boolean(updatedInspectMember && updatedInspectMember !== openInspect?.member)
    const inspectActivityApplies = Boolean(openInspect && parsed.activity && (!targetName || targetName === openInspect.member.name))
    const nextInspect =
      openInspect && (inspectMemberChanged || inspectActivityApplies)
        ? {
            ...openInspect,
            ...(updatedInspectMember ? { member: updatedInspectMember } : {}),
            ...(inspectActivityApplies && parsed.activity ? { activity: parsed.activity } : {}),
          }
        : undefined
    if (statusChanged || nextMessages !== undefined || nextInspect !== undefined) {
      const update = {
        ...(statusChanged && nextStatus !== undefined ? { swarm: nextStatus } : {}),
        ...(nextMessages !== undefined ? { swarmMessages: nextMessages } : {}),
        ...(nextInspect !== undefined ? { swarmInspect: nextInspect } : {}),
      }
      if (isActivityTick && nextMessages === undefined) this.patchSoft(update)
      else this.patch(update)
    }
    if (openInspect && (isActivityTick || parsed.kind === 'member_updated') && (!targetName || targetName === openInspect.member.name)) {
      if (isActivityTick) this.scheduleSwarmInspectRefresh(openInspect.member.name)
      else void this.inspectSwarmMember(openInspect.member.name)
    }
    // Lightweight member_activity deltas intentionally omit `status` (saving ~28 KB/tick).
    // Only trigger a fallback full refresh when we have no hydrated swarm yet or a non-activity event lacked payload.
    if (!parsed.status && !parsed.message && !parsed.member && !(isActivityTick && snap.swarm)) {
      void this.refreshSwarm({ silent: true })
    }
  }

  /** Latest request wins; session switches and workspace broadcasts invalidate old replies. */
  async refreshLearning(): Promise<void> {
    const key = this.sessionKey, revision = this.frame.sessionOpenRevision, version = ++this.learningLoadVersion
    const current = () => key === this.sessionKey && revision === this.frame.sessionOpenRevision && version === this.learningLoadVersion
    this.patch({ learningLoading: true, learningError: '' })
    try {
      const result = readLearningList(await this.bridge.call('learning.list', { session_key: key }))
      if (current()) this.patch({ learning: result, learningCounts: result.counts })
    } catch (error) {
      if (current()) this.patch({ learningError: desktopError(error) })
    } finally {
      if (current()) this.patch({ learningLoading: false })
    }
  }

  /** Only explicit UI clicks call this; a reason and status are checked before any write RPC. */
  async reviewLearning(target: LearningTarget, id: string, action: LearningAction, reason = ''): Promise<void> {
    if (this.learningReviewToken) return
    const resultKey = `${target}:${id}`
    const show = (result: LearningResult) => this.patch({ learningResults: { ...this.frame.learningResults, [resultKey]: result } })
    const proposal = this.frame.learning?.proposals.find(row => row.id === id)
    const lesson = this.frame.learning?.lessons.find(row => row.id === id)
    if (action !== 'approve' && !reason.trim()) { show({ report: '', error: 'A reason is required.' }); return }
    if (action === 'rollback' && (target !== 'lesson' || lesson?.reviewStatus !== 'approved')) { show({ report: '', error: 'Only an approved lesson can be rolled back.' }); return }
    if (target === 'proposal' && proposal?.status !== 'pending') { show({ report: '', error: 'This proposal is not pending in the current workspace. Refresh to see its outcome.' }); return }
    if (target === 'lesson' && action !== 'rollback' && lesson?.reviewStatus !== 'pending') { show({ report: '', error: 'Only pending lessons can be approved or rejected.' }); return }
    const token = {}, key = this.sessionKey, revision = this.frame.sessionOpenRevision
    this.learningReviewToken = token
    const current = () => token === this.learningReviewToken && key === this.sessionKey && revision === this.frame.sessionOpenRevision
    this.patch({ learningReviewBusy: true })
    show({ report: '', error: '' })
    try {
      // Do not use desktopCall: a refusal can carry a report with every lint finding.
      const result = await this.bridge.call('learning.review', { session_key: key, target, id, action, ...(action === 'approve' ? {} : { reason: reason.trim() }) })
      if (typeof result.report !== 'string') throw new Error(text(result.error) || 'The runtime did not return a review report. Refresh to check the outcome before reviewing again.')
      if (current()) show({ report: result.report, error: result.ok === true ? '' : text(result.error) || 'The review was not accepted.' })
    } catch (error) {
      if (current()) show({ report: '', error: desktopError(error) })
    } finally {
      if (current()) {
        await this.refreshLearning()
        if (current()) { this.learningReviewToken = null; this.patch({ learningReviewBusy: false }) }
      }
    }
  }

  /** Approve a fixed snapshot of one kind; never broaden a human's batch after the click. */
  async reviewAllLearning(target: LearningTarget): Promise<{ approved: number; failed: number }> {
    if (this.learningReviewToken || this.frame.learningLoading || this.frame.connection !== 'online') return { approved: 0, failed: 0 }
    const pending = target === 'lesson'
      ? this.frame.learning?.lessons.filter(lesson => lesson.reviewStatus === 'pending') ?? []
      : this.frame.learning?.proposals.filter(proposal => proposal.status === 'pending') ?? []
    if (!pending.length) return { approved: 0, failed: 0 }
    const token = {}, key = this.sessionKey, revision = this.frame.sessionOpenRevision
    this.learningReviewToken = token
    const current = () => token === this.learningReviewToken && key === this.sessionKey && revision === this.frame.sessionOpenRevision
    const results: Record<string, LearningResult> = { ...this.frame.learningResults }
    let approved = 0, failed = 0
    this.patch({ learningReviewBusy: true, learningError: '' })
    try {
      for (const item of pending) {
        if (!current()) break
        const resultKey = `${target}:${item.id}`
        try {
          const result = await this.bridge.call('learning.review', { session_key: key, target, id: item.id, action: 'approve' })
          if (result.ok !== true || typeof result.report !== 'string') {
            failed += 1
            results[resultKey] = { report: text(result.report), error: text(result.error) || 'The approval was not accepted.' }
          } else {
            approved += 1
            results[resultKey] = { report: result.report, error: '' }
          }
        } catch (error) {
          failed += 1
          results[resultKey] = { report: '', error: desktopError(error) }
        }
      }
      if (current()) this.patch({ learningResults: results })
    } finally {
      if (current()) {
        await this.refreshLearning()
        if (current()) {
          this.learningReviewToken = null
          this.patch({
            learningReviewBusy: false,
            learningError: failed ? `${approved} approved; ${failed} failed. Review the individual outcomes below.` : '',
          })
        }
      }
    }
    return { approved, failed }
  }

  start(bridge?: XerxesLike): void {
    if (this.started || typeof window === 'undefined') return
    this.started = true
    if (bridge) this.bridge = bridge
    this.unsubEvents = window.xerxes.onEvent(event => this.onEvent(event))
    // The app's own updates come from the main process, not the runtime.
    this.bridge.onAppUpdate?.(state => this.patch({ appUpdate: appUpdateView(state) }))
    void this.bridge.appUpdate?.('state').then(state => this.patch({ appUpdate: appUpdateView(state) })).catch(() => undefined)
    window.xerxes.onProviderRelay?.(() => this.refreshProviderRelay())
    // Workspace gate: no folder, no daemon, no initialize — the shell asks
    // for one instead of inventing a target the user never chose.
    const gate = (this.bridge as XerxesLike & { getWorkspace?: () => Promise<string | null> })
      .getWorkspace?.()
    void Promise.all([Promise.resolve(gate), this.bridge.getContextScope?.()]).then(([saved, scope]) => {
      this.patch({ storageScope: scope && scope !== 'local' ? scope + ':' : '' })
      if (saved === null) {
        this.patch({ noWorkspace: true, connection: 'offline' })
        return
      }
      this.initializeLive(typeof saved === 'string' ? saved : '')
    }).catch(error => this.wentOffline(error))
    void this.bridge.getContextHost?.().then(host => {
      if (typeof host === 'string' && host.trim()) this.patch({ contextHost: host.trim().slice(0, 128) })
    }).catch(() => undefined)
    void this.refreshContexts()
    this.heartbeat = setInterval(() => void this.beat(), HEARTBEAT_MS)
    this.heartbeat.unref?.()
    // A covered workspace skips its heartbeat (see beat()); catch up the
    // moment it is shown again instead of waiting for the next tick.
    if (typeof document !== 'undefined') {
      const catchUp = () => { if (!pageIsBackground()) void this.beat(true) }
      document.addEventListener('visibilitychange', catchUp)
      // A covered base page is marked by the main process, not by visibility.
      if (typeof MutationObserver !== 'undefined') {
        new MutationObserver(catchUp).observe(document.documentElement, { attributes: true, attributeFilter: ['data-occluded'] })
      }
    }
  }

  private initializeLive(workspace: string): void {
    this.patch({ cwd: workspace })
    void Promise.all([this.bridge.getResumeSession?.(), this.bridge.startsFresh?.(), this.bridge.freshPreset?.()]).then(async ([explicitId, fresh, preset]) => {
      const id = explicitId || (workspace && !fresh ? selectedSession((this.frame.storageScope ?? '') + workspace) : null)
      // A fresh view opened for Creator mode (while another task ran) starts on that preset.
      const start: Record<string, unknown> = id ? { resume_session_id: id } : fresh && preset ? { agent_id: preset } : {}
      try {
        return await this.initializeSelfHealing(start)
      } catch (error) {
        // Old desktop builds could save a session under the wrong workspace.
        // Keep the daemon's isolation check: never move or rebind that history.
        if (!id || connectionFailureKind(error instanceof Error ? error.message : String(error)) !== 'session') throw error
        this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
        const result = await this.initialize({})
        this.builder.push('notification', { severity: 'warning', message: 'The saved session belongs to another workspace. Opened a new session here; the original conversation is unchanged and remains available in its workspace.' })
        this.notify()
        return result
      }
    }).then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
        // The composer shows where a Hybrid task ran, so the mode is needed before Settings opens.
        void this.loadRunMode()
      },
      error => this.wentOffline(error),
    )
  }

  private refreshingContexts = false
  /** When the heartbeat last refreshed the session lists (see beat()). */
  private lastListRefresh = 0
  private stopArmTimer: ReturnType<typeof setTimeout> | undefined
  private async refreshContexts(): Promise<void> {
    if (!this.bridge.getContexts || this.refreshingContexts) return
    this.refreshingContexts = true
    try {
      const contexts = await this.bridge.getContexts()
      const prev = this.getSnapshot().contexts
      if (JSON.stringify(prev) !== JSON.stringify(contexts)) this.patch({ contexts })
    }
    catch (error) { this.patch({ workspaceError: desktopError(error) }) }
    finally { this.refreshingContexts = false }
  }

  async activateContext(id: number, sessionId?: string): Promise<void> {
    try { await this.bridge.activateContext?.(id, sessionId) }
    catch (error) { this.patch({ error: desktopError(error), workspaceError: desktopError(error) }) }
  }

  /**
   * Initialize, but survive a daemon whose memory for our session key
   * disagrees with disk (crash, restored backup, externally removed
   * transcripts). The daemon answers such binds with a transcript_generation
   * conflict; retrying under a brand-new key sidesteps the poisoned binding
   * instead of leaving the shell stuck on "Daemon offline" forever.
   */
  private async initializeSelfHealing(extra: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    try {
      return await this.initialize(extra)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      // Still running a turn with no client of ours attached (the app was
      // restarted, or the runtime updated, mid-turn): attach to it instead of
      // treating it as someone else's session and starting a new task — that
      // hid a chat that was still working.
      if (typeof extra.resume_session_id === 'string' && RUNNING_ELSEWHERE.test(message)) {
        const attached = await this.attachRunningSession(extra.resume_session_id)
        if (attached) return attached
      }
      // The conversation being resumed is not on disk: a new task that never
      // got a message when the runtime restarted, or one deleted elsewhere.
      // Retrying the same id can never succeed, so open a fresh conversation
      // in this workspace instead of stranding the window on an error.
      if (extra.resume_session_id && /saved conversation is missing/i.test(message)) {
        const hadConversation = this.builder.some(block => block.kind === 'user' || block.kind === 'agent')
        // A fresh key: the old one may still name a live session elsewhere.
        this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
        this.patch({ currentId: '', sessionKey: this.sessionKey })
        this.builder.reset()
        const result = await this.initialize({})
        if (hadConversation) {
          this.builder.push('notification', { severity: 'warning', message: 'The previous conversation is no longer saved in this workspace, so a new one was started. Open older conversations from the sidebar.' })
          this.notify()
        }
        return result
      }
      if (!/transcript_generation|divergent append/i.test(message)) throw error
      // A bare initialize with a fresh key — resuming would just re-bind the
      // poisoned session. Old chats stay on disk and in the sidebar.
      this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
      this.patch({ currentId: '', sessionKey: this.sessionKey })
      this.builder.reset()
      return this.initialize({})
    }
  }

  private async submitPrompt(
    promptText: string,
    display?: string,
    images: readonly ComposerImage[] = [],
  ): Promise<boolean> {
    const trimmed = promptText.trim()
    if (!trimmed) return false
    const shown = display?.trim() || trimmed
    if (this.frame.turnActive) return this.steer(trimmed, images)
    const sessionKey = this.sessionKey
    if (this.preparingSubmissions.has(sessionKey)) return false
    // A session bound to an external ACP runtime (e.g. Antigravity, Claude Code)
    // whose process exited or was reopened must reconnect that external runtime
    // before sending — never silently fall through to native Xerxes / local Mac.
    if (
      !this.isExternalAcpConnected()
      && this.frame.savedAcpBinding
      && this.frame.savedAcpBinding.transport !== 'native'
    ) {
      const reconnected = await this.reconnectSavedAcp()
      if (!reconnected || this.sessionKey !== sessionKey) return false
    }
    this.lastUser = shown
    const optimistic = {
      text: shown,
      id: this.builder.pushUser(shown, false, images.map(image => `data:${image.mediaType};base64,${image.data}`)),
      acknowledged: false,
    }
    this.optimisticSubmissions.set(sessionKey, optimistic)
    this.preparingSubmissions.set(sessionKey, optimistic)
    this.notify()
    try {
      // A submission id makes a busy session (a goal round, monitor or
      // schedule preparing its turn before turn_begin) refuse the submit
      // up front. Without one the daemon acknowledged it {ok:true} and only
      // then dropped it, so the bubble looked delivered and the draft was gone.
      const result = await this.bridge.call('turn.submit', {
        session_key: sessionKey,
        text: trimmed,
        submission_id: crypto.randomUUID(),
        ...(shown === trimmed ? {} : { display_text: shown }),
        ...(images.length ? { images: images.map(image => ({ media_type: image.mediaType, data: image.data })) } : {}),
      })
      if (result.ok === false) throw new Error(str(result.error) || 'Submission refused')
      if (this.sessionKey === sessionKey) this.cameOnline()
      return true
    } catch (error) {
      // The daemon rejected the submit (e.g. no provider configured): the
      // optimistic bubble never happened — roll it back rather than leave
      // a delivered-looking ghost above the error.
      if (this.preparingSubmissions.get(sessionKey) === optimistic) this.preparingSubmissions.delete(sessionKey)
      if (this.optimisticSubmissions.get(sessionKey) === optimistic) this.optimisticSubmissions.delete(sessionKey)
      if (this.sessionKey === sessionKey) {
        if (!optimistic.acknowledged) this.builder.rollbackUser(shown)
        this.fail(error)
      }
      return optimistic.acknowledged
    } finally {
      // RPC acceptance precedes snapshot preparation and turn_begin. Keep
      // the pending state until the daemon actually starts or ends the turn.
      this.notify()
    }
  }

  /**
   * Submit or steer. While a turn runs, text goes to `turn.steer` — the
   * daemon queues it between steps and the store mirrors it visibly; when
   * idle it is a normal `turn.submit`. `display` is what the transcript
   * shows when it differs from what the model is sent.
   */
  async submit(text: string, display?: string, images: readonly ComposerImage[] = []): Promise<boolean> {
    const trimmed = text.trim()
    if (!trimmed) return false
    if (!trimmed.startsWith('/')) {
      return this.submitPrompt(trimmed, display, images)
    }
    // Strict external ACP routing: any slash command or discovered skill
    // advertised by the connected ACP runtime forwards directly to that agent
    // instead of being hijacked by Xerxes's native slash handlers (`/acp` stays host-owned).
    if (shouldForwardSlashToAcp(trimmed, this.frame.acpBinding)) {
      return this.submitPrompt(trimmed, display, images)
    }
    const sideCmd = parseSideQuestionSlashCommand(trimmed)
    if (sideCmd) {
      if (sideCmd.kind === 'open') {
        this.openSideQuestion()
        return true
      }
      return this.askSideQuestion(sideCmd.question, sideCmd.model)
    }
    const swarmCmd = parseSwarmSlashCommand(trimmed)
    if (swarmCmd) {
      if (swarmCmd.kind === 'open_board') {
        this.openSwarmBoard()
        return true
      }
      if (swarmCmd.kind === 'kill') {
        const ok = await this.killSwarmMember(swarmCmd.target)
        if (!ok) {
          this.fail(new Error(this.frame.swarmError || `Could not stop ${swarmCmd.target}`))
          return false
        }
        this.builder.push('notification', {
          severity: 'info',
          message: swarmCmd.target === 'all' ? 'Stopped all Immortals workers.' : `Stopped Immortals member @${swarmCmd.target}.`,
        })
        this.notify()
        return true
      }
      if (swarmCmd.kind === 'activate_task') {
        const activated = await this.activateSwarm(true)
        if (!activated) return false
        this.openSwarmBoard()
        return this.submitPrompt(swarmCmd.task, undefined, images)
      }
    }
    // /goal routes to its dedicated durable-state RPC; everything else goes
    // to the daemon's native slash table, which owns rejection.
    if (trimmed === '/goal' || trimmed.startsWith('/goal ')) {
      const result = await this.setGoal(trimmed.slice(5).trim())
      if (!result.ok) this.fail(new Error(result.text))
      return result.ok
    }
    if (trimmed === '/plan' || trimmed.startsWith('/plan ') || trimmed === '/plan off') {
      // /plan steers planning; the mode flip itself is the RPC's job.
      const rest = trimmed.slice(5).trim()
      const acpPlan = this.acpPlanModePair()
      if (acpPlan && (!rest || rest === 'off' || rest === 'on')) {
        const target = rest === 'off' ? acpPlan.defaultId : rest === 'on' ? acpPlan.planId : acpPlan.isPlan ? acpPlan.defaultId : acpPlan.planId
        return this.setAcpMode(target)
      }
      if (!rest || rest === 'off') return this.setPlanMode(!this.frame.planMode)
      else return this.steer(rest)
    }
    // The daemon's /new and /resume re-key this connection behind the window's
    // back: the window kept submitting under its old key, and the reply landed
    // in a session it never showed. Use the window's own navigation instead.
    const [command = '', ...args] = trimmed.split(/\s+/)
    if (command === '/new' || command === '/reset') {
      this.newChat()
      return true
    }
    if (command === '/xerxes' && !args.length) { this.openLearning(); return true }
    if (command === '/resume' && args.length) return this.resumeMatching(args.join(' '))
    const sessionKey = this.sessionKey
    const previousResult = this.slashResult
    const alreadyReported = (text: string) => {
      const current = this.slashResult
      if (!current || current === previousResult || current.sessionKey !== sessionKey) return false
      // Legacy daemons describe an unknown command differently in the event
      // and RPC reply. Both identify the same command and rejection.
      const normalize = (value: string) => value.trim().replace(/^Unknown slash command:/, 'Unknown command:').replace(/ \(type \/help\)\.$/, '')
      return normalize(current.text) === normalize(text)
    }
    try {
      const result = await this.bridge.call('slash', { command: trimmed })
      if (this.sessionKey !== sessionKey) return result.ok === true
      const message = str(result.output) || str(result.error) || (result.ok === true ? 'ok' : 'command failed')
      const emitted = this.slashResult !== previousResult && this.slashResult?.sessionKey === sessionKey
      if (!alreadyReported(message) && !(result.ok === true && !result.output && emitted)) {
        this.builder.push('notification', { severity: result.ok === true ? 'info' : 'error', message })
      }
      this.notify()
      if (trimmed.startsWith('/permissions')) void this.loadProviders()
      if (trimmed === '/acp' || trimmed.startsWith('/acp ')) void this.refreshAcpAgents({ silent: true })
      void this.refreshSessions()
      return result.ok === true
    } catch (error) {
      if (this.sessionKey === sessionKey && !alreadyReported(desktopError(error))) this.fail(error)
      return false
    }
  }

  /** Queue steering text daemon-side; mirror it locally until consumed. */
  async steer(text: string, images: readonly ComposerImage[] = []): Promise<boolean> {
    const cleaned = text.trim()
    if (!cleaned && !images.length) return false
    const sessionKey = this.sessionKey
    const steerId = crypto.randomUUID()
    // The daemon's steer_input echo can land before this call returns, and the
    // echo carries only an image count. Keep the pictures with the pending
    // steer so whichever path shows the message shows them too.
    const shownImages = images.map(image => `data:${image.mediaType};base64,${image.data}`)
    const pending: { session: string; text: string; images: readonly string[]; shown: boolean; echoed: boolean; blockId?: number } = {
      session: sessionKey,
      text: cleaned,
      images: shownImages,
      shown: false,
      echoed: false,
    }
    this.pendingSteers.set(steerId, pending)
    try {
      const result = await this.bridge.call('turn.steer', { session_key: sessionKey, content: cleaned, client_steer_id: steerId, ...(images.length ? { images: images.map(image => ({ media_type: image.mediaType, data: image.data })) } : {}) })
      if (result.ok === true) {
        if (this.sessionKey === sessionKey) {
          const appliedLive = this.frame.turnActive && (result.disposition === 'live_steer' || !this.pendingSteers.has(steerId))
          if (!pending.shown) {
            pending.blockId = this.builder.pushUser(cleaned, false, shownImages)
            pending.shown = true
          }
          if (cleaned) this.recentSteerTexts.add(cleaned)
          if (appliedLive) {
            this.pendingSteers.delete(steerId)
            this.notify()
          } else if (this.frame.turnActive) {
            this.queue = [
              ...this.queue,
              {
                id: this.seq++,
                text: cleaned,
                ...(pending.blockId !== undefined ? { blockId: pending.blockId } : {}),
                ...(images.length ? { images } : {}),
              },
            ]
            this.patch({ queue: this.queue })
          } else {
            // The turn ended while this steer was in flight; turn_end came
            // first. The daemon kept the text for the next turn but started
            // none, so a "queued" entry would wait for a reply forever.
            this.builder.push('notification', { severity: 'info', message: 'The task finished before this message reached it. It is saved and goes with your next message.' })
            this.notify()
          }
        }
        return true
      } else {
        this.pendingSteers.delete(steerId)
        if (this.sessionKey === sessionKey) this.fail(new Error(str(result.error) || 'steering refused'))
        return false
      }
    } catch (error) {
      this.pendingSteers.delete(steerId)
      if (this.sessionKey === sessionKey) this.fail(error)
      return false
    }
  }

  private steeringNow = false
  private readonly recentSteerTexts = new Set<string>()

  private waitForTurnState(
    sessionKey: string,
    targetActive: boolean,
    timeoutMs: number,
  ): Promise<boolean> {
    if (this.sessionKey !== sessionKey) return Promise.resolve(false)
    if (this.frame.turnActive === targetActive) return Promise.resolve(true)
    return new Promise<boolean>(resolve => {
      let done = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const finish = (ok: boolean) => {
        if (done) return
        done = true
        if (timer !== undefined) clearTimeout(timer)
        unsubscribe()
        resolve(ok)
      }
      const unsubscribe = this.subscribe(() => {
        if (this.sessionKey !== sessionKey) { finish(false); return }
        if (this.frame.turnActive === targetActive) finish(true)
      })
      timer = setTimeout(() => finish(this.sessionKey === sessionKey && this.frame.turnActive === targetActive), timeoutMs)
    })
  }

  /**
   * Interrupt the active turn and immediately send `text` as the next turn.
   * Works across both native Xerxes and external ACP runtimes (including older
   * remote daemons) via `turn.cancel` → wait for settle → `turn.submit`.
   */
  async steerNow(
    text: string,
    images: readonly ComposerImage[] = [],
    options?: {
      readonly preserveQueueIds?: ReadonlySet<number>
      readonly promotedFromQueue?: boolean
      readonly promotedBlockId?: number
      readonly restoreQueue?: readonly QueueItem[]
    },
  ): Promise<boolean> {
    const cleaned = text.trim()
    if (!cleaned && !images.length) return false
    if (!this.frame.turnActive) {
      return this.submit(cleaned, undefined, images)
    }
    if (this.steeringNow) return false
    const sessionKey = this.sessionKey
    const priorQueue = options?.restoreQueue ? [...options.restoreQueue] : [...this.queue]
    const remaining = options?.preserveQueueIds
      ? this.queue.filter(item => options.preserveQueueIds!.has(item.id))
      : [...this.queue]
    this.steeringNow = true
    try {
      if (this.stopArmTimer !== undefined) {
        clearTimeout(this.stopArmTimer)
        this.stopArmTimer = undefined
      }
      if (this.frame.stopArmed || this.frame.approval || this.frame.question) {
        this.patch({ stopArmed: false, approval: null, question: null })
      }
      // Antigravity (`agy acp 1.5+`) supports native live mid-turn steering via
      // `turn.steer({ live: true })`. Never send `turn.cancel` before trying live
      // steer — cancelling a mid-step `agy` tool call can corrupt the trajectory.
      if (
        this.frame.acpBinding?.connected &&
        this.frame.acpBinding.provider === 'antigravity' &&
        !this.frame.approval &&
        !this.frame.question
      ) {
        const steerId = crypto.randomUUID()
        const shownImages = images.map(image => `data:${image.mediaType};base64,${image.data}`)
        const pending: { session: string; text: string; images: readonly string[]; shown: boolean; echoed: boolean; blockId?: number } = {
          session: sessionKey,
          text: cleaned,
          images: shownImages,
          shown: Boolean(options?.promotedFromQueue),
          echoed: false,
          ...(options?.promotedBlockId !== undefined ? { blockId: options.promotedBlockId } : {}),
        }
        this.pendingSteers.set(steerId, pending)
        const liveRes = await this.bridge.call('turn.steer', {
          session_key: sessionKey,
          content: cleaned,
          live: true,
          client_steer_id: steerId,
          ...(images.length ? { images: images.map(image => ({ media_type: image.mediaType, data: image.data })) } : {}),
        })
        if (liveRes.ok === true && liveRes.disposition === 'live_steer') {
          this.pendingSteers.delete(steerId)
          if (this.sessionKey === sessionKey) {
            if (!pending.shown) {
              pending.blockId = this.builder.pushUser(cleaned, false, shownImages)
              pending.shown = true
            }
            if (cleaned) this.recentSteerTexts.add(cleaned)
            this.notify()
          }
          return true
        }
        this.pendingSteers.delete(steerId)
      }
      await this.bridge.call('turn.cancel', { session_key: sessionKey, drop_pending_steers: true })
      const settled = await this.waitForTurnState(sessionKey, false, 4500)
      if (this.sessionKey !== sessionKey) return false
      if (!settled) {
        if (options?.promotedFromQueue) {
          this.queue = priorQueue
          this.patch({ queue: this.queue })
        }
        this.fail(new Error('Current step is still stopping — try Steer now again in a moment.'))
        return false
      }
      // Remove any queued user bubbles that `steer()` already mirrored into the
      // transcript — even if tool calls or streaming deltas landed after them
      // before `Steer now` ran — so promoting or re-queuing never duplicates them.
      let removedQueuedBubble = false
      if (options?.promotedFromQueue) {
        if (options.promotedBlockId !== undefined && this.builder.removeBlockById(options.promotedBlockId)) {
          removedQueuedBubble = true
        } else {
          this.builder.rollbackUser(cleaned, true)
          removedQueuedBubble = true
        }
      }
      for (const item of remaining) {
        if (item.blockId !== undefined && this.builder.removeBlockById(item.blockId)) {
          removedQueuedBubble = true
        } else {
          this.builder.rollbackUser(item.text, true)
          removedQueuedBubble = true
        }
      }
      if (removedQueuedBubble) this.notify()
      const ok = await this.submit(cleaned, undefined, images)
      if (ok && remaining.length > 0 && this.sessionKey === sessionKey) {
        const started = await this.waitForTurnState(sessionKey, true, 1500)
        if (started && this.sessionKey === sessionKey) {
          for (const item of remaining) {
            await this.steer(item.text, item.images ?? [])
          }
        }
      }
      return ok
    } catch (error) {
      if (this.sessionKey === sessionKey) this.fail(error)
      return false
    } finally {
      this.steeringNow = false
    }
  }

  /** Promote a queued follow-up message: interrupt the active step and run it now. */
  async promoteQueued(id: number): Promise<boolean> {
    if (this.steeringNow) return false
    const item = this.queue.find(entry => entry.id === id)
    if (!item) return false
    const restoreQueue = [...this.queue]
    const otherIds = new Set(this.queue.filter(entry => entry.id !== id).map(entry => entry.id))
    this.queue = this.queue.filter(entry => entry.id !== id)
    this.patch({ queue: this.queue })
    return this.steerNow(item.text, item.images ?? [], {
      preserveQueueIds: otherIds,
      promotedFromQueue: true,
      ...(item.blockId !== undefined ? { promotedBlockId: item.blockId } : {}),
      restoreQueue,
    })
  }

  dropQueued(id: number): void {
    this.queue = this.queue.filter(item => item.id !== id)
    this.patch({ queue: this.queue })
  }

  /** Escape over a running turn: arm on the first press, stop on the second. */
  escapeStop(): void {
    if (this.stopArmTimer !== undefined) clearTimeout(this.stopArmTimer)
    if (this.frame.stopArmed) {
      this.stopArmTimer = undefined
      this.patch({ stopArmed: false })
      this.cancel()
      return
    }
    this.patch({ stopArmed: true })
    this.stopArmTimer = setTimeout(() => {
      this.stopArmTimer = undefined
      this.patch({ stopArmed: false })
    }, STOP_CONFIRM_MS)
  }

  cancel(): void {
    if (this.stopArmTimer !== undefined) {
      clearTimeout(this.stopArmTimer)
      this.stopArmTimer = undefined
    }
    if (this.frame.stopArmed) this.patch({ stopArmed: false })
    // Cancelling force-rejects any pending approval/question daemon-side.
    // Leaving the card up would offer buttons that can only answer
    // "refused" while keeping the header stuck on "needs input".
    if (this.frame.approval || this.frame.question) this.patch({ approval: null, question: null })
    // `turn.cancel` routes through `cancelTrackedTurn` to cancel active native
    // and external ACP turns cleanly; only fall back to `acp.cancel` when an
    // older remote daemon returns `{ ok: false }` (e.g. out-of-turn ACP wakeup).
    const sessionId = this.frame.currentId
    const externalConnected = Boolean(this.frame.acpBinding?.connected && sessionId)
    void this.bridge
      .call('turn.cancel', { session_key: this.sessionKey })
      .then(result => {
        if (result?.ok === false && externalConnected && this.frame.currentId === sessionId) {
          return this.bridge.call('acp.cancel', { session_id: sessionId })
        }
        return undefined
      })
      .catch(error => this.fail(error))
  }

  /**
   * Fold a decision card away without answering it. The request stays
   * pending in the daemon, so it stays in the snapshot too: clearing it left
   * the turn parked on an approval nothing could bring back, with the
   * "needs input" badge gone. `showInteraction` reopens it.
   */
  dismissInteraction(): void {
    if (this.frame.approval || this.frame.question) this.patch({ interactionHidden: true })
  }

  showInteraction(): void {
    if (this.frame.interactionHidden) this.patch({ interactionHidden: false })
  }

  /** Ask the attached process to exit; DaemonRpc reconnects and launches this build. */
  /**
   * The person asked to restart now: running work stops (armed goals re-arm
   * in the new runtime). A runtime too old to re-arm is shut down instead,
   * and this conversation's goal is resumed once the new one is up.
   */
  restartRuntimeNow(): Promise<void> {
    return this.restartDaemon(true, true)
  }

  async restartDaemon(allowLegacy = true, force = false, purpose: 'update' | 'mode' = 'update'): Promise<void> {
    if (this.updatingRuntime) return
    if (this.frame.daemonWarning?.startsWith('The app is older')) {
      this.patch({ runtimeUpdate: 'failed', runtimeUpdateMessage: 'Update and relaunch the desktop app. Its bundled runtime is older than the running runtime.' })
      return
    }
    this.updatingRuntime = true
    this.patch({ runtimeUpdate: 'checking', runtimeUpdateMessage: force ? 'Stopping running work and restarting…' : purpose === 'mode' ? 'Applying the new mode: checking for running work…' : 'Checking for running work…' })
    const goalWasActive = force && parseGoal(this.frame.goal)?.phase === 'active'
    try {
      const result = await this.bridge.call('desktop.restartRuntime', { session_key: this.sessionKey, allow_legacy: allowLegacy, ...(force ? { force: true } : {}) })
      if (result.ok !== true) {
        if (result.busy === true) {
          const blockers = Array.isArray(result.blockers) ? result.blockers.filter((item): item is string => typeof item === 'string') : []
          this.patch(result.waiting_for_goal_round === true
            ? { runtimeUpdate: 'waiting', runtimeBlockers: blockers, runtimeUpdateMessage: 'Update queued. It installs when the current goal round ends, and the goal continues after the restart.' }
            : { runtimeUpdate: 'waiting', runtimeBlockers: blockers, runtimeUpdateMessage: purpose === 'mode' ? 'Mode saved. It applies when all running work finishes.' : 'Update queued. It will install automatically when all running work finishes.' })
          return
        }
        throw new Error(str(result.error) || 'This runtime cannot update automatically. Restart the workspace runtime from its terminal, then reopen the app.')
      }
      this.patch({ runtimeUpdate: 'restarting', runtimeUpdateMessage: purpose === 'mode' ? 'Restarting the runtime with the new mode…' : 'Reconnecting to the updated runtime…' })
      await this.initialize(this.frame.currentId ? { resume_session_id: this.frame.currentId } : {})
      if (this.frame.daemonWarning) throw new Error('The runtime restarted, but its build still differs. Quit and relaunch the latest app; check any custom runtime path.')
      this.patch({ runtimeUpdate: undefined, runtimeUpdateMessage: undefined, runtimeBlockers: [] })
      if (purpose === 'mode') { void this.loadRunMode(); void this.loadProviders() }
      // The old runtime could not carry the goal across; pick it back up.
      if (result.legacy_forced === true && goalWasActive) await this.submit('/goal resume')
    } catch (error) {
      this.patch({ runtimeUpdate: 'failed', runtimeUpdateMessage: error instanceof Error ? error.message : String(error) })
    } finally {
      this.updatingRuntime = false
    }
  }

  private sessionNavigation: Promise<void> = Promise.resolve()
  private sessionNavigationVersion = 0
  private openingSession = false
  private sessionNavigationNeedsRestore = false

  /** `known` is a row the caller already resolved beyond the sidebar's page (a `/resume` lookup). */
  openSession(id: string, known?: SessionRow): Promise<void> {
    const previousVersion = this.sessionNavigationVersion
    const version = ++this.sessionNavigationVersion
    this.openingSession = true
    const pending = this.sessionNavigation.then(async () => {
      if (version !== this.sessionNavigationVersion) return
      let navigated = false
      try {
        navigated = await this.openSessionNow(id, version, known)
        // A superseded click whose initialize the daemon accepted left the
        // connection bound to that chat. A click that hands off to another
        // window or workspace never re-binds it, so slash commands and the
        // other chat's events kept landing in this one until the next move.
        if (!navigated && version === this.sessionNavigationVersion) await this.restoreSupersededNavigation()
      } finally {
        if (version === this.sessionNavigationVersion) {
          // Not every click navigates: opening the already-current session
          // mid-turn is a no-op, and a busy or cross-workspace target is
          // handed to another window instead. `captureSessionRequest` treats
          // ANY advance of this counter as "the user moved on", so leaving the
          // bump in place after those exits silently discarded every reply
          // already in flight — most damagingly `setPermissionMode`, which
          // early-returns while `permissionUpdating` is set and whose buttons
          // are disabled by it, leaving no way to retry in that session.
          // Roll back only when no later click has claimed the counter; if one
          // has, that click owns it and discarding really is correct.
          if (!navigated) this.sessionNavigationVersion = previousVersion
          this.openingSession = false
          if (this.frame.connection === 'connecting') this.patch({ connection: 'online' })
        }
      }
    })
    this.sessionNavigation = pending
    return pending
  }

  /** Re-bind the connection to the chat on screen after a superseded click bound it elsewhere. */
  private async restoreSupersededNavigation(): Promise<void> {
    if (!this.sessionNavigationNeedsRestore || !this.frame.currentId) return
    try {
      await this.initialize({ resume_session_id: this.frame.currentId })
      this.sessionNavigationNeedsRestore = false
    } catch (error) {
      // Rejecting here would break the navigation chain every later click
      // waits on; the failure is shown instead, as the catch path does.
      this.fail(error)
    }
  }

  /** Resolves true only when this store's own session actually changed. */
  private async openSessionNow(id: string, version: number, known?: SessionRow): Promise<boolean> {
    try {
      const row = [...this.frame.sessions, ...this.frame.live].find(session => session.id === id) ?? known
      if (this.frame.turnActive) {
        // Clicking the running task is still an explicit open: the view returns
        // to its chat (App resets the page on this revision) without re-binding.
        if (id === this.frame.currentId) { this.patch({ sessionOpenRevision: this.frame.sessionOpenRevision + 1 }); return false }
        if (!this.bridge.openWorkspaceWindow) throw new Error('This desktop build cannot open another session window. Relaunch the updated app.')
        await this.bridge.openWorkspaceWindow(row?.cwd || this.frame.cwd, id)
        return false
      }
      if (row?.cwd && row.cwd !== this.frame.cwd) {
        if (!this.bridge.useWorkspace) throw new Error('This host cannot switch workspaces. Open the session from its project folder.')
        await this.bridge.useWorkspace(row.cwd, id)
        return false
      }
      // A resume re-keys the connection, but NOT to a string of our own
      // choosing: the daemon binds a resumed session under the session id
      // (resume_session_id wins over session_key — see its 'ignored-slot'
      // contract test), and every later submit/steer/cancel sends the key
      // explicitly. Adopt the key the daemon actually bound, or the next
      // message silently lands in a fresh, context-free session.
      // Deliberate navigation, not a dropped connection: reusing
      // 'connecting' painted "Reconnecting…" over the previous task's
      // transcript, with a Retry button that retryConnection() ignores.
      this.patch({ connection: 'connecting', navigating: 'session' })
      const result = await this.bridge.call('initialize', {
        history_limit: 100,
        session_key: `${this.sessionKey}-r${id.slice(-8)}`,
        resume_session_id: id,
        ...clientHandshake(),
      })
      if (version !== this.sessionNavigationVersion) {
        if (result.ok !== false) this.sessionNavigationNeedsRestore = true
        return true
      }
      if (result.ok === false) throw new Error(str(result.error) || 'Session initialization was rejected')
      this.sessionNavigationNeedsRestore = false
      this.applyInitializedSession(result, { resume_session_id: id }, true)
      this.patch({ sessionOpenRevision: this.frame.sessionOpenRevision + 1 })
      void this.refreshGoal()
      void this.refreshSessions()
      return true
    } catch (error) {
      if (version !== this.sessionNavigationVersion) return true
      // The chosen chat is still running its turn: watch it rather than fail.
      if (RUNNING_ELSEWHERE.test(error instanceof Error ? error.message : String(error))) {
        // A view of this window already runs it (it was handed off there):
        // go to that view. Pulling it in here made this view the busy one, so
        // the next click handed off again and an SSH window rebuilt a view
        // (new connection, full reload) every time.
        try {
          const row = [...this.frame.sessions, ...this.frame.live].find(session => session.id === id) ?? known
          if (this.frame.storageScope?.startsWith('ssh:') && await this.bridge.openWorkspaceWindow?.(row?.cwd || this.frame.cwd, id, { existingOnly: true })) {
            if (this.sessionNavigationNeedsRestore && this.frame.currentId) {
              await this.initialize({ resume_session_id: this.frame.currentId })
              this.sessionNavigationNeedsRestore = false
            }
            return false
          }
        } catch { /* attach in place below */ }
        try {
          const attached = await this.attachRunningSession(id)
          if (attached && version === this.sessionNavigationVersion) {
            this.sessionNavigationNeedsRestore = false
            this.patch({ sessionOpenRevision: this.frame.sessionOpenRevision + 1 })
            void this.refreshGoal()
            void this.refreshSessions()
            return true
          }
        } catch { /* fall through to the ordinary failure below */ }
      }
      if (this.sessionNavigationNeedsRestore && this.frame.currentId) {
        try {
          await this.initialize({ resume_session_id: this.frame.currentId })
          this.sessionNavigationNeedsRestore = false
        } catch (restoreError) {
          this.fail(restoreError)
          return true
        }
      }
      if (this.frame.turnActive) this.patch({ error: error instanceof Error ? error.message : String(error) })
      else this.fail(error)
      // A failed open still left the connection re-keyed or the UI in an error
      // state; treat it as a move rather than resurrecting stale captures.
      return true
    }
  }

  /**
   * Attach to a session whose turn is running on the daemon: the daemon keeps
   * it under its own key, and an initialize with that key is an idempotent
   * attach (the TUI's Agent View does the same). Null when it is not live.
   */
  private async attachRunningSession(id: string): Promise<Record<string, unknown> | null> {
    const listed = await this.bridge.call('session.active_list', { history_limit: 0 })
    const rows = Array.isArray(listed.sessions) ? listed.sessions as Record<string, unknown>[] : []
    const key = str(rows.find(row => str(row.id) === id)?.key)
    if (!key) return null
    // A chat bound under its own id is attached by id: a bare initialize on
    // that key makes the daemon replace it with an empty session.
    return this.initialize(key === id ? { resume_session_id: id } : { session_key: key })
  }

  /**
   * `/resume <id|name>`: the same id-prefix, key or title match the daemon
   * applies, over every saved chat. The sidebar lists hold only the newest
   * page and leave out the current chat, so matching them alone reported "no
   * match" for an older chat the daemon's own /resume used to open.
   */
  private async resumeMatching(query: string): Promise<boolean> {
    const needle = query.trim().toLowerCase()
    const currentId = this.frame.currentId
    let saved: SessionRow[]
    try {
      const listed = await this.bridge.call('session.list', { kind: 'main', scope: 'global', query: needle })
      if (listed.ok === false) throw new Error(str(listed.error) || 'Saved sessions could not be listed')
      saved = normalize(listed.sessions, currentId)
    } catch (error) {
      this.fail(error)
      return false
    }
    // Matching here too keeps an older daemon, which ignores `query` and
    // returns every saved chat, from resuming an unrelated row.
    const matches = new Map<string, SessionRow | undefined>()
    for (const row of [...this.frame.live, ...saved]) {
      if (row.id.toLowerCase().startsWith(needle) || row.key.toLowerCase() === needle || row.title.toLowerCase() === needle) matches.set(row.id, row)
    }
    // A chat that has not been saved yet is still the one on screen.
    if (currentId && currentId.toLowerCase().startsWith(needle) && !matches.has(currentId)) matches.set(currentId, undefined)
    if (matches.size !== 1) {
      // Reported directly, not through fail(): the message quotes what the
      // user typed, and fail() reads words like "connect" or "closed" as a
      // transport failure and takes the whole view offline.
      const message = matches.size
        ? `Multiple sessions match \`${query}\`; use a longer id prefix.`
        : `No saved session matches \`${query}\`.`
      this.patch({ error: message })
      this.builder.push('notification', { severity: 'error', message })
      this.notify()
      return false
    }
    const [id, row] = [...matches][0]!
    if (id === currentId) {
      this.builder.push('notification', { severity: 'info', message: 'Already in this session.' })
      this.notify()
      return true
    }
    await this.openSession(id, row)
    return true
  }

  /** A new task in `cwd`: this view for the current folder, a fresh view for another. */
  newChatIn(cwd: string): void {
    if (cwd) setThreadInboxState(current => unhideWorkspaceInState(current, cwd))
    if (!cwd || cwd === this.frame.cwd) { this.newChat(); return }
    if (!this.bridge.openWorkspaceWindow) { this.enterWorkspace(cwd); return }
    void this.bridge.openWorkspaceWindow(cwd, undefined, { fresh: true }).catch(error => this.fail(error))
  }

  newChat(): void {
    if (this.frame.connection !== 'online') { this.patch({ workspaceError: 'Not connected to this workspace yet.' }); return }
    // A running turn keeps this view's connection, so the new task opens as
    // another view in the same window — the way opening another chat mid-turn
    // does. It used to refuse ("Finish or stop the current task…") and the
    // button sat greyed out for the whole turn.
    if (this.frame.turnActive) {
      if (!this.bridge.openWorkspaceWindow) { this.patch({ workspaceError: 'This desktop build cannot open another window. Relaunch the updated app.' }); return }
      void this.bridge.openWorkspaceWindow(this.frame.cwd || undefined, undefined, { fresh: true }).catch(error => this.fail(error))
      return
    }
    void this.beginFreshTask()
  }

  /** Start an unmistakable blank Creator mode session from the shell chrome. */
  startCreatorMode(): void {
    if (this.frame.connection !== 'online') { this.patch({ workspaceError: 'Not connected to this workspace yet.' }); return }
    // Like New task: a running turn keeps this view, so Creator mode opens as
    // another view of this workspace. The button used to sit disabled mid-turn.
    if (this.frame.turnActive) {
      if (!this.bridge.openWorkspaceWindow) { this.patch({ workspaceError: 'This desktop build cannot open another window. Relaunch the updated app.' }); return }
      void this.bridge.openWorkspaceWindow(this.frame.cwd || undefined, undefined, { fresh: true, agentPreset: 'creator' }).catch(error => this.fail(error))
      return
    }
    void this.beginFreshTask('creator')
  }

  /**
   * Re-key and rebind. The daemon keys sessions by this string; reusing it
   * would reopen the previous conversation under an empty transcript. A
   * fresh task is a fresh key. Resolves true once the daemon has opened the
   * fresh session, so follow-up calls (plan ceiling, first submit) address
   * a session that exists; false means the rebind failed and the previous
   * conversation remains selected with the error visible.
   */
  private beginFreshTask(agentPreset?: string): Promise<boolean> {
    if (this.openingSession || this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)
    const sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
    this.openingFreshTask = true
    this.patch({ connection: 'connecting', navigating: 'fresh' })
    return this.initialize({ session_key: sessionKey, ...(agentPreset ? { agent_id: agentPreset } : {}) })
      .then(() => {
        void this.refreshGoal()
        void this.refreshSessions()
        return true
      })
      .catch(error => {
        this.patch({ connection: 'online' })
        this.fail(error)
        return false
      })
      .finally(() => { this.openingFreshTask = false })
  }

  private planTimer: ReturnType<typeof setTimeout> | undefined

  approvePlan(requestId: string, mode: 'accept-all' | 'manual'): void {
    const pending = this.frame.approval
    if (pending?.id !== requestId || !/^ExitPlanMode(Tool)?$/.test(pending.toolName || pending.action)) return
    this.respondApproval(requestId, 'allow_once', mode)
  }

  approve(requestId: string, response: ApprovalResponse): void {
    this.respondApproval(requestId, response)
  }

  approveWithOption(
    requestId: string,
    optionId: string,
    decision?: 'approve' | 'approve_for_session' | 'reject',
  ): void {
    const cleanOptionId = optionId.trim()
    if (!cleanOptionId) return
    const pending = this.frame.approval
    if (pending?.id !== requestId) return
    const matched = pending.options?.find(opt => opt.optionId === cleanOptionId)
    const wire: 'approve' | 'approve_for_session' | 'reject' =
      decision
      ?? (matched?.kind === 'allow_always'
        ? 'approve_for_session'
        : matched?.kind === 'reject_once' || matched?.kind === 'reject_always'
          ? 'reject'
          : 'approve')
    this.sendPermissionResponse(requestId, wire, cleanOptionId)
  }

  private respondApproval(requestId: string, response: ApprovalResponse, planMode?: 'accept-all' | 'manual'): void {
    const pending = this.frame.approval
    if (pending?.id !== requestId) return
    if (response === 'allow_session' && pending.allowPermanent === false) return
    // The daemon's vocabulary is approve / approve_for_session / reject —
    // anything else resolves as reject, so map our UI labels here.
    const wire = response === 'allow_once' ? 'approve'
      : response === 'allow_session' ? 'approve_for_session'
      : 'reject'
    const matchedOption = pending.options?.find(opt =>
      response === 'allow_once'
        ? opt.kind === 'allow_once'
        : response === 'allow_session'
          ? opt.kind === 'allow_always'
          : opt.kind === 'reject_once' || opt.kind === 'reject_always',
    )
    this.sendPermissionResponse(requestId, wire, matchedOption?.optionId, planMode)
  }

  private sendPermissionResponse(
    requestId: string,
    wire: 'approve' | 'approve_for_session' | 'reject',
    optionId?: string,
    planMode?: 'accept-all' | 'manual',
  ): void {
    const pending = this.frame.approval
    if (pending?.id !== requestId) return
    if (pending.autoAcceptAt !== undefined) {
      const { autoAcceptAt: _deadline, ...manualOnly } = pending
      this.patch({ approval: manualOnly })
    }
    const sessionKey = this.sessionKey
    const sessionId = this.frame.currentId
    const isCurrent = () => this.sessionKey === sessionKey && this.frame.currentId === sessionId
      && this.frame.approval?.id === requestId
    void this.bridge
      .call('permission_response', {
        request_id: requestId,
        response: wire,
        ...(optionId ? { option_id: optionId } : {}),
      })
      .then(result => {
        // approval_response can remove the card before the RPC reply. Only
        // the daemon-confirmed winner may change this session's policy.
        if (planMode && result.ok === true && this.sessionKey === sessionKey && this.frame.currentId === sessionId) {
          this.setPermissionMode(planMode)
        }
        if (!isCurrent()) return
        // Clear only on confirmation: a refused response (unknown id, another
        // connection owns it) leaves the request pending daemon-side, and
        // dropping the card would strand it with no surface to answer.
        if (result.ok === false) {
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'approval refused' })
          this.notify()
        } else {
          this.patch({ approval: null })
        }
      })
      .catch(error => {
        if (!isCurrent()) return
        this.fail(error)
      })
  }

  answerQuestion(requestId: string, answers: Record<string, string>): void {
    const sessionKey = this.sessionKey
    const sessionId = this.frame.currentId
    const isCurrent = () => this.sessionKey === sessionKey && this.frame.currentId === sessionId
      && this.frame.question?.requestId === requestId
    void this.bridge
      .call('question_response', { request_id: requestId, answers })
      .then(result => {
        if (!isCurrent()) return
        if (result.ok === false) {
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'answer refused' })
          this.notify()
        } else {
          this.patch({ question: null })
        }
      })
      .catch(error => {
        if (!isCurrent()) return
        this.fail(error)
      })
  }

  /**
   * Discovery is slow (the daemon probes the active profile, up to its own
   * timeout) and can fail. This used to patch no loading or error state and
   * had no in-flight guard, so the picker asserted "no models discovered
   * yet" for the whole window and after every failure, while its Retry
   * button could race a second probe on every click. The failure also went
   * only to a transcript notification — rendered behind the popover that
   * triggered it, where the person waiting cannot see it.
   */
  loadModels(force = false): void {
    if (!force && this.frame.models.length) return
    if (this.frame.modelsLoading) return
    this.patch({ modelsLoading: true, modelsError: null })
    void this.bridge
      .call('fetch_models', {for_model_selection:true})
      .then(result => {
        if (result.ok === false) {
          // 'no profile', 'provider refused' and 'offline' are not the same
          // state as 'zero models' — say which one happened.
          const message = str(result.error) || 'model discovery failed'
          this.patch({ modelsLoading: false, modelsError: message })
          this.builder.push('notification', { severity: 'error', message })
          this.notify()
          return
        }
        const ids = Array.isArray(result.models) ? (result.models as unknown[]).map(m => str(m)).filter(Boolean) : []
        this.patch({ models: toChoices(ids, result.catalog, str(result.profile_label)), modelsLoading: false, modelsError: null })
      })
      .catch(error => {
        this.patch({ modelsLoading: false, modelsError: desktopError(error) })
        this.fail(error)
      })
  }

  /** Discover one saved profile's catalog without activating that profile. */
  loadProviderModels(profileName: string, force = false): void {
    const name = profileName.trim()
    if (!name || this.frame.providerModelLoading.includes(name)) return
    if (!force && this.frame.providerModels[name]) return
    this.patch({ providerModelLoading: [...this.frame.providerModelLoading, name] })
    void this.bridge.call('provider_models', { profile_name: name }).then(result => {
      const loading = this.frame.providerModelLoading.filter(entry => entry !== name)
      if (result.ok === false) {
        this.patch({
          providerModelLoading: loading,
          providerModelWarnings: {
            ...this.frame.providerModelWarnings,
            [name]: str(result.error) || 'model discovery failed',
          },
        })
        return
      }
      const models = cachedModelsFromResult(result)
      const warnings = { ...this.frame.providerModelWarnings }
      const warning = str(result.warning)
      if (warning) warnings[name] = warning
      else delete warnings[name]
      this.patch({
        providerModelLoading: loading,
        providerModels: { ...this.frame.providerModels, [name]: models },
        providerModelWarnings: warnings,
      })
    }).catch(error => {
      const message = error instanceof Error ? error.message : String(error)
      this.patch({
        providerModelLoading: this.frame.providerModelLoading.filter(entry => entry !== name),
        providerModelWarnings: { ...this.frame.providerModelWarnings, [name]: message },
      })
    })
  }

  /** Persist or clear one cached model's input/output capacity overrides. */
  saveModelCapabilities(
    profileName: string,
    model: string,
    contextLimit: number | null,
    maxOutputTokens: number | null,
  ): void {
    const name = profileName.trim()
    const id = model.trim()
    if (!name || !id) return
    void this.bridge.call('provider_model_override', {
      profile_name: name,
      model: id,
      context_limit: contextLimit,
      max_output_tokens: maxOutputTokens,
    }).then(result => {
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'model capacity update refused',
        })
        this.notify()
        return
      }
      this.loadProviderModels(name, true)
      if (this.frame.providers.some(provider => provider.name === name && provider.active)) {
        this.loadModels(true)
      }
      this.builder.push('notification', {
        severity: 'info',
        message: `model capacities updated for \`${id}\``,
      })
      this.notify()
    }).catch(error => this.fail(error))
  }

  /**
   * Upsert a provider profile. The daemon's provider_save persists it to
   * ~/.xerxes/profiles.json AND makes it the active profile (its runtime
   * reloads onto the new credentials). Apply its model to the original
   * session without reopening it. Refused mid-turn like provider switching.
   */
  async saveProvider(profile: {
    name: string
    baseUrl: string
    model: string
    provider?: string
    apiKey?: string
  }): Promise<string | null> {
    const name = profile.name.trim()
    const baseUrl = profile.baseUrl.trim()
    const model = profile.model.trim()
    const provider = profile.provider?.trim()
    // A registry-known type supplies its default endpoint daemon-side;
    // "Provider default" (blank) is only valid then.
    const knownDefault = this.frame.providerTypes.find(t => t.name === provider)?.baseUrl ?? ''
    // The model is optional: left blank, the runtime asks the provider which
    // models this key can use and starts on the first.
    if (!name || (!baseUrl && !knownDefault)) {
      this.builder.push('notification', {
        severity: 'error',
        message: knownDefault ? 'name is required' : 'name and base_url are required',
      })
      this.notify()
      return knownDefault ? 'Name is required.' : 'Name and base URL are required.'
    }
    if (this.frame.turnActive) return 'Wait for the current turn to finish before changing providers.'
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    const params: Record<string, unknown> = { name, ...(model ? { model } : {}) }
    if (baseUrl) params.base_url = baseUrl
    if (provider) params.provider = provider
    if (profile.apiKey?.trim()) params.api_key = profile.apiKey.trim()
    return this.bridge
      .call('provider_save', params)
      .then(async result => {
        if (result.ok === false) {
          if (isCurrent()) this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'provider save refused',
          })
          this.notify()
          return str(result.error) || 'Provider save refused.'
        }
        const saved = (result.profile && typeof result.profile === 'object'
          ? result.profile
          : {}) as Record<string, unknown>
        // The save is durable even if applying its model fails. Refresh the
        // profile list and report the two outcomes separately in the editor.
        void this.loadProviders()
        this.loadModels(true)
        try {
          const applied = await this.bridge.call('set_model', {
            session_key: sessionKey, model: str(saved.model) || model, provider_profile: name,
          })
          if (applied.ok === false) throw new Error(str(applied.error) || 'Model change was refused')
          if (isCurrent()) {
            this.patch({ model: str(applied.model) || str(saved.model) || model })
            this.builder.push('notification', { severity: 'info', message: `provider \`${name}\` saved and activated` })
            this.notify()
          }
        } catch (error) {
          return `Provider saved, but its model could not be applied to the original chat: ${desktopError(error)}`
        }
        return null
      })
      .catch(error => { if (isCurrent()) this.fail(error); return desktopError(error) })
  }

  /** Delete a saved profile. The active one must be switched away from first. */
  deleteProvider(name: string): void {
    const row = this.frame.providers.find(entry => entry.name === name)
    if (!row || row.active || this.frame.turnActive) return
    void this.bridge
      .call('provider_delete', { name })
      .then(result => {
        if (result.ok === false) {
          this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'provider delete refused',
          })
          this.notify()
          return
        }
        this.builder.push('notification', {
          severity: 'info',
          message: `provider \`${name}\` deleted`,
        })
        this.notify()
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => this.fail(error))
  }

  /** The daemon's own slash catalog — every TUI command, discoverable here. */
  loadCommands(): void {
    if (this.frame.commands.length) return
    void this.bridge
      .call('commands.catalog', {})
      .then(result => {
        const pairs = Array.isArray(result.pairs) ? result.pairs : []
        const commands = pairs
          .map(pair => {
            const entry = (Array.isArray(pair) ? pair : []) as unknown[]
            const name = str(entry[0]).replace(/^\//, '')
            return name ? { name, description: str(entry[1]) } : null
          })
          .filter((entry): entry is { name: string; description: string } => entry !== null)
        this.patch({ commands })
      })
      .catch(() => {})
  }

  /**
   * Live runtime-scoped completions for a partial draft:
   * - On Xerxes Native: daemon slash commands + `/skill <name>` references.
   * - On an external ACP runtime: STRICTLY scoped to the connected ACP runtime's
   *   automatically discovered commands & skills plus `/acp` (and minimal
   *   non-colliding host commands when explicitly typed), excluding all Xerxes
   *   native commands and bundled skills (`immortals-*`, etc.).
   */
  async completeText(text: string): Promise<CompletionHintItem[]> {
    const binding = this.frame.acpBinding
    const isExternal = Boolean(binding?.connected && binding.transport !== 'native')
    const isSlashPrefix = /^\/[^\s]*$/.test(text)

    if (isExternal && binding) {
      const { commands, skills } = resolveExternalAcpCatalog(binding)
      const acpNames = new Set<string>([
        ...commands.map(cmd => cmd.name.toLowerCase()),
        ...skills.map(skill => skill.name.toLowerCase()),
        ...skills.map(skill => (skill.slash_command ?? skill.name).replace(/^\/+/, '').toLowerCase()),
      ])

      const skillPrefixMatch = /^\/skill\s+(.*)$/i.exec(text)
      if (skillPrefixMatch) {
        const needle = skillPrefixMatch[1]!.trim().toLowerCase()
        return skills
          .filter(skill =>
            !needle
            || skill.name.toLowerCase().includes(needle)
            || (skill.description?.toLowerCase().includes(needle) ?? false)
            || (skill.source?.toLowerCase().includes(needle) ?? false),
          )
          .map(skill => {
            const slash = skill.slash_command?.startsWith('/') ? skill.slash_command : `/${skill.name}`
            return {
              value: slash,
              label: slash,
              meta: skill.description || `${binding.name} skill`,
              kind: 'skill' as const,
              badge: skill.source || binding.name,
              runtimeProvider: binding.provider,
            }
          })
      }

      if (/^\/acp\s/i.test(text)) {
        const res = await this.bridge.call('complete', { text })
        const items = Array.isArray(res.completions) ? res.completions : []
        return items
          .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
          .map(item => {
            const value = str(item.value)
            return {
              value,
              label: str(item.label) || value,
              meta: str(item.meta),
              kind: 'command' as const,
              badge: 'Host',
            }
          })
          .filter(item => Boolean(item.value))
      }

      if (!isSlashPrefix) return []
      const needle = text.slice(1).toLowerCase()
      const matches = (name: string, desc?: string, source?: string): boolean =>
        !needle
        || name.toLowerCase().includes(needle)
        || (desc?.toLowerCase().includes(needle) ?? false)
        || (source?.toLowerCase().includes(needle) ?? false)

      const acpCommands: CompletionHintItem[] = commands
        .filter(cmd => matches(cmd.name, cmd.description, cmd.source))
        .map(cmd => ({
          value: `/${cmd.name}`,
          label: `/${cmd.name}`,
          meta: cmd.description || `${binding.name} command`,
          kind: 'command' as const,
          badge: cmd.source || binding.name,
          runtimeProvider: binding.provider,
        }))

      const acpSkills: CompletionHintItem[] = skills
        .filter(skill => matches(skill.name, skill.description, skill.source))
        .map(skill => {
          const slash = skill.slash_command?.startsWith('/') ? skill.slash_command : `/${skill.name}`
          return {
            value: slash,
            label: slash,
            meta: skill.description || `${binding.name} skill`,
            kind: 'skill' as const,
            badge: skill.source || 'Skill',
            runtimeProvider: binding.provider,
          }
        })

      const hostItems: CompletionHintItem[] = []
      if (matches('acp', 'Switch or manage ACP runtime', 'host')) {
        hostItems.push({
          value: '/acp',
          label: '/acp',
          meta: 'Switch or manage ACP runtime',
          kind: 'command',
          badge: 'Host',
        })
      }
      if (needle.length > 0) {
        const minimalHostCommands: readonly { readonly name: string; readonly meta: string }[] = [
          { name: 'new', meta: 'Start a new conversation' },
          { name: 'clear', meta: 'Clear the current conversation' },
          { name: 'export', meta: 'Export conversation transcript' },
        ]
        for (const hostCmd of minimalHostCommands) {
          if (acpNames.has(hostCmd.name)) continue
          if (!matches(hostCmd.name, hostCmd.meta, 'host')) continue
          hostItems.push({
            value: `/${hostCmd.name}`,
            label: `/${hostCmd.name}`,
            meta: hostCmd.meta,
            kind: 'command',
            badge: 'Host',
          })
        }
      }

      const ordered = [...acpCommands, ...acpSkills, ...hostItems]
      const seen = new Set<string>()
      return ordered.filter(entry => {
        const key = `${entry.kind}:${entry.value.trim().toLowerCase()}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
    }

    const queries = isSlashPrefix && text !== '/skill'
      ? [text, '/skill ' + text.slice(1)] : [text]
    const results = await Promise.all(queries.map(query => this.bridge.call('complete', { text: query })))
    const completions = results.flatMap(result => Array.isArray(result.completions) ? result.completions : [])
    const mapped: CompletionHintItem[] = completions
      .map(raw => {
        const entry = (raw ?? {}) as Record<string, unknown>
        const value = str(entry.value)
        return {
          value,
          label: str(entry.label) || value,
          meta: str(entry.meta),
          kind: (entry.category !== undefined ? 'command' : 'skill') as 'command' | 'skill',
        }
      })
      .filter(entry => Boolean(entry.value))

    const nativeCommands = mapped.filter(entry => entry.kind === 'command')
    const skills = mapped.filter(entry => entry.kind === 'skill')
    const ordered: CompletionHintItem[] = [...nativeCommands, ...skills]
    const seen = new Set<string>()
    return ordered.filter(entry => {
      const key =
        entry.kind === 'skill'
          ? 'skill:' + entry.label.replace(/^\/+/, '').trim().toLowerCase()
          : 'cmd:' + entry.value.trim().toLowerCase()
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }

  /**
   * Ask the shell for a workspace folder. On a pick the shell retargets its
   * daemon and reloads this window — the renderer's job ends at the request.
   */
  chooseWorkspace(): Promise<void> {
    return this.requestWorkspace(async () => {
      if (!this.bridge.chooseWorkspace) throw new Error('This host cannot open a workspace folder.')
      const picked = await this.bridge.chooseWorkspace()
      if (typeof picked === 'string' && picked) {
        setThreadInboxState(current => unhideWorkspaceInState(current, picked))
      }
    })
  }

  openWorkspaceWindow(directory?: string): Promise<void> {
    return this.requestWorkspace(async () => {
      if (!this.bridge.openWorkspaceWindow) throw new Error('This desktop build cannot open additional windows. Relaunch the updated app.')
      if (directory) setThreadInboxState(current => unhideWorkspaceInState(current, directory))
      await this.bridge.openWorkspaceWindow(directory)
    })
  }

  /** Enter a known workspace folder (sidebar header, switcher menu) — retargets the daemon. */
  enterWorkspace(cwd: string): Promise<void> {
    if (!cwd) return Promise.resolve()
    setThreadInboxState(current => unhideWorkspaceInState(current, cwd))
    return this.requestWorkspace(async () => {
      if (!this.bridge.useWorkspace) throw new Error('This host cannot switch workspace folders.')
      await this.bridge.useWorkspace(cwd)
    })
  }

  /**
   * Remove a local project folder from the sidebar and remembered workspace list.
   * Never deletes files on disk; switches to another remaining workspace if the
   * removed folder is currently open.
   */
  removeWorkspace(cwd: string): Promise<void> {
    const clean = normalizeWorkspacePath(cwd)
    if (!clean) return Promise.resolve()
    setThreadInboxState(current => hideWorkspaceInState(current, clean))
    return this.requestWorkspace(async () => {
      const remaining = this.bridge.removeWorkspace
        ? await this.bridge.removeWorkspace(clean)
        : (this.frame.workspaceDirectories ?? []).filter(dir => normalizeWorkspacePath(dir) !== clean)
      this.patch({ workspaceDirectories: remaining })
      if (normalizeWorkspacePath(this.frame.cwd) === clean) {
        const fallback = remaining.find(dir => normalizeWorkspacePath(dir) !== clean)
        if (fallback && this.bridge.useWorkspace) {
          setThreadInboxState(current => unhideWorkspaceInState(current, fallback))
          await this.bridge.useWorkspace(fallback)
        }
      }
    })
  }

  /** Restore a workspace folder to the sidebar list (Undo of removeWorkspace). */
  restoreWorkspace(cwd: string): Promise<void> {
    const clean = normalizeWorkspacePath(cwd)
    if (!clean) return Promise.resolve()
    setThreadInboxState(current => unhideWorkspaceInState(current, clean))
    return this.requestWorkspace(async () => {
      const directories = this.bridge.restoreWorkspace
        ? await this.bridge.restoreWorkspace(clean)
        : [...new Set([...(this.frame.workspaceDirectories ?? []), clean])]
      this.patch({ workspaceDirectories: directories })
    })
  }

  /** Remove a saved SSH host from the sidebar without touching the remote machine. */
  removeRemoteHost(alias: string): Promise<void> {
    const clean = alias.trim()
    if (!clean) return Promise.resolve()
    return this.requestWorkspace(async () => {
      if (!this.bridge.remote) throw new Error('This host cannot remove SSH workspaces.')
      await this.bridge.remote('remove', { alias: clean })
      await this.refreshContexts()
    })
  }

  clearWorkspaceError(): void {
    this.patch({ workspaceError: null })
  }

  private async requestWorkspace(action: () => Promise<void>): Promise<void> {
    if (this.frame.workspaceBusy) return
    this.patch({ wsMenuOpen: false, workspaceBusy: true, workspaceError: null })
    try {
      await action()
    } catch (error) {
      this.patch({ workspaceError: desktopError(error) })
    } finally {
      this.patch({ workspaceBusy: false })
    }
  }

  /** A late session-scoped reply must never update a newly selected chat. */
  private captureSessionRequest(): { sessionKey: string; isCurrent: () => boolean } {
    const sessionKey = this.sessionKey, sessionId = this.frame.currentId, navigation = this.sessionNavigationVersion
    return { sessionKey, isCurrent: () => this.sessionKey === sessionKey && this.frame.currentId === sessionId && this.sessionNavigationVersion === navigation }
  }

  pickModel(modelId: string): void {
    if (!modelId) return
    // Same mid-turn refusal as provider switching: hot-swapping the model a
    // running turn is riding on is refused daemon-side too.
    if (this.frame.turnActive) return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    // Route through the daemon's /model handler: it pins the choice to this
    // session AND persists it as the active profile's model. The old
    // initialize({ model }) path only reloaded runtime memory, so every
    // daemon restart silently fell back to the profile's stored model.
    void this.bridge.call('slash', { command: `/model ${modelId}`, session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'model change rejected',
        })
        this.notify()
        return
      }
      const applied = str(result.model)
      if (applied) this.patch({ model: applied })
      this.loadReasoningLevels({ silent: true })
    }).catch(error => { if (isCurrent()) this.fail(error) })
  }

  /**
   * Switch the daemon's active provider profile and adopt its model here.
   * Refused mid-turn: provider_select swaps the live credentials the
   * in-flight request is already riding on.
   */
  selectProvider(name: string): void {
    if (this.frame.turnActive || this.frame.providerSwitching) return
    const target = this.frame.providers.find(row => row.name === name)
    if (!target || target.active) return
    // On an SSH host a sign-in provider runs on this computer, through its
    // own login; the host's copy of that profile has no login to use.
    if (runsOnThisComputer(this.frame, target)) { this.useLocalProvider(name); return }
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ providerSwitching: name, providerSwitchError: null })
    void this.bridge
      .call('provider_select', { name })
      .then(result => {
        if (result.ok === false) {
          this.patch({ providerSwitchError: str(result.error) || 'Provider switch was refused. Select a provider to retry.' })
          return
        }
        // provider_select applies the model to its session and emits status.
        // Reinitializing here would discard loaded history and could bind a
        // different chat if navigation happened while the request was pending.
        if (isCurrent() && target.model) this.patch({ model: target.model })
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => { this.patch({ providerSwitchError: desktopError(error) }) })
      .finally(() => { this.patch({ providerSwitching: null }) })
  }

  /** Move this SSH task from this computer to the host's own working sign-in. */
  useHostProvider(name: string): void {
    if (this.frame.turnActive || this.frame.providerSwitching || !window.xerxes.remote) return
    this.patch({ providerSwitching: name, providerSwitchError: null })
    void window.xerxes.remote('provider-use-host', { profile: name })
      .then(result => {
        const row = result && typeof result === 'object' ? result as Record<string, unknown> : {}
        if (row.ok === false) { this.patch({ providerSwitchError: str(row.error) || 'Could not run that provider on the host.' }); return }
        this.refreshProviderRelay()
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => { this.patch({ providerSwitchError: desktopError(error) }) })
      .finally(() => { this.patch({ providerSwitching: null }) })
  }

  /** Put this SSH task on one of this computer's sign-in providers. */
  useLocalProvider(name: string): void {
    if (this.frame.turnActive || this.frame.providerSwitching || !window.xerxes.remote) return
    this.patch({ providerSwitching: name, providerSwitchError: null })
    void window.xerxes.remote('provider-use-local', { profile: name })
      .then(result => {
        const row = result && typeof result === 'object' ? result as Record<string, unknown> : {}
        if (row.ok === false) { this.patch({ providerSwitchError: str(row.error) || 'Could not use that provider on this computer.' }); return }
        this.refreshProviderRelay()
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => { this.patch({ providerSwitchError: desktopError(error) }) })
      .finally(() => { this.patch({ providerSwitching: null }) })
  }

  /** Toggle the session's plan mode on the daemon (plan = read-only ceiling). */
  setPlanMode(next: boolean): Promise<boolean> {
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    return this.bridge
      .call('set_plan_mode', { session_key: sessionKey, enabled: next })
      .then(result => {
        if (!isCurrent()) return result.ok !== false
        if (result.ok === false) {
          // Refused (e.g. the daemon restarted and lost the session): flip
          // nothing locally — the chip must not claim a ceiling the daemon
          // never armed.
          this.fail(new Error(str(result.error) || 'plan mode refused'))
          return false
        }
        // The daemon answers before the status echo; apply optimistically,
        // the next status_update.plan_mode is authoritative either way.
        this.patch({ planMode: next })
        return true
      })
      .catch(error => { if (isCurrent()) this.fail(error); return false })
  }

  private acpPlanModePair(): { readonly planId: string; readonly defaultId: string; readonly isPlan: boolean } | null {
    const binding = this.frame.acpBinding
    if (!binding?.connected || binding.transport === 'native' || !binding.modes?.length) return null
    const planMode =
      binding.modes.find(mode => mode.id.toLowerCase() === 'plan')
      ?? binding.modes.find(mode => mode.id.toLowerCase() === 'planning')
    if (!planMode) return null
    const defaultMode =
      binding.modes.find(mode => mode.id.toLowerCase() === 'default')
      ?? binding.modes.find(mode => mode.id !== planMode.id)
    if (!defaultMode) return null
    return {
      planId: planMode.id,
      defaultId: defaultMode.id,
      isPlan: (binding.current_mode ?? '').toLowerCase() === planMode.id.toLowerCase(),
    }
  }

  togglePlanMode(): void {
    const acpPlan = this.acpPlanModePair()
    if (acpPlan) {
      void this.setAcpMode(acpPlan.isPlan ? acpPlan.defaultId : acpPlan.planId)
      return
    }
    this.setPlanMode(!this.frame.planMode)
  }

  /** Relay the person's choice on the update prompt to the main process. */
  private relayPending = false
  private relayAgain = false
  /** Ask the desktop whether this SSH conversation runs through this computer; calls coalesce. */
  refreshProviderRelay(): void {
    if (!this.frame.storageScope?.startsWith('ssh:') || !window.xerxes.remote) {
      if (this.frame.providerRelay) this.patch({ providerRelay: null })
      return
    }
    if (this.relayPending) { this.relayAgain = true; return }
    this.relayPending = true
    void window.xerxes.remote('provider-activity', {})
      .then(value => this.patch({ providerRelay: providerRelayView(value) }))
      .catch(() => undefined)
      .finally(() => {
        this.relayPending = false
        if (this.relayAgain) { this.relayAgain = false; this.refreshProviderRelay() }
      })
  }

  async appUpdateAction(action: 'check' | 'install' | 'skip' | 'dismiss' | 'open-release'): Promise<void> {
    try {
      const state = await this.bridge.appUpdate?.(action)
      if (state !== undefined) this.patch({ appUpdate: appUpdateView(state) })
    } catch (error) {
      this.patch({ appUpdate: { phase: 'error', message: desktopError(error) } })
    }
  }

  toggleDelegationMenu(): void {
    this.patch({ delegationMenuOpen: !this.frame.delegationMenuOpen, modelMenuOpen: false, pickerOpen: false, reasoningPickerOpen: false })
  }

  closeDelegationMenu(): void {
    this.patch({ delegationMenuOpen: false })
  }

  /** Off, auto or eager for this conversation; the daemon's status echo is authoritative. */
  async setDelegationMode(mode: 'off' | 'auto' | 'eager'): Promise<void> {
    const previous = this.frame.delegationMode
    const sessionKey = this.sessionKey
    this.patch({ delegationMode: mode, delegationMenuOpen: false })
    try {
      const result = await this.bridge.call('slash', { command: `/delegate ${mode}` })
      if (result.ok !== true && this.sessionKey === sessionKey) {
        this.patch({ delegationMode: previous })
        this.fail(new Error(str(result.error) || 'The runtime did not accept the agents setting'))
      }
    } catch (error) {
      if (this.sessionKey === sessionKey) { this.patch({ delegationMode: previous }); this.fail(error) }
    }
  }

  // ── Overlay + tab actions ────────────────────────────────────────────

  setTab(tab: WorkspaceTab): void {
    this.patch({ tab })
  }

  /** Open Session edits on one file's diff (from the rail's Touched card). */
  openChange(path: string): void {
    this.patch({ tab: 'changes', changesFocus: { path, at: Date.now() } })
  }

  openSettings(tab?: SettingsTab): void {
    this.patch({ settingsOpen: true, paletteOpen: false, ...(tab ? { settingsTab: tab } : {}) })
    this.loadModels()
    void this.loadProviders()
    void this.loadRunMode()
    if (tab === 'models') void this.loadLocalSetup()
    void this.loadAgentPresets()
    if (tab === 'channels') this.loadChannels()
    if (tab === 'terminals') this.loadTerminals()
  }

  closeSettings(): void {
    this.patch({ settingsOpen: false })
  }

  setSettingsTab(tab: SettingsTab): void {
    this.patch({ settingsTab: tab })
    if (tab === 'agents') void this.loadAgentPresets()
    if (tab === 'models') {
      this.loadModels()
      void this.loadProviders()
      void this.loadRunMode()
      void this.loadLocalSetup()
    }
    if (tab === 'channels') this.loadChannels()
    if (tab === 'terminals') this.loadTerminals()
  }

  togglePalette(): void {
    this.patch({ paletteOpen: !this.frame.paletteOpen })
    this.loadModels()
    this.loadCommands()
  }

  closePalette(): void {
    this.patch({ paletteOpen: false })
  }

  openPicker(): void {
    this.patch({ pickerOpen: true, settingsOpen: false, paletteOpen: false, modelMenuOpen: false })
    this.loadModels()
  }

  closePicker(): void {
    this.patch({ pickerOpen: false })
  }

  /** Adopt a freshly observed Git branch for the active workspace without re-initializing. */
  syncBranch(branch: string | null | undefined): void {
    const clean = typeof branch === 'string' ? branch.trim() : ''
    if (clean === this.frame.branch) return
    this.patch({ branch: clean })
  }

  private branchRefreshPending = false
  private supportsGitBranch = true

  /**
   * Re-read the active workspace's live Git branch so terminal checkouts,
   * external editors, and agent commands never leave the sidebar or welcome
   * header stuck on an old branch. Uses lightweight `git.branch` when the
   * daemon supports it, falling back to `git.status` on an older live daemon.
   */
  async refreshBranch(): Promise<void> {
    if (this.branchRefreshPending || this.frame.connection !== 'online' || !this.frame.cwd) return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.branchRefreshPending = true
    try {
      if (this.supportsGitBranch) {
        try {
          const result = await this.bridge.call('git.branch', { session_key: sessionKey })
          if (!isCurrent()) return
          if (result.ok !== false && typeof result.branch === 'string') {
            this.syncBranch(result.branch)
            return
          }
          if (result.code === 'git-unknown-method' || /unknown git method/i.test(str(result.error))) {
            this.supportsGitBranch = false
          } else {
            return
          }
        } catch {
          this.supportsGitBranch = false
        }
      }
      const status = await this.bridge.call('git.status', { session_key: sessionKey })
      if (!isCurrent() || status.ok === false || !('repository' in status)) return
      const repo = status.repository
      if (repo && typeof repo === 'object' && !Array.isArray(repo)) {
        const record = repo as Record<string, unknown>
        this.syncBranch(typeof record.branch === 'string' ? record.branch : '')
      } else if (repo === null) {
        this.syncBranch('')
      }
    } catch {
      // Outside a git repository or offline daemon: leave branch untouched.
    } finally {
      this.branchRefreshPending = false
    }
  }

  /** Ask the daemon for the active session model's reasoning/thinking levels. */
  loadReasoningLevels(options: { readonly silent?: boolean } = {}): void {
    const silent = options.silent === true
    if (silent && this.frame.connection !== 'online') return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    if (!silent) this.patch({ reasoningLoading: true })
    void this.bridge.call('reasoning_levels', { session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
      if (result.ok === false) {
        if (!silent) {
          this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'reasoning levels unavailable',
          })
          this.patch({ reasoningLoading: false })
          this.notify()
        }
        return
      }
      const levels = Array.isArray(result.levels)
        ? (result.levels as unknown[]).flatMap(entry => {
            if (typeof entry !== 'object' || entry === null) return []
            const row = entry as Record<string, unknown>
            const effort = str(row.effort)
            return effort ? [{ effort, description: str(row.description) }] : []
          })
        : []
      const current = str(result.current)
      this.patch({
        ...(!silent ? { reasoningLoading: false } : {}),
        reasoningLevels: levels,
        reasoningDefault: str(result.default),
        reasoningNote: str(result.note),
        ...(current ? { reasoningEffort: current } : {}),
      })
    }).catch(error => {
      if (!isCurrent()) return
      if (!silent) {
        this.patch({ reasoningLoading: false })
        this.fail(error)
      }
    })
  }

  /** Reasoning levels are asked of the daemon per open — they differ per model. */
  openReasoningPicker(): void {
    this.patch({
      reasoningPickerOpen: true,
      settingsOpen: false,
      paletteOpen: false,
      pickerOpen: false,
      modelMenuOpen: false,
      wsMenuOpen: false,
    })
    this.loadReasoningLevels({ silent: false })
  }

  closeReasoningPicker(): void {
    this.patch({ reasoningPickerOpen: false })
  }

  toggleReasoningPicker(): void {
    if (this.frame.reasoningPickerOpen) {
      this.closeReasoningPicker()
    } else {
      this.openReasoningPicker()
    }
  }

  /** Selection rides the daemon's /thinking handler so session pinning applies. */
  pickReasoning(effort: string): void {
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({ reasoningPickerOpen: false })
    const trimmed = effort.trim()
    if (!trimmed) return
    void this.bridge.call('slash', { command: `/thinking ${trimmed}`, session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'reasoning effort rejected',
        })
        this.notify()
        return
      }
      const applied = str(result.reasoning_effort)
      if (applied) this.patch({ reasoningEffort: applied })
    }).catch(error => { if (isCurrent()) this.fail(error) })
  }

  togglePicker(): void {
    this.patch({ pickerOpen: !this.frame.pickerOpen, modelMenuOpen: false })
    this.loadModels()
  }

  /** The combined chip's dropdown; rows drill into the two pickers. */
  toggleModelMenu(): void {
    this.patch({
      modelMenuOpen: !this.frame.modelMenuOpen,
      pickerOpen: false,
      reasoningPickerOpen: false,
      paletteOpen: false,
      wsMenuOpen: false,
      contextMenuOpen: false,
    })
  }

  closeModelMenu(): void {
    this.patch({ modelMenuOpen: false })
  }

  /** Context popover: the daemon estimates the split on open, never cached. */
  toggleContextMenu(): void {
    if (this.frame.contextMenuOpen) {
      this.patch({ contextMenuOpen: false })
      return
    }
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({
      contextMenuOpen: true,
      modelMenuOpen: false,
      pickerOpen: false,
      reasoningPickerOpen: false,
      paletteOpen: false,
      wsMenuOpen: false,
      contextBreakdownLoading: true,
    })
    void this.bridge.call('context_breakdown', { session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
      if (result.ok === false) {
        this.patch({ contextBreakdownLoading: false, contextBreakdown: null })
        return
      }
      const breakdown: ContextBreakdown = {
        systemPromptTokens: Math.max(0, num(result.system_prompt_tokens) ?? 0),
        toolsTokens: Math.max(0, num(result.tools_tokens) ?? 0),
        messagesTokens: Math.max(0, num(result.messages_tokens) ?? 0),
        totalTokens: Math.max(0, num(result.total_tokens) ?? 0),
        contextLimit: Math.max(0, num(result.context_limit) ?? 0),
      }
      this.patch({ contextBreakdownLoading: false, contextBreakdown: breakdown })
    }).catch(() => {
      if (!isCurrent()) return
      this.patch({ contextBreakdownLoading: false, contextBreakdown: null })
    })
  }

  closeContextMenu(): void {
    this.patch({ contextMenuOpen: false })
  }

  toggleWorkspaceMenu(): void {
    this.patch({ wsMenuOpen: !this.frame.wsMenuOpen, paletteOpen: false, pickerOpen: false })
  }

  openWorkspaceMenu(): void {
    this.patch({ wsMenuOpen: true, paletteOpen: false, pickerOpen: false })
  }

  closeWorkspaceMenu(): void {
    this.patch({ wsMenuOpen: false })
  }

  /**
   * 'Stream thinking' — a display choice, not a policy change: the daemon
   * keeps streaming reasoning trails; the feed just stops showing them.
   */
  setStreamThinking(value: boolean): void {
    this.patch({ streamThinking: value })
    writeStreamThinking(value)
  }

  // ── New-task modal (mockup 18) ───────────────────────────────────────

  openTaskModal(): void {
    if (this.frame.turnActive) { this.patch({ workspaceError: 'Finish or stop the current task before starting a new one.' }); return }
    if (this.frame.connection !== 'online') { this.patch({ workspaceError: 'Not connected to this workspace yet.' }); return }
    this.patch({ taskModalOpen: true, paletteOpen: false, wsMenuOpen: false, sessionMenu: null })
    void this.loadAgentPresets()
    this.loadModels()
  }

  closeTaskModal(): void {
    this.patch({ taskModalOpen: false })
  }

  /**
   * Start the task the modal collected: fresh session, plan ceiling applied
   * to THAT session, then the objective submitted — in this order, so the
   * ceiling and the first message land on the session they belong to.
   */
  async startTask(objective: string, planFirst: boolean, agentPreset?: string, model?: string): Promise<void> {
    if (this.frame.turnActive || this.frame.connection !== 'online') return
    this.patch({ error: null })
    const bound = await this.beginFreshTask(agentPreset)
    if (!bound) return
    try {
      if (model) {
        const result = await this.bridge.call('slash', { session_key: this.sessionKey, command: `/model ${model}` })
        if (result.ok === false) throw new Error(str(result.error) || 'Model change rejected')
        this.patch({ model: str(result.model) || model })
      }
      if (planFirst !== this.frame.planMode) {
        const result = await this.bridge.call('set_plan_mode', { session_key: this.sessionKey, enabled: planFirst })
        if (result.ok === false) throw new Error(str(result.error) || 'Plan mode could not be enabled')
        this.patch({ planMode: planFirst })
      }
      this.patch({ taskModalOpen: false })
      const text = objective.trim()
      if (text) await this.submit(text)
    } catch (error) { this.fail(error) }
  }

  // ── Session context menu (mockup 08) ─────────────────────────────────

  openSessionMenu(row: { id: string; key: string; title: string }, x: number, y: number): void {
    // Keep the 216px menu on screen near the pointer. Viewport fallbacks
    // keep SSR/tests (no window) on a sane clamp.
    const vw = typeof window === 'object' && Number.isFinite(window.innerWidth) && window.innerWidth > 0 ? window.innerWidth : 4000
    const vh = typeof window === 'object' && Number.isFinite(window.innerHeight) && window.innerHeight > 0 ? window.innerHeight : 4000
    const clampedX = Math.min(Math.max(0, x), Math.max(0, vw - 232))
    const clampedY = Math.min(Math.max(0, y), Math.max(0, vh - 150))
    this.patch({ sessionMenu: { id: row.id, key: row.key, title: row.title, x: clampedX, y: clampedY }, wsMenuOpen: false })
  }

  closeSessionMenu(): void {
    this.patch({ sessionMenu: null })
  }

  /** Rename through the daemon's `session.title` — the wire owns the title. */
  async renameSession(key: string, title: string): Promise<void> {
    const clean = title.trim()
    const anchor = this.frame.sessionMenu
    if (anchor?.pending) return
    if (!clean) { this.closeSessionMenu(); return }
    const pending = anchor ? { ...anchor, title: clean, renaming: true, pending: true, error: '' } : null
    if (pending) this.patch({ sessionMenu: pending })
    try {
      await desktopCall(this.bridge, key, 'session.title', { title: clean })
      if (this.frame.sessionMenu === pending) this.closeSessionMenu()
      this.refreshSessions()
    } catch (error) {
      // Preserve the attempted title, but never resurrect a dismissed menu.
      if (pending && this.frame.sessionMenu === pending) {
        this.patch({ sessionMenu: { ...pending, pending: false, error: desktopError(error) } })
      } else if (!anchor) this.fail(new Error(`Rename failed: ${desktopError(error)}`))
    }
  }

  /**
   * Permanently delete a saved chat (the runtime refuses one whose turn is
   * running). Deleting the chat on screen opens a fresh task in its place.
   */
  async deleteSession(id: string): Promise<void> {
    const anchor = this.frame.sessionMenu
    if (anchor?.pending) return
    const pending = anchor ? { ...anchor, pending: true, error: '' } : null
    if (pending) this.patch({ sessionMenu: pending })
    try {
      const result = await this.bridge.call('session.delete', { session_id: id })
      if (result.ok === false) throw new Error(str(result.error) || 'The chat could not be deleted')
      if (!pending || this.frame.sessionMenu === pending) this.patch({ sessionMenu: null })
      if (id === this.frame.currentId && this.frame.connection === 'online') await this.beginFreshTask()
      this.refreshSessions()
    } catch (error) {
      // Keep the menu (and its error) only while it is still the one we opened.
      if (pending && this.frame.sessionMenu === pending) this.patch({ sessionMenu: { ...pending, pending: false, error: desktopError(error) } })
      else this.fail(new Error(`Delete failed: ${desktopError(error)}`))
    }
  }

  /**
   * Export a session transcript as a markdown download (mockup 08). The
   * record comes from the daemon's session.status; the fold is the same one
   * the replay renders, so the file shows what the transcript shows.
   */
  async exportSessionTranscript(key: string): Promise<void> {
    this.patch({ sessionMenu: null })
    try {
      await this.downloadSessionTranscript(key)
    } catch (error) {
      this.fail(new Error(`export failed: ${String(error)}`))
    }
  }

  /** Export for a surface that owns its own pending and error presentation. */
  async downloadSessionTranscript(key: string): Promise<void> {
    if (typeof document === 'undefined') throw new Error('Transcript downloads require a desktop or browser window')
    const result = await desktopCall(this.bridge, key, 'session.status')
    const record = (this.sessionOf(result) ?? result) as unknown as ExportSession
    const markdown = sessionToMarkdown(record)
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }))
    anchor.download = `${(str(record.title) || str(record.id) || 'session').replace(/[^\w-]+/g, '-').slice(0, 64) || 'session'}.md`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    setTimeout(() => URL.revokeObjectURL(anchor.href), 5_000)
  }

  /**
   * Undo recorded edits through the daemon — one file, or every file when
   * `path` is null. Undone files leave the review list; a refusal (the file
   * changed since the edit) surfaces as a visible error, never a silent one.
   */
  async undoChanges(path: string | null): Promise<void> {
    try {
      const result = await this.bridge.call('changes.undo', {
        session_key: this.sessionKey,
        ...(path ? { path } : {}),
      })
      if (result.ok === false && !Array.isArray(result.results)) {
        this.builder.push('notification', { severity: 'error', message: str(result.error) || 'undo failed' })
        this.notify()
        return
      }
      const results = Array.isArray(result.results) ? result.results : []
      const undone = new Set(
        results.filter(row => row && typeof row === 'object' && (row as Record<string, unknown>).ok === true)
          .map(row => str((row as Record<string, unknown>).path)),
      )
      const failures = results.filter(row => row && typeof row === 'object' && (row as Record<string, unknown>).ok !== true)
      this.patch({ changes: this.frame.changes.filter(change => !undone.has(change.path)) })
      const reverted = typeof result.reverted === 'number' ? result.reverted : 0
      const scope = path ? ` in ${path}` : ''
      if (failures.length) {
        const first = failures[0] as Record<string, unknown>
        this.builder.push('notification', {
          severity: 'error',
          message: `Undo refused: ${str(first.error) || 'unknown refusal'}`,
        })
      } else if (reverted > 0) {
        this.builder.push('notification', { severity: 'info', message: `Undid ${reverted} edit${reverted === 1 ? '' : 's'}${scope}` })
      }
      this.notify()
    } catch (error) {
      this.fail(error)
    }
  }

  /**
   * Create a git worktree through the daemon and switch the shell into it —
   * the shared daemon binds the next task to the isolated checkout.
   */
  async createWorktree(name: string): Promise<void> {
    try {
      const result = await this.bridge.call('workspace.worktree', { action: 'create', name })
      if (result.ok === false) {
        this.builder.push('notification', { severity: 'error', message: str(result.error) || 'worktree refused' })
        this.notify()
        return
      }
      const path = str(result.path)
      if (!path) return
      this.patch({ taskModalOpen: false })
      void this.bridge.useWorkspace?.(path)
    } catch (error) {
      this.fail(error)
    }
  }

  // ── Failure + retry ──────────────────────────────────────────────────

  /**
   * Continue the failed turn rather than send its prompt again. The daemon
   * keeps the prompt and every step that finished before the error, so
   * re-sending the prompt put it in the conversation twice — as a second
   * copy of a long instruction in the transcript — and the model started the
   * whole task over. The model is told what happened; the transcript shows
   * only "Continue".
   */
  private isExternalAcpConnected(): boolean {
    return Boolean(this.frame.acpBinding?.connected && this.frame.acpBinding.transport !== 'native')
  }

  private isAntigravityConnected(): boolean {
    return Boolean(this.frame.acpBinding?.connected && this.frame.acpBinding.provider === 'antigravity')
  }

  retryFailed(): void {
    const failed = this.frame.failed
    if (!failed) return
    this.failure = null
    this.patch({ failed: null, turnFailed: false, tab: 'activity' })
    if (!failed.lastUser) return
    const isExternalAcpSession =
      this.isExternalAcpConnected()
      || Boolean(this.frame.savedAcpBinding && this.frame.savedAcpBinding.transport !== 'native')
    if (isExternalAcpSession) {
      void this.submit('Continue')
      return
    }
    void this.submit(`Continue. Your previous reply was cut off by an error (${failed.error.slice(0, 300)}). Pick up where you stopped; do not repeat steps that already finished.`, 'Continue')
  }

  async retryCompaction(): Promise<void> {
    const key = this.sessionKey
    const failed = this.frame.failed
    if (!failed || this.frame.turnActive) return
    try {
      const result = await this.bridge.call('slash', { session_key: key, command: '/compact' })
      if (key !== this.sessionKey || this.frame.failed !== failed) return
      if (result.ok !== true) throw new Error(str(result.error) || str(result.output) || 'Compaction failed')
      this.resolveFailure()
      this.builder.push('notification', { severity: 'info', message: str(result.output) || 'Conversation compacted. You can now continue the task.' })
      this.notify()
    } catch (error) {
      if (key !== this.sessionKey || this.frame.failed !== failed) return
      this.failure = { ...failed, error: `Automatic context compaction failed: ${error instanceof Error ? error.message : String(error)}. Original conversation retained.` }
      this.patch({ failed: this.failure })
    }
  }

  resolveFailure(): void {
    this.failure = null
    this.patch({ failed: null, turnFailed: false })
  }

  // ── Settings data ────────────────────────────────────────────────────

  /** Restore persisted skill suggestions for the bound session. */
  loadSkillSuggestions(): void {
    void this.bridge.call('skill_suggestions', { session_key: this.sessionKey }).then(result => {
      if (result.ok === false) return
      const rows = Array.isArray(result.suggestions) ? result.suggestions : []
      const suggestions = rows.map(skillSuggestionOf).filter((row): row is SkillSuggestion => row !== null)
      this.patch({ skillSuggestions: suggestions })
    }).catch(() => {
      // Older daemons have no suggestions RPC; live structured events still work.
    })
  }

  /** Restore the creator-mode audit trail for the bound session. */
  loadCreatorTrace(): void {
    void this.bridge.call('creator_trace', { session_key: this.sessionKey }).then(result => {
      if (result.ok === false) return
      const rows = Array.isArray(result.trace) ? result.trace : []
      const trace = rows.map(creatorTraceOf).filter((row): row is CreatorTrace => row !== null)
      this.patch({ creatorTrace: trace })
    }).catch(() => {
      // Older daemons do not expose creator mode.
    })
  }

  clearAgentPresetError(): void {
    this.patch({ agentPresetsError: null })
  }

  async loadAgentPresets(): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.list', {})
      if (result.ok === false) throw new Error(str(result.error) || 'could not load agent presets')
      this.patch({ agentPresets: agentPresetsFromResult(result) })
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async setDefaultAgentPreset(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.setDefault', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not set default agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async copyAgentPreset(from: string, id: string, name?: string): Promise<boolean> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.copy', {
        from,
        agent_preset: id,
        ...(name?.trim() ? { name: name.trim() } : {}),
      })
      if (result.ok === false) throw new Error(str(result.error) || 'could not duplicate agent preset')
      await this.loadAgentPresets()
      return true
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  async removeAgentPreset(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.remove', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not remove agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async readAgentPreset(id: string): Promise<string> {
    const result = await this.bridge.call('agentPreset.read', { agent_preset: id })
    if (result.ok === false) throw new Error(str(result.error) || 'could not read agent preset')
    return str(result.content)
  }

  /** Persist an edited preset body (`agentPreset.write`); the daemon revalidates on save. */
  async writeAgentPreset(id: string, content: string): Promise<boolean> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.write', { agent_preset: id, content })
      if (result.ok === false) throw new Error(str(result.error) || 'could not save agent preset')
      await this.loadAgentPresets()
      return true
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  /** Bind an existing preset to the current session (`agentPreset.select`). */
  async selectAgentPreset(id: string): Promise<boolean> {
    if (!this.frame.currentId || this.frame.turnActive) return false
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.select', {
        agent_preset: id,
        session_key: this.sessionKey,
      })
      if (result.ok === false) throw new Error(str(result.error) || 'could not select agent preset')
      const applied = str(result.agent_preset)
      if (applied) this.patch({ currentAgentPreset: applied })
      return true
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  async openAgentPresetLocation(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.openDocument', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not open agent preset')
      const path = str(result.path)
      if (!path) throw new Error('The runtime did not return a preset location')
      if (!window.xerxes.openPath) throw new Error('Opening preset folders requires the desktop host')
      if (!await window.xerxes.openPath(path)) throw new Error('The system file manager could not open this preset folder')
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async draftAgentPreset(): Promise<void> {
    this.patch({ settingsOpen: false })
    await this.startTask(
      'Help me create a custom Xerxes agent preset. Ask what behavior and capabilities I want, then duplicate a suitable preset, author it, and validate it.',
      false,
      'creator',
    )
  }

  // ── Channels ─────────────────────────────────────────────────────────

  /** Load gateway status; flags stay truthful even when the manager is absent (`ok:false`). */
  loadChannels(): void {
    const version = ++this.channelsLoadVersion, key = this.sessionKey
    const current = () => version === this.channelsLoadVersion && key === this.sessionKey
    this.patch({ channelsLoading: true, channelsError: '' })
    void this.bridge.call('channel.list', {}).then(result => {
      if (!current()) return
      if (!Array.isArray(result.channels) || typeof result.channels_available !== 'boolean') throw new Error(text(result.error) || 'The runtime returned an invalid channel list.')
      this.applyChannelStatus(channelStatusFrom(result))
    }).catch(error => {
      if (current()) this.patch({ channelsError: desktopError(error) })
    }).finally(() => {
      if (current()) this.patch({ channelsLoading: false })
    })
  }

  private applyChannelStatus(status: ChannelStatus): void {
    this.patch({
      channels: status.channels,
      channelsAvailable: status.available,
      channelsConfigured: status.configured,
    })
  }

  /** Enable or disable one gateway; the response list and the `channel_status` broadcast both refresh the panel. */
  setChannelEnabled(name: string, enabled: boolean): Promise<void> {
    const method = enabled ? 'channel.enable' : 'channel.disable'
    return this.bridge
      .call(method, { name })
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || `could not ${enabled ? 'enable' : 'disable'} ${name}`)
        // Fast path: the response carries the refreshed channel LIST but no
        // availability flags — those ride the `channel_status` broadcast the
        // daemon emits right after, or the quiet reload below.
        const status = channelStatusFrom(result)
        this.patch({ channels: status.channels })
        this.loadChannels()
      })
  }

  // ── Terminals ────────────────────────────────────────────────────────

  loadTerminals(): void {
    const sessionKey = this.sessionKey
    const version = ++this.terminalsLoadVersion
    const current = () => sessionKey === this.sessionKey && version === this.terminalsLoadVersion
    this.patch({ terminalsLoading: true, terminalsError: null })
    void this.bridge
      .call('terminal.list', { session_key: sessionKey })
      .then(result => {
        if (!current()) return
        if (result.ok === false) throw new Error(str(result.error) || 'Could not list terminals')
        this.patch({ terminals: terminalsFromResult(result), terminalsLoading: false })
      })
      .catch(error => {
        if (!current()) return
        this.patch({ terminalsLoading: false, terminalsError: desktopError(error) })
      })
  }

  /** Read one terminal's retained output tail (`terminal.inspect`). */
  async inspectTerminal(id: string): Promise<TerminalDetail | null> {
    const result = await this.bridge.call('terminal.inspect', {
      terminal_id: id,
      session_key: this.sessionKey,
      max_output_chars: 24_000,
    })
    if (result.ok === false) throw new Error(str(result.error) || 'could not inspect terminal')
    return terminalDetailOf(result.terminal)
  }

  /**
   * A code block's Run: the runtime types it into one of this person's shells
   * that waits at its prompt, or opens a new one (`terminal.run`).
   */
  async runInTerminal(command: string): Promise<string> {
    let result: Record<string, unknown>
    try { result = await this.bridge.call('terminal.run', { session_key: this.sessionKey, command }) }
    catch (error) { throw new Error(desktopError(error)) }
    if (result.ok === false) {
      const message = str(result.error)
      throw new Error(/Unknown method: terminal\.run/.test(message)
        ? "This workspace's runtime is older than the app. Update it from the runtime status to run commands from the chat."
        : message || 'Could not run the command')
    }
    const id = str(result.terminal_id)
    if (!id) throw new Error('The runtime did not say which shell ran it')
    this.patch({ terminalFocus: { id, revision: (this.frame.terminalFocus?.revision ?? 0) + 1 } })
    return id
  }

  /** Send input to, interrupt, or kill one terminal (`terminal.control`). */
  async controlTerminal(id: string, action: 'write' | 'interrupt' | 'kill', chars?: string): Promise<void> {
    const params: Record<string, unknown> = {
      action,
      terminal_id: id,
      session_key: this.sessionKey,
      ...(action === 'write' ? { chars: chars ?? '' } : {}),
    }
    await this.bridge
      .call('terminal.control', params)
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || `terminal ${action} refused`)
        // The control result echoes the inspected terminal — fold its fresh
        // state (running flag, output size, exit code) into the open list.
        const detail = terminalDetailOf(result.terminal)
        this.patch({
          terminals: detail
            ? this.frame.terminals.map(row => (row.id === id ? detail : row))
            : this.frame.terminals,
        })
      })
  }

  // ── Session search ───────────────────────────────────────────────────

  openSessionSearch(): void {
    this.patch({
      searchOpen: true,
      paletteOpen: false,
      settingsOpen: false,
      taskModalOpen: false,
    })
  }

  closeSessionSearch(): void {
    this.patch({ searchOpen: false })
  }

  /**
   * Ask the daemon's transcript FTS (`session.search`). Stale answers drop:
   * a sequence counter makes the slowest response for an earlier needle
   * lose to the latest one.
   */
  runSessionSearch(query: string): void {
    const needle = query.trim()
    if (needle.length < 2) {
      this.searchSeq += 1
      this.patch({ searchResults: [], searchStats: null, searchSearching: false, searchError: null })
      return
    }
    const seq = ++this.searchSeq
    this.patch({ searchSearching: true, searchError: null })
    void this.bridge
      .call('session.search', { query: needle, limit: 24 })
      .then(result => {
        if (seq !== this.searchSeq) return
        if (result.ok === false) {
          this.patch({
            searchResults: [],
            searchStats: null,
            searchSearching: false,
            searchError: str(result.error) || 'search failed',
          })
          return
        }
        const rows = Array.isArray(result.results) ? result.results : []
        this.patch({
          searchResults: rows.map(searchHitOf).filter((row): row is SessionSearchHit => row !== null),
          searchStats: searchStatsOf(result.stats),
          searchSearching: false,
          searchError: null,
        })
      })
      .catch(error => {
        if (seq !== this.searchSeq) return
        this.patch({
          searchResults: [],
          searchStats: null,
          searchSearching: false,
          searchError: error instanceof Error ? error.message : String(error),
        })
      })
  }

  /** Private, Hybrid or Cloud, as the runtime reads its settings. A runtime without run_mode leaves it null. */
  async loadRunMode(): Promise<void> {
    try {
      const result = await this.bridge.call('run_mode.get', {})
      if (result.ok === false) {
        this.patch({ runMode: null, runModeError: result.code === 'unsupported' || str(result.error).startsWith('Unknown method') ? '' : str(result.error) })
        return
      }
      this.patch({ runMode: runModeView(result), runModeError: '' })
    } catch (error) {
      // A runtime from before run modes has no run_mode.get: nothing to show, not an error.
      const message = desktopError(error)
      this.patch({ runModeError: /unknown method/i.test(message) ? '' : message })
    }
  }

  /**
   * Choose a mode. Private and Hybrid need Bonsai set up (`needs_local_setup`),
   * Hybrid and Cloud a cloud model to lead or work (`needs_cloud_profile`); the
   * caller shows what to do. Fully local mode and the data boundary apply after
   * an idle restart, started here.
   */
  async setRunMode(mode: RunModeName, options: { workerProfile?: string } = {}): Promise<{ ok: boolean; code?: string }> {
    if (this.frame.runModeSaving || this.frame.turnActive) return { ok: false, code: 'busy' }
    this.patch({ runModeSaving: true, runModeError: '' })
    try {
      const result = await this.bridge.call('run_mode.set', {
        mode,
        ...(options.workerProfile === undefined ? {} : { worker_profile: options.workerProfile }),
      })
      if (result.ok === false) {
        const code = str(result.code)
        // These two are steps the page offers next, not errors.
        if (code !== 'needs_local_setup' && code !== 'needs_cloud_profile') this.patch({ runModeError: str(result.error) || 'The mode could not be changed.' })
        return { ok: false, code }
      }
      const view = runModeView(result)
      this.patch({ runMode: view })
      void this.loadProviders()
      this.loadModels(true)
      if (view.restartPending) void this.restartDaemon(false, false, 'mode')
      return { ok: true }
    } catch (error) {
      this.patch({ runModeError: desktopError(error) })
      return { ok: false }
    } finally {
      this.patch({ runModeSaving: false })
    }
  }

  /** The engine, model and server for Bonsai on this host, and any setup job. */
  async loadLocalSetup(): Promise<void> {
    this.patch({ localSetup: { ...this.frame.localSetup, loading: true } })
    try {
      const result = await this.bridge.call('local.status', {})
      const job = localSetupJobView(result.job)
      if (result.ok === false) {
        const unsupported = result.code === 'unsupported' || str(result.error).startsWith('Unknown method')
        this.patch({ localSetup: { ...this.frame.localSetup, job: job ?? this.frame.localSetup.job, error: unsupported ? '' : str(result.error), loading: false } })
        return
      }
      this.patch({ localSetup: { status: localStatusView(result.status), job: job ?? this.frame.localSetup.job, error: '', loading: false } })
    } catch (error) {
      this.patch({ localSetup: { ...this.frame.localSetup, error: desktopError(error), loading: false } })
    }
  }

  /**
   * Download (if needed) and start Bonsai, then switch to `thenMode`, led in
   * Hybrid by `lead` when one was picked before setup. Progress arrives as
   * `local_setup` events.
   */
  async startLocalSetup(thenMode?: 'private' | 'hybrid', lead?: string): Promise<void> {
    try {
      const params = thenMode === undefined ? {} : { then_mode: thenMode, ...(thenMode === 'hybrid' && lead ? { then_worker_profile: lead } : {}) }
      const result = await this.bridge.call('local.setup', params)
      const job = localSetupJobView(result.job)
      this.patch({ localSetup: { ...this.frame.localSetup, ...(job ? { job } : {}), error: result.ok === false ? str(result.error) || 'Setup could not start.' : '' } })
    } catch (error) {
      this.patch({ localSetup: { ...this.frame.localSetup, error: desktopError(error) } })
    }
  }

  cancelLocalSetup(): void {
    void this.bridge.call('local.setup.cancel', {}).catch(() => {})
  }

  private async localSetupFinished(): Promise<void> {
    await Promise.all([this.loadLocalSetup(), this.loadProviders(), this.loadRunMode()])
    this.loadModels(true)
    if (this.frame.runMode?.restartPending) await this.restartDaemon(false, false, 'mode')
  }

  async loadProviders(): Promise<void> {
    // "No saved provider profiles" is only true once provider_list has
    // answered; asserting it while in flight told a configured user they
    // had nothing — and Retry cleared the error first, so the retry showed
    // that same false empty state for its whole duration.
    this.patch({ providerError: '', providersLoading: true })
    const providers = this.bridge
      // In an SSH window, ask whether the host's own sign-ins work: those run there.
      .call('provider_list', this.frame.storageScope?.startsWith('ssh:') ? { with_readiness: true } : {})
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || 'Could not load provider profiles')
        const rows = Array.isArray(result.profiles) ? (result.profiles as unknown[]) : []
        return rows
          .map(raw => {
            const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
            const name = str(row.name)
            if (!name) return null
            return {
              name,
              label: str(row.label) || name,
              provider: str(row.provider),
              signsIn: row.signs_in === true,
              ...(row.ready === true ? { ready: true as boolean } : {}),
              model: str(row.model),
              // The built-in Claude Code fallback is "active" only until something is chosen; nothing runs on it.
              active: row.active === true && result.active_chosen !== false,
              baseUrl: str(row.base_url),
              ...(typeof row.local === 'boolean' ? { local: row.local } : {}),
            } satisfies ProviderRow
          })
          .filter((row): row is ProviderRow => row !== null)
      })
      .catch(error => { this.patch({ providerError: error instanceof Error ? error.message : String(error) }); return null })
    // The adapter catalog for the add/edit form; an older daemon simply
    // lacks the method, and the form falls back to a free-text type field.
    const types = this.bridge
      .call('provider_types', {})
      .then(result => {
        const rows = Array.isArray(result.types) ? (result.types as unknown[]) : []
        return rows
          .map(raw => {
            const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
            const name = str(row.name)
            if (!name) return null
            return {
              name,
              baseUrl: str(row.base_url),
              apiKeyEnv: str(row.api_key_env),
            } satisfies ProviderTypeRow
          })
          .filter((row): row is ProviderTypeRow => row !== null)
      })
      .catch(() => null)
    const status = this.bridge
      .call('runtime.status', {})
      .then(result => ({ permissionMode: str(result.permission_mode), model: str(result.model) }))
      .catch(() => null)
    await Promise.all([providers, types, status]).then(([rows, typeRows, state]) => {
      this.patch({ providersLoading: false })
      if (rows) this.patch({ providers: rows })
      if (typeRows) this.patch({ providerTypes: typeRows })
      // Daemon-wide fallback only: a session-scoped /permissions pin from
      // initialize is the truth for what this app's next tool call faces.
      if (state?.permissionMode && !this.frame.permissionMode) this.patch({ permissionMode: state.permissionMode })
    })
  }

  setPermissionMode(mode: PermissionMode): void {
    if (this.frame.permissionUpdating) return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({ permissionUpdating: true, permissionError: null })
    void this.bridge
      .call('slash', { command: `/permissions ${mode}`, session_key: sessionKey })
      .then(result => {
        if (!isCurrent()) return
        if (result.ok === false) {
          this.patch({ permissionError: str(result.error) || 'Permission change was refused. The previous mode is still active.' })
          return
        }
        // The slash result carries the pinned mode; adopt it immediately so
        // the card's ✓ marker follows the click instead of waiting for a
        // re-initialize that may never come.
        const pinned = str(result.permission_mode)
        if (result.ok !== false && pinned) this.patch({ permissionMode: pinned })
        void this.loadProviders()
      })
      .catch(error => {
        if (!isCurrent()) return
        this.patch({ permissionError: desktopError(error) })
      })
      .finally(() => { if (isCurrent()) this.patch({ permissionUpdating: false }) })
  }

  // ── Connection ───────────────────────────────────────────────────────

  private reconnecting = false
  private openingFreshTask = false
  private supportsConnectionLease = false
  async retryConnection(): Promise<void> {
    if (this.reconnecting || this.openingFreshTask || this.openingSession) return
    this.reconnecting = true
    this.patch({ connection: 'connecting' })
    // Resume the open conversation by id when one exists — a bare
    // initialize evicts the live session and the daemon would hand back a
    // fresh, context-free one under our key.
    const extra: Record<string, unknown> = this.frame.currentId
      ? { resume_session_id: this.frame.currentId }
      : {}
    await this.initializeSelfHealing(extra).then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
      },
      error => this.wentOffline(error),
    ).finally(() => { this.reconnecting = false })
  }

  private cameOnline(): void {
    if (this.frame.connection !== 'online') this.patch({ connection: 'online', error: null })
  }

  private wentOffline(error?: unknown): void {
    this.preparingSubmissions.delete(this.sessionKey)
    // A navigation that ends offline really is a connection problem now.
    if (this.frame.navigating) this.patch({ navigating: null })
    if (error !== undefined) this.patch({ error: error instanceof Error ? error.message : String(error) })
    if (this.supportsConnectionLease) {
      this.patch({ connection: 'offline' })
      return
    }
    // A daemon that died mid-turn will never send turn_end; clear the
    // acting badge (and with it the 1s tick) or Stop/⌘N stay bricked
    // against a turn that no longer exists. The live runs must fold too:
    // otherwise they sit in the builder and resurrect — blinking carets on
    // finished messages — the next time any turn renders its active fold.
    // `connecting` counts: a failed retryConnection lands here too.
    if (this.frame.connection !== 'offline' && this.frame.turnActive) {
      this.stopTick()
      this.pendingAgentCalls.clear()
      this.stopFleetPoll()
      this.builder.finalize()
      this.patch({ connection: 'offline', turnActive: false, turnSeconds: 0 })
      return
    }
    if (this.frame.connection !== 'offline') this.patch({ connection: 'offline' })
  }

  /** Cheap liveness probe; also heals the badge after a daemon restart. */
  /**
   * Keep the sidebar and connection current.
   *
   * Every workspace the window has opened keeps its own page and runtime
   * connection. Each used to list every saved and live session every five
   * seconds whether or not it could be seen — nine pages made the runtime
   * re-read the session lists nearly twice a second, and that polling was
   * most of the app's energy use. A hidden or covered page now skips the
   * beat (and catches up when shown); a visible but unfocused window
   * refreshes its lists every thirty seconds. Live events still arrive as
   * they happen either way.
   */
  private async beat(shown = false): Promise<void> {
    if (!shown && pageIsBackground()) return
    const now = Date.now()
    const unfocused = pageIsInactive()
    if (!shown && unfocused && now - this.lastListRefresh < UNFOCUSED_LIST_REFRESH_MS && this.frame.connection === 'online') return
    this.lastListRefresh = now
    void this.refreshContexts()
    if (this.frame.noWorkspace) return
    if (this.updatingRuntime || this.openingFreshTask || this.openingSession) return
    if (this.frame.connection === 'online') {
      this.refreshSessions()
      // Only a restart the person asked for is retried here (it waits for
      // running work). A build mismatch alone never restarts the runtime:
      // other clients (the VS Code extension, another app build) share it,
      // and two clients with different builds used to restart each other's
      // runtime every time it went idle, dropping unsaved chats each time.
      if (this.frame.daemonWarning && !this.frame.daemonWarning.startsWith('The app is older') && this.frame.runtimeUpdate === 'waiting') {
        await this.restartDaemon(false)
      }
      // Events would still be flowing; a silent socket only shows when a
      // call dies, which every action already routes through fail().
      return
    }
    void this.bridge.getWorkspaceDirectories?.().then(workspaceDirectories => {
      if (JSON.stringify(this.getSnapshot().workspaceDirectories) !== JSON.stringify(workspaceDirectories)) {
        this.patch({ workspaceDirectories })
      }
    }).catch(error => this.patch({ workspaceError: desktopError(error) }))
    if (connectionFailureKind(this.frame.error) !== 'transport') return
    await this.retryConnection()
  }

  // ── RPC helpers ──────────────────────────────────────────────────────

  private bridge: XerxesLike = {
    call: (method, params) => window.xerxes.call(method, params),
    getContextScope: () => window.xerxes.getContextScope?.() ?? Promise.resolve('local'),
    getContextHost: () => window.xerxes.getContextHost?.() ?? Promise.resolve(null),
    getContexts: () => window.xerxes.getContexts?.() ?? Promise.resolve([]),
    activateContext: (id, sessionId) => window.xerxes.activateContext?.(id, sessionId) ?? Promise.reject(new Error('Update the desktop app to switch contexts.')),
    // Passthrough — the wrapper must forward EVERY preload method or the
    // optional calls silently do nothing.
    getWorkspaceDirectories: () => window.xerxes.getWorkspaceDirectories?.() ?? Promise.resolve([]),
    removeWorkspace: dir => window.xerxes.removeWorkspace?.(dir) ?? Promise.resolve([]),
    restoreWorkspace: dir => window.xerxes.restoreWorkspace?.(dir) ?? Promise.resolve([]),
    remote: (action, params) => window.xerxes.remote?.(action, params) ?? Promise.reject(new Error('This host cannot manage remote workspaces.')),
    // Without these two the menu's Check for Updates… and the update prompt
    // never reached the window: the check ran and its answer was dropped.
    appUpdate: action => window.xerxes.appUpdate?.(action) ?? Promise.reject(new Error('Update the desktop app to check for updates.')),
    onAppUpdate: handler => window.xerxes.onAppUpdate?.(handler) ?? (() => {}),
    openWorkspaceWindow: (directory, resumeSessionId, options) => {
      if (!window.xerxes.openWorkspaceWindow) return Promise.reject(new Error('This desktop build cannot open additional windows. Relaunch the updated app.'))
      return window.xerxes.openWorkspaceWindow(directory, resumeSessionId, options)
    },
    startsFresh: () => window.xerxes.startsFresh?.() ?? Promise.resolve(false),
    freshPreset: () => window.xerxes.freshPreset?.() ?? Promise.resolve(null),
    chooseWorkspace: () => window.xerxes.chooseWorkspace?.() ?? Promise.resolve(null),
    useWorkspace: (dir, resumeSessionId) => window.xerxes.useWorkspace?.(dir, resumeSessionId) ?? Promise.resolve(null),
    getWorkspace: () => window.xerxes.getWorkspace?.() ?? Promise.resolve(''),
    getResumeSession: () => window.xerxes.getResumeSession?.() ?? Promise.resolve(null),
  }

  private sessionOf(result: Record<string, unknown>): Record<string, unknown> {
    return result.session && typeof result.session === 'object'
      ? (result.session as Record<string, unknown>)
      : {}
  }

  private telemetryFromSession(session: Record<string, unknown>): Partial<Snapshot> {
    const executions = Array.isArray(session.tool_executions)
      ? session.tool_executions.filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value))
      : []
    const storedToolDuration = executions.reduce((total, row) => total + Math.max(0, num(row.duration_ms) ?? 0), 0)
    const activeTurn = str(session.active_turn_id) !== ''
    const inflightRecord = activeTurn && session.inflight && typeof session.inflight === 'object' && !Array.isArray(session.inflight)
      ? (session.inflight as Record<string, unknown>)
      : null
    const inflightTools = inflightRecord && Array.isArray(inflightRecord.tools)
      ? inflightRecord.tools.filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value))
      : []
    const inflightToolDuration = inflightTools.reduce((total, row) => total + Math.max(0, num(row.duration_ms) ?? 0), 0)
    const inflightHasWorkingTool = inflightTools.some(row => typeof row.ok !== 'boolean' && typeof row.duration_ms !== 'number' && typeof row.error !== 'string')
    const inflightStart = activeTurn ? inflightStartedAtMs(inflightRecord) : null
    const storedLlmDuration = Math.max(0, num(session.llm_duration_ms) ?? 0)
    this.ttftSamples = Math.max(0, Math.trunc(num(session.ttft_samples) ?? 0))
    this.ttftTotalMs = Math.max(0, num(session.ttft_total_ms) ?? 0)
    const persistedTurns = Math.max(0, Math.trunc(num(session.turn_count) ?? 0))
    this.turnCount = Math.max(persistedTurns, activeTurn ? 1 : 0)
    const ttftAverage = num(session.ttft_avg_ms)
      ?? (this.ttftSamples > 0 ? this.ttftTotalMs / this.ttftSamples : null)
    const tokensPerSecond = num(session.tokens_per_second)
    const cacheHitRate = num(session.cache_hit_rate)
    const now = Date.now()
    const totalToolDuration = Math.max(0, num(session.tool_duration_ms) ?? 0, storedToolDuration + inflightToolDuration)
    const totalToolSteps = Math.max(0, Math.trunc(num(session.tool_steps) ?? 0), executions.length + inflightTools.length)
    const inferredLlmSteps = Math.max(
      0,
      Math.trunc(num(session.llm_steps) ?? num(session.calls) ?? 0),
      activeTurn ? Math.max(1, totalToolSteps > 0 ? totalToolSteps : 1) : 0,
    )
    const phaseAnchor = activeTurn
      ? (inflightStart !== null && storedLlmDuration === 0
          ? Math.min(now, inflightStart + inflightToolDuration)
          : now)
      : null
    return {
      turnCount: this.turnCount,
      llmDurationMs: storedLlmDuration,
      llmSteps: inferredLlmSteps,
      toolDurationMs: totalToolDuration,
      toolSteps: totalToolSteps,
      inputTokens: Math.max(0, num(session.input_tokens) ?? 0),
      outputTokens: Math.max(0, num(session.output_tokens) ?? 0),
      metricPhase: activeTurn ? (inflightHasWorkingTool ? 'tool' : 'llm') : null,
      metricPhaseStartedAt: phaseAnchor,
      tokensPerSecond: tokensPerSecond === null ? null : Math.max(0, tokensPerSecond),
      cacheHitRate: cacheHitRate === null ? null : Math.max(0, Math.min(1, cacheHitRate)),
      cacheReadTokens: num(session.cache_read_tokens),
      cacheWriteTokens: num(session.cache_write_tokens),
      ttftMs: ttftAverage === null ? null : Math.max(0, ttftAverage),
    }
  }

  private adoptHistory(session: Record<string, unknown>, preserve = false): void {
    const page = readHistoryPage(session.history)
    const previousCursor = this.historyBefore
    this.historyGeneration++
    if (!preserve) this.historyBlockIds.clear()
    this.legacyHistory = []
    if (page) {
      const overlap = preserve && page.actions.length ? this.historyActions.findIndex(action => action.id === page.actions[0]!.id && JSON.stringify(action) === JSON.stringify(page.actions[0])) : -1
      const older = overlap > 0 ? this.historyActions.slice(0, overlap) : []
      this.historyActions = [...older, ...page.actions]
      this.historyBefore = older.length ? previousCursor : page.before
      this.historySeen = new Set(this.historyActions.map(action => action.id))
      this.builder.reset(historyBlocks({ ...page, actions: this.historyActions }, this.historyBlockIds))
    } else {
      // Older runtimes lack server paging. Bound rendering until the runtime
      // can safely update; retain the received legacy data for manual paging.
      const blocks = blocksFromStoredMessages(session.transcript ?? session.messages, { executions: session.tool_executions, thinking: session.thinking_content })
      this.legacyHistory = blocks.slice(0, -100)
      this.builder.reset(blocks.slice(-100))
      this.historyActions = []
      this.historyBefore = null
      this.historySeen.clear()
    }
    if (str(session.active_turn_id) !== '' && session.inflight) {
      const hydrated = applyInflightTurn(this.builder, session.inflight)
      if (hydrated?.user) this.lastUser = hydrated.user
    }
    if (!this.lastUser) {
      const snapshotBlocks = this.builder.snapshot(str(session.active_turn_id) !== '')
      for (let i = snapshotBlocks.length - 1; i >= 0; i -= 1) {
        const block = snapshotBlocks[i]!
        if (block.kind === 'user' && block.text.trim()) {
          this.lastUser = block.text.trim()
          break
        }
      }
    }
    this.patch({ historyMore: Boolean(this.historyBefore || this.legacyHistory.length), historyLoading: false, historyError: null })
    this.refoldChangesFromHistory(session, preserve)
    if (!preserve) void this.revealLatestUserMessage()
  }

  /**
   * Page back until the person's own latest message is on screen. A chat opens
   * on its newest 100 actions, and a goal or workflow session can run hundreds
   * of tool actions after the last thing the person said: the opened chat was
   * tool output and a question with no trace of the conversation, which read as
   * the whole chat having been lost. Bounded, and harness-written prompts (goal
   * rounds) do not count.
   */
  private async revealLatestUserMessage(): Promise<void> {
    const generation = this.historyGeneration
    for (let page = 0; page < REVEAL_USER_MESSAGE_MAX_PAGES; page++) {
      if (generation !== this.historyGeneration || this.frame.historyError || !this.frame.historyMore) return
      if (this.builder.some(block => block.kind === 'user')) return
      await this.loadOlderHistory()
    }
  }

  /**
   * Rebuild the session-edits fold from restored history.
   *
   * `changes` was written only by the live `tool_call` event, so reopening
   * a task showed "No files changed in this session yet" while the git
   * working tree panel listed exactly those files as dirty — and the
   * session-scoped undo had nothing to act on.
   */
  private refoldChangesFromHistory(session: Readonly<Record<string, unknown>>, preserve = false): void {
    const legacyExecutions = Array.isArray(session.tool_executions) ? session.tool_executions : []
    const executions = legacyExecutions.length > 0
      ? legacyExecutions
      : this.historyActions.flatMap(action => action.executions)
    if (!executions.length) return
    let clearedForPreserve = false
    const clearBeforeFold = (): void => {
      if (!preserve || clearedForPreserve) return
      clearedForPreserve = true
      this.changes.clear()
      this.foldedToolIds.clear()
    }
    let folded = false
    for (const raw of executions) {
      if (!raw || typeof raw !== 'object') continue
      const execution = raw as Record<string, unknown>
      if (execution.permitted === false || execution.error) continue
      const args = parseArgs(execution.arguments ?? execution.inputs ?? execution.input)
      const diffs = parseStructuredDiffs(execution.diffs, execution.diff)
      if (diffs.length > 1) {
        for (const entry of diffs) {
          const entryStats = editStatsOf(execution.name, {}, entry, execution.locations)
          if (!entryStats) continue
          clearBeforeFold()
          this.foldChange(entryStats, {}, str(execution.name), entry)
          folded = true
        }
        continue
      }
      const structuredDiff = diffs[0] ?? null
      const stats = editStatsOf(execution.name, args, structuredDiff, execution.locations)
      if (!stats) continue
      clearBeforeFold()
      this.foldChange(stats, args, str(execution.name), structuredDiff)
      folded = true
    }
    if (folded) this.patch({ changes: [...this.changes.values()].sort((a, b) => a.path.localeCompare(b.path)) })
  }

  async loadOlderHistory(): Promise<void> {
    if (this.frame.historyLoading || !this.frame.historyMore) return
    const generation = this.historyGeneration
    const key = this.sessionKey
    this.patch({ historyLoading: true, historyError: null })
    try {
      if (this.legacyHistory.length) {
        this.builder.prepend(this.legacyHistory.splice(-100))
        this.patch({ historyMore: this.legacyHistory.length > 0 })
      } else {
        const result = await this.bridge.call('session.history', { session_key: key, before: this.historyBefore, history_limit: 100 })
        if (generation !== this.historyGeneration || key !== this.sessionKey) return
        if (result.ok !== true) throw new Error(str(result.error) || 'Could not load older history')
        const page = readHistoryPage(result.history)
        if (!page || page.before === this.historyBefore) throw new Error('History page did not advance')
        if (page.actions.some(action => this.historySeen.has(action.id))) throw new Error('History changed. Reopen this session to load its current history.')
        this.builder.prepend(historyBlocks(page, this.historyBlockIds))
        this.historyActions = [...page.actions, ...this.historyActions]
        for (const action of page.actions) this.historySeen.add(action.id)
        this.historyBefore = page.before
        this.patch({ historyMore: page.has_more })
      }
    } catch (error) {
      if (generation === this.historyGeneration && key === this.sessionKey) this.patch({ historyError: error instanceof Error ? error.message : String(error) })
    } finally {
      if (generation === this.historyGeneration && key === this.sessionKey) this.patch({ historyLoading: false })
    }
  }

  private async initialize(extra: Record<string, unknown>): Promise<Record<string, unknown>> {
    const result = await this.bridge.call('initialize', {
      history_limit: 100,
      session_key: this.sessionKey,
      ...extra,
      ...clientHandshake(),
    })
    if (result.ok === false) throw new Error(str(result.error) || 'Session initialization was rejected')
    return this.applyInitializedSession(result, extra)
  }

  /** One adoption path for startup, reconnect and explicit session navigation. */
  private applyInitializedSession(result: Record<string, unknown>, extra: Record<string, unknown>, explicitNavigation = false): Record<string, unknown> {
    this.supportsConnectionLease = result.connection_lease_supported === true
    const session = this.sessionOf(result)
    const replay = Array.isArray(result.reconnect_events) ? result.reconnect_events : null
    const preserveLiveTranscript = !explicitNavigation && replay !== null && this.frame.currentId === str(result.session_id ?? session.id)
    // The key every session-scoped call uses must be the one the daemon bound
    // to the session now on screen, whatever this view asked for. Adopting it
    // only when a key was passed left a plain initialize showing one session
    // while goal, background and status calls still addressed another.
    const adoptedKey = str(session.key) || (typeof extra.session_key === 'string' ? extra.session_key : '')
    if (!extra.resume_session_id && adoptedKey && adoptedKey !== this.sessionKey) {
      this.sessionKey = adoptedKey
      this.resetWorkspaceFolds()
      this.patch({ goal: '', approval: null, question: null, failed: null, compacting: undefined })
    }
    // A resume binds the session under the session id, not our requested
    // key — every session-scoped call below must target what the daemon
    // actually bound or it silently addresses a fresh session.
    const boundKey = str(session.key) || str(result.session_id)
    if (extra.resume_session_id && boundKey) {
      this.sessionKey = boundKey
      if (!preserveLiveTranscript) {
        // Reconnecting to the session already on screen is not a session
        // change. Resetting the folds here discarded the Changes review list,
        // the captured plan and the event log, and yanked the user off the
        // Changes tab — while the transcript this same call preserved still
        // showed the edits those folds described. Only the request state that
        // died with the old connection has to go.
        const sameSession = !explicitNavigation && this.frame.currentId === str(result.session_id ?? session.id)
        this.adoptHistory(session, sameSession)
        if (sameSession) this.resetInFlightRequests()
        else this.resetWorkspaceFolds()
      }
    }
    if (!extra.resume_session_id && !preserveLiveTranscript) this.adoptHistory(session)
    const reportedContextLimit = num(result.context_limit)
    const contextLimit = reportedContextLimit !== null && reportedContextLimit > 0
      ? reportedContextLimit
      : null
    // The daemon owns turn truth (`active_turn_id` exists only mid-turn).
    // A fold that still believes it is acting after a restart or reconnect
    // would show ▶ act forever with blinking carets on finished messages —
    // reconcile to the daemon and close any stranded live runs.
    const daemonInTurn = str(session.active_turn_id) !== ''
    const inflightStartedAt = daemonInTurn ? inflightStartedAtMs(session.inflight) : null
    if (daemonInTurn && !this.frame.turnActive) this.startTurn(inflightStartedAt ?? undefined)
    if (!daemonInTurn && this.frame.turnActive) {
      this.stopTick()
      this.builder.finalize()
    }
    const resolvedTurnStartedAt = daemonInTurn
      ? (inflightStartedAt ?? this.frame.turnStartedAt ?? Date.now())
      : null
    const resolvedTurnSeconds = resolvedTurnStartedAt !== null
      ? Math.max(0, Math.floor((Date.now() - resolvedTurnStartedAt) / 1000))
      : 0
    this.patch({
      connection: 'online',
      navigating: null,
      error: null,
      currentId: str(result.session_id ?? session.id),
      currentTitle: str(session.title),
      sessionKey: this.sessionKey,
      cwd: str(result.cwd ?? session.cwd),
      model: str(result.model ?? session.model),
      planMode: (result.plan_mode ?? session.plan_mode) === true,
      delegationMode: delegationModeValue(result.delegation_mode ?? session.delegation_mode),
      ...(str(result.reasoning_effort)
        ? { reasoningEffort: str(result.reasoning_effort) }
        : {}),
      currentAgentPreset: str(result.agent_name ?? session.agent_id) || this.frame.currentAgentPreset,
      branch: str(result.branch),
      daemonWarning: daemonCompatibilityWarning(result),
      costUsd: costUsdFromPayload(result),
      costBasis: (result.cost_basis ?? session.cost_basis) === 'subscription' ? 'subscription' : null,
      contextMax: contextLimit,
      approval: null,
      ...(this.frame.currentId !== str(result.session_id ?? session.id) ? { question: null, goal: '' } : {}),
      ...(daemonInTurn ? { turnStartedAt: resolvedTurnStartedAt, turnSeconds: resolvedTurnSeconds } : { turnActive: false, turnSeconds: 0, turnStartedAt: null }),
      // initialize reports THIS session's policy — /permissions pins the
      // mode per session, so the daemon-wide runtime.status would lie about
      // what the next tool call will actually face.
      ...(str(result.permission_mode) ? { permissionMode: str(result.permission_mode) } : {}),
      ...this.telemetryFromSession(session),
    })
    this.adoptFleet(session)
      this.adoptTodos(session)
    if (preserveLiveTranscript) {
      for (const raw of replay ?? []) {
        const frame = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
        const params = frame.params && typeof frame.params === 'object' ? frame.params as Record<string, unknown> : {}
        if (frame.method === 'event' && typeof params.type === 'string' && params.payload && typeof params.payload === 'object') {
          this.onEvent({ type: params.type, payload: params.payload as Record<string, unknown> })
        }
      }
    }
    for (const raw of Array.isArray(result.pending_interactions) ? result.pending_interactions : []) {
      if (raw && typeof raw === 'object' && (raw.type === 'approval_request' || raw.type === 'question_request') && raw.payload && typeof raw.payload === 'object') {
        this.onEvent({ type: raw.type, payload: raw.payload as Record<string, unknown> })
      }
    }
    this.loadSkillSuggestions()
    this.loadCreatorTrace()
    const inlineSwarm = parseSwarmStatus(result.swarm ?? session.swarm)
    if (explicitNavigation || !preserveLiveTranscript) {
      this.swarmLoadVersion++
      this.patch({
        swarm: inlineSwarm,
        swarmMessages: [],
        swarmInspect: null,
        swarmInspectTarget: '',
        swarmLoading: false,
        swarmError: null,
        swarmOpen: false,
      })
    } else if (inlineSwarm) {
      this.patch({ swarm: inlineSwarm })
    }
    void this.refreshSwarm({ silent: true })
    const sessionMetadata =
      session.metadata && typeof session.metadata === 'object' && !Array.isArray(session.metadata)
        ? (session.metadata as Readonly<Record<string, unknown>>)
        : null
    const rawLiveAcp = parseAcpSessionBinding(result.acp_binding ?? session.acp_binding)
    const rawSavedAcp = parseAcpSessionBinding(result.saved_acp_binding ?? session.saved_acp_binding)
    const rawMetaAcp = parseAcpSessionBinding(sessionMetadata?.acp_binding)
    const inlineAcpBinding = rawLiveAcp?.connected
      ? rawLiveAcp
      : !rawLiveAcp && !rawSavedAcp && rawMetaAcp?.connected
        ? rawMetaAcp
        : null
    const savedAcpBinding = inlineAcpBinding
      ? null
      : rawSavedAcp
        ? { ...rawSavedAcp, connected: false }
        : rawLiveAcp && !rawLiveAcp.connected
          ? rawLiveAcp
          : rawMetaAcp && !rawMetaAcp.connected
            ? rawMetaAcp
            : null
    if (explicitNavigation || !preserveLiveTranscript) {
      this.acpLoadVersion++
      this.patch({
        acpBinding: inlineAcpBinding,
        savedAcpBinding,
        acpConnecting: false,
        acpError: rawLiveAcp?.error ?? null,
        acpAuthRequired: false,
        acpAuthMethods: [],
      })
    } else if (inlineAcpBinding || savedAcpBinding) {
      this.patch({
        ...(inlineAcpBinding ? { acpBinding: inlineAcpBinding, savedAcpBinding: null } : { savedAcpBinding }),
      })
    }
    void this.refreshAcpAgents({ silent: true })
    this.learningReviewToken = null
    this.learningLoadVersion++
    this.patch({ learning: null, learningCounts: { pending_proposals: 0, pending_lessons: 0 }, learningResults: {}, learningReviewBusy: false, learningOpen: false })
    void this.refreshLearning()
    return result
  }

  /**
   * Fleet truth: the parent session's subagent_snapshots panel — the
   * sessions map behind session.active_list only holds client-opened
   * sessions, so filtering it for kind 'subagent' shows nothing while
   * subagents actually run.
   */
  private adoptTodos(session: Readonly<Record<string, unknown>>): void {
    const structured = todoItemsOf(session.todos)
    if (structured !== null) { this.patch({ todos: structured }); return }
    if (!Array.isArray(session.tool_executions)) return
    for (const execution of [...session.tool_executions].reverse()) {
      if (!execution || execution.permitted === false || execution.error) continue
      const todos = todosFromResult(execution.name, execution.result ?? execution.return_value)
      if (todos !== null) { this.patch({ todos }); return }
    }
  }

  /** The daemon's `subagent_revision` of the fleet last adopted, per session key. */
  private fleetRevision: { readonly key: string; readonly revision: string } | null = null

  private adoptFleet(session: Readonly<Record<string, unknown>>, requestedAt = Date.now()): void {
    // The daemon left the rows out because they match the revision we sent: keep what we have.
    if (session.subagent_snapshots_unchanged === true) return
    const revision = str(session.subagent_revision)
    this.fleetRevision = revision ? { key: this.sessionKey, revision } : null
    const raw = Array.isArray(session.subagent_snapshots) ? session.subagent_snapshots : []
    const previousById = new Map(this.frame.fleet.map(agent => [agent.id, agent] as const))
    const fleet = raw
      .map(item => {
        const row = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
        const id = str(row.id)
        if (!id) return null
        const previous = previousById.get(id)
        const updatedAt = Date.parse(str(row.updated_at))
        const liveWins = (previous?.agentDetails?.lastEventAt !== undefined && (!Number.isFinite(updatedAt) || updatedAt <= previous.agentDetails.lastEventAt)) || (previous?.agentDetails?.lastReceiptAt !== undefined && requestedAt <= previous.agentDetails.lastReceiptAt)
        const label = str(row.title) || str(row.name) || str(row.agent_id) || `#${id.slice(0, 6)}`
        const count = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
        const paths = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
        const entry: SessionRow = {
          id,
          key: id,
          title: label,
          status: liveWins ? previous.status : str(row.status) || 'running',
          age: '',
          current: false,
          kind: 'subagent',
          turns: 0,
          messages: 0,
          cwd: '',
          untitled: false,
          agentDetails: {
            ...previous?.agentDetails,
            ...(agentGroupOf(row.group) ?? previous?.agentDetails?.group ? { group: (agentGroupOf(row.group) ?? previous?.agentDetails?.group)! } : {}),
            summary: str(row.summary), error: str(row.error), model: str(row.model),
            baseAgent: str(row.agent_id) || previous?.agentDetails?.baseAgent || '',
            parentId: str(row.parent_id) || str(row.creator_id),
            providerProfile: str(row.provider_profile) || previous?.agentDetails?.providerProfile || '',
            reasoningEffort: str(row.reasoning_effort) || previous?.agentDetails?.reasoningEffort || '',
            toolCount: count(row.tool_count), inputTokens: count(row.input_tokens), outputTokens: count(row.output_tokens),
            filesRead: paths(row.files_read), filesWritten: paths(row.files_written),
            ...(liveWins ? previous.agentDetails : {}),
          },
        }
        return entry
      })
      .filter(row => row !== null)
    const merged = [...fleet, ...this.frame.fleet.filter(row => (row.agentDetails?.lastEventAt !== undefined || row.agentDetails?.lastReceiptAt !== undefined) && !fleet.some(saved => saved.id === row.id))]
    this.patch({ fleet: merged })
    this.syncAgentMembersFromFleet(merged)
  }

  /**
   * Fold daemon snapshot statuses into the agents card. Snapshots persist in
   * session metadata, so this is the terminal-status path for children that
   * outlive their turn. Unseen rows are adopted only while LIVE — the
   * previous turn's terminal snapshots must not open a stale card at the
   * next turn's start.
   */
  /** A finished Workflow reports its cost; the run's card shows it in the header. */
  private foldWorkflowResult(value: unknown): void {
    let result: unknown = value
    if (typeof result === 'string') { try { result = JSON.parse(result) } catch { return } }
    if (!result || typeof result !== 'object') return
    const run = result as Record<string, unknown>
    const id = str(run.workflow_id)
    const known = typeof run.cost_usd === 'number' ? run.cost_usd : typeof run.cost_usd_known === 'number' ? run.cost_usd_known : undefined
    if (!id || known === undefined) return
    const partial = typeof run.unpriced_agents === 'number' && run.unpriced_agents > 0
    let touched = false
    for (const [key, member] of this.agentMembers) {
      if (member.group?.id !== id) continue
      this.agentMembers.set(key, { ...member, group: { ...member.group, costUsd: known, ...(partial ? { costPartial: true } : {}) } })
      touched = true
    }
    if (touched) { this.builder.pushAgents([...this.agentMembers.values()]); this.notify() }
  }

  private syncAgentMembersFromFleet(fleet: readonly SessionRow[]): void {
    if (!syncAgentMembers(this.agentMembers, fleet)) return
    const members = [...this.agentMembers.values()]
    this.builder.pushAgents(members)
    if (members.some(m => m.status === 'working')) this.startFleetPoll()
    this.notify()
  }

  /**
   * Events tagged with a foreign session_id are background turns (the daemon
   * runs them as bg-* sessions on this connection's pipe). They never enter
   * the foreground fold — they move the header jobs chip only.
   */
  private onBackgroundEvent(type: string, payload: Readonly<Record<string, unknown>>, sessionId: string): void {
    if (type === 'turn_begin') {
      const title = (str(payload.user_input) || str(payload.text) || 'background task').slice(0, 80)
      this.patch({
        backgroundJobs: [
          ...this.frame.backgroundJobs.filter(job => job.id !== sessionId),
          { id: sessionId, title, status: 'working' },
        ],
      })
      return
    }
    if (type === 'session_title') {
      const title = str(payload.title)
      if (title) {
        this.patch({
          backgroundJobs: this.frame.backgroundJobs.map(job => (job.id === sessionId ? { ...job, title } : job)),
        })
      }
      return
    }
    if (type === 'turn_end') {
      // Running-only chip: settled work leaves the list.
      this.patch({ backgroundJobs: this.frame.backgroundJobs.filter(job => job.id !== sessionId) })
      void this.refreshSessions()
    }
  }

  /**
   * Attach-mid-run seed: the event stream only sees turns that START while
   * we're attached, so the chip also reads `session.active_list` — bg-*
   * sessions with an active turn are working background jobs by definition.
   */
  private seedBackgroundJobs(rows: unknown): void {
    if (!Array.isArray(rows)) return
    const seeded: BackgroundJob[] = []
    for (const raw of rows) {
      const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
      const id = str(row.id ?? row.session_id)
      if (!id || id === this.frame.currentId) continue
      if (!str(row.key).startsWith('bg-')) continue
      if (!str(row.active_turn_id)) continue
      seeded.push({ id, title: str(row.title) || 'background task', status: 'working' })
    }
    // Event-driven entries win (they carry the prompt as title).
    const live = this.frame.backgroundJobs.filter(job => !seeded.some(s => s.id === job.id))
    const merged = [...seeded, ...live]
    const same =
      merged.length === this.frame.backgroundJobs.length &&
      merged.every((job, index) => job.id === this.frame.backgroundJobs[index]?.id && job.status === this.frame.backgroundJobs[index]?.status)
    if (!same) this.patch({ backgroundJobs: merged })
  }

  private async refreshGoal(): Promise<void> {
    const key = this.sessionKey
    try {
      const result = await this.bridge.call('session.goal', { session_key: key, input: '' })
      // A reply for the session this view has since left belongs to that
      // session: applied here, a goal round ending as New task was pressed
      // put the old goal on the new, empty task.
      if (key !== this.sessionKey) return
      this.patch({ goal: str(result.text) })
    } catch {
      // Best-effort view; absence renders an empty goal card.
    }
  }

  /**
   * Stop every live agent of this session (`subagent.interrupt` without a
   * task). Interrupting the turn no longer does this: agents outlive a
   * stopped turn, so this is the explicit way to end them all.
   */
  async stopAllAgents(): Promise<string> {
    const key = this.sessionKey
    const result = await desktopCall(this.bridge, key, 'subagent.interrupt', {})
    const stopped = typeof result.interrupted === 'number' ? result.interrupted : 0
    if (key === this.sessionKey) this.refreshFleet()
    if (result.found !== true || stopped === 0) return 'No agents were running.'
    return `Stopping ${stopped} agent${stopped === 1 ? '' : 's'}.`
  }

  /** Mid-turn fleet refresh: snapshots only move while a turn runs. */
  async controlAgent(id: string, action: 'stop' | 'retry', message = ''): Promise<string> {
    if (!this.frame.fleet.some(row => row.id === id)) throw new Error('This agent is no longer in the current session')
    const key = this.sessionKey
    const started = Date.now()
    const result = await desktopCall(this.bridge, key, action === 'retry' ? 'subagent.retry' : 'subagent.interrupt', { task: id, ...(message.trim() ? { message: message.trim() } : {}) })
    if (action === 'stop' && result.found !== true) throw new Error('The runtime did not confirm the stop request')
    if (action === 'retry' && result.ok !== true) throw new Error('The runtime did not confirm the retry')
    if (key === this.sessionKey) {
      if (action === 'retry') {
        // An acknowledgement is not a live agent event. Giving it a timestamp
        // can mask a completed snapshot produced before the RPC reply arrives.
        this.patch({ fleet: this.frame.fleet.map(row => row.id === id && (row.agentDetails?.lastEventAt ?? 0) <= started ? { ...row, status: 'running', ...(row.agentDetails ? { agentDetails: { ...row.agentDetails, error: '', summary: '', toolCalls: [], thinking: [], notes: [], startedAt: Date.now(), lastEventAt: undefined } } : {}) } : row) })
        this.syncAgentMembersFromFleet(this.frame.fleet)
        this.startFleetPoll()
      }
      this.refreshFleet()
    }
    return action === 'retry' ? 'Retry accepted. Waiting for agent progress.' : 'Stop requested. Waiting for the runtime to confirm it ended.'
  }

  /**
   * Send a message to a running child. It joins the agent's context at its
   * next provider/tool boundary — the same semantics as steering the main
   * turn, and equally not a conversation: the agent may or may not act on it.
   *
   * `delivered:false` is the interesting case and is reported as a failure,
   * because the daemon returns it for a child that finished between the user
   * reading the roster and pressing send. Treating that as success would tell
   * someone their redirect landed when nothing will ever read it.
   */
  async steerAgent(id: string, message: string): Promise<string> {
    const cleaned = message.trim()
    if (!cleaned) throw new Error('Type a message to send')
    if (!this.frame.fleet.some(row => row.id === id)) throw new Error('This agent is no longer in the current session')
    const key = this.sessionKey
    const result = await desktopCall(this.bridge, key, 'subagent.steer', { task: id, message: cleaned })
    if (result.ok !== true) throw new Error(text(result.error) || 'The runtime rejected the message')
    if (result.delivered !== true) throw new Error('This agent is no longer running, so the message was not delivered')
    if (key === this.sessionKey) this.refreshFleet()
    return 'Sent. The agent reads it at its next step.'
  }

  /** One fleet read in flight at a time; calls meanwhile ask for exactly one read after it. */
  private fleetRead: { again: boolean } | null = null

  /**
   * Re-read the parent's agent rows. Single-flight: on a slow link (an SSH
   * workspace behind a lossy tunnel) a reply can take longer than the poll
   * interval, and starting a new read every tick queued replies behind each
   * other until every RPC on the link took seconds. Calls made while a read is
   * pending collapse into one follow-up read.
   */
  private refreshFleet(): void {
    if (this.fleetRead) { this.fleetRead.again = true; return }
    const read = { again: false }
    this.fleetRead = read
    const key = this.sessionKey
    const requestedAt = Date.now()
    const revision = this.frame.sessionOpenRevision
    // Send the revision we hold so an unchanged fleet comes back as a few bytes, not every row.
    const known = this.fleetRevision?.key === key ? this.fleetRevision.revision : undefined
    void this.bridge
      .call('session.status', { session_key: key, history_limit: 0, ...(known ? { subagent_revision: known } : {}) })
      .then(result => {
        if (key !== this.sessionKey || revision !== this.frame.sessionOpenRevision) return
        const session = this.sessionOf(result)
        this.adoptFleet(Object.keys(session).length ? session : result, requestedAt)
      })
      .catch(() => {})
      .finally(() => {
        if (this.fleetRead === read) this.fleetRead = null
        if (read.again) this.refreshFleet()
      })
  }

  /**
   * Poll while agent-family tools are in flight. A subagent spawn is
   * persisted inside tool execution, so the refresh fired at the `tool_call`
   * event itself can still come back empty — the poll is what guarantees the
   * rail shows the child while it actually runs.
   */
  private startFleetPoll(): void {
    if (this.fleetPoll) return
    this.fleetPoll = setInterval(() => {
      const anyWorking = [...this.agentMembers.values()].some(m => m.status === 'working')
      // Background children outlive their turn's tool calls — the poll only
      // stops only after confirmed terminal state, including long-running work.
      if (this.pendingAgentCalls.size === 0 && !anyWorking) {
        this.stopFleetPoll()
        return
      }
      if (pageIsInactive()) return
      this.refreshFleet()
    }, FLEET_POLL_MS)
  }

  private stopFleetPoll(): void {
    if (this.fleetPoll) {
      clearInterval(this.fleetPoll)
      this.fleetPoll = null
    }
  }

  /** Fetch MCP statuses without presenting an unavailable response as empty. */
  refreshMcpStatus(): Promise<void> {
    return desktopCall(this.bridge, this.sessionKey, 'session.status', { history_limit: 0 })
      .then(result => {
        const raw = (this.sessionOf(result).mcp_status ?? (result as Record<string, unknown>).mcp_status) as unknown
        if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('MCP status is unavailable from this runtime. Check the runtime version and connection.')
        const statuses: Record<string, McpServerStatus> = {}
        for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
          if (!value || typeof value !== 'object') continue
          const record = value as Record<string, unknown>
          statuses[name] = {
            connected: record.connected === true,
            tools: typeof record.tools === 'number' ? record.tools : 0,
            resources: typeof record.resources === 'number' ? record.resources : 0,
            prompts: typeof record.prompts === 'number' ? record.prompts : 0,
            ...(typeof record.lastError === 'string' && record.lastError ? { lastError: record.lastError } : {}),
          }
        }
        this.patch({ mcpStatus: statuses })
      })
  }

  /** Reconnect every configured MCP server through the daemon's slash RPC. */
  async reloadMcp(): Promise<void> {
    try {
      await desktopCall(this.bridge, this.sessionKey, 'slash', { command: '/reload-mcp' })
    } catch (reloadError) {
      try { await this.refreshMcpStatus() }
      catch (statusError) {
        throw new AggregateError([reloadError, statusError], `MCP reload failed and current server status could not be refreshed. ${desktopError(reloadError)}; ${desktopError(statusError)}`, { cause: reloadError })
      }
      throw reloadError
    }
    await this.refreshMcpStatus()
  }

  private sessionsRefresh = 0
  /** One sidebar read in flight at a time (see refreshFleet); calls meanwhile ask for one read after it. */
  private sessionsRead: { again: boolean } | null = null

  private refreshSessions(): void {
    if (this.sessionsRead) { this.sessionsRead.again = true; return }
    const read = { again: false }
    this.sessionsRead = read
    const revision = ++this.sessionsRefresh
    void this.bridge.getWorkspaceDirectories?.().then(workspaceDirectories => {
      if (JSON.stringify(this.getSnapshot().workspaceDirectories) !== JSON.stringify(workspaceDirectories)) {
        this.patch({ workspaceDirectories })
      }
    }).catch(error => this.patch({ workspaceError: desktopError(error) }))
    const saved = this.bridge
      .call('session.list', { kind: 'main', scope: 'global', limit: 60 })
      .then(result => normalize(result.sessions, this.frame.currentId))
      .catch(() => null)
    const active = this.bridge
      .call('session.active_list', { history_limit: 0 })
      .then(result => {
        this.seedBackgroundJobs(result.sessions)
        return normalize(result.sessions, this.frame.currentId)
      })
      .catch(() => null)
    void Promise.all([saved, active]).then(([savedRows, activeRows]) => {
      if (revision !== this.sessionsRefresh) return
      if (!savedRows && !activeRows) return
      // The attached session is not 'fleet' and not a history row — it is
      // what the chat column is already showing.
      const currentId = this.frame.currentId
      const all = activeRows ?? this.frame.live
      // An untitled 0-turn live row is an empty shell — a session some client
      // opened and never spoke in. It is not a task; listing it as one is how
      // the sidebar filled with "0 turns" ghosts.
      const live = all.filter(
        row => row.kind === 'main' && row.id !== currentId && !(row.turns === 0 && row.untitled),
      )
      // Fleet comes from the parent's subagent_snapshots (adoptFleet) —
      // active_list only holds client-opened sessions and would blank the
      // panel while subagents actually run.
      const liveIds = new Set(live.map(row => row.id))
      const history = (savedRows ?? this.frame.sessions).filter(row => !liveIds.has(row.id) && row.id !== currentId)
      const sessionActivity: Record<string, number> = { ...this.frame.sessionActivity }
      for (const row of [...(savedRows ?? []), ...all]) {
        if (row.activeAt !== undefined) sessionActivity[row.id] = Math.max(sessionActivity[row.id] ?? 0, row.activeAt)
      }
      const prev = this.getSnapshot()
      if (
        JSON.stringify(prev.live) !== JSON.stringify(live)
        || JSON.stringify(prev.sessions) !== JSON.stringify(history)
        || JSON.stringify(prev.sessionActivity) !== JSON.stringify(sessionActivity)
      ) {
        this.patch({ live, sessions: history, sessionActivity })
      }
      this.enrichUntitled(history)
    }).finally(() => {
      if (this.sessionsRead === read) this.sessionsRead = null
      if (read.again) this.refreshSessions()
    })
  }

  /**
   * Give untitled history rows a real identity: the first user message,
   * fetched lazily through session.status and capped. Rows whose sessions
   * are no longer loaded answer `{ok:false}` and keep the short-id label.
   */
  private enrichUntitled(rows: readonly SessionRow[]): void {
    for (const row of rows) {
      // Missing previews are common for unopened sessions. Do not refetch
      // every one on every heartbeat or redraw the transcript for empty replies.
      if (this.enriching.size >= 4) break
      if (row.untitled && !this.snippets[row.id] && !this.enriching.has(row.id) && (this.snippetRetryAt.get(row.id) ?? 0) <= Date.now()) {
        this.enriching.add(row.id)
        let changed = false
        void this.bridge
          .call('session.status', { session_key: row.key, history_limit: 0 })
          .then(result => {
            const session = this.sessionOf(result)
            if (typeof session.preview === 'string' && session.preview.trim()) {
              this.snippets = { ...this.snippets, [row.id]: session.preview.slice(0, SNIPPET_CAP) }
              changed = true
              return
            }
            const transcript = session.transcript ?? session.messages
            if (!Array.isArray(transcript)) return
            for (const message of transcript as unknown[]) {
              if (!message || typeof message !== 'object') continue
              const record = message as Record<string, unknown>
              if (record.role !== 'user') continue
              const content = record.content
              const text = typeof content === 'string'
                ? content
                : Array.isArray(content)
                  ? content.map(part => (part && typeof part === 'object' && typeof (part as Record<string, unknown>).text === 'string' ? (part as Record<string, unknown>).text as string : '')).join('')
                  : ''
              const cleaned = text.replace(/\s+/g, ' ').trim()
              if (cleaned) {
                this.snippets = { ...this.snippets, [row.id]: cleaned.length > SNIPPET_CAP ? `${cleaned.slice(0, SNIPPET_CAP - 1)}…` : cleaned }
                changed = true
                break
              }
            }
          })
          .catch(() => {})
          .finally(() => {
            this.enriching.delete(row.id)
            if (!changed) {
              this.snippetRetryAt.delete(row.id)
              this.snippetRetryAt.set(row.id, Date.now() + 60_000)
              if (this.snippetRetryAt.size > 256) this.snippetRetryAt.delete(this.snippetRetryAt.keys().next().value!)
            } else {
              this.snippetRetryAt.delete(row.id)
              this.notify()
            }
          })
      }
    }
  }

  private async setGoal(input: string): Promise<{ ok: boolean; text: string }> {
    try {
      const key = this.sessionKey
      const result = await this.bridge.call('session.goal', { session_key: key, input })
      const ok = result.ok === true
      const text = str(result.text) || (ok ? 'ok' : 'command failed')
      if (key === this.sessionKey) this.patch({ goal: ok ? text : this.frame.goal })
      this.builder.push('notification', { severity: ok ? 'info' : 'error', message: text })
      this.notify()
      return { ok, text }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      this.fail(error)
      return { ok: false, text }
    }
  }

  // ── Event stream ─────────────────────────────────────────────────────

  private onEvent(event: DaemonEvent): void {
    const { type, payload } = event
    if (type === 'desktop_connection') {
      if (payload.online === false) this.wentOffline(new Error('Connection lost. Reconnecting to your session.'))
      else if (payload.online === true && this.frame.connection !== 'online' && !this.frame.noWorkspace) this.retryConnection()
      return
    }
    if (type === 'session_attention') {
      // Background thread status changed (finished, stopped, failed, needs input, or saved an artifact).
      // Refresh sidebar state without polluting the active transcript or diagnostics ring.
      void this.refreshSessions()
      return
    }
    this.pushLog(type, payload)
    // Learning is workspace-scoped even when another session made the change.
    if (type === 'learning_changed') {
      if (str(payload.workspace) !== (this.frame.learning?.workspace ?? this.frame.cwd)) return
      try { this.patch({ learningCounts: learningCounts(payload.counts) }) }
      catch (error) { this.patch({ learningError: desktopError(error) }) }
      void this.refreshLearning()
      return
    }
    if (type === 'swarm_event') {
      const sid = str(payload.session_id)
      const knownIds = [this.frame.currentId, this.sessionKey, this.frame.swarm?.session_id ?? ''].filter(Boolean)
      if (sid && knownIds.length > 0 && !knownIds.includes(sid)) return
      this.applySwarmEvent(payload)
      return
    }
    if (type === 'btw_event') {
      const sid = str(payload.session_id)
      const knownIds = [this.frame.currentId, this.sessionKey].filter(Boolean)
      if (sid && knownIds.length > 0 && !knownIds.includes(sid)) return
      this.applyBtwEvent(payload)
      return
    }
    if (type === 'acp_status') {
      const sid = str(payload.session_id)
      const knownIds = [this.frame.currentId, this.sessionKey].filter(Boolean)
      if (sid && knownIds.length > 0 && !knownIds.includes(sid)) return
      const binding = parseAcpSessionBinding(payload.binding)
      const agents = Array.isArray(payload.agents) ? parseAcpExternalAgents(payload.agents) : undefined
      const wasExternal = Boolean(this.frame.acpBinding?.connected && this.frame.acpBinding.transport !== 'native')
      const isConnectedExternal = Boolean(binding?.connected && binding.transport !== 'native')
      const runtimeTransitioned = wasExternal !== isConnectedExternal
      const statusError = binding?.error ?? (str(payload.error) || null)
      const statusAuthMethods = parseAcpAuthMethods(payload.auth_methods ?? payload.authMethods)
      const statusAuthRequired =
        !isConnectedExternal
        && (payload.auth_required === true || payload.authRequired === true || statusAuthMethods.length > 0)
      const fallbackSaved =
        !isConnectedExternal && binding && binding.transport !== 'native'
          ? { ...binding, connected: false }
          : !isConnectedExternal && statusError && this.frame.acpBinding && this.frame.acpBinding.transport !== 'native'
            ? { ...this.frame.acpBinding, connected: false }
            : undefined
      this.patch({
        acpBinding: isConnectedExternal ? binding : null,
        ...(isConnectedExternal
          ? { savedAcpBinding: null }
          : fallbackSaved !== undefined
            ? { savedAcpBinding: fallbackSaved }
            : {}),
        acpConnecting: false,
        acpError: statusError,
        acpAuthRequired: statusAuthRequired,
        acpAuthMethods: statusAuthRequired ? statusAuthMethods : [],
        ...(agents ? { acpAgents: agents } : {}),
        ...(isConnectedExternal
          ? {
              modelMenuOpen: false,
              pickerOpen: false,
              reasoningPickerOpen: false,
              delegationMenuOpen: false,
              skillSuggestions: [],
            }
          : {}),
        ...(runtimeTransitioned
          ? {
              swarm: null,
              swarmMessages: [],
              swarmInspect: null,
              swarmInspectTarget: null,
            }
          : {}),
      })
      if (runtimeTransitioned) {
        void this.refreshSwarm({ silent: true })
      }
      return
    }
    // Background turns (bg-* sessions) share this connection's event pipe
    // with a session_id tag. They must never touch the foreground fold —
    // the TUI applies the same filter. They drive the jobs chip instead.
    const eventSession = str(payload.session_id)
    if (eventSession && this.frame.currentId && eventSession !== this.frame.currentId) {
      this.onBackgroundEvent(type, payload, eventSession)
      return
    }
    switch (type) {
      case 'subagent_event': {
        const current = this.getSnapshot()
        const fleet = foldAgentEvent(current.fleet, payload)
        const status = str(payload.status).toLowerCase()
        const terminal = status === 'completed' || status === 'succeeded' || status === 'failed' || status === 'cancelled' || status === 'interrupted'
        if (terminal) this.patch({ fleet })
        else this.patchSoft({ fleet })
        this.syncAgentMembersFromFleet(fleet)
        return
      }
      case 'turn_begin': {
        this.turnCount += 1
        this.turnStartTotals = this.changeTotals()
        // A new attempt supersedes the previous failure card; retry clears
        // it explicitly, and any other submit means the human moved on.
        this.failure = null
        this.turnError = null
        // Defensive: an interrupted previous turn (daemon restart, dropped
        // turn_end) may have left live runs in the builder. Commit them as
        // closed blocks BEFORE the active fold renders, or they come back
        // as blinking streaming rows riding this new turn. No-op when the
        // previous turn ended cleanly.
        this.builder.finalize()
        // Current daemons echo the prompt as `text`; older ones said
        // `user_input`. Either way the local optimistic line may already
        // cover it — dedupe on content.
        const user = typeof payload.user_input === 'string'
          ? payload.user_input
          : typeof payload.text === 'string'
            ? payload.text
            : ''
        if (user) {
          const cleanUser = withoutTurnContext(user).trim()
          const unprefixed = /^user:\s+/i.test(cleanUser) ? cleanUser.replace(/^user:\s+/i, '').trim() : ''
          const matchesRecentSteer = Boolean(
            (cleanUser && this.recentSteerTexts.has(cleanUser))
            || (unprefixed && this.recentSteerTexts.has(unprefixed)),
          )
          const displayUser = unprefixed && matchesRecentSteer ? unprefixed : cleanUser
          this.lastUser = displayUser
          const last = this.frame.blocks.at(-1)
          const optimistic = this.optimisticSubmissions.get(this.sessionKey)
          const matchesOptimistic = Boolean(optimistic && (optimistic.text === user || optimistic.text === displayUser))
          const alreadyShown = matchesOptimistic && this.frame.blocks.some(block => block.kind === 'user' && block.id === optimistic!.id && (block.text === user || block.text === displayUser))
          if (matchesOptimistic && optimistic) {
            optimistic.acknowledged = true
            this.optimisticSubmissions.delete(this.sessionKey)
          }
          const alreadyAtTail = Boolean(
            last && last.kind === 'user' && (last.text === user || last.text === displayUser || (unprefixed && last.text === unprefixed)),
          )
          const alreadyQueuedNearTail = Boolean(
            (matchesRecentSteer || Boolean(unprefixed))
            && this.frame.blocks.slice(-4).some(block => block.kind === 'user' && (block.text === displayUser || (unprefixed && block.text === unprefixed))),
          )
          if (matchesRecentSteer) {
            this.recentSteerTexts.delete(displayUser)
            if (cleanUser) this.recentSteerTexts.delete(cleanUser)
          }
          // A prompt the harness wrote (goal round, monitor, schedule) is not
          // something the user said, so it is not shown.
          if (!alreadyShown && !harnessOriginOf(payload.origin) && !alreadyAtTail && !alreadyQueuedNearTail) {
            this.builder.pushUser(displayUser)
          }
        }
        this.preparingSubmissions.delete(this.sessionKey)
        // A late delta that landed after the previous turn_end sits in the
        // scratch runs; left alone it re-enters the stream as a "live" block
        // of THIS turn. Drain it committed before the tail opens.
        this.builder.finalize()
        // The new turn's first spawn opens a fresh agents card; the previous
        // card's committed copy keeps its terminal states.
        this.builder.closeAgentsCard()
        this.agentMembers.clear()
        this.startTurn()
        // First fleet read of the turn; agent-family tool calls below keep
        // the panel polling while spawns actually run.
        this.refreshFleet()
        return
      }
      case 'text_part': {
        this.setMetricPhase('llm', true)
        if (typeof payload.text === 'string' && payload.text && this.agentText.length < MAX_AGENT_TEXT_CHARS) {
          this.agentText += payload.text.slice(0, MAX_AGENT_TEXT_CHARS - this.agentText.length)
        }
        this.builder.push(type, payload)
        this.notify()
        return
      }
      case 'think_part':
      case 'tool_result':
      case 'notification':
        if (type === 'think_part') this.setMetricPhase('llm', true)
        if (type === 'tool_result' && payload.permitted !== false && !payload.error) {
          const todos = todosFromResult(payload.name, payload.return_value)
          if (todos !== null) this.patch({ todos })
        }
        // Goal tools change the goal mid-turn (create, resume, remove limits).
        // The card used to refresh only when the turn ended, so it went on
        // showing "blocked · Rounds 0/1" for a goal the agent had already
        // resumed with no limits.
        if (type === 'tool_result' && GOAL_WRITE_TOOLS.has(str(payload.name))) void this.refreshGoal()
        if (type === 'tool_result' && /(^|[.:/])(swarm|post_message|swarm_overview)$/i.test(str(payload.name))) {
          void this.refreshSwarm({ silent: true })
        }
        if (type === 'tool_result' && /(^|[.:])workflow$/i.test(str(payload.name))) this.foldWorkflowResult(payload.return_value ?? payload.result)
        if (type === 'tool_result' && payload.permitted !== false && !payload.error && Array.isArray(payload.display_blocks)) {
          for (const block of payload.display_blocks) {
            if (block?.type !== 'todo') continue
            const todos = todoItemsOf(block.items)
            if (todos !== null) this.patch({ todos })
          }
        }
        if (type === 'notification') {
          if (payload.category === 'reflection' && payload.type === 'proposal') void this.refreshLearning()
          if (payload.category === 'slash' && payload.type === 'result') {
            this.slashResult = { sessionKey: this.sessionKey, text: str(payload.body) || str(payload.message) }
          }
          const suggestion = skillSuggestionOf(payload.skill)
          if (suggestion) {
            this.patch({
              skillSuggestions: [
                ...this.getSnapshot().skillSuggestions.filter(row => row.skillName !== suggestion.skillName),
                suggestion,
              ].slice(-32),
            })
          }
          const body = str(payload.body) || str(payload.message)
          const severity = String(payload.severity ?? payload.level ?? 'info').toLowerCase()
          if (body && (severity.includes('error') || severity.includes('fatal'))) this.preparingSubmissions.delete(this.sessionKey)
          if (body && (severity.includes('error') || severity.includes('fatal')) && this.frame.turnActive) {
            // A failed turn announces itself as an error notification — never
            // as silence; text may still have streamed before the failure.
            this.turnError = body
          }
        }
        // Individual tool_result errors (e.g. "file does not exist", non-zero exit
        // codes, or retries) are normal tool feedback that the agent recovers from
        // mid-turn and already render on their tool row; they must never mark a
        // completed turn as failed.
        if (type === 'tool_result') {
          // A foreground agent's terminal status lands with its result:
          // refresh now, then let the poll die once nothing agent-side runs.
          const id = str(payload.tool_call_id)
          if (id) {
            this.pendingAgentCalls.delete(id)
            this.activeMetricTools.delete(id)
          }
          if (payload.permitted !== false && !payload.error && (!id || !this.foldedToolIds.has(id))) {
            const resultDiffs = parseStructuredDiffs(payload.diffs, payload.diff)
            const resultArgs = parseArgs(payload.arguments)
            if (resultDiffs.length > 1) {
              let foldedAny = false
              for (const entry of resultDiffs) {
                const entryStats = editStatsOf(payload.name, {}, entry, payload.locations)
                if (entryStats) {
                  foldedAny = true
                  this.foldChange(entryStats, {}, str(payload.name), entry)
                }
              }
              if (foldedAny && id) this.foldedToolIds.add(id)
            } else {
              const resultDiff = resultDiffs[0] ?? null
              const resultStats = editStatsOf(payload.name, resultArgs, resultDiff, payload.locations)
              if (resultStats) {
                if (id) this.foldedToolIds.add(id)
                this.foldChange(resultStats, resultArgs, str(payload.name), resultDiff)
              }
            }
          }
          const durationMs = Math.max(0, num(payload.duration_ms) ?? 0)
          const current = this.getSnapshot()
          this.patchSoft({
            toolDurationMs: current.toolDurationMs + durationMs,
            toolSteps: current.toolSteps + 1,
            ...(this.activeMetricTools.size === 0 && current.metricPhase !== 'llm'
              ? { metricPhase: 'llm' as const, metricPhaseStartedAt: Date.now() }
              : {}),
          })
          if (isAgentFamilyTool(payload.name)) {
            const reports = agentReceiptRows(payload.return_value ?? payload.result)
            const members = [...this.agentMembers.values()].filter(member => member.key.startsWith(`${id}:`))
            for (const [index, value] of reports.entries()) {
              const report = value && typeof value === 'object' ? value as Record<string, unknown> : {}
              const runtimeId = str(report.id) || str(report.conversation_id) || str(report.conversationId)
              if (!runtimeId || !str(report.status)) continue
              const member = members[index]
              const previous = this.getSnapshot().fleet.find(row => row.id === runtimeId)
              const title = str(report.title) || previous?.title || member?.title || runtimeId
              if (member) {
                this.agentMembers.set(member.key, {...member, runtimeId, title, status:agentStatusOf(str(report.status))})
              }
              const row: SessionRow = {...(previous ?? {id:runtimeId,key:runtimeId,age:'',current:false,kind:'subagent',turns:0,messages:0,cwd:'',untitled:false}),title,status:str(report.status),
                agentDetails:{filesRead:[],filesWritten:[],...previous?.agentDetails,summary:str(report.summary),error:str(report.error),model:str(report.model)||previous?.agentDetails?.model||'',baseAgent:str(report.agent_id)||member?.baseAgent||previous?.agentDetails?.baseAgent||'',goal:member?.prompt||previous?.agentDetails?.goal||'',requestKey:member?.key||previous?.agentDetails?.requestKey||"",lastReceiptAt:Date.now()}}
              this.patch({fleet:[...this.getSnapshot().fleet.filter(row=>row.id!==runtimeId),row]})
            }
            if (reports.length) this.builder.pushAgents([...this.agentMembers.values()])
            this.refreshFleet()
          }
          // A failed spawn never reached the manifest — the card's only
          // honest terminal signal is the result itself.
          const spawnError = toolFailureText(payload)
          if (id && spawnError && this.agentMembers.size > 0) {
            let touched = false
            for (const [key, member] of this.agentMembers) {
              if (!key.startsWith(`${id}:`) || member.runtimeId || member.status !== 'working') continue
              this.agentMembers.set(key, { ...member, status: 'failed', error: spawnError })
              touched = true
            }
            if (touched) this.builder.pushAgents([...this.agentMembers.values()])
          }
          if (this.pendingAgentCalls.size === 0 && ![...this.agentMembers.values()].some(m => m.status === 'working')) this.stopFleetPoll()
        }
        this.builder.push(type, payload)
        this.notify()
        return
      case 'tool_update': {
        const metricToolId = str(payload.tool_call_id) || str(payload.id)
        if (metricToolId) this.activeMetricTools.add(metricToolId)
        this.setMetricPhase('tool', true)
        this.builder.push('tool_update', payload)
        this.notify()
        return
      }
      case 'tool_call': {
        // The first computer-use step is when its macOS permissions matter.
        if (payload.name === 'computer_use' && !this.frame.computerUseWanted) this.patch({ computerUseWanted: true })
        const callDiffs = parseStructuredDiffs(payload.diffs, payload.diff)
        const parsedCallArgs = parseArgs(payload.arguments ?? payload.inputs)
        const metricToolId = str(payload.tool_call_id) || str(payload.id)
        if (callDiffs.length > 1) {
          let foldedAny = false
          for (const entry of callDiffs) {
            const entryStats = editStatsOf(payload.name, {}, entry, payload.locations)
            if (entryStats) {
              foldedAny = true
              this.foldChange(entryStats, {}, str(payload.name), entry)
            }
          }
          if (foldedAny && metricToolId) this.foldedToolIds.add(metricToolId)
        } else {
          const structuredDiff = callDiffs[0] ?? null
          const stats = editStatsOf(payload.name, parsedCallArgs, structuredDiff, payload.locations)
          if (stats) {
            if (metricToolId) this.foldedToolIds.add(metricToolId)
            this.foldChange(stats, parsedCallArgs, str(payload.name), structuredDiff)
          }
        }
        this.builder.push(type, payload)
        if (metricToolId) this.activeMetricTools.add(metricToolId)
        this.setMetricPhase('tool', true)
        // Agent-family calls move the fleet panel while they run — spawn
        // manifests persist inside tool execution, so refresh now AND keep
        // polling until the matching tool_result lands.
        if (isAgentFamilyTool(payload.name)) {
          const id = str(payload.tool_call_id) || str(payload.id)
          if (id) this.pendingAgentCalls.add(id)
          // The in-chat agents card is event-driven, not manifest-driven:
          // even when the spawn dies inside the daemon (provider down,
          // policy refusal) and subagent_snapshots stays empty, the batch
          // appears here — and the error result below marks it failed.
          if (id && isSpawnTool(payload.name)) {
            for (const member of spawnMembersOf(payload.name, payload.arguments, id)) {
              this.agentMembers.set(member.key, member)
            }
            this.builder.pushAgents([...this.agentMembers.values()])
          }
          this.refreshFleet()
          this.startFleetPoll()
        }
        this.notify()
        return
      }
      case 'steer_applied': {
        // The model has the steer now: it is no longer queued. Match by text,
        // one queued entry per applied steer, oldest first.
        const contents = Array.isArray(payload.contents) ? payload.contents.filter((value): value is string => typeof value === 'string') : []
        if (!contents.length) break
        let queue = this.queue
        let shownOther = false
        for (const content of contents) {
          const trimmed = content.trim()
          if (trimmed) this.recentSteerTexts.add(trimmed)
          const text = content.trim()
          let ours = false
          const index = queue.findIndex(item => item.text === text)
          if (index >= 0) { queue = [...queue.slice(0, index), ...queue.slice(index + 1)]; ours = true }
          for (const [id, steer] of this.pendingSteers) {
            // @-mentions expand the applied text, so a pending steer contained in it is still ours.
            if (steer.session === this.sessionKey && (steer.text === text || (steer.text !== '' && text.includes(steer.text)))) { this.pendingSteers.delete(id); ours = true; break }
          }
          // A steer from the phone, the Hub or another client: its acceptance echo went only to the
          // sender, so this is the first time this window hears of it. Show it where the model saw it.
          if (!ours && text) { this.builder.pushUser(text); shownOther = true }
        }
        if (queue !== this.queue) { this.queue = queue; this.patch({ queue }) }
        else if (shownOther) this.notify()
        return
      }
      case 'steer_input': {
        // Acceptance echo, not consumption: the daemon emits this the moment
        // it queues the text, and offers no "consumed" signal — steers drain
        // silently at the next step boundary. The mirror therefore lives
        // until turn_end, which is the one boundary we do observe.
        const content = str(payload.content)
        if (!content) return
        const id = str(payload.client_steer_id)
        const pending = id ? this.pendingSteers.get(id) : [...this.pendingSteers.values()].find(item => item.session === this.sessionKey && item.text === content && !item.echoed)
        if (pending) pending.echoed = true
        if (!pending?.shown) {
          const blockId = this.builder.pushUser(content, false, pending?.images ?? [])
          if (pending) {
            pending.shown = true
            pending.blockId = blockId
          }
          this.notify()
        }
        return
      }
      case 'session_title': {
        // Broadcast event: every session that earns a title (subagents,
        // other clients) lands here. Only ours may retitle the header.
        const title = str(payload.title)
        const forUs = !payload.session_id || str(payload.session_id) === this.frame.currentId
        if (title && forUs) this.patch({ currentTitle: title })
        return
      }
      case 'agent_preset_selected': {
        const selected = str(payload.agent_preset)
        if (selected) this.patch({ currentAgentPreset: selected })
        return
      }
      case 'channel_status': {
        // Daemon-wide broadcast (no session tag): any enable/disable — from
        // this shell, the TUI, or the webhook itself — keeps the panel true.
        this.applyChannelStatus(channelStatusFrom(payload))
        return
      }
      case 'status_update': {
        this.refreshProviderRelay()
        const current = this.getSnapshot()
        const patch: Record<string, unknown> = {}
        // Progress while compacting; the full status echo that follows
        // (it carries turn_count) is sent once the task is free again.
        if (payload.kind === 'compressing') patch.compacting = str(payload.text) || 'Compacting the conversation…'
        else if (current.compacting && payload.turn_count !== undefined) patch.compacting = undefined
        if (payload.kind === 'network_retry') patch.networkRetrying = true
        if (payload.kind === 'provider_ready') patch.networkRetrying = false
        if (payload.kind === 'plan') {
          const planTodos = todoItemsOf(payload.todos)
          if (planTodos !== null) patch.todos = planTodos
        }
        if (payload.kind === 'focus') {
          const focus = str(payload.text).trim()
          patch.liveFocus = focus || null
        }
        if (typeof payload.model === 'string' && payload.model) patch.model = payload.model
        if (typeof payload.context_tokens === 'number') patch.contextTokens = payload.context_tokens
        if (typeof payload.max_context === 'number') {
          patch.contextMax = payload.max_context > 0 ? payload.max_context : null
        }
        // The daemon sends two status_update shapes under one type: per-round
        // deltas (usage frames — no turn_count/llm_steps) and cumulative
        // session echoes (every settled turn, plus mode/config changes —
        // those carry turn_count/llm_steps). Only the first kind may be
        // added; adopting a cumulative echo as a delta inflated model time
        // roughly N-fold (the TUI adapter discriminates the same way).
        const cumulativeTelemetry = payload.llm_steps !== undefined || payload.turn_count !== undefined
        if (typeof payload.llm_duration_ms === 'number' && Number.isFinite(payload.llm_duration_ms)) {
          if (cumulativeTelemetry) {
            patch.llmDurationMs = Math.max(0, payload.llm_duration_ms)
            patch.llmSteps = Math.max(0, Math.trunc(typeof payload.llm_steps === 'number' ? payload.llm_steps : current.llmSteps))
          } else {
            patch.llmDurationMs = current.llmDurationMs + Math.max(0, payload.llm_duration_ms)
            patch.llmSteps = current.llmSteps + 1
            patch.metricPhase = 'llm'
            patch.metricPhaseStartedAt = Date.now()
          }
        }
        if (typeof payload.ttft_ms === 'number' && Number.isFinite(payload.ttft_ms)) {
          this.ttftSamples += 1
          this.ttftTotalMs += Math.max(0, payload.ttft_ms)
          patch.ttftMs = this.ttftTotalMs / this.ttftSamples
        }
        if (typeof payload.total_input_tokens === 'number' && Number.isFinite(payload.total_input_tokens)) {
          patch.inputTokens = Math.max(0, payload.total_input_tokens)
        } else if (typeof payload.input_tokens === 'number' && Number.isFinite(payload.input_tokens)) {
          patch.inputTokens = Math.max(0, payload.input_tokens)
        }
        if (typeof payload.total_output_tokens === 'number' && Number.isFinite(payload.total_output_tokens)) {
          patch.outputTokens = Math.max(0, payload.total_output_tokens)
        } else if (typeof payload.output_tokens === 'number' && Number.isFinite(payload.output_tokens)) {
          patch.outputTokens = Math.max(0, payload.output_tokens)
        }
        if (typeof payload.tokens_per_second === 'number' && Number.isFinite(payload.tokens_per_second)) {
          patch.tokensPerSecond = payload.tokens_per_second
        }
        const cacheReadTokens = num(payload.cache_read_tokens)
        const cacheWriteTokens = num(payload.cache_creation_tokens)
        const cumulativeInputTokens = num(payload.total_input_tokens) ?? num(payload.input_tokens)
        if (cacheReadTokens !== null) patch.cacheReadTokens = cacheReadTokens
        if (cacheWriteTokens !== null) patch.cacheWriteTokens = cacheWriteTokens
        // The daemon's rate uses only cache-instrumented calls. total_input_tokens
        // also includes uncached/unknown calls, so recomputing with it understates
        // the provider's reported cache hit rate on long sessions.
        if (typeof payload.cache_hit_rate === 'number' && Number.isFinite(payload.cache_hit_rate)) {
          patch.cacheHitRate = Math.max(0, Math.min(1, payload.cache_hit_rate))
        } else {
          const promptTokens = (cacheReadTokens ?? 0) + (cacheWriteTokens ?? 0) + (cumulativeInputTokens ?? 0)
          if ((cacheReadTokens !== null || cacheWriteTokens !== null) && cumulativeInputTokens !== null && promptTokens > 0) {
            patch.cacheHitRate = (cacheReadTokens ?? 0) / promptTokens
          }
        }
        // Live cost estimate rides the same echo the ctx counter uses.
        const statusCostUsd = costUsdFromPayload(payload)
        if (statusCostUsd !== null) patch.costUsd = statusCostUsd
        // Every status echo states the basis; its absence means per-token billing.
        if ('model' in payload) patch.costBasis = payload.cost_basis === 'subscription' ? 'subscription' : null
        // The daemon logs plan-mode flips; its status echo is authoritative.
        if (typeof payload.plan_mode === 'boolean') patch.planMode = payload.plan_mode
        if (payload.delegation_mode === 'off' || payload.delegation_mode === 'auto' || payload.delegation_mode === 'eager') patch.delegationMode = payload.delegation_mode
        // The daemon echoes effort flips here; session pinning means the
        // picker may not have been the one to change it.
        if (typeof payload.reasoning_effort === 'string' && payload.reasoning_effort) {
          patch.reasoningEffort = payload.reasoning_effort
        }
        // /permissions pins per session and echoes here — without this the
        // Permissions card's ✓ marker stays on the mode it booted with.
        if (typeof payload.permission_mode === 'string' && payload.permission_mode) {
          patch.permissionMode = payload.permission_mode
        }
        this.patch(patch)
        return
      }
      case 'usage': {
        const patch: Record<string, unknown> = {}
        const usedTokens = num(payload.context_tokens) ?? num(payload.used)
        const maxTokens = num(payload.max_context) ?? num(payload.size)
        if (usedTokens !== null) patch.contextTokens = Math.max(0, usedTokens)
        if (maxTokens !== null) patch.contextMax = maxTokens > 0 ? maxTokens : null
        const inputToks = num(payload.total_input_tokens) ?? num(payload.input_tokens)
        const outputToks = num(payload.total_output_tokens) ?? num(payload.output_tokens)
        if (inputToks !== null) patch.inputTokens = Math.max(0, inputToks)
        if (outputToks !== null) patch.outputTokens = Math.max(0, outputToks)
        const cacheRead = num(payload.cache_read_tokens)
        const cacheWrite = num(payload.cache_creation_tokens)
        if (cacheRead !== null) patch.cacheReadTokens = Math.max(0, cacheRead)
        if (cacheWrite !== null) patch.cacheWriteTokens = Math.max(0, cacheWrite)
        const usageCostUsd = costUsdFromPayload(payload)
        if (usageCostUsd !== null) patch.costUsd = usageCostUsd
        if (Object.keys(patch).length > 0) this.patchSoft(patch)
        return
      }
      case 'approval_request': {
        const id = str(payload.id) || str(payload.request_id)
        if (!id || this.frame.approval?.id === id) return
        const name = str(payload.tool_name) || str(payload.name) || str(payload.action)
        const autoAcceptAt = /^ExitPlanMode(Tool)?$/.test(name) && !this.isAntigravityConnected() ? Date.now() + 180_000 : undefined
        // The daemon attaches the real arguments as `inputs` (and older
        // builds as `arguments`). Keep them structured for the card to
        // render per tool; the flat description is only the last resort.
        const inputs = payload.inputs && typeof payload.inputs === 'object' && !Array.isArray(payload.inputs)
          ? payload.inputs as Record<string, unknown>
          : payload.arguments && typeof payload.arguments === 'object' && !Array.isArray(payload.arguments)
            ? payload.arguments as Record<string, unknown>
            : null
        const diffs = parseStructuredDiffs(payload.diffs, payload.diff)
        const diff = diffs[0] ?? null
        const options: ApprovalOption[] = Array.isArray(payload.options)
          ? payload.options
              .map(raw => {
                if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
                const row = raw as Readonly<Record<string, unknown>>
                const optionId = str(row.option_id) || str(row.optionId)
                if (!optionId) return null
                const warning = str(row.warning)
                return {
                  optionId,
                  name: str(row.name) || optionId,
                  kind: str(row.kind) || 'allow_once',
                  ...(warning ? { warning } : {}),
                }
              })
              .filter((opt): opt is ApprovalOption => opt !== null)
          : []
        const allowPermanent =
          typeof payload.allow_permanent === 'boolean'
            ? payload.allow_permanent
            : options.length > 0
              ? options.some(opt => opt.kind === 'allow_always')
              : undefined
        const description =
          str(payload.description) ||
          `${str(payload.name)} ${str(JSON.stringify(payload.arguments ?? ''))}`.trim()
        this.patch({
          approval: {
            id,
            ...(autoAcceptAt === undefined ? {} : { autoAcceptAt }),
            action: str(payload.action),
            description,
            ...(str(payload.tool_call_id) ? { toolCallId: str(payload.tool_call_id) } : {}),
            ...(str(payload.tool_name) || str(payload.name) ? { toolName: str(payload.tool_name) || str(payload.name) } : {}),
            ...(inputs ? { inputs } : {}),
            ...(str(payload.cwd) ? { cwd: str(payload.cwd) } : {}),
            ...(str(payload.reason) ? { reason: str(payload.reason) } : {}),
            ...(str(payload.permission_mode) ? { permissionMode: str(payload.permission_mode) } : {}),
            ...(allowPermanent !== undefined ? { allowPermanent } : {}),
            ...(diff ? { diff } : {}),
            ...(diffs.length > 0 ? { diffs } : {}),
            ...(options.length > 0 ? { options } : {}),
          },
          interactionHidden: false,
        })
        return
      }
      case 'question_request': {
        const requestId = str(payload.id)
        const mode: 'form' | 'url' | undefined =
          payload.mode === 'url' ? 'url' : payload.mode === 'form' ? 'form' : undefined
        const url = str(payload.url) || undefined
        const elicitationId = str(payload.elicitation_id) || str(payload.elicitationId) || undefined
        const message = str(payload.message) || undefined
        const rawItems = Array.isArray(payload.questions) ? payload.questions : []
        const parsedItems = rawItems.map(raw => {
          const item = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
          return {
            id: str(item.id) || 'answer',
            question: str(item.question),
            options: Array.isArray(item.options) ? item.options.filter(o => typeof o === 'string') : [],
            allowFreeform: item.allow_free_form !== false,
            ...(item.multi_select === true || item.multiSelect === true ? { multiSelect: true } : {}),
            ...(str(item.placeholder) ? { placeholder: str(item.placeholder) } : {}),
          }
        }).filter(item => item.question)
        const items = parsedItems.length > 0
          ? parsedItems
          : mode === 'url' && url
            ? [{
                id: 'url_consent',
                question: message || 'Open external link to continue',
                options: ['Open link & continue', 'Decline'],
                allowFreeform: false,
              }]
            : []
        if (!requestId || items.length === 0) return
        const question = {
          requestId,
          toolCallId: str(payload.tool_call_id),
          items,
          ...(mode ? { mode } : {}),
          ...(url ? { url } : {}),
          ...(elicitationId ? { elicitationId } : {}),
          ...(message ? { message } : {}),
        }
        this.patch({ question, interactionHidden: false })
        // A plan review captures the proposal as the session's working plan.
        if (mode !== 'url' && isPlanReview(question) && this.agentText.trim()) {
          this.capturePlan(this.agentText)
        }
        return
      }
      // Every window showing the request hears its answer, including one
      // answered from another window or the TUI: the card is dead from then on.
      case 'question_response': {
        const id = str(payload.id)
        if (this.frame.question?.requestId === id) this.patch({ question: null })
        return
      }
      case 'question_resolved': {
        const id = str(payload.id) || str(payload.request_id)
        const elicitationId = str(payload.elicitation_id) || str(payload.elicitationId)
        const current = this.frame.question
        if (
          current
          && ((id && current.requestId === id)
            || (elicitationId && current.elicitationId === elicitationId))
        ) {
          this.patch({ question: null })
        }
        return
      }
      case 'approval_response': {
        // Answered here or in another window on the same session: the card can
        // no longer be answered, so it must not keep saying it waits for you.
        const id = str(payload.request_id) || str(payload.id)
        if (id && this.frame.approval?.id === id) this.patch({ approval: null })
        return
      }
      case 'turn_end': {
        for (const [id, steer] of this.pendingSteers) if (steer.session === this.sessionKey) this.pendingSteers.delete(id)
        this.preparingSubmissions.delete(this.sessionKey)
        // Plan mode: whatever the agent reasoned toward in text is the plan
        // artifact — capture it before the buffer resets for the next turn.
        if (this.frame.planMode && this.agentText.trim()) this.capturePlan(this.agentText)
        this.agentText = ''
        this.builder.finalize()
        // `changes` is session-cumulative: a read-only turn after an editing
        // one must not stamp a checkpoint after its answer.
        const totals = this.changeTotals()
        if (totals.adds !== this.turnStartTotals.adds || totals.dels !== this.turnStartTotals.dels) {
          this.builder.pushCheckpoint(this.turnCount, totals)
        }
        const turnEndError = typeof payload.error === 'string' && payload.error.trim() ? payload.error.trim() : ''
        const stopReason = typeof payload.stop_reason === 'string' ? payload.stop_reason : ''
        const turnEndFailed = payload.failed === true || stopReason === 'provider_failed' || stopReason === 'turn_failed'
        const failureError = turnEndError
          || this.turnError
          || (turnEndFailed ? (stopReason === 'provider_failed' ? 'The provider request failed.' : 'Turn failed.') : '')
        if (failureError) {
          this.failure = { error: failureError, turn: this.turnCount, lastUser: this.lastUser }
        }
        this.turnError = null
        if (this.tick) {
          clearInterval(this.tick)
          this.tick = null
        }
        this.queue = []
        this.pendingAgentCalls.clear()
        this.activeMetricTools.clear()
        // Background children may still be working — the poll only stops
        // here when the card is already terminal; their terminal statuses
        // land on the committed card as the snapshots settle.
        if (![...this.agentMembers.values()].some(m => m.status === 'working')) this.stopFleetPoll()
        const elapsedSeconds = this.frame.turnStartedAt != null
          ? Math.max(this.frame.turnSeconds, Math.floor((Date.now() - this.frame.turnStartedAt) / 1000))
          : this.frame.turnSeconds
        this.patch({
          turnActive: false,
          liveFocus: null,
          turnSeconds: elapsedSeconds,
          turnStartedAt: null,
          turnFailed: this.failure !== null,
          failed: this.failure,
          queue: this.queue,
          metricPhase: null,
          metricPhaseStartedAt: null,
          // The daemon force-rejects any pending interaction when the turn
          // ends, so a card left on screen answers into a dead request and
          // pins the header on "needs input" for every later turn.
          approval: null,
          question: null,
        })
        void this.refreshGoal()
        this.refreshFleet()
        this.loadSkillSuggestions()
        this.loadCreatorTrace()
        void this.refreshSessions()
        return
      }
      case 'local_setup': {
        const job = localSetupJobView(payload.job)
        if (!job) break
        this.patch({ localSetup: { ...this.frame.localSetup, job } })
        if (job.state === 'done') void this.localSetupFinished()
        return
      }
      default:
        break
    }
    this.notify()
  }

  // ── Workspace folds ──────────────────────────────────────────────────

  private foldChange(
    stats: { path: string; adds: number; dels: number; isNew: boolean },
    args: Record<string, unknown>,
    name: string,
    structuredDiff?: StructuredDiffPayload | null,
  ): void {
    const previous = this.changes.get(stats.path)
    const argOld =
      typeof args.old_string === 'string'
        ? args.old_string
        : typeof args.Target === 'string'
          ? args.Target
          : typeof args.old_text === 'string'
            ? args.old_text
            : typeof args.oldText === 'string'
              ? args.oldText
              : undefined
    const argNew =
      typeof args.new_string === 'string'
        ? args.new_string
        : typeof args.Replacement === 'string'
          ? args.Replacement
          : typeof args.new_text === 'string'
            ? args.new_text
            : typeof args.newText === 'string'
              ? args.newText
              : typeof args.content === 'string'
                ? args.content
                : typeof args.Content === 'string'
                  ? args.Content
                  : undefined
    const hasArgEdit = argOld !== undefined || argNew !== undefined
    let oldLines: readonly string[] = []
    let newLines: readonly string[] = []
    if (hasArgEdit) {
      oldLines = argOld ? argOld.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
      newLines = argNew ? argNew.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
    } else if (structuredDiff) {
      if (structuredDiff.oldText && structuredDiff.newText) {
        const span = trimmedChangedLines(structuredDiff.oldText, structuredDiff.newText, 0)
        oldLines = span.before
        newLines = span.after
      } else {
        oldLines = structuredDiff.oldText ? structuredDiff.oldText.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
        newLines = structuredDiff.newText ? structuredDiff.newText.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
      }
    }
    const hunk: DiffLine[] = [{ kind: 'hunk', text: `@@ ${stats.path} @@` }]
    for (const line of oldLines) hunk.push({ kind: 'del', text: line })
    for (const line of newLines) hunk.push({ kind: 'add', text: line })
    const combinedHunks = [...(previous?.hunks ?? []), ...hunk]
    const boundedHunks = combinedHunks.length > MAX_DIFF_LINES_PER_FILE
      ? [
          ...combinedHunks.slice(0, MAX_DIFF_LINES_PER_FILE - 1),
          { kind: 'hunk' as const, text: `@@ … (${combinedHunks.length - MAX_DIFF_LINES_PER_FILE + 1} lines truncated) @@` },
        ]
      : combinedHunks
    const merged: DiffFile = {
      path: stats.path,
      adds: (previous?.adds ?? 0) + stats.adds,
      dels: (previous?.dels ?? 0) + stats.dels,
      isNew: previous?.isNew ?? (stats.isNew || normalizeToolName(name) === 'writefile'),
      hunks: boundedHunks,
      turn: this.turnCount,
    }
    this.changes.set(stats.path, merged)
    // A new edit during review reopens the surface — "Keep all" is an
    // acknowledgment, not a permanent verdict.
    this.patch({
      changes: [...this.changes.values()].sort((a, b) => a.path.localeCompare(b.path)),
      changesKept: false,
    })
  }

  /** Acknowledge the current change set (survives tab switches). */
  ackChanges(): void {
    this.patch({ changesKept: true })
  }

  private changeTotals(): { adds: number; dels: number } {
    let adds = 0
    let dels = 0
    for (const file of this.changes.values()) {
      adds += file.adds
      dels += file.dels
    }
    return { adds, dels }
  }

  private capturePlan(markdown: string): void {
    const cleaned = markdown.trim()
    this.planState = { markdown: cleaned, items: planItemsOf(cleaned), turn: this.turnCount }
    this.patch({ plan: this.planState })
  }

  private pushLog(type: string, payload: Readonly<Record<string, unknown>>): void {
    this.logRing.push({ id: this.seq++, turn: this.turnCount, type, summary: summarize(type, payload) })
    if (this.logRing.length > LOG_CAP) {
      this.logRing.splice(0, this.logRing.length - LOG_CAP)
    }
    this.logDirty = true
  }

  /**
   * Drop request state that a dropped socket has already invalidated.
   *
   * Safe on a same-session reconnect, unlike the session-scoped folds below:
   * every one of these flags gates a control (the reasoning picker, the
   * context popover, the permission buttons) whose reply died with the old
   * connection, and `setPermissionMode` early-returns while its flag is set.
   */
  private resetInFlightRequests(): void {
    this.patch({
      reasoningPickerOpen: false,
      reasoningLoading: false,
      contextMenuOpen: false,
      contextBreakdownLoading: false,
      permissionUpdating: false,
      permissionError: null,
    })
  }

  private resetWorkspaceFolds(): void {
    this.resetInFlightRequests()
    this.queue = []
    this.recentSteerTexts.clear()
    this.changes.clear()
    this.foldedToolIds.clear()
    this.turnStartTotals = { adds: 0, dels: 0 }
    this.logRing = []
    this.logView = []
    this.logDirty = false
    if (this.swarmInspectTimer) {
      clearTimeout(this.swarmInspectTimer)
      this.swarmInspectTimer = null
    }
    this.lastSwarmInspectAt = 0
    this.planState = null
    this.failure = null
    this.turnCount = 0
    this.agentText = ''
    this.lastUser = ''
    this.turnError = null
    this.ttftTotalMs = 0
    this.ttftSamples = 0
    this.activeMetricTools.clear()
    // Bound cross-session preview snippet cache so long desktop lifetimes stay lean.
    const snippetKeys = Object.keys(this.snippets)
    if (snippetKeys.length > SNIPPET_MAP_CAP) {
      const keep: Record<string, string> = {}
      for (const key of snippetKeys.slice(-SNIPPET_MAP_CAP)) keep[key] = this.snippets[key]!
      this.snippets = keep
    }
    // Agent bookkeeping belongs to the session that spawned it. Only turn_begin
    // used to clear these, so a session left with a still-working subagent
    // carried its members into the next session: the fleet poll kept running
    // against the NEW session while its stop condition read the OLD session's
    // members, and `syncAgentMembersFromFleet` could commit the old session's
    // agents card into the new session's transcript. Title collisions also
    // folded the new session's status updates onto the old session's entries.
    this.stopFleetPoll()
    this.agentMembers.clear()
    this.pendingAgentCalls.clear()
    this.patch({
      reasoningPickerOpen: false,
      reasoningLoading: false,
      reasoningLevels: [],
      reasoningDefault: '',
      reasoningNote: '',
      contextMenuOpen: false,
      contextBreakdownLoading: false,
      contextBreakdown: null,
      permissionUpdating: false,
      permissionError: null,
      fleet: [],
      queue: this.queue,
      changes: [],
      changesKept: false,
      log: this.logView,
      plan: null,
      todos: null,
      liveFocus: null,
      failed: null,
      turnCount: 0,
      turnFailed: false,
      turnSeconds: 0,
      turnStartedAt: null,
      ttftMs: null,
      tokensPerSecond: null,
      llmDurationMs: 0,
      llmSteps: 0,
      toolDurationMs: 0,
      toolSteps: 0,
      inputTokens: 0,
      outputTokens: 0,
      metricPhase: null,
      metricPhaseStartedAt: null,
      cacheHitRate: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      skillSuggestions: [],
      creatorTrace: [],
      swarm: null,
      swarmMessages: [],
      swarmInspect: null,
      swarmInspectTarget: '',
      swarmLoading: false,
      swarmError: null,
      swarmOpen: false,
      sideQuestionTurns: [],
      sideQuestionStreaming: false,
      sideQuestionError: null,
      sideQuestionOpen: false,
      acpBinding: null,
      acpConnecting: false,
      acpError: null,
      acpAuthRequired: false,
      acpAuthMethods: [],
      tab: 'activity',
    })
  }

  private setMetricPhase(phase: 'llm' | 'tool', soft = false): void {
    const current = this.getSnapshot()
    if (current.metricPhase === phase) return
    const update = { metricPhase: phase, metricPhaseStartedAt: Date.now() }
    if (soft) this.patchSoft(update)
    else this.patch(update)
  }

  private startTurn(startedAt?: number): void {
    this.activeMetricTools.clear()
    this.stopTick()
    const now = Date.now()
    const anchor = typeof startedAt === 'number' && Number.isFinite(startedAt) && startedAt > 0
      ? Math.min(now, startedAt)
      : now
    this.patch({
      turnActive: true, networkRetrying: false,
      liveFocus: null,
      turnFailed: false,
      turnSeconds: Math.max(0, Math.floor((now - anchor) / 1000)),
      turnStartedAt: anchor,
      metricPhase: 'llm',
      metricPhaseStartedAt: now,
    })
  }

  private stopTick(): void {
    if (this.tick) {
      clearInterval(this.tick)
      this.tick = null
    }
  }

  private fail(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error)
    if (/connect|socket|offline|disposed|closed|not ready|launch/i.test(message)) {
      this.wentOffline(error)
      return
    }
    this.builder.push('notification', { severity: 'error', message })
    this.patch({ error: message })
  }

  // ── Snapshot plumbing ────────────────────────────────────────────────

  private materializeFrame(): Snapshot {
    const before = this.frame
    const merge = this.pendingPatch
    this.pendingPatch = null
    this.frameDirty = false
    if (this.logDirty) {
      this.logView = this.logRing.slice()
      this.logDirty = false
    }
    const nextTurnActive = merge && typeof merge.turnActive === 'boolean' ? merge.turnActive : before.turnActive
    this.frame = this.frozen({
      ...before,
      ...(merge ?? {}),
      blocks: this.builder.snapshot(nextTurnActive),
      log: this.logView,
      turnCount: this.turnCount,
      snippets: this.snippets,
    })
    const pending = this.frame.approval
    const agyActive = this.isAntigravityConnected()
    const wasAgyActive = Boolean(before.acpBinding?.connected && before.acpBinding.provider === 'antigravity')
    if (pending !== before.approval || before.currentId !== this.frame.currentId || this.frame.connection === 'offline' || agyActive !== wasAgyActive) {
      clearTimeout(this.planTimer)
      this.planTimer = undefined
      if (pending !== before.approval && pending?.autoAcceptAt !== undefined && this.frame.connection !== 'offline' && !agyActive) {
        const { isCurrent } = this.captureSessionRequest()
        this.planTimer = setTimeout(() => {
          if (isCurrent() && this.frame.approval === pending && !this.isAntigravityConnected()) this.approvePlan(pending.id, 'accept-all')
        }, Math.max(0, pending.autoAcceptAt - Date.now()))
      }
    }
    if (merge && ('currentId' in merge || 'cwd' in merge)) {
      rememberSession((this.frame.storageScope ?? '') + this.frame.cwd, this.frame.currentId)
    }
    return this.frame
  }

  private patch(merge: Partial<Snapshot> | Record<string, unknown>): void {
    this.pendingPatch = this.pendingPatch ? { ...this.pendingPatch, ...merge } : { ...merge }
    this.frameDirty = true
    this.materializeFrame()
    // Structural change (connection, approval, tab, session, terminal events): render now.
    // Only the per-token / high-frequency streaming path below waits for the next frame.
    this.emit(true)
  }

  /** Coalesced state update for high-frequency streaming deltas (`btw_event`, `member_activity`, `subagent_event`, metrics). */
  private patchSoft(merge: Partial<Snapshot> | Record<string, unknown>): void {
    this.pendingPatch = this.pendingPatch ? { ...this.pendingPatch, ...merge } : { ...merge }
    this.frameDirty = true
    this.emit(false)
  }

  private notify(): void {
    this.frameDirty = true
    this.emit(false)
  }

  private frozen(value: Snapshot | Record<string, unknown>): Snapshot {
    return Object.freeze({ ...value, submissionPending: this.preparingSubmissions.has(String(value.sessionKey ?? this.sessionKey)) } as Snapshot)
  }

  /**
   * Coalesce streaming notifications onto frame boundaries (max ~20/s when
   * visible, ~4/s when hidden/occluded) with lazy snapshot materialization.
   *
   * `getSnapshot()` materializes lazily on demand so synchronous readers always
   * see current state, while bursts of 50–200 incoming tokens execute only one
   * `BlockBuilder.snapshot()` + `Object.freeze()` + React render per frame.
   * Terminal/structural events (`turn_end`, `turn_begin`, `error`, `user_send`,
   * approvals) call `emit(true)` to flush immediately.
   */
  private emitScheduled: ReturnType<typeof setTimeout> | null = null
  private emitRafId: number | null = null
  private emitToken = 0
  private lastEmitAt = 0

  private cancelScheduledEmit(): void {
    this.emitToken += 1
    if (this.emitScheduled !== null) {
      clearTimeout(this.emitScheduled)
      this.emitScheduled = null
    }
    if (this.emitRafId !== null) {
      if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.emitRafId)
      this.emitRafId = null
    }
  }

  private emit(immediate = false): void {
    if (immediate) {
      this.cancelScheduledEmit()
      if (this.frameDirty) this.materializeFrame()
      this.lastEmitAt = Date.now()
      for (const listener of this.listeners) listener()
      return
    }
    if (this.emitScheduled !== null || this.emitRafId !== null) return
    // No frames outside a page (headless tests, SSR): stay synchronous and deterministic.
    if (typeof requestAnimationFrame !== 'function') {
      if (this.frameDirty) this.materializeFrame()
      this.lastEmitAt = Date.now()
      for (const listener of this.listeners) listener()
      return
    }
    const frameMs = pageIsBackground() ? BACKGROUND_STREAM_FRAME_MS : STREAM_FRAME_MS
    const wait = Math.max(0, frameMs - (Date.now() - this.lastEmitAt))
    const token = ++this.emitToken
    const dispatch = (): void => {
      if (token !== this.emitToken) return
      this.emitScheduled = null
      this.emitRafId = null
      if (this.frameDirty) this.materializeFrame()
      this.lastEmitAt = Date.now()
      for (const listener of this.listeners) listener()
    }
    if (wait === 0 && !pageIsBackground()) {
      this.emitRafId = requestAnimationFrame(dispatch)
      return
    }
    this.emitScheduled = setTimeout(() => {
      if (token !== this.emitToken) return
      this.emitScheduled = null
      if (!pageIsBackground() && typeof requestAnimationFrame === 'function') {
        this.emitRafId = requestAnimationFrame(dispatch)
      } else {
        dispatch()
      }
    }, wait)
  }
}

/** Shape the store actually needs from the bridge (matches types.ts). */
export interface XerxesLike {
  /** The app's own GitHub-release update (main process); absent in test bridges. */
  appUpdate?(action: 'state' | 'check' | 'install' | 'skip' | 'dismiss' | 'open-release'): Promise<unknown>
  onAppUpdate?(handler: (state: unknown) => void): () => void
  getContextScope?(): Promise<string>
  getContextHost?(): Promise<string | null>
  getContexts?(): Promise<WorkspaceContext[]>
  activateContext?(id: number, sessionId?: string): Promise<void>
  call(method: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>
  /** Present on the real preload bridge; test bridges may omit it. */
  chooseWorkspace?(): Promise<unknown>
  getWorkspaceDirectories?(): Promise<string[]>
  removeWorkspace?(dir: string): Promise<string[]>
  restoreWorkspace?(dir: string): Promise<string[]>
  remote?(action: string, params?: Record<string, unknown>): Promise<unknown>
  openWorkspaceWindow?(dir?: string, resumeSessionId?: string, options?: { fresh?: boolean; existingOnly?: boolean; agentPreset?: string }): Promise<unknown>
  useWorkspace?(dir: string, resumeSessionId?: string): Promise<unknown>
  getWorkspace?(): Promise<string | null>
  getResumeSession?(): Promise<string | null>
  /** True for a window opened to start a fresh task: skip the remembered chat. */
  startsFresh?(): Promise<boolean>
  /** The agent preset that fresh window starts with (Creator mode opened mid-turn). */
  freshPreset?(): Promise<string | null>
}

export const store = new Store()
