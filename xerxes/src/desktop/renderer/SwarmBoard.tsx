// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Calm, flat-hairline Native Swarm surface for Xerxes Desktop and VS Code.
 *
 * Consumes Daemon v35 `swarm.*` RPCs and live `'swarm_event'` stream updates
 * through `store.ts` (or injected `actions` for isolated testing). Never
 * imports runtime `src/swarm/` modules directly.
 */

import {
  Fragment,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
  type RefObject,
} from 'react'

import { isExternalAcpBinding } from './AcpRuntimePicker.js'
import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { PickerLayer } from './Overlays.js'
import { store, type Snapshot } from './store.js'
import {
  formatInspectorToolArgs,
  formatInspectorToolResult,
  stripInspectorSystemEnvelope,
} from './swarmInspectorFormat.js'
import { SwarmGraph } from './SwarmGraph.js'
import { parseSwarmTimestampMs, type SwarmGraphModeInput } from './swarmGraphModel.js'
import {
  applySwarmMentionCandidate,
  filterSwarmMembers,
  formatSwarmShortTime,
  parseSwarmMessageBlocks,
  previewSwarmMessage,
  sortSwarmMembers,
  swarmChannelsOf,
  swarmMemberCounts,
  swarmMentionCandidates,
  tokenizeSwarmMessage,
  type SwarmBoardMessage,
  type SwarmChannelSnapshot,
  type SwarmMemberCounts,
  type SwarmMemberFilter,
  type SwarmMemberInspectResult,
  type SwarmMemberSnapshot,
  type SwarmStatusSnapshot,
  type SwarmTextToken,
} from './swarmState.js'
import type { ModelChoice } from './types.js'
import {
  resolveSimurghRole,
  SimurghSprite,
  type SimurghSpriteRole,
  type SimurghSpriteState,
} from './XerxesSprites.js'

export type SwarmSurfaceTab = 'overview' | 'board' | 'members' | 'graph'
export type SwarmBoardQuickFilter = 'all' | 'mentions' | 'done' | 'stall'

/** Number of trailing board posts rendered eagerly before progressive reveal (`Show earlier posts (N)`). */
export const SWARM_BOARD_TAIL_LIMIT = 50

export interface SwarmBoardActions {
  refresh(channel?: string): Promise<void> | void
  post(input: {
    readonly text: string
    readonly channel: string
    readonly asFeedback?: boolean
  }): Promise<boolean> | boolean
  kill(name: string): Promise<boolean> | boolean
  switchModel(name: string, model?: string): Promise<boolean> | boolean
  inspect?(name: string): Promise<SwarmMemberInspectResult | null> | SwarmMemberInspectResult | null
}

export interface SwarmBoardProps {
  readonly snap?: Snapshot
  readonly status?: SwarmStatusSnapshot | null
  readonly messages?: readonly SwarmBoardMessage[]
  readonly inspect?: SwarmMemberInspectResult | null
  readonly models?: readonly ModelChoice[]
  readonly online?: boolean
  readonly loading?: boolean
  readonly error?: string | null
  readonly actions?: SwarmBoardActions
  readonly initialTab?: SwarmSurfaceTab
  /** Optional initial mode (`'battlefield' | 'mentions' | 'hive'`) when opening the `Graph` tab. */
  readonly initialGraphMode?: SwarmGraphModeInput
  /** Initial channel view (`'*'` shows all channels; `'main'` or custom channel filters). */
  readonly initialChannel?: string
  /** Optional initial state for testing Narrator feedback composer. */
  readonly initialFeedbackOpen?: boolean
  readonly initialFeedbackDraft?: string
  readonly initialPostDraft?: string
  /** Optional initial member name with Switch Model expanded. */
  readonly initialModelSwitchMember?: string
  readonly initialModelDrafts?: Readonly<Record<string, string>>
  /** Optional initial inspected member name. */
  readonly initialInspectMember?: string
  /** Optional initial expanded long-message sequence numbers. */
  readonly initialExpandedSeqs?: readonly number[]
  /** Optional initial member filter/search state for large rosters (`> 8` members). */
  readonly initialMemberFilter?: SwarmMemberFilter
  readonly initialMemberQuery?: string
  /** Optional initial armed kill target (`'all'` or member name) for confirmation testing. */
  readonly initialArmedKillTarget?: string | null
  /** Optional initial split-view mode (`>= 780px`) for deterministic layout testing. */
  readonly initialSplitView?: boolean
}

export interface SwarmBoardViewProps {
  readonly status: SwarmStatusSnapshot | null
  readonly channels: readonly SwarmChannelSnapshot[]
  readonly members: readonly SwarmMemberSnapshot[]
  readonly counts: SwarmMemberCounts
  readonly messages: readonly SwarmBoardMessage[]
  readonly visibleMessages: readonly SwarmBoardMessage[]
  /** Optional count of trailing messages shown before collapsing earlier posts behind `Show earlier posts (N)`. */
  readonly visibleMessageLimit?: number
  readonly onShowEarlierMessages?: () => void
  /** Render board post & member rows via `React.memo` components (enabled by `<SwarmBoard>`, off for direct JSX tree tests). */
  readonly memoizeRows?: boolean
  readonly models: readonly ModelChoice[]
  readonly modelsLoading?: boolean
  readonly modelListId?: string
  readonly online: boolean
  readonly loading: boolean
  readonly busyAction: string
  readonly error: string
  readonly activeTab?: SwarmSurfaceTab
  readonly isSplitView?: boolean
  readonly boardRef?: RefObject<HTMLElement | null>
  readonly armedKillTarget?: string | null
  readonly onArmKill?: (target: string) => void
  readonly onCancelArmKill?: () => void
  readonly initialGraphMode?: SwarmGraphModeInput
  readonly selectedChannel: string
  readonly highlightedSeq: number | null
  readonly expandedSeqs?: ReadonlySet<number>
  readonly memberFilter?: SwarmMemberFilter
  readonly memberQuery?: string
  readonly boardQuery?: string
  readonly boardFilter?: SwarmBoardQuickFilter
  readonly boardFilterOpen?: boolean
  readonly feedbackOpen: boolean
  readonly feedbackDraft: string
  readonly postDraft: string
  readonly mentionIndex?: number
  readonly mentionsDismissed?: boolean
  readonly switchingMember: string
  readonly modelDrafts: Readonly<Record<string, string>>
  readonly inspectedMember?: string
  readonly inspectResult?: SwarmMemberInspectResult | null
  readonly inspectLoading?: boolean
  readonly feedRef?: RefObject<HTMLDivElement | null>
  readonly showJumpLatest?: boolean
  readonly unseenLatestCount?: number
  readonly onFeedScroll?: () => void
  readonly onJumpToLatest?: () => void
  readonly composerInputRef?: RefObject<HTMLTextAreaElement | null>
  readonly modelAnchorRef?: RefObject<HTMLElement | null>
  readonly onSelectTab?: (tab: SwarmSurfaceTab) => void
  readonly onRefresh: () => void
  readonly onStopAll: () => void
  readonly onToggleFeedback: () => void
  readonly onChangeFeedbackDraft: (value: string) => void
  readonly onSubmitFeedback: (event?: FormEvent) => void
  readonly onCloseFeedback: () => void
  readonly onSelectChannel: (channel: string) => void
  readonly onInsertMention: (name: string) => void
  readonly onPickMentionCandidate?: (name: string) => void
  readonly onMoveMentionIndex?: (delta: number, total: number) => void
  readonly onDismissMentions?: () => void
  readonly onInsertSeqRef: (seq: number) => void
  readonly onJumpToSeq: (seq: number) => void
  readonly onToggleExpandMessage?: (seq: number) => void
  readonly onChangeMemberFilter?: (filter: SwarmMemberFilter) => void
  readonly onChangeMemberQuery?: (query: string) => void
  readonly onChangeBoardQuery?: (query: string) => void
  readonly onChangeBoardFilter?: (filter: SwarmBoardQuickFilter) => void
  readonly onToggleBoardFilterOpen?: () => void
  readonly onKillMember: (name: string) => void
  readonly onInspectMember?: (name: string) => void
  readonly onCloseInspect?: () => void
  readonly onToggleModelSwitch: (member: SwarmMemberSnapshot) => void
  readonly onPickModel?: (member: SwarmMemberSnapshot, modelId: string) => void
  readonly onChangeModelDraft: (name: string, value: string) => void
  readonly onSubmitModelSwitch: (member: SwarmMemberSnapshot, event?: FormEvent) => void
  readonly onCancelModelSwitch: () => void
  readonly onChangePostDraft: (value: string) => void
  readonly onSubmitPost: (event?: FormEvent) => void
}

const DEFAULT_STORE_ACTIONS: SwarmBoardActions = {
  refresh(channel) {
    return store.refreshSwarm(channel ? { channel } : {})
  },
  post(input) {
    return store.postToSwarm(input)
  },
  kill(name) {
    return store.killSwarmMember(name)
  },
  switchModel(name, model) {
    return store.switchSwarmMemberModel(name, model)
  },
  inspect(name) {
    return store.inspectSwarmMember(name)
  },
}

export function swarmLiveBadgeState(
  online: boolean,
  status: Pick<SwarmStatusSnapshot, 'active'> | null,
  counts: Pick<SwarmMemberCounts, 'working' | 'waiting' | 'stopped'>,
): { readonly label: 'Live' | 'Paused' | 'Offline' | 'Idle'; readonly live: boolean; readonly ariaLabel: string } {
  if (!online) return { label: 'Offline', live: false, ariaLabel: 'Immortals offline' }
  if (status?.active) return { label: 'Live', live: true, ariaLabel: 'Live Immortals updates' }
  if (counts.stopped > 0 && counts.working === 0 && counts.waiting === 0) {
    return { label: 'Paused', live: false, ariaLabel: 'Immortals paused' }
  }
  return { label: 'Idle', live: false, ariaLabel: 'Immortals idle' }
}

/** True when the Board message feed is visible and should follow live posts. */
export function shouldEnableSwarmFeedScroll(
  activeTab: SwarmSurfaceTab,
  isSplitView: boolean,
  inspecting: boolean,
): boolean {
  if (inspecting) return false
  return activeTab === 'board' || (isSplitView && activeTab === 'members')
}

function stateLabelOf(member: SwarmMemberSnapshot): string {
  if (member.killed || member.state === 'killed') return 'killed'
  return member.state
}

function roleLabelOf(role: SwarmMemberSnapshot['role']): string {
  if (role === 'coordinator') return 'Coordinator'
  if (role === 'narrator') return 'Narrator'
  if (role === 'monitor') return 'Monitor'
  return 'worker'
}

function workspaceBadgeLabel(mode: SwarmMemberSnapshot['workspace']): string {
  if (mode === 'share') return 'worktree'
  if (mode === 'branch') return 'clone'
  return 'inherit'
}

type SwarmAvatarTint = 'accent' | 'violet' | 'emerald' | 'amber' | 'rose'

const WORKER_TINTS: readonly SwarmAvatarTint[] = ['violet', 'emerald', 'rose', 'amber', 'accent']

function memberAvatarTint(name: string, role: SwarmMemberSnapshot['role']): SwarmAvatarTint {
  if (role === 'coordinator') return 'accent'
  if (role === 'narrator') return 'violet'
  if (role === 'monitor') return 'amber'
  const resolved = resolveSimurghRole(name, role)
  if (resolved === 'verifier') return 'emerald'
  let hash = 0
  for (let i = 0; i < name.length; i += 1) {
    hash = ((hash * 31) + name.charCodeAt(i)) >>> 0
  }
  return WORKER_TINTS[hash % WORKER_TINTS.length] ?? 'violet'
}

function cleanSingleLineSnippet(raw: string, maxChars = 120): string {
  const firstLine = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .find(line => line.length > 0) ?? ''
  const stripped = firstLine
    .replace(/^(?:(?:@[a-z0-9_#-]+|\[done\]|\[stall\]|#+\s+)+\s*)+/i, '')
    .replace(/[*_`~]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  const chosen = stripped || firstLine.replace(/\s+/g, ' ').trim()
  if (chosen.length <= maxChars) return chosen
  return `${chosen.slice(0, maxChars - 1).trimEnd()}…`
}

interface SwarmMemberInsight {
  readonly latestPost: SwarmBoardMessage | null
  readonly latestSnippet: string
  readonly briefTitle: string
  readonly done: boolean
}

function buildMemberInsights(
  members: readonly SwarmMemberSnapshot[],
  messages: readonly SwarmBoardMessage[],
): ReadonlyMap<string, SwarmMemberInsight> {
  const latestByAgent = new Map<string, SwarmBoardMessage>()
  const briefByMember = new Map<string, string>()

  for (const message of messages) {
    latestByAgent.set(message.agent, message)
    if (message.mentions.length > 0) {
      const headline = cleanSingleLineSnippet(message.text, 96)
      if (headline) {
        for (const target of message.mentions) {
          if (!briefByMember.has(target)) briefByMember.set(target, headline)
        }
      }
    }
  }

  const insights = new Map<string, SwarmMemberInsight>()
  for (const member of members) {
    const latestPost = latestByAgent.get(member.name) ?? null
    const latestSnippet = latestPost ? cleanSingleLineSnippet(latestPost.text, 140) : ''
    const briefTitle =
      briefByMember.get(member.name) ||
      (member.current_activity ? cleanSingleLineSnippet(member.current_activity, 96) : '') ||
      (member.group ? `${member.group} · @${member.name}` : '')
    const postedDone = Boolean(latestPost && /\[done\]/i.test(latestPost.text))
    const done =
      member.killed ||
      member.state === 'killed' ||
      (postedDone && member.state !== 'working')
    insights.set(member.name, { latestPost, latestSnippet, briefTitle, done })
  }
  return insights
}

function messageRecipientLabels(message: SwarmBoardMessage): readonly string[] {
  if (message.broadcast) {
    const extras = message.mentions.filter(name => name !== 'all').slice(0, 2).map(name => `@${name}`)
    return ['@all', ...extras]
  }
  return message.mentions.slice(0, 3).map(name => `@${name}`)
}

function matchesBoardQuery(
  message: SwarmBoardMessage,
  query: string,
  filter: SwarmBoardQuickFilter,
): boolean {
  if (filter === 'mentions' && message.mentions.length === 0 && !message.broadcast) return false
  if (filter === 'done' && !/\[done\]/i.test(message.text)) return false
  if (filter === 'stall' && !/\[stall\]/i.test(message.text)) return false
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  if (`#${message.seq}` === needle) return true
  return (
    message.agent.toLowerCase().includes(needle) ||
    message.channel.toLowerCase().includes(needle) ||
    message.text.toLowerCase().includes(needle) ||
    message.mentions.some(name => name.toLowerCase().includes(needle))
  )
}

function SwarmAvatarTile({
  name,
  role,
  state,
  size = 'md',
  showDot = false,
  dotState,
  showSleep = false,
  active = false,
  className = '',
}: {
  readonly name: string
  readonly role: SwarmMemberSnapshot['role']
  readonly state: SimurghSpriteState
  readonly size?: 'sm' | 'md' | 'lg' | 'tile'
  readonly showDot?: boolean
  readonly dotState?: string
  readonly showSleep?: boolean
  readonly active?: boolean
  readonly className?: string
}): ReactElement {
  const spriteRole: SimurghSpriteRole = resolveSimurghRole(name, role)
  const tint = memberAvatarTint(name, role)
  const iconSize = size === 'sm' ? 15 : size === 'lg' ? 24 : size === 'tile' ? 22 : 20
  return (
    <span
      className={`swarm-avatar-tile swarm-avatar-tile--${size}${active ? ' is-active' : ''}${className ? ` ${className}` : ''}`}
      data-tint={tint}
      data-state={dotState ?? state}
    >
      {showSleep && (
        <span className="swarm-avatar-tile__sleep" aria-hidden="true">
          z
        </span>
      )}
      <SimurghSprite
        role={spriteRole}
        state={state}
        size={iconSize}
        animate={state === 'working'}
        className="swarm-member__sprite"
      />
      {showDot && (
        <span
          className="swarm-member__dot"
          data-state={dotState ?? state}
          aria-hidden="true"
        />
      )}
    </span>
  )
}

interface MessageRenderOptions {
  readonly onMentionClick: (name: string) => void
  readonly onChannelClick: (channel: string) => void
  readonly onJumpToSeq: (seq: number) => void
}

interface InlineFormatFlags {
  readonly allowBold: boolean
  readonly allowItalic: boolean
  readonly allowStrike: boolean
}

const ALL_INLINE_FORMATS: InlineFormatFlags = { allowBold: true, allowItalic: true, allowStrike: true }

function renderMessageTokens(
  text: string,
  options: MessageRenderOptions,
  formats: InlineFormatFlags = ALL_INLINE_FORMATS,
): ReactElement[] {
  const tokens: readonly SwarmTextToken[] = tokenizeSwarmMessage(text, formats)
  return tokens.map((token, index) => {
    switch (token.kind) {
      case 'code':
        return (
          <code
            key={index}
            className="swarm-msg__code"
            style={{ fontFamily: 'var(--mono)' }}
          >
            {token.code}
          </code>
        )
      case 'bold':
        return (
          <strong key={index} style={{ color: 'var(--x-title)', fontWeight: 600 }}>
            {renderMessageTokens(token.text, options, { ...formats, allowBold: false })}
          </strong>
        )
      case 'italic':
        return (
          <em key={index} className="swarm-msg__em">
            {renderMessageTokens(token.text, options, { ...formats, allowItalic: false })}
          </em>
        )
      case 'strike':
        return (
          <del key={index} className="swarm-msg__del">
            {renderMessageTokens(token.text, options, { ...formats, allowStrike: false })}
          </del>
        )
      case 'done':
        return (
          <span key={index} className="swarm-badge swarm-badge--done">
            {token.raw}
          </span>
        )
      case 'stall':
        return (
          <span key={index} className="swarm-badge swarm-badge--stall">
            {token.raw}
          </span>
        )
      case 'broadcast':
        return (
          <span key={index} className="swarm-token swarm-token--broadcast">
            {token.raw}
          </span>
        )
      case 'channel_mention':
        return (
          <button
            key={index}
            type="button"
            className="swarm-token swarm-token--channel"
            title={`Filter channel #${token.channel}`}
            onClick={() => options.onChannelClick(token.channel)}
          >
            {token.raw}
          </button>
        )
      case 'mention':
        return (
          <button
            key={index}
            type="button"
            className="swarm-token swarm-token--mention"
            title={`Mention @${token.name} in composer`}
            onClick={() => options.onMentionClick(token.name)}
          >
            {token.raw}
          </button>
        )
      case 'seq_ref':
        return (
          <button
            key={index}
            type="button"
            className="swarm-token swarm-token--seq"
            aria-label={`Jump to message #${token.seq}`}
            title={`Jump to message #${token.seq}`}
            onClick={() => options.onJumpToSeq(token.seq)}
          >
            {token.raw}
          </button>
        )
      case 'text':
      default:
        return <Fragment key={index}>{token.raw}</Fragment>
    }
  })
}

function renderSwarmMessageText(text: string, options: MessageRenderOptions): ReactElement[] {
  const blocks = parseSwarmMessageBlocks(text)
  if (blocks.length === 0) return []
  if (blocks.length === 1 && blocks[0]?.kind === 'paragraph') {
    return renderMessageTokens(blocks[0].text, options)
  }
  return blocks.map((block, index) => {
    const isLast = index === blocks.length - 1
    switch (block.kind) {
      case 'heading': {
        const Tag = (block.level <= 2 ? 'h3' : 'h4') as 'h3' | 'h4'
        return (
          <Tag
            key={index}
            className="swarm-msg__heading"
            data-level={block.level}
          >
            {renderMessageTokens(block.text, options)}
          </Tag>
        )
      }
      case 'list':
        return (
          <ul
            key={index}
            className="swarm-msg__list"
            style={{
              margin: isLast ? '4px 0 0' : '4px 0 6px',
              paddingLeft: '18px',
              whiteSpace: 'normal',
            }}
          >
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex} style={{ margin: '2px 0' }}>
                {renderMessageTokens(item, options)}
              </li>
            ))}
          </ul>
        )
      case 'ordered_list':
        return (
          <ol
            key={index}
            className="swarm-msg__list swarm-msg__list--ordered"
            start={block.start !== 1 ? block.start : undefined}
            style={{
              margin: isLast ? '4px 0 0' : '4px 0 6px',
              paddingLeft: '20px',
              whiteSpace: 'normal',
            }}
          >
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex} style={{ margin: '2px 0' }}>
                {renderMessageTokens(item, options)}
              </li>
            ))}
          </ol>
        )
      case 'code':
        return (
          <pre key={index} className="swarm-msg__pre" data-lang={block.language || undefined}>
            <code>{block.code}</code>
          </pre>
        )
      case 'quote':
        return (
          <blockquote key={index} className="swarm-msg__quote">
            {renderMessageTokens(block.text, options)}
          </blockquote>
        )
      case 'hr':
        return <hr key={index} className="swarm-msg__hr" />
      case 'table':
        return (
          <div key={index} className="swarm-msg__tablewrap">
            <table className="swarm-msg__table">
              <thead>
                <tr>
                  {block.headers.map((cell, cellIndex) => (
                    <th key={cellIndex}>{renderMessageTokens(cell, options)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{renderMessageTokens(cell, options)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      case 'paragraph':
      default:
        return (
          <p
            key={index}
            className="swarm-msg__para"
            style={{ margin: isLast ? 0 : '0 0 6px' }}
          >
            {renderMessageTokens(block.text, options)}
          </p>
        )
    }
  })
}

type InspectorToolCategory = 'read' | 'edit' | 'run' | 'search' | 'swarm' | 'web' | 'tool'

function inspectorToolCategory(toolName: string): {
  readonly kind: InspectorToolCategory
  readonly label: string
} {
  const lower = toolName.toLowerCase()
  if (lower.includes('swarm') || lower.includes('immortal') || lower.includes('board') || lower.includes('message')) {
    return { kind: 'swarm', label: 'Immortals' }
  }
  if (lower.includes('read') || lower.includes('view') || lower.includes('cat')) {
    return { kind: 'read', label: 'Read' }
  }
  if (lower.includes('edit') || lower.includes('write') || lower.includes('patch') || lower.includes('replace')) {
    return { kind: 'edit', label: 'Edit' }
  }
  if (lower.includes('grep') || lower.includes('glob') || lower.includes('find') || lower.includes('search')) {
    return { kind: 'search', label: 'Search' }
  }
  if (lower.includes('bash') || lower.includes('exec') || lower.includes('run') || lower.includes('shell') || lower.includes('cmd')) {
    return { kind: 'run', label: 'Run' }
  }
  if (lower.includes('web') || lower.includes('fetch') || lower.includes('http') || lower.includes('browser')) {
    return { kind: 'web', label: 'Web' }
  }
  return { kind: 'tool', label: 'Tool' }
}

function formatInspectorDuration(startedAt: string, finishedAt?: string): string {
  if (!finishedAt) return ''
  const start = parseSwarmTimestampMs(startedAt)
  const end = parseSwarmTimestampMs(finishedAt)
  if (start === null || end === null) return ''
  const ms = Math.max(0, end - start)
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`
  const mins = Math.floor(ms / 60_000)
  const secs = Math.round((ms % 60_000) / 1000)
  return `${mins}m ${String(secs).padStart(2, '0')}s`
}

function summarizeToolArgs(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
    try {
      const parsed: unknown = JSON.parse(trimmed)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        const record = parsed as Record<string, unknown>
        const primary =
          record.file_path ??
          record.path ??
          record.command ??
          record.pattern ??
          record.query ??
          record.url ??
          record.name ??
          record.text
        if (typeof primary === 'string' && primary.trim()) {
          return primary.trim().slice(0, 96)
        }
        const pairs = Object.entries(record)
          .slice(0, 3)
          .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
        if (pairs.length > 0) return pairs.join(' · ').slice(0, 96)
      }
    } catch {
      // Fall back to trimmed preview.
    }
  }
  return trimmed.slice(0, 96)
}

/** Deduplicate `models` by ID and ensure the member's current model appears when missing. */
export function resolveSwarmModelChoices(
  models: readonly ModelChoice[],
  member: SwarmMemberSnapshot,
): { readonly choices: readonly ModelChoice[]; readonly activeId: string } {
  const seen = new Set<string>()
  const choices: ModelChoice[] = []
  for (const choice of models) {
    const id = choice.id.trim()
    if (!id) continue
    const key = id.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    choices.push({ ...choice, id })
  }

  const currentRaw = member.model.trim()
  const currentLower = currentRaw.toLowerCase()
  const displayLower = member.model_display_name.trim().toLowerCase()

  let activeId = ''
  if (currentLower) {
    const exact = choices.find(c => c.id.toLowerCase() === currentLower)
    if (exact) {
      activeId = exact.id
    } else {
      const byBareOrLabel = choices.find(
        c =>
          (c.id.split('/').pop() ?? c.id).toLowerCase() === currentLower ||
          (displayLower && c.label?.trim().toLowerCase() === displayLower),
      )
      if (byBareOrLabel) {
        activeId = byBareOrLabel.id
      } else {
        const provider = currentRaw.includes('/') ? currentRaw.split('/')[0]! : ''
        const fallbackChoice: ModelChoice = {
          id: currentRaw,
          provider,
          ...(member.model_display_name && member.model_display_name !== currentRaw
            ? { label: member.model_display_name }
            : {}),
        }
        choices.unshift(fallbackChoice)
        activeId = currentRaw
      }
    }
  }

  return { choices, activeId }
}

export interface SwarmModelPickerProps {
  readonly member: SwarmMemberSnapshot
  readonly models: readonly ModelChoice[]
  readonly modelsLoading?: boolean
  readonly online: boolean
  readonly busy: boolean
  readonly customDraft: string
  readonly anchorRef?: RefObject<HTMLElement | null>
  readonly onPickModel: (modelId: string) => void
  readonly onChangeCustomDraft: (value: string) => void
  readonly onSubmitCustom: (event?: FormEvent) => void
  readonly onClose: () => void
}

/**
 * Sleek floating Vibes model popover for switching an Immortals member's model.
 * Single-click picks immediately; optional filter bar appears when > 6 models;
 * quiet custom model ID footer preserves manual entry via Enter.
 */
export function SwarmModelPicker({
  member,
  models,
  modelsLoading = false,
  online,
  busy,
  customDraft,
  anchorRef,
  onPickModel,
  onChangeCustomDraft,
  onSubmitCustom,
  onClose,
}: SwarmModelPickerProps): ReactElement {
  const [query, setQuery] = useState('')
  const popoverRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const { choices, activeId } = useMemo(
    () => resolveSwarmModelChoices(models, member),
    [models, member],
  )
  const showSearch = choices.length > 6
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return choices
    return choices.filter(
      choice =>
        choice.id.toLowerCase().includes(needle) ||
        (choice.label?.toLowerCase().includes(needle) ?? false) ||
        choice.provider.toLowerCase().includes(needle),
    )
  }, [choices, query])

  useLayoutEffect(() => {
    const popup = popoverRef.current
    const anchor = anchorRef?.current
    if (!popup || !anchor || typeof window === 'undefined') return
    const place = (): void => {
      const rect = anchor.getBoundingClientRect()
      popup.style.position = 'fixed'
      popup.style.right = 'auto'
      popup.style.bottom = 'auto'
      popup.style.maxWidth = 'calc(100vw - 24px)'
      popup.style.maxHeight = `${Math.max(160, window.innerHeight - 24)}px`
      const width = popup.getBoundingClientRect().width || 280
      const height = popup.getBoundingClientRect().height || 240
      const left = Math.max(12, Math.min(rect.right - width, window.innerWidth - width - 12))
      const belowTop = rect.bottom + 6
      const fitsBelow = belowTop + height <= window.innerHeight - 12
      const aboveTop = rect.top - height - 6
      const top = fitsBelow
        ? belowTop
        : aboveTop >= 12
          ? aboveTop
          : Math.max(12, Math.min(belowTop, window.innerHeight - height - 12))
      popup.style.left = `${left}px`
      popup.style.top = `${top}px`
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    observer?.observe(popup)
    observer?.observe(anchor)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [anchorRef, filtered.length])

  useEffect(() => {
    if (showSearch) searchInputRef.current?.focus()
    else popoverRef.current?.focus()
  }, [showSearch])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const onEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        onClose()
        anchorRef?.current?.focus()
      }
    }
    window.addEventListener('keydown', onEscape, true)
    return () => window.removeEventListener('keydown', onEscape, true)
  }, [onClose, anchorRef])

  return (
    <PickerLayer>
      <div
        className="backdrop backdrop--clear"
        onClick={() => {
          onClose()
          anchorRef?.current?.focus()
        }}
      />
      <div
        ref={popoverRef}
        className="swarm-model-picker"
        role="dialog"
        aria-label={`Switch model for @${member.name}`}
        tabIndex={-1}
        onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
          if (event.key === 'Escape') {
            event.preventDefault()
            event.stopPropagation()
            onClose()
            anchorRef?.current?.focus()
          }
        }}
      >
        <header className="swarm-model-picker__head">
          <span className="swarm-model-picker__title">Switch model</span>
          <span className="swarm-model-picker__member">@{member.name}</span>
          <button
            type="button"
            className="swarm-model-picker__close"
            aria-label="Close model picker"
            onClick={() => {
              onClose()
              anchorRef?.current?.focus()
            }}
          >
            <Icon name="close" size={11} />
          </button>
        </header>

        {showSearch && (
          <label className="swarm-model-picker__search">
            <Icon name="search" size={12} />
            <input
              ref={searchInputRef}
              type="text"
              className="swarm-model-picker__filter"
              aria-label="Filter models"
              placeholder="Filter models…"
              spellCheck={false}
              value={query}
              onChange={event => setQuery(event.target.value)}
            />
          </label>
        )}

        <div
          className="swarm-model-picker__list"
          role="listbox"
          aria-label={`Available models for ${member.name}`}
        >
          {modelsLoading && models.length === 0 && (
            <p className="swarm-model-picker__empty" role="status">
              Loading models…
            </p>
          )}
          {filtered.length === 0 ? (
            <p className="swarm-model-picker__empty">
              {query.trim()
                ? 'No matching models — enter a custom ID below.'
                : 'No discovered models — enter a custom ID below.'}
            </p>
          ) : (
            filtered.map(choice => {
              const isActive = choice.id === activeId
              const displayName =
                choice.label?.trim() || (choice.id.split('/').pop() || choice.id)
              const providerTag =
                choice.provider.trim() ||
                (choice.id.includes('/') ? choice.id.split('/')[0]! : '')
              return (
                <button
                  key={choice.id}
                  type="button"
                  role="option"
                  aria-selected={isActive}
                  className={`swarm-model-picker__item${isActive ? ' is-active' : ''}`}
                  title={choice.id}
                  disabled={!online || busy}
                  onClick={() => onPickModel(choice.id)}
                >
                  <span className="swarm-model-picker__check" aria-hidden="true">
                    {isActive ? <Icon name="check" size={12} /> : null}
                  </span>
                  <span className="swarm-model-picker__name">{displayName}</span>
                  {providerTag && (
                    <span className="swarm-model-picker__provider">{providerTag}</span>
                  )}
                </button>
              )
            })
          )}
        </div>

        <form
          className="swarm-model-picker__custom"
          onSubmit={event => {
            event.preventDefault()
            if (!online || busy || !customDraft.trim()) return
            onSubmitCustom(event)
          }}
        >
          <input
            type="text"
            className="swarm-model-picker__custom-input"
            aria-label={`Target model for ${member.name}`}
            placeholder="Custom model ID…"
            spellCheck={false}
            value={customDraft}
            disabled={!online || busy}
            onChange={event => onChangeCustomDraft(event.target.value)}
          />
          <button
            type="submit"
            className="swarm-model-picker__custom-submit"
            aria-label={`Apply model for ${member.name}`}
            title="Switch to custom model ID"
            disabled={!online || busy || !customDraft.trim()}
          >
            <Icon name="enter" size={12} />
          </button>
        </form>
      </div>
    </PickerLayer>
  )
}

export interface ScrollMetrics {
  readonly scrollTop: number
  readonly clientHeight: number
  readonly scrollHeight: number
}

/** True when a scroll container is within `thresholdPx` of its bottom edge (or has no overflow). */
export function isNearScrollBottom(metrics: ScrollMetrics, thresholdPx = 48): boolean {
  if (metrics.scrollHeight <= metrics.clientHeight + 1) return true
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= thresholdPx
}

/** Track unseen items arriving while scrolled away from the bottom; clears immediately when pinned. */
export function nextUnseenCount(
  currentUnseen: number,
  previousCount: number,
  nextCount: number,
  pinnedAtBottom: boolean,
): number {
  if (pinnedAtBottom) return 0
  const delta = nextCount - previousCount
  return delta > 0 ? currentUnseen + delta : currentUnseen
}

function scrollElementToBottom(el: HTMLElement | null, behavior: ScrollBehavior): void {
  if (!el) return
  if (typeof el.scrollTo === 'function') {
    el.scrollTo({ top: el.scrollHeight, behavior })
  } else {
    el.scrollTop = el.scrollHeight
  }
}

/**
 * Sticky auto-follow hook for live scrollable feeds (Immortals board & worker inspector).
 * Follows new entries automatically while pinned at the bottom, never yanks the user
 * when scrolled up to read history, and tracks how many unseen items arrived since scrolling up.
 */
function useFollowLatestScroll(options: {
  readonly ref: RefObject<HTMLDivElement | null>
  readonly itemCount: number
  readonly resetKey: string
  readonly enabled?: boolean
  readonly thresholdPx?: number
  readonly initialAtBottom?: boolean
  readonly initialUnseenCount?: number
}): {
  readonly atBottom: boolean
  readonly unseenCount: number
  readonly handleScroll: () => void
  readonly jumpToLatest: (behavior?: ScrollBehavior) => void
  readonly pauseFollow: () => void
} {
  const {
    ref,
    itemCount,
    resetKey,
    enabled = true,
    thresholdPx = 48,
    initialAtBottom = true,
    initialUnseenCount = 0,
  } = options
  const [atBottom, setAtBottom] = useState<boolean>(initialAtBottom)
  const [unseenCount, setUnseenCount] = useState<number>(initialUnseenCount)
  const atBottomRef = useRef<boolean>(initialAtBottom)
  const prevCountRef = useRef<number>(itemCount)
  const prevResetKeyRef = useRef<string>(resetKey)
  const smoothScrollUntilRef = useRef<number>(0)
  const lastScrollTopRef = useRef<number>(0)

  const handleScroll = useCallback((): void => {
    const el = ref.current
    if (!el) return
    const maxTop = Math.max(0, el.scrollHeight - el.clientHeight)
    const clampedTop = Math.min(Math.max(0, el.scrollTop), maxTop)
    const clampedPrev = Math.min(Math.max(0, lastScrollTopRef.current), maxTop)
    lastScrollTopRef.current = clampedTop
    // When there is no scroll overflow or we are resting on (or elastically past)
    // the bottom edge, stay pinned — never treat macOS rubber-band bounce as scrolling up.
    if (maxTop <= 2 || maxTop - clampedTop <= 2) {
      smoothScrollUntilRef.current = 0
      atBottomRef.current = true
      setAtBottom(true)
      setUnseenCount(0)
      return
    }
    if (clampedTop < clampedPrev - 1) {
      smoothScrollUntilRef.current = 0
      atBottomRef.current = false
      setAtBottom(false)
      return
    }
    const near = isNearScrollBottom({ scrollTop: clampedTop, clientHeight: el.clientHeight, scrollHeight: el.scrollHeight }, thresholdPx)
    if (near && (clampedTop >= clampedPrev || atBottomRef.current)) {
      smoothScrollUntilRef.current = 0
      atBottomRef.current = true
      setAtBottom(true)
      setUnseenCount(0)
      return
    }
    // Ignore intermediate frames while smoothly animating to bottom, unless the user scrolls back up.
    if (Date.now() < smoothScrollUntilRef.current && clampedTop >= clampedPrev) {
      return
    }
    smoothScrollUntilRef.current = 0
    atBottomRef.current = false
    setAtBottom(false)
  }, [ref, thresholdPx])

  const jumpToLatest = useCallback((behavior: ScrollBehavior = 'smooth'): void => {
    atBottomRef.current = true
    setAtBottom(true)
    setUnseenCount(0)
    if (behavior === 'smooth') {
      smoothScrollUntilRef.current = Date.now() + 700
    }
    const run = (): void => {
      const el = ref.current
      if (el) lastScrollTopRef.current = el.scrollTop
      scrollElementToBottom(el, behavior)
    }
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run)
    else run()
  }, [ref])

  const pauseFollow = useCallback((): void => {
    atBottomRef.current = false
    setAtBottom(false)
  }, [])

  // Reset stickiness and scroll straight to the tail when switching channels / inspected member.
  useLayoutEffect(() => {
    if (!enabled) return
    const snapBottom = (): void => {
      scrollElementToBottom(ref.current, 'auto')
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(() => {
          if (atBottomRef.current) scrollElementToBottom(ref.current, 'auto')
        })
      }
    }
    const resetChanged = prevResetKeyRef.current !== resetKey
    if (resetChanged) {
      prevResetKeyRef.current = resetKey
      prevCountRef.current = itemCount
      atBottomRef.current = true
      setAtBottom(true)
      setUnseenCount(0)
      snapBottom()
      return
    }
    const prevCount = prevCountRef.current
    prevCountRef.current = itemCount
    if (atBottomRef.current) {
      if (itemCount !== prevCount) {
        snapBottom()
      }
      return
    }
    if (itemCount > prevCount) {
      setUnseenCount(current => nextUnseenCount(current, prevCount, itemCount, false))
    }
  }, [enabled, itemCount, resetKey, ref])

  // Snap to latest on initial mount (including after flex settle), unpin immediately on upward wheel, and keep pinned across resizes.
  useLayoutEffect(() => {
    if (!enabled) return
    const el = ref.current
    if (!el) return
    const snapAndSync = (): void => {
      const node = ref.current
      if (!node || !atBottomRef.current) return
      const maxTop = Math.max(0, node.scrollHeight - node.clientHeight)
      if (node.scrollTop < maxTop - 1) {
        scrollElementToBottom(node, 'auto')
      }
      lastScrollTopRef.current = maxTop
    }
    if (initialAtBottom && atBottomRef.current) {
      snapAndSync()
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(snapAndSync)
      }
    }
    const onWheel = (event: WheelEvent): void => {
      if (event.deltaY < 0 && el.scrollHeight > el.clientHeight + 1) {
        smoothScrollUntilRef.current = 0
        atBottomRef.current = false
        setAtBottom(false)
      }
    }
    el.addEventListener('wheel', onWheel, { passive: true })
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(snapAndSync) : null
    observer?.observe(el)
    return () => {
      el.removeEventListener('wheel', onWheel)
      observer?.disconnect()
    }
  }, [enabled, initialAtBottom, ref])

  return { atBottom, unseenCount, handleScroll, jumpToLatest, pauseFollow }
}

/** Floating bottom-right `↓ Latest` pill anchored over scrollable Swarm feeds. */
export function SwarmJumpLatestButton({
  visible,
  unseenCount = 0,
  onClick,
  label = 'Latest',
}: {
  readonly visible: boolean
  readonly unseenCount?: number
  readonly onClick: () => void
  readonly label?: string
}): ReactElement | null {
  if (!visible) return null
  return (
    <button
      type="button"
      className="swarm-jump-latest"
      aria-label={unseenCount > 0 ? `Jump to latest (${unseenCount} new)` : 'Jump to latest'}
      onClick={onClick}
    >
      <Icon name="arrowDown" size={14} />
      <span className="swarm-jump-latest__label">{label}</span>
      {unseenCount > 0 && (
        <span className="swarm-jump-latest__count" aria-hidden="true">
          {unseenCount}
        </span>
      )}
    </button>
  )
}

function SwarmMemberInspector({
  member,
  inspect,
  loading,
  online,
  busyAction,
  models,
  modelsLoading = false,
  switchingMember,
  modelDrafts,
  armedKillTarget = null,
  onArmKill,
  onCancelArmKill,
  onBack,
  onKillMember,
  onToggleModelSwitch,
  onPickModel,
  onChangeModelDraft,
  onSubmitModelSwitch,
  onCancelModelSwitch,
}: {
  readonly member: SwarmMemberSnapshot
  readonly inspect: SwarmMemberInspectResult | null
  readonly loading: boolean
  readonly online: boolean
  readonly busyAction: string
  readonly models: readonly ModelChoice[]
  readonly modelsLoading?: boolean
  readonly switchingMember: string
  readonly modelDrafts: Readonly<Record<string, string>>
  readonly armedKillTarget?: string | null
  readonly onArmKill?: (target: string) => void
  readonly onCancelArmKill?: () => void
  readonly onBack: () => void
  readonly onKillMember: (name: string) => void
  readonly onToggleModelSwitch: (member: SwarmMemberSnapshot) => void
  readonly onPickModel?: (member: SwarmMemberSnapshot, modelId: string) => void
  readonly onChangeModelDraft: (name: string, value: string) => void
  readonly onSubmitModelSwitch: (member: SwarmMemberSnapshot, event?: FormEvent) => void
  readonly onCancelModelSwitch: () => void
}): ReactElement {
  const effectiveMember = inspect?.member ?? member
  const stateLabel = stateLabelOf(effectiveMember)
  const spriteRole = resolveSimurghRole(effectiveMember.name, effectiveMember.role)
  const spriteState = effectiveMember.error ? 'error' : effectiveMember.state
  const isKilled = effectiveMember.killed || effectiveMember.state === 'killed'
  const canKill = !isKilled && effectiveMember.role !== 'coordinator'
  const canSwitchModel =
    !isKilled && (effectiveMember.state === 'waiting' || effectiveMember.state === 'stopped')
  const modelFormOpen = canSwitchModel && switchingMember === effectiveMember.name
  const modelAnchorRef = useRef<HTMLElement | null>(null)
  const inspectorBodyRef = useRef<HTMLDivElement>(null)
  const activity = inspect?.activity
  const activeTools = activity?.active_tools ?? []
  const recentTools = activity?.recent_tools ?? []
  const transcript = inspect?.messages ?? []
  const inspectorItemCount =
    transcript.length +
    activeTools.length +
    recentTools.length +
    (activity?.live_text_preview?.length ?? 0)
  const inspectorScroll = useFollowLatestScroll({
    ref: inspectorBodyRef,
    itemCount: inspectorItemCount,
    resetKey: effectiveMember.name,
  })
  const liveSubtitle =
    effectiveMember.current_activity ||
    activity?.summary ||
    (activity?.phase && activity.phase !== 'idle' ? activity.phase : '')

  return (
    <section className="swarm-inspector" aria-label={`Worker inspector for @${effectiveMember.name}`}>
      <header className="swarm-inspector__bar">
        <button
          type="button"
          className="swarm-btn swarm-btn--ghost swarm-btn--xs"
          aria-label="Back to members"
          onClick={onBack}
        >
          ← Back
        </button>
        <span className="swarm-member__dot" data-state={stateLabel} aria-hidden="true" />
        <SimurghSprite
          role={spriteRole}
          state={spriteState}
          size={18}
          animate={spriteState === 'working'}
          className="swarm-member__sprite"
        />
        <strong className="swarm-inspector__name">@{effectiveMember.name}</strong>
        <span className="swarm-role" data-role={effectiveMember.role}>
          {roleLabelOf(effectiveMember.role)}
        </span>
        <span className="swarm-state" data-state={stateLabel}>
          {stateLabel}
        </span>
        <span className="swarm-inspector__spacer" />
        {canKill && (
          armedKillTarget === effectiveMember.name ? (
            <>
              <button
                type="button"
                className="swarm-btn swarm-btn--xs swarm-btn--danger"
                aria-label={`Confirm stop ${effectiveMember.name}`}
                disabled={!online || Boolean(busyAction)}
                onClick={() => onKillMember(effectiveMember.name)}
              >
                Confirm stop?
              </button>
              <button
                type="button"
                className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                aria-label={`Cancel stop ${effectiveMember.name}`}
                onClick={() => onCancelArmKill?.()}
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              type="button"
              className="swarm-btn swarm-btn--xs swarm-btn--danger"
              aria-label={`Kill ${effectiveMember.name}`}
              disabled={!online || Boolean(busyAction)}
              onClick={() => (onArmKill ? onArmKill(effectiveMember.name) : onKillMember(effectiveMember.name))}
            >
              Stop
            </button>
          )
        )}
      </header>

      <div className="swarm-inspector__facts">
        {canSwitchModel ? (
          <button
            ref={el => {
              if (el) modelAnchorRef.current = el
            }}
            type="button"
            className={`swarm-member__model swarm-member__model--btn${modelFormOpen ? ' is-open' : ''}`}
            title={`Model: ${effectiveMember.model} — click to switch`}
            aria-label={`Switch model for ${effectiveMember.name}`}
            aria-haspopup="dialog"
            aria-expanded={modelFormOpen}
            disabled={!online || Boolean(busyAction)}
            onClick={event => {
              modelAnchorRef.current = event.currentTarget
              onToggleModelSwitch(effectiveMember)
            }}
          >
            <span>{effectiveMember.model_display_name || effectiveMember.model || 'Switch model'}</span>
            <Icon name="caretDown" size={9} />
          </button>
        ) : (
          <span>{effectiveMember.model_display_name || effectiveMember.model}</span>
        )}
        <span>·</span>
        <span>{workspaceBadgeLabel(effectiveMember.workspace)}</span>
        <span>·</span>
        <span>
          read #{effectiveMember.last_read_seq}
          {effectiveMember.last_post_seq !== undefined ? ` · posted #${effectiveMember.last_post_seq}` : ''}
        </span>
        {liveSubtitle && (
          <>
            <span>·</span>
            <span className="swarm-member__activity">{liveSubtitle}</span>
          </>
        )}
      </div>

      {modelFormOpen && (
        <SwarmModelPicker
          member={effectiveMember}
          models={models}
          modelsLoading={modelsLoading}
          online={online}
          busy={Boolean(busyAction)}
          customDraft={modelDrafts[effectiveMember.name] ?? ''}
          anchorRef={modelAnchorRef}
          onPickModel={modelId => {
            if (onPickModel) {
              onPickModel(effectiveMember, modelId)
            } else {
              onChangeModelDraft(effectiveMember.name, modelId)
              onSubmitModelSwitch(effectiveMember)
            }
          }}
          onChangeCustomDraft={value => onChangeModelDraft(effectiveMember.name, value)}
          onSubmitCustom={event => onSubmitModelSwitch(effectiveMember, event)}
          onClose={onCancelModelSwitch}
        />
      )}

      {effectiveMember.error && (
        <p className="swarm-member__error" role="alert">
          {effectiveMember.error}
        </p>
      )}

      <div className="swarm-inspector__stage">
        <div
          className="swarm-inspector__body"
          ref={inspectorBodyRef}
          onScroll={inspectorScroll.handleScroll}
        >
        {(activeTools.length > 0 || recentTools.length > 0 || activity?.live_text_preview) && (
          <section className="swarm-inspector__section" aria-label="Live worker activity">
            <h4 className="swarm-inspector__cap">Activity</h4>
            {activity?.live_text_preview && (
              <p className="swarm-inspector__preview">{activity.live_text_preview}</p>
            )}
            <div className="swarm-inspector__tools">
              {[...activeTools, ...recentTools.filter(tool => !activeTools.some(active => active.id === tool.id))]
                .slice(0, 12)
                .map(tool => {
                  const category = inspectorToolCategory(tool.name)
                  const duration =
                    tool.status === 'running'
                      ? 'live'
                      : formatInspectorDuration(tool.started_at, tool.finished_at)
                  const hasOutput = Boolean(tool.result_preview?.trim())
                  const rowBody = (
                    <>
                      <span className="swarm-inspector__tool-dot" data-status={tool.status} aria-hidden="true" />
                      <span className="swarm-inspector__tool-badge" data-category={category.kind}>
                        {category.label}
                      </span>
                      <code className="swarm-inspector__tool-name">{tool.name}</code>
                      {tool.summary && (
                        <span className="swarm-inspector__tool-sum" title={tool.summary}>
                          {tool.summary}
                        </span>
                      )}
                      {duration && <span className="swarm-inspector__tool-dur">{duration}</span>}
                      <span className="swarm-inspector__tool-state" data-status={tool.status}>
                        {tool.status}
                      </span>
                    </>
                  )
                  return hasOutput ? (
                    <details
                      key={tool.id}
                      className="swarm-inspector__tool-details"
                      data-status={tool.status}
                    >
                      <summary className="swarm-inspector__tool">{rowBody}</summary>
                      <pre className="swarm-inspector__tool-output">{tool.result_preview}</pre>
                    </details>
                  ) : (
                    <div key={tool.id} className="swarm-inspector__tool" data-status={tool.status}>
                      {rowBody}
                    </div>
                  )
                })}
            </div>
          </section>
        )}

        <section className="swarm-inspector__section" aria-label="Worker conversation">
          <h4 className="swarm-inspector__cap">
            Conversation
            {inspect ? ` · ${inspect.total_messages}` : ''}
          </h4>
          {loading && !inspect ? (
            <p className="swarm-board__empty" role="status">
              Loading worker conversation…
            </p>
          ) : transcript.length === 0 ? (
            <p className="swarm-board__empty">
              {effectiveMember.role === 'coordinator'
                ? 'Coordinator runs in the main conversation on the left.'
                : 'No worker turns recorded yet.'}
            </p>
          ) : (
            <div className="swarm-inspector__turns">
              {transcript.map(entry => {
                if (entry.role === 'tool') {
                  const formatted = formatInspectorToolResult(entry.tool_name, entry.text)
                  const category = inspectorToolCategory(entry.tool_name ?? 'tool')
                  return (
                    <div key={entry.index} className="swarm-inspector__turn swarm-inspector__turn--tool" data-role="tool">
                      {formatted.collapsible ? (
                        <details className="swarm-inspector__call-details swarm-inspector__result-details">
                          <summary className="swarm-inspector__call-row">
                            <span className="swarm-inspector__tool-badge" data-category={category.kind}>
                              {category.label}
                            </span>
                            <code className="swarm-inspector__tool-name">{entry.tool_name || 'tool'}</code>
                            <span className="swarm-inspector__tool-sum" title={formatted.summary}>
                              {formatted.summary}
                            </span>
                          </summary>
                          <div className="swarm-inspector__result-body">
                            {formatted.isJson ? (
                              <pre className="swarm-inspector__tool-output">{formatted.body}</pre>
                            ) : (
                              <div className="swarm-inspector__turn-text">{formatted.body}</div>
                            )}
                          </div>
                        </details>
                      ) : (
                        <div className="swarm-inspector__result-inline">
                          <span className="swarm-inspector__tool-badge" data-category={category.kind}>
                            {category.label}
                          </span>
                          <code className="swarm-inspector__tool-name">{entry.tool_name || 'tool'}</code>
                          <span className="swarm-inspector__result-summary">{formatted.summary}</span>
                        </div>
                      )}
                    </div>
                  )
                }

                const cleanedPrompt = entry.role === 'user'
                  ? stripInspectorSystemEnvelope(entry.text)
                  : { text: entry.text.trim(), isSystemWake: false }

                if (cleanedPrompt.isSystemWake) {
                  return (
                    <div key={entry.index} className="swarm-inspector__turn swarm-inspector__turn--wake" data-role="system">
                      <span className="swarm-inspector__wake-badge">board</span>
                      <span className="swarm-inspector__wake-text">{cleanedPrompt.text}</span>
                    </div>
                  )
                }

                return (
                  <div key={entry.index} className="swarm-inspector__turn" data-role={entry.role}>
                    <header className="swarm-inspector__turn-head">
                      <span className="swarm-inspector__turn-role">{entry.role}</span>
                    </header>
                    {cleanedPrompt.text && (
                      entry.role === 'assistant' ? (
                        <Markdown text={cleanedPrompt.text} className="swarm-inspector__md" />
                      ) : (
                        <div className="swarm-inspector__turn-text">{cleanedPrompt.text}</div>
                      )
                    )}
                    {entry.tool_calls && entry.tool_calls.length > 0 && (
                      <div className="swarm-inspector__turn-calls">
                        {entry.tool_calls.map(call => {
                          const category = inspectorToolCategory(call.name)
                          const summary = summarizeToolArgs(call.args_preview)
                          const prettyArgs = formatInspectorToolArgs(call.args_preview)
                          return (
                            <details key={call.id} className="swarm-inspector__call-details">
                              <summary className="swarm-inspector__call-row">
                                <span className="swarm-inspector__tool-badge" data-category={category.kind}>
                                  {category.label}
                                </span>
                                <code className="swarm-inspector__tool-name">{call.name}</code>
                                {summary && <span className="swarm-inspector__tool-sum">{summary}</span>}
                              </summary>
                              {prettyArgs && (
                                <pre className="swarm-inspector__tool-output">{prettyArgs}</pre>
                              )}
                            </details>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </section>
      </div>
        <SwarmJumpLatestButton
          visible={!inspectorScroll.atBottom}
          unseenCount={inspectorScroll.unseenCount}
          onClick={() => inspectorScroll.jumpToLatest('smooth')}
        />
      </div>
    </section>
  )
}

interface SwarmBoardPostRowProps {
  readonly message: SwarmBoardMessage
  readonly authorMember?: SwarmMemberSnapshot
  readonly hasCustomChannels: boolean
  readonly highlighted: boolean
  readonly expanded: boolean
  readonly onInsertSeqRef: (seq: number) => void
  readonly onInsertMention: (name: string) => void
  readonly onSelectChannel: (channel: string) => void
  readonly onJumpToSeq: (seq: number) => void
  readonly onToggleExpandMessage?: (seq: number) => void
}

function renderSwarmBoardPostArticle(
  {
    message,
    authorMember,
    hasCustomChannels,
    highlighted,
    expanded,
    onInsertSeqRef,
    onInsertMention,
    onSelectChannel,
    onToggleExpandMessage,
  }: SwarmBoardPostRowProps,
  preview: ReturnType<typeof previewSwarmMessage>,
  renderedBody: readonly ReactElement[],
): ReactElement {
  const isExpanded = !preview.foldable || expanded
  const authorRole: SwarmMemberSnapshot['role'] =
    authorMember?.role ??
    (message.agent === 'coordinator'
      ? 'coordinator'
      : message.agent === 'narrator'
        ? 'narrator'
        : message.agent === 'monitor'
          ? 'monitor'
          : 'worker')
  const authorState: SimurghSpriteState = authorMember
    ? authorMember.error
      ? 'error'
      : authorMember.state
    : 'waiting'
  const recipients = messageRecipientLabels(message)
  return (
    <article
      key={message.seq}
      id={`swarm-msg-${message.seq}`}
      data-seq={message.seq}
      data-highlighted={highlighted ? 'true' : undefined}
      data-folded={preview.foldable && !isExpanded ? 'true' : undefined}
      className="swarm-msg"
    >
      <div className="swarm-msg__avatar" aria-hidden="true">
        <SwarmAvatarTile
          name={message.agent}
          role={authorRole}
          state={authorState}
          size="sm"
        />
      </div>
      <div className="swarm-msg__main">
        <header className="swarm-msg__head">
          <button
            type="button"
            className="swarm-msg__author"
            data-agent={message.agent}
            title={`Reply to @${message.agent}`}
            onClick={() => onInsertMention(message.agent)}
          >
            @{message.agent}
          </button>
          {recipients.length > 0 && (
            <>
              <span className="swarm-msg__arrow" aria-hidden="true">→</span>
              <span className="swarm-msg__recipients">{recipients.join(' ')}</span>
            </>
          )}
          {(hasCustomChannels || message.channel !== 'main') && (
            <button
              type="button"
              className="swarm-msg__channel"
              title={`Filter channel #${message.channel}`}
              onClick={() => onSelectChannel(message.channel)}
            >
              #{message.channel}
            </button>
          )}
          {message.broadcast && <span className="swarm-badge swarm-badge--broadcast">@all</span>}
          <span className="swarm-msg__spacer" />
          <button
            type="button"
            className="swarm-msg__seq"
            title={`Reference #${message.seq}`}
            onClick={() => onInsertSeqRef(message.seq)}
          >
            #{message.seq}
          </button>
          {message.time && (
            <>
              <span className="swarm-msg__sep" aria-hidden="true">·</span>
              <time className="swarm-msg__time" dateTime={message.time} title={message.time}>
                {formatSwarmShortTime(message.time)}
              </time>
            </>
          )}
        </header>
        <div className="swarm-msg__body">{renderedBody}</div>
        {preview.foldable && (
          <button
            type="button"
            className="swarm-btn swarm-btn--ghost swarm-btn--xs swarm-msg__fold"
            aria-expanded={isExpanded}
            aria-label={
              isExpanded
                ? `Show less of message #${message.seq}`
                : `Show more of message #${message.seq}`
            }
            onClick={() => onToggleExpandMessage?.(message.seq)}
          >
            {isExpanded ? 'Show less' : `Show more · ${preview.displayLines} lines`}
          </button>
        )}
      </div>
    </article>
  )
}

function renderSwarmBoardPostInline(props: SwarmBoardPostRowProps): ReactElement {
  const preview = previewSwarmMessage(props.message.text)
  const isExpanded = !preview.foldable || props.expanded
  const displayText = isExpanded ? props.message.text : preview.preview
  const renderedBody = renderSwarmMessageText(displayText, {
    onMentionClick: props.onInsertMention,
    onChannelClick: props.onSelectChannel,
    onJumpToSeq: props.onJumpToSeq,
  })
  return renderSwarmBoardPostArticle(props, preview, renderedBody)
}

const SwarmBoardPostRow = memo(function SwarmBoardPostRow(
  props: SwarmBoardPostRowProps,
): ReactElement {
  const { message, expanded, onInsertMention, onSelectChannel, onJumpToSeq } = props
  const preview = useMemo(() => previewSwarmMessage(message.text), [message.text])
  const isExpanded = !preview.foldable || expanded
  const displayText = isExpanded ? message.text : preview.preview
  const renderedBody = useMemo(
    () =>
      renderSwarmMessageText(displayText, {
        onMentionClick: onInsertMention,
        onChannelClick: onSelectChannel,
        onJumpToSeq,
      }),
    [displayText, onInsertMention, onSelectChannel, onJumpToSeq],
  )
  return renderSwarmBoardPostArticle(props, preview, renderedBody)
})

interface SwarmMemberRowProps {
  readonly member: SwarmMemberSnapshot
  readonly insight?: SwarmMemberInsight
  readonly online: boolean
  readonly busy: boolean
  readonly modelFormOpen: boolean
  readonly killArmed?: boolean
  readonly models: readonly ModelChoice[]
  readonly modelsLoading: boolean
  readonly customDraft: string
  readonly modelAnchorRef?: RefObject<HTMLElement | null>
  readonly onInsertMention: (name: string) => void
  readonly onSelectTab?: (tab: SwarmSurfaceTab) => void
  readonly onInspectMember?: (name: string) => void
  readonly onArmKill?: (name: string) => void
  readonly onCancelArmKill?: () => void
  readonly onKillMember: (name: string) => void
  readonly onToggleModelSwitch: (member: SwarmMemberSnapshot) => void
  readonly onPickModel?: (member: SwarmMemberSnapshot, modelId: string) => void
  readonly onChangeModelDraft: (name: string, value: string) => void
  readonly onSubmitModelSwitch: (member: SwarmMemberSnapshot, event?: FormEvent) => void
  readonly onCancelModelSwitch: () => void
}

function renderSwarmMemberRow({
  member,
  insight,
  online,
  busy,
  modelFormOpen,
  killArmed = false,
  models,
  modelsLoading,
  customDraft,
  modelAnchorRef,
  onInsertMention,
  onSelectTab,
  onInspectMember,
  onArmKill,
  onCancelArmKill,
  onKillMember,
  onToggleModelSwitch,
  onPickModel,
  onChangeModelDraft,
  onSubmitModelSwitch,
  onCancelModelSwitch,
}: SwarmMemberRowProps): ReactElement {
  const stateLabel = stateLabelOf(member)
  const isKilled = member.killed || member.state === 'killed'
  const canKill = !isKilled && member.role !== 'coordinator'
  const canSwitchModel =
    !isKilled && (member.state === 'waiting' || member.state === 'stopped')
  const isHelper = Boolean(
    member.added_by &&
      member.added_by !== 'coordinator' &&
      member.role !== 'coordinator',
  )
  const customChannels = member.channels.filter(ch => ch !== 'main' && ch !== '*')
  const spriteState: SimurghSpriteState = member.error ? 'error' : member.state
  const latestSnippet = insight?.latestSnippet ?? ''
  const briefTitle = insight?.briefTitle ?? ''
  const showActiveSubtitle = Boolean(member.current_activity && stateLabel === 'working')

  return (
    <div
      key={member.name}
      role="listitem"
      className={`swarm-member${isHelper ? ' swarm-member--helper' : ''}${member.role === 'coordinator' ? ' swarm-member--coordinator' : ''}${modelFormOpen || killArmed ? ' is-picker-open' : ''}`}
      data-state={stateLabel}
      data-role={member.role}
    >
      <div className="swarm-member__avatar">
        <SwarmAvatarTile
          name={member.name}
          role={member.role}
          state={spriteState}
          size="md"
          showDot
          dotState={stateLabel}
          showSleep={stateLabel === 'waiting' && member.role !== 'coordinator'}
          active={stateLabel === 'working'}
        />
      </div>

      <div className="swarm-member__content">
        <div className="swarm-member__top">
          <button
            type="button"
            className="swarm-member__name"
            title={`Mention @${member.name} in board composer`}
            onClick={() => {
              onInsertMention(member.name)
              onSelectTab?.('board')
            }}
          >
            @{member.name}
          </button>
          {member.role !== 'worker' && (
            <span className="swarm-role" data-role={member.role}>
              {roleLabelOf(member.role)}
            </span>
          )}
          {member.role === 'coordinator' && stateLabel === 'working' && (
            <span className="swarm-member__live-tag">· Working</span>
          )}
          {isHelper && (
            <span
              className="swarm-member__lineage is-helper"
              title={`Helper spawned by @${member.added_by}`}
            >
              ↳ helper of @{member.added_by}
            </span>
          )}
          {!isHelper && member.role !== 'coordinator' && briefTitle && (
            <span className="swarm-member__brief" title={briefTitle}>
              {briefTitle}
            </span>
          )}
          <span className="swarm-member__spacer" />
          <div className="swarm-member__right-meta">
            {canSwitchModel ? (
              <button
                ref={el => {
                  if (modelFormOpen && modelAnchorRef && el) {
                    modelAnchorRef.current = el
                  }
                }}
                type="button"
                className={`swarm-member__model swarm-member__model--btn${modelFormOpen ? ' is-open' : ''}`}
                title={member.model ? `Model: ${member.model} — click to switch` : 'Click to switch model'}
                aria-label={`Switch model for ${member.name}`}
                aria-haspopup="dialog"
                aria-expanded={modelFormOpen}
                disabled={!online || busy}
                onClick={event => {
                  if (modelAnchorRef) modelAnchorRef.current = event.currentTarget
                  onToggleModelSwitch(member)
                }}
              >
                <span>{member.model_display_name || member.model || 'Switch model'}</span>
                <Icon name="caretDown" size={9} />
              </button>
            ) : (member.model_display_name || member.model) ? (
              <span className="swarm-member__model" title={member.model}>
                {member.model_display_name || member.model}
              </span>
            ) : null}
            {member.last_post_seq !== undefined && (
              <>
                <span className="swarm-member__seq-sep" aria-hidden="true">·</span>
                <span className="swarm-member__seq-tag">#{member.last_post_seq}</span>
              </>
            )}
          </div>
        </div>

        {showActiveSubtitle ? (
          <div className="swarm-member__activity" title={member.current_activity}>
            {member.current_activity}
          </div>
        ) : latestSnippet ? (
          <div className="swarm-member__preview" title={latestSnippet}>
            {latestSnippet}
          </div>
        ) : (
          <div className="swarm-member__preview swarm-member__preview--muted">
            {stateLabel === 'waiting'
              ? 'Standing by'
              : stateLabel === 'stopped'
                ? 'Paused — wakes on mention'
                : stateLabel === 'killed'
                  ? 'Stopped'
                  : 'Ready'}
          </div>
        )}

        <div className="swarm-member__meta">
          <span className="swarm-seq-meta">
            read #{member.last_read_seq}
            {member.last_post_seq !== undefined ? ` · posted #${member.last_post_seq}` : ''}
          </span>
          <span
            className="swarm-workspace"
            data-workspace={member.workspace}
            title={member.workspace_path || `Workspace mode: ${member.workspace}`}
          >
            {workspaceBadgeLabel(member.workspace)}
          </span>
          <span className="swarm-state" data-state={stateLabel}>
            {stateLabel}
          </span>
          {customChannels.map(ch => (
            <span key={ch} className="swarm-chip swarm-chip--channel">
              #{ch}
            </span>
          ))}
          <span className="swarm-member__spacer" />
          <div className="swarm-member__actions">
            {member.role !== 'coordinator' && onInspectMember && (
              <button
                type="button"
                className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                aria-label={`Inspect ${member.name}`}
                onClick={() => onInspectMember(member.name)}
              >
                Inspect
              </button>
            )}
            {canKill && (
              killArmed ? (
                <>
                  <button
                    type="button"
                    className="swarm-btn swarm-btn--xs swarm-btn--danger"
                    aria-label={`Confirm stop ${member.name}`}
                    disabled={!online || busy}
                    onClick={() => onKillMember(member.name)}
                  >
                    Confirm stop?
                  </button>
                  <button
                    type="button"
                    className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                    aria-label={`Cancel stop ${member.name}`}
                    onClick={() => onCancelArmKill?.()}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="swarm-btn swarm-btn--xs swarm-btn--danger"
                  aria-label={`Kill ${member.name}`}
                  disabled={!online || busy}
                  onClick={() => (onArmKill ? onArmKill(member.name) : onKillMember(member.name))}
                >
                  Stop
                </button>
              )
            )}
          </div>
        </div>

        {member.error && (
          <p className="swarm-member__error" role="alert">
            {member.error}
          </p>
        )}

        {modelFormOpen && (
          <SwarmModelPicker
            member={member}
            models={models}
            modelsLoading={modelsLoading}
            online={online}
            busy={busy}
            customDraft={customDraft}
            {...(modelAnchorRef ? { anchorRef: modelAnchorRef } : {})}
            onPickModel={modelId => {
              if (onPickModel) {
                onPickModel(member, modelId)
              } else {
                onChangeModelDraft(member.name, modelId)
                onSubmitModelSwitch(member)
              }
            }}
            onChangeCustomDraft={value => onChangeModelDraft(member.name, value)}
            onSubmitCustom={event => onSubmitModelSwitch(member, event)}
            onClose={onCancelModelSwitch}
          />
        )}
      </div>
    </div>
  )
}

const SwarmMemberRow = memo(function SwarmMemberRow(props: SwarmMemberRowProps): ReactElement {
  return renderSwarmMemberRow(props)
})

/** Pure presentational Swarm Board view (testable both via SSR and direct element callbacks). */
export function SwarmBoardView(props: SwarmBoardViewProps): ReactElement {
  const {
    status,
    channels,
    members,
    counts,
    messages,
    visibleMessages,
    visibleMessageLimit,
    onShowEarlierMessages,
    memoizeRows = false,
    models,
    modelsLoading = false,
    online,
    loading,
    busyAction,
    error,
    activeTab = 'overview',
    isSplitView = false,
    boardRef,
    armedKillTarget = null,
    onArmKill,
    onCancelArmKill,
    initialGraphMode,
    selectedChannel,
    highlightedSeq,
    expandedSeqs,
    memberFilter = 'all',
    memberQuery = '',
    boardQuery = '',
    boardFilter = 'all',
    boardFilterOpen = false,
    feedbackOpen,
    feedbackDraft,
    postDraft,
    mentionIndex = 0,
    mentionsDismissed = false,
    switchingMember,
    modelDrafts,
    inspectedMember = '',
    inspectResult = null,
    inspectLoading = false,
    feedRef,
    showJumpLatest = false,
    unseenLatestCount = 0,
    onFeedScroll,
    onJumpToLatest,
    composerInputRef,
    modelAnchorRef,
    onSelectTab,
    onRefresh,
    onStopAll,
    onChangeFeedbackDraft,
    onSubmitFeedback,
    onCloseFeedback,
    onSelectChannel,
    onInsertMention,
    onPickMentionCandidate,
    onMoveMentionIndex,
    onDismissMentions,
    onInsertSeqRef,
    onJumpToSeq,
    onToggleExpandMessage,
    onChangeMemberFilter,
    onChangeMemberQuery,
    onChangeBoardQuery,
    onChangeBoardFilter,
    onToggleBoardFilterOpen,
    onKillMember,
    onInspectMember,
    onCloseInspect,
    onToggleModelSwitch,
    onPickModel,
    onChangeModelDraft,
    onSubmitModelSwitch,
    onCancelModelSwitch,
    onChangePostDraft,
    onSubmitPost,
  } = props

  const postChannel = selectedChannel === '*' ? 'main' : selectedChannel
  const postChannelMeta = channels.find(entry => entry.name === postChannel)
  const postChannelArchived = postChannelMeta?.archived === true
  const hasCustomChannels = channels.some(channel => channel.name !== 'main' || channel.archived)
  const title = status?.title.trim() || 'Immortals'
  const overview = status?.overview ?? null
  const hasNarrator = members.some(member => member.role === 'narrator' && !member.killed)
  const showNarratorFeedback = Boolean(hasNarrator || overview?.markdown.trim() || feedbackOpen)
  const mentionCandidates = !postChannelArchived && !mentionsDismissed
    ? swarmMentionCandidates(postDraft, members)
    : []
  const activeMentionIndex =
    mentionCandidates.length > 0
      ? ((mentionIndex % mentionCandidates.length) + mentionCandidates.length) % mentionCandidates.length
      : 0

  const openMemberForInspect = inspectedMember
    ? members.find(member => member.name === inspectedMember) ?? inspectResult?.member
    : undefined

  const memberByName = new Map<string, SwarmMemberSnapshot>()
  for (const member of members) memberByName.set(member.name, member)
  const memberInsights = buildMemberInsights(members, messages)

  const showMemberFilter = members.length > 8
  const filteredMembers = showMemberFilter
    ? filterSwarmMembers(members, memberFilter, memberQuery)
    : memberQuery.trim()
      ? filterSwarmMembers(members, 'all', memberQuery)
      : members
  const retiredCount = counts.killed + counts.stopped

  // Partition members cleanly for AGY-style hierarchy: Active/Command -> STANDING BY -> RETIRED
  const activeCommandMembers: SwarmMemberSnapshot[] = []
  const standingByMembers: SwarmMemberSnapshot[] = []
  const retiredMembers: SwarmMemberSnapshot[] = []
  for (const member of filteredMembers) {
    const isRetired = member.killed || member.state === 'killed' || member.state === 'stopped'
    if (member.role === 'coordinator' || (!isRetired && member.state === 'working')) {
      activeCommandMembers.push(member)
    } else if (isRetired) {
      retiredMembers.push(member)
    } else {
      standingByMembers.push(member)
    }
  }

  // Overview derived cards
  const coordinatorMember = members.find(member => member.role === 'coordinator') ?? null
  const workerMembers = members.filter(member => member.role !== 'coordinator')
  const runningWorkers = workerMembers.filter(member => !member.killed && member.state === 'working')
  const completedWorkers = workerMembers.filter(member => memberInsights.get(member.name)?.done === true)
  const threadGroupMap = new Map<string, SwarmMemberSnapshot[]>()
  for (const worker of workerMembers) {
    const rawGroup = worker.group?.trim()
    const groupTitle = rawGroup ? `${rawGroup.slice(0, 1).toUpperCase()}${rawGroup.slice(1)}` : 'Tasks'
    const bucket = threadGroupMap.get(groupTitle)
    if (bucket) bucket.push(worker)
    else threadGroupMap.set(groupTitle, [worker])
  }
  const threadGroups = [...threadGroupMap.entries()]

  const isBoardFiltering = Boolean(boardQuery.trim()) || boardFilter !== 'all'
  const filteredBoardMessages = isBoardFiltering
    ? visibleMessages.filter(message => matchesBoardQuery(message, boardQuery, boardFilter))
    : visibleMessages

  const effectiveMessageLimit = isBoardFiltering
    ? Math.max(filteredBoardMessages.length, visibleMessageLimit ?? SWARM_BOARD_TAIL_LIMIT)
    : (visibleMessageLimit ?? SWARM_BOARD_TAIL_LIMIT)
  const hiddenEarlierCount = Math.max(0, filteredBoardMessages.length - effectiveMessageLimit)
  const windowedMessages = hiddenEarlierCount > 0 ? filteredBoardMessages.slice(hiddenEarlierCount) : filteredBoardMessages
  const handleShowEarlierMessages = onShowEarlierMessages

  const liveBadge = swarmLiveBadgeState(online, status, counts)

  const renderMemberList = (list: readonly SwarmMemberSnapshot[]): ReactElement[] =>
    list.map(member => {
      const isKilled = member.killed || member.state === 'killed'
      const canSwitchModel =
        !isKilled && (member.state === 'waiting' || member.state === 'stopped')
      const modelFormOpen = canSwitchModel && switchingMember === member.name
      const insight = memberInsights.get(member.name)
      const rowProps: SwarmMemberRowProps = {
        member,
        ...(insight ? { insight } : {}),
        online,
        busy: Boolean(busyAction),
        modelFormOpen,
        killArmed: armedKillTarget === member.name,
        models,
        modelsLoading,
        customDraft: modelDrafts[member.name] ?? '',
        ...(modelAnchorRef ? { modelAnchorRef } : {}),
        onInsertMention,
        ...(onSelectTab ? { onSelectTab } : {}),
        ...(onInspectMember ? { onInspectMember } : {}),
        ...(onArmKill ? { onArmKill } : {}),
        ...(onCancelArmKill ? { onCancelArmKill } : {}),
        onKillMember,
        onToggleModelSwitch,
        ...(onPickModel ? { onPickModel } : {}),
        onChangeModelDraft,
        onSubmitModelSwitch,
        onCancelModelSwitch,
      }
      return memoizeRows ? (
        <SwarmMemberRow key={member.name} {...rowProps} />
      ) : (
        renderSwarmMemberRow(rowProps)
      )
    })

  return (
    <section
      ref={boardRef}
      className="swarm-board"
      data-tab={activeTab}
      data-split={isSplitView ? 'true' : undefined}
      data-inspecting={openMemberForInspect ? 'true' : undefined}
      aria-label="Immortals Board"
    >
      {/* ── AGY-Inspired Header + Minimal Underline Tabs ── */}
      <header className="swarm-board__header">
        <div className="swarm-board__title-row">
          <h3 className="swarm-board__title" title={title}>
            {title}
          </h3>
          <span className="swarm-board__header-spacer" />
          <div className="swarm-board__controls">
            <span
              className="swarm-board__live"
              data-live={liveBadge.live ? 'true' : undefined}
              data-status={liveBadge.label.toLowerCase()}
              aria-label={liveBadge.ariaLabel}
            >
              <span className="swarm-board__live-dot" aria-hidden="true" />
              <span>{liveBadge.label}</span>
            </span>
            <button
              type="button"
              className="swarm-btn swarm-btn--ghost swarm-btn--xs swarm-board__icon-btn"
              aria-label="Refresh Immortals Board"
              title="Refresh Immortals status and board"
              disabled={!online || Boolean(busyAction)}
              onClick={onRefresh}
            >
              <Icon name="retry" size={12} />
            </button>
            {armedKillTarget === 'all' ? (
              <>
                <button
                  type="button"
                  className="swarm-btn swarm-btn--xs swarm-btn--danger swarm-board__stop-btn"
                  aria-label="Confirm stop all Immortals workers"
                  title="Permanently stop all living workers and delete their share/branch workspaces"
                  disabled={!online || Boolean(busyAction) || counts.livingWorkers === 0}
                  onClick={onStopAll}
                >
                  <span>Confirm stop?</span>
                </button>
                <button
                  type="button"
                  className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                  aria-label="Cancel stop all Immortals workers"
                  onClick={() => onCancelArmKill?.()}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                className={`swarm-btn swarm-btn--xs ${counts.livingWorkers > 0 ? 'swarm-btn--danger' : 'swarm-btn--ghost'} swarm-board__stop-btn`}
                aria-label="Stop all Immortals workers"
                title={counts.livingWorkers > 0 ? 'Stop all living workers permanently' : 'No living workers to stop'}
                disabled={!online || Boolean(busyAction) || counts.livingWorkers === 0}
                onClick={() => (onArmKill ? onArmKill('all') : onStopAll())}
              >
                <Icon name="stop" size={12} />
                <span>Stop</span>
              </button>
            )}
          </div>
        </div>

        <div className="swarm-board__bar">
          <div className="swarm-tabs" role="tablist" aria-label="Immortals views">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'overview'}
              className={`swarm-tab${activeTab === 'overview' ? ' is-selected' : ''}`}
              onClick={() => {
                onCloseInspect?.()
                onSelectTab?.('overview')
              }}
            >
              <span>Overview</span>
              {overview?.markdown.trim() && <span className="swarm-tab__dot" aria-hidden="true" />}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={isSplitView ? activeTab === 'board' || activeTab === 'members' : activeTab === 'board'}
              className={`swarm-tab${(isSplitView ? activeTab === 'board' || activeTab === 'members' : activeTab === 'board') ? ' is-selected' : ''}`}
              onClick={() => {
                onCloseInspect?.()
                onSelectTab?.('board')
              }}
            >
              <span>{isSplitView ? 'Board & Members' : 'Board'}</span>
              {messages.length > 0 && <span className="swarm-tab__count">{messages.length}</span>}
            </button>
            {!isSplitView && (
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'members'}
                className={`swarm-tab${activeTab === 'members' ? ' is-selected' : ''}`}
                onClick={() => onSelectTab?.('members')}
              >
                <span>Members</span>
                {members.length > 0 && <span className="swarm-tab__count">{members.length}</span>}
              </button>
            )}
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'graph'}
              className={`swarm-tab${activeTab === 'graph' ? ' is-selected' : ''}`}
              onClick={() => {
                onCloseInspect?.()
                onSelectTab?.('graph')
              }}
            >
              <span>Graph</span>
            </button>
          </div>
        </div>
      </header>

      {(loading || busyAction) && (
        <div className="swarm-board__status" role="status" aria-live="polite">
          <span className="studio-spinner" aria-hidden="true" />
          <span>{busyAction ? 'Updating Immortals…' : 'Syncing Immortals…'}</span>
        </div>
      )}

      {error && (
        <p className="swarm-board__error studio-error" role="alert">
          {error}
        </p>
      )}

      {/* ── Live Worker Inspector (when a member is inspected) ── */}
      {openMemberForInspect && (
        <SwarmMemberInspector
          member={openMemberForInspect}
          inspect={inspectResult && inspectResult.member.name === openMemberForInspect.name ? inspectResult : null}
          loading={inspectLoading}
          online={online}
          busyAction={busyAction}
          models={models}
          modelsLoading={modelsLoading}
          switchingMember={switchingMember}
          modelDrafts={modelDrafts}
          armedKillTarget={armedKillTarget}
          {...(onArmKill ? { onArmKill } : {})}
          {...(onCancelArmKill ? { onCancelArmKill } : {})}
          onBack={() => onCloseInspect?.()}
          onKillMember={onKillMember}
          onToggleModelSwitch={onToggleModelSwitch}
          {...(onPickModel ? { onPickModel } : {})}
          onChangeModelDraft={onChangeModelDraft}
          onSubmitModelSwitch={onSubmitModelSwitch}
          onCancelModelSwitch={onCancelModelSwitch}
        />
      )}

      {/* ── Interactive Immortals Graph (Map + Mentions + Replay + Export GIF) ── */}
      {activeTab === 'graph' && !openMemberForInspect && (
        <section
          className="swarm-board__graph"
          data-pane="graph"
          aria-label="Immortals graph"
        >
          <SwarmGraph
            status={status}
            messages={messages}
            {...(initialGraphMode ? { initialMode: initialGraphMode } : {})}
            {...(onInspectMember ? { onInspectMember } : {})}
          />
        </section>
      )}

      {/* ── AGY-Inspired Live Immortals Overview ── */}
      <section
        className="swarm-board__overview"
        data-pane="overview"
        aria-label="Narrator overview"
      >
        <div className="swarm-board__overview-scroll">
          {members.length === 0 && !overview?.markdown.trim() ? (
            <div className="swarm-ov-card swarm-ov-card--empty">
              <SwarmAvatarTile
                name="coordinator"
                role="coordinator"
                state="waiting"
                size="lg"
              />
              <div className="swarm-ov-empty__copy">
                <h4 className="swarm-ov-empty__title">Immortals ready</h4>
                <p className="swarm-ov-empty__sub">
                  Run <code>/immortals &lt;task&gt;</code> in chat to dispatch parallel workers on a shared board.
                </p>
              </div>
            </div>
          ) : (
            <div className="swarm-ov-stack">
              {/* 1. Running Card (live active workers) */}
              {runningWorkers.length > 0 && (
                <div className="swarm-ov-card swarm-ov-card--running">
                  <div className="swarm-ov-card__head swarm-ov-card__head--static">
                    <span className="swarm-ov-dot swarm-ov-dot--working" aria-hidden="true" />
                    <span className="swarm-ov-card__title">Running</span>
                    <span className="swarm-ov-card__count">{runningWorkers.length}</span>
                  </div>
                  <div className="swarm-ov-list">
                    {runningWorkers.map(worker => {
                      const insight = memberInsights.get(worker.name)
                      const headline =
                        insight?.briefTitle ||
                        worker.current_activity ||
                        `@${worker.name}`
                      const snippet =
                        worker.current_activity && worker.current_activity !== headline
                          ? worker.current_activity
                          : insight?.latestSnippet || ''
                      return (
                        <button
                          key={worker.name}
                          type="button"
                          className="swarm-ov-task-row"
                          onClick={() => onInspectMember?.(worker.name)}
                        >
                          <div className="swarm-ov-task-row__top">
                            <span className="swarm-ov-spin" aria-hidden="true">
                              <Icon name="spinner" size={14} />
                            </span>
                            <span className="swarm-ov-task-row__title">{headline}</span>
                            <span className="swarm-ov-pill swarm-ov-pill--working">Working</span>
                            <span className="swarm-ov-member-chip">
                              <SimurghSprite
                                role={resolveSimurghRole(worker.name, worker.role)}
                                state="working"
                                size={13}
                                animate={false}
                              />
                              <span>{worker.name}</span>
                            </span>
                          </div>
                          {snippet && (
                            <p className="swarm-ov-task-row__snippet">{snippet}</p>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* 2. Compact Progress Summary Bar (without duplicating completed workers listed in Threads) */}
              {members.length > 0 && (
                <div className="swarm-ov-card swarm-ov-card--progress">
                  <div className="swarm-ov-card__head swarm-ov-card__head--static">
                    <span className="swarm-ov-ring" data-complete={workerMembers.length > 0 && completedWorkers.length === workerMembers.length ? 'true' : undefined} aria-hidden="true" />
                    <span className="swarm-ov-progress__label">
                      <strong>{completedWorkers.length} of {workerMembers.length}</strong> tasks done
                    </span>
                    <span className="swarm-board__counts" aria-label="Member status counts">
                      <span className="swarm-count" data-tone={counts.working > 0 ? 'working' : undefined}>
                        <strong>{counts.working}</strong> active
                      </span>
                      <span className="swarm-count" data-tone={counts.waiting > 0 ? 'waiting' : undefined}>
                        <strong>{counts.waiting}</strong> waiting
                      </span>
                      {counts.stopped > 0 && (
                        <span className="swarm-count" data-tone="stopped">
                          <strong>{counts.stopped}</strong> stopped
                        </span>
                      )}
                      {counts.killed > 0 && (
                        <span className="swarm-count" data-tone="killed">
                          <strong>{counts.killed}</strong> killed
                        </span>
                      )}
                      {status && status.head > 0 && (
                        <span className="swarm-board__head-seq" title="Latest message sequence on board">
                          #{status.head}
                        </span>
                      )}
                    </span>
                  </div>
                </div>
              )}

              {/* 3. Unified Threads / Task Groups (deduplicated: running workers live in Running above) */}
              {threadGroups.length > 0 && (
                <div className="swarm-ov-section">
                  <h4 className="swarm-ov-section-title">Threads</h4>
                  <div className="swarm-ov-threads">
                    {threadGroups.map(([groupName, groupWorkers]) => {
                      const doneInGroup = groupWorkers.filter(w => memberInsights.get(w.name)?.done === true).length
                      const workingInGroup = groupWorkers.some(w => !w.killed && w.state === 'working')
                      const nonRunningWorkers = groupWorkers.filter(w => !(!w.killed && w.state === 'working'))
                      const allDone = groupWorkers.length > 0 && doneInGroup === groupWorkers.length
                      const pct = groupWorkers.length > 0 ? Math.round((doneInGroup / groupWorkers.length) * 100) : 0
                      const groupStatus = workingInGroup ? 'Moving' : allDone ? 'Done' : 'Standing by'
                      if (nonRunningWorkers.length === 0) {
                        return (
                          <div key={groupName} className="swarm-ov-card swarm-ov-thread">
                            <div className="swarm-ov-card__head swarm-ov-card__head--static">
                              <span
                                className={`swarm-ov-dot${workingInGroup ? ' swarm-ov-dot--working' : allDone ? ' swarm-ov-dot--done' : ''}`}
                                aria-hidden="true"
                              />
                              <span className="swarm-ov-card__title">{groupName}</span>
                              <span className="swarm-ov-thread__status">{groupStatus}</span>
                              <span className="swarm-ov-thread__progress">
                                <span className="swarm-ov-bar" aria-hidden="true">
                                  <span
                                    className="swarm-ov-bar__fill"
                                    data-working={workingInGroup ? 'true' : undefined}
                                    style={{ width: `${Math.max(workingInGroup ? 35 : 0, pct)}%` }}
                                  />
                                </span>
                                <span className="swarm-ov-thread__frac">
                                  {doneInGroup}/{groupWorkers.length}
                                </span>
                              </span>
                            </div>
                          </div>
                        )
                      }
                      return (
                        <details key={groupName} open className="swarm-ov-card swarm-ov-thread">
                          <summary className="swarm-ov-card__head">
                            <span
                              className={`swarm-ov-dot${workingInGroup ? ' swarm-ov-dot--working' : allDone ? ' swarm-ov-dot--done' : ''}`}
                              aria-hidden="true"
                            />
                            <span className="swarm-ov-card__title">{groupName}</span>
                            <span className="swarm-ov-thread__status">{groupStatus}</span>
                            <span className="swarm-ov-thread__progress">
                              <span className="swarm-ov-bar" aria-hidden="true">
                                <span
                                  className="swarm-ov-bar__fill"
                                  data-working={workingInGroup ? 'true' : undefined}
                                  style={{ width: `${Math.max(workingInGroup ? 35 : 0, pct)}%` }}
                                />
                              </span>
                              <span className="swarm-ov-thread__frac">
                                {doneInGroup}/{groupWorkers.length}
                              </span>
                            </span>
                            <span className="swarm-ov-caret" aria-hidden="true">
                              <Icon name="caretDown" size={12} />
                            </span>
                          </summary>
                          <div className="swarm-ov-card__body">
                            <div className="swarm-ov-list">
                              {nonRunningWorkers.map(worker => {
                                const insight = memberInsights.get(worker.name)
                                const isDone = insight?.done === true
                                const headline =
                                  insight?.briefTitle ||
                                  worker.current_activity ||
                                  insight?.latestSnippet ||
                                  `@${worker.name}`
                                return (
                                  <button
                                    key={worker.name}
                                    type="button"
                                    className="swarm-ov-task-row swarm-ov-task-row--compact"
                                    onClick={() => onInspectMember?.(worker.name)}
                                  >
                                    <div className="swarm-ov-task-row__top">
                                      {isDone ? (
                                        <span className="swarm-ov-check" aria-hidden="true">
                                          <Icon name="checkCircle" size={14} />
                                        </span>
                                      ) : (
                                        <span className="swarm-ov-idle-ring" aria-hidden="true" />
                                      )}
                                      <span className="swarm-ov-task-row__title">{headline}</span>
                                      <span
                                        className={`swarm-ov-pill${isDone ? ' swarm-ov-pill--done' : ''}`}
                                      >
                                        {isDone ? 'Done' : stateLabelOf(worker)}
                                      </span>
                                      <span className="swarm-ov-member-chip">
                                        <SimurghSprite
                                          role={resolveSimurghRole(worker.name, worker.role)}
                                          state={worker.error ? 'error' : worker.state}
                                          size={13}
                                          animate={false}
                                        />
                                        <span>{worker.name}</span>
                                      </span>
                                    </div>
                                  </button>
                                )
                              })}
                            </div>
                          </div>
                        </details>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* 4. Narrator Live Summary (when curated) */}
              {overview?.markdown.trim() && (
                <div className="swarm-ov-card swarm-ov-card--doc">
                  <article className="artdoc swarm-overview-doc">
                    <header className="artdoc__head swarm-overview-doc__head">
                      <div className="artdoc__chips">
                        {overview.author && <span className="artdoc__chip">@{overview.author}</span>}
                        {overview.updated_at && (
                          <span className="artdoc__chip" title={overview.updated_at}>
                            {formatSwarmShortTime(overview.updated_at)}
                          </span>
                        )}
                      </div>
                      {overview.summary && overview.summary.trim() !== overview.markdown.trim() && (
                        <p className="swarm-board__overview-summary">{overview.summary}</p>
                      )}
                    </header>
                    <Markdown
                      text={overview.markdown}
                      className="artdoc__body swarm-board__overview-md"
                    />
                  </article>
                </div>
              )}

              {/* 5. The Team (Coordinator card + SimurghSprite avatar tile strip) */}
              {members.length > 0 && (
                <div className="swarm-ov-section">
                  <h4 className="swarm-ov-section-title">The team</h4>
                  {coordinatorMember && (
                    <button
                      type="button"
                      className="swarm-ov-card swarm-ov-coord"
                      onClick={() => onSelectTab?.('members')}
                    >
                      <SwarmAvatarTile
                        name={coordinatorMember.name}
                        role="coordinator"
                        state={coordinatorMember.error ? 'error' : coordinatorMember.state}
                        size="lg"
                        showDot
                        dotState={stateLabelOf(coordinatorMember)}
                        active={coordinatorMember.state === 'working'}
                      />
                      <div className="swarm-ov-coord__body">
                        <div className="swarm-ov-coord__name">Coordinator</div>
                        <div className="swarm-ov-coord__sub">
                          {coordinatorMember.current_activity ||
                            memberInsights.get(coordinatorMember.name)?.latestSnippet ||
                            coordinatorMember.model_display_name ||
                            coordinatorMember.model}
                        </div>
                      </div>
                    </button>
                  )}

                  {workerMembers.length > 0 && (
                    <div className="swarm-ov-roster" role="list" aria-label="Immortals team tiles">
                      <button
                        type="button"
                        className="swarm-ov-tile"
                        title="View all members"
                        onClick={() => onSelectTab?.('members')}
                      >
                        <span className="swarm-avatar-tile swarm-avatar-tile--tile swarm-avatar-tile--all" aria-hidden="true">
                          <span className="swarm-avatar-cluster">
                            <SimurghSprite role="coordinator" state="waiting" size={11} animate={false} />
                            <SimurghSprite role="worker" state="waiting" size={11} animate={false} />
                            <SimurghSprite role="verifier" state="waiting" size={11} animate={false} />
                            <SimurghSprite role="narrator" state="waiting" size={11} animate={false} />
                          </span>
                        </span>
                        <span className="swarm-ov-tile__label">All</span>
                      </button>

                      {workerMembers.map(worker => {
                        const stateLabel = stateLabelOf(worker)
                        const isWorking = !worker.killed && worker.state === 'working'
                        return (
                          <button
                            key={worker.name}
                            type="button"
                            className={`swarm-ov-tile${isWorking ? ' is-working' : ''}`}
                            title={`@${worker.name} (${stateLabel}) — click to inspect`}
                            onClick={() => onInspectMember?.(worker.name)}
                          >
                            <SwarmAvatarTile
                              name={worker.name}
                              role={worker.role}
                              state={worker.error ? 'error' : worker.state}
                              size="tile"
                              showSleep={stateLabel === 'waiting'}
                              showDot={isWorking}
                              dotState={stateLabel}
                              active={isWorking}
                            />
                            <span className="swarm-ov-tile__label">@{worker.name}</span>
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Subtle narrator hint when members exist but narrator hasn't published markdown yet */}
              {!overview?.markdown.trim() && members.length > 0 && (
                <p className="swarm-board__overview-empty">
                  {hasNarrator
                    ? 'Waiting for @narrator to publish the first overview…'
                    : 'No narrator overview yet — send feedback below or spawn a narrator worker to curate a live summary.'}
                </p>
              )}
            </div>
          )}
        </div>

        {showNarratorFeedback && (
          <form
            className="swarm-board__feedback-form"
            aria-label="Feedback to Narrator"
            onSubmit={onSubmitFeedback}
          >
            <div className="swarm-pill-composer">
              <input
                type="text"
                className="swarm-input swarm-pill-composer__input"
                aria-label="Narrator feedback message"
                placeholder="Feedback to narrator…"
                value={feedbackDraft}
                disabled={!online || Boolean(busyAction)}
                onChange={event => onChangeFeedbackDraft(event.target.value)}
                onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
                  if (event.key === 'Escape') {
                    event.preventDefault()
                    event.stopPropagation()
                    if (feedbackDraft) onChangeFeedbackDraft('')
                    onCloseFeedback()
                  }
                }}
              />
              <button
                type="submit"
                className="swarm-btn swarm-btn--primary swarm-pill-composer__send"
                aria-label="Send narrator feedback"
                title="Send feedback to narrator"
                disabled={!online || Boolean(busyAction) || !feedbackDraft.trim()}
              >
                <Icon name="arrowUp" size={13} />
              </button>
            </div>
          </form>
        )}
      </section>

      {/* ── Main Content: Members + Board (single-surface < 780px, 2-col split >= 780px) ── */}
      <div className="swarm-board__main">
        {/* Members View */}
        <section
          className="swarm-board__roster"
          data-pane="members"
          aria-label="Member roster"
        >
          <header className="swarm-board__section-head">
            <span>Members</span>
            <span className="swarm-muted">
              {filteredMembers.length !== members.length
                ? `${filteredMembers.length} / ${members.length}`
                : members.length}
            </span>
          </header>

          {showMemberFilter ? (
            <div
              className="swarm-board__channels swarm-roster__filters"
              role="search"
              aria-label="Filter Immortals members"
            >
              <div className="swarm-filter-pills" role="group" aria-label="Member status filter">
                {(
                  [
                    { id: 'all', label: 'All', count: members.length },
                    { id: 'active', label: 'Active', count: counts.working },
                    { id: 'waiting', label: 'Waiting', count: counts.waiting },
                    { id: 'retired', label: 'Retired', count: retiredCount },
                  ] as const
                ).map(option => {
                  const selected = memberFilter === option.id
                  return (
                    <button
                      key={option.id}
                      type="button"
                      aria-pressed={selected}
                      className={`swarm-filter-pill${selected ? ' is-selected' : ''}`}
                      onClick={() => onChangeMemberFilter?.(option.id)}
                    >
                      <span>{option.label}</span>
                      <span className="swarm-tab__count">{option.count}</span>
                    </button>
                  )
                })}
              </div>
              <label className="swarm-searchbar swarm-searchbar--compact">
                <Icon name="search" size={12} />
                <input
                  type="search"
                  className="swarm-searchbar__input"
                  aria-label="Filter members"
                  placeholder="Filter members…"
                  spellCheck={false}
                  value={memberQuery}
                  onChange={event => onChangeMemberQuery?.(event.target.value)}
                />
              </label>
            </div>
          ) : members.length > 0 ? (
            <div className="swarm-board__searchrow">
              <label className="swarm-searchbar">
                <Icon name="search" size={13} />
                <input
                  type="search"
                  className="swarm-searchbar__input"
                  aria-label="Search members"
                  placeholder="Search members"
                  spellCheck={false}
                  value={memberQuery}
                  onChange={event => onChangeMemberQuery?.(event.target.value)}
                />
              </label>
            </div>
          ) : null}

          {members.length === 0 ? (
            <p className="swarm-board__empty">
              No active Immortals run in this session yet. Run <code>/immortals &lt;task&gt;</code> in the composer to coordinate workers on a shared board.
            </p>
          ) : filteredMembers.length === 0 ? (
            <div className="swarm-board__empty swarm-board__empty--row">
              <span>No members match the current filter.</span>
              <button
                type="button"
                className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                onClick={() => {
                  onChangeMemberFilter?.('all')
                  onChangeMemberQuery?.('')
                }}
              >
                Clear filter
              </button>
            </div>
          ) : (
            <div className="swarm-board__members" role="list">
              {activeCommandMembers.length > 0 && renderMemberList(activeCommandMembers)}
              {standingByMembers.length > 0 && (
                <>
                  <div className="swarm-roster__group-head" role="presentation">
                    <span>STANDING BY</span>
                    <span>{standingByMembers.length}</span>
                  </div>
                  {renderMemberList(standingByMembers)}
                </>
              )}
              {retiredMembers.length > 0 && (
                <>
                  <div className="swarm-roster__group-head" role="presentation">
                    <span>RETIRED</span>
                    <span>{retiredMembers.length}</span>
                  </div>
                  {renderMemberList(retiredMembers)}
                </>
              )}
            </div>
          )}
        </section>

        {/* Live Message Board Timeline + AGY Search & Clean Bottom Composer */}
        <section
          className="swarm-board__feed-col"
          data-pane="board"
          aria-label="Message board"
        >
          {/* Search + Filter Bar */}
          <div className="swarm-board__searchrow">
            <label className="swarm-searchbar">
              <Icon name="search" size={13} />
              <input
                type="search"
                className="swarm-searchbar__input"
                aria-label="Search board messages"
                placeholder="Search"
                spellCheck={false}
                value={boardQuery}
                onChange={event => onChangeBoardQuery?.(event.target.value)}
              />
            </label>
            <button
              type="button"
              className={`swarm-board__filter-btn${boardFilterOpen || boardFilter !== 'all' ? ' is-active' : ''}`}
              aria-label="Filter board messages"
              aria-pressed={boardFilterOpen || boardFilter !== 'all'}
              title="Filter posts"
              onClick={() => onToggleBoardFilterOpen?.()}
            >
              <Icon name="listBullets" size={14} />
            </button>
          </div>

          {(boardFilterOpen || boardFilter !== 'all') && (
            <div className="swarm-board__quickfilters" role="group" aria-label="Board post filter">
              {(
                [
                  { id: 'all', label: 'All' },
                  { id: 'mentions', label: '@mentions' },
                  { id: 'done', label: '[done]' },
                  { id: 'stall', label: '[stall]' },
                ] as const
              ).map(option => (
                <button
                  key={option.id}
                  type="button"
                  aria-pressed={boardFilter === option.id}
                  className={`swarm-filter-pill${boardFilter === option.id ? ' is-selected' : ''}`}
                  onClick={() => onChangeBoardFilter?.(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}

          {/* Hide channel strip when only #main exists */}
          {hasCustomChannels && (
            <div className="swarm-board__channels" role="tablist" aria-label="Immortals channels">
              <button
                type="button"
                role="tab"
                aria-selected={selectedChannel === '*'}
                className={`swarm-channel${selectedChannel === '*' ? ' is-selected' : ''}`}
                onClick={() => onSelectChannel('*')}
              >
                <span className="swarm-channel__name">All</span>
                <span className="swarm-channel__count">{messages.length}</span>
              </button>
              {channels.map(channel => {
                const selected = selectedChannel === channel.name
                return (
                  <button
                    key={channel.name}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    data-archived={channel.archived ? 'true' : undefined}
                    className={`swarm-channel${selected ? ' is-selected' : ''}${channel.archived ? ' is-archived' : ''}`}
                    title={
                      channel.topic
                        ? `#${channel.name} — ${channel.topic}${channel.archived ? ' (archived)' : ''}`
                        : `#${channel.name}${channel.archived ? ' (archived)' : ''}`
                    }
                    onClick={() => onSelectChannel(channel.name)}
                  >
                    <span className="swarm-channel__name">#{channel.name}</span>
                    <span className="swarm-channel__count" title={`${channel.subscriber_count} subscribers`}>
                      {channel.subscriber_count}
                    </span>
                    {channel.archived && <span className="swarm-channel__archived">archived</span>}
                  </button>
                )
              })}
            </div>
          )}

          <div className="swarm-board__feed-stage">
            <div
              className="swarm-board__feed"
              ref={feedRef}
              onScroll={onFeedScroll}
              role="log"
              aria-live="polite"
              aria-relevant="additions"
            >
              {filteredBoardMessages.length === 0 ? (
                <p className="swarm-board__empty">
                  {isBoardFiltering
                    ? 'No board messages match your search.'
                    : selectedChannel === '*'
                      ? 'No board messages posted yet.'
                      : `No messages in #${selectedChannel} yet.`}
                </p>
              ) : (
                <>
                  {hiddenEarlierCount > 0 && (
                    <div className="swarm-board__earlier">
                      <button
                        type="button"
                        className="swarm-btn swarm-btn--ghost swarm-btn--xs"
                        onClick={handleShowEarlierMessages}
                      >
                        Show earlier posts ({hiddenEarlierCount})
                      </button>
                    </div>
                  )}
                  {windowedMessages.map(message => {
                    const authorMember = memberByName.get(message.agent)
                    const rowProps: SwarmBoardPostRowProps = {
                      message,
                      ...(authorMember ? { authorMember } : {}),
                      hasCustomChannels,
                      highlighted: highlightedSeq === message.seq,
                      expanded: Boolean(expandedSeqs?.has(message.seq)),
                      onInsertSeqRef,
                      onInsertMention,
                      onSelectChannel,
                      onJumpToSeq,
                      ...(onToggleExpandMessage ? { onToggleExpandMessage } : {}),
                    }
                    return memoizeRows ? (
                      <SwarmBoardPostRow key={message.seq} {...rowProps} />
                    ) : (
                      renderSwarmBoardPostInline(rowProps)
                    )
                  })}
                </>
              )}
            </div>
            {onJumpToLatest && (
              <SwarmJumpLatestButton
                visible={showJumpLatest}
                unseenCount={unseenLatestCount}
                onClick={onJumpToLatest}
              />
            )}
          </div>

          {/* Clean Bottom Composer with @ Autocomplete Popover */}
          <form
            className="swarm-board__composer"
            aria-label="Post to Immortals Board"
            onSubmit={onSubmitPost}
          >
            {mentionCandidates.length > 0 && (
              <div
                className="swarm-mentions-pop"
                role="listbox"
                aria-label="Mention an Immortals member"
              >
                {mentionCandidates.map((candidate, index) => (
                  <button
                    key={candidate.name}
                    type="button"
                    role="option"
                    aria-selected={index === activeMentionIndex}
                    className={`swarm-mentions-pop__row${index === activeMentionIndex ? ' is-on' : ''}`}
                    onMouseDown={event => {
                      event.preventDefault()
                      if (onPickMentionCandidate) onPickMentionCandidate(candidate.name)
                      else onInsertMention(candidate.name)
                    }}
                  >
                    <span className="swarm-mentions-pop__name">@{candidate.name}</span>
                    <span className="swarm-mentions-pop__meta">{candidate.detail}</span>
                  </button>
                ))}
              </div>
            )}

            <div className="swarm-pill-composer swarm-board__composer-row">
              {hasCustomChannels && (
                <span className="swarm-board__target-channel" title={`Posting onto #${postChannel}`}>
                  #{postChannel}
                </span>
              )}
              <textarea
                ref={composerInputRef}
                rows={1}
                className="swarm-board__textarea swarm-pill-composer__input"
                aria-label={`Post message to #${postChannel}`}
                placeholder={
                  postChannelArchived
                    ? `Channel #${postChannel} is archived (read-only)`
                    : `Message #${postChannel} — type @ to mention…`
                }
                value={postDraft}
                disabled={!online || Boolean(busyAction) || postChannelArchived}
                onChange={event => onChangePostDraft(event.target.value)}
                onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                  if (mentionCandidates.length > 0) {
                    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                      event.preventDefault()
                      onMoveMentionIndex?.(event.key === 'ArrowDown' ? 1 : -1, mentionCandidates.length)
                      return
                    }
                    if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey)) {
                      const picked = mentionCandidates[activeMentionIndex]
                      if (picked) {
                        event.preventDefault()
                        if (onPickMentionCandidate) onPickMentionCandidate(picked.name)
                        else onInsertMention(picked.name)
                        return
                      }
                    }
                    if (event.key === 'Escape') {
                      event.preventDefault()
                      event.stopPropagation()
                      onDismissMentions?.()
                      return
                    }
                  }
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    onSubmitPost()
                  } else if (event.key === 'Escape' && postDraft) {
                    event.preventDefault()
                    event.stopPropagation()
                  }
                }}
              />
              <button
                type="submit"
                className="swarm-btn swarm-btn--primary swarm-pill-composer__send"
                aria-label="Post message"
                title="Post message"
                disabled={!online || Boolean(busyAction) || postChannelArchived || !postDraft.trim()}
              >
                <Icon name="arrowUp" size={13} />
              </button>
            </div>
          </form>
        </section>
      </div>
    </section>
  )
}

/**
 * Resolve the model catalog for the Immortals member model switcher.
 * When connected to an external ACP runtime (such as Antigravity) that advertises
 * its own models, present those runtime models instead of native Xerxes models.
 */
export function resolveSwarmBoardModels(
  snap: Pick<Snapshot, 'acpBinding' | 'models'> | undefined,
  overrideModels?: readonly ModelChoice[],
): readonly ModelChoice[] {
  if (overrideModels !== undefined) return overrideModels
  const binding = snap?.acpBinding
  if (isExternalAcpBinding(binding) && binding.models?.length) {
    const providerLabel = binding.name.trim() || binding.provider
    return binding.models.map(option => ({
      id: option.id,
      provider: providerLabel,
      ...(option.name && option.name !== option.id ? { label: option.name } : {}),
      ...(option.description ? { hint: option.description } : {}),
    }))
  }
  return snap?.models ?? []
}

export function SwarmBoard(props: SwarmBoardProps): ReactElement {
  const { snap } = props
  const status = props.status !== undefined ? props.status : (snap?.swarm ?? null)
  const messages = props.messages !== undefined ? props.messages : (snap?.swarmMessages ?? [])
  const externalInspect = props.inspect !== undefined ? props.inspect : (snap?.swarmInspect ?? null)
  const externalAcp = isExternalAcpBinding(snap?.acpBinding)
  const models = useMemo(
    () => resolveSwarmBoardModels(snap, props.models),
    [snap?.acpBinding, snap?.models, props.models],
  )
  const online = props.online !== undefined ? props.online : snap ? snap.connection === 'online' : true
  const loading = props.loading !== undefined ? props.loading : Boolean(snap?.swarmLoading)
  const externalError = props.error !== undefined ? props.error : (snap?.swarmError ?? null)
  const actions = props.actions ?? DEFAULT_STORE_ACTIONS

  const [activeTab, setActiveTab] = useState<SwarmSurfaceTab>(props.initialTab ?? 'overview')
  const [selectedChannel, setSelectedChannel] = useState<string>(props.initialChannel ?? '*')
  const [boardQuery, setBoardQuery] = useState<string>('')
  const [boardFilter, setBoardFilter] = useState<SwarmBoardQuickFilter>('all')
  const [boardFilterOpen, setBoardFilterOpen] = useState<boolean>(false)
  const [feedbackOpen, setFeedbackOpen] = useState<boolean>(props.initialFeedbackOpen ?? false)
  const [feedbackDraft, setFeedbackDraft] = useState<string>(props.initialFeedbackDraft ?? '')
  const [postDraft, setPostDraft] = useState<string>(props.initialPostDraft ?? '')
  const [mentionIndex, setMentionIndex] = useState<number>(0)
  const [mentionsDismissedFor, setMentionsDismissedFor] = useState<string>('')
  const [switchingMember, setSwitchingMember] = useState<string>(props.initialModelSwitchMember ?? '')
  const [modelDrafts, setModelDrafts] = useState<Readonly<Record<string, string>>>(
    props.initialModelDrafts ?? {},
  )
  const [inspectedMember, setInspectedMember] = useState<string>(
    props.initialInspectMember ?? snap?.swarmInspectTarget ?? '',
  )
  const [localInspect, setLocalInspect] = useState<SwarmMemberInspectResult | null>(null)
  const [inspectLoading, setInspectLoading] = useState<boolean>(false)
  const [busyAction, setBusyAction] = useState<string>('')
  const [localError, setLocalError] = useState<string>('')
  const [highlightedSeq, setHighlightedSeq] = useState<number | null>(null)
  const [expandedSeqs, setExpandedSeqs] = useState<ReadonlySet<number>>(
    () => new Set(props.initialExpandedSeqs ?? []),
  )
  const [memberFilter, setMemberFilter] = useState<SwarmMemberFilter>(props.initialMemberFilter ?? 'all')
  const [memberQuery, setMemberQuery] = useState<string>(props.initialMemberQuery ?? '')
  const [visibleMessageLimit, setVisibleMessageLimit] = useState<number>(SWARM_BOARD_TAIL_LIMIT)
  const [armedKillTarget, setArmedKillTarget] = useState<string | null>(props.initialArmedKillTarget ?? null)
  const [isSplitView, setIsSplitView] = useState<boolean>(Boolean(props.initialSplitView))

  const boardRef = useRef<HTMLElement>(null)
  const feedRef = useRef<HTMLDivElement>(null)
  const composerInputRef = useRef<HTMLTextAreaElement>(null)
  const modelAnchorRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const node = boardRef.current
    if (!node) return
    const update = (): void => {
      const width = node.getBoundingClientRect().width
      if (width > 0) setIsSplitView(width >= 780)
    }
    update()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(update)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!armedKillTarget) return
    const timer = setTimeout(() => setArmedKillTarget(null), 6000)
    return () => clearTimeout(timer)
  }, [armedKillTarget])

  useEffect(() => {
    if (!snap || props.actions || !online) return
    void store.refreshSwarm()
  }, [snap?.sessionKey, snap?.currentId, online, Boolean(props.actions)])

  useEffect(() => {
    if (!switchingMember || !snap || props.actions || !online || externalAcp || models.length > 0) return
    store.loadModels()
  }, [switchingMember, snap, Boolean(props.actions), online, externalAcp, models.length])

  useEffect(() => {
    const target = snap?.swarmInspectTarget?.trim()
    if (!target) return
    setInspectedMember(target)
    if (!online || !actions.inspect) return
    if (externalInspect?.member.name === target) return
    setInspectLoading(true)
    void Promise.resolve(actions.inspect(target))
      .then(result => {
        if (result) setLocalInspect(result)
      })
      .catch(() => {
        // Keep roster summary visible if daemon does not support swarm.inspect yet.
      })
      .finally(() => {
        setInspectLoading(false)
      })
  }, [snap?.swarmInspectTarget, snap?.swarmOpenRevision])

  const channels = useMemo(
    () => swarmChannelsOf(status, messages),
    [status, messages],
  )

  const members = useMemo(
    () => sortSwarmMembers(status?.members ?? []),
    [status?.members],
  )

  const counts = useMemo(
    () => swarmMemberCounts(status?.members ?? []),
    [status?.members],
  )

  const visibleMessages = useMemo(
    () =>
      selectedChannel === '*'
        ? messages
        : messages.filter(message => message.channel === selectedChannel),
    [messages, selectedChannel],
  )

  const feedScroll = useFollowLatestScroll({
    ref: feedRef,
    itemCount: visibleMessages.length,
    resetKey: `${snap?.currentId ?? ''}:${selectedChannel}`,
    enabled: shouldEnableSwarmFeedScroll(activeTab, isSplitView, Boolean(inspectedMember)),
  })

  // Expand tail-windowed board posts when in-page Find (`xerxes:find`) opens.
  useEffect(() => {
    if (visibleMessages.length <= visibleMessageLimit || typeof window === 'undefined') return
    const expandAll = (): void => {
      setVisibleMessageLimit(current => Math.max(current, visibleMessages.length))
    }
    window.addEventListener('xerxes:find', expandAll)
    return () => window.removeEventListener('xerxes:find', expandAll)
  }, [visibleMessages.length, visibleMessageLimit])

  const postChannel = selectedChannel === '*' ? 'main' : selectedChannel
  const postChannelArchived = channels.find(entry => entry.name === postChannel)?.archived === true

  const runAction = async (
    label: string,
    work: () => Promise<boolean | void> | boolean | void,
  ): Promise<boolean> => {
    if (busyAction) return false
    setBusyAction(label)
    setLocalError('')
    try {
      const outcome = await work()
      return outcome !== false
    } catch (failure) {
      setLocalError(desktopError(failure))
      return false
    } finally {
      setBusyAction('')
    }
  }

  const pickChannel = useCallback((nextChannel: string): void => {
    setSelectedChannel(nextChannel)
    setVisibleMessageLimit(SWARM_BOARD_TAIL_LIMIT)
    setActiveTab('board')
    if (nextChannel !== '*' && online) {
      void Promise.resolve(actions.refresh(nextChannel)).catch(() => {})
    }
  }, [online, actions])

  const insertMention = useCallback((name: string): void => {
    const token = `@${name} `
    setPostDraft(current => {
      if (!current.trim()) return token
      if (current.includes(`@${name}`)) return current
      return `${current.replace(/\s*$/, ' ')}${token}`
    })
    setMentionIndex(0)
    composerInputRef.current?.focus()
  }, [])

  const pickMentionCandidate = useCallback((name: string): void => {
    setPostDraft(current => applySwarmMentionCandidate(current, name))
    setMentionIndex(0)
    setMentionsDismissedFor('')
    composerInputRef.current?.focus()
  }, [])

  const openInspector = useCallback(async (name: string): Promise<void> => {
    setInspectedMember(name)
    if (!online || !actions.inspect) return
    setInspectLoading(true)
    try {
      const result = await actions.inspect(name)
      if (result) setLocalInspect(result)
    } catch {
      // Keep roster summary visible if daemon does not support swarm.inspect yet.
    } finally {
      setInspectLoading(false)
    }
  }, [online, actions])

  const closeInspector = useCallback((): void => {
    setInspectedMember('')
    setLocalInspect(null)
    if (snap && !props.actions) store.closeSwarmInspect()
  }, [snap, props.actions])

  const insertSeqRef = useCallback((seq: number): void => {
    setPostDraft(current => (current ? `${current.replace(/\s*$/, ' ')}#${seq} ` : `#${seq} `))
    composerInputRef.current?.focus()
  }, [])

  const messagesRef = useRef(messages)
  messagesRef.current = messages
  const selectedChannelRef = useRef(selectedChannel)
  selectedChannelRef.current = selectedChannel

  const jumpToSeq = useCallback((seq: number): void => {
    const currentMessages = messagesRef.current
    const targetIndex = currentMessages.findIndex(entry => entry.seq === seq)
    const targetMessage = targetIndex >= 0 ? currentMessages[targetIndex] : undefined
    if (targetMessage && selectedChannelRef.current !== '*' && targetMessage.channel !== selectedChannelRef.current) {
      setSelectedChannel('*')
    }
    if (targetIndex >= 0) {
      const neededFromTail = currentMessages.length - targetIndex
      setVisibleMessageLimit(current => Math.max(current, neededFromTail))
    }
    setActiveTab('board')
    setHighlightedSeq(seq)
    feedScroll.pauseFollow()
    const scrollCard = (): void => {
      const element =
        feedRef.current?.querySelector<HTMLElement>(`[data-seq="${seq}"]`) ??
        (typeof document !== 'undefined' ? document.getElementById(`swarm-msg-${seq}`) : null)
      element?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
    }
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(scrollCard)
    else scrollCard()
  }, [feedScroll.pauseFollow])

  const toggleExpandMessage = useCallback((seq: number): void => {
    setExpandedSeqs(current => {
      const next = new Set(current)
      if (next.has(seq)) next.delete(seq)
      else next.add(seq)
      return next
    })
  }, [])

  const showEarlierMessages = useCallback((): void => {
    setVisibleMessageLimit(current => current + SWARM_BOARD_TAIL_LIMIT)
  }, [])

  const toggleModelSwitch = useCallback((member: SwarmMemberSnapshot): void => {
    setSwitchingMember(current => (current === member.name ? '' : member.name))
  }, [])

  const cancelModelSwitch = useCallback((): void => {
    setSwitchingMember('')
  }, [])

  const changeModelDraft = useCallback((name: string, value: string): void => {
    setModelDrafts(current => ({ ...current, [name]: value }))
  }, [])

  const submitFeedback = async (event?: FormEvent): Promise<void> => {
    event?.preventDefault()
    const clean = feedbackDraft.trim()
    if (!clean || !online) return
    const ok = await runAction('feedback', () =>
      actions.post({ text: clean, channel: 'main', asFeedback: true }),
    )
    if (ok) {
      setFeedbackDraft('')
      setFeedbackOpen(false)
    }
  }

  const submitBoardPost = async (event?: FormEvent): Promise<void> => {
    event?.preventDefault()
    const clean = postDraft.trim()
    if (!clean || !online || postChannelArchived) return
    const ok = await runAction('post', () =>
      actions.post({ text: clean, channel: postChannel, asFeedback: false }),
    )
    if (ok) {
      setPostDraft('')
      setMentionIndex(0)
      setMentionsDismissedFor('')
      feedScroll.jumpToLatest('smooth')
    }
  }

  const pickMemberModel = async (
    member: SwarmMemberSnapshot,
    modelId: string,
  ): Promise<void> => {
    if (!online) return
    const clean = modelId.trim()
    const ok = await runAction(`model:${member.name}`, () =>
      actions.switchModel(member.name, clean || undefined),
    )
    if (ok) {
      setSwitchingMember('')
      setModelDrafts(current => ({ ...current, [member.name]: '' }))
    }
  }

  const submitModelSwitch = async (
    member: SwarmMemberSnapshot,
    event?: FormEvent,
  ): Promise<void> => {
    event?.preventDefault()
    if (!online) return
    const draft = (modelDrafts[member.name] ?? '').trim()
    await pickMemberModel(member, draft)
  }

  const runActionRef = useRef(runAction)
  runActionRef.current = runAction
  const actionsRef = useRef(actions)
  actionsRef.current = actions
  const openInspectorRef = useRef(openInspector)
  openInspectorRef.current = openInspector
  const pickMemberModelRef = useRef(pickMemberModel)
  pickMemberModelRef.current = pickMemberModel
  const submitModelSwitchRef = useRef(submitModelSwitch)
  submitModelSwitchRef.current = submitModelSwitch

  const handleKillMember = useCallback((name: string): void => {
    setArmedKillTarget(null)
    void runActionRef.current(`kill:${name}`, () => actionsRef.current.kill(name))
  }, [])
  const handleInspectMember = useCallback((name: string): void => {
    void openInspectorRef.current(name)
  }, [])
  const handlePickModel = useCallback((member: SwarmMemberSnapshot, modelId: string): void => {
    void pickMemberModelRef.current(member, modelId)
  }, [])
  const handleSubmitModelSwitch = useCallback((member: SwarmMemberSnapshot, event?: FormEvent): void => {
    void submitModelSwitchRef.current(member, event)
  }, [])

  const effectiveInspect = externalInspect ?? localInspect

  return (
    <SwarmBoardView
      status={status}
      channels={channels}
      members={members}
      counts={counts}
      messages={messages}
      visibleMessages={visibleMessages}
      visibleMessageLimit={visibleMessageLimit}
      onShowEarlierMessages={showEarlierMessages}
      memoizeRows
      models={models}
      modelsLoading={externalAcp ? false : Boolean(snap?.modelsLoading)}
      online={online}
      loading={loading}
      busyAction={busyAction}
      error={localError || externalError || ''}
      activeTab={activeTab}
      isSplitView={isSplitView}
      boardRef={boardRef}
      armedKillTarget={armedKillTarget}
      onArmKill={setArmedKillTarget}
      onCancelArmKill={() => setArmedKillTarget(null)}
      {...(props.initialGraphMode ? { initialGraphMode: props.initialGraphMode } : {})}
      selectedChannel={selectedChannel}
      highlightedSeq={highlightedSeq}
      expandedSeqs={expandedSeqs}
      memberFilter={memberFilter}
      memberQuery={memberQuery}
      boardQuery={boardQuery}
      boardFilter={boardFilter}
      boardFilterOpen={boardFilterOpen}
      feedbackOpen={feedbackOpen}
      feedbackDraft={feedbackDraft}
      postDraft={postDraft}
      mentionIndex={mentionIndex}
      mentionsDismissed={Boolean(mentionsDismissedFor) && mentionsDismissedFor === postDraft}
      switchingMember={switchingMember}
      modelDrafts={modelDrafts}
      inspectedMember={inspectedMember}
      inspectResult={effectiveInspect}
      inspectLoading={inspectLoading}
      feedRef={feedRef}
      showJumpLatest={!feedScroll.atBottom}
      unseenLatestCount={feedScroll.unseenCount}
      onFeedScroll={feedScroll.handleScroll}
      onJumpToLatest={() => feedScroll.jumpToLatest('smooth')}
      composerInputRef={composerInputRef}
      modelAnchorRef={modelAnchorRef}
      onSelectTab={setActiveTab}
      onRefresh={() =>
        void runAction('refresh', () =>
          actions.refresh(selectedChannel === '*' ? undefined : selectedChannel),
        )
      }
      onStopAll={() => {
        setArmedKillTarget(null)
        void runAction('kill:all', () => actions.kill('all'))
      }}
      onToggleFeedback={() => setFeedbackOpen(open => !open)}
      onChangeFeedbackDraft={setFeedbackDraft}
      onSubmitFeedback={event => void submitFeedback(event)}
      onCloseFeedback={() => setFeedbackOpen(false)}
      onSelectChannel={pickChannel}
      onInsertMention={insertMention}
      onPickMentionCandidate={pickMentionCandidate}
      onMoveMentionIndex={(delta, total) => {
        if (total <= 0) return
        setMentionIndex(current => ((current + delta) % total + total) % total)
      }}
      onDismissMentions={() => setMentionsDismissedFor(postDraft)}
      onInsertSeqRef={insertSeqRef}
      onJumpToSeq={jumpToSeq}
      onToggleExpandMessage={toggleExpandMessage}
      onChangeMemberFilter={setMemberFilter}
      onChangeMemberQuery={setMemberQuery}
      onChangeBoardQuery={setBoardQuery}
      onChangeBoardFilter={setBoardFilter}
      onToggleBoardFilterOpen={() => setBoardFilterOpen(open => !open)}
      onKillMember={handleKillMember}
      onInspectMember={handleInspectMember}
      onCloseInspect={closeInspector}
      onToggleModelSwitch={toggleModelSwitch}
      onPickModel={handlePickModel}
      onChangeModelDraft={changeModelDraft}
      onSubmitModelSwitch={handleSubmitModelSwitch}
      onCancelModelSwitch={cancelModelSwitch}
      onChangePostDraft={value => {
        setPostDraft(value)
        setMentionIndex(0)
        if (mentionsDismissedFor) setMentionsDismissedFor('')
      }}
      onSubmitPost={event => void submitBoardPost(event)}
    />
  )
}
