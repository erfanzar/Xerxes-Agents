// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure deterministic replay, layout, and animated GIF export engine for the
 * Xerxes Immortals Graph (`Battlefield` tactical imperial command map +
 * `Mentions` directed graph).
 *
 * Operates strictly on renderer `SwarmStatusSnapshot` / `SwarmBoardMessage`
 * snapshots — never imports runtime `src/swarm/` modules.
 */

import { downloadGifBytes, encodeAnimatedGif, type GifFrame } from './gifEncoder.js'
import {
  tokenizeSwarmMessage,
  type SwarmBoardMessage,
  type SwarmMemberRole,
  type SwarmMemberSnapshot,
  type SwarmStatusSnapshot,
} from './swarmState.js'
import {
  drawSimurghSpriteOnCanvas,
  resolveSimurghRole,
  type SimurghSpriteRole,
  type SimurghSpriteState,
} from './XerxesSprites.js'

export type SwarmGraphViewMode = 'battlefield' | 'mentions'
export type SwarmGraphModeInput = SwarmGraphViewMode | 'hive'

/** Normalize legacy `'hive'` mode input cleanly into `'battlefield'`. */
export function normalizeSwarmGraphMode(mode?: SwarmGraphModeInput): SwarmGraphViewMode {
  return mode === 'mentions' ? 'mentions' : 'battlefield'
}
export type SwarmReplayPacing = 'even' | 'realtime'
export type SwarmReplaySpeed = 0.5 | 1 | 2 | 4
export type SwarmGraphMemberState = 'working' | 'waiting' | 'stopped' | 'killed' | 'done' | 'error'

export interface SwarmGraphMemberNode {
  readonly name: string
  readonly role: SwarmMemberRole
  readonly spriteRole: SimurghSpriteRole
  readonly spriteState: SimurghSpriteState
  readonly state: SwarmGraphMemberState
  readonly model: string
  readonly isCoordinator: boolean
  readonly postCount: number
  readonly allCount: number
  readonly postedWithoutMention: boolean
  readonly receivedCount: number
  readonly isActiveSender: boolean
  readonly isActiveRecipient: boolean
  readonly lastSeq: number | null
  readonly error?: string
  readonly currentActivity?: string
}

export interface SwarmGraphEdge {
  readonly id: string
  readonly from: string
  readonly to: string
  readonly count: number
  readonly kind: 'mention' | 'broadcast'
  readonly lastSeq: number
  readonly isActive: boolean
}

export interface SwarmGraphBuckets {
  readonly coordinator: SwarmGraphMemberNode | null
  readonly working: readonly SwarmGraphMemberNode[]
  readonly waiting: readonly SwarmGraphMemberNode[]
  readonly done: readonly SwarmGraphMemberNode[]
  readonly error: readonly SwarmGraphMemberNode[]
}

export interface SwarmGraphFrame {
  readonly step: number
  readonly totalSteps: number
  readonly isLive: boolean
  readonly currentMessage: SwarmBoardMessage | null
  readonly activeSender: string | null
  readonly activeRecipients: readonly string[]
  readonly activeIsBroadcast: boolean
  readonly activeIsDone: boolean
  readonly members: readonly SwarmGraphMemberNode[]
  readonly edges: readonly SwarmGraphEdge[]
  readonly broadcastEdges: readonly SwarmGraphEdge[]
  readonly allCounts: Readonly<Record<string, number>>
  readonly postedWithoutMention: ReadonlySet<string>
  readonly buckets: SwarmGraphBuckets
  readonly doneCount: number
  readonly totalWorkers: number
  readonly postCount: number
  readonly elapsedLabel: string
}

export interface SwarmTimeline {
  readonly totalPosts: number
  readonly durationMs: number
  readonly durationLabel: string
  readonly timestampsMs: readonly (number | null)[]
  readonly frameAt: (step: number) => SwarmGraphFrame
  readonly stepDelayMs: (step: number, pacing: SwarmReplayPacing, speed: SwarmReplaySpeed) => number
}

export interface MentionsLayoutOptions {
  readonly hideInactive: boolean
  readonly expandAll: boolean
  readonly width: number
  readonly height: number
}

export interface PositionedMentionNode {
  readonly member: SwarmGraphMemberNode
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly rx: number
  readonly displayLabel: string
}

export interface PositionedMentionEdge {
  readonly edge: SwarmGraphEdge
  readonly path: string
  readonly sx: number
  readonly sy: number
  readonly cx: number
  readonly cy: number
  readonly ex: number
  readonly ey: number
  readonly labelX: number
  readonly labelY: number
  readonly strokeWidth: number
  readonly dashed: boolean
}

export interface MentionsGraphLayout {
  readonly width: number
  readonly height: number
  readonly nodes: readonly PositionedMentionNode[]
  readonly edges: readonly PositionedMentionEdge[]
  readonly hiddenCount: number
}

export type BattlefieldZoneKind = 'coordinator' | 'working' | 'waiting' | 'done' | 'error'
export type BattlefieldZoneTone = 'emerald' | 'accent' | 'slate' | 'rose' | 'gold'

export interface BattlefieldMemberSlot {
  readonly member: SwarmGraphMemberNode
  readonly x: number
  readonly y: number
  readonly spriteSize: number
  readonly displayLabel: string
  readonly caption?: string
}

export interface BattlefieldZone {
  readonly id: BattlefieldZoneKind
  readonly label: string
  readonly sublabel: string
  readonly statusTag: string
  readonly count: number
  readonly cx: number
  readonly cy: number
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly chamfer: number
  readonly radius: number
  readonly points: string
  readonly innerPoints: string
  readonly vertices: ReadonlyArray<readonly [number, number]>
  readonly innerVertices: ReadonlyArray<readonly [number, number]>
  readonly cornerPips: ReadonlyArray<readonly [number, number]>
  readonly tone: BattlefieldZoneTone
  readonly members: readonly SwarmGraphMemberNode[]
  readonly slots: readonly BattlefieldMemberSlot[]
  readonly overflowCount: number
  readonly progressRatio?: number
  readonly activeRatio?: number
}

export type BattlefieldDispatchKind = 'tasks_done' | 'message_sent' | 'waiting_on'

export interface BattlefieldDispatchVector {
  readonly id: string
  readonly kind: BattlefieldDispatchKind
  readonly from: BattlefieldZoneKind
  readonly to: BattlefieldZoneKind
  readonly path: string
  readonly sx: number
  readonly sy: number
  readonly cx: number
  readonly cy: number
  readonly ex: number
  readonly ey: number
  readonly dashed: boolean
}

export interface BattlefieldRangeRing {
  readonly cx: number
  readonly cy: number
  readonly r: number
  readonly dashed: boolean
}

export interface BattlefieldGraphLayout {
  readonly width: number
  readonly height: number
  readonly mapPadding: number
  readonly gridStep: number
  readonly commandCenter: readonly [number, number]
  readonly rings: readonly BattlefieldRangeRing[]
  readonly zones: readonly BattlefieldZone[]
  readonly dispatches: readonly BattlefieldDispatchVector[]
}

/** Backwards-compatible type aliases for pre-Battlefield callers. */
export type HiveCellKind = BattlefieldZoneKind
export type HiveCellTone = BattlefieldZoneTone
export type HiveMemberSlot = BattlefieldMemberSlot
export type HiveCell = BattlefieldZone
export type HiveFlightKind = BattlefieldDispatchKind
export type HiveFlightArc = BattlefieldDispatchVector
export type HiveGraphLayout = BattlefieldGraphLayout

export interface CanvasLike {
  width: number
  height: number
  getContext(contextId: '2d'): CanvasRenderingContext2D | null
}

export interface ExportSwarmGraphGifOptions {
  readonly timeline: SwarmTimeline
  readonly mode: SwarmGraphModeInput
  readonly pacing: SwarmReplayPacing
  readonly speed: SwarmReplaySpeed
  readonly hideInactive: boolean
  readonly expandAll: boolean
  readonly title?: string
  readonly width?: number
  readonly height?: number
  readonly maxFrames?: number
  readonly filename?: string
  readonly createCanvas?: (width: number, height: number) => CanvasLike | null
  readonly download?: (bytes: Uint8Array, filename: string) => boolean
}

export interface ExportSwarmGraphGifResult {
  readonly ok: boolean
  readonly frames: number
  readonly bytes: number
  readonly filename: string
  readonly error?: string
}

const DAY_MS = 86_400_000

/** Parse board timestamps (`YYYY-MM-DD HH:MM:SS`, ISO 8601, or `HH:MM[:SS]`) into epoch/relative ms. */
export function parseSwarmTimestampMs(raw: string): number | null {
  const trimmed = raw.trim()
  if (!trimmed) return null

  const clockOnly = /^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?$/.exec(trimmed)
  if (clockOnly) {
    const h = Number.parseInt(clockOnly[1] ?? '0', 10)
    const m = Number.parseInt(clockOnly[2] ?? '0', 10)
    const s = Number.parseInt(clockOnly[3] ?? '0', 10)
    const ms = Number.parseInt((clockOnly[4] ?? '0').padEnd(3, '0'), 10)
    if (h >= 0 && h < 24 && m >= 0 && m < 60 && s >= 0 && s < 60) {
      return ((h * 3600 + m * 60 + s) * 1000) + ms
    }
    return null
  }

  const normalized = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(trimmed)
    ? trimmed.replace(/\s+/, 'T')
    : trimmed
  const parsed = Date.parse(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

/** Format elapsed duration cleanly (`"21h 02m"`, `"14m 08s"`, `"42s"`, `"0s"`). */
export function formatSwarmDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '0s'
  const totalSec = Math.max(0, Math.round(ms / 1000))
  if (totalSec < 60) return `${totalSec}s`
  const totalMin = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  if (totalMin < 60) {
    return `${totalMin}m ${String(sec).padStart(2, '0')}s`
  }
  const hours = Math.floor(totalMin / 60)
  const mins = totalMin % 60
  return `${hours}h ${String(mins).padStart(2, '0')}m`
}

/** Truncate an agent identifier smoothly while preserving legibility. */
export function truncateLabel(label: string, maxChars: number): string {
  if (maxChars <= 1) return label.slice(0, 1)
  if (label.length <= maxChars) return label
  return `${label.slice(0, maxChars - 1)}…`
}

function inferRoleFromName(name: string): SwarmMemberRole {
  if (name === 'coordinator') return 'coordinator'
  if (name === 'narrator' || /^narrator-\d+$/.test(name)) return 'narrator'
  if (name === 'monitor' || /^monitor-\d+$/.test(name)) return 'monitor'
  return 'worker'
}

interface ParsedMessageSignals {
  readonly mentions: readonly string[]
  readonly isBroadcast: boolean
  readonly isDone: boolean
}

function parseMessageSignals(message: SwarmBoardMessage): ParsedMessageSignals {
  const mentionSet = new Set<string>()
  for (const name of message.mentions) {
    const clean = name.trim()
    if (clean && clean.toLowerCase() !== 'all' && clean !== message.agent) {
      mentionSet.add(clean)
    }
  }
  let isBroadcast = message.broadcast === true
  let isDone = false

  for (const token of tokenizeSwarmMessage(message.text, { allowBold: false, allowItalic: false, allowStrike: false })) {
    if (token.kind === 'broadcast') {
      isBroadcast = true
    } else if (token.kind === 'done') {
      isDone = true
    } else if (token.kind === 'mention') {
      const clean = token.name.trim()
      if (clean && clean.toLowerCase() !== 'all' && clean !== message.agent) {
        mentionSet.add(clean)
      }
    }
  }

  return {
    mentions: [...mentionSet],
    isBroadcast,
    isDone,
  }
}

function buildMonotonicTimestamps(messages: readonly SwarmBoardMessage[]): readonly (number | null)[] {
  const out: (number | null)[] = []
  let dayOffset = 0
  let prevClockOnly = false
  let prevMs: number | null = null

  for (const msg of messages) {
    const isClock = /^\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?$/.test(msg.time.trim())
    const parsed = parseSwarmTimestampMs(msg.time)
    if (parsed === null) {
      out.push(null)
      continue
    }
    let adjusted = parsed + (isClock ? dayOffset : 0)
    if (isClock && prevClockOnly && prevMs !== null && adjusted + 60_000 < prevMs) {
      dayOffset += DAY_MS
      adjusted += DAY_MS
    }
    prevMs = adjusted
    prevClockOnly = isClock
    out.push(adjusted)
  }
  return out
}

function mapStateToSpriteState(
  state: SwarmGraphMemberState,
  isCoordinator: boolean,
  hasWorkingPeers: boolean,
  isActiveSender: boolean,
): SimurghSpriteState {
  if (state === 'error') return 'error'
  if (isCoordinator) {
    if (state === 'killed') return 'killed'
    return hasWorkingPeers || isActiveSender ? 'working' : 'waiting'
  }
  if (state === 'working') return 'working'
  if (state === 'waiting') return 'waiting'
  if (state === 'stopped') return 'stopped'
  return 'killed'
}

/**
 * Build a deterministic timeline over board messages and live swarm status so
 * the Graph UI and GIF exporter can scrub smoothly across any step `0..N`.
 */
export function computeSwarmTimeline(
  status: SwarmStatusSnapshot | null | undefined,
  messages: readonly SwarmBoardMessage[],
): SwarmTimeline {
  const sortedMessages = [...messages].sort((a, b) => a.seq - b.seq)
  const parsedSignals = sortedMessages.map(parseMessageSignals)
  const timestampsMs = buildMonotonicTimestamps(sortedMessages)

  let firstTs: number | null = null
  let lastTs: number | null = null
  for (const ts of timestampsMs) {
    if (ts === null) continue
    if (firstTs === null) firstTs = ts
    lastTs = ts
  }
  const durationMs = firstTs !== null && lastTs !== null ? Math.max(0, lastTs - firstTs) : 0
  const durationLabel = formatSwarmDuration(durationMs)

  const rosterByName = new Map<string, SwarmMemberSnapshot>()
  for (const member of status?.members ?? []) {
    rosterByName.set(member.name, member)
  }

  // Ensure all participants from board messages have a stable descriptor.
  const allNames = new Set<string>(rosterByName.keys())
  for (let i = 0; i < sortedMessages.length; i += 1) {
    const msg = sortedMessages[i]!
    const sig = parsedSignals[i]!
    if (msg.agent) allNames.add(msg.agent)
    for (const target of sig.mentions) allNames.add(target)
  }

  const coordinatorName =
    [...rosterByName.values()].find(m => m.role === 'coordinator')?.name ??
    (allNames.has('coordinator') ? 'coordinator' : null)

  const frameCache = new Map<number, SwarmGraphFrame>()

  const frameAt = (rawStep: number): SwarmGraphFrame => {
    const totalSteps = sortedMessages.length
    const step = Math.max(0, Math.min(totalSteps, Math.trunc(rawStep)))
    const cached = frameCache.get(step)
    if (cached) return cached

    const isLive = step === totalSteps
    const currentMessage = step > 0 ? (sortedMessages[step - 1] ?? null) : null
    const currentSignal = step > 0 ? (parsedSignals[step - 1] ?? null) : null

    const appeared = new Set<string>()
    if (coordinatorName) appeared.add(coordinatorName)
    if (isLive) {
      for (const name of allNames) appeared.add(name)
    }

    const postCount = new Map<string, number>()
    const receivedCount = new Map<string, number>()
    const allCounts: Record<string, number> = {}
    const postedWithoutMention = new Set<string>()
    const lastSeqByMember = new Map<string, number>()
    const edgeMap = new Map<string, { from: string; to: string; count: number; lastSeq: number }>()
    const replayState = new Map<string, SwarmGraphMemberState>()
    const donePosted = new Set<string>()

    if (coordinatorName) {
      replayState.set(coordinatorName, 'waiting')
    }

    for (let i = 0; i < step; i += 1) {
      const msg = sortedMessages[i]!
      const sig = parsedSignals[i]!
      const sender = msg.agent
      if (!sender) continue

      appeared.add(sender)
      postCount.set(sender, (postCount.get(sender) ?? 0) + 1)
      lastSeqByMember.set(sender, msg.seq)

      if (sig.isBroadcast) {
        allCounts[sender] = (allCounts[sender] ?? 0) + 1
      }
      if (sig.mentions.length === 0 && !sig.isBroadcast) {
        postedWithoutMention.add(sender)
      }

      if (sig.isDone && sender !== coordinatorName) {
        donePosted.add(sender)
        replayState.set(sender, 'done')
      } else if (sender !== coordinatorName) {
        replayState.set(sender, sig.mentions.length > 0 ? 'waiting' : 'working')
      } else {
        replayState.set(sender, 'working')
      }

      for (const target of sig.mentions) {
        appeared.add(target)
        receivedCount.set(target, (receivedCount.get(target) ?? 0) + 1)
        const key = `${sender}->${target}`
        const existing = edgeMap.get(key)
        if (existing) {
          existing.count += 1
          existing.lastSeq = msg.seq
        } else {
          edgeMap.set(key, { from: sender, to: target, count: 1, lastSeq: msg.seq })
        }
        if (target !== coordinatorName) {
          // Being explicitly mentioned wakes a worker back up.
          donePosted.delete(target)
          replayState.set(target, 'working')
        }
      }

      if (sig.isBroadcast) {
        for (const name of appeared) {
          if (name === sender || name === coordinatorName) continue
          if (replayState.get(name) !== 'done') {
            replayState.set(name, 'working')
          }
        }
      }
    }

    const activeSender = currentMessage?.agent ?? null
    const activeRecipients = currentSignal?.mentions ?? []
    const activeRecipientSet = new Set(activeRecipients)
    const activeIsBroadcast = currentSignal?.isBroadcast ?? false
    const activeIsDone = currentSignal?.isDone ?? false

    // Resolve per-member state.
    const resolvedStates = new Map<string, SwarmGraphMemberState>()
    for (const name of appeared) {
      const live = rosterByName.get(name)
      const isCoord = name === coordinatorName || live?.role === 'coordinator'
      if (isLive && live) {
        if (live.killed || live.state === 'killed') {
          resolvedStates.set(name, isCoord ? 'killed' : 'done')
        } else if (live.error && !isCoord) {
          resolvedStates.set(name, 'error')
        } else if (live.state === 'working') {
          resolvedStates.set(name, 'working')
        } else if (live.state === 'stopped') {
          resolvedStates.set(name, donePosted.has(name) && !isCoord ? 'done' : 'waiting')
        } else {
          // live.state === 'waiting'
          resolvedStates.set(name, donePosted.has(name) && !isCoord ? 'done' : 'waiting')
        }
      } else {
        const inferred = replayState.get(name) ?? (donePosted.has(name) && !isCoord ? 'done' : 'waiting')
        resolvedStates.set(name, inferred)
      }
    }

    const anyWorkerWorking = [...appeared].some(
      name => name !== coordinatorName && resolvedStates.get(name) === 'working',
    )

    const members: SwarmGraphMemberNode[] = [...appeared]
      .sort((a, b) => {
        if (a === coordinatorName) return -1
        if (b === coordinatorName) return 1
        return a.localeCompare(b)
      })
      .map(name => {
        const live = rosterByName.get(name)
        const role = live?.role ?? inferRoleFromName(name)
        const isCoordinator = role === 'coordinator' || name === coordinatorName
        const state = resolvedStates.get(name) ?? 'waiting'
        const isActiveSender = activeSender === name
        const isActiveRecipient = activeRecipientSet.has(name) || (activeIsBroadcast && activeSender !== null && name !== activeSender)
        const spriteRole = resolveSimurghRole(name, role)
        const spriteState = mapStateToSpriteState(state, isCoordinator, anyWorkerWorking, isActiveSender)
        return {
          name,
          role,
          spriteRole,
          spriteState,
          state,
          model: live?.model_display_name || live?.model || '',
          isCoordinator,
          postCount: postCount.get(name) ?? 0,
          allCount: allCounts[name] ?? 0,
          postedWithoutMention: postedWithoutMention.has(name),
          receivedCount: receivedCount.get(name) ?? 0,
          isActiveSender,
          isActiveRecipient,
          lastSeq: lastSeqByMember.get(name) ?? live?.last_post_seq ?? null,
          ...(live?.error ? { error: live.error } : {}),
          ...(live?.current_activity ? { currentActivity: live.current_activity } : {}),
        }
      })

    const edges: SwarmGraphEdge[] = [...edgeMap.values()]
      .sort((a, b) => a.from.localeCompare(b.from) || a.to.localeCompare(b.to))
      .map(entry => ({
        id: `m:${entry.from}->${entry.to}`,
        from: entry.from,
        to: entry.to,
        count: entry.count,
        kind: 'mention',
        lastSeq: entry.lastSeq,
        isActive:
          currentMessage !== null &&
          entry.from === currentMessage.agent &&
          activeRecipientSet.has(entry.to),
      }))

    const broadcastEdges: SwarmGraphEdge[] = []
    const appearedList = members.map(m => m.name)
    for (const [author, count] of Object.entries(allCounts)) {
      if (count <= 0) continue
      for (const target of appearedList) {
        if (target === author) continue
        broadcastEdges.push({
          id: `b:${author}->${target}`,
          from: author,
          to: target,
          count,
          kind: 'broadcast',
          lastSeq: lastSeqByMember.get(author) ?? step,
          isActive: activeIsBroadcast && activeSender === author,
        })
      }
    }

    let coordinator: SwarmGraphMemberNode | null = null
    const working: SwarmGraphMemberNode[] = []
    const waiting: SwarmGraphMemberNode[] = []
    const done: SwarmGraphMemberNode[] = []
    const error: SwarmGraphMemberNode[] = []

    for (const member of members) {
      if (member.isCoordinator) {
        coordinator = member
      } else if (member.state === 'error') {
        error.push(member)
      } else if (member.state === 'working') {
        working.push(member)
      } else if (member.state === 'done' || member.state === 'killed') {
        done.push(member)
      } else {
        waiting.push(member)
      }
    }

    const stepTs = step > 0 ? (timestampsMs[step - 1] ?? null) : firstTs
    const elapsedMs = firstTs !== null && stepTs !== null ? Math.max(0, stepTs - firstTs) : 0
    const elapsedLabel = step === totalSteps ? durationLabel : formatSwarmDuration(elapsedMs)

    const frame: SwarmGraphFrame = {
      step,
      totalSteps,
      isLive,
      currentMessage,
      activeSender,
      activeRecipients,
      activeIsBroadcast,
      activeIsDone,
      members,
      edges,
      broadcastEdges,
      allCounts,
      postedWithoutMention,
      buckets: { coordinator, working, waiting, done, error },
      doneCount: done.length,
      totalWorkers: working.length + waiting.length + done.length + error.length,
      postCount: step,
      elapsedLabel,
    }

    frameCache.set(step, frame)
    return frame
  }

  const stepDelayMs = (
    step: number,
    pacing: SwarmReplayPacing,
    speed: SwarmReplaySpeed,
  ): number => {
    const safeSpeed = speed > 0 ? speed : 1
    if (pacing === 'even' || step <= 1 || step > sortedMessages.length) {
      return Math.max(90, Math.round(520 / safeSpeed))
    }
    const prevTs = timestampsMs[step - 2] ?? null
    const currTs = timestampsMs[step - 1] ?? null
    if (prevTs === null || currTs === null) {
      return Math.max(90, Math.round(520 / safeSpeed))
    }
    const delta = Math.max(0, currTs - prevTs)
    // Compress multi-hour real-time gaps logarithmically so replay never stalls.
    const scaled =
      delta <= 2000
        ? Math.max(180, delta * 0.65)
        : 1300 + Math.log10(delta / 2000 + 1) * 750
    return Math.max(90, Math.min(2600, Math.round(scaled / safeSpeed)))
  }

  return {
    totalPosts: sortedMessages.length,
    durationMs,
    durationLabel,
    timestampsMs,
    frameAt,
    stepDelayMs,
  }
}

function pillEdgeAnchor(
  from: PositionedMentionNode,
  to: PositionedMentionNode,
  pad: number,
): readonly [number, number] {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const len = Math.hypot(dx, dy)
  if (len < 0.001) return [from.x, from.y]
  const ux = dx / len
  const uy = dy / len
  const hw = from.width / 2 + pad
  const hh = from.height / 2 + pad
  const tx = Math.abs(ux) > 1e-4 ? hw / Math.abs(ux) : Number.POSITIVE_INFINITY
  const ty = Math.abs(uy) > 1e-4 ? hh / Math.abs(uy) : Number.POSITIVE_INFINITY
  const t = Math.min(tx, ty)
  return [from.x + ux * t, from.y + uy * t]
}

/**
 * Compute deterministic non-overlapping Mentions directed graph coordinates
 * matching the top-Coordinator + balanced staggered worker rows layout.
 */
export function layoutMentionsGraph(
  frame: SwarmGraphFrame,
  options: MentionsLayoutOptions,
): MentionsGraphLayout {
  const width = Math.max(360, Math.round(options.width))
  const minHeight = Math.max(280, Math.round(options.height))

  const activeEdgesAll = options.expandAll
    ? [...frame.edges, ...frame.broadcastEdges]
    : [...frame.edges]

  const connectedNames = new Set<string>()
  for (const edge of activeEdgesAll) {
    connectedNames.add(edge.from)
    connectedNames.add(edge.to)
  }

  const visibleMembers = frame.members.filter(member => {
    if (!options.hideInactive) return true
    if (member.isCoordinator) return true
    if (member.state === 'working' || member.state === 'waiting' || member.state === 'error') {
      return true
    }
    return connectedNames.has(member.name) || member.postCount > 0
  })

  const hiddenCount = Math.max(0, frame.members.length - visibleMembers.length)
  const coordinator = visibleMembers.find(m => m.isCoordinator) ?? null
  const workers = visibleMembers.filter(m => !m.isCoordinator)

  const nodes: PositionedMentionNode[] = []
  const nodeByName = new Map<string, PositionedMentionNode>()

  const padX = 22
  const usableW = Math.max(300, width - padX * 2)

  if (coordinator) {
    const coordW = Math.min(168, Math.max(132, Math.round(coordinator.name.length * 7.6 + 52)))
    const coordNode: PositionedMentionNode = {
      member: coordinator,
      x: width / 2,
      y: 44,
      width: coordW,
      height: 32,
      rx: 16,
      displayLabel: truncateLabel(coordinator.name, 16),
    }
    nodes.push(coordNode)
    nodeByName.set(coordinator.name, coordNode)
  }

  const workerCount = workers.length
  let totalHeight = minHeight

  if (workerCount > 0) {
    const maxColsByWidth = Math.max(2, Math.min(5, Math.floor((usableW + 14) / 132)))
    const cols =
      workerCount <= maxColsByWidth
        ? workerCount
        : workerCount <= 8
          ? Math.min(4, maxColsByWidth)
          : maxColsByWidth

    const gapX = cols >= 5 ? 12 : cols === 4 ? 16 : 22
    const pillW = Math.min(152, Math.max(96, Math.floor((usableW - gapX * (cols - 1)) / cols)))
    const pillH = 28
    const maxChars = Math.max(8, Math.floor((pillW - 40) / 6.8))
    const staggerY = cols >= 3 && workerCount >= 6 ? 12 : 0
    const rowGap = Math.max(54, pillH + staggerY + 24)
    const startY = coordinator ? 116 : 46
    const rows = Math.ceil(workerCount / cols)

    for (let index = 0; index < workerCount; index += 1) {
      const worker = workers[index]!
      const row = Math.floor(index / cols)
      const col = index % cols
      const rowCount = row === rows - 1 ? workerCount - row * cols : cols
      const rowWidth = rowCount * pillW + (rowCount - 1) * gapX
      const rowStartX = (width - rowWidth) / 2 + pillW / 2
      const x = rowStartX + col * (pillW + gapX)
      const colStagger = rowCount >= 3 && col % 2 === 1 ? staggerY : 0
      const y = startY + row * rowGap + colStagger

      const positioned: PositionedMentionNode = {
        member: worker,
        x: Math.round(x * 10) / 10,
        y: Math.round(y * 10) / 10,
        width: pillW,
        height: pillH,
        rx: 14,
        displayLabel: truncateLabel(worker.name, maxChars),
      }
      nodes.push(positioned)
      nodeByName.set(worker.name, positioned)
    }

    const maxNodeY = nodes.reduce((max, n) => Math.max(max, n.y + n.height / 2), 0)
    totalHeight = Math.max(minHeight, Math.ceil(maxNodeY + 30))
  }

  const visibleEdges = activeEdgesAll.filter(
    edge => nodeByName.has(edge.from) && nodeByName.has(edge.to) && edge.from !== edge.to,
  )

  const directedPairs = new Set(visibleEdges.map(e => `${e.from}->${e.to}`))
  const placedLabels: Array<{ x: number; y: number }> = []

  const positionedEdges: PositionedMentionEdge[] = visibleEdges.map((edge, idx) => {
    const fromNode = nodeByName.get(edge.from)!
    const toNode = nodeByName.get(edge.to)!
    const [sx, sy] = pillEdgeAnchor(fromNode, toNode, 1)
    const [ex, ey] = pillEdgeAnchor(toNode, fromNode, 4)

    const dx = ex - sx
    const dy = ey - sy
    const dist = Math.hypot(dx, dy) || 1
    const nx = -dy / dist
    const ny = dx / dist

    const isBidirectional = directedPairs.has(`${edge.to}->${edge.from}`)
    let offset = 0
    if (isBidirectional) {
      offset = Math.min(26, Math.max(14, dist * 0.14))
    } else if (edge.kind === 'broadcast') {
      offset = (idx % 2 === 0 ? 1 : -1) * Math.min(22, dist * 0.09)
    } else if (Math.abs(dx) < 28 && dist > 90) {
      // Gently bow long vertical edges so they route cleanly around middle nodes.
      offset = (idx % 2 === 0 ? 1 : -1) * Math.min(28, dist * 0.12)
    } else {
      offset = Math.min(14, dist * 0.06) * (fromNode.x <= toNode.x ? 1 : -1)
    }

    const cx = (sx + ex) / 2 + nx * offset
    const cy = (sy + ey) / 2 + ny * offset

    // Evaluate quadratic Bezier slightly off-center (t = 0.44) for cleaner count labels.
    const t = 0.44
    const omt = 1 - t
    let labelX = omt * omt * sx + 2 * omt * t * cx + t * t * ex
    let labelY = omt * omt * sy + 2 * omt * t * cy + t * t * ey

    for (const prev of placedLabels) {
      if (Math.hypot(prev.x - labelX, prev.y - labelY) < 15) {
        labelX += nx * 11
        labelY += ny * 11
      }
    }
    if (edge.count > 1) {
      placedLabels.push({ x: labelX, y: labelY })
    }

    const strokeWidth =
      edge.kind === 'broadcast'
        ? 1.15
        : Math.round(Math.min(3.8, 1.15 + Math.log2(edge.count + 1) * 0.85) * 100) / 100

    const r1 = (n: number): number => Math.round(n * 10) / 10

    return {
      edge,
      path: `M ${r1(sx)} ${r1(sy)} Q ${r1(cx)} ${r1(cy)} ${r1(ex)} ${r1(ey)}`,
      sx: r1(sx),
      sy: r1(sy),
      cx: r1(cx),
      cy: r1(cy),
      ex: r1(ex),
      ey: r1(ey),
      labelX: r1(labelX),
      labelY: r1(labelY),
      strokeWidth,
      dashed: edge.kind === 'broadcast',
    }
  })

  return {
    width,
    height: totalHeight,
    nodes,
    edges: positionedEdges,
    hiddenCount,
  }
}

/** Generate 8 clockwise vertices for a crisp chamfered-octagon (cut-corner) tactical zone. */
export function chamferedOctagonVertices(
  x: number,
  y: number,
  width: number,
  height: number,
  chamfer: number,
): ReadonlyArray<readonly [number, number]> {
  const c = Math.max(0, Math.min(chamfer, width / 2, height / 2))
  const r1 = (n: number): number => Math.round(n * 10) / 10
  return [
    [r1(x + c), r1(y)],
    [r1(x + width - c), r1(y)],
    [r1(x + width), r1(y + c)],
    [r1(x + width), r1(y + height - c)],
    [r1(x + width - c), r1(y + height)],
    [r1(x + c), r1(y + height)],
    [r1(x), r1(y + height - c)],
    [r1(x), r1(y + c)],
  ]
}

/** Exact containment check for a chamfered octagonal zone with optional outer clearance `pad`. */
export function isPointInsideChamferedRect(
  px: number,
  py: number,
  zone: Pick<BattlefieldZone, 'cx' | 'cy' | 'width' | 'height' | 'chamfer'>,
  pad = 0,
): boolean {
  const hx = zone.width / 2
  const hy = zone.height / 2
  const dx = Math.abs(px - zone.cx)
  const dy = Math.abs(py - zone.cy)
  return (
    dx <= hx + pad &&
    dy <= hy + pad &&
    dx + dy <= hx + hy - zone.chamfer + pad * Math.SQRT2
  )
}

function chamferedRectRayDistance(
  width: number,
  height: number,
  chamfer: number,
  ux: number,
  uy: number,
): number {
  const hx = width / 2
  const hy = height / 2
  const ax = Math.abs(ux)
  const ay = Math.abs(uy)
  const tx = ax > 1e-6 ? hx / ax : Number.POSITIVE_INFINITY
  const ty = ay > 1e-6 ? hy / ay : Number.POSITIVE_INFINITY
  const td = ax + ay > 1e-6 ? (hx + hy - chamfer) / (ax + ay) : Number.POSITIVE_INFINITY
  return Math.min(tx, ty, td)
}

function battlefieldZoneHeight(memberCount: number): number {
  const shown = Math.max(1, Math.min(6, memberCount))
  const overflow = memberCount > 6 ? 26 : 10
  return 34 + 6 + shown * 28 + overflow
}

function memberRowCaption(member: SwarmGraphMemberNode): string {
  if (member.error) return member.error
  if (member.currentActivity) return member.currentActivity
  if (member.state === 'working') return 'working'
  if (member.state === 'waiting') return 'waiting'
  if (member.state === 'done' || member.state === 'killed') return 'done'
  if (member.state === 'error') return 'failed'
  return 'idle'
}

function buildBattlefieldWorkerSlots(
  members: readonly SwarmGraphMemberNode[],
  x: number,
  y: number,
  width: number,
  _height: number,
): { readonly slots: readonly BattlefieldMemberSlot[]; readonly overflowCount: number } {
  const maxVisible = 6
  const shown = members.slice(0, maxVisible)
  const overflowCount = Math.max(0, members.length - shown.length)
  if (shown.length === 0) return { slots: [], overflowCount }

  const r1 = (n: number): number => Math.round(n * 10) / 10
  const rowH = 28
  const startY = y + 34 + 6 + rowH / 2
  const availTextW = Math.max(100, width - 38 - 28)
  const maxNameChars = Math.max(12, Math.min(22, Math.floor((availTextW * 0.62) / 6.8)))

  const slots: BattlefieldMemberSlot[] = shown.map((member, rowIdx) => {
    const displayLabel = truncateLabel(member.name, maxNameChars)
    const namePx = displayLabel.length * 7.2
    const remPx = availTextW - namePx - 14
    const rawCaption = memberRowCaption(member)
    const caption =
      remPx >= 42 ? truncateLabel(rawCaption, Math.max(6, Math.floor(remPx / 5.8))) : undefined

    return {
      member,
      x: r1(x + width / 2),
      y: r1(startY + rowIdx * rowH),
      spriteSize: 14,
      displayLabel,
      ...(caption ? { caption } : {}),
    }
  })

  return { slots, overflowCount }
}

function makeBattlefieldZone(input: {
  readonly id: BattlefieldZoneKind
  readonly label: string
  readonly sublabel: string
  readonly statusTag: string
  readonly count: number
  readonly cx: number
  readonly cy: number
  readonly width: number
  readonly height: number
  readonly chamfer: number
  readonly tone: BattlefieldZoneTone
  readonly members: readonly SwarmGraphMemberNode[]
  readonly progressRatio?: number
  readonly activeRatio?: number
}): BattlefieldZone {
  const r1 = (n: number): number => Math.round(n * 10) / 10
  const x = r1(input.cx - input.width / 2)
  const y = r1(input.cy - input.height / 2)
  const vertices = chamferedOctagonVertices(x, y, input.width, input.height, input.chamfer)
  const points = vertices.map(([vx, vy]) => `${vx},${vy}`).join(' ')
  const radius = 10

  if (input.id === 'coordinator') {
    const coord = input.members[0]
    const slots: BattlefieldMemberSlot[] = coord
      ? [
          {
            member: coord,
            x: r1(x + 23),
            y: r1(y + 24),
            spriteSize: 14,
            displayLabel: coord.name,
            caption: input.sublabel,
          },
        ]
      : []
    return {
      ...input,
      x,
      y,
      radius,
      points,
      innerPoints: points,
      vertices,
      innerVertices: vertices,
      cornerPips: [],
      slots,
      overflowCount: 0,
    }
  }

  const { slots, overflowCount } = buildBattlefieldWorkerSlots(
    input.members,
    x,
    y,
    input.width,
    input.height,
  )
  return {
    ...input,
    x,
    y,
    radius,
    points,
    innerPoints: points,
    vertices,
    innerVertices: vertices,
    cornerPips: [],
    slots,
    overflowCount,
  }
}

/**
 * Compute deterministic clean developer-tool topology layout for the Map view.
 *
 * - Single-bucket or all-completed swarms render a clean 2-tier vertical stack
 *   (`Completed` / active sector above `Coordinator`).
 * - Multi-state swarms arrange `Completed` (top), `Active` (left) & `Waiting`
 *   (right), `Coordinator` (bottom center), and `Errors` (below when present)
 *   as rounded `rx=10` cards with guaranteed unobstructed connector corridors.
 */
export function layoutBattlefieldGraph(
  frame: SwarmGraphFrame,
  options: { readonly width: number; readonly height: number },
): BattlefieldGraphLayout {
  const minW = Math.max(660, Math.round(options.width))
  const minH = Math.max(430, Math.round(options.height))
  const mapPadding = 12
  const gridStep = 32

  const { coordinator, working, waiting, done, error } = frame.buckets
  const activeBuckets: Array<{
    readonly id: Exclude<BattlefieldZoneKind, 'coordinator'>
    readonly label: string
    readonly statusTag: string
    readonly tone: BattlefieldZoneTone
    readonly members: readonly SwarmGraphMemberNode[]
  }> = []

  if (done.length > 0) {
    activeBuckets.push({
      id: 'done',
      label: 'Completed',
      statusTag: `${done.length} done`,
      tone: 'emerald',
      members: done,
    })
  }
  if (working.length > 0) {
    activeBuckets.push({
      id: 'working',
      label: 'Active',
      statusTag: `${working.length} active`,
      tone: 'accent',
      members: working,
    })
  }
  if (waiting.length > 0) {
    activeBuckets.push({
      id: 'waiting',
      label: 'Waiting',
      statusTag: `${waiting.length} waiting`,
      tone: 'slate',
      members: waiting,
    })
  }
  if (error.length > 0) {
    activeBuckets.push({
      id: 'error',
      label: 'Errors',
      statusTag: `${error.length} failed`,
      tone: 'rose',
      members: error,
    })
  }

  const allDone = frame.totalWorkers > 0 && frame.doneCount === frame.totalWorkers
  const coordTone: BattlefieldZoneTone = allDone
    ? 'emerald'
    : working.length > 0
      ? 'accent'
      : error.length > 0
        ? 'rose'
        : 'slate'

  const coordSublabel =
    frame.totalWorkers === 0
      ? 'Standing by'
      : `${frame.doneCount} of ${frame.totalWorkers} completed`

  const coordStatusTag = allDone
    ? 'Completed'
    : working.length > 0
      ? `${working.length} active`
      : error.length > 0
        ? 'Needs attention'
        : frame.totalWorkers > 0
          ? 'Waiting'
          : 'Idle'

  const progressRatio = frame.totalWorkers > 0 ? frame.doneCount / frame.totalWorkers : 0
  const activeRatio = frame.totalWorkers > 0 ? working.length / frame.totalWorkers : 0

  interface RawZonePlacement {
    readonly id: BattlefieldZoneKind
    readonly label: string
    readonly sublabel: string
    readonly statusTag: string
    readonly count: number
    readonly cx: number
    readonly cy: number
    readonly width: number
    readonly height: number
    readonly chamfer: number
    readonly tone: BattlefieldZoneTone
    readonly members: readonly SwarmGraphMemberNode[]
    readonly progressRatio?: number
    readonly activeRatio?: number
  }

  const rawZones: RawZonePlacement[] = []
  const usableW = Math.min(760, minW) - mapPadding * 2
  const cmdHeight = frame.totalWorkers > 0 ? 88 : 68

  if (activeBuckets.length === 0) {
    if (coordinator) {
      rawZones.push({
        id: 'coordinator',
        label: 'Coordinator',
        sublabel: coordSublabel,
        statusTag: coordStatusTag,
        count: 1,
        cx: 0,
        cy: 0,
        width: 312,
        height: cmdHeight,
        chamfer: 0,
        tone: coordTone,
        members: [coordinator],
        progressRatio,
        activeRatio,
      })
    }
  } else if (activeBuckets.length === 1) {
    const only = activeBuckets[0]!
    const bannerW = Math.min(344, Math.max(304, Math.floor(usableW * 0.52)))
    const bannerH = battlefieldZoneHeight(only.members.length)
    const tierGap = 44
    const totalStack = bannerH + tierGap + cmdHeight
    const topY = -totalStack / 2

    rawZones.push({
      id: only.id,
      label: only.label,
      sublabel: only.statusTag,
      statusTag: only.statusTag,
      count: only.members.length,
      cx: 0,
      cy: topY + bannerH / 2,
      width: bannerW,
      height: bannerH,
      chamfer: 0,
      tone: only.tone,
      members: only.members,
    })

    if (coordinator) {
      rawZones.push({
        id: 'coordinator',
        label: 'Coordinator',
        sublabel: coordSublabel,
        statusTag: coordStatusTag,
        count: 1,
        cx: 0,
        cy: topY + bannerH + tierGap + cmdHeight / 2,
        width: bannerW,
        height: cmdHeight,
        chamfer: 0,
        tone: coordTone,
        members: [coordinator],
        progressRatio,
        activeRatio,
      })
    }
  } else {
    const flankW = Math.min(282, Math.max(256, Math.floor(usableW * 0.39)))
    const victoryW = Math.min(304, Math.max(272, Math.floor(usableW * 0.43)))
    const cmdW = Math.min(316, Math.max(292, Math.floor(usableW * 0.45)))
    const errorW = Math.min(276, Math.max(248, Math.floor(usableW * 0.39)))

    const flankOuter = Math.min(usableW / 2 - 12, 360)
    const flankOffsetX = Math.max(flankW / 2 + 56, flankOuter - flankW / 2)
    const tierGap = 48

    const doneBucket = activeBuckets.find(b => b.id === 'done') ?? null
    const workingBucket = activeBuckets.find(b => b.id === 'working') ?? null
    const waitingBucket = activeBuckets.find(b => b.id === 'waiting') ?? null
    const errorBucket = activeBuckets.find(b => b.id === 'error') ?? null

    let cursorY = 0

    if (doneBucket) {
      const h = battlefieldZoneHeight(doneBucket.members.length)
      rawZones.push({
        id: 'done',
        label: doneBucket.label,
        sublabel: doneBucket.statusTag,
        statusTag: doneBucket.statusTag,
        count: doneBucket.members.length,
        cx: 0,
        cy: cursorY + h / 2,
        width: victoryW,
        height: h,
        chamfer: 0,
        tone: doneBucket.tone,
        members: doneBucket.members,
      })
      cursorY += h + tierGap
    }

    if (workingBucket || waitingBucket) {
      const workH = workingBucket ? battlefieldZoneHeight(workingBucket.members.length) : 0
      const waitH = waitingBucket ? battlefieldZoneHeight(waitingBucket.members.length) : 0
      const flankH = Math.max(workH, waitH)
      const flankCy = cursorY + flankH / 2

      if (workingBucket) {
        rawZones.push({
          id: 'working',
          label: workingBucket.label,
          sublabel: workingBucket.statusTag,
          statusTag: workingBucket.statusTag,
          count: workingBucket.members.length,
          cx: -flankOffsetX,
          cy: flankCy,
          width: flankW,
          height: flankH,
          chamfer: 0,
          tone: workingBucket.tone,
          members: workingBucket.members,
        })
      }
      if (waitingBucket) {
        rawZones.push({
          id: 'waiting',
          label: waitingBucket.label,
          sublabel: waitingBucket.statusTag,
          statusTag: waitingBucket.statusTag,
          count: waitingBucket.members.length,
          cx: flankOffsetX,
          cy: flankCy,
          width: flankW,
          height: flankH,
          chamfer: 0,
          tone: waitingBucket.tone,
          members: waitingBucket.members,
        })
      }
      cursorY += flankH + tierGap
    }

    if (coordinator) {
      rawZones.push({
        id: 'coordinator',
        label: 'Coordinator',
        sublabel: coordSublabel,
        statusTag: coordStatusTag,
        count: 1,
        cx: 0,
        cy: cursorY + cmdHeight / 2,
        width: cmdW,
        height: cmdHeight,
        chamfer: 0,
        tone: coordTone,
        members: [coordinator],
        progressRatio,
        activeRatio,
      })
      cursorY += cmdHeight + (errorBucket ? tierGap : 0)
    }

    if (errorBucket) {
      const errH = battlefieldZoneHeight(errorBucket.members.length)
      rawZones.push({
        id: 'error',
        label: errorBucket.label,
        sublabel: errorBucket.statusTag,
        statusTag: errorBucket.statusTag,
        count: errorBucket.members.length,
        cx: 0,
        cy: cursorY + errH / 2,
        width: errorW,
        height: errH,
        chamfer: 0,
        tone: errorBucket.tone,
        members: errorBucket.members,
      })
    }
  }

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const z of rawZones) {
    minX = Math.min(minX, z.cx - z.width / 2)
    maxX = Math.max(maxX, z.cx + z.width / 2)
    minY = Math.min(minY, z.cy - z.height / 2)
    maxY = Math.max(maxY, z.cy + z.height / 2)
  }

  const contentW = Number.isFinite(minX) ? maxX - minX : 0
  const contentH = Number.isFinite(minY) ? maxY - minY : 0
  const targetMinH = activeBuckets.length === 0 ? 152 : minH
  const finalW = Math.max(minW, Math.ceil(contentW + 76))
  const finalH = Math.max(targetMinH, Math.ceil(contentH + 84))
  const offsetX = Number.isFinite(minX) ? (finalW - contentW) / 2 - minX : finalW / 2
  const offsetY = Number.isFinite(minY) ? (finalH - contentH) / 2 - minY : finalH / 2

  const r1 = (n: number): number => Math.round(n * 10) / 10

  const zones: BattlefieldZone[] = rawZones.map(z =>
    makeBattlefieldZone({
      id: z.id,
      label: z.label,
      sublabel: z.sublabel,
      statusTag: z.statusTag,
      count: z.count,
      cx: r1(z.cx + offsetX),
      cy: r1(z.cy + offsetY),
      width: z.width,
      height: z.height,
      chamfer: z.chamfer,
      tone: z.tone,
      members: z.members,
      ...(z.progressRatio !== undefined ? { progressRatio: z.progressRatio } : {}),
      ...(z.activeRatio !== undefined ? { activeRatio: z.activeRatio } : {}),
    }),
  )

  const zoneById = new Map<BattlefieldZoneKind, BattlefieldZone>(zones.map(z => [z.id, z]))
  const memberZone = new Map<string, BattlefieldZoneKind>()
  for (const zone of zones) {
    for (const member of zone.members) {
      memberZone.set(member.name, zone.id)
    }
  }

  const cmdZone = zoneById.get('coordinator')
  const commandCenter: readonly [number, number] = cmdZone
    ? [cmdZone.cx, cmdZone.cy]
    : [r1(finalW / 2), r1(finalH * 0.72)]

  const rings: BattlefieldRangeRing[] = []

  const dispatches: BattlefieldDispatchVector[] = []
  const seenDispatches = new Set<string>()
  const pairUsed = new Set<string>()

  const addDispatch = (
    from: BattlefieldZoneKind,
    to: BattlefieldZoneKind,
    kind: BattlefieldDispatchKind,
  ): void => {
    if (from === to) return
    const key = `${kind}:${from}->${to}`
    if (seenDispatches.has(key)) return
    const zFrom = zoneById.get(from)
    const zTo = zoneById.get(to)
    if (!zFrom || !zTo) return
    seenDispatches.add(key)

    const dx = zTo.cx - zFrom.cx
    const dy = zTo.cy - zFrom.cy
    const dist = Math.hypot(dx, dy) || 1
    const ux = dx / dist
    const uy = dy / dist
    const nx = -uy
    const ny = ux
    const theta = Math.atan2(dy, dx)
    const otherZones = zones.filter(z => z.id !== from && z.id !== to)

    const pairKey = [from, to].sort().join('<->')
    const hasReverseOrTwin = pairUsed.has(pairKey)
    pairUsed.add(pairKey)

    interface CandidateVector {
      readonly sx: number
      readonly sy: number
      readonly cx: number
      readonly cy: number
      readonly ex: number
      readonly ey: number
    }

    const straightCandidate = (lateral = 0): CandidateVector => {
      const startDist = chamferedRectRayDistance(zFrom.width, zFrom.height, zFrom.chamfer, ux, uy) + 3
      const endDist = chamferedRectRayDistance(zTo.width, zTo.height, zTo.chamfer, -ux, -uy) + 6
      const sx = zFrom.cx + ux * startDist + nx * lateral
      const sy = zFrom.cy + uy * startDist + ny * lateral
      const ex = zTo.cx - ux * endDist + nx * lateral
      const ey = zTo.cy - uy * endDist + ny * lateral
      return { sx, sy, cx: (sx + ex) / 2, cy: (sy + ey) / 2, ex, ey }
    }

    const bowedCandidate = (side: 1 | -1, shoulder: number, bowPx: number): CandidateVector => {
      const aStart = theta + side * shoulder
      const aEnd = theta + Math.PI - side * shoulder
      const sux = Math.cos(aStart)
      const suy = Math.sin(aStart)
      const eux = Math.cos(aEnd)
      const euy = Math.sin(aEnd)
      const sDist = chamferedRectRayDistance(zFrom.width, zFrom.height, zFrom.chamfer, sux, suy) + 3
      const eDist = chamferedRectRayDistance(zTo.width, zTo.height, zTo.chamfer, eux, euy) + 6
      const sx = zFrom.cx + sux * sDist
      const sy = zFrom.cy + suy * sDist
      const ex = zTo.cx + eux * eDist
      const ey = zTo.cy + euy * eDist
      return {
        sx,
        sy,
        cx: (sx + ex) / 2 + nx * side * bowPx,
        cy: (sy + ey) / 2 + ny * side * bowPx,
        ex,
        ey,
      }
    }

    const flankBypassCandidate = (side: 1 | -1, clearanceX: number): CandidateVector => {
      const sux = side
      const suy = 0
      const eux = side
      const euy = 0
      const sDist = chamferedRectRayDistance(zFrom.width, zFrom.height, zFrom.chamfer, sux, suy) + 3
      const eDist = chamferedRectRayDistance(zTo.width, zTo.height, zTo.chamfer, eux, euy) + 6
      const sx = zFrom.cx + sux * sDist
      const sy = zFrom.cy + suy * sDist
      const ex = zTo.cx + eux * eDist
      const ey = zTo.cy + euy * eDist
      const targetX = side > 0 ? finalW - clearanceX : clearanceX
      return {
        sx,
        sy,
        cx: targetX,
        cy: (sy + ey) / 2,
        ex,
        ey,
      }
    }

    const isUnobstructed = (vec: CandidateVector, pad: number): boolean => {
      if (otherZones.length === 0) return true
      for (let i = 1; i <= 23; i += 1) {
        const t = i / 24
        const omt = 1 - t
        const px = omt * omt * vec.sx + 2 * omt * t * vec.cx + t * t * vec.ex
        const py = omt * omt * vec.sy + 2 * omt * t * vec.cy + t * t * vec.ey
        if (px < mapPadding + 2 || px > finalW - mapPadding - 2) return false
        if (py < mapPadding + 2 || py > finalH - mapPadding - 2) return false
        for (const other of otherZones) {
          if (isPointInsideChamferedRect(px, py, other, pad)) {
            return false
          }
        }
      }
      return true
    }

    const preferredSide: 1 | -1 = kind === 'waiting_on' ? 1 : -1
    const otherSide: 1 | -1 = preferredSide === 1 ? -1 : 1

    const candidates: CandidateVector[] = hasReverseOrTwin
      ? [
          bowedCandidate(preferredSide, 0.24, 24),
          bowedCandidate(otherSide, 0.24, 24),
          straightCandidate(preferredSide * 10),
          bowedCandidate(preferredSide, 0.36, 56),
          bowedCandidate(otherSide, 0.36, 56),
          flankBypassCandidate(preferredSide, 22),
          flankBypassCandidate(otherSide, 22),
        ]
      : [
          straightCandidate(0),
          bowedCandidate(preferredSide, 0.22, 22),
          bowedCandidate(otherSide, 0.22, 22),
          bowedCandidate(preferredSide, 0.34, 54),
          bowedCandidate(otherSide, 0.34, 54),
          bowedCandidate(preferredSide, 0.44, 118),
          bowedCandidate(otherSide, 0.44, 118),
          flankBypassCandidate(preferredSide, 20),
          flankBypassCandidate(otherSide, 20),
        ]

    const chosen =
      candidates.find(vec => isUnobstructed(vec, 6)) ??
      candidates.find(vec => isUnobstructed(vec, 2)) ??
      straightCandidate(0)

    const { sx, sy, cx, cy, ex, ey } = chosen

    dispatches.push({
      id: key,
      kind,
      from,
      to,
      path: `M ${r1(sx)} ${r1(sy)} Q ${r1(cx)} ${r1(cy)} ${r1(ex)} ${r1(ey)}`,
      sx: r1(sx),
      sy: r1(sy),
      cx: r1(cx),
      cy: r1(cy),
      ex: r1(ex),
      ey: r1(ey),
      dashed: kind === 'waiting_on',
    })
  }

  // 1. Live or scrubbed active message dispatch vectors
  if (frame.activeSender) {
    const fromZone = memberZone.get(frame.activeSender)
    if (fromZone) {
      const targetZones = new Set<BattlefieldZoneKind>()
      for (const recipient of frame.activeRecipients) {
        const toZone = memberZone.get(recipient)
        if (toZone && toZone !== fromZone) targetZones.add(toZone)
      }
      if (frame.activeIsBroadcast) {
        for (const zone of zones) {
          if (zone.id !== fromZone) targetZones.add(zone.id)
        }
      }
      const dispatchKind: BattlefieldDispatchKind = frame.activeIsDone
        ? 'tasks_done'
        : 'message_sent'
      for (const toZone of targetZones) {
        addDispatch(fromZone, toZone, dispatchKind)
      }
    }
  }

  // 2. Ensure live active command dispatch from Command Pavilion -> Vanguard is visible
  if (
    zoneById.has('coordinator') &&
    zoneById.has('working') &&
    !dispatches.some(d => d.kind === 'message_sent') &&
    (frame.isLive || frame.edges.some(e => memberZone.get(e.from) === 'coordinator' && memberZone.get(e.to) === 'working'))
  ) {
    addDispatch('coordinator', 'working', 'message_sent')
  }

  // 3. Ensure completed Victory -> Command Pavilion signal is visible when workers completed
  if (
    zoneById.has('done') &&
    zoneById.has('coordinator') &&
    frame.doneCount > 0 &&
    frame.isLive &&
    !dispatches.some(d => d.kind === 'tasks_done')
  ) {
    addDispatch('done', 'coordinator', 'tasks_done')
  }

  // 4. Show "Awaiting handoff" vector whenever Reserve awaits active Vanguard or Command
  if (zoneById.has('waiting')) {
    if (zoneById.has('working')) {
      addDispatch('waiting', 'working', 'waiting_on')
    } else if (zoneById.has('coordinator')) {
      addDispatch('waiting', 'coordinator', 'waiting_on')
    }
  }

  return {
    width: finalW,
    height: finalH,
    mapPadding,
    gridStep,
    commandCenter,
    rings,
    zones,
    dispatches,
  }
}

/** Backwards-compatible alias for `layoutBattlefieldGraph`. */
export function layoutHiveGraph(
  frame: SwarmGraphFrame,
  options: { readonly width: number; readonly height: number },
): BattlefieldGraphLayout {
  return layoutBattlefieldGraph(frame, options)
}

/* ── Canvas Renderer & Animated GIF Exporter ────────────────────────────── */

const GIF_BG = '#0e1015'
const GIF_CARD = '#151821'
const GIF_CHIP = '#1d222e'
const GIF_GRID = '#1d222c'
const GIF_BORDER = '#343b4c'
const GIF_TEXT = '#e8ecf3'
const GIF_SECONDARY = '#a5aec1'
const GIF_MUTED = '#8892a6'
const GIF_EDGE = '#717a8c'
const GIF_EMERALD = '#57ca85'
const GIF_ACCENT = '#6ea8fe'
const GIF_GOLD = '#d8ae58'
const GIF_ROSE = '#f47067'
const GIF_SLATE = '#8892a6'

function toneStrokeHex(tone: BattlefieldZoneTone): string {
  switch (tone) {
    case 'emerald':
      return GIF_EMERALD
    case 'accent':
      return GIF_ACCENT
    case 'rose':
      return GIF_ROSE
    case 'gold':
      return GIF_GOLD
    default:
      return GIF_SLATE
  }
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + radius, y)
  ctx.lineTo(x + w - radius, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius)
  ctx.lineTo(x + w, y + h - radius)
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h)
  ctx.lineTo(x + radius, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius)
  ctx.lineTo(x, y + radius)
  ctx.quadraticCurveTo(x, y, x + radius, y)
  ctx.closePath()
}

function drawStatusDot(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  fill: string,
): void {
  ctx.fillStyle = fill
  ctx.beginPath()
  if (typeof ctx.arc === 'function') {
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
  } else {
    ctx.moveTo(cx, cy - r)
    ctx.lineTo(cx + r, cy)
    ctx.lineTo(cx, cy + r)
    ctx.lineTo(cx - r, cy)
    ctx.closePath()
  }
  ctx.fill()
}

function drawArrowHead(
  ctx: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string,
  size = 5.5,
): void {
  const angle = Math.atan2(toY - fromY, toX - fromX)
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(toX, toY)
  ctx.lineTo(toX - size * Math.cos(angle - Math.PI / 6), toY - size * Math.sin(angle - Math.PI / 6))
  ctx.lineTo(toX - size * Math.cos(angle + Math.PI / 6), toY - size * Math.sin(angle + Math.PI / 6))
  ctx.closePath()
  ctx.fill()
}

function renderFrameToCanvas(
  ctx: CanvasRenderingContext2D,
  frame: SwarmGraphFrame,
  options: {
    readonly width: number
    readonly height: number
    readonly mode: SwarmGraphViewMode
    readonly hideInactive: boolean
    readonly expandAll: boolean
    readonly title: string
    readonly frameIndex: number
    readonly durationLabel: string
  },
): void {
  const { width, height, mode, title, frameIndex } = options
  ctx.fillStyle = GIF_BG
  ctx.fillRect(0, 0, width, height)

  // Clean top title bar
  ctx.fillStyle = GIF_TEXT
  ctx.font = '600 12.5px -apple-system, BlinkMacSystemFont, sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  const cleanTitle = title.trim()
  const modeLabel = mode === 'battlefield' ? 'Map' : 'Mentions'
  const canvasHeader =
    mode === 'battlefield'
      ? 'XERXES IMMORTALS · MAP'
      : cleanTitle && !/^xerxes\s+immortals$/i.test(cleanTitle)
        ? `XERXES IMMORTALS · ${cleanTitle}`
        : `XERXES IMMORTALS · ${modeLabel.toUpperCase()}`
  ctx.fillText(truncateLabel(canvasHeader, 46), 18, 22)

  ctx.fillStyle = GIF_MUTED
  ctx.font = '500 10.5px Menlo, monospace'
  ctx.textAlign = 'right'
  ctx.fillText(
    `#${frame.step}/${frame.totalSteps} · ${frame.elapsedLabel}`,
    width - 18,
    22,
  )

  // Hairline divider
  ctx.strokeStyle = GIF_GRID
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(0, 38)
  ctx.lineTo(width, 38)
  ctx.stroke()

  const stageY = 42
  const stageH = height - 76

  ctx.save()
  ctx.translate(0, stageY)

  if (mode === 'battlefield') {
    const layout = layoutBattlefieldGraph(frame, { width: Math.max(660, width), height: Math.max(430, stageH) })
    const scale = Math.min(1, width / layout.width, stageH / layout.height)
    const ox = (width - layout.width * scale) / 2
    const oy = (stageH - layout.height * scale) / 2
    ctx.translate(ox, oy)
    ctx.scale(scale, scale)

    // 1. Clean rounded cards (`rx = 10`, hairline border, zero chamfer/HUD chrome)
    for (const zone of layout.zones) {
      const toneColor = toneStrokeHex(zone.tone)
      const isCoord = zone.id === 'coordinator'

      drawRoundedRect(ctx, zone.x, zone.y, zone.width, zone.height, zone.radius)
      ctx.fillStyle = GIF_CARD
      ctx.fill()
      ctx.lineWidth = 1
      ctx.strokeStyle = GIF_BORDER
      ctx.stroke()

      if (isCoord) {
        const slot = zone.slots[0]
        if (slot) {
          drawSimurghSpriteOnCanvas(
            ctx,
            slot.member.spriteRole,
            slot.member.spriteState,
            zone.x + 16,
            zone.y + 17,
            14,
            frameIndex,
          )
        }

        const rightEdge = zone.x + zone.width - 16
        ctx.fillStyle = GIF_TEXT
        ctx.font = '600 12.5px -apple-system, sans-serif'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(zone.label, zone.x + 38, zone.y + 24)

        drawStatusDot(ctx, rightEdge - 3, zone.y + 24, 3, toneColor)
        ctx.fillStyle = GIF_MUTED
        ctx.font = '500 10.5px -apple-system, sans-serif'
        ctx.textAlign = 'right'
        ctx.fillText(zone.statusTag, rightEdge - 12, zone.y + 24)

        ctx.fillStyle = GIF_SECONDARY
        ctx.font = '400 11px -apple-system, sans-serif'
        ctx.textAlign = 'left'
        ctx.fillText(zone.sublabel, zone.x + 38, zone.y + 46)

        if (frame.totalWorkers > 0) {
          const pct = Math.round((zone.progressRatio ?? 0) * 100)
          ctx.fillStyle = GIF_MUTED
          ctx.font = '500 10px Menlo, monospace'
          ctx.textAlign = 'right'
          ctx.fillText(`${pct}%`, rightEdge, zone.y + 46)

          const barX = zone.x + 16
          const barY = zone.y + 68
          const barW = zone.width - 32
          const barH = 3.5
          drawRoundedRect(ctx, barX, barY, barW, barH, 1.75)
          ctx.fillStyle = GIF_CHIP
          ctx.fill()

          const doneW = Math.round(barW * Math.max(0, Math.min(1, zone.progressRatio ?? 0)))
          if (doneW > 0) {
            drawRoundedRect(ctx, barX, barY, Math.max(4, doneW), barH, 1.75)
            ctx.fillStyle = GIF_EMERALD
            ctx.fill()
          }
          const activeW = Math.round(
            barW * Math.max(0, Math.min(1 - (zone.progressRatio ?? 0), zone.activeRatio ?? 0)),
          )
          if (activeW > 0) {
            ctx.fillStyle = GIF_ACCENT
            ctx.fillRect(barX + doneW, barY, Math.max(2, activeW), barH)
          }
        }
      } else {
        drawStatusDot(ctx, zone.x + 16, zone.y + 17, 3, toneColor)
        ctx.fillStyle = GIF_MUTED
        ctx.font = '500 11.5px -apple-system, sans-serif'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(zone.label, zone.x + 26, zone.y + 17)

        ctx.fillStyle = GIF_MUTED
        ctx.font = '400 10.5px -apple-system, sans-serif'
        ctx.textAlign = 'right'
        ctx.fillText(zone.statusTag, zone.x + zone.width - 14, zone.y + 17)

        // Edge-to-edge hairline divider
        ctx.strokeStyle = GIF_GRID
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(zone.x, zone.y + 34)
        ctx.lineTo(zone.x + zone.width, zone.y + 34)
        ctx.stroke()

        // Clean horizontal member list rows
        for (const slot of zone.slots) {
          drawSimurghSpriteOnCanvas(
            ctx,
            slot.member.spriteRole,
            slot.member.spriteState,
            zone.x + 14,
            slot.y - 7,
            14,
            frameIndex,
          )
          ctx.fillStyle =
            slot.member.state === 'done' || slot.member.state === 'killed' ? GIF_SECONDARY : GIF_TEXT
          ctx.font = '500 12px -apple-system, sans-serif'
          ctx.textAlign = 'left'
          ctx.textBaseline = 'middle'
          ctx.fillText(slot.displayLabel, zone.x + 36, slot.y)

          const dotColor =
            slot.member.state === 'error'
              ? GIF_ROSE
              : slot.member.state === 'done' || slot.member.state === 'killed'
                ? GIF_EMERALD
                : slot.member.state === 'working'
                  ? GIF_ACCENT
                  : GIF_MUTED
          drawStatusDot(ctx, zone.x + zone.width - 16, slot.y, 3, dotColor)
          if (slot.caption) {
            ctx.fillStyle = GIF_MUTED
            ctx.font = '400 10px -apple-system, sans-serif'
            ctx.textAlign = 'right'
            ctx.fillText(slot.caption, zone.x + zone.width - 26, slot.y)
          }
        }

        if (zone.overflowCount > 0) {
          const pillText = `+${zone.overflowCount} more`
          ctx.fillStyle = GIF_MUTED
          ctx.font = '500 10px -apple-system, sans-serif'
          ctx.textAlign = 'left'
          ctx.fillText(pillText, zone.x + 14, zone.y + zone.height - 14)
        }
      }
    }

    // 2. Subtle smooth connector edges between cards
    for (let dIdx = 0; dIdx < layout.dispatches.length; dIdx += 1) {
      const dispatch = layout.dispatches[dIdx]!
      const color =
        dispatch.kind === 'tasks_done'
          ? GIF_EMERALD
          : dispatch.kind === 'waiting_on'
            ? GIF_MUTED
            : GIF_ACCENT
      ctx.save()
      ctx.strokeStyle = color
      ctx.lineWidth = 1.4
      ctx.globalAlpha = 0.72
      if (dispatch.dashed) ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(dispatch.sx, dispatch.sy)
      ctx.quadraticCurveTo(dispatch.cx, dispatch.cy, dispatch.ex, dispatch.ey)
      ctx.stroke()
      ctx.restore()

      drawArrowHead(ctx, dispatch.cx, dispatch.cy, dispatch.ex, dispatch.ey, color, 5.5)
    }
  } else {
    const layout = layoutMentionsGraph(frame, {
      hideInactive: options.hideInactive,
      expandAll: options.expandAll,
      width,
      height: stageH,
    })
    const scale = Math.min(1, width / layout.width, stageH / layout.height)
    const ox = (width - layout.width * scale) / 2
    const oy = (stageH - layout.height * scale) / 2
    ctx.translate(ox, oy)
    ctx.scale(scale, scale)

    // Draw edges
    for (const pe of layout.edges) {
      const color = pe.edge.isActive
        ? GIF_ACCENT
        : pe.dashed
          ? '#64748b'
          : GIF_EDGE
      ctx.save()
      ctx.strokeStyle = color
      ctx.lineWidth = pe.edge.isActive ? Math.max(2, pe.strokeWidth) : pe.strokeWidth
      ctx.globalAlpha = pe.edge.isActive ? 0.96 : pe.dashed ? 0.55 : 0.68
      if (pe.dashed) ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(pe.sx, pe.sy)
      ctx.quadraticCurveTo(pe.cx, pe.cy, pe.ex, pe.ey)
      ctx.stroke()
      ctx.restore()

      drawArrowHead(ctx, pe.cx, pe.cy, pe.ex, pe.ey, color, pe.edge.isActive ? 7 : 5)

      if (pe.edge.count > 1 && !pe.dashed) {
        ctx.fillStyle = GIF_BG
        drawRoundedRect(ctx, pe.labelX - 9, pe.labelY - 6, 18, 12, 4)
        ctx.fill()
        ctx.fillStyle = GIF_TEXT
        ctx.font = '600 9px Menlo, monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(pe.edge.count), pe.labelX, pe.labelY)
      }
    }

    // Draw nodes
    for (const node of layout.nodes) {
      const x = node.x - node.width / 2
      const y = node.y - node.height / 2
      ctx.save()
      drawRoundedRect(ctx, x, y, node.width, node.height, node.rx)
      ctx.fillStyle = node.member.isActiveSender ? GIF_CHIP : GIF_CARD
      ctx.fill()
      ctx.lineWidth = node.member.isCoordinator || node.member.isActiveSender ? 1.6 : 1.1
      ctx.strokeStyle = node.member.isActiveSender
        ? GIF_ACCENT
        : node.member.isCoordinator
          ? GIF_GOLD
          : GIF_BORDER
      if (node.member.postedWithoutMention) ctx.setLineDash([3, 3])
      ctx.stroke()
      ctx.restore()

      const spriteSize = 16
      drawSimurghSpriteOnCanvas(
        ctx,
        node.member.spriteRole,
        node.member.spriteState,
        x + 8,
        node.y - spriteSize / 2,
        spriteSize,
        frameIndex,
      )

      ctx.fillStyle = node.member.state === 'done' || node.member.state === 'killed' ? GIF_MUTED : GIF_TEXT
      ctx.font = node.member.isCoordinator ? '600 11px Menlo, monospace' : '500 10.5px Menlo, monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillText(node.displayLabel, x + 28, node.y)

      if (node.member.allCount > 0) {
        const badgeText = `@all ${node.member.allCount}`
        const badgeW = Math.max(36, badgeText.length * 6 + 10)
        const bx = x + node.width - badgeW + 6
        const by = y - 10
        drawRoundedRect(ctx, bx, by, badgeW, 14, 7)
        ctx.fillStyle = '#232832'
        ctx.fill()
        ctx.strokeStyle = GIF_BORDER
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.fillStyle = GIF_MUTED
        ctx.font = '500 9px Menlo, monospace'
        ctx.textAlign = 'center'
        ctx.fillText(badgeText, bx + badgeW / 2, by + 7)
      }
    }
  }

  ctx.restore()

  // Bottom footer bar
  ctx.fillStyle = GIF_MUTED
  ctx.font = '400 10px -apple-system, sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(
    mode === 'battlefield'
      ? '— Completed   — Dispatch sent   --> Awaiting handoff'
      : '→ @mention, width = count   ⋯→ via @all   ◌ posted with no mention',
    18,
    height - 16,
  )
}

/**
 * Sample replay frames from `timeline`, render them onto a 2D `<canvas>` using
 * crisp Xerxes Simurgh sprites, encode a looping GIF89a binary, and trigger download.
 */
export function exportSwarmGraphGif(options: ExportSwarmGraphGifOptions): ExportSwarmGraphGifResult {
  const mode = normalizeSwarmGraphMode(options.mode)
  const width = Math.max(240, Math.round(options.width ?? 720))
  const height = Math.max(180, Math.round(options.height ?? 520))
  const maxFrames = Math.max(2, Math.min(36, Math.round(options.maxFrames ?? 32)))
  const rawSlug = (options.title || '')
    .toLowerCase()
    .replace(/^xerxes[-_\s]+immortals[-_\s]*/i, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32)
  const slug = rawSlug || 'session'
  const filename = options.filename || `xerxes-immortals-${slug}-${mode}.gif`

  const canvasFactory =
    options.createCanvas ??
    ((w: number, h: number): CanvasLike | null => {
      if (typeof document === 'undefined') return null
      const el = document.createElement('canvas')
      el.width = w
      el.height = h
      return el
    })

  const canvas = canvasFactory(width, height)
  const ctx = canvas?.getContext('2d') ?? null
  if (!canvas || !ctx) {
    return { ok: false, frames: 0, bytes: 0, filename, error: '2D canvas unavailable for GIF export.' }
  }

  const total = options.timeline.totalPosts
  const stepIndices: number[] = []
  if (total === 0) {
    stepIndices.push(0)
  } else if (total + 1 <= maxFrames) {
    for (let s = 0; s <= total; s += 1) stepIndices.push(s)
  } else {
    for (let i = 0; i < maxFrames; i += 1) {
      const s = Math.round((i * total) / (maxFrames - 1))
      if (stepIndices.at(-1) !== s) stepIndices.push(s)
    }
    if (stepIndices.at(-1) !== total) stepIndices.push(total)
  }

  const gifFrames: GifFrame[] = []
  for (let i = 0; i < stepIndices.length; i += 1) {
    const step = stepIndices[i]!
    const frame = options.timeline.frameAt(step)
    renderFrameToCanvas(ctx, frame, {
      width,
      height,
      mode,
      hideInactive: options.hideInactive,
      expandAll: options.expandAll,
      title: options.title ?? 'XERXES IMMORTALS',
      frameIndex: i,
      durationLabel: options.timeline.durationLabel,
    })
    const img = ctx.getImageData(0, 0, width, height)
    const isLast = i === stepIndices.length - 1
    const baseDelay = options.timeline.stepDelayMs(step, options.pacing, options.speed)
    gifFrames.push({
      width,
      height,
      rgba: img.data,
      delayMs: isLast ? Math.max(1200, baseDelay * 2) : Math.max(120, Math.min(900, baseDelay)),
    })
  }

  const bytes = encodeAnimatedGif(gifFrames, { loop: 0, defaultDelayMs: 360 })
  const downloader = options.download ?? downloadGifBytes
  downloader(bytes, filename)

  return {
    ok: true,
    frames: gifFrames.length,
    bytes: bytes.byteLength,
    filename,
  }
}
