// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { focusComposer } from '../app/composerFocus.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import {
  $swarmStore,
  applySwarmBoard,
  applySwarmEvent,
  applySwarmStatus,
  getSwarmSliceForSession,
  refreshSwarmSession
} from '../app/swarmStore.js'
import { $uiSessionId } from '../app/uiStore.js'
import type {
  GatewayEvent,
  SwarmBoardMessage,
  SwarmChannelSnapshot,
  SwarmKillResponse,
  SwarmMemberInspectResponse,
  SwarmMemberSnapshot,
  SwarmMemberState,
  SwarmModelResponse,
  SwarmPostResponse
} from '../gatewayTypes.js'
import { isPageDownKey, isPageUpKey } from '../lib/pageKeys.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import {
  collectSwarmChannelTabs,
  formatSwarmTime,
  normalizeSwarmInspectResponse,
  summarizeSwarmCounts,
  tokenizeSwarmMessageText
} from '../lib/swarmFormat.js'
import type { Theme } from '../theme.js'

import { DialogEmpty, DialogFooter, DialogHeader } from './dialogChrome.js'
import { OVERLAY_PANEL_SPECS, overlayPanelSize, windowItems } from './overlayLayout.js'
import { Box, Span, Text } from './primitives.js'

export type SwarmTab = 'board' | 'members' | 'overview'
type ComposeMode = 'post' | 'feedback' | 'model' | null

const STATE_GLYPH: Record<SwarmMemberState, string> = {
  working: '●',
  waiting: '○',
  stopped: '◌',
  killed: '×'
}

function memberStateColor(state: SwarmMemberState, t: Theme): string {
  if (state === 'working') return t.color.accent
  if (state === 'waiting') return t.color.ok
  if (state === 'stopped') return t.color.warn
  return t.color.muted
}

function clampText(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, Math.max(1, max - 1))}…` : value
}

export function extractComposeMentionQuery(draft: string): { prefix: string; query: string } | null {
  const match = draft.match(/^(.*?)(?:^|\s)@([a-z0-9_#-]*)$/i)
  if (!match) return null
  const query = match[2] ?? ''
  const atIndex = draft.lastIndexOf(`@${query}`)
  if (atIndex < 0) return null
  return { prefix: draft.slice(0, atIndex), query }
}

export function computeSwarmMentionCandidates(
  query: string,
  members: readonly SwarmMemberSnapshot[],
  channels: readonly SwarmChannelSnapshot[]
): string[] {
  const living = members
    .filter(member => !member.killed && member.state !== 'killed')
    .map(member => `@${member.name}`)
  const customChannels = channels
    .filter(ch => ch.name !== 'main' && !ch.archived)
    .map(ch => `@#${ch.name}`)
  const allTargets = ['@all', ...living, ...customChannels]
  const unique = [...new Set(allTargets)]
  const needle = query.trim().toLowerCase()
  if (!needle) return unique.slice(0, 6)
  return unique
    .filter(candidate => candidate.slice(1).toLowerCase().startsWith(needle))
    .slice(0, 6)
}

function BoardMessageRow({
  msg,
  showChannelBadge,
  t
}: {
  msg: SwarmBoardMessage
  showChannelBadge: boolean
  t: Theme
}) {
  const time = formatSwarmTime(msg.time)
  const tokens = useMemo(() => tokenizeSwarmMessageText(msg.text), [msg.text])

  return (
    <Box flexDirection="column" flexShrink={0} marginBottom={1}>
      <Text wrap="truncate-end">
        <Span bold color={t.color.brandGold}>{`#${msg.seq}`}</Span>
        <Span bold color={msg.agent === 'coordinator' ? t.color.accent : t.color.text}>{`  @${msg.agent}`}</Span>
        {time ? <Span color={t.ds.meta}>{`  ${time}`}</Span> : null}
        {showChannelBadge ? <Span color={t.ds.secondary}>{`  #${msg.channel}`}</Span> : null}
        {msg.broadcast ? <Span bold color={t.color.warn}>{'  @all'}</Span> : null}
      </Text>
      <Box paddingLeft={2}>
        <Text color={t.color.text} wrap="wrap">
          {tokens.map((tok, index) => {
            if (tok.kind === 'done') {
              return (
                <Span bold color={t.color.ok} key={index}>
                  {tok.value}
                </Span>
              )
            }
            if (tok.kind === 'stall') {
              return (
                <Span bold color={t.color.warn} key={index}>
                  {tok.value}
                </Span>
              )
            }
            if (tok.kind === 'broadcast') {
              return (
                <Span bold color={t.color.warn} key={index}>
                  {tok.value}
                </Span>
              )
            }
            if (tok.kind === 'mention' || tok.kind === 'channel_mention') {
              return (
                <Span bold color={t.color.accent} key={index}>
                  {tok.value}
                </Span>
              )
            }
            if (tok.kind === 'seq_ref') {
              return (
                <Span color={t.color.brandGold} key={index}>
                  {tok.value}
                </Span>
              )
            }
            return <Span key={index}>{tok.value}</Span>
          })}
        </Text>
      </Box>
    </Box>
  )
}

function RosterMemberRow({
  compact,
  inSplitPane = false,
  member,
  onSelect,
  selected,
  t
}: {
  compact: boolean
  inSplitPane?: boolean
  member: SwarmMemberSnapshot
  onSelect: () => void
  selected: boolean
  t: Theme
}) {
  const glyph = STATE_GLYPH[member.state]
  const stateColor = memberStateColor(member.state, t)
  const modelLabel = clampText(
    member.model_display_name || member.model || 'default',
    compact ? 10 : inSplitPane ? 13 : 20
  )
  const lineage =
    member.added_by && member.added_by !== 'coordinator'
      ? ` ← @${member.added_by}`
      : member.role !== 'worker'
        ? ` · ${member.role}`
        : ''
  const wsSuffix = !compact && member.workspace !== 'inherit' ? ` · ws:${member.workspace}` : ''
  const rightMeta = compact ? member.state : `${member.state} · ${modelLabel}${wsSuffix}`

  return (
    <Box
      backgroundColor={selected ? t.ds.selected : undefined}
      flexDirection="column"
      flexShrink={0}
      onMouseDown={onSelect}
      paddingX={1}
    >
      <Box flexDirection="row" justifyContent="space-between" gap={1}>
        <Text wrap="truncate-end">
          <Span color={selected ? t.color.accent : t.ds.meta}>{selected ? '› ' : '  '}</Span>
          <Span color={stateColor}>{`${glyph} `}</Span>
          <Span bold={selected} color={member.killed ? t.color.muted : t.color.text}>{`@${member.name}`}</Span>
          {lineage ? <Span color={t.ds.secondary}>{lineage}</Span> : null}
        </Text>
        <Text color={t.ds.meta} wrap="truncate-end">
          {rightMeta}
        </Text>
      </Box>
      {member.state === 'working' && member.current_activity ? (
        <Text color={t.ds.secondary} wrap="truncate-end">
          {`    ↳ ${member.current_activity}`}
        </Text>
      ) : null}
      {member.error ? (
        <Text color={t.color.warn} wrap="truncate-end">
          {`    ! ${member.error}`}
        </Text>
      ) : null}
    </Box>
  )
}

export function SwarmOverlay({ sessionId, t }: { sessionId?: string | null; t: Theme }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const uiSid = useStore($uiSessionId)
  const swarmStore = useStore($swarmStore)

  const sid = sessionId?.trim() || uiSid?.trim() || ''
  const slice = getSwarmSliceForSession(swarmStore, sid)
  const status = slice.status
  const messages = slice.messages

  const size = overlayPanelSize(terminal, { ...OVERLAY_PANEL_SPECS.swarm, maxHeight: 44, minHeight: 14 })
  // Spacious split view only on wide >= 120 col terminals; calm single-pane tabs below 120 cols.
  const wideSplit = terminal.width >= 120 && size.height >= 22
  const compact = size.width < 64 || size.height < 20

  const [selectedChannel, setSelectedChannel] = useState<string>('*')
  const [activeTab, setActiveTab] = useState<SwarmTab>('board')
  const [selectedMemberIdx, setSelectedMemberIdx] = useState(0)
  const [tailOffset, setTailOffset] = useState(0)
  const [composeMode, setComposeMode] = useState<ComposeMode>(null)
  const [draft, setDraft] = useState('')
  const [confirmKillTarget, setConfirmKillTarget] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(status === null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [refreshTick, setRefreshTick] = useState(0)
  const [inspectedMemberName, setInspectedMemberName] = useState<string | null>(null)
  const [inspectData, setInspectData] = useState<SwarmMemberInspectResponse | null>(null)
  const [inspectLoading, setInspectLoading] = useState(false)

  const alive = useRef(true)
  const boardScroll = useRef<ScrollBoxRenderable | null>(null)
  const overviewScroll = useRef<ScrollBoxRenderable | null>(null)
  const inspectScroll = useRef<ScrollBoxRenderable | null>(null)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  // Subscribe directly when a live GatewayClient emitter is supplied so
  // standalone overlay renders receive 'swarm_event' frames immediately too.
  useEffect(() => {
    const gw = gateway?.gw as unknown as {
      on?: (event: string, fn: (ev: GatewayEvent) => void) => void
      off?: (event: string, fn: (ev: GatewayEvent) => void) => void
    } | undefined
    if (!gw?.on || !gw?.off) return
    const onSwarmEvent = (ev: GatewayEvent) => {
      if (ev.type === 'swarm_event') {
        const eventSid = ev.payload?.session_id || ev.session_id || sid
        if (!sid || !eventSid || eventSid === sid) {
          applySwarmEvent(ev.payload, sid)
        }
      }
    }
    gw.on('swarm_event', onSwarmEvent)
    return () => {
      gw.off?.('swarm_event', onSwarmEvent)
    }
  }, [gateway?.gw, sid])

  useEffect(() => {
    let current = true
    if (!gateway) {
      setLoading(false)
      setError('Connect to a daemon to inspect the Immortals Board.')
      return
    }
    setLoading(status === null)
    void (async () => {
      try {
        const sessionParam = sid ? { session_id: sid } : {}
        const [rawStatus, rawBoard] = await Promise.all([
          gateway.rpc<Record<string, unknown>>('swarm.status', sessionParam, { reportError: false }),
          gateway.rpc<Record<string, unknown>>('swarm.board', { ...sessionParam, limit: 100 }, { reportError: false })
        ])
        if (!current) return
        if (!rawStatus) {
          setError('Could not load Immortals status.')
          return
        }
        if (rawStatus.ok === false) {
          setError(String(rawStatus.error ?? 'Immortals status request was refused.'))
          return
        }
        const appliedStatus = applySwarmStatus(rawStatus, sid)
        if (rawBoard && rawBoard.ok !== false) {
          applySwarmBoard(appliedStatus.session_id || sid, rawBoard, { resetIfEmptyHead: true })
        }
        setError('')
      } catch (cause) {
        if (current) setError(rpcErrorMessage(cause))
      } finally {
        if (current) setLoading(false)
      }
    })()

    return () => {
      current = false
    }
  }, [gateway, sid, refreshTick])

  // Load bounded transcript + live tool activity when inspecting a member.
  useEffect(() => {
    let current = true
    if (!inspectedMemberName || !gateway) {
      setInspectData(null)
      setInspectLoading(false)
      return
    }
    setInspectLoading(prev => (inspectData?.member.name === inspectedMemberName ? prev : true))
    void (async () => {
      try {
        const raw = await gateway.rpc<Record<string, unknown>>(
          'swarm.inspect',
          { ...(sid ? { session_id: sid } : {}), name: inspectedMemberName, limit: 60 },
          { reportError: false }
        )
        if (!current) return
        if (raw && raw.ok !== false) {
          setInspectData(normalizeSwarmInspectResponse(raw, sid, inspectedMemberName))
        }
      } catch {
        // Keep last snapshot or member row fallback if daemon predates swarm.inspect.
      } finally {
        if (current) setInspectLoading(false)
      }
    })()

    return () => {
      current = false
    }
  }, [gateway, sid, inspectedMemberName, refreshTick, slice.updatedAt])

  const channelTabs = useMemo(() => collectSwarmChannelTabs(status, messages), [status, messages])
  // Hide the channel strip completely when only #main exists.
  const hasCustomChannels = useMemo(
    () => channelTabs.some(ch => ch.name !== 'main'),
    [channelTabs]
  )
  const filterOrder = useMemo(
    () => (hasCustomChannels ? ['*', ...channelTabs.map(ch => ch.name)] : ['*']),
    [channelTabs, hasCustomChannels]
  )

  useEffect(() => {
    if (!filterOrder.includes(selectedChannel)) {
      setSelectedChannel('*')
      setTailOffset(0)
    }
  }, [filterOrder, selectedChannel])

  const members = status?.members ?? []
  const clampedMemberIdx = Math.min(selectedMemberIdx, Math.max(0, members.length - 1))
  const inspectedMember = inspectedMemberName
    ? members.find(m => m.name === inspectedMemberName) ?? inspectData?.member
    : undefined
  const selectedMember = inspectedMember ?? members[clampedMemberIdx]
  const canKillSelected = Boolean(
    selectedMember &&
      selectedMember.role !== 'coordinator' &&
      !selectedMember.killed &&
      selectedMember.state !== 'killed'
  )

  const filteredMessages = useMemo(
    () => (selectedChannel === '*' ? messages : messages.filter(m => m.channel === selectedChannel)),
    [messages, selectedChannel]
  )

  const hasOverview = Boolean(status?.overview && (status.overview.markdown.trim() || status.overview.summary))
  const overviewLines = useMemo(() => {
    if (!status?.overview) return []
    const raw = status.overview.markdown.trim() || status.overview.summary || ''
    return raw.split('\n')
  }, [status?.overview])

  const mentionState = useMemo(
    () => (composeMode === 'post' ? extractComposeMentionQuery(draft) : null),
    [composeMode, draft]
  )
  const mentionCandidates = useMemo(
    () =>
      composeMode === 'post'
        ? computeSwarmMentionCandidates(mentionState?.query ?? '', members, channelTabs)
        : [],
    [composeMode, mentionState?.query, members, channelTabs]
  )

  // Clean row budgeting without overlapping chrome.
  const borderAndPadRows = 2
  const headerRows = size.height >= 26 ? 4 : 1
  const footerRows = size.height >= 26 ? 2 : 1
  const tabBarRows = 1
  const channelBarRows = hasCustomChannels ? 1 : 0
  const overviewRibbonRows = wideSplit && activeTab !== 'overview' && hasOverview && !compact ? 1 : 0
  const extraBannerRows =
    (composeMode ? 3 : 0) + (confirmKillTarget ? 1 : 0) + (error || note ? 1 : 0)
  const chromeRows =
    borderAndPadRows +
    headerRows +
    footerRows +
    tabBarRows +
    channelBarRows +
    overviewRibbonRows +
    extraBannerRows
  const bodyRows = Math.max(4, size.height - chromeRows)

  const rosterVisibleCount = Math.max(2, bodyRows - (wideSplit ? 2 : 1))
  const rosterWindow = windowItems(members, clampedMemberIdx, rosterVisibleCount)

  const visibleMessageCount = Math.max(1, Math.floor((bodyRows - (wideSplit ? 1 : 0)) / 3))
  const maxTailOffset = Math.max(0, filteredMessages.length - visibleMessageCount)
  const clampedTailOffset = Math.min(tailOffset, maxTailOffset)
  const boardEndIdx = filteredMessages.length - clampedTailOffset
  const boardStartIdx = Math.max(0, boardEndIdx - visibleMessageCount)
  const visibleMessages = filteredMessages.slice(boardStartIdx, boardEndIdx)
  const pageStep = Math.max(2, visibleMessageCount)

  const closeOverlay = () => {
    patchOverlayState({ swarm: false })
    focusComposer()
  }

  const cycleChannel = (delta: number) => {
    if (!hasCustomChannels) return
    const idx = Math.max(0, filterOrder.indexOf(selectedChannel))
    const next = filterOrder[(idx + delta + filterOrder.length) % filterOrder.length] ?? '*'
    setSelectedChannel(next)
    setTailOffset(0)
  }

  const cycleTab = (delta: number) => {
    const order: SwarmTab[] = ['board', 'members', 'overview']
    const idx = Math.max(0, order.indexOf(activeTab))
    setActiveTab(order[(idx + delta + order.length) % order.length] ?? 'board')
  }

  const executeKill = async (targetName: string) => {
    if (!gateway || busy) return
    setBusy(true)
    setError('')
    setNote('')
    setConfirmKillTarget(null)
    try {
      const res = await gateway.rpc<SwarmKillResponse>(
        'swarm.kill',
        { ...(sid ? { session_id: sid } : {}), name: targetName },
        { reportError: false }
      )
      if (!alive.current) return
      if (!res || res.ok === false) {
        setError(res?.error || `Could not kill @${targetName}.`)
        return
      }
      const killedList =
        Array.isArray(res.killed) && res.killed.length > 0
          ? res.killed.map(n => `@${n}`).join(', ')
          : `@${targetName}`
      setNote(`Stopped ${killedList}`)
      await refreshSwarmSession(gateway.rpc, sid, { reportError: false })
    } catch (cause) {
      if (alive.current) setError(rpcErrorMessage(cause))
    } finally {
      if (alive.current) setBusy(false)
    }
  }

  const requestKillSelected = () => {
    if (!selectedMember) return
    if (selectedMember.role === 'coordinator') {
      setNote('Coordinator cannot be killed.')
      return
    }
    if (selectedMember.killed || selectedMember.state === 'killed') {
      setNote(`@${selectedMember.name} is already stopped.`)
      return
    }
    if (
      (selectedMember.role === 'narrator' || selectedMember.role === 'monitor') &&
      confirmKillTarget !== selectedMember.name
    ) {
      setError('')
      setNote('')
      setConfirmKillTarget(selectedMember.name)
      return
    }
    void executeKill(selectedMember.name)
  }

  const submitCompose = async () => {
    if (!gateway || busy || !composeMode) return
    const text = draft.trim()
    if (composeMode !== 'model' && !text) {
      setError(composeMode === 'feedback' ? 'Enter feedback for @narrator.' : 'Enter a message to post.')
      return
    }
    setBusy(true)
    setError('')
    setNote('')
    try {
      if (composeMode === 'post') {
        const channel = selectedChannel === '*' ? 'main' : selectedChannel
        const res = await gateway.rpc<SwarmPostResponse & { ok?: boolean; error?: string }>(
          'swarm.post',
          { ...(sid ? { session_id: sid } : {}), text, channel },
          { reportError: false }
        )
        if (!alive.current) return
        if (!res || res.ok === false) {
          setError(res?.error || 'Failed to post message to board.')
          return
        }
        setComposeMode(null)
        setDraft('')
        setTailOffset(0)
        setNote(`Posted #${res.seq} to #${res.channel || channel}`)
        await refreshSwarmSession(gateway.rpc, sid, { reportError: false })
        return
      }

      if (composeMode === 'feedback') {
        const res = await gateway.rpc<SwarmPostResponse & { ok?: boolean; error?: string }>(
          'swarm.post',
          { ...(sid ? { session_id: sid } : {}), text, as_feedback: true },
          { reportError: false }
        )
        if (!alive.current) return
        if (!res || res.ok === false) {
          setError(res?.error || 'Failed to send feedback to @narrator.')
          return
        }
        setComposeMode(null)
        setDraft('')
        setTailOffset(0)
        setNote(`Sent feedback to @narrator (#${res.seq})`)
        await refreshSwarmSession(gateway.rpc, sid, { reportError: false })
        return
      }

      if (composeMode === 'model' && selectedMember) {
        const res = await gateway.rpc<SwarmModelResponse>(
          'swarm.model',
          {
            ...(sid ? { session_id: sid } : {}),
            name: selectedMember.name,
            ...(text ? { model: text } : {})
          },
          { reportError: false }
        )
        if (!alive.current) return
        if (!res || res.ok === false) {
          setError(res?.error || `Failed to switch model for @${selectedMember.name}.`)
          return
        }
        setComposeMode(null)
        setDraft('')
        setNote(`Switched @${selectedMember.name}${res.model ? ` → ${res.model}` : ''}`)
        await refreshSwarmSession(gateway.rpc, sid, { reportError: false })
      }
    } catch (cause) {
      if (alive.current) setError(rpcErrorMessage(cause))
    } finally {
      if (alive.current) setBusy(false)
    }
  }

  useKeyboard(event => {
    if (event.eventType === 'release') return
    const key = event.name
    const seq = event.sequence ?? ''

    if (key === 'escape') {
      event.preventDefault()
      event.stopPropagation()
      if (composeMode) {
        setComposeMode(null)
        setDraft('')
        setError('')
        return
      }
      if (confirmKillTarget) {
        setConfirmKillTarget(null)
        return
      }
      if (inspectedMemberName) {
        setInspectedMemberName(null)
        setInspectData(null)
        return
      }
      closeOverlay()
      return
    }

    if (busy) {
      event.preventDefault()
      event.stopPropagation()
      return
    }

    // While an inline <input> is focused, allow Tab @mention completion and let <input> own text editing.
    if (composeMode) {
      if (key === 'tab' && composeMode === 'post' && mentionState && mentionCandidates.length > 0) {
        event.preventDefault()
        event.stopPropagation()
        setDraft(`${mentionState.prefix}${mentionCandidates[0]} `)
      }
      return
    }

    if (confirmKillTarget) {
      event.preventDefault()
      event.stopPropagation()
      if (
        key === 'y' ||
        key === 'return' ||
        key === 'enter' ||
        key === 'x' ||
        seq === 'K' ||
        (key === 'k' && event.shift)
      ) {
        void executeKill(confirmKillTarget)
      } else if (key === 'n') {
        setConfirmKillTarget(null)
      }
      return
    }

    if (key === 'tab') {
      event.preventDefault()
      event.stopPropagation()
      if (hasCustomChannels) {
        cycleChannel(event.shift ? -1 : 1)
      } else {
        cycleTab(event.shift ? -1 : 1)
      }
      return
    }

    if (seq === '[' || seq === ']') {
      event.preventDefault()
      event.stopPropagation()
      if (hasCustomChannels) {
        cycleChannel(seq === '[' ? -1 : 1)
      } else {
        cycleTab(seq === '[' ? -1 : 1)
      }
      return
    }

    if (inspectedMemberName && (key === 'backspace' || key === 'delete')) {
      event.preventDefault()
      event.stopPropagation()
      setInspectedMemberName(null)
      setInspectData(null)
      return
    }

    if (
      !inspectedMemberName &&
      activeTab === 'members' &&
      (key === 'return' || key === 'enter') &&
      selectedMember
    ) {
      event.preventDefault()
      event.stopPropagation()
      setError('')
      setNote('')
      setInspectedMemberName(selectedMember.name)
      return
    }

    if (seq === '1') {
      event.preventDefault()
      event.stopPropagation()
      setInspectedMemberName(null)
      setActiveTab('board')
      return
    }

    if (seq === '2') {
      event.preventDefault()
      event.stopPropagation()
      setInspectedMemberName(null)
      setActiveTab('members')
      return
    }

    if (seq === '3') {
      event.preventDefault()
      event.stopPropagation()
      setInspectedMemberName(null)
      setActiveTab('overview')
      return
    }

    if ((key === 'left' || key === 'right') && !event.ctrl && !event.meta && !event.shift) {
      event.preventDefault()
      event.stopPropagation()
      cycleTab(key === 'left' ? -1 : 1)
      return
    }

    if ((key === 'r' && !event.meta) || (event.ctrl && key === 'r')) {
      event.preventDefault()
      event.stopPropagation()
      setError('')
      setNote('')
      setRefreshTick(v => v + 1)
      return
    }

    if (key === 'o' && !event.ctrl && !event.meta) {
      event.preventDefault()
      event.stopPropagation()
      setActiveTab(prev => (prev === 'overview' ? 'board' : 'overview'))
      return
    }

    if (key === 'i' && !event.ctrl && !event.meta) {
      event.preventDefault()
      event.stopPropagation()
      const activeChannelObj = channelTabs.find(ch => ch.name === selectedChannel)
      if (selectedChannel !== '*' && activeChannelObj?.archived) {
        setError(`Channel #${selectedChannel} is archived — switch to #main to post.`)
        return
      }
      setError('')
      setNote('')
      setDraft('')
      setComposeMode('post')
      return
    }

    if (key === 'f' && !event.ctrl && !event.meta) {
      event.preventDefault()
      event.stopPropagation()
      setError('')
      setNote('')
      setDraft('')
      setComposeMode('feedback')
      return
    }

    if (key === 'm' && !event.ctrl && !event.meta && selectedMember) {
      event.preventDefault()
      event.stopPropagation()
      if (selectedMember.killed || selectedMember.state === 'killed') {
        setNote(`@${selectedMember.name} is killed and cannot switch models.`)
        return
      }
      if (selectedMember.state === 'working') {
        setNote(`Wait for @${selectedMember.name} to park before switching models.`)
        return
      }
      setError('')
      setNote('')
      setDraft('')
      setComposeMode('model')
      return
    }

    if ((key === 'x' && !event.ctrl && !event.meta) || seq === 'K' || (key === 'k' && event.shift)) {
      event.preventDefault()
      event.stopPropagation()
      requestKillSelected()
      return
    }

    if (isPageUpKey(event)) {
      event.preventDefault()
      event.stopPropagation()
      if (inspectedMemberName) {
        inspectScroll.current?.scrollBy(-6)
      } else if (activeTab === 'overview') {
        overviewScroll.current?.scrollBy(-6)
      } else {
        setTailOffset(prev => Math.min(maxTailOffset, prev + pageStep))
        boardScroll.current?.scrollBy(-pageStep)
      }
      return
    }

    if (isPageDownKey(event)) {
      event.preventDefault()
      event.stopPropagation()
      if (inspectedMemberName) {
        inspectScroll.current?.scrollBy(6)
      } else if (activeTab === 'overview') {
        overviewScroll.current?.scrollBy(6)
      } else {
        setTailOffset(prev => Math.max(0, prev - pageStep))
        boardScroll.current?.scrollBy(pageStep)
      }
      return
    }

    if (key === 'home') {
      event.preventDefault()
      event.stopPropagation()
      if (inspectedMemberName) {
        inspectScroll.current?.scrollTo(0)
      } else if (activeTab === 'members' && members.length > 0) {
        setSelectedMemberIdx(0)
      } else if (activeTab === 'overview') {
        overviewScroll.current?.scrollTo(0)
      } else {
        setTailOffset(maxTailOffset)
        boardScroll.current?.scrollTo(0)
      }
      return
    }

    if (key === 'end') {
      event.preventDefault()
      event.stopPropagation()
      if (inspectedMemberName) {
        inspectScroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      } else if (activeTab === 'members' && members.length > 0) {
        setSelectedMemberIdx(Math.max(0, members.length - 1))
      } else if (activeTab === 'overview') {
        overviewScroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      } else {
        setTailOffset(0)
        boardScroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      }
      return
    }

    if (key === 'up' || key === 'down') {
      event.preventDefault()
      event.stopPropagation()
      const delta = key === 'up' ? -1 : 1
      if (inspectedMemberName) {
        inspectScroll.current?.scrollBy(delta * 2)
      } else if (activeTab === 'members' && members.length > 0) {
        setSelectedMemberIdx(prev => Math.max(0, Math.min(members.length - 1, prev + delta)))
      } else if (activeTab === 'overview') {
        overviewScroll.current?.scrollBy(delta * 2)
      } else {
        setTailOffset(prev => Math.max(0, Math.min(maxTailOffset, prev - delta)))
        boardScroll.current?.scrollBy(delta * 2)
      }
    }
  })

  const counts = summarizeSwarmCounts(status)
  const rawTitle = status?.title.trim() || 'Immortals Board'
  const titleText = compact ? clampText(rawTitle, 20) : rawTitle
  const headSeq = Math.max(status?.head ?? 0, slice.head)
  const headerCounts = compact
    ? `${counts.working} working · ${counts.waiting} waiting · #${headSeq}`
    : `${counts.working} working · ${counts.waiting} waiting${counts.stopped ? ` · ${counts.stopped} stopped` : ''}${counts.killed ? ` · ${counts.killed} killed` : ''} · #${headSeq}`
  const emptySwarm = !loading && !status?.active && members.length === 0 && messages.length === 0
  const postChannel = selectedChannel === '*' ? 'main' : selectedChannel

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={180}
    >
      <Box
        backgroundColor={t.color.overlayBg}
        borderColor={t.color.border}
        borderStyle="round"
        flexDirection="column"
        height={emptySwarm && !composeMode ? Math.min(22, size.height) : size.height}
        paddingX={1}
        width={emptySwarm && !composeMode ? Math.min(90, size.width) : size.width}
      >
        <DialogHeader
          t={t}
          title={
            <>
              {titleText}
              <Span color={t.ds.secondary}>{` · ${headerCounts}`}</Span>
            </>
          }
        />

        {/* Clean Primary View Tabs */}
        {!emptySwarm ? (
          <Box flexDirection="row" flexShrink={0} gap={2} overflow="hidden" paddingBottom={hasCustomChannels ? 0 : compact ? 0 : 1}>
            <Box onMouseDown={() => setActiveTab('board')}>
              <Text bold={activeTab === 'board'} color={activeTab === 'board' ? t.color.accent : t.ds.secondary}>
                {`${activeTab === 'board' ? '› ' : ''}1 Board${selectedChannel !== '*' ? ` (#${selectedChannel})` : filteredMessages.length > 0 ? ` (${filteredMessages.length})` : ''}`}
              </Text>
            </Box>
            <Text color={t.ds.separator}>{'·'}</Text>
            <Box onMouseDown={() => setActiveTab('members')}>
              <Text bold={activeTab === 'members'} color={activeTab === 'members' ? t.color.accent : t.ds.secondary}>
                {`${activeTab === 'members' ? '› ' : ''}2 Members (${members.length})`}
              </Text>
            </Box>
            <Text color={t.ds.separator}>{'·'}</Text>
            <Box onMouseDown={() => setActiveTab('overview')}>
              <Text bold={activeTab === 'overview'} color={activeTab === 'overview' ? t.color.accent : hasOverview ? t.ds.secondary : t.color.muted}>
                {`${activeTab === 'overview' ? '› ' : ''}3 Overview${hasOverview && status?.overview?.author ? ` (@${status.overview.author})` : ''}`}
              </Text>
            </Box>
          </Box>
        ) : null}

        {/* Channel Strip — hidden when only #main exists */}
        {!emptySwarm && hasCustomChannels ? (
          <Box flexDirection="row" flexShrink={0} gap={1} overflow="hidden" paddingBottom={compact ? 0 : 1}>
            {filterOrder.map(chKey => {
              const isSelected = selectedChannel === chKey
              const chMeta = chKey === '*' ? undefined : channelTabs.find(c => c.name === chKey)
              const label =
                chKey === '*'
                  ? 'All'
                  : `#${chKey}${chMeta?.archived ? ' [archived]' : ''}`
              return (
                <Box
                  backgroundColor={isSelected ? t.ds.selected : undefined}
                  flexShrink={0}
                  key={chKey}
                  onMouseDown={() => {
                    setSelectedChannel(chKey)
                    setTailOffset(0)
                  }}
                  paddingX={1}
                >
                  <Text bold={isSelected} color={isSelected ? t.color.accent : chMeta?.archived ? t.color.muted : t.ds.secondary}>
                    {label}
                  </Text>
                </Box>
              )
            })}
          </Box>
        ) : null}

        {/* Quiet 1-line Narrator ribbon on wide split view */}
        {!emptySwarm && wideSplit && activeTab !== 'overview' && hasOverview && status?.overview && !compact ? (
          <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1} onMouseDown={() => setActiveTab('overview')}>
            <Text wrap="truncate-end">
              <Span color={t.color.accent}>{'Overview'}</Span>
              <Span color={t.ds.secondary}>
                {` · ${status.overview.summary || overviewLines[0] || 'Updated'}`}
              </Span>
            </Text>
            <Text color={t.ds.meta}>{'3 read · f feedback'}</Text>
          </Box>
        ) : null}

        {error ? (
          <Text color={t.color.warn} wrap="truncate-end">
            {error}
          </Text>
        ) : note ? (
          <Text color={t.color.ok} wrap="truncate-end">
            {note}
          </Text>
        ) : null}

        {confirmKillTarget ? (
          <Box backgroundColor={t.color.completionMetaBg} flexShrink={0} paddingX={1}>
            <Text color={t.color.warn} wrap="truncate-end">
              {`Confirm killing ${selectedMember?.role ?? 'member'} @${confirmKillTarget}? Press y / Enter to confirm, Esc / n to cancel.`}
            </Text>
          </Box>
        ) : null}

        {loading && !status ? (
          <Box alignItems="center" flexGrow={1} justifyContent="center">
            <Text color={t.ds.secondary}>Loading Immortals Board…</Text>
          </Box>
        ) : emptySwarm ? (
          <DialogEmpty
            action="i Post to #main"
            description="Run /immortals <task> in chat to launch a parallel Immortals force, or press i to post on #main."
            onAction={() => {
              setComposeMode('post')
              setDraft('')
            }}
            symbol="✦"
            t={t}
            title="No active Immortals run in this session."
          />
        ) : inspectedMemberName && selectedMember ? (
          /* Live Worker Inspector Detail View (Enter on member; Esc/Backspace returns to Members) */
          (() => {
            const liveActivity =
              slice.activities[inspectedMemberName] ?? inspectData?.activity
            const activeTools = liveActivity?.active_tools ?? []
            const recentTools = liveActivity?.recent_tools ?? []
            const inspectMessages = inspectData?.messages ?? []
            const modelText = selectedMember.model_display_name || selectedMember.model || 'default'
            return (
              <Box flexDirection="column" flexGrow={1} minHeight={0}>
                <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
                  <Text wrap="truncate-end">
                    <Span color={t.ds.secondary}>{'← Members · '}</Span>
                    <Span bold color={t.color.accent}>{`@${selectedMember.name}`}</Span>
                    <Span color={memberStateColor(selectedMember.state, t)}>{` · ${selectedMember.state}`}</Span>
                    <Span color={t.ds.secondary}>
                      {` · ${selectedMember.role}${selectedMember.added_by ? ` ← @${selectedMember.added_by}` : ''} · ${modelText} · ws:${selectedMember.workspace}`}
                    </Span>
                  </Text>
                  <Text color={t.ds.meta}>
                    {`m model${canKillSelected ? ' · x stop' : ''} · Esc back`}
                  </Text>
                </Box>

                {liveActivity && (liveActivity.phase !== 'idle' || liveActivity.summary || activeTools.length > 0 || recentTools.length > 0) ? (
                  <Box flexDirection="column" flexShrink={0} marginBottom={1} paddingLeft={1}>
                    <Text wrap="truncate-end">
                      <Span bold color={t.color.accent}>{'Activity'}</Span>
                      <Span color={t.ds.secondary}>
                        {` · ${liveActivity.summary || liveActivity.phase}`}
                      </Span>
                    </Text>
                    {activeTools.map(tool => (
                      <Text color={t.color.accent} key={`active-${tool.id}`} wrap="truncate-end">
                        {`  ● ${tool.name}${tool.summary ? ` · ${tool.summary}` : ''}`}
                      </Text>
                    ))}
                    {recentTools.slice(-4).map(tool => (
                      <Text color={tool.status === 'error' ? t.color.warn : t.ds.meta} key={`recent-${tool.id}`} wrap="truncate-end">
                        {`  ${tool.status === 'error' ? '!' : '✓'} ${tool.name}${tool.summary ? ` · ${tool.summary}` : ''}${tool.result_preview ? ` → ${tool.result_preview}` : ''}`}
                      </Text>
                    ))}
                    {liveActivity.live_text_preview ? (
                      <Text color={t.color.text} wrap="truncate-end">
                        {`  ▍ ${liveActivity.live_text_preview.replace(/\s+/g, ' ').trim()}`}
                      </Text>
                    ) : null}
                  </Box>
                ) : null}

                <scrollbox
                  contentOptions={{ flexDirection: 'column' }}
                  ref={inspectScroll}
                  stickyScroll
                  stickyStart="bottom"
                  style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
                >
                  {inspectLoading && inspectMessages.length === 0 ? (
                    <Text color={t.ds.secondary}>{`Loading @${selectedMember.name} conversation…`}</Text>
                  ) : inspectMessages.length === 0 ? (
                    <Text color={t.ds.meta}>
                      {`No conversation turns recorded yet for @${selectedMember.name}.`}
                    </Text>
                  ) : (
                    inspectMessages.map(turn => (
                      <Box flexDirection="column" flexShrink={0} key={turn.index} marginBottom={1}>
                        <Text wrap="truncate-end">
                          <Span
                            bold
                            color={
                              turn.role === 'user'
                                ? t.color.brandGold
                                : turn.role === 'tool'
                                  ? t.ds.secondary
                                  : t.color.accent
                            }
                          >
                            {turn.role === 'tool'
                              ? `tool:${turn.tool_name || 'result'}`
                              : turn.role}
                          </Span>
                        </Text>
                        {turn.text ? (
                          <Box paddingLeft={2}>
                            <Text color={t.color.text} wrap="wrap">
                              {turn.text}
                            </Text>
                          </Box>
                        ) : null}
                        {turn.tool_calls?.map(call => (
                          <Box key={call.id} paddingLeft={2}>
                            <Text color={t.ds.secondary} wrap="truncate-end">
                              {`→ ${call.name}${call.args_preview ? ` (${call.args_preview})` : ''}`}
                            </Text>
                          </Box>
                        ))}
                      </Box>
                    ))
                  )}
                </scrollbox>
              </Box>
            )
          })()
        ) : activeTab === 'overview' ? (
          /* Dedicated Editorial Overview Tab */
          <Box flexDirection="column" flexGrow={1} minHeight={0}>
            <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
              <Text wrap="truncate-end">
                <Span bold color={t.color.accent}>{'Narrator Overview'}</Span>
                {status?.overview?.summary ? (
                  <Span color={t.ds.secondary}>{` · ${status.overview.summary}`}</Span>
                ) : null}
                {status?.overview?.author ? (
                  <Span color={t.ds.meta}>{` (@${status.overview.author})`}</Span>
                ) : null}
              </Text>
              <Text color={t.ds.meta}>{'f feedback'}</Text>
            </Box>
            <scrollbox
              contentOptions={{ flexDirection: 'column' }}
              ref={overviewScroll}
              style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
            >
              {!hasOverview ? (
                <Text color={t.ds.meta}>
                  No narrator overview published yet. Press f to send instructions to @narrator.
                </Text>
              ) : (
                overviewLines.map((line, idx) => (
                  <Text color={t.color.text} key={idx} wrap="wrap">
                    {line || ' '}
                  </Text>
                ))
              )}
            </scrollbox>
          </Box>
        ) : (
          /* Spacious Split View on Wide (>=120 col) Terminals; Clean Single-Pane on Narrow/Medium */
          <Box flexDirection="row" flexGrow={1} gap={2} minHeight={0} overflow="hidden">
            {/* Board Timeline */}
            {(wideSplit || activeTab === 'board') ? (
              <Box flexDirection="column" flexGrow={1} minHeight={0} minWidth={0}>
                {wideSplit ? (
                  <Box flexDirection="row" justifyContent="space-between" marginBottom={1} onMouseDown={() => setActiveTab('board')}>
                    <Text bold={activeTab === 'board'} color={activeTab === 'board' ? t.color.accent : t.ds.secondary}>
                      {`${activeTab === 'board' ? '› ' : ''}Board${selectedChannel !== '*' ? ` · #${selectedChannel}` : ''}`}
                    </Text>
                    {filteredMessages.length > 0 ? (
                      <Text color={t.ds.meta}>
                        {clampedTailOffset > 0
                          ? `#${visibleMessages[0]?.seq ?? 1}–#${visibleMessages.at(-1)?.seq ?? headSeq} of ${filteredMessages.length} (↑ scrolled)`
                          : `${filteredMessages.length} msg${filteredMessages.length === 1 ? '' : 's'}`}
                      </Text>
                    ) : null}
                  </Box>
                ) : clampedTailOffset > 0 && filteredMessages.length > 0 ? (
                  <Box flexDirection="row" justifyContent="flex-end">
                    <Text color={t.ds.meta}>
                      {`#${visibleMessages[0]?.seq ?? 1}–#${visibleMessages.at(-1)?.seq ?? headSeq} of ${filteredMessages.length} (↑ scrolled)`}
                    </Text>
                  </Box>
                ) : null}
                <scrollbox
                  contentOptions={{ flexDirection: 'column' }}
                  ref={boardScroll}
                  stickyScroll={clampedTailOffset === 0}
                  stickyStart="bottom"
                  style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
                >
                  {visibleMessages.length === 0 ? (
                    <Text color={t.ds.meta}>
                      {selectedChannel === '*'
                        ? 'No messages posted to the board yet.'
                        : `No messages in #${selectedChannel} yet.`}
                    </Text>
                  ) : (
                    visibleMessages.map(msg => (
                      <BoardMessageRow
                        key={msg.seq}
                        msg={msg}
                        showChannelBadge={hasCustomChannels && (selectedChannel === '*' || msg.channel !== 'main')}
                        t={t}
                      />
                    ))
                  )}
                </scrollbox>
              </Box>
            ) : null}

            {/* Members List */}
            {(wideSplit || activeTab === 'members') ? (
              <Box
                borderColor={wideSplit ? t.ds.hairline : undefined}
                borderSides={wideSplit ? ['left'] : undefined}
                flexDirection="column"
                flexShrink={0}
                minHeight={0}
                overflow="hidden"
                paddingLeft={wideSplit ? 1 : 0}
                width={wideSplit ? Math.min(58, Math.max(44, Math.floor(size.width * 0.45))) : '100%'}
              >
                {wideSplit ? (
                  <Box flexDirection="row" justifyContent="space-between" marginBottom={1} onMouseDown={() => setActiveTab('members')}>
                    <Text bold={activeTab === 'members'} color={activeTab === 'members' ? t.color.accent : t.ds.secondary}>
                      {`${activeTab === 'members' ? '› ' : ''}Members (${members.length})`}
                    </Text>
                    <Text color={t.ds.meta}>
                      {`Enter inspect · m model${canKillSelected ? ' · x stop' : ''}`}
                    </Text>
                  </Box>
                ) : null}
                {members.length === 0 ? (
                  <Text color={t.ds.meta}>No workers added yet.</Text>
                ) : (
                  rosterWindow.items.map((member, idx) => {
                    const actualIdx = rosterWindow.offset + idx
                    return (
                      <RosterMemberRow
                        compact={compact}
                        inSplitPane={wideSplit}
                        key={member.name}
                        member={member}
                        onSelect={() => {
                          setActiveTab('members')
                          setSelectedMemberIdx(actualIdx)
                        }}
                        selected={actualIdx === clampedMemberIdx}
                        t={t}
                      />
                    )
                  })
                )}
              </Box>
            ) : null}
          </Box>
        )}

        {/* Inline Composer with @mention completion hint */}
        {composeMode ? (
          <Box flexDirection="column" flexShrink={0} marginTop={compact ? 0 : 1}>
            <Box flexDirection="row" justifyContent="space-between" gap={1}>
              <Text color={t.color.accent} wrap="truncate-end">
                {composeMode === 'feedback'
                  ? 'Feedback to @narrator (Enter send · Esc cancel)'
                  : composeMode === 'model'
                    ? `Switch model for @${selectedMember?.name ?? 'worker'} (Enter apply · Esc cancel)`
                    : `Message #${postChannel} (Enter post · Esc cancel)`}
              </Text>
              {composeMode === 'post' && mentionCandidates.length > 0 ? (
                <Text color={t.ds.meta} wrap="truncate-end">
                  {mentionState
                    ? `Tab complete: ${mentionCandidates.join(' · ')}`
                    : `Mentions: ${mentionCandidates.slice(0, 4).join(' ')}`}
                </Text>
              ) : null}
            </Box>
            <input
              backgroundColor={t.color.completionMetaBg}
              focused={!busy}
              onInput={setDraft}
              onSubmit={() => {
                void submitCompose()
              }}
              placeholder={
                composeMode === 'feedback'
                  ? 'Describe how @narrator should adjust the overview…'
                  : composeMode === 'model'
                    ? 'Enter model ID (or leave blank for coordinator model)…'
                    : `Message #${postChannel} — type @ to mention…`
              }
              textColor={t.color.text}
              value={draft}
            />
          </Box>
        ) : null}

        <DialogFooter t={t}>
          <Text color={t.ds.secondary} wrap="truncate-end">
            {busy
              ? 'Applying Immortals action…'
              : composeMode
                ? mentionState && mentionCandidates.length > 0
                  ? 'Tab complete @mention · Enter submit · Esc cancel'
                  : 'Enter submit · Esc cancel'
                : inspectedMemberName
                  ? `↑↓/PgUp/PgDn scroll · m model${canKillSelected ? ' · x stop' : ''} · Backspace/Esc back to Members`
                  : compact
                    ? `1/2/3 view · i post · f narrator${canKillSelected ? ' · x stop' : ''} · Esc close`
                    : `1/2/3 view${hasCustomChannels ? ' · Tab/[ ] channel' : ''} · ↑↓ navigate${activeTab === 'members' ? ' · Enter inspect' : ''} · i post · f feedback · m model${canKillSelected ? ' · x stop' : ''} · r refresh · Esc close`}
          </Text>
        </DialogFooter>
      </Box>
    </box>
  )
}
