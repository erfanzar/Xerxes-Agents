// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppLayoutActions } from '../app/interfaces.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme } from '../app/uiStore.js'
import type {
  LiveSessionStatus,
  SessionActiveItem,
  SessionActiveListResponse,
  SessionListItem,
  SessionListResponse
} from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

const MAX_VISIBLE = 12
const MIN_PANEL_WIDTH = 56
const MAX_PANEL_WIDTH = 104

export type SessionPickerActions = Pick<AppLayoutActions, 'activateLiveSession' | 'newLiveSession' | 'resumeById'>

type SessionView = 'agents' | 'chats'

export interface SessionPickerProps {
  actions: SessionPickerActions
  currentSessionId?: null | string
  onCancel?: () => void
  t?: Theme
}

type SessionRow =
  | { id: 'new'; kind: 'new' }
  | { id: string; item: SessionActiveItem; kind: 'live'; parentTitle?: string }
  | { id: string; item: SessionListItem; kind: 'saved'; parentTitle?: string }

const STATUS_GLYPH: Record<LiveSessionStatus, string> = {
  idle: '✓',
  starting: '…',
  waiting: '?',
  working: '◆'
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

const shortId = (id: string) => (id.length > 10 ? `${id.slice(0, 9)}…` : id)

const shortModel = (model = '') => model.replace(/^.*\//, '') || 'default'

const compact = (value: string, max: number) =>
  value.length > max ? `${value.slice(0, Math.max(1, max - 1)).trimEnd()}…` : value

const isAgentSession = (item: SessionActiveItem | SessionListItem) => {
  const kind = item.kind?.trim().toLowerCase()

  return kind === 'subagent' || Boolean(item.subagent_id?.trim())
}

const isResumableAgentHistory = (item: SessionListItem) => {
  const status = item.status?.trim().toLowerCase()

  return item.resumable !== false && status !== 'running' && status !== 'working' && status !== 'starting'
}

const linkedParentId = (item: SessionActiveItem | SessionListItem) =>
  item.parent_session_id?.trim() || item.root_session_id?.trim() || ''

const itemTitle = (item: SessionActiveItem | SessionListItem) => item.title || item.preview || '(untitled)'

const agentStatusColor = (status: string, t: Theme) => {
  const normalized = status.toLowerCase()

  if (normalized === 'completed' || normalized === 'idle') return t.color.ok
  if (normalized === 'running' || normalized === 'working' || normalized === 'starting') return t.color.accent
  if (normalized === 'failed' || normalized === 'error') return t.color.error
  if (normalized === 'cancelled' || normalized === 'interrupted' || normalized === 'timeout') return t.color.warn

  return t.color.muted
}

const agentStatusGlyph = (status: string) => {
  const normalized = status.toLowerCase()

  if (normalized === 'completed' || normalized === 'idle') return '✓'
  if (normalized === 'running' || normalized === 'working' || normalized === 'starting') return '◆'
  if (normalized === 'failed' || normalized === 'error') return '×'
  if (normalized === 'cancelled' || normalized === 'interrupted' || normalized === 'timeout') return '!'

  return '·'
}

const agentRowLabel = ({
  age,
  maxWidth,
  messages,
  parent,
  profile,
  status,
  title
}: {
  age: string
  maxWidth: number
  messages: number
  parent: string
  profile: string
  status: string
  title: string
}) => {
  const full = `↳ ${status} · ${title} · ${profile} · ${messages} msgs${age ? ` · ${age}` : ''} · ← ${compact(parent, 22)}`

  if (full.length <= maxWidth) return full

  const narrow = (rowTitle: string) =>
    `↳ ${agentStatusGlyph(status)} ${rowTitle} · ${compact(profile, 8)} · ${messages}msg${age ? ` · ${age}` : ''} ← ${compact(parent, 8)}`
  const candidate = narrow(title)

  if (candidate.length <= maxWidth) return candidate

  const titleWidth = Math.max(6, title.length - (candidate.length - maxWidth))

  return compact(narrow(compact(title, titleWidth)), maxWidth)
}

const relativeAge = (timestamp?: number) => {
  if (!timestamp) {
    return ''
  }

  const elapsed = Math.max(0, Date.now() / 1000 - timestamp)
  const minutes = Math.floor(elapsed / 60)

  if (minutes < 1) {
    return 'now'
  }

  if (minutes < 60) {
    return `${minutes}m`
  }

  const hours = Math.floor(minutes / 60)

  if (hours < 24) {
    return `${hours}h`
  }

  return `${Math.floor(hours / 24)}d`
}

const windowItems = <T,>(items: readonly T[], selected: number, visible: number) => {
  if (visible <= 0) {
    return { items: [] as readonly T[], offset: 0 }
  }

  const offset = Math.max(0, Math.min(selected - Math.floor(visible / 2), items.length - visible))

  return { items: items.slice(offset, offset + visible), offset }
}

function ModalShell({
  children,
  height,
  panelHeight,
  panelWidth,
  t,
  width
}: {
  children: ReactNode
  height: number
  panelHeight: number
  panelWidth: number
  t: Theme
  width: number
}) {
  const top = Math.max(0, Math.floor((height - panelHeight) / 2))

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height={height}
      left={0}
      paddingTop={top}
      position="absolute"
      top={0}
      width={width}
      zIndex={200}
    >
      <box
        backgroundColor={t.color.statusBg}
        flexDirection="column"
        flexShrink={0}
        height={panelHeight}
        paddingBottom={1}
        paddingTop={1}
        width={panelWidth}
      >
        <box flexDirection="row" flexShrink={0} justifyContent="space-between" paddingLeft={2} paddingRight={2}>
          <text fg={t.color.accent} flexShrink={0}>
            <b>Sessions</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc
          </text>
        </box>
        {children}
      </box>
    </box>
  )
}

function InfoRow({ children, color }: { children: ReactNode; color: string }) {
  return (
    <box flexShrink={0} height={1} paddingLeft={2} paddingRight={2}>
      <text fg={color} flexShrink={0} truncate width="100%" wrapMode="none">
        {children}
      </text>
    </box>
  )
}

function SessionListRow({
  index,
  maxLabelWidth,
  row,
  selected,
  t
}: {
  index: number
  maxLabelWidth: number
  row: SessionRow
  selected: boolean
  t: Theme
}) {
  let label: string
  let color = t.color.muted

  if (row.kind === 'new') {
    label = '+  new session'
    color = t.color.label
  } else if (isAgentSession(row.item)) {
    const title = itemTitle(row.item)
    const profile = row.item.agent_id?.trim() || shortModel(row.item.model)
    const status = row.item.status?.trim() || (row.kind === 'live' ? 'idle' : 'saved')
    const age = relativeAge(row.kind === 'live' ? row.item.last_active ?? row.item.started_at : row.item.started_at)
    const messages = row.item.message_count ?? 0
    const parent = row.parentTitle || shortId(linkedParentId(row.item)) || 'main chat'
    label = agentRowLabel({ age, maxWidth: maxLabelWidth, messages, parent, profile, status, title })
    color = agentStatusColor(status, t)
  } else if (row.kind === 'live') {
    const current = Boolean(row.item.current)
    const title = itemTitle(row.item)
    const age = relativeAge(row.item.last_active ?? row.item.started_at)
    const identity = current ? 'current' : shortId(row.item.id)
    const ageSuffix = age ? ` · ${age}` : ''
    label = `${STATUS_GLYPH[row.item.status] ?? '·'}  ${identity} · ${row.item.status} · ${shortModel(row.item.model)}${ageSuffix} · ${title}`
    color = current ? t.color.warn : row.item.status === 'working' ? t.color.ok : t.color.text
  } else {
    const title = itemTitle(row.item)
    const age = relativeAge(row.item.started_at)
    const ageSuffix = age ? ` · ${age}` : ''
    label = `↻  ${shortId(row.item.id)}${ageSuffix} · ${row.item.message_count} msgs · ${title}`
  }

  return (
    <box
      backgroundColor={selected ? t.color.completionCurrentBg : undefined}
      flexShrink={0}
      height={1}
      paddingLeft={2}
      paddingRight={2}
      width="100%"
    >
      <text fg={selected ? t.color.accent : color} flexShrink={0} truncate width="100%" wrapMode="none">
        {selected ? '›' : ' '} {String(index + 1).padStart(2)}. {label}
      </text>
    </box>
  )
}

export function SessionPicker({ actions, currentSessionId, onCancel, t: suppliedTheme }: SessionPickerProps) {
  const { gw } = useGateway()
  const storeSessionId = useStore($uiSessionId)
  const storeTheme = useStore($uiTheme)
  const { height, width } = useTerminalDimensions()
  const t = suppliedTheme ?? storeTheme
  const effectiveSessionId = currentSessionId === undefined ? storeSessionId : currentSessionId

  const [activeSessions, setActiveSessions] = useState<SessionActiveItem[]>([])
  const [savedSessions, setSavedSessions] = useState<SessionListItem[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(0)
  const [view, setView] = useState<SessionView>('chats')

  const close = useCallback(() => {
    patchOverlayState({ sessions: false })
    onCancel?.()
  }, [onCancel])

  useEffect(() => {
    let mounted = true

    setLoading(true)
    setError('')

    void Promise.allSettled([
      gw.request<SessionActiveListResponse>('session.active_list', {
        current_session_id: effectiveSessionId
      }),
      // The daemon already reads every transcript to build summaries. Fetch
      // both kinds in one pass so large swarms do not double filesystem work
      // or disappear behind an arbitrary child-history cap.
      gw.request<SessionListResponse>('session.list', { kind: 'all', limit: 0 })
    ]).then(results => {
      if (!mounted) {
        return
      }

      const liveResult = results[0]
      const historyResult = results[1]
      const errors: string[] = []
      let nextActive: SessionActiveItem[] = []
      let nextSaved: SessionListItem[] = []

      if (liveResult.status === 'fulfilled') {
        const parsed = asRpcResult<SessionActiveListResponse>(liveResult.value)

        if (parsed) {
          nextActive = parsed.sessions ?? []
        } else {
          errors.push('invalid response: session.active_list')
        }
      } else {
        errors.push(rpcErrorMessage(liveResult.reason))
      }

      if (historyResult.status === 'fulfilled') {
        const parsed = asRpcResult<SessionListResponse>(historyResult.value)

        if (parsed) {
          nextSaved = parsed.sessions ?? []
        } else {
          errors.push('invalid response: session.list')
        }
      } else {
        errors.push(rpcErrorMessage(historyResult.reason))
      }

      const liveIds = new Set(nextActive.map(session => session.id))
      const savedIds = new Set<string>()
      nextSaved = nextSaved.filter(session => {
        if (liveIds.has(session.id) || savedIds.has(session.id)) return false
        savedIds.add(session.id)

        return true
      })

      setActiveSessions(nextActive)
      setSavedSessions(nextSaved)
      setError(errors.join(' · '))
      const current = nextActive.find(
        session => Boolean(session.current) || (!!effectiveSessionId && session.id === effectiveSessionId)
      ) ?? nextSaved.find(session => !!effectiveSessionId && session.id === effectiveSessionId)
      const nextView: SessionView = current && isAgentSession(current) ? 'agents' : 'chats'
      setView(nextView)
      setSelected(() => {
        const visible = [...nextActive, ...nextSaved].filter(session => isAgentSession(session) === (nextView === 'agents'))
        const currentIndex = visible.findIndex(session => (
          Boolean('current' in session && session.current) || (!!effectiveSessionId && session.id === effectiveSessionId)
        ))

        return currentIndex >= 0 ? currentIndex + (nextView === 'chats' ? 1 : 0) : 0
      })
      setLoading(false)
    })

    return () => {
      mounted = false
    }
  }, [effectiveSessionId, gw])

  const titleById = useMemo(
    () => new Map(
      [...activeSessions, ...savedSessions]
        .filter(item => item.id.trim())
        .map(item => [item.id, itemTitle(item)] as const)
    ),
    [activeSessions, savedSessions]
  )
  const withParentTitle = useCallback(
    <T extends SessionActiveItem | SessionListItem>(item: T) => {
      const parentId = linkedParentId(item)

      return parentId ? titleById.get(parentId) : undefined
    },
    [titleById]
  )
  const chatRows = useMemo<SessionRow[]>(
    () => [
      { id: 'new', kind: 'new' },
      ...activeSessions
        .filter(item => !isAgentSession(item))
        .map(item => ({ id: item.id, item, kind: 'live' }) as const),
      ...savedSessions
        .filter(item => !isAgentSession(item))
        .map(item => ({ id: item.id, item, kind: 'saved' }) as const)
    ],
    [activeSessions, savedSessions]
  )
  const agentRows = useMemo<SessionRow[]>(
    () => [
      ...activeSessions
        .filter(isAgentSession)
        .map(item => ({ id: item.id, item, kind: 'live', parentTitle: withParentTitle(item) }) as const),
      ...savedSessions
        .filter(isAgentSession)
        .map(item => ({ id: item.id, item, kind: 'saved', parentTitle: withParentTitle(item) }) as const)
    ],
    [activeSessions, savedSessions, withParentTitle]
  )
  const rows = view === 'chats' ? chatRows : agentRows

  useEffect(() => {
    setSelected(index => Math.max(0, Math.min(index, rows.length - 1)))
  }, [rows.length])

  const switchView = useCallback((next: SessionView) => {
    setView(next)
    setSelected(0)
  }, [])

  const choose = useCallback(() => {
    const row = rows[selected]

    if (!row) {
      return
    }

    if (row.kind === 'saved' && isAgentSession(row.item) && !isResumableAgentHistory(row.item)) {
      setError('That agent is still running in its parent chat. Wait for it to finish before resuming its history.')

      return
    }

    if (row.kind === 'new') {
      close()
      actions.newLiveSession()
    } else if (row.kind === 'live') {
      close()
      actions.activateLiveSession(row.item.id)
    } else {
      close()
      actions.resumeById(row.item.id)
    }
  }, [actions, close, rows, selected])

  const handleKey = useCallback(
    (event: KeyEvent) => {
      const name = event.name.toLowerCase()
      const sequence = event.sequence ?? ''
      const isEscape = name === 'escape'
      const isQuit = !event.ctrl && !event.meta && !event.super && (name === 'q' || sequence === 'q')

      if (isEscape || isQuit) {
        consume(event)
        close()

        return
      }

      if (loading) {
        return
      }

      if (name === 'tab') {
        consume(event)
        switchView(view === 'chats' ? 'agents' : 'chats')

        return
      }

      if (name === 'left') {
        consume(event)
        switchView('chats')

        return
      }

      if (name === 'right') {
        consume(event)
        switchView('agents')

        return
      }

      if (name === 'up') {
        consume(event)
        setSelected(index => Math.max(0, index - 1))

        return
      }

      if (name === 'down') {
        consume(event)
        setSelected(index => Math.max(0, Math.min(rows.length - 1, index + 1)))

        return
      }

      if (name === 'home') {
        consume(event)
        setSelected(0)

        return
      }

      if (name === 'end') {
        consume(event)
        setSelected(rows.length - 1)

        return
      }

      if (name === 'return' || name === 'enter' || name === 'kpenter') {
        consume(event)
        choose()
      }
    },
    [choose, close, loading, rows.length, switchView, view]
  )

  useKeyboard(handleKey)

  const panelWidth = Math.max(
    1,
    Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, width - 6), Math.max(1, width - 2))
  )
  const visible = Math.max(1, Math.min(MAX_VISIBLE, height - 10))
  const { items: visibleRows, offset } = windowItems(rows, selected, visible)
  const panelHeight = Math.min(height, visible + 7 + (error ? 1 : 0))

  if (loading) {
    return (
      <ModalShell height={height} panelHeight={5} panelWidth={panelWidth} t={t} width={width}>
        <InfoRow color={t.color.muted}>loading live and saved sessions…</InfoRow>
      </ModalShell>
    )
  }

  return (
    <ModalShell height={height} panelHeight={panelHeight} panelWidth={panelWidth} t={t} width={width}>
      <InfoRow color={t.color.muted}>
        {view === 'chats'
          ? `[Chats ${Math.max(0, chatRows.length - 1)}] · Agents ${agentRows.length}`
          : `[Agents ${agentRows.length}] · Chats ${Math.max(0, chatRows.length - 1)}`}
        {` · ${activeSessions.length} live`}
      </InfoRow>
      {error ? <InfoRow color={t.color.error}>error: {error}</InfoRow> : null}
      <InfoRow color={t.color.muted}>{offset > 0 ? `↑ ${offset} more` : ' '}</InfoRow>

      {visibleRows.map((row, index) => {
        const absoluteIndex = offset + index

        return (
          <SessionListRow
            index={absoluteIndex}
            key={`${row.kind}:${row.id}`}
            maxLabelWidth={Math.max(12, panelWidth - 10)}
            row={row}
            selected={selected === absoluteIndex}
            t={t}
          />
        )
      })}

      {!rows.length ? <InfoRow color={t.color.muted}>No agent histories yet.</InfoRow> : null}
      {Array.from({ length: Math.max(0, visible - visibleRows.length - (rows.length ? 0 : 1)) }, (_, index) => (
        <InfoRow color={t.color.muted} key={`session-pad-${index}`}>
          {' '}
        </InfoRow>
      ))}

      <InfoRow color={t.color.muted}>
        {offset + visible < rows.length ? `↓ ${rows.length - offset - visible} more` : ' '}
      </InfoRow>
      <InfoRow color={t.color.muted}>Tab/←/→ views · ↑/↓ select · Enter open · Esc/q close</InfoRow>
    </ModalShell>
  )
}
