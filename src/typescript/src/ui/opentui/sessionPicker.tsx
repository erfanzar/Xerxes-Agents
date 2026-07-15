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

export interface SessionPickerProps {
  actions: SessionPickerActions
  currentSessionId?: null | string
  onCancel?: () => void
  t?: Theme
}

type SessionRow =
  | { id: 'new'; kind: 'new' }
  | { id: string; item: SessionActiveItem; kind: 'live' }
  | { id: string; item: SessionListItem; kind: 'saved' }

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

function SessionListRow({ index, row, selected, t }: { index: number; row: SessionRow; selected: boolean; t: Theme }) {
  let label: string
  let color = t.color.muted

  if (row.kind === 'new') {
    label = '+  new session'
    color = t.color.label
  } else if (row.kind === 'live') {
    const current = Boolean(row.item.current)
    const title = row.item.title || row.item.preview || '(untitled)'
    const age = relativeAge(row.item.last_active ?? row.item.started_at)
    const identity = current ? 'current' : shortId(row.item.id)
    const ageSuffix = age ? ` · ${age}` : ''
    label = `${STATUS_GLYPH[row.item.status] ?? '·'}  ${identity} · ${row.item.status} · ${shortModel(row.item.model)}${ageSuffix} · ${title}`
    color = current ? t.color.warn : row.item.status === 'working' ? t.color.ok : t.color.text
  } else {
    const title = row.item.title || row.item.preview || '(untitled)'
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
      gw.request<SessionListResponse>('session.list', { limit: 200 })
    ]).then(results => {
      if (!mounted) {
        return
      }

      const liveResult = results[0]
      const savedResult = results[1]
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

      if (savedResult.status === 'fulfilled') {
        const parsed = asRpcResult<SessionListResponse>(savedResult.value)

        if (parsed) {
          nextSaved = parsed.sessions ?? []
        } else {
          errors.push('invalid response: session.list')
        }
      } else {
        errors.push(rpcErrorMessage(savedResult.reason))
      }

      const liveIds = new Set(nextActive.map(session => session.id))
      nextSaved = nextSaved.filter(session => !liveIds.has(session.id))

      setActiveSessions(nextActive)
      setSavedSessions(nextSaved)
      setError(errors.join(' · '))
      setSelected(() => {
        const currentIndex = nextActive.findIndex(
          session => Boolean(session.current) || (!!effectiveSessionId && session.id === effectiveSessionId)
        )

        return currentIndex >= 0 ? currentIndex + 1 : 0
      })
      setLoading(false)
    })

    return () => {
      mounted = false
    }
  }, [effectiveSessionId, gw])

  const rows = useMemo<SessionRow[]>(
    () => [
      { id: 'new', kind: 'new' },
      ...activeSessions.map(item => ({ id: item.id, item, kind: 'live' }) as const),
      ...savedSessions.map(item => ({ id: item.id, item, kind: 'saved' }) as const)
    ],
    [activeSessions, savedSessions]
  )

  useEffect(() => {
    setSelected(index => Math.max(0, Math.min(index, rows.length - 1)))
  }, [rows.length])

  const choose = useCallback(() => {
    const row = rows[selected]

    if (!row) {
      return
    }

    close()

    if (row.kind === 'new') {
      actions.newLiveSession()
    } else if (row.kind === 'live') {
      actions.activateLiveSession(row.item.id)
    } else {
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

      if (name === 'up') {
        consume(event)
        setSelected(index => Math.max(0, index - 1))

        return
      }

      if (name === 'down') {
        consume(event)
        setSelected(index => Math.min(rows.length - 1, index + 1))

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
    [choose, close, loading, rows.length]
  )

  useKeyboard(handleKey)

  const panelWidth = Math.max(
    1,
    Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, width - 6), Math.max(1, width - 2))
  )
  const visible = Math.max(1, Math.min(MAX_VISIBLE, height - 10))
  const sessionVisible = Math.max(0, visible - 1)
  const listSelected = Math.max(0, selected - 1)
  const sessionRows = rows.slice(1)
  const { items: visibleSessionRows, offset } = windowItems(sessionRows, listSelected, sessionVisible)
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
        {activeSessions.length} live · {savedSessions.length} resumable
      </InfoRow>
      {error ? <InfoRow color={t.color.error}>error: {error}</InfoRow> : null}
      <InfoRow color={t.color.muted}>{offset > 0 ? `↑ ${offset} more` : ' '}</InfoRow>

      <SessionListRow index={0} row={rows[0]!} selected={selected === 0} t={t} />
      {visibleSessionRows.map((row, index) => {
        const absoluteIndex = offset + index + 1

        return (
          <SessionListRow
            index={absoluteIndex}
            key={`${row.kind}:${row.id}`}
            row={row}
            selected={selected === absoluteIndex}
            t={t}
          />
        )
      })}

      {Array.from({ length: Math.max(0, sessionVisible - visibleSessionRows.length) }, (_, index) => (
        <InfoRow color={t.color.muted} key={`session-pad-${index}`}>
          {' '}
        </InfoRow>
      ))}

      <InfoRow color={t.color.muted}>
        {offset + sessionVisible < sessionRows.length ? `↓ ${sessionRows.length - offset - sessionVisible} more` : ' '}
      </InfoRow>
      <InfoRow color={t.color.muted}>↑/↓ select · Enter open · Esc/q close</InfoRow>
    </ModalShell>
  )
}
