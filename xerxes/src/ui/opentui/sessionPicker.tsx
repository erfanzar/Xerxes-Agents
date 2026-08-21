// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, usePaste, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppLayoutActions } from '../app/interfaces.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme, patchUiState } from '../app/uiStore.js'
import type {
  BackgroundStartResponse,
  GatewayTranscriptMessage,
  LiveSessionStatus,
  PromptSubmitResponse,
  SessionActiveItem,
  SessionActiveListResponse,
  SessionListItem,
  SessionListResponse,
  SessionPeekResponse,
  SessionSteerResponse
} from '../gatewayTypes.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { compactPreview as compact } from '../lib/text.js'
import type { Theme } from '../theme.js'
import { windowItems } from './overlayLayout.js'

const POLL_MS = 1_500
// The one place a per-row placeholder is genuinely needed — a column of
// blanks is unusable. An em-dash reads as "not named yet" rather than
// pretending to be a name, which is what seven identical "Untitled chat"
// rows did.
const UNTITLED = '—'

export type SessionPickerActions = Pick<AppLayoutActions, 'activateLiveSession' | 'resumeById'>

export interface SessionPickerProps {
  actions: SessionPickerActions
  currentSessionId?: null | string
  onCancel?: () => void
  t?: Theme
}

type SessionGroup = 'needs-input' | 'working' | 'review'
type SessionRow =
  | { group: SessionGroup; id: string; item: SessionActiveItem; kind: 'live' }
  | { group: SessionGroup; id: string; item: SessionListItem; kind: 'saved' }

interface PeekState {
  inflight?: null | { assistant?: string; user?: string }
  messages: GatewayTranscriptMessage[]
  rowId: string
  status: LiveSessionStatus
}

const STATUS_GLYPH: Record<LiveSessionStatus, string> = {
  idle: '✓',
  starting: '…',
  waiting: '?',
  working: '◆'
}

const GROUP_LABEL: Record<SessionGroup, string> = {
  'needs-input': 'NEEDS INPUT',
  review: 'READY',
  working: 'WORKING'
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

const decodePaste = (bytes: Uint8Array): string => new TextDecoder().decode(bytes)

const shortId = (id: string) => (id.length > 10 ? `${id.slice(0, 9)}…` : id)

const shortModel = (model = '') => model.replace(/^.*\//, '') || 'default'

const isMainSession = (item: SessionActiveItem | SessionListItem) => {
  const kind = item.kind?.trim().toLowerCase()

  return kind !== 'subagent' && !item.subagent_id?.trim()
}

// Session content is deliberately excluded. A chat remains visibly untitled
// until the daemon's post-first-exchange model call returns a generated title.
const itemTitle = (item: SessionActiveItem | SessionListItem) => item.title?.trim() || UNTITLED

const relativeAge = (timestamp?: number) => {
  if (!timestamp) return ''

  const elapsed = Math.max(0, Date.now() / 1000 - timestamp)
  const minutes = Math.floor(elapsed / 60)

  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`

  const hours = Math.floor(minutes / 60)

  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`
}

const groupForLive = (status: LiveSessionStatus): SessionGroup => {
  if (status === 'waiting') return 'needs-input'
  if (status === 'starting' || status === 'working') return 'working'

  return 'review'
}

const groupOrder = (group: SessionGroup) => (
  group === 'needs-input' ? 0 : group === 'working' ? 1 : 2
)

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
  firstInGroup,
  maxLabelWidth,
  row,
  selected,
  t
}: {
  firstInGroup: boolean
  maxLabelWidth: number
  row: SessionRow
  selected: boolean
  t: Theme
}) {
  const title = itemTitle(row.item)
  const groupColumn = (firstInGroup ? GROUP_LABEL[row.group] : '').padEnd(11)
  let details: string
  let color = t.color.text

  if (row.kind === 'live') {
    const age = relativeAge(row.item.last_active ?? row.item.started_at)
    const activity = row.item.activity?.trim() || row.item.status
    const current = row.item.current ? ' · attached' : ''
    details = `${STATUS_GLYPH[row.item.status]} ${title} · ${compact(activity, 28)} · ${shortModel(row.item.model)}${age ? ` · ${age}` : ''}${current} · ${shortId(row.item.id)}`
    color = row.item.status === 'waiting'
      ? t.color.warn
      : row.item.status === 'working' || row.item.status === 'starting'
        ? t.color.ok
        : t.color.text
  } else {
    const age = relativeAge(row.item.last_message_at ?? row.item.started_at)
    details = `↻ ${title} · saved · ${row.item.message_count} msgs${age ? ` · ${age}` : ''} · ${shortId(row.item.id)}`
    color = t.color.muted
  }

  const prefix = `${groupColumn} ${selected ? '›' : ' '} `
  const label = prefix + compact(details, Math.max(8, maxLabelWidth - prefix.length))

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
        {label}
      </text>
    </box>
  )
}

const previewLines = (peek: PeekState, width: number, maxRows: number): string[] => {
  const messages = peek.messages
    .filter(message => message.text?.trim() || message.thinking?.trim() || message.error?.trim())
    .map(message => {
      const role = message.role === 'assistant' ? 'assistant' : message.role === 'user' ? 'you' : message.role
      const text = message.text?.trim() || message.thinking?.trim() || message.error?.trim() || ''

      return `${role}: ${compact(text, Math.max(12, width - role.length - 8))}`
    })
  const inflight = peek.inflight?.assistant?.trim()
  if (inflight) messages.push(`assistant: ${compact(inflight, Math.max(12, width - 15))}`)

  return messages.slice(-Math.max(1, maxRows))
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
  const [draft, setDraft] = useState('')
  const draftRef = useRef('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [notice, setNotice] = useState('')
  const [peek, setPeek] = useState<null | PeekState>(null)
  const [selected, setSelected] = useState(0)
  const selectedIdRef = useRef('')
  const [submitting, setSubmitting] = useState(false)
  const mountedRef = useRef(true)
  const peekGenerationRef = useRef(0)
  const pendingPeekRowIdRef = useRef('')

  const invalidatePeek = useCallback(() => {
    peekGenerationRef.current += 1
    pendingPeekRowIdRef.current = ''
  }, [])

  const updateDraft = useCallback((update: string | ((value: string) => string)) => {
    const next = typeof update === 'function' ? update(draftRef.current) : update
    draftRef.current = next
    setDraft(next)
  }, [])

  const close = useCallback(() => {
    invalidatePeek()
    patchOverlayState({ sessions: false })
    onCancel?.()
  }, [invalidatePeek, onCancel])

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      invalidatePeek()
    }
  }, [invalidatePeek])

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      gw.request<SessionActiveListResponse>('session.active_list', {
        current_session_id: effectiveSessionId
      }),
      gw.request<SessionListResponse>('session.list', { kind: 'main', limit: 0 })
    ])
    const errors: string[] = []
    let nextActive: SessionActiveItem[] = []
    let nextSaved: SessionListItem[] = []
    const liveResult = results[0]
    const historyResult = results[1]

    if (liveResult.status === 'fulfilled') {
      const parsed = asRpcResult<SessionActiveListResponse>(liveResult.value)
      if (parsed) nextActive = (parsed.sessions ?? []).filter(isMainSession)
      else errors.push('invalid response: session.active_list')
    } else {
      errors.push(rpcErrorMessage(liveResult.reason))
    }

    if (historyResult.status === 'fulfilled') {
      const parsed = asRpcResult<SessionListResponse>(historyResult.value)
      if (parsed) nextSaved = (parsed.sessions ?? []).filter(isMainSession)
      else errors.push('invalid response: session.list')
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

    if (mountedRef.current) {
      setActiveSessions(nextActive)
      setSavedSessions(nextSaved)
      setError(errors.join(' · '))
      setLoading(false)
    }

    return { active: nextActive, saved: nextSaved }
  }, [effectiveSessionId, gw])

  useEffect(() => {
    let stopped = false
    let inFlight = false
    let initialized = false

    const poll = () => {
      if (stopped || inFlight) return
      inFlight = true
      void refresh()
        .then(({ active, saved }) => {
          if (stopped || initialized) return
          initialized = true
          const all = [
            ...active.map(item => ({ group: groupForLive(item.status), id: item.id, item, kind: 'live' as const })),
            ...saved.map(item => ({ group: 'review' as const, id: item.id, item, kind: 'saved' as const }))
          ].sort((a, b) => groupOrder(a.group) - groupOrder(b.group))
          const current = all.findIndex(row =>
            (row.kind === 'live' && row.item.current) || (!!effectiveSessionId && row.id === effectiveSessionId)
          )
          const initial = current >= 0 ? current : 0
          selectedIdRef.current = all[initial]?.id ?? ''
          setSelected(initial)
        })
        .finally(() => {
          inFlight = false
        })
    }

    poll()
    const timer = setInterval(poll, POLL_MS)

    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [effectiveSessionId, refresh])

  const rows = useMemo<SessionRow[]>(
    () => [
      ...activeSessions.map(item => ({ group: groupForLive(item.status), id: item.id, item, kind: 'live' as const })),
      ...savedSessions.map(item => ({ group: 'review' as const, id: item.id, item, kind: 'saved' as const }))
    ].sort((a, b) => groupOrder(a.group) - groupOrder(b.group)),
    [activeSessions, savedSessions]
  )

  useEffect(() => {
    const selectedStillExists = !selectedIdRef.current || rows.some(row => row.id === selectedIdRef.current)
    const pendingStillExists = !pendingPeekRowIdRef.current || rows.some(row => row.id === pendingPeekRowIdRef.current)
    const peekStillExists = !peek || rows.some(row => row.id === peek.rowId)
    if (!selectedStillExists || !pendingStillExists || !peekStillExists) invalidatePeek()

    setSelected(index => {
      const preserved = rows.findIndex(row => row.id === selectedIdRef.current)
      const next = preserved >= 0 ? preserved : Math.max(0, Math.min(index, Math.max(0, rows.length - 1)))
      selectedIdRef.current = rows[next]?.id ?? ''

      return next
    })
    if (!peekStillExists) setPeek(null)
  }, [invalidatePeek, peek, rows])

  const select = useCallback((next: number | ((index: number) => number)) => {
    setSelected(index => {
      const candidate = typeof next === 'function' ? next(index) : next
      const clamped = Math.max(0, Math.min(candidate, Math.max(0, rows.length - 1)))
      const nextId = rows[clamped]?.id ?? ''
      if (nextId !== selectedIdRef.current) {
        invalidatePeek()
        setNotice('')
      }
      selectedIdRef.current = nextId

      return clamped
    })
  }, [invalidatePeek, rows])

  const attach = useCallback(() => {
    const row = rows[selected]
    if (!row) return

    close()
    if (row.kind === 'live') actions.activateLiveSession(row.item.id)
    else actions.resumeById(row.item.id)
  }, [actions, close, rows, selected])

  const openPeek = useCallback(() => {
    const row = rows[selected]
    if (!row) return
    const generation = peekGenerationRef.current + 1
    peekGenerationRef.current = generation
    pendingPeekRowIdRef.current = ''
    if (row.kind === 'saved') {
      setNotice('Attach to a saved chat to inspect or reply.')

      return
    }

    pendingPeekRowIdRef.current = row.item.id
    setNotice('loading preview…')
    void gw.request<SessionPeekResponse>('session.peek', { session_id: row.item.id })
      .then(raw => {
        if (!mountedRef.current || generation !== peekGenerationRef.current) return
        const result = asRpcResult<SessionPeekResponse>(raw)
        if (!result?.messages || !result.status) throw new Error('invalid response: session.peek')
        pendingPeekRowIdRef.current = ''
        setPeek({
          inflight: result.inflight,
          messages: result.messages,
          rowId: row.item.id,
          status: result.status
        })
        setNotice('')
      })
      .catch(cause => {
        if (mountedRef.current && generation === peekGenerationRef.current) {
          pendingPeekRowIdRef.current = ''
          setNotice(`preview failed: ${rpcErrorMessage(cause)}`)
        }
      })
  }, [gw, rows, selected])

  const submitDraft = useCallback(() => {
    const text = draftRef.current.trim()
    if (!text || submitting) return

    setSubmitting(true)
    setError('')

    if (peek) {
      if (peek.status === 'waiting') {
        setNotice('This chat needs structured input. Attach to answer it.')
        setSubmitting(false)

        return
      }

      const method = peek.status === 'working' || peek.status === 'starting' ? 'session.steer' : 'prompt.submit'
      void gw.request<PromptSubmitResponse | SessionSteerResponse>(method, {
        session_id: peek.rowId,
        text
      })
        .then(raw => {
          const result = asRpcResult<PromptSubmitResponse | SessionSteerResponse>(raw)
          if (result?.ok === false || (result && 'status' in result && result.status === 'rejected')) {
            throw new Error('the running chat rejected that instruction')
          }
          updateDraft('')
          setNotice(method === 'session.steer' ? 'instruction sent to running chat' : 'reply sent')
          void refresh()
        })
        .catch(cause => setError(rpcErrorMessage(cause)))
        .finally(() => setSubmitting(false))

      return
    }

    void gw.request<BackgroundStartResponse>('prompt.background', {
      session_id: effectiveSessionId,
      text
    })
      .then(raw => {
        const result = asRpcResult<BackgroundStartResponse>(raw)
        if (!result?.task_id) throw new Error('invalid response: prompt.background')
        patchUiState(state => ({ ...state, bgTasks: new Set(state.bgTasks).add(result.task_id!) }))
        updateDraft('')
        setNotice(`dispatched ${shortId(result.task_id)}`)
        void refresh()
      })
      .catch(cause => setError(rpcErrorMessage(cause)))
      .finally(() => setSubmitting(false))
  }, [effectiveSessionId, gw, peek, refresh, submitting, updateDraft])

  const handleKey = useCallback((event: KeyEvent) => {
    const name = event.name.toLowerCase()
    const sequence = event.sequence ?? ''

    if (name === 'escape') {
      consume(event)
      if (peek) {
        invalidatePeek()
        setPeek(null)
        updateDraft('')
        setNotice('')
      } else {
        close()
      }

      return
    }

    if (name === 'backspace') {
      consume(event)
      updateDraft(value => Array.from(value).slice(0, -1).join(''))

      return
    }

    if (event.ctrl && name === 'u') {
      consume(event)
      updateDraft('')

      return
    }

    if (!draftRef.current && !peek && name === 'up') {
      consume(event)
      select(index => index - 1)

      return
    }

    if (!draftRef.current && !peek && name === 'down') {
      consume(event)
      select(index => index + 1)

      return
    }

    if (!draftRef.current && !peek && name === 'home') {
      consume(event)
      select(0)

      return
    }

    if (!draftRef.current && !peek && name === 'end') {
      consume(event)
      select(rows.length - 1)

      return
    }

    if (!draftRef.current && !peek && (name === 'space' || sequence === ' ')) {
      consume(event)
      openPeek()

      return
    }

    if (!draftRef.current && !peek && name === 'right') {
      consume(event)
      attach()

      return
    }

    if (name === 'return' || name === 'enter' || name === 'kpenter') {
      consume(event)
      if (draftRef.current.trim()) submitDraft()
      else if (!peek) attach()

      return
    }

    if (!event.ctrl && !event.meta && !event.super && sequence && !/[\x00-\x1f\x7f]/u.test(sequence)) {
      consume(event)
      updateDraft(value => value + sequence)
    }
  }, [attach, close, invalidatePeek, openPeek, peek, rows.length, select, submitDraft, updateDraft])

  useKeyboard(handleKey)
  usePaste(event => {
    event.preventDefault()
    event.stopPropagation()
    updateDraft(value => value + decodePaste(event.bytes))
  })

  const chromeRows = error || notice ? 7 : 6
  // Budget, not quota: only real rows are rendered and a flex spacer absorbs
  // the remainder, so a three-chat list is three rows rather than three rows
  // padded out with twenty-nine blanks.
  const listRows = Math.max(1, height - chromeRows)
  const { items: visibleRows, offset } = windowItems(rows, selected, Math.min(listRows, rows.length))
  const hiddenAbove = offset
  const hiddenBelow = Math.max(0, rows.length - offset - visibleRows.length)
  const peekRows = listRows
  const peekText = peek ? previewLines(peek, width, peekRows) : []
  const peekRow = peek ? rows.find(row => row.id === peek.rowId) : undefined

  return (
    <box
      backgroundColor={t.color.statusBg}
      flexDirection="column"
      height={height}
      left={0}
      position="absolute"
      top={0}
      width={width}
      zIndex={200}
    >
      <box flexDirection="row" flexShrink={0} height={1} justifyContent="space-between" paddingLeft={2} paddingRight={2}>
        <text fg={t.color.accent} flexShrink={0}><b>Agent view</b></text>
        <text fg={t.color.muted} flexShrink={0}>{activeSessions.length} live · {rows.length} chats</text>
      </box>
      <InfoRow color={t.color.muted}>
        {peek && peekRow
          ? `${itemTitle(peekRow.item)} · ${peek.status} · preview`
          : 'Detached sessions keep working. Subagents stay inside their parent chat.'}
      </InfoRow>
      {error ? <InfoRow color={t.color.error}>error: {error}</InfoRow> : notice ? <InfoRow color={t.color.warn}>{notice}</InfoRow> : null}

      {loading ? (
        <InfoRow color={t.color.muted}>loading live and saved chats…</InfoRow>
      ) : peek ? (
        <>
          {peekText.map((line, index) => <InfoRow color={t.color.text} key={`peek-${index}`}>{line}</InfoRow>)}
        </>
      ) : rows.length ? (
        <>
          {hiddenAbove > 0 ? <InfoRow color={t.color.muted}>{`  ↑ ${hiddenAbove} more`}</InfoRow> : null}
          {visibleRows.map((row, index) => {
            const absoluteIndex = offset + index
            const previous = rows[absoluteIndex - 1]

            return (
              <SessionListRow
                firstInGroup={!previous || previous.group !== row.group}
                key={`${row.kind}:${row.id}`}
                maxLabelWidth={Math.max(12, width - 4)}
                row={row}
                selected={selected === absoluteIndex}
                t={t}
              />
            )
          })}
          {hiddenBelow > 0 ? <InfoRow color={t.color.muted}>{`  ↓ ${hiddenBelow} more`}</InfoRow> : null}
        </>
      ) : (
        <>
          <InfoRow color={t.color.muted}>No chats yet. Type below to dispatch one.</InfoRow>
        </>
      )}

      {/* Directly below the list, not pinned to the terminal's last row. The
          old layout padded the gap with blank rows and left ~32 dead rows
          between a four-chat list and its input; a flex spacer reproduced
          exactly the same look. Stacking is what actually fixes it. */}
      <box borderColor={t.color.border} borderStyle="single" flexShrink={0} height={3} marginTop={1} paddingLeft={1} paddingRight={1}>
        <text fg={draft ? t.color.text : t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
          {draft || (peek ? 'Reply or steer this chat…' : 'Dispatch a new independent chat…')}{submitting ? ' …' : ''}
        </text>
      </box>
      <InfoRow color={t.color.muted}>
        {peek
          ? 'type + Enter reply/steer · Esc back'
          : 'type + Enter dispatch · ↑/↓ select · Space peek · →/Enter attach · Esc exit'}
      </InfoRow>
    </box>
  )
}
