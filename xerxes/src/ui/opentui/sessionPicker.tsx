// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { createTextAttributes } from '@opentui/core'
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
import { GLYPH, leaderRun, type NocturneState, stateSkin } from '../domain/nocturne.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { compactPreview as compact } from '../lib/text.js'
import type { Theme } from '../theme.js'
import { GroupCaption } from './nocturne.js'
import { windowItems } from './overlayLayout.js'

const POLL_MS = 1_500
// The one place a per-row placeholder is genuinely needed — a column of
// blanks is unusable. An em-dash reads as "not named yet" rather than
// pretending to be a name, which is what seven identical "Untitled chat"
// rows did.
// Deliberately an em dash, not a word: a column of identical "untitled"
// strings reads as a NAME and makes the list unnavigable, while `—` reads as
// "not named yet". Changing this breaks sessionPicker.test.ts on purpose.
const UNTITLED = '—'

export type SessionPickerActions = Pick<AppLayoutActions, 'activateLiveSession' | 'resumeById'>

export interface SessionPickerProps {
  actions: SessionPickerActions
  currentSessionId?: null | string
  onCancel?: () => void
  t?: Theme
}

/**
 * A saved chat is not an agent.
 *
 * Every saved session used to be forced into `review`, so a machine with 37
 * chats on disk opened this screen to `READY TO REVIEW · 37` — borrowing the
 * agents vocabulary for history that nobody is reviewing and nothing is
 * waiting on. `saved` is its own group and sorts last, behind everything that
 * is actually live.
 */
type SessionGroup = 'needs-input' | 'working' | 'review' | 'saved'
type SessionRow =
  | { group: SessionGroup; id: string; item: SessionActiveItem; kind: 'live' }
  | { group: SessionGroup; id: string; item: SessionListItem; kind: 'saved' }

interface PeekState {
  inflight?: null | { assistant?: string; user?: string }
  messages: GatewayTranscriptMessage[]
  rowId: string
  status: LiveSessionStatus
}

const GROUP_LABEL: Record<SessionGroup, string> = {
  'needs-input': 'NEEDS INPUT',
  review: 'READY TO REVIEW',
  saved: 'SAVED CHATS',
  working: 'WORKING'
}

/** The Nocturne state each group wears; the colour follows from there. */
const GROUP_STATE: Record<SessionGroup, NocturneState> = {
  'needs-input': 'needsInput',
  review: 'done',
  saved: 'working',
  working: 'working'
}

/** Voice color for a live row's dot and title, shared with the F6 cards. */
const statusVoice = (
  status: LiveSessionStatus,
  t: Theme
): { budget: string; voice: string } => {
  if (status === 'waiting') return { budget: 'needs you', voice: t.color.warn }

  if (status === 'starting' || status === 'working') {
    return { budget: 'working', voice: t.color.accent }
  }

  return { budget: 'idle', voice: t.color.ok }
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

/**
 * Mockup 04: needs-input cards get a thicker accent edge so the "unblock me"
 * group is scannable before anything is selected. This is the same left-edge
 * strip the agent-rail cards paint, in the mode accent (lapis after the v3
 * pivot) rather than the mockup's gold; other groups stay on the selection bg.
 */
export const sessionRowAccent = (group: SessionGroup, t: Theme): string | undefined =>
  group === 'needs-input' ? t.color.accent : undefined

const BOLD = createTextAttributes({ bold: true })

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

const groupOrder = (group: SessionGroup) =>
  group === 'needs-input' ? 0 : group === 'working' ? 1 : group === 'review' ? 2 : 3

function InfoRow({ children, color }: { children: ReactNode; color: string }) {
  return (
    <box flexShrink={0} height={1} paddingLeft={2} paddingRight={2}>
      <text fg={color} flexShrink={0} truncate width="100%" wrapMode="none">
        {children}
      </text>
    </box>
  )
}

/**
 * The inspector — screen 03's right pane.
 *
 * It answers in the order you ask: what is this, what is it doing, what has
 * it said. The list and the inspector are on screen together on purpose: a
 * view that REPLACES the list with a detail makes you remember which row you
 * were on and back out to check, which is exactly what the peek view used to
 * do here.
 */
function SessionInspector({
  peekRow,
  peekText,
  peeking,
  row,
  status,
  t,
  width
}: {
  peekRow?: SessionRow | undefined
  peekText: string[]
  peeking: boolean
  row?: SessionRow | undefined
  status?: LiveSessionStatus | undefined
  t: Theme
  width: number
}) {
  const shown = peeking ? (peekRow ?? row) : row

  if (!shown) {
    return (
      <box flexDirection="column" flexShrink={0} paddingLeft={2} paddingRight={2} width={width}>
        <text fg={t.ds.caption} flexShrink={0} truncate width="100%" wrapMode="none">
          nothing selected
        </text>
      </box>
    )
  }

  const live = shown.kind === 'live' ? shown.item : null
  const saved = shown.kind === 'saved' ? shown.item : null
  const skin = stateSkin(GROUP_STATE[shown.group], t.ds)
  // Only facts the row actually carries — the canvas is explicit that a
  // custom endpoint gets no badges rather than invented ones.
  const chips = [
    live ? shortModel(live.model) : '',
    saved ? `${saved.message_count} msgs` : '',
    live?.current ? 'attached' : '',
    shortId(shown.item.id)
  ].filter(Boolean)

  return (
    <box
      backgroundColor={t.ds.sunken}
      flexDirection="column"
      flexShrink={0}
      paddingLeft={2}
      paddingRight={2}
      width={width}
    >
      <text flexShrink={0} truncate width="100%" wrapMode="none">
        <span fg={skin.dot}>{`${GLYPH.state} `}</span>
        <span attributes={BOLD} fg={t.ds.title}>
          {compact(itemTitle(shown.item), Math.max(8, width - 6))}
        </span>
      </text>
      <text fg={t.ds.caption} flexShrink={0} truncate width="100%" wrapMode="none">
        {GROUP_LABEL[shown.group]}
      </text>
      {chips.length ? (
        <text fg={t.ds.meta} flexShrink={0} truncate width="100%" wrapMode="none">
          {chips.join(` ${GLYPH.separator} `)}
        </text>
      ) : null}
      <box flexShrink={0} height={1} />
      <GroupCaption label={peeking ? 'TRANSCRIPT' : 'LATEST'} t={t} width={Math.max(8, width - 4)} />
      {peeking && peekText.length ? (
        peekText.map((line, index) => (
          <text fg={t.ds.prose} flexShrink={0} key={`peek-${index}`} truncate width="100%" wrapMode="none">
            {line}
          </text>
        ))
      ) : (
        <text fg={t.ds.meta} flexShrink={0} truncate width="100%" wrapMode="none">
          {live?.activity?.trim() || (status ?? '') || 'press Space to read this chat'}
        </text>
      )}
    </box>
  )
}

/** Exported for the assembled-screen test; `SessionPicker` is its only user. */
export function SessionListRow({
  compactMode,
  counts,
  firstInGroup,
  maxLabelWidth,
  row,
  selected,
  t
}: {
  compactMode?: boolean
  counts: Record<SessionGroup, number>
  firstInGroup: boolean
  maxLabelWidth: number
  row: SessionRow
  selected: boolean
  t: Theme
}) {
  const title = itemTitle(row.item)
  const selectionBg = selected ? t.color.completionCurrentBg : undefined

  // Tiny panes get one line per chat — no captions, no second line — so the
  // manager stays usable instead of clipping its own cards.
  if (compactMode) {
    const voice = row.kind === 'live' ? statusVoice(row.item.status, t) : null
    const age = relativeAge(
      row.kind === 'live' ? row.item.last_active ?? row.item.started_at : row.item.last_message_at
    )
    const label = compact(
      `${voice ? '● ' : '↻ '}${title}${age ? ` · ${age}` : ''}`,
      Math.max(8, maxLabelWidth - 6)
    )

    return (
      <box backgroundColor={selectionBg} flexShrink={0} height={1} paddingLeft={2} paddingRight={2} width="100%">
        <text flexShrink={0} truncate width="100%" wrapMode="none">
          <span fg={voice ? voice.voice : t.color.muted}>{label.slice(0, 2)}</span>
          <span attributes={BOLD} fg={t.color.text}>
            {compact(title, Math.max(8, maxLabelWidth - 10))}
          </span>
          {age ? <span fg={t.color.muted}>{` · ${age}`}</span> : null}
        </text>
      </box>
    )
  }

  // Group captions are real rows — `NEEDS INPUT · 2`, then a rule to the edge
  // — so the action order reads at a glance and the count lands before you
  // open the group. Same component every other grouped list uses.
  const caption = firstInGroup ? (
    <box flexShrink={0} height={1} paddingLeft={2} paddingRight={2} width="100%">
      <GroupCaption
        count={counts[row.group]}
        label={GROUP_LABEL[row.group]}
        t={t}
        tone={stateSkin(GROUP_STATE[row.group], t.ds).dot}
        width={maxLabelWidth}
      />
    </box>
  ) : null

  if (row.kind === 'saved') {
    const age = relativeAge(row.item.last_message_at ?? row.item.started_at)
    // The group caption already says these are saved chats, so the row does
    // not repeat the word. What it owes you is the size and the age, and
    // those hang RIGHT on a dotted leader so thirty-seven of them stack into
    // a column you can read vertically instead of a ragged left-packed edge.
    const right = `${row.item.message_count} msgs${age ? ` · ${age}` : ''}`
    const name = compact(title, Math.max(8, maxLabelWidth - right.length - 8))
    const dots = leaderRun(maxLabelWidth, name.length + 3, right.length)

    return (
      <>
        {caption}
        <box backgroundColor={selectionBg} flexDirection="row" flexShrink={0} width="100%">
          {/* Shared edge column: the needs-input accent strip paints here
              without shifting any other row's text column by one cell. */}
          <box flexShrink={0} width={1} />
          <box flexShrink={1} height={1} minWidth={0} overflow="hidden">
            <text flexShrink={0} truncate width="100%" wrapMode="none">
              <span fg={t.ds.separator}>{`${GLYPH.collapsed} `}</span>
              <span fg={t.ds.secondary}>{name}</span>
              <span fg={t.ds.leaderQuiet}>{dots}</span>
              <span fg={t.ds.numeric}>{` ${right}`}</span>
            </text>
          </box>
        </box>
      </>
    )
  }

  const age = relativeAge(row.item.last_active ?? row.item.started_at)
  const voice = statusVoice(row.item.status, t)
  const activity = row.item.activity?.trim()
  const working = row.item.status === 'working' || row.item.status === 'starting'
  const budget = `${voice.budget}${age ? ` · ${age}` : ''}`
  const nameMax = Math.max(8, maxLabelWidth - budget.length - 6)
  const edgeColor = sessionRowAccent(row.group, t)

  return (
    <>
      {caption}
      <box backgroundColor={selectionBg} flexDirection="row" flexShrink={0} width="100%">
        {/* Mockup 04: needs-input cards get a thicker accent edge so the
            "unblock me" group reads before anything is selected — the same
            left strip the agent-rail cards paint. */}
        <box backgroundColor={edgeColor} flexShrink={0} width={1} />
        <box flexDirection="column" flexGrow={1} flexShrink={1} minWidth={0}>
        {/* Line 1: voice dot, bold title, attached marker — state hangs right. */}
        <box flexDirection="row" flexShrink={0} height={1} paddingLeft={1} paddingRight={2}>
          <box flexGrow={1} flexShrink={1} overflow="hidden">
            <text flexShrink={0} truncate width="100%" wrapMode="none">
              <span fg={voice.voice}>{'● '}</span>
              <span attributes={BOLD} fg={t.color.text}>
                {compact(title, nameMax)}
              </span>
              {row.item.current ? <span fg={t.color.muted}>{'  · attached'}</span> : null}
            </text>
          </box>
          <box flexShrink={0}>
            <text fg={working ? t.color.accent : t.color.muted} flexShrink={0}>
              {budget}
            </text>
          </box>
        </box>
        {/* Line 2: the one substance line — violet while it works. */}
        <box flexShrink={0} height={1} paddingLeft={3} paddingRight={2}>
          <text fg={working ? t.color.thinking : t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
            {activity
              ? `${working ? '└ ' : ''}${compact(activity, Math.max(8, maxLabelWidth - 6))}`
              : `${shortModel(row.item.model)}${row.item.id ? ` · ${shortId(row.item.id)}` : ''}`}
          </text>
        </box>
        </box>
      </box>
    </>
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
  // Closing the overlay is terminal for this component instance. Keeping that
  // fact outside React state prevents a queued Space/open event or deferred
  // session.peek response from repainting the manager after Esc has restored
  // the transcript, even when keyboard events and promises settle together.
  const closedRef = useRef(false)
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
    closedRef.current = true
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
            ...saved.map(item => ({ group: 'saved' as const, id: item.id, item, kind: 'saved' as const }))
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
      ...savedSessions.map(item => ({ group: 'saved' as const, id: item.id, item, kind: 'saved' as const }))
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
    if (closedRef.current) return
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
        if (closedRef.current || !mountedRef.current || generation !== peekGenerationRef.current) return
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

    // Mockup 04 composer meta: "n new chat". While a preview is open, typed
    // text replies into THAT chat — n leaves it and refocuses the dispatch
    // well for a fresh detached chat. With no preview open the composer
    // already owns every keystroke, so n is left alone and simply begins
    // the new chat's first word rather than being swallowed; the
    // empty-draft guard keeps the binding from ever firing mid-message.
    if (!draftRef.current && peek && !event.ctrl && !event.meta && !event.super && name === 'n') {
      consume(event)
      invalidatePeek()
      setPeek(null)
      updateDraft('')
      setNotice('')

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

  // The header hairline is chrome, and chrome yields before content: on a
  // short terminal it is not drawn at all, because one rule is not worth a
  // row of the list it is ruling off. Counted here so the budget and the
  // paint cannot disagree.
  const showHeaderRule = height >= 20
  const chromeRows = (error || notice ? 8 : 7) + (showHeaderRule ? 1 : 0)
  // Budget in RENDERED LINES, not entries: live rows are two lines tall and
  // group captions add one more, so a naive per-entry window overflowed the
  // pane and pushed the dispatch composer off-screen. Below four available
  // lines the rows collapse to a one-line compact mode instead.
  const listRows = Math.max(1, height - chromeRows)
  const compactMode = listRows < 4
  const anchored = windowItems(rows, selected, Math.min(listRows, rows.length))
  const rowLines = (row: SessionRow, isFirst: boolean): number => {
    if (compactMode) return 1
    const base = row.kind === 'live' ? 2 : 1

    return isFirst ? base + 1 : base
  }
  let usedLines = 0
  const visibleRows: SessionRow[] = []
  anchored.items.forEach((row, index) => {
    const absoluteIndex = anchored.offset + index
    const previous = rows[absoluteIndex - 1]
    const isFirst = !previous || previous.group !== row.group
    const cost = rowLines(row, isFirst)

    // Always admit the anchored row so the selection is never empty.
    if (usedLines + cost <= listRows || visibleRows.length === 0) {
      visibleRows.push(row)
      usedLines += cost
    }
  })
  const offset = anchored.offset
  const hiddenAbove = offset
  const hiddenBelow = Math.max(0, rows.length - offset - visibleRows.length)
  // The inspector is a side panel, and screen 09 makes side panels the FIRST
  // thing a narrowing terminal gives up — the list is what you came for.
  // The threshold is above `densityFor`'s generic 100 on purpose: this pane
  // needs ~40 columns of its own, and taking those out of a 100-column
  // terminal leaves a list too narrow to read the titles it exists to show.
  const showInspector = width >= 120 && rows.length > 0
  const inspectorWidth = showInspector ? Math.max(32, Math.min(52, Math.floor(width * 0.36))) : 0
  const listWidth = Math.max(20, width - inspectorWidth - 1)
  const inspectRow = rows[selected]
  const peekRows = listRows
  const peekText = peek ? previewLines(peek, inspectorWidth || width, peekRows) : []
  const peekRow = peek ? rows.find(row => row.id === peek.rowId) : undefined
  // Caption counts per action group, computed once for the whole list.
  const groupCounts: Record<SessionGroup, number> = { 'needs-input': 0, review: 0, saved: 0, working: 0 }
  for (const row of rows) {
    groupCounts[row.group] += 1
  }

  const workingCount = activeSessions.filter(
    item => item.status === 'working' || item.status === 'starting'
  ).length

  // Mockup 04 header: the attached main chat's generated title anchors the
  // right edge ("main session · refactor-auth"), muted with a lapis title.
  // Only data the picker already polls is used, and an unnamed chat stays
  // quiet rather than printing its not-named-yet placeholder.
  const currentActive =
    activeSessions.find(item => item.current) ??
    activeSessions.find(item => !!effectiveSessionId && item.id === effectiveSessionId)
  const mainSessionTitle = currentActive?.title?.trim() ?? ''

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
      <box flexDirection="row" flexShrink={0} height={1} paddingLeft={2} paddingRight={2}>
        <box flexShrink={0}>
          <text flexShrink={0}>
            <span fg={t.color.accent}>✦ </span>
            <b>Agent View</b>
          </text>
        </box>
        {/* Counts yield before the title does: their own clipped box means a
            40-column terminal still reads the brand + name intact. */}
        <box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
          <text fg={t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
            {`  ${activeSessions.length} chats · ${workingCount} working`}
          </text>
        </box>
        {mainSessionTitle ? (
          <>
            <box flexShrink={0}>
              <text flexShrink={0}>{'main session · '}</text>
            </box>
            {/* Single-span truncate inside its own clipped box: span-heavy
                truncation blanks whole runs at narrow widths, so only the
                title itself shrinks — the muted label never clips mid-glyph. */}
            <box flexShrink={1} minWidth={0} overflow="hidden">
              <text fg={t.color.accent} flexShrink={0} truncate width="100%" wrapMode="none">
                {mainSessionTitle}
              </text>
            </box>
          </>
        ) : null}
        <box flexShrink={0}>
          <text fg={workingCount ? t.color.accent : t.color.muted} flexShrink={0}>
            {workingCount ? 'live' : 'idle'}
          </text>
        </box>
      </box>
      <InfoRow color={t.color.muted}>
        {peek && peekRow
          ? `${itemTitle(peekRow.item)} · ${peek.status} · preview`
          : 'Detached sessions keep working. Subagents stay inside their parent chat.'}
      </InfoRow>
      {/* A hairline under the header, as the canvas rules its panel head off
          from the content below it. */}
      {showHeaderRule ? (
        // A run of `─`, not a bordered box: a box with `height={1}` and a
        // border draws all four edges, so the "hairline" came out as
        // `┌────────┐`.
        <box flexShrink={0} height={1} paddingLeft={2} paddingRight={2}>
          <text fg={t.ds.divider} flexShrink={0} truncate width="100%" wrapMode="none">
            {'─'.repeat(Math.max(0, width - 4))}
          </text>
        </box>
      ) : null}
      {error ? <InfoRow color={t.color.error}>error: {error}</InfoRow> : notice ? <InfoRow color={t.color.warn}>{notice}</InfoRow> : null}

      {/* Screen 03's body: a grouped list on the left and an inspector on the
          right, both visible at once, filling everything between the header
          and the composer. The panel used to stack its children at their own
          heights and leave the rest of the terminal empty — a four-chat list
          with thirty dead rows under it. */}
      <box flexDirection="row" flexGrow={1} minHeight={0} width="100%">
        <box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} minWidth={0} overflow="hidden">
          {loading ? (
            <InfoRow color={t.color.muted}>loading live and saved chats…</InfoRow>
          ) : peek && !showInspector ? (
            /* Too narrow for a side panel, so the peek takes the list's place
               — screen 09's order of sacrifice gives up side panels first,
               but never the thing you actually asked to read. */
            peekText.map((line, index) => (
              <InfoRow color={t.color.text} key={`peek-${index}`}>
                {line}
              </InfoRow>
            ))
          ) : rows.length ? (
            <box flexDirection="column" flexShrink={1} minHeight={0} overflow="hidden">
              {!compactMode && hiddenAbove > 0 ? (
                <InfoRow color={t.color.muted}>{`  ↑ ${hiddenAbove} more`}</InfoRow>
              ) : null}
              {visibleRows.map((row, index) => {
                const absoluteIndex = offset + index
                const previous = rows[absoluteIndex - 1]

                return (
                  <SessionListRow
                    compactMode={compactMode}
                    counts={groupCounts}
                    firstInGroup={!previous || previous.group !== row.group}
                    key={`${row.kind}:${row.id}`}
                    maxLabelWidth={Math.max(12, listWidth - 4)}
                    row={row}
                    selected={selected === absoluteIndex}
                    t={t}
                  />
                )
              })}
              {!compactMode && hiddenBelow > 0 ? (
                <InfoRow color={t.color.muted}>{`  ↓ ${hiddenBelow} more`}</InfoRow>
              ) : null}
            </box>
          ) : (
            <InfoRow color={t.color.muted}>No chats yet. Type below to dispatch one.</InfoRow>
          )}
        </box>
        {showInspector ? (
          <>
            {/* A filled column, not a per-side border: OpenTUI paints an edge
                THROUGH the text when a bordered child sits inside a framed
                parent, which is how the composer's identity row once came out
                as `╰─◆─code─mode─·─…─╯`. */}
            <box backgroundColor={t.ds.hairline} flexShrink={0} width={1} />
            <SessionInspector
              peekRow={peekRow}
              peekText={peekText}
              peeking={Boolean(peek)}
              row={inspectRow}
              status={peek?.status}
              t={t}
              width={inspectorWidth}
            />
          </>
        ) : null}
      </box>

      {/* Directly below the list, not pinned to the terminal's last row. The
          old layout padded the gap with blank rows and left ~32 dead rows
          between a four-chat list and its input; a flex spacer reproduced
          exactly the same look. Stacking is what actually fixes it. */}
      <box borderColor={t.color.border} borderStyle="rounded" flexShrink={0} height={3} marginTop={1} paddingLeft={1} paddingRight={1}>
        <text flexShrink={0} truncate width="100%" wrapMode="none">
          <span fg={draft ? t.color.text : t.color.accent}>❯ </span>
          <span fg={draft ? t.color.text : t.color.muted}>
            {/* Multi-span truncate middle-ellipsizes instead of clipping the
                tail, so narrow terminals get a shorter placeholder rather
                than a mangled one. */}
            {draft ||
              (peek
                ? width < 48
                  ? 'Reply or steer…'
                  : 'Reply or steer this chat…'
                : width < 48
                  ? 'Dispatch a new chat…'
                  : 'Dispatch a new independent chat…')}
            {submitting ? ' …' : ''}
          </span>
        </text>
      </box>
      <InfoRow color={t.color.muted}>
        {peek
          ? 'type + Enter reply/steer · Esc back'
          : 'type + Enter dispatch · ↑/↓ select · Space peek · n new chat · →/Enter attach · Esc exit'}
      </InfoRow>
    </box>
  )
}
