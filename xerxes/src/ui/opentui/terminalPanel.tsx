// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useStore } from '@nanostores/react'
import { useKeyboard, usePaste, useTerminalDimensions } from '@opentui/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useOptionalGateway } from '../app/gatewayContext.js'
import {
  $panelWidthDelta,
  adjustPanelWidth,
  PANEL_WIDTH_STEP,
  withPanelWidthDelta
} from '../app/panelSizeStore.js'

import {
  orderTerminals,
  TERMINAL_GROUP_LABEL,
  TERMINAL_GROUP_STATE,
  terminalGroup,
  terminalHeading
} from '../lib/terminalGroups.js'
import { stateSkin } from '../domain/nocturne.js'

import { OVERLAY_PANEL_SPECS, overlayPanelSize } from './overlayLayout.js'
import {
  controlTerminal,
  inspectTerminal,
  listTerminals,
  terminalAge,
  terminalState,
  type TerminalInspection,
  type TerminalSummary
} from '../lib/terminals.js'
import type { Theme } from '../theme.js'

import { isPanelResizeKey } from './diffPanel.js'
import { isPageDownKey, isPageUpKey } from '../lib/pageKeys.js'

import { GroupCaption } from './nocturne.js'
import { Box, Span, Text } from './primitives.js'

/**
 * F8 terminal viewer: every shell Xerxes is driving — background commands,
 * foreground `exec_command` runs, and persistent PTY sessions — with a live
 * output tail per terminal.
 *
 * The list/detail split mirrors the agents panel on purpose: Enter drills in,
 * Esc comes back out, Esc again closes. Watching is read-only by construction;
 * the daemon mirrors output rather than draining the buffer the model reads,
 * so having this open cannot change what the agent sees.
 *
 * Destructive keys are two-step per mockup 06: the first `k`/`K` arms the
 * kill with an inline confirm on that row, the repeat press executes it
 * (`k` → SIGTERM, `K` → SIGKILL for the rare force case), and Esc or moving
 * the selection steps back cleanly.
 */

type KeyEvent = Parameters<Parameters<typeof useKeyboard>[0]>[0]

/** Poll cadences. Fast enough to read as live, slow enough to stay cheap. */
const LIST_POLL_MS = 1_200
const OUTPUT_POLL_MS = 700

/**
 * Mockup 06: destructive keys are two-step. The first press arms the kill and
 * shows the inline confirm on that row; the repeat inside this window
 * executes it. The window keeps a stray arm from lurking invisibly behind
 * scrolling output — forget it and it simply expires.
 */
const KILL_ARM_WINDOW_MS = 6_000

/** An armed destructive action waiting for its confirming repeat press. */
interface KillArm {
  at: number
  force: boolean
  terminalId: string
}

/** The arm still valid for `terminalId` at instant `at`, or null once stale. */
const activeKillArm = (
  arm: null | KillArm,
  terminalId: string | undefined,
  at: number
): null | KillArm =>
  arm !== null && arm.terminalId === terminalId && at - arm.at <= KILL_ARM_WINDOW_MS ? arm : null

/** The shared wording for the row hint, the detail hint and the notice. */
const killPrompt = (label: string, force: boolean): string =>
  `${force ? 'force kill' : 'kill'} ${label}? ${force ? 'K' : 'k'} again to confirm · esc cancel`

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

/** Addressable row id, so a selection moved below the fold can be scrolled to. */
const terminalRowId = (terminalId: string): string => `terminal-row:${terminalId}`

export function TerminalPanelHotkey({
  disabled,
  open,
  onToggle
}: {
  disabled: boolean
  open: boolean
  onToggle: (open: boolean) => void
}) {
  useKeyboard(event => {
    if (disabled || event.name !== 'f8') return
    consumeKey(event)
    onToggle(!open)
  })

  return null
}

const KIND_LABEL: Record<TerminalSummary['kind'], string> = {
  background: 'bg',
  foreground: 'run',
  pty: 'pty'
}

function stateColor(entry: TerminalSummary, t: Theme): string {
  const state = terminalState(entry)
  if (state === 'running') return t.color.accent
  return state === 'failed' ? t.color.error : t.color.ok
}

function stateGlyph(entry: TerminalSummary): string {
  const state = terminalState(entry)
  return state === 'running' ? '●' : state === 'failed' ? '✗' : '✓'
}

/** "1.2k" / "3.4M" so a chatty build does not push the row's columns around. */
function fmtChars(count: number): string {
  if (count < 1_000) return `${count}`
  if (count < 1_000_000) return `${(count / 1_000).toFixed(count < 10_000 ? 1 : 0)}k`
  return `${(count / 1_000_000).toFixed(1)}M`
}

function exitLabel(entry: TerminalSummary): string {
  if (entry.running) return 'running'
  return entry.exitCode === null ? 'ended' : `exit ${entry.exitCode}`
}

/** Right-aligned state budget for the row: what it did and how long ago. */
function stateBudget(entry: TerminalSummary, now: number): { color: string; dim: boolean; text: string } {
  const age = terminalAge(entry, now)

  if (entry.running) {
    return { color: 'muted', dim: true, text: `running · ${age}` }
  }

  if (entry.exitCode === 0) {
    // No tick here: the row's state glyph is already a ✓, and printing a
    // second one rendered as `✓ session 4a91 … ✓ exit 0 · 10s`. One mark per
    // fact — the glyph says it succeeded, the budget says what it cost.
    return { color: 'ok', dim: false, text: `exit 0 · ${age}` }
  }

  return {
    color: 'error',
    dim: false,
    text: `${entry.exitCode === null ? 'ended' : `exit ${entry.exitCode}`} · ${age}`
  }
}

function TerminalRow({
  arm,
  entry,
  now,
  selected,
  t
}: {
  arm: null | KillArm
  entry: TerminalSummary
  now: number
  selected: boolean
  t: Theme
}) {
  const color = stateColor(entry, t)
  const budget = stateBudget(entry, now)
  const budgetColor = budget.color === 'muted' ? t.color.muted : budget.color === 'ok' ? t.color.ok : t.color.error

  return (
    <Box flexDirection="column" flexShrink={0}>
      <Box
        backgroundColor={selected ? t.color.selectionBg : undefined}
        flexDirection="row"
        flexShrink={0}
        id={terminalRowId(entry.id)}
        paddingX={1}
      >
        {/* Dot, owner, command — with the state budget hanging off the right
            edge instead of eating a second line. Every row names its owner:
            a shell nobody claims is a bug in the product, not a row in the
            list. */}
        <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
          <Text wrap="truncate-end">
            <Span color={color}>{`${stateGlyph(entry)} `}</Span>
            <Span bold color={t.ds.title}>
              {entry.label}
            </Span>
            {entry.command ? <Span color={t.ds.meta}>{`  ${entry.command}`}</Span> : null}
          </Text>
        </Box>
        <Box flexShrink={0}>
          <Text color={budgetColor} dimColor={budget.dim}>
            {` ${budget.text}`}
          </Text>
        </Box>
      </Box>
      {/* pid and cwd: the two facts that let you go find this shell yourself
          when the panel is not enough. */}
      {entry.pid || entry.cwd ? (
        <Box backgroundColor={selected ? t.color.selectionBg : undefined} flexShrink={0} paddingLeft={3}>
          <Text color={t.ds.caption} wrap="truncate-end">
            {[entry.pid ? `pid ${entry.pid}` : '', entry.cwd].filter(Boolean).join(' · ')}
          </Text>
        </Box>
      ) : null}
      {/* Armed destructive action: the mockup's two-step confirm lives on the
          row itself, warn-colored, until the repeat press or Esc resolves it. */}
      {arm ? (
        <Box backgroundColor={selected ? t.color.selectionBg : undefined} flexShrink={0} paddingLeft={2}>
          <Text color={t.color.warn} wrap="truncate-end">
            {killPrompt(entry.label, arm.force)}
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

function TerminalListView({
  armedKill,
  captionWidth,
  entries,
  loading,
  now,
  scrollRef,
  selectedId,
  t
}: {
  armedKill: null | KillArm
  /** Columns a group caption's rule may run to. */
  captionWidth: number
  entries: readonly TerminalSummary[]
  loading: boolean
  now: number
  scrollRef: React.MutableRefObject<ScrollBoxRenderable | null>
  selectedId: string | undefined
  t: Theme
}) {
  if (loading && !entries.length) {
    return <Text color={t.color.muted}>reading terminals…</Text>
  }

  if (!entries.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>No terminals yet</Text>
        <Text color={t.color.muted} dimColor>
          Shell commands and background processes Xerxes starts appear here.
        </Text>
      </Box>
    )
  }

  return (
    <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
      <Box flexDirection="column" flexShrink={0}>
        {entries.map((entry, index) => {
          const heading = terminalHeading(entries, index)

          return (
            <Box flexDirection="column" flexShrink={0} key={entry.id}>
              {heading ? (
                <GroupCaption
                  count={entries.filter(row => terminalGroup(row) === terminalGroup(entry)).length}
                  label={TERMINAL_GROUP_LABEL[terminalGroup(entry)]}
                  t={t}
                  tone={stateSkin(TERMINAL_GROUP_STATE[terminalGroup(entry)], t.ds).dot}
                  width={captionWidth}
                />
              ) : null}
              <TerminalRow
                arm={activeKillArm(armedKill, entry.id, now)}
                entry={entry}
                now={now}
                selected={entry.id === selectedId}
                t={t}
              />
            </Box>
          )
        })}
      </Box>
    </scrollbox>
  )
}

function TerminalDetailView({
  arm,
  detail,
  draft,
  now,
  outputRef,
  typing,
  t
}: {
  arm: null | KillArm
  detail: null | TerminalInspection
  draft: string
  now: number
  outputRef: React.MutableRefObject<ScrollBoxRenderable | null>
  typing: boolean
  t: Theme
}) {
  if (!detail) {
    return <Text color={t.color.muted}>loading output…</Text>
  }

  const lines = detail.output.replaceAll('\r\n', '\n').replaceAll('\r', '\n').split('\n')

  return (
    <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
      {/* Detail head: dot + bold command, meta under it — the mirror note
          captions the output pane, per the mockup. */}
      <Box flexDirection="column" flexShrink={0} marginBottom={1}>
        <Box flexDirection="row" flexShrink={0}>
          <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
            <Text wrap="truncate-end">
              <Span color={stateColor(detail, t)}>{`${stateGlyph(detail)} `}</Span>
              <Span bold color={t.color.text}>
                {detail.label}
              </Span>
              {detail.command && detail.command !== detail.label ? (
                <Span color={t.color.muted}>{`  ${detail.command}`}</Span>
              ) : null}
            </Text>
          </Box>
          <Box flexShrink={0}>
            <Text color={t.color.muted} dimColor>
              {` ${terminalAge(detail, now)}`}
            </Text>
          </Box>
        </Box>
        <Text color={t.color.muted} dimColor wrap="truncate-end">
          {KIND_LABEL[detail.kind]} · {exitLabel(detail)} · {fmtChars(detail.outputChars)} chars
          {detail.pid ? ` · pid ${detail.pid}` : ''} · {detail.cwd}
        </Text>
      </Box>
      {arm ? (
        <Text color={t.color.warn} wrap="truncate-end">
          {killPrompt(detail.label, arm.force)}
        </Text>
      ) : null}
      <Text color={t.color.muted} dimColor wrap="truncate-end">
        {`OUTPUT — ${detail.label} (mirror)`}
      </Text>
      <scrollbox ref={outputRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          {detail.outputTruncated ? (
            <Text color={t.color.muted} dimColor>
              …earlier output dropped from the viewer's buffer…
            </Text>
          ) : null}
          {lines.map((line, index) => (
            <Text key={index} color={t.color.text} wrap="truncate-end">
              {line || ' '}
            </Text>
          ))}
        </Box>
      </scrollbox>
      {typing ? (
        <Box flexShrink={0} marginTop={1}>
          <Text color={t.color.accent} wrap="truncate-end">
            stdin ▸ {draft}▏
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

export function TerminalPanelOverlay({ onClose, t }: { onClose: () => void; t: Theme }) {
  const gateway = useOptionalGateway()
  const listRef = useRef<ScrollBoxRenderable | null>(null)
  const outputRef = useRef<ScrollBoxRenderable | null>(null)
  const { height, width } = useTerminalDimensions()
  useStore($panelWidthDelta)
  const [entries, setEntries] = useState<readonly TerminalSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [openId, setOpenId] = useState<null | string>(null)
  const [detail, setDetail] = useState<null | TerminalInspection>(null)
  const [notice, setNotice] = useState<null | string>(null)
  // Two-step destructive keys (mockup 06): the kill is armed by the first
  // press and only the confirming repeat sends the signal.
  const [armedKill, setArmedKill] = useState<null | KillArm>(null)
  // Shrink only when the list is empty, so "no terminals yet" is a compact
  // box rather than a full-screen void. A populated panel keeps its full
  // allowance so the output pane and footer hints have room.
  const { height: panelHeight, width: fittedWidth } = overlayPanelSize(
    { height, width },
    entries.length
      ? OVERLAY_PANEL_SPECS.terminals
      : { ...OVERLAY_PANEL_SPECS.terminals, desiredHeight: 0 }
  )
  const panelWidth = withPanelWidthDelta(fittedWidth, width)
  const page = Math.max(4, Math.floor(height * 0.6))
  const [typing, setTyping] = useState(false)
  const [draft, setDraft] = useState('')
  const [now, setNow] = useState(() => Date.now())
  // Pinned to the newest output by default, like `tail -f`. Any manual scroll
  // releases the pin, so reading back through a build log is not fought by the
  // next poll yanking the viewport to the bottom again.
  const followOutput = useRef(true)

  const selected = entries[Math.min(selectedIndex, Math.max(0, entries.length - 1))]

  const refreshList = useCallback(() => {
    if (!gateway) {
      setLoading(false)
      return
    }
    void listTerminals(gateway.rpc)
      .then(next => {
        // Ordered on arrival, not at render: selection, arrow keys and the
        // detail view all index into this array, so a separate display order
        // would drift from what the keyboard is actually moving through.
        setEntries(orderTerminals(next))
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [gateway])

  useEffect(() => {
    refreshList()
    const timer = setInterval(refreshList, LIST_POLL_MS)
    return () => clearInterval(timer)
  }, [refreshList])

  // A running terminal needs a clock of its own: nothing else re-renders while
  // a build sits quiet, and a frozen "12s" next to a spinner reads as a hang.
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!gateway || !openId) {
      setDetail(null)
      return
    }
    let cancelled = false
    const load = () => {
      void inspectTerminal(gateway.rpc, openId)
        .then(next => {
          if (!cancelled) setDetail(next)
        })
        .catch(() => {})
    }
    load()
    const timer = setInterval(load, OUTPUT_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [gateway, openId])

  useEffect(() => {
    if (detail && followOutput.current) outputRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
  }, [detail])

  // Keep the selected row on screen. Attempted twice because on the commit that
  // first mounts a row Yoga has not positioned it yet, so the immediate call
  // computes a zero delta and does nothing.
  const selectedId = selected?.id
  useEffect(() => {
    if (!selectedId || openId) return
    const target = terminalRowId(selectedId)
    listRef.current?.scrollChildIntoView(target)
    const settle = setTimeout(() => listRef.current?.scrollChildIntoView(target), 0)
    return () => clearTimeout(settle)
  }, [entries.length, openId, selectedId])

  const act = useCallback(
    (action: 'interrupt' | 'kill', force = false) => {
      const target = openId ? detail : selected
      if (!gateway || !target) return
      if (!target.running) {
        setNotice('that terminal has already exited')
        return
      }
      if (action === 'kill' && !target.canKill) {
        setNotice('this terminal cannot be killed from here')
        return
      }
      if (action === 'interrupt' && !target.canInterrupt) {
        setNotice('only interactive sessions accept an interrupt')
        return
      }
      setNotice(action === 'kill' ? (force ? 'sending SIGKILL…' : 'sending SIGTERM…') : 'sending Ctrl+C…')
      void controlTerminal(gateway.rpc, target.id, action, { force }).then(error => {
        setNotice(error ?? (action === 'kill' ? 'signal sent' : 'interrupt sent'))
        refreshList()
      })
    },
    [detail, gateway, openId, refreshList, selected]
  )

  const sendDraft = useCallback(() => {
    if (!gateway || !detail) return
    const chars = `${draft}\n`
    setDraft('')
    setTyping(false)
    followOutput.current = true
    void controlTerminal(gateway.rpc, detail.id, 'write', { chars }).then(error =>
      setNotice(error ?? 'sent')
    )
  }, [detail, draft, gateway])

  useKeyboard(event => {
    const name = event.name?.toLowerCase() ?? ''
    const sequence = event.sequence ?? ''

    // Typing into a live shell owns the keyboard: every printable key is input,
    // and only Esc and Enter mean anything to the panel.
    if (typing) {
      consumeKey(event)
      if (name === 'escape') {
        setTyping(false)
        setDraft('')
      } else if (name === 'return' || name === 'enter' || name === 'kpenter') {
        sendDraft()
      } else if (name === 'backspace') {
        setDraft(current => current.slice(0, -1))
      } else if (sequence && sequence.length === 1 && sequence >= ' ' && !event.ctrl && !event.meta) {
        setDraft(current => current + sequence)
      }
      return
    }

    if (isPanelResizeKey(event)) {
      consumeKey(event)
      adjustPanelWidth(event.name === 'right' ? PANEL_WIDTH_STEP : -PANEL_WIDTH_STEP)
      return
    }

    if (name === 'escape' || name === 'f8' || (sequence === 'q' && !event.ctrl && !event.meta)) {
      consumeKey(event)
      // An armed kill is cancelled by the first Esc; only then does Esc back
      // out a level. Cancelling must never take the panel down with it.
      if (armedKill) {
        setArmedKill(null)
        setNotice(null)
        return
      }
      // Esc backs out one level before it closes: the detail view is a place
      // you can be, not a modal on top of the panel.
      if (openId) {
        setOpenId(null)
        setNotice(null)
        followOutput.current = true
      } else {
        onClose()
      }
      return
    }

    if (name === 'return' || name === 'enter' || name === 'kpenter' || name === 'right') {
      consumeKey(event)
      if (!openId && selected) {
        setOpenId(selected.id)
        setDetail(null)
        setNotice(null)
        setArmedKill(null)
        followOutput.current = true
      }
      return
    }

    if (name === 'left' && openId) {
      consumeKey(event)
      setOpenId(null)
      setNotice(null)
      setArmedKill(null)
      return
    }

    // Mockup 06: "destructive keys are two-step; K exists for the rare force
    // case." Both k and K arm on the first press and execute on the repeat —
    // force is a different signal, not a different level of caution. Any other
    // interaction (Esc, moving selection, opening another terminal) disarms.
    if ((sequence === 'k' && !event.ctrl && !event.meta) || sequence === 'K') {
      consumeKey(event)
      const target = openId ? detail : selected
      const force = sequence === 'K'

      if (!target) return

      if (!target.running) {
        setArmedKill(null)
        setNotice('that terminal has already exited')
        return
      }

      if (!target.canKill) {
        setArmedKill(null)
        setNotice('this terminal cannot be killed from here')
        return
      }

      const at = Date.now()
      const armed =
        armedKill !== null && armedKill.terminalId === target.id && at - armedKill.at <= KILL_ARM_WINDOW_MS
          ? armedKill
          : null

      if (!armed || armed.force !== force) {
        setArmedKill({ at, force, terminalId: target.id })
        setNotice(killPrompt(target.label, force))
        return
      }

      setArmedKill(null)
      act('kill', force)
      return
    }

    if (sequence === 'c' && !event.ctrl && !event.meta) {
      consumeKey(event)
      act('interrupt')
      return
    }

    if (sequence === 'i' && !event.ctrl && !event.meta) {
      consumeKey(event)
      if (!openId || !detail) {
        setNotice('open a terminal first (Enter)')
      } else if (!detail.canWrite) {
        setNotice('only interactive PTY sessions accept input')
      } else {
        setTyping(true)
        setDraft('')
      }
      return
    }

    if (sequence === 'r' && !event.ctrl && !event.meta) {
      consumeKey(event)
      refreshList()
      return
    }

    const scroller = openId ? outputRef : listRef
    if (name === 'up' || name === 'down') {
      consumeKey(event)
      if (openId) {
        followOutput.current = false
        scroller.current?.scrollBy(name === 'up' ? -1 : 1)
      } else {
        // Moving the selection abandons any armed kill along with its
        // now-stale confirm notice.
        setArmedKill(null)
        setNotice(null)
        setSelectedIndex(index =>
          Math.max(0, Math.min(entries.length - 1, index + (name === 'up' ? -1 : 1)))
        )
      }
    } else if (isPageUpKey(event) || isPageDownKey(event)) {
      consumeKey(event)
      if (openId) followOutput.current = false
      scroller.current?.scrollBy(isPageUpKey(event) ? -page : page)
    } else if (name === 'home') {
      consumeKey(event)
      if (openId) followOutput.current = false
      else {
        setArmedKill(null)
        setNotice(null)
        setSelectedIndex(0)
      }
      scroller.current?.scrollTo(0)
    } else if (name === 'end') {
      consumeKey(event)
      if (openId) followOutput.current = true
      else {
        setArmedKill(null)
        setNotice(null)
        setSelectedIndex(Math.max(0, entries.length - 1))
      }
      scroller.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    }
  })

  usePaste(event => {
    if (!typing) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    setDraft(current => current + new TextDecoder().decode(event.bytes))
  })

  const liveCount = entries.filter(entry => entry.running).length
  const title = openId ? 'Terminal' : 'Terminals'
  const footer = typing
    ? 'type to send · Enter submit · Esc cancel'
    : openId
      ? '↑↓ scroll · i input · c interrupt · k kill ×2 · K force ×2 · Esc back'
      : '↑↓ select · Enter open · k kill ×2 · K force ×2 · r refresh · Esc close'

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={186}
    >
      <Box
        backgroundColor={t.color.completionBg}
        borderColor={t.color.border}
        borderStyle="round"
        flexDirection="column"
        height={panelHeight}
        paddingX={2}
        paddingY={1}
        width={panelWidth}
      >
        {/* Same header identity as the agents panel: brand mark, title, count
            budget left; the read-only promise on the right, per the mockup. */}
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
          <Box flexDirection="row" flexShrink={1} minWidth={0} overflow="hidden">
            <Text bold color={t.color.text} wrap="truncate-end">
              <Span color={t.color.accent}>✦ </Span>
              {title}
            </Text>
            {openId ? null : (
              <Text color={t.color.muted} wrap="truncate-end">
                {`  ${entries.length} tracked · ${liveCount} running`}
              </Text>
            )}
          </Box>
          <Box flexShrink={0}>
            {openId ? (
              <Text color={liveCount ? t.color.accent : t.color.muted}>{liveCount ? 'running' : 'ended'}</Text>
            ) : (
              <Text color={t.color.muted} dimColor>
                read-only mirror
              </Text>
            )}
          </Box>
        </Box>
        {gateway ? null : (
          <Text color={t.color.warn}>not connected to a daemon — nothing to inspect</Text>
        )}
        {openId ? (
          <TerminalDetailView
            arm={activeKillArm(armedKill, detail?.id, now)}
            detail={detail}
            draft={draft}
            now={now}
            outputRef={outputRef}
            t={t}
            typing={typing}
          />
        ) : (
          <TerminalListView
            armedKill={armedKill}
            captionWidth={Math.max(24, panelWidth - 6)}
            entries={entries}
            loading={loading}
            now={now}
            scrollRef={listRef}
            selectedId={selected?.id}
            t={t}
          />
        )}
        {/* The notice owns a row of its own. Sharing one with the footer meant
            a long hint and a long error ran into each other rather than one
            truncating, and the two texts read as a single garbled line. */}
        {notice ? (
          <Box flexShrink={0} marginTop={1}>
            <Text color={t.color.warn} wrap="truncate-end">
              {notice}
            </Text>
          </Box>
        ) : null}
        <Box flexShrink={0} marginTop={notice ? 0 : 1}>
          <Text color={t.color.muted} wrap="truncate-end">
            {footer}
          </Text>
        </Box>
      </Box>
    </box>
  )
}
