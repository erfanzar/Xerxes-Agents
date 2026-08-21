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
 */

type KeyEvent = Parameters<Parameters<typeof useKeyboard>[0]>[0]

/** Poll cadences. Fast enough to read as live, slow enough to stay cheap. */
const LIST_POLL_MS = 1_200
const OUTPUT_POLL_MS = 700

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

function TerminalRow({
  entry,
  now,
  selected,
  t
}: {
  entry: TerminalSummary
  now: number
  selected: boolean
  t: Theme
}) {
  const color = stateColor(entry, t)

  return (
    <Box
      backgroundColor={selected ? t.color.selectionBg : undefined}
      flexDirection="row"
      flexShrink={0}
      id={terminalRowId(entry.id)}
      paddingX={1}
    >
      <Box flexShrink={0} width={2}>
        <Text color={color}>{stateGlyph(entry)}</Text>
      </Box>
      <Box flexDirection="column" flexGrow={1} flexShrink={1}>
        <Text color={t.color.text} wrap="truncate-end">
          <Span color={t.color.muted}>{KIND_LABEL[entry.kind]} </Span>
          {entry.label}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {exitLabel(entry)} · {terminalAge(entry, now)} · {fmtChars(entry.outputChars)} chars
          {entry.pid ? ` · pid ${entry.pid}` : ''}
        </Text>
      </Box>
    </Box>
  )
}

function TerminalListView({
  entries,
  loading,
  now,
  scrollRef,
  selectedId,
  t
}: {
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
        {entries.map(entry => (
          <TerminalRow key={entry.id} entry={entry} now={now} selected={entry.id === selectedId} t={t} />
        ))}
      </Box>
    </scrollbox>
  )
}

function TerminalDetailView({
  detail,
  draft,
  now,
  outputRef,
  typing,
  t
}: {
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
      <Box flexDirection="column" flexShrink={0} marginBottom={1}>
        <Text color={t.color.text} wrap="truncate-end">
          <Span color={stateColor(detail, t)}>{stateGlyph(detail)} </Span>
          {detail.command || detail.label}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {KIND_LABEL[detail.kind]} · {exitLabel(detail)} · {terminalAge(detail, now)} ·{' '}
          {fmtChars(detail.outputChars)} chars{detail.pid ? ` · pid ${detail.pid}` : ''}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {detail.cwd}
        </Text>
      </Box>
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
        setEntries(next)
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
        followOutput.current = true
      }
      return
    }

    if (name === 'left' && openId) {
      consumeKey(event)
      setOpenId(null)
      return
    }

    if (sequence === 'k' && !event.ctrl && !event.meta) {
      consumeKey(event)
      act('kill')
      return
    }

    if (sequence === 'K') {
      consumeKey(event)
      act('kill', true)
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
        setSelectedIndex(index =>
          Math.max(0, Math.min(entries.length - 1, index + (name === 'up' ? -1 : 1)))
        )
      }
    } else if (name === 'pageup' || name === 'pagedown') {
      consumeKey(event)
      if (openId) followOutput.current = false
      scroller.current?.scrollBy(name === 'pageup' ? -page : page)
    } else if (name === 'home') {
      consumeKey(event)
      if (openId) followOutput.current = false
      else setSelectedIndex(0)
      scroller.current?.scrollTo(0)
    } else if (name === 'end') {
      consumeKey(event)
      if (openId) followOutput.current = true
      else setSelectedIndex(Math.max(0, entries.length - 1))
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
      ? '↑↓ scroll · i input · c interrupt · k kill (K force) · Esc back'
      : '↑↓ select · Enter open · k kill (K force) · r refresh · Esc close'

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
        backgroundColor={t.color.statusBg}
        borderColor={t.color.brandGold}
        borderStyle="single"
        flexDirection="column"
        height={panelHeight}
        paddingX={2}
        paddingY={1}
        width={panelWidth}
      >
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
          <Text bold color={t.color.accent}>
            ▌ {title}
          </Text>
          <Text color={liveCount ? t.color.accent : t.color.muted}>
            {liveCount ? `${liveCount} running` : `${entries.length} tracked`}
          </Text>
        </Box>
        {gateway ? null : (
          <Text color={t.color.warn}>not connected to a daemon — nothing to inspect</Text>
        )}
        {openId ? (
          <TerminalDetailView
            detail={detail}
            draft={draft}
            now={now}
            outputRef={outputRef}
            t={t}
            typing={typing}
          />
        ) : (
          <TerminalListView
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
