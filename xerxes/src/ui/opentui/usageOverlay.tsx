// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// `/usage` as a popup. It opens at once — mid-turn included — and fills in
// when the runtime answers, so checking your plan never waits on the agent
// and never lands in the transcript between streamed lines. While open it
// keeps the session numbers live and re-reads plan windows every minute.
import { useCallback, useEffect, useRef, useState } from 'react'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'

import { parseUsageReport, type UsageReportView } from '../../auth/usageView.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { patchUiState } from '../app/uiStore.js'
import { usagePanelSections } from '../domain/usagePanel.js'
import type { Theme } from '../theme.js'

import { DialogFooter, DialogHeader } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'
import { PanelMessage } from './panelView.js'
import { Box, Text } from './primitives.js'

/** Session numbers move every step; plan windows are cached by the runtime for a minute. */
const SESSION_POLL_MS = 5_000

function ago(at: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - at) / 1_000))
  return seconds < 10 ? 'just now' : seconds < 60 ? `${seconds}s ago` : `${Math.round(seconds / 60)}m ago`
}

export function UsageOverlay({ t, initialRefresh = false }: { t: Theme; initialRefresh?: boolean }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 84, minWidth: 40, maxHeight: 40 })
  const [report, setReport] = useState<UsageReportView | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const alive = useRef(true)
  useEffect(() => () => { alive.current = false }, [])

  const load = useCallback(async (refresh: boolean) => {
    if (!gateway) { setError('Connect to the runtime to see usage.'); return }
    setLoading(true)
    try {
      // Raw request: an older runtime answers `{ ok: false, error: "Unknown method" }`.
      const raw = await gateway.gw.request<Record<string, unknown>>('usage.report', refresh ? { refresh: true } : {})
      if (!alive.current) return
      if (!raw || typeof raw !== 'object' || raw.ok === false) {
        const reason = String((raw as Record<string, unknown> | null)?.error ?? 'Usage is unavailable.')
        setError(/unknown method/i.test(reason) ? 'Plan limits need the latest runtime. Restart the runtime to update it.' : reason)
        return
      }
      const next = parseUsageReport(raw)
      setReport(next); setError(''); setNow(Date.now())
      const session = raw.session as Record<string, number> | undefined
      if (session) patchUiState({ usage: { calls: session.calls ?? 0, input: session.input ?? 0, output: session.output ?? 0, total: session.total ?? 0 } })
    } catch (failure) {
      if (alive.current) setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      if (alive.current) setLoading(false)
    }
  }, [gateway])

  useEffect(() => {
    void load(initialRefresh)
    const timer = setInterval(() => { void load(false) }, SESSION_POLL_MS)
    return () => clearInterval(timer)
  }, [load, initialRefresh])

  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (!['escape', 'q', 'r', 'up', 'down', 'pageup', 'pagedown', 'home', 'end'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape' || key.name === 'q') patchOverlayState({ usage: false })
    else if (key.name === 'r') void load(true)
    else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else scroll.current?.scrollBy((key.name === 'up' || key.name === 'pageup' ? -1 : 1) * (key.name.startsWith('page') ? Math.max(1, size.height - 8) : 1))
  })

  const inner = Math.max(30, size.width - 4)
  const status = loading && !report ? 'checking your plans…' : report ? `updated ${ago(report.fetchedAt, now)}${loading ? ' · refreshing…' : ''}` : ''
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.statusBg}>
      <DialogHeader t={t} title={<>Usage{status ? <span fg={t.ds.meta}>{` · ${status}`}</span> : null}</>} subtitle="This session, and the limits on every provider you have imported." />
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      <scrollbox ref={scroll} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
        {report ? <PanelMessage panel={{ title: 'Usage', sections: usagePanelSections(report, now) }} t={t} titled={false} width={inner} /> : null}
      </scrollbox>
      <DialogFooter t={t}><Text color={t.ds.secondary}>R refresh plans · ↑↓ scroll · Esc close</Text></DialogFooter>
    </Box>
  </box>
}
