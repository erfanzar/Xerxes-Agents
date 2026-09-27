// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { listMonitors, monitorAction, type MonitorView } from '../lib/monitors.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { MonitorPolicy } from './monitorPolicy.js'
import { MonitorCreate } from './monitorCreate.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty, DialogSection } from './dialogChrome.js'

export function MonitorOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const [editing, setEditing] = useState<MonitorView | null>(null)
  const [creating, setCreating] = useState(false)
  const size = overlayPanelSize(terminal, { maxWidth: creating ? 120 : 124, maxHeight: 38, minWidth: 32, ...(creating ? { desiredHeight: 32 } : {}) })
  const [rows, setRows] = useState<MonitorView[]>([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState<MonitorView | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        if (!gateway) throw new Error('Connect to a daemon to inspect monitors.')
        const values = await listMonitors(gateway.rpc)
        if (!current) return
        setRows(values)
        setSelected(id => values.some(row => row.id === id) ? id : values[0]?.id ?? '')
      } catch (failure) { if (current) setError(String(failure)) }
      finally { if (current) timer = setTimeout(() => { void load() }, 2000) }
    }
    void load()
    return () => { current = false; if (timer) clearTimeout(timer) }
  }, [gateway, refresh])
  useEffect(() => {
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    setDetail(null)
    scroll.current?.scrollTo(0)
    const load = async () => {
      if (!gateway || !selected) return
      try {
        const value = await monitorAction(gateway.rpc, selected, 'inspect')
        if (current) setDetail(value)
      } catch (failure) { if (current) setError(String(failure)) }
      finally { if (current) timer = setTimeout(() => { void load() }, 2000) }
    }
    void load()
    return () => { current = false; if (timer) clearTimeout(timer) }
  }, [gateway, selected, refresh])
  const selectedView = detail?.id === selected ? detail : rows.find(row => row.id === selected)
  const action = selectedView?.stopAction
  const actionLabel = action === 'stop-watch' ? 'S stop' : action === 'cancel-reactions' ? 'S cancel reactions' : selectedView?.state === 'detached' ? 'Owned by another daemon' : 'No cancellable work'
  const stop = () => {
    if (!gateway || !selected || busy || !action) return
    setBusy(true)
    void monitorAction(gateway.rpc, selected, 'stop').then(() => {
      if (alive.current) setRefresh(value => value + 1)
    }).catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { if (alive.current) setBusy(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release' || creating || editing) return
    if (!['escape', 'up', 'down', 's', 'r', 'n', 'e', 'pageup', 'pagedown', 'home', 'end'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') patchOverlayState({ monitors: false })
    else if (key.name === 'e' && selectedView?.policy) setEditing(selectedView)
    else if (key.name === 'n') setCreating(true)
    else if (key.name === 's') stop()
    else if (key.name === 'r') { setError(''); setRefresh(value => value + 1) }
    else if (key.name === 'up' || key.name === 'down') {
      const next = Math.max(0, Math.min(rows.length - 1, rows.findIndex(row => row.id === selected) + (key.name === 'up' ? -1 : 1)))
      setSelected(rows[next]?.id ?? '')
    } else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, size.height - 10))
  })
  const wide = size.width >= 100
  const count = wide ? Math.max(1, Math.floor((size.height - 13) / 2)) : 3
  const start = Math.max(0, rows.findIndex(row => row.id === selected) - count + 1)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!rows.length && !creating && !editing ? Math.min(88, size.width) : size.width} height={!rows.length && !creating && !editing ? Math.min(24, size.height) : size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      {editing ? <MonitorPolicy t={t} monitor={editing} onClose={() => setEditing(null)} onSaved={() => { setEditing(null); setRefresh(value => value + 1) }} /> : creating ? <MonitorCreate t={t} onClose={() => setCreating(false)} onCreated={id => { setCreating(false); setSelected(id); setRefresh(value => value + 1) }} /> : <>
      <DialogHeader t={t} title={<>Monitors · {rows.filter(row => row.state === 'watching').length} watching</>} subtitle="Watch for changes. Keep the important events in view." />
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {!rows.length ? <DialogEmpty t={t} title="No watches in this session." description="Watch a file, terminal or connected source." action="+ N  Create monitor" onAction={() => setCreating(true)} symbol="◎" /> : (<Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? '35%' : '100%'} height={wide ? '100%' : Math.min(6, rows.length * 2 + 1)} flexDirection="column" paddingRight={1}>
          {rows.length ? rows.slice(start, start + count).map(row => <Box key={row.id} height={2} flexDirection="column" backgroundColor={row.id === selected ? t.ds.selected : undefined} onMouseDown={() => setSelected(row.id)}>
            <Text color={t.color.text} wrap="truncate-end">{row.source?.kind === 'file' ? `File changes · ${row.source.path}` : row.source?.kind === 'websocket' ? `Websocket · ${row.source.url}` : row.source?.kind === 'webhook' ? `Webhook · ${row.source.name}` : row.match}</Text>
            <Text color={t.ds.secondary} wrap="truncate-end">{row.state} · {row.source?.kind === 'file' ? row.source.workspace : row.source?.kind === 'websocket' ? 'server push' : row.source?.kind === 'webhook' ? 'configured source' : row.terminalId}</Text>
          </Box>) : <DialogEmpty t={t} title="No watches in this session." description="Watch a file, terminal or connected source." action="+ N  Create monitor" onAction={() => setCreating(true)} symbol="◎" />}
        </Box>
        <scrollbox ref={scroll} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
          {detail ? <Box flexDirection="column" flexShrink={0}>
            <DialogSection t={t}>SOURCE</DialogSection><Text bold color={t.color.text} wrap="wrap">{detail.source?.kind === 'file' ? `File changes: ${detail.source.path}` : detail.source?.kind === 'websocket' ? `Websocket: ${detail.source.url}` : detail.source?.kind === 'webhook' ? `Webhook: ${detail.source.name}` : `Match: ${detail.match}`}</Text>
            <Text color={t.ds.secondary} wrap="wrap">{detail.state} · {detail.reaction}</Text>
            {detail.source?.kind === 'file' ? <Text color={t.ds.secondary} wrap="wrap">Workspace: {detail.source.workspace} · metadata changes only</Text> : null}
            {detail.source?.kind === 'websocket' ? <Text color={t.ds.secondary} wrap="wrap">Text-only server push · duplicates suppressed · reconnect gaps possible</Text> : null}
            {detail.source?.kind === 'webhook' ? <Text color={t.ds.secondary} wrap="wrap">Configured webhook · text-only payloads · delivery gaps possible</Text> : null}
            <Text color={t.ds.secondary} wrap="wrap">Expires: {new Date(detail.expiresAt).toLocaleString()}</Text>
            {detail.sourceStatus ? <Text color={t.ds.secondary} wrap="wrap">{detail.sourceStatus}</Text> : null}
            {detail.error ? <Text color={t.color.warn} wrap="wrap">{detail.error}</Text> : null}
            {detail.omittedEvents ? <Text color={t.ds.secondary}>Earlier events: {detail.omittedEvents} · inspect Runs for history</Text> : null}
            <DialogSection t={t}>RECENT EVENTS</DialogSection><Text color={t.color.text} wrap="wrap">{detail.events.join('\n\n') || 'No matching events yet.'}</Text>
          </Box> : <Text color={t.ds.secondary}>Select a watch to inspect.</Text>}
        </scrollbox>
      </Box>)}

      <DialogFooter t={t}><Text color={t.ds.secondary}>{busy ? 'Applying cancellation…' : !rows.length ? 'N new · R refresh · Esc close' : size.width < 60 ? actionLabel : `N new · ↑↓ select · ${actionLabel}`}</Text>
      {rows.length ? <Text color={t.ds.secondary}>{size.width < 60 ? 'E limits · N new · Esc close' : 'E edit limits · R refresh · PgUp/Dn · Esc close'}</Text> : null}</DialogFooter>
      </>}
    </Box>
  </box>
}
