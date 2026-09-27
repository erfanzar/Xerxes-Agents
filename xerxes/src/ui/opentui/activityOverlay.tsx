// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useEffect, useState } from 'react'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'
import { isActive, useActivity, type ActivityRow } from './useActivity.js'

export function activityAction(row: ActivityRow): { method: string; params: Record<string, unknown> } | null {
  if (!row.action) return null
  if (row.kind === 'shell') return { method: 'terminal.control', params: { terminal_id: row.id, action: 'kill' } }
  if (row.kind === 'watcher') return { method: 'monitor.stop', params: { monitor_id: row.id } }
  return { method: row.action === 'cancel' ? 'schedule.cancel' : 'schedule.pause', params: { schedule_id: row.id, revision: row.revision } }
}
export function ActivityOverlay({ sessionId, t }: { sessionId: string | null; t: Theme }) {
  const { rows, omitted, now, error, loading, refresh, gateway } = useActivity(sessionId)
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 120, maxHeight: 36, minWidth: 32 })
  const [selected, setSelected] = useState('')
  const [inspecting, setInspecting] = useState(false)
  const [output, setOutput] = useState('')
  const [filter, setFilter] = useState(0)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')
  const filters = ['all','shell','watcher','schedule']
  const visible = rows.filter(row => filter === 0 || row.kind === filters[filter])
  const index = Math.max(0, visible.findIndex(row => row.kind + ':' + row.id === selected))
  const row = visible[index]
  const actionLabel = row?.action === 'pause' ? 'Pause schedule' : row?.action === 'cancel' ? 'Cancel current run' : row?.kind === 'watcher' ? 'Stop watch / reactions' : 'Stop shell'
  const stop = async () => {
    if (!row || !sessionId || !gateway || busy) return
    const action = activityAction(row)
    if (!action) return
    setBusy(true); setActionError('')
    try {
      const result = await gateway.gw.request<{ ok?: boolean; error?: string; requested?: boolean }>(action.method, { ...action.params, session_id: sessionId })
      if (!result?.ok || result.requested === false) throw new Error(result?.error ?? 'Work already stopped; refresh its status.')
      refresh()
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }
  const inspect = () => { if (row) setInspecting(true) }
  useEffect(() => {
    if (!inspecting || !row || !gateway) return
    let active = true
    setOutput('Loading details…')
    const method = row.kind === 'shell' ? 'terminal.inspect' : row.kind === 'watcher' ? 'monitor.inspect' : 'schedule.inspect'
    const params = row.kind === 'shell' ? { terminal_id: row.id, max_output_chars: 20000 } : row.kind === 'watcher' ? { monitor_id: row.id } : { schedule_id: row.id }
    void gateway.gw.request<Record<string, unknown>>(method, { ...params, session_id: sessionId }).then(result => {
      if (result.ok !== true) throw new Error(String(result.error ?? 'Inspection failed'))
      const value = result.terminal ?? result.monitor ?? result.job ?? result
      const detail = value && typeof value === 'object' ? value as Record<string, unknown> : {}
      const body = row.kind === 'shell' ? String(detail.output || '(No output yet)') : row.kind === 'watcher'
        ? [row.detail, 'State: ' + row.state, ...(Array.isArray(detail.events) ? detail.events.map((event: unknown) => event && typeof event === 'object' && 'text' in event ? String(event.text) : '') : [])].join('\n\n')
        : [row.detail, 'State: ' + row.state, row.nextRunAt ? 'Next run: ' + new Date(row.nextRunAt).toLocaleString() : 'No next run', row.lastState ? 'Last run: ' + row.lastState : 'No previous run'].join('\n\n')
      if (active) setOutput(body.slice(0, 30000))
    }).catch(error => { if (active) setOutput(String(error)) })
    return () => { active = false }
  }, [inspecting, row?.id, row?.state, sessionId, gateway])
  useKeyboard(key => {
    if (key.eventType === 'release' || !['escape','up','down','tab','s','r','return','enter'].includes(key.name)) return
    if (inspecting && ['up','down'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { if (inspecting) setInspecting(false); else patchOverlayState({ activity: false }) }
    else if (key.name === 'tab') setFilter(n => (n + 1) % filters.length)
    else if (key.name === 's') void stop()
    else if (key.name === 'r') { setInspecting(false); refresh() }
    else if (key.name === 'return' || key.name === 'enter') inspect()
    else { const next = visible[Math.max(0, Math.min(visible.length - 1, index + (key.name === 'down' ? 1 : -1)))]; if (next) setSelected(next.kind + ':' + next.id) }
  })
  const count = Math.max(1, Math.floor((size.height - 13) / 3))
  const start = Math.max(0, index - count + 1)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} paddingX={2} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg} flexDirection="column">
      <DialogHeader t={t} title="Background activity" subtitle="Shells and watches in this chat · schedules in this workspace" />
      <Box flexDirection="row" gap={2} flexShrink={0}>{filters.map((name,i) => <Box key={name} onClick={() => setFilter(i)}><Text color={filter === i ? t.color.brandGold : t.ds.secondary}>{name === 'all' ? 'All activity' : name + 's'}</Text></Box>)}</Box>
      <Text color={t.ds.secondary}>{rows.filter(isActive).length} active · {rows.filter(row => row.state === 'scheduled').length} scheduled{omitted ? ` · ${omitted} older entries omitted` : ''}</Text>
      {inspecting ? <scrollbox style={{ flexGrow: 1, minHeight: 0 }}><Text color={t.color.text} wrap="wrap">{row?.title + '\n\n' + output}</Text></scrollbox> : <Box flexGrow={1} minHeight={0} flexDirection="column" paddingTop={1}>
        {loading ? <Text color={t.ds.secondary}>Loading activity…</Text> : !visible.length && !error ? <DialogEmpty t={t} title="Nothing running here." description="Background commands, watches and schedules appear automatically." /> : null}
        {visible.slice(start, start + count).map(item => <Box key={item.kind + ':' + item.id} flexShrink={0} height={3} flexDirection="column" paddingX={1} backgroundColor={item === row ? t.ds.selected : undefined} onClick={() => setSelected(item.kind + ':' + item.id)}>
          <Text color={item.state === 'failed' ? t.color.warn : isActive(item) ? t.color.brandGold : t.color.text} wrap="truncate-end">{isActive(item) ? '● ' : '○ '}{item.title}</Text>
          <Text color={t.ds.secondary} wrap="truncate-end">{item.kind} · {item.state}{item.startedAt !== null ? ` · ${Math.max(0, Math.floor(((item.endedAt ?? now) - item.startedAt) / 1000))}s` : ''}{item.scope === 'workspace' ? ' · workspace' : ''}</Text>
        </Box>)}
      </Box>}
      {row ? <Box flexDirection="column" flexShrink={0} paddingY={1}>
        <Text color={t.color.text} wrap="truncate-end">{row.title}</Text>
        <Text color={t.ds.secondary} wrap="truncate-end">{row.detail}</Text>
        {row.nextRunAt ? <Text color={t.ds.secondary} wrap="truncate-end">Next: {new Date(row.nextRunAt).toLocaleString()}</Text> : null}
        <Box flexDirection="row" gap={2}><Box onClick={inspect}><Text color={t.color.brandGold}>Enter · Details / output</Text></Box>{row.action ? <Box onClick={() => void stop()}><Text color={t.color.warn}>{busy ? 'Requesting…' : 'S · ' + actionLabel}</Text></Box> : null}</Box>
      </Box> : null}
      {error || actionError ? <Text color={t.color.warn} wrap="wrap">{actionError || error}</Text> : null}
      <DialogFooter t={t}><Text color={t.ds.secondary}>↑↓ select · Tab filter · R refresh · Esc close</Text></DialogFooter>
    </Box>
  </box>
}
