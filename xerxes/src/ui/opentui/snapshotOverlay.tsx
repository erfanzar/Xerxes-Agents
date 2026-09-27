// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useStore } from '@nanostores/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId } from '../app/uiStore.js'
import { parseUnifiedDiff } from '../lib/gitDiff.js'
import type { Theme } from '../theme.js'
import { DiffRow } from './diffPanel.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'

interface Snapshot { id: string; label: string; created_at: string; session_id?: string; turn_index?: number }
interface Preview { id: string; revision: string; diff: string; truncated: boolean; path?: string; action?: 'restore' | 'remove' }
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid snapshot response')
  const result = value as Record<string, unknown>
  if (result.ok === false) throw new Error(typeof result.error === 'string' ? result.error : 'Snapshot request failed')
  return result
}
export function SnapshotOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const sid = useStore($uiSessionId)
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 220, minWidth: 32 })
  const [recovery, setRecovery] = useState<Array<{ backupId: string; phase: string }>>([])
  const [rows, setRows] = useState<Snapshot[]>([])
  const [selected, setSelected] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [filePath, setFilePath] = useState<string | undefined>()
  const [preview, setPreview] = useState<Preview | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [applying, setApplying] = useState(false)
  const busy = useRef(false)
  const alive = useRef(true)
  const [focusDiff, setFocusDiff] = useState(false)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const rpc = gateway?.rpc
  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(''); setRows([]); setSelected(''); setFilePath(undefined); setFiles([]); setPreview(null); setConfirm(false)
    if (!rpc) { setError('Daemon unavailable'); setLoading(false); return }
    void rpc('snapshot.list', { ...(sid ? { session_id: sid } : {}) }).then(value => {
      if (cancelled) return
      const data = record(value)
      if (!Array.isArray(data.snapshots)) throw new Error('Invalid snapshot list')
      if (data.restore_attempts !== undefined && !Array.isArray(data.restore_attempts)) throw new Error('Invalid restore recovery history')
      const attempts = (data.restore_attempts ?? []) as unknown[]
      setRecovery(attempts.map(value => {
        const row = record(value)
        if (typeof row.backupId !== 'string' || typeof row.phase !== 'string') throw new Error('Invalid restore recovery record')
        return { backupId: row.backupId, phase: row.phase }
      }).filter(row => row.phase === 'prepared' || row.phase === 'failed'))
      const next = data.snapshots.map(item => {
        const row = record(item)
        if (typeof row.id !== 'string' || typeof row.label !== 'string' || typeof row.created_at !== 'string') throw new Error('Invalid snapshot record')
        return { id: row.id, label: row.label, created_at: row.created_at,
          ...(typeof row.session_id === 'string' ? { session_id: row.session_id } : {}),
          ...(typeof row.turn_index === 'number' ? { turn_index: row.turn_index } : {}) }
      }).reverse()
      setRows(next); setSelected(next[0]?.id ?? '')
    }).catch(cause => { if (!cancelled) setError(String(cause)) }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [rpc, sid, refresh])
  useEffect(() => {
    let cancelled = false
    setPreview(null); setConfirm(false); setError('')
    if (!rpc || !selected) return
    void rpc('snapshot.preview', { snapshot_id: selected, ...(filePath === undefined ? {} : { path: filePath }), ...(sid ? { session_id: sid } : {}) }).then(value => {
      if (cancelled) return
      const data = record(value)
      if (data.snapshot_id !== selected || typeof data.revision !== 'string' || !/^[a-f0-9]{64}$/.test(data.revision) || typeof data.diff !== 'string') throw new Error('Invalid snapshot preview')
      if (data.files !== undefined && (!Array.isArray(data.files) || !data.files.every(path => typeof path === 'string'))) throw new Error('Invalid snapshot file list')
      if (filePath !== undefined && data.action !== 'restore' && data.action !== 'remove') throw new Error('Invalid file restore action')
      setFiles((data.files ?? []) as string[])
      setPreview({ ...(filePath === undefined ? {} : { path: filePath, action: data.action as 'restore' | 'remove' }), id: selected, revision: data.revision, diff: data.diff, truncated: data.truncated === true })
      scroll.current?.scrollTo(0)
    }).catch(cause => { if (!cancelled) setError(String(cause)) })
    return () => { cancelled = true }
  }, [rpc, sid, selected, refresh, filePath])
  const apply = () => {
    if (!rpc || !confirm || !preview || preview.id !== selected || preview.path !== filePath || busy.current) return
    busy.current = true; setApplying(true); setConfirm(false)
    const request = preview.path === undefined
      ? rpc('slash.exec', { command: `rollback apply ${preview.id} ${preview.revision}`, ...(sid ? { session_id: sid } : {}) })
      : rpc('snapshot.restoreFile', { snapshot_id: preview.id, path: preview.path, revision: preview.revision, ...(sid ? { session_id: sid } : {}) })
    void request.then(value => {
      if (record(value).ok !== true) throw new Error('Invalid restore response; refresh to inspect current files')
      if (alive.current) { setNotice('Files restored. Conversation unchanged.'); setRefresh(value => value + 1) }
    }).catch(cause => { if (alive.current) { setError(String(cause)); setPreview(null) } }).finally(() => {
      busy.current = false
      if (alive.current) setApplying(false)
    })
  }
  useKeyboard(key => {
    if (busy.current) return
    if (key.name === 'escape') { if (confirm) setConfirm(false); else patchOverlayState({ snapshots: false }); return }
    if (confirm) { if (key.name === 'y') apply(); return }
    if (key.name === 'b' && recovery.length) { setFilePath(undefined); setSelected(recovery[0]!.backupId); return }
    if (key.name === 'r') { setRefresh(value => value + 1); return }
    if (key.name === 'a' && preview?.id === selected && preview.path === filePath) { setConfirm(true); return }
    if (key.name === 'f' && files.length) { setFilePath(path => files[files.indexOf(path ?? '') + 1]); return }
    if (key.name === 'tab') { setFocusDiff(value => !value); return }
    if (key.name === 'up' || key.name === 'down') {
      const step = key.name === 'down' ? 1 : -1
      if (focusDiff) scroll.current?.scrollBy(step)
      else { setFilePath(undefined); setSelected(id => rows[Math.max(0, Math.min(rows.length - 1, rows.findIndex(row => row.id === id) + step))]?.id ?? '') }
    }
    if (key.name === 'pageup' || key.name === 'pagedown') scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, size.height - 8))
    if (key.name === 'left' || key.name === 'right') scroll.current?.scrollBy({ x: key.name === 'right' ? 12 : -12, y: 0 })
  })
  const parsed = useMemo(() => parseUnifiedDiff(preview?.diff ?? ''), [preview])
  const wide = size.width >= 100
  const count = wide ? Math.max(1, Math.floor((size.height - 7) / 2)) : 1
  const index = rows.findIndex(row => row.id === selected)
  const start = Math.max(0, index - count + 1)
  const codeWidth = Math.max(1, size.width - (wide ? 46 : 8), ...parsed.lines.map(line => Bun.stringWidth(line.text) + 18))
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!rows.length && !confirm && !recovery.length ? Math.min(88, size.width) : size.width} height={!rows.length && !confirm && !recovery.length ? Math.min(24, size.height) : size.height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      <DialogHeader t={t} title={<>Snapshot timeline · {rows.length}</>} subtitle="Browse saved file states and preview a restore." />
      <Text color={t.ds.secondary} wrap="truncate-end">Workspace files · conversation unchanged</Text>
      {error ? <Text color={t.color.warn} wrap="truncate-end">{error}</Text> : null}
      {recovery.length ? <Text color={t.color.warn} wrap='truncate-end'>{recovery.length} unfinished restore(s) · B backup</Text> : null}
      {notice ? <Text color={t.color.accent} wrap="truncate-end">{notice}</Text> : null}
      {!rows.length ? <DialogEmpty t={t} title={loading ? "Loading…" : "No snapshots."} description="Save a file state you can return to." action="Use /snapshot to capture one." symbol="▣" /> : (<Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? 38 : '100%'} height={wide ? '100%' : 2} flexDirection="column" flexShrink={0}>
          {rows.slice(start, start + count).map(row => <Box key={row.id} flexDirection="column" flexShrink={0} backgroundColor={row.id === selected ? t.color.selectionBg : undefined} onMouseDown={() => { if (!busy.current && !confirm) { setSelected(row.id); setFilePath(undefined); setFocusDiff(false) } }}>
            <Text color={t.color.text} wrap="truncate-end">{row.id === selected ? '› ' : '  '}{row.label || 'Snapshot'}</Text>
            <Text color={t.ds.secondary} wrap="truncate-end">{row.turn_index === undefined ? 'Manual' : `Before turn ${row.turn_index}`} · {row.created_at}{row.session_id ? ` · ${row.session_id}` : ''}</Text>
          </Box>)}
          {!rows.length ? <DialogEmpty t={t} title={loading ? 'Loading…' : 'No snapshots.'} description="Save a file state you can return to." action="Use /snapshot to capture one." symbol="▣" /> : null}
        </Box>
        <Box flexDirection="column" flexGrow={1} minHeight={0} minWidth={0}>
          <Text color={t.color.accent} wrap="truncate-end">{focusDiff ? '› ' : ''}{filePath === undefined ? `All changed files (${files.length})` : `${preview?.action === 'remove' ? 'Remove' : 'Restore'}: ${filePath}`}{preview?.truncated ? ' · truncated' : ''}</Text>
          <scrollbox ref={scroll} flexGrow={1} minHeight={0} scrollX scrollY>
            <Box width={codeWidth} flexDirection="column">{parsed.lines.map((line, i) => <DiffRow key={i} line={line} t={t} />)}</Box>
            {!preview ? <Text color={t.ds.secondary}>{selected && !error ? 'Loading preview…' : 'Select a snapshot'}</Text> : !preview.diff ? <Text color={t.ds.secondary}>No captured-file changes.</Text> : null}
          </scrollbox>
        </Box>
      </Box>)}

      <DialogFooter t={t}><Text color={t.ds.secondary} wrap="truncate-end">Ignored uncaptured files excluded.</Text>
      {confirm ? <Box flexDirection="column"><Text color={t.color.warn} wrap="wrap">{preview?.path === undefined ? 'Restore all captured files?' : `${preview.action === 'remove' ? 'Remove' : 'Restore'} ${preview.path}?`} Backup is saved first.</Text><Text color={t.color.accent}>Y restore · Esc cancel</Text></Box> : <Text color={t.ds.secondary} wrap="wrap">{applying ? 'Restoring…' : size.width < 60 ? '↑↓ select · F file · A restore\nTab diff · R refresh · Esc close' : '↑↓ select · Tab diff · F file · A restore · R refresh · Esc close'}</Text>}
      </DialogFooter>
    </Box>
  </box>
}
