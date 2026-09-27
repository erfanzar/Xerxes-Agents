// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { listWorkspaceIntegrations, inspectWorkspaceIntegration, recoverWorkspaceIntegration, type WorkspaceIntegration } from '../lib/workspaces.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter } from './dialogChrome.js'

export function WorkspaceRecovery({ t, width, height, close }: { t: Theme; width: number; height: number; close: () => void }) {
  const gateway = useOptionalGateway()
  const [rows, setRows] = useState<WorkspaceIntegration[]>([])
  const [selected, setSelected] = useState(0)
  const [cursor, setCursor] = useState<string | undefined>()
  const [pages, setPages] = useState<Array<string | undefined>>([])
  const [next, setNext] = useState<string | undefined>()
  const [refresh, setRefresh] = useState(0)
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)
  const [confirm, setConfirm] = useState(false)
  const [files, setFiles] = useState<Awaited<ReturnType<typeof inspectWorkspaceIntegration>>>([])
  const [inspection, setInspection] = useState('')
  const [filePage, setFilePage] = useState(0)
  const fileScroll = useRef<ScrollBoxRenderable | null>(null)
  const busy = useRef(false)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => {
    let current = true
    setLoading(true); setRows([]); setSelected(0); setConfirm(false)
    if (!gateway) { setMessage('Connect to a daemon first.'); setLoading(false); return }
    void listWorkspaceIntegrations(gateway.rpc, cursor).then(page => { if (current) { setRows(page.records); setNext(page.next) } })
      .catch(error => { if (current) setMessage(String(error)) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [gateway, cursor, refresh])
  const row = rows[selected]
  useEffect(() => {
    let current = true
    setFiles([]); setFilePage(0); setInspection('Checking affected files…')
    fileScroll.current?.scrollTo({ x: 0, y: 0 })
    if (!row || !gateway) return
    void inspectWorkspaceIntegration(gateway.rpc, row.id).then(value => { if (current) { setFiles(value); setInspection(value.length ? 'Check-time preview; recovery rechecks files.' : 'No file restoration required.') } })
      .catch(error => { if (current) setInspection(String(error)) })
    return () => { current = false }
  }, [gateway, row])
  const recoverable = row && !row.error && ['preparing', 'prepared', 'needs-recovery', 'applied', 'rolled-back', 'abandoned'].includes(row.status)
  const lockOnly = row && ['applied', 'rolled-back', 'abandoned'].includes(row.status)
  const recover = () => {
    if (!gateway || !row || !confirm || !recoverable || busy.current) return
    busy.current = true; setConfirm(false); setMessage(lockOnly ? 'Checking leftover lock…' : 'Recovering original files…')
    void recoverWorkspaceIntegration(gateway.rpc, row.id).then(result => { if (mounted.current) setMessage(result) })
      .catch(error => { if (mounted.current) setMessage(String(error)) })
      .finally(() => { busy.current = false; if (mounted.current) setRefresh(value => value + 1) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    key.preventDefault(); key.stopPropagation()
    if (busy.current) return
    if (confirm) { if (key.name === 'y') recover(); else if (key.name === 'escape' || key.name === 'n') setConfirm(false); return }
    if (key.name === 'escape' || key.name === 'i') { close(); return }
    if (key.name === 'r') { setMessage(''); setRefresh(value => value + 1) }
    if (key.name === 'up' || key.name === 'down') setSelected(value => Math.max(0, Math.min(rows.length - 1, value + (key.name === 'down' ? 1 : -1))))
    if (key.name === 'n' && next && !loading) { setPages(value => [...value, cursor]); setCursor(next) }
    if (key.name === 'p' && pages.length && !loading) { setCursor(pages.at(-1)); setPages(value => value.slice(0, -1)) }
    if (key.name === 'b' && recoverable) setConfirm(true)
    if (key.name === 'pagedown') setFilePage(value => Math.min(Math.max(0, Math.ceil(files.length / 3) - 1), value + 1))
    if (key.name === 'pageup') setFilePage(value => Math.max(0, value - 1))
    if (key.name === 'left' || key.name === 'right') fileScroll.current?.scrollBy({ x: key.name === 'right' ? 12 : -12, y: 0 })
  })
  const count = Math.max(1, Math.floor((height - 12) / 2)), start = Math.max(0, selected - count + 1)
  return <Box width={width} height={height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
    <DialogHeader t={t} title={<> Integration recovery · Page {pages.length + 1}</>} />
    <Text color={t.ds.secondary} wrap="wrap">Restore interrupted applies. Completed applies cannot be undone here.</Text>
    {message ? <Text color={t.color.warn} wrap="wrap">{message}</Text> : null}
    {confirm ? <Box flexDirection="column" flexShrink={0}><Text color={t.color.warn} wrap="wrap">{row?.status === 'preparing' ? 'Abandon incomplete preparation? Destination files stay unchanged.' : lockOnly ? 'Release a leftover lock? Current files will be preserved.' : `Restore original files in ${row?.destination}? Newer edits will be preserved.`}</Text><Box onMouseDown={recover}><Text color={t.color.accent}>Y confirm · Esc cancel</Text></Box></Box> : null}
    <Box flexGrow={1} minHeight={0} flexDirection="column">
      {rows.slice(start, start + count).map((item, offset) => <Box key={item.id} flexShrink={0} flexDirection="column" backgroundColor={selected === start + offset ? t.color.selectionBg : undefined} onMouseDown={() => { if (!busy.current && !confirm) setSelected(start + offset) }}>
        <Text color={t.color.text} wrap="truncate-end">{selected === start + offset ? '› ' : '  '}{item.status} · {item.id}</Text>
        <Text color={t.ds.secondary} wrap="truncate-end">  {item.error || `${item.paths.length} affected paths · ${item.destination}`}</Text>
      </Box>)}
      {!rows.length ? <Text color={t.ds.secondary}>{loading ? 'Loading integrations…' : 'No integration records.'}</Text> : null}
    </Box>
    {row ? <Text color={t.ds.secondary} wrap="wrap">Backup: {row.backupPath}</Text> : null}
    {row ? <Box flexDirection="column" flexShrink={0}><Text color={t.ds.secondary} wrap="wrap">{inspection}</Text>{files.length ? <scrollbox ref={fileScroll} scrollX height={4} flexShrink={0}><Box flexDirection="column" width={Math.max(width - 8, ...files.slice(filePage * 3, filePage * 3 + 3).map(file => Bun.stringWidth(`${file.action} · ${file.path} · ${file.reason}`) + 2))}>{files.slice(filePage * 3, filePage * 3 + 3).map(file => <Text key={file.path} color={file.action === 'conflict' ? t.color.warn : t.color.text}>{file.action} · {file.path} · {file.reason}</Text>)}</Box></scrollbox> : null}{files.length ? <Text color={t.ds.secondary}>←→ text · PgUp/PgDn files {filePage + 1}/{Math.ceil(files.length / 3)}</Text> : null}</Box> : null}
    {row?.error ? <Text color={t.color.warn} wrap="wrap">{row.error}</Text> : null}
    {recoverable ? <Box onMouseDown={() => { if (!busy.current) setConfirm(true) }}><Text color={t.color.accent}>{row.status === 'preparing' ? 'B · Abandon preparation' : lockOnly ? 'B · Release leftover lock' : 'B · Restore original files'}</Text></Box> : null}
    <Text color={t.ds.secondary}>↑↓ select · N/P pages · R refresh</Text>
    <DialogFooter t={t}><Text color={t.ds.secondary}>Esc back to workspace review</Text></DialogFooter>
  </Box>
}
