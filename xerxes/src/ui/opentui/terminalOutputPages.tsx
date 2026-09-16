// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter } from './dialogChrome.js'

type Cursor = { streamId: string; offset: number }
type Page = { text: string; cursor: Cursor; droppedChars: number; hasMore: boolean; running: boolean }
function parsePage(value: unknown): Page {
  if (!value || typeof value !== 'object') throw new Error('Invalid output page')
  const p = value as Record<string, unknown>, c = p.cursor as Record<string, unknown> | undefined
  if (typeof p.text !== 'string' || typeof p.hasMore !== 'boolean' || typeof p.running !== 'boolean'
    || typeof p.droppedChars !== 'number' || !Number.isSafeInteger(p.droppedChars) || p.droppedChars < 0
    || !c || typeof c.streamId !== 'string' || typeof c.offset !== 'number' || !Number.isSafeInteger(c.offset) || c.offset < 0) throw new Error('Invalid output page')
  return { text: p.text, hasMore: p.hasMore, running: p.running, droppedChars: p.droppedChars, cursor: { streamId: c.streamId, offset: c.offset } }
}

/** Independent read cursor; never drains a tool or another viewer's output. */
export function TerminalOutputPages({ t, terminalId, onClose }: { t: Theme; terminalId: string; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const [page, setPage] = useState<Page | null>(null)
  const [cursors, setCursors] = useState<Array<Cursor | undefined>>([undefined])
  const [error, setError] = useState(''), [loading, setLoading] = useState(false)
  const epoch = useRef(0), busy = useRef(false), scroll = useRef<ScrollBoxRenderable | null>(null)
  const lines = (page?.text || (page ? 'No output in this page.' : '')).split('\n')
  const outputWidth = Math.max(1, ...lines.map(line => Bun.stringWidth(line)))
  const retry = useRef<Array<Cursor | undefined> | null>(null)
  const load = async (next: Array<Cursor | undefined>) => {
    if (!gateway || busy.current) return
    busy.current = true; setLoading(true); setError(''); retry.current = next
    const request = ++epoch.current, cursor = next.at(-1)
    try {
      const result = await gateway.rpc('terminal.output', { terminal_id: terminalId, max_output_chars: 8000, ...(cursor ? { cursor } : {}) })
      if (request !== epoch.current) return
      if (result?.ok !== true) throw new Error(String(result?.error ?? 'Output is unavailable'))
      const value = parsePage(result.page)
      setPage(value); setCursors(next); retry.current = null
      if (next.length !== cursors.length) scroll.current?.scrollTo(0)
    } catch (failure) { if (request === epoch.current) setError(String(failure)) }
    finally { if (request === epoch.current) { busy.current = false; setLoading(false) } }
  }
  useEffect(() => {
    busy.current = false; setPage(null); setCursors([undefined]); retry.current = null
    if (gateway) void load([undefined]); else setError('Connect to a daemon to read output.')
    return () => { epoch.current++ }
  }, [gateway, terminalId])
  useKeyboard(key => {
    if (key.eventType === 'release') return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape' || key.name === 'b') onClose()
    else if (key.name === 'r') void load(retry.current ?? cursors)
    else if (key.name === 'n' && page && (page.hasMore || page.running)) void load([...cursors, page.cursor])
    else if (key.name === 'p' && cursors.length > 1) void load(cursors.slice(0, -1))
    else if (key.name === 'home' || key.name === 'end') scroll.current?.scrollTo(key.name === 'home' ? 0 : Number.MAX_SAFE_INTEGER)
    else if (key.name === 'up' || key.name === 'down') scroll.current?.scrollBy(key.name === 'up' ? -1 : 1)
    else if (key.name === 'left' || key.name === 'right') scroll.current?.scrollBy({ x: key.name === 'left' ? -12 : 12, y: 0 })
    else if (key.name === 'pageup' || key.name === 'pagedown') scroll.current?.scrollBy(key.name === 'pageup' ? -8 : 8)
  })
  return <Box flexGrow={1} minHeight={0} flexDirection="column">
    <DialogHeader t={t} title={`Output · Page ${cursors.length}`} subtitle="Browse retained output without consuming it." />
    {loading ? <Text color={t.ds.secondary}>Reading output…</Text> : null}
    {error ? <Text color={t.color.warn} wrap="wrap">{error} · R retry</Text> : null}
    {page?.droppedChars ? <Text color={t.color.warn} wrap="wrap">{page.droppedChars} earlier characters were dropped by retention.</Text> : null}
    <scrollbox ref={scroll} scrollX viewportCulling style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}>
      <Box flexDirection="column" width={outputWidth} flexShrink={0}>{lines.map((line, index) => <Text key={index} color={t.color.text} wrap="truncate-end">{line || ' '}</Text>)}</Box>
    </scrollbox>
    <Text color={t.ds.secondary}>{page?.running ? 'Still running · N read more' : page?.hasMore ? 'More retained output · N next' : 'End of retained output'}</Text>
    <DialogFooter t={t}><Text color={t.ds.secondary}>N/P pages · R retry · Esc back</Text><Text color={t.ds.secondary}>↑↓ / PgUp/PgDn · ←→ pan</Text></DialogFooter>
  </Box>
}
