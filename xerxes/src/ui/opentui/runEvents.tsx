// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { readRunEvents, type RunEventPage } from '../lib/runEvents.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'

export function RunEvents({ t, runId, title, scope, onClose }: {
  t: Theme; runId: string; title: string; scope: 'session' | 'workspace'; onClose: () => void
}) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const [page, setPage] = useState<RunEventPage | null>(null)
  const [cursors, setCursors] = useState<number[]>([0])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const generation = useRef(0)
  const busy = useRef(false)
  const retryPage = useRef<number[] | null>(null)
  const load = async (next: number[]) => {
    if (!gateway || busy.current) return
    const request = ++generation.current
    busy.current = true; setLoading(true); setError('')
    retryPage.current = next
    try {
      const result = await readRunEvents(gateway.rpc, runId, scope, next.at(-1)!)
      if (request !== generation.current) return
      const changed = next.at(-1) !== cursors.at(-1)
      setPage(result); setCursors(next)
      retryPage.current = null
      if (changed) scroll.current?.scrollTo(0)
    } catch (failure) {
      if (request === generation.current) setError(String(failure))
    } finally {
      if (request === generation.current) { busy.current = false; setLoading(false) }
    }
  }
  useEffect(() => {
    busy.current = false; retryPage.current = null; setPage(null); setCursors([0])
    if (!gateway) setError('Connect to a daemon to read run events.')
    else void load([0])
    return () => { generation.current++ }
  }, [gateway, runId, scope])
  const next = () => { if (page?.hasMore) void load([...cursors, page.nextCursor]) }
  const previous = () => { if (cursors.length > 1) void load(cursors.slice(0, -1)) }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape' || key.name === 'b') { onClose(); return }
    if (key.name === 'n') next()
    else if (key.name === 'p') previous()
    else if (key.name === 'r') void load(retryPage.current ?? cursors)
    else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else if (key.name === 'up' || key.name === 'down') scroll.current?.scrollBy(key.name === 'up' ? -1 : 1)
    else if (key.name === 'pageup' || key.name === 'pagedown') scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, terminal.height - 10))
  })
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <DialogHeader t={t} title={`Events · Page ${cursors.length}`} subtitle={title} />
    <Text color={t.ds.secondary} wrap="wrap">Saved evidence · {scope} scope · reading does not acknowledge this run</Text>
    {loading ? <Text color={t.ds.secondary}>Loading events…</Text> : null}
    {error ? <Text color={t.color.warn} wrap="wrap">{error} · R retry</Text> : null}
    <scrollbox ref={scroll} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      {page?.events.map(event => <Box key={event.sequence} flexDirection="column" flexShrink={0} marginBottom={1}>
        <Text color={t.color.accent} wrap="wrap">Event {event.sequence} · {new Date(event.at).toLocaleString()}</Text>
        <Text color={t.color.text} wrap="wrap">{event.text || '(empty event)'}</Text>
      </Box>)}
      {page && !page.events.length ? <DialogEmpty t={t} title="No saved events." description="This run may have output without a separate event history." symbol="≡" /> : null}
    </scrollbox>
    <Box flexDirection="row" gap={2} flexShrink={0}>
      <Box onMouseDown={previous}><Text color={cursors.length > 1 ? t.color.accent : t.ds.secondary}>P Previous</Text></Box>
      <Box onMouseDown={next}><Text color={page?.hasMore ? t.color.accent : t.ds.secondary}>{page?.hasMore ? 'N Next' : 'End of saved events'}</Text></Box>
    </Box>
    <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">R refresh · Esc back</Text><Text color={t.ds.secondary}>↑↓ / PgUp/PgDn scroll</Text></DialogFooter>
  </Box>
}
