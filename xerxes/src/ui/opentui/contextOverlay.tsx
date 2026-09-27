// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useEffect, useRef, useState } from 'react'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogSection } from './dialogChrome.js'
import type { Theme } from '../theme.js'

const sections = ['instructions', 'memory', 'conversation', 'tools', 'compaction'] as const
interface ContextPage {
  controls_revision?: number
  generation: string; note: string; section: string; offset: number; next_offset: number | null
  sections: Array<{ id: string; count: number; available: boolean; estimated_tokens: number; provenance: string }>
  entries: Array<{ index: number; title: string; text: string; truncated: boolean; estimated_tokens: number;
    control?: { scope: string; path: string; pinned: boolean; excluded: boolean } }>
}
function parse(value: unknown): ContextPage {
  if (!value || typeof value !== 'object') throw new Error('Invalid context response')
  const row = value as Record<string, unknown>
  if (row.ok !== true) throw new Error(typeof row.error === 'string' ? row.error : 'Context unavailable')
  if (typeof row.generation !== 'string' || typeof row.note !== 'string' || !sections.includes(row.section as typeof sections[number])
    || typeof row.offset !== 'number' || !Number.isSafeInteger(row.offset) || row.offset < 0
    || (row.next_offset !== null && (typeof row.next_offset !== 'number' || !Number.isSafeInteger(row.next_offset) || row.next_offset <= row.offset))
    || !Array.isArray(row.sections) || row.sections.length !== sections.length || !Array.isArray(row.entries) || row.entries.length > 20) throw new Error('Invalid context response')
  for (const item of row.sections) if (!item || typeof item !== 'object' || typeof item.id !== 'string' || typeof item.available !== 'boolean' || typeof item.count !== 'number' || typeof item.estimated_tokens !== 'number' || typeof item.provenance !== 'string') throw new Error('Invalid context section')
  for (const item of row.entries) if (!item || typeof item !== 'object' || typeof item.title !== 'string' || typeof item.text !== 'string' || typeof item.truncated !== 'boolean' || typeof item.estimated_tokens !== 'number') throw new Error('Invalid context entry')
  for (const item of row.entries) if (item.control !== undefined) {
    const control = item.control
    if (!control || typeof control !== 'object' || !['global', 'project'].includes(control.scope) || typeof control.path !== 'string' || typeof control.pinned !== 'boolean' || typeof control.excluded !== 'boolean'
      || typeof row.controls_revision !== 'number' || !Number.isSafeInteger(row.controls_revision) || row.controls_revision < 0) throw new Error('Invalid context control')
  }
  return row as unknown as ContextPage
}
export function ContextOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 128, minWidth: 32, maxHeight: 42 })
  const [section, setSection] = useState(0)
  const [offset, setOffset] = useState(0)
  const [refresh, setRefresh] = useState(0)
  const [page, setPage] = useState<ContextPage | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(0)
  const [notice, setNotice] = useState('')
  const mutating = useRef(false)
  const generation = useRef<string | undefined>(undefined)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => {
    let current = true
    setLoading(true); setError('')
    if (!gateway) { setError('Connect to a daemon to inspect context.'); setLoading(false); return }
    void gateway.rpc<Record<string, unknown>>('context.inspect', { section: sections[section], offset, ...(generation.current ? { generation: generation.current } : {}) })
      .then(value => {
        if (!current) return
        const next = parse(value)
        if (next.section !== sections[section] || next.offset !== offset) throw new Error('Context response does not match the requested page; refresh the inspector')
        generation.current = next.generation; setPage(next); setSelected(0); scroll.current?.scrollTo(0)
      })
      .catch(failure => { if (current) setError(failure instanceof Error ? failure.message : String(failure)) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [gateway, section, offset, refresh])
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (!['escape', 'tab', 'left', 'right', 'n', 'p', 'r', 'j', 'k', 'i', 'x', 'up', 'down', 'pageup', 'pagedown', 'home', 'end'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') patchOverlayState({ contextInspector: false })
    else if (mutating.current) return
    else if (['j', 'k'].includes(key.name) && page?.section === 'memory') setSelected(value => Math.max(0, Math.min(page.entries.length - 1, value + (key.name === 'j' ? 1 : -1))))
    else if (['i', 'x'].includes(key.name) && !loading && !error && page?.section === 'memory' && sections[section] === 'memory') {
      const control = page.entries[selected]?.control
      if (!control || !gateway) return
      mutating.current = true; setNotice('Saving…')
      const action = key.name === 'i' ? control.pinned ? 'unpin' : 'pin' : control.excluded ? 'include' : 'exclude'
      void gateway.rpc<Record<string, unknown>>('context.control', { action, scope: control.scope, path: control.path, revision: page.controls_revision, generation: page.generation })
        .then(result => {
          if (result?.ok !== true) throw new Error(typeof result?.error === 'string' ? result.error : 'Context control failed')
          setNotice('Saved · applies on the next turn'); generation.current = undefined; setRefresh(value => value + 1)
        }).catch(failure => { setNotice(''); setError(failure instanceof Error ? failure.message : String(failure)) })
        .finally(() => { mutating.current = false })
    }
    else if (key.name === 'r') { generation.current = undefined; setOffset(0); setRefresh(value => value + 1) }
    else if (key.name === 'tab' || key.name === 'left' || key.name === 'right') { setSection(value => (value + (key.name === 'left' || key.shift ? sections.length - 1 : 1)) % sections.length); setOffset(0) }
    else if (!loading && !error && page?.section === sections[section] && page.offset === offset && key.name === 'n' && page.next_offset !== null) setOffset(page.next_offset)
    else if (!loading && !error && page?.section === sections[section] && page.offset === offset && key.name === 'p') setOffset(Math.max(0, page.offset - 20))
    else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    else if (['up', 'down', 'pageup', 'pagedown'].includes(key.name)) scroll.current?.scrollBy((key.name === 'up' || key.name === 'pageup' ? -1 : 1) * (key.name.startsWith('page') ? Math.max(1, size.height - 8) : 1))
  })
  const summary = page?.sections.find(item => item.id === page.section)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      <DialogHeader t={t} title={<>Context · {loading ? 'loading…' : page?.section ?? sections[section]}</>} subtitle="Understand what the agent sees. Inspect sources and memory." />
      <Text color={t.ds.secondary} wrap="wrap">{summary ? summary.available ? `${summary.count} entries · ${page?.section === 'compaction' ? 'not in model context' : '~' + summary.estimated_tokens + ' tokens'} · ${summary.provenance}` : 'Not assembled yet' : 'Read-only inspection'}</Text>
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {notice ? <Text color={t.ds.secondary} wrap="wrap">{notice}</Text> : null}
      <scrollbox ref={scroll} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
        <Text color={t.ds.secondary} wrap="wrap">{page?.note ?? 'Local estimates; no provider request is made.'}</Text>
        {page?.entries.map((entry, index) => <Box key={entry.index} flexDirection="column" flexShrink={0} marginTop={1}>
          <DialogSection t={t}><span>{entry.control && index === selected ? '› ' : ''}{entry.title}{entry.control ? entry.control.pinned ? ' · pinned' : entry.control.excluded ? ' · excluded' : ' · optional source' : page.section === 'compaction' ? '' : ' · ~' + entry.estimated_tokens + ' tokens'}</span></DialogSection>
          {!entry.control || index === selected ? <Text color={t.color.text} wrap="wrap">{entry.text}</Text> : null}
          {entry.truncated && (!entry.control || index === selected) ? <Text color={t.color.warn}>Excerpt truncated at 8000 characters</Text> : null}
        </Box>)}
      </scrollbox>
      <DialogFooter t={t}><Text color={t.ds.secondary}>Tab section · N/P page · R refresh</Text>
      {page?.section === 'memory' ? <Text color={t.ds.secondary} wrap="wrap">J/K select · I pin/unpin · X exclude/include</Text> : null}
      {page?.section === 'conversation' ? <Text color={t.ds.secondary} wrap="wrap">Branch after closing: /branch --through-turn N</Text> : null}
      <Text color={t.ds.secondary}>↑↓ scroll · Esc close</Text></DialogFooter>
    </Box>
  </box>
}
