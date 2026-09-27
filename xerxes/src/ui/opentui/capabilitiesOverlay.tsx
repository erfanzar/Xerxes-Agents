// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useStore } from '@nanostores/react'
import { $uiState } from '../app/uiStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'

type Entry = { name: string; description: string; uses: number; source?: string; tags?: string[]; platform_supported?: boolean; exposure?: string }
type Catalog = { ok: boolean; skills: Entry[]; tools: Entry[]; total_skills?: number; tools_source?: string; error?: string; model?: string; provider_profile?: string; reasoning_effort?: string }
export function toolGroup(name: string): string {
  if (/mcp/i.test(name)) return 'MCP integrations'
  if (/skill|memory/i.test(name)) return 'Skills & memory'
  if (/agent|spawn|delegate|taskget|tasklist|await/i.test(name)) return 'Agents & delegation'
  if (/shell|exec|terminal|process|command/i.test(name)) return 'Terminal & processes'
  if (/file|read|write|edit|patch|grep|glob|dir/i.test(name)) return 'Files & code'
  if (/browser|web|search|fetch/i.test(name)) return 'Web & browser'
  if (/goal|todo|schedule|cron|monitor|watch/i.test(name)) return 'Planning & background work'
  return 'Other tools'
}
const destinations = [
  { name: 'Models & provider profiles', detail: 'Choose the provider profile first, then a model. /model', key: 'modelPicker' },
  { name: 'Reasoning effort', detail: 'Choose the effort supported by the current model. /reasoning', key: 'reasoningPicker' },
  { name: 'MCP connections', detail: 'Inspect connected servers, tools and configuration. /mcp', key: 'mcpSettings' },
  { name: 'Schedules & templates', detail: 'Start from a task example, preview timing and choose delivery. /schedules', key: 'schedules' },
  { name: 'Custom agents', detail: 'Inspect, edit or generate a specialist from a description. /custom-agents', key: 'customAgentEditor' },
  { name: 'Remote workspaces', detail: 'Browse SSH hosts and remote project folders. /machine', key: 'machinePicker' },
] as const

export function CapabilitiesOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const ui = useStore($uiState)
  const terminal = useTerminalDimensions()
  const size = overlayPanelSize(terminal, { maxWidth: 140, maxHeight: 42, minWidth: 32 })
  const [tab, setTab] = useState(0)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const [popular, setPopular] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState('')
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => {
    let current = true
    setCatalog(null); setError('')
    if (!gateway) { setError('Connect to a daemon to inspect capabilities.'); return }
    void gateway.rpc<Catalog>('capabilities.list', {}).then(result => {
      if (!current) return
      if (!result?.ok || !Array.isArray(result.skills) || !Array.isArray(result.tools)) throw new Error(result?.error || 'Invalid capability catalog')
      if (![...result.skills, ...result.tools].every(row => typeof row?.name === 'string' && typeof row.uses === 'number')) throw new Error('Invalid capability record')
      setCatalog(result)
    }).catch(failure => { if (current) setError(String(failure)) })
    return () => { current = false }
  }, [gateway, ui.sid, refresh])
  const source: Entry[] = tab === 0 ? catalog?.skills ?? [] : tab === 1 ? catalog?.tools ?? [] : destinations.map(row => ({ name: row.name, description: row.detail, uses: 0 }))
  const rows = source.filter(row => `${row.name} ${row.description} ${tab === 1 ? toolGroup(row.name) : row.tags?.join(' ') ?? ''}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => (tab === 1 ? toolGroup(a.name).localeCompare(toolGroup(b.name)) : 0) || (popular ? b.uses - a.uses : 0) || a.name.localeCompare(b.name))
  const index = Math.min(selected, Math.max(0, rows.length - 1))
  const row = rows[index]
  useEffect(() => {
    let current = true
    scroll.current?.scrollTo(0)
    setPreview('')
    if (tab !== 0 || !row || !gateway) return
    setPreview('Loading instructions…')
    void gateway.rpc<{ ok: boolean; instructions?: string; truncated?: boolean; error?: string }>('capabilities.inspect', { name: row.name }).then(result => {
      if (current) setPreview(result?.ok ? `${result.instructions ?? ''}${result.truncated ? '\n[Preview shortened; open the source for full instructions.]' : ''}` : result?.error || 'Preview unavailable')
    }).catch(failure => { if (current) setPreview(String(failure)) })
    return () => { current = false }
  }, [tab, row?.name, gateway, refresh, ui.sid])
  const changeTab = (value: number) => { setTab(value); setQuery(''); setSelected(0) }
  const open = () => {
    if (tab !== 2 || !row) return
    const destination = destinations.find(value => value.name === row.name)
    if (destination) patchOverlayState({ capabilities: false, [destination.key]: true })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'escape') { key.preventDefault(); key.stopPropagation(); patchOverlayState({ capabilities: false }) }
    else if (key.name === 'tab') { key.preventDefault(); key.stopPropagation(); changeTab((tab + (key.shift ? 2 : 1)) % 3) }
    else if (key.name === 'up' || key.name === 'down') { key.preventDefault(); key.stopPropagation(); setSelected(Math.max(0, Math.min(rows.length - 1, index + (key.name === 'up' ? -1 : 1)))) }
    else if (key.name === 'pageup' || key.name === 'pagedown') { key.preventDefault(); key.stopPropagation(); scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * 8) }
    else if (key.ctrl && key.name === 'r') { key.preventDefault(); key.stopPropagation(); setRefresh(value => value + 1) }
    else if (key.ctrl && key.name === 's') { key.preventDefault(); key.stopPropagation(); setPopular(value => !value) }
    else if (key.name === 'return') { key.preventDefault(); key.stopPropagation(); open() }
  })
  const wide = size.width >= 95
  const count = wide ? Math.max(1, Math.floor((size.height - 15) / 2)) : 2
  const start = Math.max(0, index - count + 1)
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={180} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} paddingX={1} flexDirection="column" borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      <DialogHeader t={t} title="Capabilities" subtitle="Discover expertise, inspect tools and configure your workspace." />
      <Box flexDirection="row" gap={2} flexShrink={0}>{['Skills', 'Tools', 'Controls'].map((label, id) => <Box key={label} onMouseDown={() => changeTab(id)}><Text bold color={tab === id ? t.color.accent : t.ds.secondary}>{label}{id < 2 ? ` ${id === 0 ? catalog?.skills.length ?? '…' : catalog?.tools.length ?? '…'}` : ''}</Text></Box>)}</Box>
      <input value={query} onInput={value => { setQuery(value); setSelected(0) }} focused placeholder="Search names, descriptions or groups…" backgroundColor={t.color.completionMetaBg} textColor={t.color.text} />
      <Text color={t.ds.meta} wrap={wide ? "wrap" : "truncate-end"}>{tab === 2 ? `${catalog?.model || ui.info?.model || 'Model unavailable'} · profile: ${catalog?.provider_profile || 'not reported'} · reasoning: ${catalog?.reasoning_effort || ui.info?.reasoning_effort || 'not reported'}` : `${rows.length} matches · ${popular ? 'Most used' : 'Name order'} · counts: retained session history`}</Text>
      {error ? <Text color={t.color.warn} wrap="wrap">{error} · Ctrl+R retry</Text> : null}
      {tab === 0 && (catalog?.total_skills ?? 0) > (catalog?.skills.length ?? 0) ? <Text color={t.ds.meta}>Showing the first {catalog?.skills.length} of {catalog?.total_skills} skills.</Text> : null}
      {tab === 1 && catalog?.tools_source === 'unavailable' ? <Text color={t.color.warn}>Tool inventory unavailable for this session.</Text> : null}
      {!catalog && !error ? <Text color={t.ds.secondary}>Loading capabilities…</Text> : null}
      <Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0} gap={1}>
        <Box width={wide ? '38%' : '100%'} flexDirection="column" flexShrink={0}>
          {!rows.length && catalog ? <Text color={t.ds.secondary}>No matching capabilities.</Text> : null}
          {rows.slice(start, start + count).map((item, offset) => <Box key={item.name} flexDirection="column" flexShrink={0} backgroundColor={index === start + offset ? t.color.completionCurrentBg : undefined} paddingX={1} onMouseDown={() => setSelected(start + offset)}>
            <Text color={index === start + offset ? t.color.accent : t.color.text} wrap="truncate-end">{index === start + offset ? '› ' : '  '}{item.name}{tab < 2 ? ` · ${item.uses}` : ''}</Text>
            {wide ? <Text color={t.ds.meta} wrap="truncate-end">{tab === 1 ? toolGroup(item.name) : item.tags?.join(' · ') || item.description}</Text> : null}
          </Box>)}
        </Box>
        <scrollbox ref={scroll} flexGrow={1} minHeight={0} contentOptions={{ flexDirection: 'column' }}>
          {row ? <><Text bold color={t.color.accent} wrap="wrap">{row.name}</Text><Text wrap="wrap">{row.description || 'No description supplied.'}</Text>
            {tab === 0 ? <><Text color={t.ds.meta} wrap="wrap">{row.source}</Text><Text color={t.ds.secondary} wrap="wrap">{row.platform_supported === false ? 'Unsupported on this host' : 'Discovered · tool readiness not checked'} · /skill {row.name}</Text><Text wrap="wrap">{preview}</Text></> : tab === 1 ? <><Text color={t.ds.secondary} wrap="wrap">{toolGroup(row.name)} · {row.exposure || 'Registered'} · {row.uses} calls in retained history</Text><Text color={t.ds.meta} wrap="wrap">Registration does not imply permission or connection readiness.</Text></> : <Text color={t.color.accent}>Enter to open</Text>}</> : null}
        </scrollbox>
      </Box>
      <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">{wide ? 'Tab category · ↑↓ select · PgUp/PgDn details · Ctrl+S sort · Ctrl+R refresh · Esc close' : '↑↓ select · Tab tabs · Esc close\n^R refresh · ^S sort · PgUp/Dn'}</Text></DialogFooter>
    </Box>
  </box>
}
