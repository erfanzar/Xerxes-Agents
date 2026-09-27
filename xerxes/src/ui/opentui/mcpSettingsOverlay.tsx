// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable, TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, SettingRow, DialogEmpty } from './dialogChrome.js'

interface Server { name: string; enabled: boolean; transport: string; timeout_ms: number | null; configured_fields: string[] }
interface Settings { ok: boolean; revision: string; servers: Server[]; source?: string; error?: string }
const fields = ['enabled', 'transport', 'command', 'args', 'url', 'timeoutMs', 'env', 'headers'] as const
const labels = ['Enabled', 'Transport', 'Command', 'Arguments (JSON array)', 'URL', 'Timeout (milliseconds)', 'Environment (JSON object)', 'Headers (JSON object)']
export function McpSettingsOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const compact = terminal.height < 26
  const size = overlayPanelSize(terminal, { maxWidth: 108, minWidth: 32, maxHeight: 36 })
  const [settings, setSettings] = useState<Settings | null>(null)
  const [selected, setSelected] = useState(0)
  const [field, setField] = useState(0)
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [editing, setEditing] = useState(false)
  const [creating, setCreating] = useState(false)
  const [naming, setNaming] = useState(false)
  const [newName, setNewName] = useState('')
  const [message, setMessage] = useState('Loading…')
  const [busy, setBusy] = useState(false)
  const fieldScroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => {
    const timer = setTimeout(() => fieldScroll.current?.scrollChildIntoView(editing ? 'server-edit' : 'server-field-' + field), 0)
    return () => clearTimeout(timer)
  }, [field, editing, selected])
  const input = useRef<TextareaRenderable | null>(null)
  const alive = useRef(true), pending = useRef(false)
  const server: Server | undefined = creating ? { name: newName, enabled: true, transport: 'stdio', timeout_ms: null, configured_fields: [] } : settings?.servers[selected]
  const key = fields[field]!
  const load = async () => {
    if (!gateway || pending.current) return
    pending.current = true; setBusy(true)
    try {
      const result = await gateway.rpc<Settings>('mcp.settings.get', {})
      if (!result?.ok) throw new Error(result?.error || 'Could not load MCP settings')
      if (alive.current) { setSettings(result); setSelected(0); setDraft({}); setCreating(false); setNaming(false); setEditing(false); setMessage('') }
    } catch (error) { if (alive.current) setMessage(String(error)) }
    finally { pending.current = false; if (alive.current) setBusy(false) }
  }
  useEffect(() => { alive.current = true; if (gateway) void load(); else setMessage('MCP gateway unavailable'); return () => { alive.current = false } }, [gateway])
  const acceptInput = () => {
    const value = input.current?.plainText ?? ''
    if (naming) {
      if (!value.trim() || settings?.servers.some(server => server.name === value.trim())) { setMessage('Enter a unique server name'); return }
      setNewName(value.trim()); setCreating(true); setNaming(false); setEditing(false); setDraft({}); setField(2); setMessage('Enter a command, or change transport and enter a URL. Ctrl+S connects and saves.'); return
    }
    try {
      let parsed: unknown = value
      if (value === '') { setDraft(current => { const next = { ...current }; delete next[key]; return next }); setEditing(false); return }
      if (['args', 'env', 'headers'].includes(key)) parsed = JSON.parse(value)
      if (key === 'timeoutMs') { parsed = Number(value); if (!Number.isFinite(parsed)) throw new Error('Enter a number') }
      setDraft(current => ({ ...current, [key]: parsed })); setEditing(false); setMessage('Draft updated · Ctrl+S to save')
    } catch { setMessage('Invalid value. Arguments need a JSON array; environment and headers need JSON objects.') }
  }
  const save = async () => {
    if (!gateway || !settings || !server || pending.current || editing || !Object.keys(draft).length) return
    pending.current = true; setBusy(true); setMessage('Connecting and saving…')
    try {
      const result = await gateway.rpc<{ ok: boolean; revision: string; warnings?: string[]; error?: string }>('mcp.settings.save', { name: server.name, revision: settings.revision, changes: draft, ...(creating ? { create: true } : {}) })
      if (!result?.ok) throw new Error(result?.error || 'Could not save MCP settings')
      if (alive.current) {
        const updatedServers = creating ? [...settings.servers, server] : settings.servers
        const updatedIndex = creating ? updatedServers.length - 1 : selected
        setSettings({ ...settings, revision: result.revision, servers: updatedServers.map((value, i) => i !== updatedIndex ? value : {
          ...value, enabled: 'enabled' in draft ? draft.enabled !== false : value.enabled,
          transport: 'transport' in draft ? String(draft.transport ?? 'stdio') : value.transport,
          timeout_ms: 'timeoutMs' in draft ? typeof draft.timeoutMs === 'number' ? draft.timeoutMs : null : value.timeout_ms,
        }) });
        setSelected(updatedIndex); setCreating(false)
        setDraft({}); setMessage(result.warnings?.length ? result.warnings.join(' · ') : 'Saved. Connected tools refreshed.')
      }
    } catch (error) { if (alive.current) setMessage(`${String(error)} · Draft kept. F5 reload discards it.`) }
    finally { pending.current = false; if (alive.current) setBusy(false) }
  }
  const beginNew = () => {
    if (pending.current) return
    if (!settings || Object.keys(draft).length || creating) { setMessage('Save or reload the current draft first.'); return }
      setNaming(true); setEditing(true); setMessage('Name the new MCP server'); return
  }
  useKeyboard(event => {
    if (editing && !['escape', 'return'].includes(event.name)) return
    if (!['escape', 'return', 'tab', 'up', 'down', 'f2', 'f5', 's', 'delete'].includes(event.name)) return
    if (event.name === 's' && !event.ctrl) return
    event.preventDefault(); event.stopPropagation()
    if (pending.current) return
    if (event.name === 'escape') { if (editing) { setEditing(false); setNaming(false) } else patchOverlayState({ mcpSettings: false }); return }
    if (event.name === 'f5') { void load(); return }
    if (event.name === 'f2') { beginNew(); return }
    if (editing) { acceptInput(); return }
    if (!server) return
    if (event.name === 's') { void save(); return }
    if (event.name === 'tab') { setField(value => (value + (event.shift ? fields.length - 1 : 1)) % fields.length); return }
    if (event.name === 'up' || event.name === 'down') {
      if (Object.keys(draft).length || creating) { setMessage('Save this draft or press F5 to discard it before switching servers.'); return }
      setSelected(value => (value + (event.name === 'up' ? settings!.servers.length - 1 : 1)) % settings!.servers.length); return
    }
    if (event.name === 'delete') { setDraft(value => ({ ...value, [key]: null })); setMessage('Field will be removed on save'); return }
    if (key === 'enabled') setDraft(value => ({ ...value, enabled: !(typeof value.enabled === 'boolean' ? value.enabled : server.enabled) }))
    else if (key === 'transport') setDraft(value => ({ ...value, transport: ['stdio', 'sse', 'streamable_http'][(['stdio', 'sse', 'streamable_http'].indexOf(String(value.transport ?? server.transport)) + 1) % 3] }))
    else setEditing(true)
  })
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!server && !editing ? Math.min(88, size.width) : size.width} height={!server && !editing ? Math.min(24, size.height) : size.height} backgroundColor={t.color.overlayBg} borderStyle="round" borderColor={t.color.border} padding={1} flexDirection="column">
      <DialogHeader t={t} title="Configuration · MCP servers" subtitle="Connect tools and services to your agent." />
      {server || editing ? <Text wrap="wrap">{naming ? 'New server · enter name' : server ? `${creating ? 'New' : `${selected + 1}/${settings!.servers.length}`} · ${server.name}` : 'User settings'}</Text> : null}
      {!server && !editing ? <DialogEmpty t={t} title="No MCP servers configured" description="Add a server to make its tools available." action="F2  Add server" onAction={beginNew} /> : <scrollbox ref={fieldScroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
        {!compact ? <Text wrap="wrap" color={t.ds.secondary}>Existing launch and credential values stay hidden. Blank input keeps the saved value. Delete removes a field.</Text> : null}
        {server ? fields.map((name, index) => compact && index !== field ? null : <Box key={name} id={`server-field-${index}`} flexDirection="column" flexShrink={0}><SettingRow t={t} label={labels[index]!} selected={field === index}>
          {name in draft ? draft[name] === null ? '(remove)' : ['enabled', 'transport', 'timeoutMs'].includes(name) ? String(draft[name]) : '(replacement entered)' : name === 'enabled' ? String(server.enabled) : name === 'transport' ? server.transport : name === 'timeoutMs' ? String(server.timeout_ms ?? 'default') : '(keep saved value)'}
        </SettingRow>{editing && !naming && index === field ? <textarea id="server-edit" key={`${selected}-${field}`} ref={input} focused minHeight={2} maxHeight={5} /> : null}</Box>) : !editing ? <DialogEmpty t={t} title="No MCP servers configured" description="Add a server to make its tools available." action="F2  Add server" onAction={beginNew} /> : null}
        {editing && naming ? <textarea id="server-edit" key={`${naming}-${selected}-${field}`} ref={input} focused minHeight={2} maxHeight={5} /> : null}
      </scrollbox>}
      <Text wrap="wrap" color={t.color.warn}>{message}</Text>
      <DialogFooter t={t}><Text wrap="wrap">{busy ? 'Saving/loading… Please wait.' : !server && !editing ? 'F2 add server · F5 reload · Esc close' : editing ? 'Enter apply draft · Esc cancel edit' : '↑/↓ server · F2 new · Tab field · Enter edit · Ctrl+S save · F5 reload · Esc close'}</Text></DialogFooter>
    </Box>
  </box>
}
