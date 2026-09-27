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

interface Server { name: string; enabled: boolean; languageId: string; extensions: string[]; timeoutMs: number; configuredFields: string[] }
interface Settings { ok: boolean; revision: string; servers: Server[]; source?: string; error?: string }
const fields = ['enabled', 'languageId', 'extensions', 'command', 'args', 'timeoutMs', 'env'] as const
const labels = ['Enabled', 'Language ID', 'File suffixes (JSON array)', 'Command', 'Arguments (JSON array)', 'Timeout (milliseconds)', 'Environment (JSON object)']
export function LspSettingsOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 108, minWidth: 32, maxHeight: 36 })
  const compact = size.width < 60 || size.height < 24
  const [settings, setSettings] = useState<Settings | null>(null)
  const [selected, setSelected] = useState(0)
  const [field, setField] = useState(0)
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [editing, setEditing] = useState(false)
  const [removing, setRemoving] = useState(false)
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
  const server: Server | undefined = creating ? { name: newName, enabled: true, languageId: '', extensions: [], timeoutMs: 30000, configuredFields: [] } : settings?.servers[selected]
  const key = fields[field]!
  const load = async () => {
    if (!gateway || pending.current) return
    pending.current = true; setBusy(true)
    try {
      const result = await gateway.rpc<Settings>('lsp.settings.get', {})
      if (!result?.ok) throw new Error(result?.error || 'Could not load LSP settings')
      if (alive.current) { setSettings(result); setSelected(0); setDraft({}); setCreating(false); setRemoving(false); setNaming(false); setEditing(false); setMessage('') }
    } catch (error) { if (alive.current) setMessage(String(error)) }
    finally { pending.current = false; if (alive.current) setBusy(false) }
  }
  useEffect(() => { alive.current = true; if (gateway) void load(); else setMessage('LSP gateway unavailable'); return () => { alive.current = false } }, [gateway])
  const acceptInput = () => {
    const value = input.current?.plainText ?? ''
    if (naming) {
      if (!value.trim() || settings?.servers.some(server => server.name === value.trim())) { setMessage('Enter a unique server name'); return }
      setNewName(value.trim()); setCreating(true); setNaming(false); setEditing(false); setDraft({}); setField(1); setMessage('Enter language ID, file suffixes and command. Ctrl+S saves and applies.'); return
    }
    try {
      let parsed: unknown = value
      if (value === '') { setDraft(current => { const next = { ...current }; delete next[key]; return next }); setEditing(false); return }
      if (['args', 'env', 'extensions'].includes(key)) parsed = JSON.parse(value)
      if (key === 'timeoutMs') { parsed = Number(value); if (!Number.isFinite(parsed)) throw new Error('Enter a number') }
      setDraft(current => ({ ...current, [key]: parsed })); setEditing(false); setMessage('Draft updated · Ctrl+S to save')
    } catch { setMessage('Invalid value. Arguments and suffixes need JSON arrays; environment needs a JSON object.') }
  }
  const save = async (remove = false) => {
    if (!gateway || !settings || !server || pending.current || editing || (!remove && !Object.keys(draft).length)) return
    pending.current = true; setBusy(true); setRemoving(false); setMessage('Applying settings…')
    try {
      const result = await gateway.rpc<Settings & { warnings?: string[] }>('lsp.settings.save', { name: server.name, revision: settings.revision, action: remove ? 'remove' : creating ? 'create' : 'update', ...(!remove ? { changes: draft } : {}) })
      if (!result?.ok) throw new Error(result?.error || 'Could not save LSP settings')
      if (alive.current) {
        setSettings(result); setSelected(Math.max(0, result.servers.findIndex(row => row.name === server.name))); setCreating(false)
        setDraft({}); setMessage(result.warnings?.length ? result.warnings.join(' · ') : 'Saved. Settings applied; tools refreshed.')
      }
    } catch (error) { if (alive.current) setMessage(`${String(error)} · Draft kept. F5 reload discards it.`) }
    finally { pending.current = false; if (alive.current) setBusy(false) }
  }
  const beginNew = () => {
    if (pending.current) return
    if (!settings || Object.keys(draft).length || creating) { setMessage('Save or reload the current draft first.'); return }
      setNaming(true); setEditing(true); setMessage('Name the new LSP server'); return
  }
  useKeyboard(event => {
    if (removing) {
      event.preventDefault(); event.stopPropagation()
      if (pending.current) return
      if (event.name === 'y') void save(true)
      else if (event.name === 'n' || event.name === 'escape') setRemoving(false)
      return
    }
    if (editing && !['escape', 'return'].includes(event.name)) return
    if (!['escape', 'return', 'tab', 'up', 'down', 'f2', 'f4', 'f5', 's', 'delete'].includes(event.name)) return
    if (event.name === 's' && !event.ctrl) return
    event.preventDefault(); event.stopPropagation()
    if (pending.current) return
    if (event.name === 'escape') { if (editing) { setEditing(false); setNaming(false) } else patchOverlayState({ lspSettings: false }); return }
    if (event.name === 'f5') { void load(); return }
    if (event.name === 'f2') { beginNew(); return }
    if (editing) { acceptInput(); return }
    if (!server) return
    if (event.name === 'f4') { if (creating || Object.keys(draft).length) setMessage('Save or reload the draft first.'); else { setRemoving(true); setMessage('Remove this server from user settings? Y confirms · N cancels'); } return }
    if (event.name === 's') { void save(); return }
    if (event.name === 'tab') { setField(value => (value + (event.shift ? fields.length - 1 : 1)) % fields.length); return }
    if (event.name === 'up' || event.name === 'down') {
      if (Object.keys(draft).length || creating) { setMessage('Save this draft or press F5 to discard it before switching servers.'); return }
      setSelected(value => (value + (event.name === 'up' ? settings!.servers.length - 1 : 1)) % settings!.servers.length); return
    }
    if (event.name === 'delete') { if (!['args', 'env', 'timeoutMs'].includes(key)) { setMessage('This field is required; edit its value.'); return } setDraft(value => ({ ...value, [key]: null })); setMessage('Field will be removed on save'); return }
    if (key === 'enabled') setDraft(value => ({ ...value, enabled: !(typeof value.enabled === 'boolean' ? value.enabled : server.enabled) }))
    else setEditing(true)
  })
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!server && !editing ? Math.min(88, size.width) : size.width} height={!server && !editing ? Math.min(24, size.height) : size.height} backgroundColor={t.color.overlayBg} borderStyle="round" borderColor={t.color.border} padding={1} flexDirection="column">
      <DialogHeader t={t} title="Configuration · LSP servers" subtitle="Language intelligence for your project." />
      {server || editing ? <Text wrap="wrap">{naming ? 'New server · enter name' : server ? `${creating ? 'New' : `${selected + 1}/${settings!.servers.length}`} · ${server.name}` : 'User settings'}</Text> : null}
      {!server && !editing ? <DialogEmpty t={t} title="No LSP servers configured" description="Add a language server for diagnostics and navigation." action="F2  Add server" onAction={beginNew} /> : <scrollbox ref={fieldScroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
        {!compact ? <Text wrap="wrap" color={t.ds.secondary}>Existing launch and credential values stay hidden. Blank input keeps the saved value. Delete removes optional fields.</Text> : null}
        {server ? fields.map((name, index) => compact && index !== field ? null : <Box key={name} id={`server-field-${index}`} flexDirection="column" flexShrink={0}><SettingRow t={t} label={labels[index]!} selected={field === index}>
          {name in draft ? draft[name] === null ? '(remove)' : ['enabled', 'languageId', 'extensions', 'timeoutMs'].includes(name) ? String(draft[name]) : '(replacement entered)' : name === 'enabled' ? String(server.enabled) : name === 'languageId' ? server.languageId : name === 'extensions' ? server.extensions.join(', ') : name === 'timeoutMs' ? String(server.timeoutMs) : '(keep saved value)'}
        </SettingRow>{editing && !naming && index === field ? <textarea id="server-edit" key={`${selected}-${field}`} ref={input} focused minHeight={2} maxHeight={5} /> : null}</Box>) : !editing ? <DialogEmpty t={t} title="No LSP servers configured" description="Add a language server for diagnostics and navigation." action="F2  Add server" onAction={beginNew} /> : null}
        {editing && naming ? <textarea id="server-edit" key={`${naming}-${selected}-${field}`} ref={input} focused minHeight={2} maxHeight={5} /> : null}
      </scrollbox>}
      <Text wrap="wrap" color={t.color.warn}>{message}</Text>
      <DialogFooter t={t}><Text wrap="wrap">{busy ? 'Saving/loading… Please wait.' : !server && !editing ? 'F2 add server · F5 reload · Esc close' : removing ? 'Y remove server · N cancel' : editing ? 'Enter apply draft · Esc cancel edit' : compact ? 'Tab/Enter edit · Ctrl+S save · F2 new · F4 remove · F5 reload · Esc' : '↑/↓ server · F2 new · F4 remove · Tab field · Enter edit · Ctrl+S save · F5 reload · Esc close'}</Text></DialogFooter>
    </Box>
  </box>
}
