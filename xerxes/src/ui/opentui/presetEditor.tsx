// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { TextareaRenderable, ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'

type Preset = { id: string; name: string; description: string; manageable: boolean; broken?: string }
type Draft = { original: string; content: string }
// Drafts survive closing the editor and session switches, but never cross session identities.
const drafts = new Map<string, Draft>()
export function PresetEditor({ t, sessionId, onClose }: { t: Theme; sessionId: string; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 120, minWidth: 32, maxHeight: 38 })
  const [rows, setRows] = useState<Preset[]>([])
  const [selected, setSelected] = useState(0)
  const [content, setContent] = useState('')
  const [guarded, setGuarded] = useState(false)
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [refresh, setRefresh] = useState(0)
  const input = useRef<TextareaRenderable | null>(null)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const alive = useRef(true)
  const action = useRef(false)
  const active = rows[selected]
  const draftKey = `${sessionId}:${active?.id ?? ''}`
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    if (!gateway) { setError('Connect to a daemon to inspect presets.'); return }
    setBusy(true)
    void gateway.rpc('agentPreset.list', {}).then(result => {
      if (!current) return
      if (!result?.ok || !Array.isArray(result.presets)) throw new Error(String(result?.error || 'Could not load presets'))
      const parsed = result.presets.map((value: unknown): Preset => {
        if (!value || typeof value !== 'object') throw new Error('Invalid preset')
        const row = value as Record<string, unknown>
        if (typeof row.id !== 'string' || typeof row.name !== 'string' || typeof row.description !== 'string' || typeof row.manageable !== 'boolean') throw new Error('Invalid preset')
        return { id: row.id, name: row.name, description: row.description, manageable: row.manageable, ...(typeof row.broken === 'string' ? { broken: row.broken } : {}) }
      })
      setRows(parsed); setSelected(previous => Math.min(previous, Math.max(0, parsed.length - 1)))
    }).catch(failure => { if (current) setError(String(failure)) }).finally(() => { if (current) setBusy(false) })
    return () => { current = false }
  }, [gateway, refresh])
  useEffect(() => {
    let current = true
    setContent(''); setGuarded(false); setError(''); setNotice(''); scroll.current?.scrollTo(0)
    if (!gateway || !active) return
    setBusy(true)
    void gateway.rpc('agentPreset.read', { agent_preset: active.id }).then(result => {
      if (!current) return
      if (!result?.ok || typeof result.content !== 'string') throw new Error(String(result?.error || 'Could not read composition'))
      setContent(result.content); setGuarded(result.guarded_write === true)
    }).catch(failure => { if (current) setError(String(failure)) }).finally(() => { if (current) setBusy(false) })
    return () => { current = false }
  }, [gateway, active?.id, refresh])
  const edit = () => {
    if (busy || !active?.manageable || !guarded || !content) return
    if (!drafts.has(draftKey)) drafts.set(draftKey, { original: content, content })
    setEditing(true); setError(''); setNotice('')
  }
  const save = async () => {
    const draft = drafts.get(draftKey)
    if (!gateway || !active || !draft || action.current) return
    action.current = true; setBusy(true); setError('')
    draft.content = input.current?.plainText ?? draft.content
    try {
      const result = await gateway.rpc('agentPreset.write', { agent_preset: active.id, content: draft.content, expected_content: draft.original })
      if (!result?.ok) throw new Error(String(result?.error || 'Could not save composition'))
      drafts.delete(draftKey)
      if (alive.current) { setContent(draft.content); setEditing(false); setNotice('Composition saved. Existing conversations keep their selected preset.'); setRefresh(value => value + 1) }
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { action.current = false; if (alive.current) setBusy(false) }
  }
  const locate = async () => {
    if (!gateway || !active?.manageable || action.current) return
    action.current = true; setBusy(true); setError('')
    try {
      const result = await gateway.rpc('agentPreset.openDocument', { agent_preset: active.id })
      if (!result?.ok || typeof result.path !== 'string') throw new Error(String(result?.error || 'Document location unavailable'))
      if (alive.current) setNotice(`Document folder: ${result.path}`)
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { action.current = false; if (alive.current) setBusy(false) }
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (editing) {
      if (key.name === 'escape') { key.preventDefault(); key.stopPropagation(); if (!busy) setEditing(false) }
      else if (key.ctrl && key.name === 's') { key.preventDefault(); key.stopPropagation(); void save() }
      else if (key.ctrl && key.name === 'd') { key.preventDefault(); key.stopPropagation(); if (!busy) { drafts.delete(draftKey); setEditing(false); setRefresh(value => value + 1) } }
      return
    }
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { onClose(); return }
    if (busy) return
    if (key.name === 'up' || key.name === 'down') setSelected(value => Math.max(0, Math.min(rows.length - 1, value + (key.name === 'up' ? -1 : 1))))
    else if (key.name === 'e' || key.name === 'return') edit()
    else if (key.name === 'o') void locate()
    else if (key.name === 'r') setRefresh(value => value + 1)
    else if (key.name === 'pageup' || key.name === 'pagedown') scroll.current?.scrollBy(key.name === 'pageup' ? -8 : 8)
    else if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(Number.MAX_SAFE_INTEGER)
  })
  const wide = size.width >= 90
  const count = Math.max(1, wide ? size.height - 12 : 3)
  const start = Math.max(0, selected - count + 1)
  return <box position="absolute" width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.statusBg}>
      <DialogHeader t={t} title={editing ? `Edit composition · ${active?.id}` : 'Agent compositions'} subtitle="Complete preset instructions, tools and configuration." />
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {notice ? <Text color={t.color.accent} wrap="wrap">{notice}</Text> : null}
      {editing ? <textarea key={draftKey} ref={input} initialValue={drafts.get(draftKey)?.content ?? content} onContentChange={() => { const draft = drafts.get(draftKey); if (draft && input.current) draft.content = input.current.plainText }} focused={!busy} flexGrow={1} minHeight={1} focusedBackgroundColor={t.color.statusBg} focusedTextColor={t.color.text} /> : !rows.length ? <DialogEmpty t={t} title={busy ? 'Loading compositions…' : 'No compositions available.'} description="Press R to reload." symbol="≡" /> : <Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? 30 : '100%'} flexDirection="column" flexShrink={0} paddingRight={1}>
          {rows.slice(start, start + count).map((row, i) => <Box key={row.id} backgroundColor={selected === start + i ? t.ds.selected : undefined} onMouseDown={() => { if (!busy) setSelected(start + i) }}><Text color={t.color.text} wrap="truncate-end">{row.name}{row.broken ? ' !' : ''}</Text></Box>)}
        </Box>
        <scrollbox ref={scroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
          <Text color={t.color.accent} bold wrap="wrap">{active?.id}</Text>
          <Text color={t.ds.secondary} wrap="wrap">{active?.description}</Text>
          {active?.broken ? <Text color={t.color.warn} wrap="wrap">{active.broken}</Text> : null}
          <Text color={t.ds.secondary} wrap="wrap">{active?.manageable ? guarded ? 'E edit · O document folder' : 'Read only: this daemon lacks guarded writes.' : `Shipped preset · copy with /preset copy ${active?.id} my-agent to edit.`}</Text>
          <Text color={t.color.text} wrap="wrap">{content || (busy ? 'Loading composition…' : '')}</Text>
        </scrollbox>
      </Box>}
      <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">{busy ? 'Loading / saving…' : editing ? 'Ctrl+S save · Esc keep draft · Ctrl+D discard' : '↑↓ select · E edit · O folder · R reload · Esc close'}</Text>{!editing ? <Text color={t.ds.secondary}>PgUp/PgDn scroll composition</Text> : null}</DialogFooter>
    </Box>
  </box>
}
