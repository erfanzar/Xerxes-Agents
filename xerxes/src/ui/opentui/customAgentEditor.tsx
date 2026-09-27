// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable, TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'

type Row = { id: string; description: string; error?: string }
type Draft = { id: string; content: string; revision: string | null }
export function CustomAgentEditor({ t, onClose }: { t: Theme; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const terminal = useTerminalDimensions()
  const size = overlayPanelSize(terminal, { maxWidth: 120, minWidth: 32, maxHeight: 34 })
  const [rows, setRows] = useState<Row[]>([])
  const [selected, setSelected] = useState(0)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [generating, setGenerating] = useState(false)
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const input = useRef<TextareaRenderable | null>(null)
  const details = useRef<ScrollBoxRenderable | null>(null)
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    if (!gateway) { setError('Connect to a daemon to edit agents.'); setLoading(false); return }
    setLoading(true)
    void gateway.rpc('agentPreset.projectList', {}).then(result => {
      if (!current) return
      if (!result?.ok || !Array.isArray(result.agents)) throw new Error(String(result?.error || 'Could not load agents'))
      const agents = result.agents.map((value: unknown): Row => {
        if (!value || typeof value !== 'object' || !('id' in value) || typeof value.id !== 'string' || !('description' in value) || typeof value.description !== 'string') throw new Error('Invalid custom agent record')
        return { id: value.id, description: value.description, ...('error' in value && typeof value.error === 'string' ? { error: value.error } : {}) }
      })
      setRows(agents); setSelected(value => Math.min(value, Math.max(0, agents.length - 1)))
    }).catch(failure => { if (current) setError(String(failure)) }).finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [gateway, refresh])
  useEffect(() => { if (draft) input.current?.setText(draft.content) }, [draft?.id])
  useEffect(() => { details.current?.scrollTo(0) }, [selected])
  const edit = async () => {
    if (!gateway || !rows[selected]) return
    setBusy(true); setError('')
    try {
      const result = await gateway.rpc('agentPreset.projectRead', { id: rows[selected]!.id })
      if (!result?.ok || typeof result.content !== 'string' || typeof result.revision !== 'string') throw new Error(String(result?.error || 'Could not read agent'))
      if (alive.current) setDraft({ id: rows[selected]!.id, content: result.content, revision: result.revision })
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { if (alive.current) setBusy(false) }
  }
  const save = async () => {
    if (!gateway || !draft) return
    setBusy(true); setError('')
    const content = input.current?.plainText ?? draft.content
    const id = draft.revision === null ? '' : draft.id
    try {
      const result = await gateway.rpc('agentPreset.projectWrite', { id, content, revision: draft.revision })
      if (!result?.ok) throw new Error(String(result?.error || 'Could not save agent'))
      if (alive.current) { setDraft(null); setRefresh(value => value + 1) }
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { if (alive.current) setBusy(false) }
  }
  const beginNew = () => { if (busy) return;  setError(''); setDraft({ id: 'new-agent', revision: null, content: '---\nname: new-agent\ndescription: Describe when to delegate to this specialist.\n---\nYou are a specialist. Describe your instructions here.\n' })  }
  const beginGenerate = () => { if (!busy) { setError(''); setGenerating(true) } }
  const generate = async () => {
    if (!gateway || busy) return
    const text = input.current?.plainText ?? description
    setDescription(text)
    if (!text.trim()) { setError('Describe the specialist you want to create.'); return }
    setBusy(true); setError('')
    try {
      const result = await gateway.rpc('agentPreset.projectGenerate', { description: text })
      if (!result?.ok || typeof result.id !== 'string' || typeof result.content !== 'string') throw new Error(String(result?.error || 'Could not generate agent'))
      if (alive.current) { setGenerating(false); setDraft({ id: result.id, content: result.content, revision: null }) }
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { if (alive.current) setBusy(false) }
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'escape') { key.preventDefault(); key.stopPropagation(); if (!busy) { setError(''); if (generating) { setDescription(input.current?.plainText ?? description); setGenerating(false) } else if (draft) setDraft(null); else onClose() }; return }
    if (generating) {
      if (key.ctrl && key.name === 'g') { key.preventDefault(); key.stopPropagation(); void generate() }
      return
    }
    if (draft) {
      if (key.ctrl && key.name === 's') { key.preventDefault(); key.stopPropagation(); if (!busy) void save() }
      return
    }
    if (!['g', 'n', 'return', 'up', 'down', 'pageup', 'pagedown'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (busy) return
    if (key.name === 'pageup' || key.name === 'pagedown') { details.current?.scrollBy(key.name === 'pageup' ? -5 : 5); return }
    if (key.name === 'up') setSelected(value => Math.max(0, value - 1))
    else if (key.name === 'down') setSelected(value => Math.min(Math.max(0, rows.length - 1), value + 1))
    else if (key.name === 'return') void edit()
    else if (key.name === 'g') beginGenerate()
    else beginNew()
  })
  const wide = size.width >= 80
  const count = Math.max(1, size.height - (wide ? 12 : 18))
  const start = Math.max(0, selected - count + 1)
  const active = rows[selected]
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      <DialogHeader t={t} title={generating ? 'Generate a specialist' : draft ? `Edit agent · ${draft.id}` : 'Custom agents'} subtitle={generating ? 'Describe the job. Your current model writes the first draft.' : draft ? 'Review the instructions before saving.' : `${rows.length} specialists · this project`} />
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {generating ? <Box flexGrow={1} minHeight={0} flexDirection="column">
        <Text color={t.color.accent}>WHAT SHOULD THIS AGENT DO?</Text>
        <Text color={t.ds.secondary} wrap="wrap">Example: A JAX reviewer who finds sharding mistakes, checks shapes, and proposes focused regression tests.</Text>
        <textarea key="description" ref={input} flexGrow={1} minHeight={1} focused={!busy} initialValue={description} focusedBackgroundColor={t.color.overlayBg} focusedTextColor={t.color.text} />
        <Text color={t.ds.secondary} wrap="wrap">Generated instructions open for review. Files are saved only with Ctrl+S.</Text>
      </Box> : draft ? <textarea key={draft.id} ref={input} flexGrow={1} minHeight={1} focused={!busy} initialValue={draft.content} focusedBackgroundColor={t.color.overlayBg} focusedTextColor={t.color.text} /> : loading ? <Box flexGrow={1}><Text color={t.ds.secondary}>Loading specialists…</Text></Box> : rows.length ? <Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0}>
        <Box width={wide ? 34 : '100%'} flexDirection="column" paddingRight={wide ? 2 : 0} minHeight={0}>
          <Text color={t.ds.secondary}>SPECIALISTS · {selected + 1}/{rows.length}</Text>
          {rows.slice(start, start + count).map((row, index) => <Box key={row.id} backgroundColor={start + index === selected ? t.color.completionCurrentBg : t.color.overlayBg} onMouseDown={() => setSelected(start + index)}>
            <Text color={start + index === selected ? t.color.accent : t.color.text} wrap="truncate-end">{start + index === selected ? '▸ ' : '  '}{row.id}{row.error ? ' !' : ''}</Text>
          </Box>)}
        </Box>
        <Box flexDirection="column" flexGrow={1} minWidth={0} minHeight={0} paddingLeft={wide ? 2 : 0} borderStyle="single" borderSides={wide ? ['left'] : ['top']} borderColor={t.color.border}>
          <Text color={t.color.accent} bold>{active?.id}</Text>
          <scrollbox ref={details} flexGrow={1} minHeight={1}>
            <Text color={t.ds.secondary}>WHEN TO DELEGATE</Text>
            <Text color={active?.error ? t.color.warn : t.color.text} wrap="wrap">{active?.error || active?.description || 'No description yet. Press Enter to add one.'}</Text>
            <Text color={t.ds.secondary} wrap="wrap">{`
.xerxes/agents/${active?.id}.md`}</Text>
          </scrollbox>
        </Box>
      </Box> : <DialogEmpty t={t} title={error ? 'Could not load agents.' : 'No custom agents.'} description="Give a specialist a role, instructions and tools." action="Press N to create one." onAction={beginNew} symbol="✦" />}
      <DialogFooter t={t}>
        {busy ? <Text color={t.color.accent}>{generating ? 'Generating draft with your current model…' : 'Saving / loading…'}</Text> : generating ? <Box onMouseDown={() => void generate()}><Text color={t.color.accent}>Ctrl+G generate · Esc back</Text></Box> : draft ? <Text color={t.ds.secondary}>Ctrl+S save · Esc discard draft</Text> : <Box flexDirection="column">
          <Box flexDirection="row"><Box onMouseDown={beginGenerate}><Text color={t.color.accent} bold> G Generate </Text></Box><Box onMouseDown={beginNew}><Text color={t.color.text}> N New blank </Text></Box></Box>
          <Text color={t.ds.secondary} wrap="wrap">↑↓ select · PgUp/PgDn details · Enter edit · Esc close</Text>
        </Box>}
      </DialogFooter>
    </Box>
  </box>
}
