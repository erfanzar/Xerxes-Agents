// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { forgePackage, type ForgePackage, type ForgeParameter } from '../lib/forge.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty } from './dialogChrome.js'
import { overlayPanelSize } from './overlayLayout.js'
import { RecordForm, type RecordField } from './recordForm.js'

const definitionFields: RecordField[] = [
  { key: 'name', label: 'Package name', help: 'A reusable template name, for example release-note.' },
  { key: 'version', label: 'Version', help: 'Use a new version such as 1.0.0. Saved versions are immutable.' },
  { key: 'description', label: 'Description', help: 'Explain what this package produces.' },
  { key: 'template', label: 'Template', help: 'Text with {{parameter}} placeholders. Add their declarations in review.' },
]
const parameterFields: RecordField[] = [
  { key: 'name', label: 'Parameter name', help: 'Matches {{name}} in the template.' },
  { key: 'description', label: 'Parameter description' },
  { key: 'required', label: 'Required?', help: 'Enter yes or no.' },
  { key: 'useDefault', label: 'Use a default?', help: 'Enter yes or no. An empty default is different from no default.' },
  { key: 'default', label: 'Default value', help: 'Used only when Use a default is yes.' },
]
type Definition = { values: Record<string, string>; parameters: ForgeParameter[] }
const drafts = new Map<string, Definition>()
type Mode = 'browse' | 'define' | 'review' | 'parameter' | 'run' | 'remove' | 'output'
export function ForgeOverlay({ t, sessionId, onClose }: { t: Theme; sessionId: string; onClose: () => void }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 120, minWidth: 32, maxHeight: 38 })
  const [rows, setRows] = useState<ForgePackage[]>([])
  const [selected, setSelected] = useState(0)
  const [detail, setDetail] = useState<ForgePackage | null>(null)
  const [mode, setMode] = useState<Mode>('browse')
  const [definition, setDefinition] = useState<Definition>(() => drafts.get(sessionId) ?? { values: { name: '', version: '1.0.0', description: '', template: '' }, parameters: [] })
  const [parameterIndex, setParameterIndex] = useState(0)
  const [parameter, setParameter] = useState<Record<string, string>>({})
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [output, setOutput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true)
  const action = useRef(false)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const active = rows[selected]
  const remember = (next: Definition) => { drafts.set(sessionId, next); setDefinition(next) }
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => {
    let current = true
    if (!gateway) { setError('Connect to a daemon to use Forge.'); return }
    setBusy(true)
    void gateway.rpc('forge.list', {}).then(result => {
      if (!current) return
      if (!result?.ok || !Array.isArray(result.packages)) throw new Error(String(result?.error || 'Could not load Forge packages'))
      setRows(result.packages.map(value => forgePackage(value)))
      setSelected(value => Math.min(value, Math.max(0, result.packages!.length - 1)))
      setError('')
    }).catch(failure => { if (current) setError(String(failure)) }).finally(() => { if (current) setBusy(false) })
    return () => { current = false }
  }, [gateway, refresh])
  useEffect(() => {
    let current = true
    setDetail(null)
    if (!gateway || !active) return
    setBusy(true)
    void gateway.rpc('forge.inspect', { name: active.name, version: active.version }).then(result => {
      if (!current) return
      if (!result?.ok) throw new Error(String(result?.error || 'Could not inspect package'))
      const parsed = forgePackage(result.package, true)
      if (parsed.name !== active.name || parsed.version !== active.version) throw new Error('Package changed. Refresh before continuing.')
      setDetail(parsed); setError('')
    }).catch(failure => { if (current) setError(String(failure)) }).finally(() => { if (current) setBusy(false) })
    return () => { current = false }
  }, [gateway, active?.name, active?.version, refresh])
  useEffect(() => { scroll.current?.scrollTo(0) }, [selected, mode])
  const perform = async (method: 'forge.define' | 'forge.run' | 'forge.undefine') => {
    if (!gateway || action.current) return
    if (method !== 'forge.define' && !detail) return
    action.current = true; setBusy(true); setError('')
    try {
      const params = method === 'forge.define' ? { ...definition.values, parameters: definition.parameters, confirm: true }
        : { name: detail!.name, version: detail!.version, ...(method === 'forge.run' ? { input: inputs } : { confirm: true }) }
      const result = await gateway.rpc(method, params)
      if (!result?.ok) throw new Error(String(result?.error || 'Forge request failed'))
      if (!alive.current) return
      if (method === 'forge.run') {
        if (typeof result.output !== 'string') throw new Error('Forge did not return text output')
        setOutput(result.output); setMode('output')
      } else {
        if (method === 'forge.define') { drafts.delete(sessionId); setDefinition({ values: { name: '', version: '1.0.0', description: '', template: '' }, parameters: [] }) }
        setMode('browse'); setRefresh(value => value + 1)
      }
    } catch (failure) { if (alive.current) setError(String(failure)) }
    finally { action.current = false; if (alive.current) setBusy(false) }
  }
  const editParameter = (index: number) => {
    const p = definition.parameters[index]
    setParameterIndex(index); setParameter(p ? { name: p.name, description: p.description, required: p.required ? 'yes' : 'no', useDefault: p.default === undefined ? 'no' : 'yes', default: p.default ?? '' }
      : { name: '', description: '', required: 'yes', useDefault: 'no', default: '' })
    setMode('parameter'); setError('')
  }
  const saveParameter = () => {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(parameter.name ?? '') || !['yes', 'no'].includes(parameter.required ?? '') || !['yes', 'no'].includes(parameter.useDefault ?? '')) { setError('Use a valid parameter name and yes/no for both options.'); return }
    if (definition.parameters.some((p, i) => i !== parameterIndex && p.name === parameter.name)) { setError('Parameter names must be unique.'); return }
    const next = [...definition.parameters]
    next[parameterIndex] = { name: parameter.name!, description: parameter.description ?? '', required: parameter.required === 'yes', ...(parameter.useDefault === 'yes' ? { default: parameter.default ?? '' } : {}) }
    remember({ ...definition, parameters: next }); setMode('review'); setError('')
  }
  useKeyboard(key => {
    if (key.eventType === 'release' || mode === 'define' || mode === 'parameter' || mode === 'run') return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { setError(''); if (mode === 'browse') onClose(); else setMode('browse'); return }
    if (busy) return
    if (key.name === 'pageup' || key.name === 'pagedown') { scroll.current?.scrollBy(key.name === 'pageup' ? -8 : 8); return }
    if (key.name === 'home' || key.name === 'end') { scroll.current?.scrollTo(key.name === 'home' ? 0 : Number.MAX_SAFE_INTEGER); return }
    if (mode === 'remove') { if (key.name === 'y') void perform('forge.undefine'); else if (key.name === 'n') setMode('browse'); return }
    if (mode === 'review') {
      if (key.name === 'a') editParameter(definition.parameters.length)
      else if (key.name === 'e') editParameter(parameterIndex)
      else if (key.name === 'up' || key.name === 'down') setParameterIndex(value => Math.max(0, Math.min(definition.parameters.length - 1, value + (key.name === 'up' ? -1 : 1))))
      else if (key.name === 'd') { remember({ ...definition, parameters: definition.parameters.filter((_, i) => i !== parameterIndex) }); setParameterIndex(0) }
      else if (key.name === 'b') setMode('define')
      else if (key.ctrl && key.name === 's') void perform('forge.define')
      return
    }
    if (mode === 'output') { if (key.name === 'b') setMode('run'); return }
    if (key.name === 'up' || key.name === 'down') setSelected(value => Math.max(0, Math.min(rows.length - 1, value + (key.name === 'up' ? -1 : 1))))
    else if (key.name === 'n') { setMode('define'); setError('') }
    else if (key.name === 'r') setRefresh(value => value + 1)
    else if (key.name === 'd' && detail) setMode('remove')
    else if ((key.name === 'return' || key.name === 'x') && detail) { setInputs(Object.fromEntries(detail.parameters.filter(p => p.default !== undefined).map(p => [p.name, p.default!]))); setMode('run'); setError('') }
  })
  const count = Math.max(1, Math.min(6, Math.floor(size.height / 5)))
  const start = Math.max(0, selected - count + 1)
  return <box position="absolute" width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={size.width} height={size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.statusBg}>
      <DialogHeader t={t} title={mode === 'browse' ? 'Forge packages' : mode === 'define' ? 'Define package' : mode === 'review' ? 'Review definition' : mode === 'parameter' ? 'Parameter' : mode === 'run' ? `Run · ${detail?.name}` : mode === 'remove' ? 'Remove package?' : 'Package output'} subtitle="Reusable text templates. No shell execution or provider call." />
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {mode === 'define' ? <RecordForm t={t} fields={definitionFields} values={definition.values} onChange={(key, value) => remember({ ...definition, values: { ...definition.values, [key]: value } })} onSubmit={() => { setMode('review'); setError('') }} onBack={() => setMode('browse')} busy={busy} submitLabel="review" />
        : mode === 'parameter' ? <RecordForm t={t} fields={parameterFields} values={parameter} onChange={(key, value) => setParameter(previous => ({ ...previous, [key]: value }))} onSubmit={saveParameter} onBack={() => setMode('review')} busy={busy} submitLabel="keep parameter" />
        : mode === 'run' ? <RecordForm t={t} fields={detail?.parameters.map(p => ({ key: p.name, label: p.name + (p.required ? ' · required' : ' · optional'), help: p.description + (p.default === undefined ? '' : ` Default: ${p.default}`) })) ?? []} values={inputs} onChange={(key, value) => setInputs(previous => ({ ...previous, [key]: value }))} onSubmit={() => void perform('forge.run')} onBack={() => setMode('browse')} busy={busy} submitLabel="run" />
        : <>
          {mode === 'browse' ? rows.slice(start, start + count).map((row, i) => <Box key={`${row.name}@${row.version}`} backgroundColor={selected === start + i ? t.ds.selected : undefined} onMouseDown={() => { if (!busy) setSelected(start + i) }}><Text color={t.color.text} wrap="truncate-end">{row.name} · {row.version}</Text></Box>) : null}
          <scrollbox ref={scroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
            {mode === 'browse' ? detail ? <>
              <Text color={t.color.accent} bold wrap="wrap">{detail.name} · {detail.version}</Text><Text color={t.color.text} wrap="wrap">{detail.description}</Text>
              <Text color={t.ds.secondary}>Created {detail.createdAt}</Text>
              {detail.parameters.map(p => <Text key={p.name} color={t.ds.secondary} wrap="wrap">{p.name} · {p.required ? 'required' : 'optional'}{p.default === undefined ? '' : ` · default: ${p.default}`} — {p.description}</Text>)}
              <Text color={t.color.text} wrap="wrap">{detail.template}</Text>
            </> : <DialogEmpty t={t} title={busy ? 'Loading packages…' : error ? 'Package unavailable.' : 'No Forge packages.'} description="N defines a reusable text template. R retries discovery." symbol="≡" />
              : mode === 'review' ? <>
                <Text color={t.color.accent} wrap="wrap">{definition.values.name} · {definition.values.version}</Text>
                <Text color={t.color.text} wrap="wrap">{definition.values.description}</Text>
                <Text color={t.color.text} wrap="wrap">{definition.values.template}</Text>
                <Text color={t.ds.secondary}>Parameters · {definition.parameters.length}</Text>
                {definition.parameters.map((p, i) => <Text key={p.name} color={i === parameterIndex ? t.color.accent : t.color.text} wrap="wrap">{i === parameterIndex ? '> ' : ''}{p.name} · {p.required ? 'required' : 'optional'}{p.default === undefined ? '' : ` · default: ${p.default}`} — {p.description}</Text>)}
                <Text color={t.color.warn} wrap="wrap">Ctrl+S confirms saving this immutable version.</Text>
              </> : mode === 'remove' ? <Text color={t.color.warn} wrap="wrap">Remove {detail?.name}@{detail?.version}? This deletes the saved package version. Y confirms; N or Escape cancels.</Text>
                : <Text color={t.color.text} wrap="wrap">{output || '(empty output)'}</Text>}
          </scrollbox>
          <DialogFooter t={t}><Text color={t.ds.secondary} wrap="wrap">{busy ? 'Working…' : mode === 'browse' ? '↑↓ select · Enter run · N new/draft · D remove · R refresh · Esc close' : mode === 'review' ? 'A add parameter · ↑↓ select · E edit · D remove · B definition · Ctrl+S save · Esc keep draft' : mode === 'output' ? 'B inputs · Esc packages' : 'Y remove · N cancel'}</Text><Text color={t.ds.secondary}>PgUp/PgDn scroll</Text></DialogFooter>
        </>}
    </Box>
  </box>
}
