// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogSection, SettingRow } from './dialogChrome.js'
import { RoutingNoteEditor } from './routingNoteEditor.js'
const tiers = ['light', 'balanced', 'smart'] as const
interface Tier { model: string; provider_profile: string; reasoning_effort: string }
interface Settings { default?: string; light?: string | Partial<Tier>; balanced?: string | Partial<Tier>; smart?: string | Partial<Tier> }
interface Response { ok: boolean; revision: number; settings?: Settings; profiles?: { name: string; provider: string; model: string }[]; error?: string }
export function AgentSettingsOverlay({ t }: { t: Theme }) {
  const gateway = useOptionalGateway()
  const size = overlayPanelSize(useTerminalDimensions(), { maxWidth: 110, minWidth: 32, maxHeight: 34 })
  const [rows, setRows] = useState<Tier[]>(tiers.map(() => ({ model: '', provider_profile: '', reasoning_effort: '' })))
  const [noteTarget, setNoteTarget] = useState<{ profile: string; model: string } | null>(null)
  const [field, setField] = useState(0)
  const [loaded, setLoaded] = useState<Response | null>(null)
  const [defaultMode, setDefaultMode] = useState('inherit')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [efforts, setEfforts] = useState<string[]>([])
  const [optionsError, setOptionsError] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [modelStatus, setModelStatus] = useState('')
  const [modelRefresh, setModelRefresh] = useState(0)
  const pending = useRef(false)
  const alive = useRef(true)
  const input = useRef<TextareaRenderable | null>(null)
  const keys = ['provider_profile', 'model', 'reasoning_effort'] as const
  const row = Math.floor(field / 3), key = keys[field % 3]!
  useEffect(() => {
    alive.current = true
    void gateway?.rpc<Response>('agent.settings.get', {}).then(result => {
      if (!alive.current) return
      if (!result?.ok) throw new Error(result?.error || 'Could not load settings')
      setLoaded(result)
      setDefaultMode(result.settings?.default ?? 'inherit')
      setRows(tiers.map(tier => { const value = result.settings?.[tier]; return { provider_profile: '', reasoning_effort: '', model: '', ...(typeof value === 'string' ? { model: value } : value) } }))
    }).catch(error => { if (alive.current) setError(String(error)) })
    return () => { alive.current = false }
  }, [gateway])
  useEffect(() => { input.current?.setText(rows[row]?.[key] ?? '') }, [field, loaded, noteTarget])
  const selected = rows[row]!
  useEffect(() => {
    let cancelled = false
    setModels([]); setModelStatus('')
    if (!loaded || !gateway || key !== 'model') return
    setModelStatus('Loading models…')
    const profile = selected.provider_profile.trim()
    void gateway.rpc<{ ok: boolean; models?: string[]; warning?: string; error?: string }>('fetch_models', profile ? { profile_name: profile } : {}).then(result => {
      if (cancelled) return
      if (!result?.ok) throw new Error(result?.error || 'Could not load models')
      const available = [...new Set(result.models ?? [])].filter(Boolean)
      setModels(available)
      setModelStatus(result.warning || (available.length ? `${available.length} models · F5 refresh` : 'No models returned; enter a model ID.'))
    }).catch(error => { if (!cancelled) setModelStatus(`${String(error)} · F5 retry or enter a model ID`) })
    return () => { cancelled = true }
  }, [gateway, loaded, key, selected.provider_profile, modelRefresh])
  useEffect(() => {
    let cancelled = false
    setEfforts([]); setOptionsError('')
    if (!loaded || !gateway || key !== 'reasoning_effort') return
    void gateway.rpc<{ ok: boolean; reasoning_efforts?: string[]; error?: string }>('agent.settings.options', {
      provider_profile: selected.provider_profile.trim(), model: selected.model.trim(),
    }).then(result => {
      if (cancelled) return
      if (!result?.ok) throw new Error(result?.error || 'Could not load reasoning choices')
      setEfforts(result.reasoning_efforts ?? [])
    }).catch(error => { if (!cancelled) setOptionsError(String(error)) })
    return () => { cancelled = true }
  }, [gateway, loaded, row, key, selected.provider_profile, selected.model])
  const choices = key === 'provider_profile' ? ['', ...(loaded?.profiles?.map(profile => profile.name) ?? [])]
    : key === 'reasoning_effort' ? ['', ...efforts] : models
  const save = () => {
    if (!loaded || !gateway || pending.current) return
    const settings: Record<string, unknown> = { default: defaultMode }
    for (let i = 0; i < tiers.length; i++) {
      const value = rows[i]!
      if (!value.model.trim()) {
        if (value.provider_profile.trim() || value.reasoning_effort.trim()) { setError(`Enter a model for ${tiers[i]}`); return }
        continue
      }
      settings[tiers[i]!] = { model: value.model.trim(), ...(value.provider_profile.trim() ? { provider_profile: value.provider_profile.trim() } : {}), ...(value.reasoning_effort.trim() ? { reasoning_effort: value.reasoning_effort.trim() } : {}) }
    }
    if (defaultMode !== 'inherit' && !settings[defaultMode]) { setError('Configure the default mode or choose inherit with F2'); return }
    pending.current = true; setBusy(true); setError('')
    void gateway.rpc<Response>('agent.settings.save', { revision: loaded.revision, settings }).then(result => {
      if (!result?.ok) throw new Error(result?.error || 'Could not save settings')
      if (alive.current) { setLoaded({ ...result, profiles: loaded.profiles }); setError('Saved. New turns use these settings; running agents are unchanged.') }
    }).catch(error => { if (alive.current) setError(String(error)) }).finally(() => { pending.current = false; if (alive.current) setBusy(false) })
  }
  useKeyboard(event => {
    if (noteTarget || event.eventType === 'release') return
    if (event.name === 'f6') {
      event.preventDefault(); event.stopPropagation()
      if (!loaded || busy) return
      const profile = selected.provider_profile.trim()
      if (!loaded.profiles?.some(value => value.name === profile)) { setError('Select an explicit provider profile before editing routing notes.'); return }
      setNoteTarget({ profile, model: selected.model.trim() }); return
    }
    if (event.name === 'f2' || event.name === 'f4') {
      event.preventDefault(); event.stopPropagation()
      if (!loaded || busy) return
      if (event.name === 'f2') {
        const modes = ['inherit', ...tiers.filter((_, i) => rows[i]!.model.trim())]
        setDefaultMode(current => modes[(modes.indexOf(current) + 1) % modes.length]!)
        setError('')
      } else {
        setRows(current => current.map((value, i) => i === row ? { model: '', provider_profile: '', reasoning_effort: '' } : value))
        input.current?.setText('')
        if (defaultMode === tiers[row]) setDefaultMode('inherit')
        setError(`${tiers[row]} disabled in draft. Enter saves; Esc discards.`)
      }
      return
    }
    if (event.name === 'f5' && key === 'model') {
      event.preventDefault(); event.stopPropagation()
      if (!busy) setModelRefresh(value => value + 1)
      return
    }
    if (['up', 'down'].includes(event.name) && choices.length > 0 && loaded) {
      event.preventDefault(); event.stopPropagation()
      if (busy) return
      const index = choices.indexOf(selected[key])
      const nextIndex = index < 0 ? (event.name === 'up' ? choices.length - 1 : 0)
        : (index + (event.name === 'up' ? choices.length - 1 : 1)) % choices.length
      const next = choices[nextIndex]!
      setRows(current => current.map((value, i) => i === row ? { ...value, [key]: next } : value))
      input.current?.setText(next)
      return
    }
    if (!['escape', 'tab', 'return'].includes(event.name)) return
    event.preventDefault(); event.stopPropagation()
    if (event.name === 'escape') { patchOverlayState({ agentSettings: false }); return }
    if (busy) return
    if (event.name === 'tab') setField(value => (value + (event.shift ? 8 : 1)) % 9)
    else save()
  })
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center"><Box width={size.width} height={size.height} backgroundColor={t.color.overlayBg} borderStyle="round" borderColor={t.color.border} padding={1} flexDirection="column">
    {noteTarget ? <RoutingNoteEditor t={t} {...noteTarget} onClose={() => setNoteTarget(null)} /> : <>
    <DialogHeader t={t} title="Configuration · Agent modes" subtitle="Match each task to the right amount of intelligence." />
    <Box flexDirection="row" gap={1} flexShrink={0}>{tiers.map((tier, i) => <Box key={tier} paddingX={1} backgroundColor={i === row ? t.color.completionCurrentBg : undefined}><Text bold={i === row} color={i === row ? t.color.accent : t.ds.secondary}>{i === row ? `[${tier.toUpperCase()}]` : tier}</Text></Box>)}</Box>
    <Text wrap="wrap">Default: {defaultMode} · F2 change · F4 disable selected mode</Text>
    <scrollbox style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      <Text wrap="wrap" color={t.ds.secondary}>Blank provider/effort: inherit. Empty tier: disabled.</Text>
      <Text wrap="wrap">Provider profiles: {loaded?.profiles?.map(profile => `${profile.name} (${profile.provider})`).join(' · ') || 'Loading…'}</Text>
      {tiers.map((tier, i) => i !== row ? null : <Box key={tier} flexDirection="column" marginTop={1}>
        <DialogSection t={t}>{tier.toUpperCase()}</DialogSection>
        {keys.map((name, j) => <SettingRow key={name} t={t} selected={field === i * 3 + j} label={['Provider profile', 'Model ID', 'Reasoning effort'][j]!}>{rows[i]?.[name] || '(inherit / unset)'}</SettingRow>)}
      </Box>)}
      {loaded ? <textarea key={field} ref={input} focused={!busy} minHeight={1} maxHeight={2} onContentChange={() => { const text = input.current?.plainText ?? ''; setRows(current => current.map((value, i) => i === row ? { ...value, [key]: text } : value)) }} /> : null}
      {choices.length > 0 ? <Text wrap="wrap" color={t.ds.secondary}>↑/↓ choose: {choices.slice(Math.max(0, choices.indexOf(selected[key]) - 2), Math.max(0, choices.indexOf(selected[key]) - 2) + 6).map(value => value || '(inherit)').join(' · ')}{choices.length > 6 ? ' …' : ''}</Text> : null}
      {key === 'model' && modelStatus ? <Text wrap="wrap" color={t.ds.secondary}>{modelStatus}</Text> : null}
      {optionsError && key === 'reasoning_effort' ? <Text wrap="wrap" color={t.color.warn}>{optionsError}</Text> : null}
      {error ? <Text wrap="wrap" color={t.color.warn}>{error}</Text> : null}
    </scrollbox>
    <DialogFooter t={t}><Text wrap="wrap">{busy ? 'Saving…' : 'Tab field/mode · F6 routing notes · Enter save · Esc close'}</Text></DialogFooter>
    </>}
  </Box></box>
}
