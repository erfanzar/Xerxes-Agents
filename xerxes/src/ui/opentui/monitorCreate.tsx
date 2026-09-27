// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { createMonitor, listMonitorWebhooks } from '../lib/monitors.js'
import { listTerminals, type TerminalSummary } from '../lib/terminals.js'
import type { Theme } from '../theme.js'
import { Text } from './primitives.js'
import { SettingsFormLayout } from './settingsFormLayout.js'

export function MonitorCreate({ t, onClose, onCreated }: { t: Theme; onClose: () => void; onCreated: (id: string) => void }) {
  const gateway = useOptionalGateway()
  const [terminals, setTerminals] = useState<TerminalSummary[]>([])
  const [terminalIndex, setTerminalIndex] = useState(0)
  const [field, setField] = useState(0)
  const [values, setValues] = useState(['terminal', '', '', '3600', '', '3', '60', '', '', '', '', ''])
  const [sourceKind, setSourceKind] = useState<'terminal' | 'file' | 'websocket' | 'webhook'>('terminal')
  const [webhooks, setWebhooks] = useState<string[]>([])
  const [webhookIndex, setWebhookIndex] = useState(0)
  const [completion, setCompletion] = useState(false)
  const [react, setReact] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const alive = useRef(true)
  const input = useRef<TextareaRenderable | null>(null)
  const editable = [2, 3, 5, 6, 8, 9, 10].includes(field)
  const visibleFields = sourceKind === 'file' ? [0, 9, 3, 4, 5, 6, 8] : sourceKind === 'websocket' ? [0, 10, 2, 3, 4, 5, 6, 8] : sourceKind === 'webhook' ? [0, 11, 2, 3, 4, 5, 6, 8] : [0, 1, 2, 3, 4, 5, 6, 7, 8]
  useEffect(() => {
    alive.current = true
    if (gateway) void listTerminals(gateway.rpc).then(rows => { if (alive.current) setTerminals(rows) })
      .catch(failure => { if (alive.current) setError(String(failure)) })
    return () => { alive.current = false }
  }, [gateway])
  useEffect(() => {
    if (sourceKind !== 'webhook' || !gateway) return
    let current = true
    setWebhooks([])
    setWebhookIndex(0)
    void listMonitorWebhooks(gateway.rpc).then(names => {
      if (!current) return
      setWebhooks(names)
      setValues(previous => previous.map((value, index) => index === 11 ? names[0] ?? '' : value))
    }).catch(failure => { if (current) setError(String(failure)) })
    return () => { current = false }
  }, [gateway, sourceKind])
  useEffect(() => { if (editable) input.current?.setText(values[field] ?? '') }, [field, editable])
  const submit = () => {
    if (!gateway || busy) return
    const source = terminals[terminalIndex]
    if (sourceKind === 'terminal' && !source) { setError('Start a terminal command first.'); return }
    if (sourceKind === 'terminal' && !completion && !values[2]?.trim()) { setError('Enter a literal match.'); return }
    if (sourceKind === 'file' && !values[9]?.trim()) { setError('Enter a file path.'); return }
    if (sourceKind === 'websocket' && (!values[10]?.trim() || !values[2]?.trim())) { setError('Enter a websocket URL and literal match.'); return }
    if (sourceKind === 'webhook' && (!webhooks[webhookIndex] || !values[2]?.trim())) { setError(webhooks.length ? 'Enter a literal match.' : 'No configured webhook sources are available.'); return }
    const tokens = values[8]?.trim() ? Number(values[8]) : undefined
    if (tokens !== undefined && (!react || !Number.isSafeInteger(tokens) || tokens < 1)) { setError('Token threshold requires automatic reactions and a positive whole number.'); return }
    const duration = Number(values[3]), attempts = Number(values[5]), timeout = Number(values[6])
    if (!Number.isSafeInteger(duration) || duration < 1 || duration > 86400 || !Number.isSafeInteger(attempts) || attempts < 1 || attempts > 10 || !Number.isSafeInteger(timeout) || timeout < 1 || timeout > 120) {
      setError('Duration: 1–86400; attempts: 1–10; timeout: 1–120 seconds.'); return
    }
    setBusy(true); setError('')
    const settings = { ...(tokens === undefined ? {} : { max_total_tokens: tokens }), ...(sourceKind === 'file'
      ? { source_kind: 'file' as const, file_path: values[9].trim(), trigger: 'change' as const }
      : sourceKind === 'websocket'
        ? { source_kind: 'websocket' as const, websocket_url: values[10].trim(), trigger: 'output' as const, match: values[2].trim() }
        : sourceKind === 'webhook'
          ? { source_kind: 'webhook' as const, webhook_name: webhooks[webhookIndex]!, trigger: 'output' as const, match: values[2].trim() }
        : { terminal_id: source!.id, trigger: completion ? 'completion' as const : 'output' as const, match: values[2].trim() }), duration_seconds: duration, react, max_reactions: attempts, reaction_timeout_seconds: timeout }
    void createMonitor(gateway.rpc, settings)
      .then(watch => { if (alive.current) onCreated(watch.id) })
      .catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { if (alive.current) setBusy(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release') return
    if (key.name === 'escape' || key.name === 'tab' || key.name === 'return' || (!editable && ['left', 'right', 'space'].includes(key.name))) {
      key.preventDefault(); key.stopPropagation()
      if (key.name === 'escape') { if (!busy) onClose() }
      else if (key.name === 'tab') setField(value => {
        const index = visibleFields.indexOf(value)
        return visibleFields[(index + (key.shift ? visibleFields.length - 1 : 1)) % visibleFields.length]!
      })
      else if (key.name === 'return') submit()
      else if (field === 7) setCompletion(value => !value)
      else if (field === 4) setReact(value => !value)
      else if (field === 1) setTerminalIndex(value => terminals.length ? (value + (key.name === 'left' ? terminals.length - 1 : 1)) % terminals.length : 0)
      else if (field === 11) setWebhookIndex(value => webhooks.length ? (value + (key.name === 'left' ? webhooks.length - 1 : 1)) % webhooks.length : 0)
      else if (field === 0) setSourceKind(value => key.name === 'left'
        ? value === 'terminal' ? 'webhook' : value === 'file' ? 'terminal' : value === 'websocket' ? 'file' : 'websocket'
        : value === 'terminal' ? 'file' : value === 'file' ? 'websocket' : value === 'websocket' ? 'webhook' : 'terminal')
    }
  })
  const labels = ['Source', 'Terminal', 'Match', 'Duration seconds', 'Automatic reactions', 'Max reactions', 'Reaction timeout seconds', 'Trigger', 'Lifetime token threshold', 'File path', 'Websocket URL', 'Webhook name']
  const shown = [sourceKind === 'file' ? 'File changes' : sourceKind === 'websocket' ? 'Websocket server push' : sourceKind === 'webhook' ? 'Configured webhook' : 'Terminal output', terminals[terminalIndex]?.label ?? 'No terminals', sourceKind === 'terminal' && completion ? '(unused for completion)' : values[2] || '(required)', values[3], react ? 'Enabled' : 'Notifications only', values[5], values[6], sourceKind === 'terminal' && completion ? 'Command completion' : sourceKind === 'file' ? 'Metadata changes' : 'Output match', values[8] || 'unlimited', values[9] || '(required)', values[10] || '(required)', webhooks[webhookIndex] || (sourceKind === 'webhook' ? '(no configured webhooks)' : '(required)')]
  return <SettingsFormLayout t={t} title="New monitor" subtitle="Choose a source, then decide when and how to react."
    fields={visibleFields.map(id => ({ id, label: labels[id]!, value: shown[id]!, group: [0, 1, 2, 9, 10, 11].includes(id) ? 'SOURCE & MATCH' : 'REACTIONS & LIMITS' }))}
    selected={field} onSelect={setField} busy={busy ? 'Creating…' : ''} error={error}
    editor={editable ? <textarea key={field} ref={input} focused={!busy} minHeight={1} maxHeight={2} focusedBackgroundColor={t.color.overlayBg} focusedTextColor={t.color.text} onContentChange={() => {
        const text = input.current?.plainText ?? ''
        setValues(previous => previous.map((value, index) => index === field ? text : value))
      }} /> : null}
    help={<>
      <Text color={t.ds.secondary} wrap="wrap">{sourceKind === 'file' ? 'File watches report metadata changes only; rapid changes may coalesce. Contents are never read.' : sourceKind === 'websocket' ? 'Websocket watches receive text-only server push. Duplicates are suppressed; reconnects may create observation gaps. URL credentials and query strings are unsupported.' : sourceKind === 'webhook' ? 'Webhook watches use a configured host source. Payloads are text-only; delivery gaps may occur during reconnects.' : 'Reactions share the session model. Token thresholds block new calls; in-flight calls may overshoot. Unknown usage blocks further reactions.'}</Text>
</>}
  />
}
