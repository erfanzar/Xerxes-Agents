// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard } from '@opentui/react'
import type { TextareaRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import type { MonitorView } from '../lib/monitors.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, SettingRow } from './dialogChrome.js'

export function MonitorPolicy({ t, monitor, onClose, onSaved }: { t: Theme; monitor: MonitorView; onClose: () => void; onSaved: () => void }) {
  const gateway = useOptionalGateway()
  const policy = monitor.policy!
  const [values, setValues] = useState([String(policy.maxReactions), String(policy.maxDurationMs / 1000), policy.maxTotalTokens == null ? '' : String(policy.maxTotalTokens)])
  const [field, setField] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const pending = useRef(false), alive = useRef(true), input = useRef<TextareaRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => { input.current?.setText(values[field]!) }, [field])
  const save = () => {
    if (!gateway || pending.current) return
    const attempts = Number(values[0]), timeout = Number(values[1]), tokens = values[2]!.trim() ? Number(values[2]) : null
    if (!Number.isSafeInteger(attempts) || attempts < 1 || attempts > 10 || !Number.isSafeInteger(timeout) || timeout < 1 || timeout > 86400 || (tokens !== null && (!Number.isSafeInteger(tokens) || tokens < 1))) {
      setError('Attempts 1–10; timeout 1–86400 seconds; tokens positive or blank.'); return
    }
    pending.current = true; setBusy(true); setError('')
    void gateway.rpc<{ ok: boolean; error?: string }>('monitor.update', { monitor_id: monitor.id, revision: policy.revision, max_reactions: attempts, reaction_timeout_seconds: timeout, max_total_tokens: tokens })
      .then(result => { if (!result?.ok) throw new Error(result?.error || 'Could not update policy'); if (alive.current) onSaved() })
      .catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { pending.current = false; if (alive.current) setBusy(false) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release' || !['escape', 'tab', 'return'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (busy) return
    if (key.name === 'escape') onClose()
    else if (key.name === 'tab') setField(value => (value + (key.shift ? 2 : 1)) % 3)
    else save()
  })
  return <Box flexDirection="column" flexGrow={1} minHeight={0}>
    <DialogHeader t={t} title={<> Reaction policy · {monitor.match}</>} />
    <scrollbox style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
      {['Lifetime attempts', 'Timeout seconds', 'Lifetime token threshold'].map((label, index) => <SettingRow key={label} t={t} selected={field === index} label={label}>{values[index] || 'unlimited'}</SettingRow>)}
      <textarea key={field} ref={input} focused={!busy} minHeight={1} maxHeight={2} onContentChange={() => { const text = input.current?.plainText ?? ''; setValues(previous => previous.map((value, index) => index === field ? text : value)) }} />
      <Text wrap="wrap">Usage history is retained. Saving may start pending reactions. In-flight calls may overshoot token thresholds.</Text>
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
    </scrollbox>
    <DialogFooter t={t}><Text>{busy ? 'Saving…' : 'Tab field · Enter save · Esc back'}</Text></DialogFooter>
  </Box>
}
