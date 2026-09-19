// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useRef, useState } from 'react'
import type { RemoteTaskDecision, RemoteTaskReview } from '../lib/remoteTaskSetup.js'
import type { Theme } from '../theme.js'
import { ModalShell } from './pickerChrome.js'

const durations = [15, 60, 240] as const
const requests = [50, 200, 1000] as const
const outputs = [4096, 16384, 65536] as const
const concurrent = [1, 4, 8] as const

export function RemoteProviderReview({value, t, onChoose, onCancel}: {
  value: RemoteTaskReview; t: Theme; onChoose: (decision: RemoteTaskDecision) => void; onCancel: () => void
}) {
  const {width, height} = useTerminalDimensions()
  const [profileIndex, setProfileIndex] = useState(-1)
  const [field, setField] = useState(0)
  const [duration, setDuration] = useState(1)
  const [request, setRequest] = useState(1)
  const [output, setOutput] = useState(1)
  const [concurrency, setConcurrency] = useState(1)
  const [confirm, setConfirm] = useState(false)
  const [providerControlled, setProviderControlled] = useState(false)
  const [error, setError] = useState('')
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const profile = value.profiles[profileIndex]
  const currentRoute = value.localRequirement || `Remote provider: ${value.remoteProfile || 'not configured'}`
  const keepSetup = value.localRequirement
    ? 'Keep this task’s local-provider requirement. No local access is granted. Choose a local profile to authorize again, or open the task and use /provider to select remote credentials explicitly.'
    : 'Use the selected task’s existing remote credentials, model and integrations. No local provider access is granted. After connecting, use /provider to configure remote credentials if needed.'
  const allowed = value.localBindingSupported && !value.running && profile?.supported
  useKeyboard(key => {
    if (!['escape', 'return', 'enter', 'tab', 'left', 'right', 'up', 'down', 'pageup', 'pagedown', 'home', 'end', 'a', 'c'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { if (confirm) {setConfirm(false);setError('')} else onCancel(); return }
    if (['pageup', 'pagedown', 'home', 'end'].includes(key.name)) {
      if (key.name === 'home') scroll.current?.scrollTo(0)
      else if (key.name === 'end') scroll.current?.scrollTo(1_000_000)
      else scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, height - 8))
      return
    }
    if (confirm) {
      if (key.name === 'up' || key.name === 'down') {scroll.current?.scrollBy(key.name === 'up' ? -1 : 1);return}
      if (key.name === 'c' && profile?.providerControlledOutput) {setProviderControlled(v => !v);setError('');return}
      if (key.name !== 'a' || !profile || !allowed) return
      if (profile.providerControlledOutput && !providerControlled) {setError('Press C to explicitly accept provider-controlled output length first.');return}
      onChoose({kind:'local',profile:profile.name,durationMinutes:durations[duration]!,maxRequests:requests[request]!,
        maxOutputTokens:profile.providerControlledOutput ? null : outputs[output]!,maxConcurrent:concurrent[concurrency]!,consentProviderControlledOutput:providerControlled})
      return
    }
    if (key.name === 'tab') {setField(v => (v + (key.shift ? 4 : 1)) % 5);return}
    if (key.name === 'return' || key.name === 'enter') {
      if (!profile) {onChoose({kind:'remote'});return}
      if (!allowed) {setError(value.running ? 'This task is running. Use its current setup or return after it stops.' : profile.setup || 'Local provider binding is unavailable. Update the remote runtime or use remote setup.');return}
      setConfirm(true);setProviderControlled(false);setError('');scroll.current?.scrollTo(0);return
    }
    const delta = key.name === 'left' || key.name === 'up' ? -1 : key.name === 'right' || key.name === 'down' ? 1 : 0
    if (!delta) return
    setError('')
    if (field === 0) setProfileIndex(v => Math.max(-1, Math.min(value.profiles.length - 1, v + delta)))
    else if (field === 1) setDuration(v => (v + delta + durations.length) % durations.length)
    else if (field === 2) setRequest(v => (v + delta + requests.length) % requests.length)
    else if (field === 3) setOutput(v => (v + delta + outputs.length) % outputs.length)
    else setConcurrency(v => (v + delta + concurrent.length) % concurrent.length)
  })
  const fields = [
    `Provider: ${profile ? 'local / ' + profile.name + ' · ' + (profileIndex + 1) + '/' + value.profiles.length : 'keep current task setup'}`,
    `Expires: ${durations[duration]} minutes after approval`,
    `Requests: ${requests[request]} total (includes subagents and summaries)`,
    `Output: ${profile?.providerControlledOutput ? 'provider-controlled; separate consent required' : outputs[output] + ' tokens per request maximum'}`,
    `Concurrent requests: ${concurrent[concurrency]}`,
  ]
  return <ModalShell t={t} title={confirm ? 'Authorize local provider' : 'Choose provider location'} width={width} height={height} panelWidth={Math.min(88,width)} panelHeight={Math.min(30,height)}>
    <box flexDirection="column" flexGrow={1} minHeight={0} paddingX={2}>
      <scrollbox ref={scroll} style={{flexGrow:1,minHeight:0}}>
        <text fg={t.color.text} wrapMode="word">{`SSH host: ${value.destination}\nRemote workspace: ${value.workspace}\nTask: ${value.sessionId}${value.running ? ' · running' : ''}\n${currentRoute}\nModel: ${value.remoteModel || 'not selected'}\n`}</text>
        {fields.map((text,i) => <text key={i} fg={!confirm && field === i ? t.color.accent : t.color.text} wrapMode="word">{`${!confirm && field === i ? '› ' : '  '}${text}`}</text>)}
        <text fg={t.color.text} wrapMode="word">{profile ? `\nModel: ${profile.model}\nCredential source: ${profile.credentialSource}\nProvider requests and token refresh run on this local workstation. Code and tools run on the remote host. Only this task and its delegated provider work use this grant. Requests send this task's context to the selected provider.\n\nCredentials are not copied. The grant stays in memory; closing this SSH window, expiry, revocation or connection loss ends access. Remote history keeps the provider requirement. Reconnect requires a new review; it never switches to remote credentials. Other remote tasks keep their own setup.\n\n${profile.setup}` : '\n' + keepSetup}</text>
        {value.localRequirement ? <text fg={t.color.warn} wrapMode="word">{value.localRequirement + '. Choose a local profile to authorize again, or open the task and use /provider to choose remote credentials explicitly.'}</text> : null}
        {value.inventoryError ? <text fg={t.color.warn} wrapMode="word">{value.inventoryError}</text> : null}
        {!value.localBindingSupported ? <text fg={t.color.warn} wrapMode="word">Remote runtime does not report local provider binding support.</text> : null}
        {confirm && profile?.providerControlledOutput ? <text fg={t.color.warn} wrapMode="word">{`${providerControlled ? '[accepted]' : '[not accepted]'} C · Accept provider-controlled output length; no per-request output cap can be enforced.`}</text> : null}
        {error ? <text fg={t.color.error} wrapMode="word">{error}</text> : null}
      </scrollbox>
      <text flexShrink={0} fg={t.color.muted}>PgUp/PgDn scroll · Esc {confirm ? 'edit' : 'cancel'}</text>
      <text flexShrink={0} fg={t.color.accent} wrapMode="word">{confirm ? 'A authorize this local provider scope' : 'Tab field · ←→ change · Enter ' + (profile ? 'review consent' : 'keep current setup')}</text>
    </box>
  </ModalShell>
}
