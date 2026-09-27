// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import type { ScrollBoxRenderable } from '@opentui/core'
import { useEffect, useRef, useState } from 'react'
import { useStore } from '@nanostores/react'
import { $uiState } from '../app/uiStore.js'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { overlayPanelSize } from './overlayLayout.js'
import { Box, Text } from './primitives.js'
import { DialogHeader, DialogFooter, DialogEmpty, DialogSection } from './dialogChrome.js'

import { DeliveryPanel } from './deliveryPanel.js'
import { RunOverlay } from './runOverlay.js'
import { ScheduleTemplatePicker } from './scheduleTemplatePicker.js'
import { scheduleDescription, type ScheduleTemplate } from '../lib/scheduleTemplates.js'
import { ScheduleForm, type ScheduleDraft } from './scheduleForm.js'
import { parseScheduleTokens, scheduleTokensLabel } from '../lib/scheduleTokens.js'

type Job = { stop_condition?: string | null; target_session_id?: string | null; expires_at?: string | null; max_runs?: number | null; runs_started?: number; max_model_calls?: number | null; deliver?: string; recipient?: string; missed_run_policy?: "coalesce" | "skip"; misfire_grace_seconds?: number; timezone?: string; id: string; revision?: string; interval_seconds?: number | null; timeout_seconds?: number | null; max_retries?: number | null; prompt: string; schedule: string; paused: boolean; execution_state: string; next_run_at?: string; last_run_at?: string; metadata?: Record<string, unknown> }
function parseJobs(value: unknown): Job[] {
  if (!Array.isArray(value)) throw new Error('Invalid schedule list')
  return value.map(row => {
    if (!row || typeof row !== 'object' || typeof row.id !== 'string' || typeof row.prompt !== 'string' || typeof row.schedule !== 'string' || typeof row.paused !== 'boolean' || !['idle', 'running', 'cancelling'].includes(row.execution_state)) throw new Error('Invalid schedule record')
    parseScheduleTokens(row.token_budget)
    return row as Job
  })
}
export function scheduleUsageLabel(value: unknown): string {
  if (!value || typeof value !== 'object') return 'Last attempt tokens: unavailable'
  const usage = value as Record<string, unknown>
  if (![usage.input_tokens, usage.output_tokens].every(n => typeof n === 'number' && Number.isSafeInteger(n) && n >= 0)) return 'Last attempt tokens: unavailable'
  return `Last attempt tokens: ${usage.input_tokens} input · ${usage.output_tokens} output${usage.complete === true ? '' : ' · partial; some usage unavailable'}`
}
export function ScheduleOverlay({ t, followupsOnly = false }: { t: Theme; followupsOnly?: boolean }) {
  const gateway = useOptionalGateway()
  const sid = useStore($uiState).sid
  const terminal = useTerminalDimensions()
  const [deliveryId, setDeliveryId] = useState<string | null>(null)
  const [historyId, setHistoryId] = useState<string | null>(null)
  const [templates, setTemplates] = useState(false)
  const [template, setTemplate] = useState<ScheduleTemplate | undefined>()
  const [editing, setEditing] = useState<ScheduleDraft | 'new' | null>(null)
  const size = overlayPanelSize(terminal, { maxWidth: editing ? 132 : 124, maxHeight: 40, minWidth: 32, ...(editing ? { desiredHeight: 38 } : {}) })
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState('')
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true)
  const pending = useRef(new Set<string>())
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => { setJobs([]); setSelected(''); setError('') }, [sid, followupsOnly])
  useEffect(() => {
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        if (!gateway) throw new Error('Connect to a daemon to manage schedules.')
        const result = await gateway.rpc<{ ok: boolean; jobs?: unknown; error?: string }>('schedule.list', followupsOnly ? { scope: 'session', ...(sid ? { owner_session_id: sid } : {}) } : {})
        if (!result?.ok) throw new Error(result?.error || 'Schedules unavailable')
        const rows = parseJobs(result.jobs)
        if (current) { setError(''); setJobs(rows); setSelected(id => rows.some(row => row.id === id) ? id : rows[0]?.id ?? '') }
      } catch (failure) { if (current) setError(String(failure)) }
      finally { if (current) timer = setTimeout(() => { void load() }, 2000) }
    }
    void load()
    return () => { current = false; if (timer) clearTimeout(timer) }
  }, [gateway, refresh, followupsOnly, sid])
  const job = jobs.find(row => row.id === selected)
  const action = (name: 'pause' | 'resume' | 'run' | 'cancel') => {
    if (!gateway || !job) return
    const key = `${job.id}:${name}`
    if (pending.current.has(key)) return
    pending.current.add(key)
    void gateway.rpc<{ ok: boolean; error?: string }>(`schedule.${name}`, { schedule_id: job.id, ...(followupsOnly ? { scope: 'session', ...(sid ? { owner_session_id: sid } : {}) } : {}) }).then(result => {
      if (!result?.ok) throw new Error(result?.error || 'Schedule action failed')
      if (alive.current) setRefresh(value => value + 1)
    }).catch(failure => { if (alive.current) setError(String(failure)) })
      .finally(() => { pending.current.delete(key) })
  }
  useKeyboard(key => {
    if (key.eventType === 'release' || editing || templates || historyId || deliveryId || !['b', 'd', 'h', 'n', 'e', 'escape', 'up', 'down', 'p', 'r', 'x', 'g', 'pageup', 'pagedown'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') patchOverlayState({ schedules: false, loops: false })
    else if (key.name === 'd' && job) setDeliveryId(job.id)
    else if (key.name === 'h' && job) setHistoryId(job.id)
    else if (key.name === 'b') setTemplates(true)
    else if (key.name === 'n') { setTemplate(undefined); setEditing('new') }
    else if (key.name === 'e' && job?.revision) setEditing({ ...job, revision: job.revision })
    else if (key.name === 'p' && job) action(job.paused ? 'resume' : 'pause')
    else if (key.name === 'x') action('cancel')
    else if (key.name === 'g') action('run')
    else if (key.name === 'r') { setError(''); setRefresh(value => value + 1) }
    else if (key.name === 'up' || key.name === 'down') {
      const index = jobs.findIndex(row => row.id === selected)
      setSelected(jobs[Math.max(0, Math.min(jobs.length - 1, index + (key.name === 'up' ? -1 : 1)))]?.id ?? '')
      scroll.current?.scrollTo(0)
    } else scroll.current?.scrollBy((key.name === 'pageup' ? -1 : 1) * Math.max(1, size.height - 10))
  })
  const wide = size.width >= 100
  const count = wide ? Math.max(1, size.height - 14) : 3
  const start = Math.max(0, jobs.findIndex(row => row.id === selected) - count + 1)
  if (historyId) return <RunOverlay t={t} scheduleId={historyId} onClose={() => setHistoryId(null)} />
  return <box position="absolute" left={0} top={0} width="100%" height="100%" zIndex={150} backgroundColor="#000000cc" alignItems="center" justifyContent="center">
    <Box width={!jobs.length && !editing && !deliveryId && !templates ? Math.min(88, size.width) : size.width} height={!jobs.length && !editing && !deliveryId && !templates ? Math.min(24, size.height) : size.height} flexDirection="column" paddingX={1} borderStyle="round" borderColor={t.color.border} backgroundColor={t.color.overlayBg}>
      {templates ? <ScheduleTemplatePicker t={t} onClose={() => setTemplates(false)} onSelect={value => { setTemplate(value); setTemplates(false); setEditing('new') }} /> : deliveryId ? <DeliveryPanel t={t} scheduleId={deliveryId} onClose={() => { setDeliveryId(null); setRefresh(value => value + 1) }} /> : editing ? <ScheduleForm t={t} template={template} followup={followupsOnly} initial={editing === 'new' ? undefined : editing} onClose={() => setEditing(null)} onSaved={id => { setEditing(null); setSelected(id); setRefresh(value => value + 1) }} /> : <>
      <DialogHeader t={t} title={followupsOnly ? "Follow-ups · this conversation" : "Schedules · current workspace"} subtitle="Set a rhythm for your work. Review results when you return." />
      <Box onMouseDown={() => setTemplates(true)}><Text color={t.color.accent}>B  Browse templates · briefings, reviews and test reports</Text></Box>
      {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      {!jobs.length ? <DialogEmpty t={t} title={followupsOnly ? "No conversation follow-ups." : "No workspace schedules."} description="Schedule a task and return to its results." action="+ N  Create schedule" onAction={() => { setTemplate(undefined); setEditing('new') }} symbol="◷" /> : (<Box flexDirection={wide ? 'row' : 'column'} flexGrow={1} minHeight={0} gap={wide ? 2 : 0}>
        <Box width={wide ? '35%' : '100%'} flexDirection="column" paddingRight={1}>
          {jobs.length ? jobs.slice(start, start + count).map(row => <Text key={row.id} color={row.id === selected ? t.color.accent : t.color.text} wrap="truncate-end">{row.id === selected ? '› ' : '  '}{row.paused ? 'Ⅱ ' : '● '}{row.prompt}</Text>) : <DialogEmpty t={t} title={followupsOnly ? "No conversation follow-ups." : "No workspace schedules."} description="Give a recurring task its own schedule." action="+ N  Create schedule" onAction={() => { setTemplate(undefined); setEditing('new') }} symbol="◷" />}
        </Box>
        <scrollbox ref={scroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
          {job ? <Box flexDirection="column" flexShrink={0}>
                        {typeof job.metadata?.last_error === 'string' ? <Text color={t.color.warn} wrap="wrap">{job.metadata.last_error}</Text> : null}
            {job.metadata?.execution_recovery_required === true ? <Text color={t.color.warn} wrap="wrap">Review run output before resuming. Resume acknowledges possible prior effects; a one-shot may execute again.</Text> : null}
            {job.metadata?.delivery_state === 'failed' ? <>
              <Text color={t.color.warn} wrap="wrap">Output delivery failed: {String(job.metadata.delivery_error ?? 'Unknown delivery error')}</Text>
              {typeof job.metadata.delivery_archive === 'string' ? <Text color={t.ds.secondary} wrap="wrap">Archived: {job.metadata.delivery_archive}</Text> : null}
            </> : null}
<DialogSection t={t}>TASK</DialogSection><Text bold wrap="wrap">{job.prompt}</Text>
            <Text wrap="wrap">{job.paused ? 'Paused' : 'Enabled'} · {job.execution_state}</Text>
            <DialogSection t={t}>TIMING</DialogSection><Text wrap="wrap">Schedule: {job.interval_seconds != null ? scheduleDescription('', '', job.interval_seconds) : job.schedule ? scheduleDescription(job.schedule, job.timezone ?? 'UTC') : 'One-shot'}</Text>
            <Text wrap="wrap">Next: {job.next_run_at || 'Not scheduled'}</Text>
            <Text wrap="wrap">Missed runs: {job.missed_run_policy ?? "coalesce"} · Allowance: {job.misfire_grace_seconds ?? 300}s · Overlap: forbidden</Text>
            {job.metadata?.last_missed_run ? <Text color={t.color.warn} wrap="wrap">An overdue occurrence was skipped. Review timing before resuming a one-shot.</Text> : null}
            <DialogSection t={t}>DELIVERY</DialogSection><Text wrap="wrap">Delivery: {job.deliver && job.deliver !== "none" && job.deliver !== "workspace" ? `${job.deliver} → ${job.recipient || "recipient missing"}` : "Archive only"}</Text>
            <Text wrap="wrap">Target: {job.target_session_id ? "Conversation " + job.target_session_id : "Independent"}</Text>
            {job.stop_condition ? <Text wrap="wrap">Stop when: {job.stop_condition}</Text> : null}
            {job.metadata?.followup_completion ? <Text color={t.color.ok} wrap="wrap">Condition reported met by model. Future wakes stopped; explicit resume rearms.</Text> : null}
            {Array.isArray(job.metadata?.followup_completions) ? job.metadata.followup_completions.slice(-3).map((item, index) => {
              const value = item && typeof item === 'object' ? item as Record<string, unknown> : {}
              return typeof value.evidence === 'string' ? <Text key={index} wrap="wrap">Model report ({String(value.at ?? '')}): {value.evidence}</Text> : null
            }) : null}
            <DialogSection t={t}>LIMITS & USAGE</DialogSection><Text wrap="wrap">Expiry: {job.expires_at ?? "none"} (new attempts only)</Text>
            <Text wrap="wrap">Lifetime attempts: {job.runs_started ?? 0} / {job.max_runs ?? "unlimited"}</Text>
            <Text wrap="wrap">Model calls per run: {job.max_model_calls ?? "unlimited"}</Text>
            <Text wrap="wrap">{scheduleUsageLabel(job.metadata?.token_usage)}</Text>
            <Text wrap="wrap">{scheduleTokensLabel(parseScheduleTokens((job as Job & { token_budget?: unknown }).token_budget))}</Text>
            <Text wrap="wrap">Token threshold blocks new calls; in-flight calls may overshoot.</Text>
            <DialogSection t={t}>LAST RUN</DialogSection><Text wrap="wrap">Last: {job.last_run_at || 'Never'}</Text>
            <Text wrap="wrap">Timeout: {job.timeout_seconds == null ? 'daemon default' : `${job.timeout_seconds}s`} · One-shot retries: {job.max_retries ?? 'daemon default'}</Text>
            <Text color={t.ds.secondary} wrap="wrap">Pause stops future runs. Cancel requests cleanup of the active run.</Text>
          </Box> : <Text color={t.ds.secondary} wrap="wrap">Press N to create a job. Jobs run while the owning daemon is online.</Text>}
        </scrollbox>
      </Box>)}

      <DialogFooter t={t}>{jobs.length ? <><Text color={t.ds.secondary}>D deliveries</Text>
      <Text color={t.ds.secondary}>P pause/resume · G run · X cancel</Text>
      <Text color={t.ds.secondary}>N new · E edit · H history · Esc</Text></> : <Text color={t.ds.secondary}>N new · R refresh · Esc close</Text>}</DialogFooter>
      </>}
    </Box>
  </box>
}
