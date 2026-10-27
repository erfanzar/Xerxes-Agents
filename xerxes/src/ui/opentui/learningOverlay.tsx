// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import type { Lesson } from '../../memory/lessons.js'
import type { StoredReflectionProposal } from '../../runtime/reflection.js'
import { useGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme, getUiState } from '../app/uiStore.js'
import { applyLearningCounts } from '../app/useLearningSync.js'
import type { LearningListResponse, LearningReviewParams, LearningReviewResponse } from '../gatewayTypes.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import { overlayPanelSize, windowItems } from './overlayLayout.js'
import { Box, Text } from './primitives.js'

type ReviewRow = { target: 'proposal'; item: StoredReflectionProposal } | { target: 'lesson'; item: Lesson }
const rowStatus = (row: ReviewRow) => row.target === 'proposal' ? row.item.status : row.item.reviewStatus
const rowLabel = (row: ReviewRow) => row.target === 'proposal' ? row.item.proposal.target : row.item.cue

/** Full, plain-text evidence and content, never Markdown-rendered executable instructions. */
export function learningDetail(row: ReviewRow): string {
  if (row.target === 'lesson') {
    const item = row.item
    const evidence = item.evidence.kind === 'test'
      ? `${item.evidence.command} · exit ${item.evidence.exitCode}\n${item.evidence.summary}`
      : item.evidence.kind === 'shared'
        ? `Shared from ${item.evidence.host} via ${item.evidence.source}\n${item.evidence.summary}`
        : `${item.evidence.toolName} · ${item.evidence.toolCallId}\n${item.evidence.summary}`
    return `Cue: ${item.cue}\nStatus: ${item.reviewStatus}\nConfidence: ${item.confidence}\n\nLesson\n${item.lesson}\n\nEvidence\n${evidence}`
  }
  const item = row.item
  const proposal = item.proposal
  const content = proposal.kind === 'agent_update' ? proposal.agent.yaml
    : proposal.kind === 'plugin_idea' ? `Idea only; no code is written.\n${proposal.plugin.description}`
    : `Description\n${proposal.skill.description}\n\nInstructions\n${proposal.skill.instructions}`
      + Object.entries(proposal.skill.files ?? {}).map(([path, text]) => `\n\nFile: ${path}\n${text}`).join('')
  return `${proposal.kind} ${proposal.target}\nStatus: ${item.status}\n\nRationale\n${proposal.rationale}\n\nQuoted evidence\n${proposal.evidence.map(line => `> ${line}`).join('\n')}\n\nProposed content\n${content}`
    + (item.reason ? `\n\nReason\n${item.reason}` : '')
    + (item.outcome ? `\n\nOutcome\n${JSON.stringify(item.outcome, null, 2)}` : '')
    + (item.hint ? `\n\n${item.hint}` : '')
}

/** A modal projection over the transcript. Closing never replaces history or the composer. */
export function LearningOverlay() {
  const { rpc } = useGateway()
  const t = useStore($uiTheme)
  const sid = useStore($uiSessionId)
  const terminal = useTerminalDimensions()
  const size = overlayPanelSize(terminal, { maxWidth: 100, minWidth: 36, maxHeight: 38, gutterY: 2 })
  const [data, setData] = useState<Extract<LearningListResponse, { ok: true }> | null>(null)
  const [tab, setTab] = useState<'proposal' | 'lesson'>('proposal')
  const [selected, setSelected] = useState(0)
  const [detail, setDetail] = useState(false)
  const [reasonAction, setReasonAction] = useState<'reject' | 'rollback' | null>(null)
  const [reason, setReason] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const actionLock = useRef(false)
  const epoch = useRef(0)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  const rows: ReviewRow[] = tab === 'proposal'
    ? (data?.proposals ?? []).map(item => ({ target: 'proposal', item }))
    : (data?.lessons ?? []).map(item => ({ target: 'lesson', item }))
  const selectedIndex = Math.min(selected, Math.max(0, rows.length - 1))
  const row = rows[selectedIndex]
  const pending = row && rowStatus(row) === 'pending'
  const canRollback = row?.target === 'lesson' && row.item.reviewStatus === 'approved'

  const refresh = useCallback(async (token: number) => {
    const result = await rpc<LearningListResponse>('learning.list', sid ? { session_key: sid } : {}, { reportError: false })
    if (token !== epoch.current || getUiState().sid !== sid) return
    if (!result?.ok) throw new Error(result?.error || 'No learning list response; retry with Ctrl+R.')
    setData(result)
    applyLearningCounts(result.workspace, result.counts)
  }, [rpc, sid])

  useEffect(() => {
    const token = ++epoch.current
    actionLock.current = false
    setBusy(false); setData(null); setSelected(0); setDetail(false); setReasonAction(null); setError(''); setMessage('')
    void refresh(token).catch(cause => { if (token === epoch.current) setError(rpcErrorMessage(cause)) })
    return () => { ++epoch.current }
  }, [refresh])

  const review = async (action: LearningReviewParams['action']) => {
    if (!row || actionLock.current || (action === 'rollback' ? !canRollback : !pending)) return
    if (action !== 'approve' && !reason.trim()) { setError('Reason required'); return }
    actionLock.current = true
    setBusy(true); setError('')
    const token = epoch.current
    try {
      const result = await rpc<LearningReviewResponse>('learning.review', {
        ...(sid ? { session_key: sid } : {}), target: row.target, id: row.item.id, action,
        ...(action === 'approve' ? {} : { reason: reason.trim() })
      }, { reportError: false })
      if (token !== epoch.current || getUiState().sid !== sid) return
      if (!result) throw new Error('No review response; refresh before retrying.')
      setMessage(result.report || (result.ok ? 'Review saved' : 'Review refused'))
      if (!result.ok) setError(result.error || 'Review refused')
      else { setReasonAction(null); setReason('') }
      // Refresh even a failed application: the proposal may now have a failed
      // outcome. Preserve the report if that second request itself fails.
      await refresh(token)
    } catch (cause) {
      if (token === epoch.current) setError(rpcErrorMessage(cause))
    } finally {
      if (token === epoch.current) { actionLock.current = false; setBusy(false) }
    }
  }

  useKeyboard(event => {
    const key = event.name
    if (key === 'escape') {
      event.preventDefault(); event.stopPropagation()
      if (reasonAction) { setReasonAction(null); setReason('') }
      else if (detail) setDetail(false)
      else patchOverlayState({ learning: false })
      return
    }
    if (busy) { event.preventDefault(); event.stopPropagation(); return }
    if (reasonAction) return // The focused inline input owns editing and submit.
    event.preventDefault(); event.stopPropagation()
    if (key === 'tab') { setTab(tab === 'proposal' ? 'lesson' : 'proposal'); setSelected(0); setDetail(false); return }
    if (event.ctrl && key === 'r') {
      const token = ++epoch.current
      setError('')
      void refresh(token).catch(cause => { if (token === epoch.current) setError(rpcErrorMessage(cause)) })
      return
    }
    if (key === 'return' || key === 'enter') { setDetail(!detail); return }
    if (key === 'a' && pending) { void review('approve'); return }
    if ((key === 'r' && pending) || (key === 'b' && canRollback)) {
      setReasonAction(key === 'r' ? 'reject' : 'rollback'); setReason(''); setError(''); return
    }
    if (detail) {
      if (key === 'up' || key === 'down') scroll.current?.scrollBy(key === 'up' ? -1 : 1)
      if (key === 'pageup' || key === 'pagedown') scroll.current?.scrollBy((key === 'pageup' ? -1 : 1) * Math.max(1, size.height - 8))
      if (key === 'home') scroll.current?.scrollTo(0)
      if (key === 'end') scroll.current?.scrollTo(1_000_000)
    } else {
      if (key === 'up' || key === 'down') setSelected(Math.max(0, Math.min(rows.length - 1, selectedIndex + (key === 'up' ? -1 : 1))))
      if (key === 'home' || key === 'end') setSelected(key === 'home' ? 0 : Math.max(0, rows.length - 1))
    }
  })

  const window = windowItems(rows, selectedIndex, Math.max(1, size.height - 10))
  const compact = terminal.width < 60
  return <box alignItems="center" backgroundColor="#000000cc" height="100%" justifyContent="center" left={0} position="absolute" top={0} width="100%" zIndex={180}>
    <Box backgroundColor={t.color.overlayBg} borderStyle="round" borderColor={t.color.border} flexDirection="column" paddingX={1} width={size.width} height={size.height}>
      <Text bold color={t.color.accent}>What Xerxes learned</Text>
      <Text color={t.ds.secondary} wrap="truncate-end">{`${tab === 'proposal' ? '› ' : ''}Proposals ${data?.proposals.length ?? 0} · ${tab === 'lesson' ? '› ' : ''}Lessons ${data?.lessons.length ?? 0}`}</Text>
      {!compact ? <Text color={t.ds.meta}>Nothing applies without your approval. /xerxes help: typed subcommands.</Text> : null}
      {detail && (row || message || error) ? <scrollbox key={row ? `${row.target}:${row.item.id}` : 'review-report'} ref={scroll} style={{ flexGrow: 1, minHeight: 0 }} contentOptions={{ flexDirection: 'column' }}>
        <Text color={t.color.text} wrap="wrap">{[row ? learningDetail(row) : '', message ? `Review report\n${message}` : '', error ? `Error\n${error}` : ''].filter(Boolean).join('\n\n')}</Text>
      </scrollbox> : <Box flexDirection="column" flexGrow={1} minHeight={0} overflow="hidden">
        {!data ? <Text color={t.ds.meta}>{error ? 'List unavailable · Ctrl+R retry' : 'Loading…'}</Text> : !rows.length ? <Text color={t.ds.meta}>Nothing here to review.</Text> : window.items.map((item, index) => <Box key={item.item.id} backgroundColor={selectedIndex === window.offset + index ? t.ds.selected : undefined} flexShrink={0}>
          <Text color={t.color.text} wrap="truncate-end">{`${selectedIndex === window.offset + index ? '› ' : '  '}${rowLabel(item)} · ${rowStatus(item)}`}</Text>
        </Box>)}
      </Box>}
      {reasonAction ? <Box flexDirection="column" flexShrink={0}>
        <Text color={t.color.warn}>{reasonAction === 'reject' ? 'Reject reason (required)' : 'Rollback reason (required)'}</Text>
        <input focused={!busy} value={reason} onInput={setReason} onSubmit={() => { void review(reasonAction) }} placeholder="Reason required" backgroundColor={t.color.completionMetaBg} textColor={t.color.text} />
      </Box> : null}
      {message || error ? <scrollbox style={{ maxHeight: Math.max(2, Math.floor(size.height / 4)), flexShrink: 0 }} contentOptions={{ flexDirection: 'column' }}>
        {message ? <Text color={t.ds.secondary} wrap="wrap">{message}</Text> : null}
        {error ? <Text color={t.color.warn} wrap="wrap">{error}</Text> : null}
      </scrollbox> : null}
      <Text color={t.ds.secondary} wrap="truncate-end">{reasonAction ? 'Enter save · Esc cancel' : 'Tab section · ↑↓ move · Enter details'}</Text>
      <Text color={t.ds.secondary} wrap="truncate-end">{busy ? 'Saving… · Esc back' : `${pending ? 'a approve · r reject · ' : canRollback ? 'b rollback · ' : ''}Esc ${detail ? 'back' : 'close'}`}</Text>
    </Box>
  </box>
}
