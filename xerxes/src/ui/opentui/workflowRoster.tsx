// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { patchOverlayState } from '../app/overlayStore.js'
import { $spawnHistory, spawnHistoryForSession } from '../app/spawnHistoryStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { fmtDuration, subagentElapsedSeconds } from '../lib/subagentElapsed.js'
import { fmtK } from '../lib/text.js'
import { summarizeWorkflowRoster } from '../lib/workflowRoster.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

import { Box, Span, Text } from './primitives.js'

const MAX_RUNNING_ROWS = 6
const MAX_FAILED_ROWS = 4

/**
 * Under a Workflow row: one line per phase with its tally, then a line for
 * each agent still running and each that failed. Thousands of agents stay a
 * handful of lines; the agents panel (F6) holds the full list.
 */
export function WorkflowRoster({ archived = [], label, t }: { archived?: readonly SubagentProgress[]; label: string; t: Theme }) {
  const live = useTurnSelector(state => state.subagents)
  const history = useStore($spawnHistory)
  const sessionId = useStore($uiState).sid
  const summary = useMemo(() => {
    const past = [...spawnHistoryForSession(history, sessionId)].reverse().flatMap(snapshot => snapshot.subagents)
    return summarizeWorkflowRoster([...past, ...archived, ...live], label)
  }, [archived, history, label, live, sessionId])
  if (!summary.total) return null
  const row = (agent: SubagentProgress, color: string, glyph: string) => {
    const elapsed = subagentElapsedSeconds(agent)
    const tokens = agent.inputTokens !== undefined ? ` · ${fmtK(agent.inputTokens + (agent.outputTokens ?? 0))} tok` : ''
    const model = agent.model ? ` · ${agent.model.split('/').pop()}` : ''
    const detail = agent.status === 'running' ? agent.notes.at(-1) ?? '' : agent.summary ?? agent.status
    return (
      <Box key={agent.id} onClick={() => patchOverlayState({ agents: true, agentsInspectId: agent.id })}>
        <Text wrap="truncate-end">
          <Span color={color}>{`      ${glyph} `}</Span>
          <Span color={t.color.text}>{agent.title || agent.name || agent.id}</Span>
          <Span color={t.color.muted}>{`${model}${elapsed === null ? '' : ` [${fmtDuration(elapsed)}]`}${tokens}${detail ? ` · ${detail.split('\n')[0]}` : ''}`}</Span>
        </Text>
      </Box>
    )
  }
  return (
    <Box flexDirection="column" flexShrink={0}>
      {summary.phases.map(phase => (
        <Text key={phase.name || 'batch'} wrap="truncate-end">
          <Span color={phase.working ? '#6487ff' : phase.failed ? t.color.error : t.color.ok}>{phase.working ? '    ▸ ' : phase.failed ? '    ✗ ' : '    ✓ '}</Span>
          <Span color={t.color.text}>{phase.name || 'agents'}</Span>
          <Span color={t.color.muted}>{` · ${phase.total} agent${phase.total === 1 ? '' : 's'}${phase.working ? ` · ${phase.working} running` : ''}${phase.done ? ` · ${phase.done} done` : ''}${phase.failed ? ` · ${phase.failed} failed` : ''}`}</Span>
        </Text>
      ))}
      {summary.running.slice(0, MAX_RUNNING_ROWS).map(agent => row(agent, '#6487ff', '⣾'))}
      {summary.failures.slice(0, MAX_FAILED_ROWS).map(agent => row(agent, t.color.error, '✗'))}
      {summary.running.length + summary.failures.length > MAX_RUNNING_ROWS + MAX_FAILED_ROWS ? (
        <Text color={t.color.muted} wrap="truncate-end">{'      … the rest in the agents panel (F6)'}</Text>
      ) : null}
    </Box>
  )
}
