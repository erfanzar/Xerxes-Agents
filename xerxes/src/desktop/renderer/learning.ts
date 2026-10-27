// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Lesson } from '../../memory/lessons.js'
import type { ReflectionProposal, StoredReflectionProposal } from '../../runtime/reflection.js'
import { record, records, text } from './desktopRpc.js'

export type { Lesson, ReflectionProposal, StoredReflectionProposal }
export type LearningTarget = 'proposal' | 'lesson'
export type LearningAction = 'approve' | 'reject' | 'rollback'
export interface LearningCounts { readonly pending_proposals: number; readonly pending_lessons: number }
export interface LearningList {
  readonly workspace: string
  readonly proposals: readonly StoredReflectionProposal[]
  readonly lessons: readonly Lesson[]
  readonly counts: LearningCounts
}
export interface LearningResult { readonly report: string; readonly error: string }
export interface ReflectionNotice {
  readonly id: string
  readonly kind: ReflectionProposal['kind']
  readonly target: string
  readonly rationale: string
  readonly evidence_count: number
}

export function proposalKindLabel(kind: ReflectionProposal['kind']): string {
  return { skill_create: 'New skill', skill_update: 'Skill update', agent_update: 'Agent update', plugin_idea: 'Plugin idea' }[kind]
}
function proposalKind(value: unknown): value is ReflectionProposal['kind'] {
  return value === 'skill_create' || value === 'skill_update' || value === 'agent_update' || value === 'plugin_idea'
}
function count(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 }
function hasText(value: unknown): value is string { return typeof value === 'string' && value.trim().length > 0 }
export function learningCounts(value: unknown): LearningCounts {
  const row = record(value)
  if (!count(row.pending_proposals) || !count(row.pending_lessons)) throw new Error('Invalid learned-item counts from the runtime.')
  return { pending_proposals: row.pending_proposals, pending_lessons: row.pending_lessons }
}
export function reflectionNotice(value: unknown): ReflectionNotice | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = record(value)
  if (!hasText(row.id) || !proposalKind(row.kind) || !hasText(row.target) || !hasText(row.rationale) || !count(row.evidence_count)) return null
  return { id: row.id, kind: row.kind, target: row.target, rationale: row.rationale, evidence_count: row.evidence_count }
}
function readProposal(value: unknown): StoredReflectionProposal {
  const row = record(value), proposal = record(row.proposal)
  const content = proposal.kind === 'agent_update' ? record(proposal.agent) : proposal.kind === 'plugin_idea' ? record(proposal.plugin) : record(proposal.skill)
  if (!hasText(row.id) || !hasText(row.workspace) || !['pending', 'applying', 'applied', 'failed', 'rejected', 'acknowledged'].includes(text(row.status))
    || typeof row.createdAt !== 'number' || !Number.isFinite(row.createdAt) || typeof row.updatedAt !== 'number' || !Number.isFinite(row.updatedAt)
    || !proposalKind(proposal.kind) || !hasText(proposal.target) || !hasText(proposal.rationale)
    || !Array.isArray(proposal.evidence) || !proposal.evidence.every(line => typeof line === 'string')
    || (proposal.kind === 'agent_update' ? !hasText(content.yaml) : proposal.kind === 'plugin_idea' ? !hasText(content.description) : !hasText(content.description) || !hasText(content.instructions))) {
    throw new Error('Invalid reflection proposal from the runtime. Refresh before reviewing it.')
  }
  if (content.files !== undefined && !Object.values(record(content.files)).every(body => typeof body === 'string')) throw new Error('Invalid proposed skill files.')
  for (const key of ['reason', 'hint'] as const) if (row[key] !== undefined && typeof row[key] !== 'string') throw new Error('Invalid proposal outcome.')
  if (row.outcome !== undefined) record(row.outcome)
  return row as unknown as StoredReflectionProposal
}
function readLesson(value: unknown): Lesson {
  const row = record(value), evidence = record(row.evidence)
  if (!hasText(row.id) || !hasText(row.workspace) || !hasText(row.cue) || !hasText(row.lesson) || !hasText(row.agentId)
    || typeof row.createdAt !== 'number' || !Number.isFinite(row.createdAt) || typeof row.confidence !== 'number' || !Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1
    || (row.reviewStatus !== 'pending' && row.reviewStatus !== 'approved') || !hasText(evidence.summary)
    || (evidence.kind === 'tool_call' ? !hasText(evidence.toolName) || !hasText(evidence.toolCallId)
      : evidence.kind === 'shared' ? !hasText(evidence.host) || (evidence.source !== 'repo' && evidence.source !== 'hub')
        : evidence.kind !== 'test' || !hasText(evidence.command) || typeof evidence.exitCode !== 'number' || !Number.isSafeInteger(evidence.exitCode))) {
    throw new Error('Invalid lesson from the runtime. Refresh before reviewing it.')
  }
  return row as unknown as Lesson
}
export function readLearningList(value: unknown): LearningList {
  const row = record(value)
  if (row.ok !== true) throw new Error(text(row.error) || 'Could not load what Xerxes learned.')
  if (!hasText(row.workspace)) throw new Error('The runtime did not identify this learning workspace.')
  const proposals = records(row.proposals).map(readProposal), lessons = records(row.lessons).map(readLesson)
  if (proposals.some(item => item.workspace !== row.workspace)
    || lessons.some(item => item.workspace !== row.workspace && !(typeof row.repo === 'string' && /^git:[0-9a-f]{32}$/u.test(row.repo) && item.repo === row.repo))
    || new Set(proposals.map(item => item.id)).size !== proposals.length || new Set(lessons.map(item => item.id)).size !== lessons.length) {
    throw new Error('The runtime returned duplicate or cross-workspace learned items. Refresh before reviewing.')
  }
  return { workspace: row.workspace, proposals, lessons, counts: learningCounts(row.counts) }
}

/** Exact proposed fields, never a Markdown interpretation of model-written content. */
export function proposalPreview(proposal: ReflectionProposal): readonly { title: string; content: string }[] {
  if (proposal.kind === 'agent_update') return [{ title: 'Agent YAML', content: proposal.agent.yaml }]
  if (proposal.kind === 'plugin_idea') return [{ title: 'Plugin description (idea only; no code is written)', content: proposal.plugin.description }]
  return [{ title: 'Skill description', content: proposal.skill.description }, { title: 'Skill instructions', content: proposal.skill.instructions }, ...Object.entries(proposal.skill.files ?? {}).map(([title, content]) => ({ title, content }))]
}
