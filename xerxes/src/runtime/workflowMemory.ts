// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { AgentMemory, AgentMemoryScope } from '../memory/agentMemory.js'
import { findCredentialPatterns } from '../security/promptScanner.js'

/** Legacy single-file workflow store, still read so pre-topic captures dedupe. */
export const WORKFLOW_MEMORY_FILE = 'WORKFLOW.md'

/** Directory holding one topic file per captured workflow instruction. */
export const WORKFLOW_TOPIC_DIRECTORY = 'topics'

/**
 * Ceiling on a single captured instruction.
 *
 * An explicit "remember this" attached to a pasted log or file dump would
 * otherwise commit the whole paste to durable memory. Above the ceiling the
 * capture declines and the agent can still save a distilled note deliberately.
 */
export const MAX_WORKFLOW_MEMORY_CHARACTERS = 2_000

export const EXPLICIT_MEMORY_MARKERS = Object.freeze([
  'remember',
  'keep in memory',
  'save this',
  'note this',
  'note that',
  'for your memory',
  'my workflow',
  'real workflow',
])

export const PROJECT_WORKFLOW_MARKERS = Object.freeze([
  'workflow',
  'big project',
  'big projects',
  'large project',
  'large projects',
  'large repo',
  'large repos',
  'codebase',
  'codebases',
  'project',
])

export const PROJECT_MEMORY_INTENTS = Object.freeze([
  'remember',
  'save',
  'learn',
  'understand',
  'use this',
  'always know',
  'keep track',
])

export interface WorkflowMemoryCapture {
  readonly captured: boolean
  readonly path?: string
  readonly reason?:
    | 'credential_blocked'
    | 'duplicate'
    | 'empty'
    | 'memory_unavailable'
    | 'no_signal'
    | 'too_long'
    | 'write_rejected'
  readonly scope?: AgentMemoryScope
}

export interface CaptureWorkflowMemoryOptions {
  /** Injectable clock keeps note timestamps testable and avoids hidden wall-clock coupling. */
  readonly clock?: () => Date
  /** Project root is rendered as context only; memory scope comes from the memory instance. */
  readonly projectRoot?: string
}

/**
 * Persist an explicit user request to remember workflow/project information.
 *
 * Honouring an explicit save request is the entire point of this function:
 * when the user says "remember this", refusing is the wrong answer. The
 * guards are about hygiene, not intent — an oversized paste, a credential, and
 * an unbounded append into an always-injected file are the three ways this
 * turns into a liability, so each is handled instead of declining the request.
 * The note lands in its own topic file, which reaches the prompt as a single
 * manifest line rather than as a growing timestamped log.
 */
export async function captureUserWorkflowMemory(
  userMessage: string,
  memory: AgentMemory | undefined,
  options: CaptureWorkflowMemoryOptions = {},
): Promise<WorkflowMemoryCapture> {
  const message = userMessage.trim()
  if (!message) return { captured: false, reason: 'empty' }
  if (!shouldCaptureWorkflowMemory(message)) return { captured: false, reason: 'no_signal' }
  if (!memory) return { captured: false, reason: 'memory_unavailable' }
  if (message.length > MAX_WORKFLOW_MEMORY_CHARACTERS) return { captured: false, reason: 'too_long' }
  if (findCredentialPatterns(message).length > 0) return { captured: false, reason: 'credential_blocked' }

  const scope = memory.hasProjectScope() ? AgentMemoryScope.PROJECT : AgentMemoryScope.GLOBAL
  const path = workflowMemoryTopicPath(message)
  if (await workflowMemoryAlreadyRecorded(memory, scope, path, message)) {
    return { captured: false, reason: 'duplicate', scope }
  }

  const note = formatWorkflowMemoryNote(message, {
    ...(options.clock === undefined ? {} : { clock: options.clock }),
    ...(options.projectRoot === undefined ? {} : { projectRoot: options.projectRoot }),
  })
  try {
    const result = await memory.write(scope, path, note)
    return { captured: true, scope, path: result.path }
  } catch (error) {
    if (error instanceof ValidationError) return { captured: false, reason: 'write_rejected', scope }
    throw error
  }
}

/** Return whether a message explicitly asks Xerxes to retain workflow information. */
export function shouldCaptureWorkflowMemory(message: string): boolean {
  const normalized = message.trim().toLowerCase()
  if (!normalized) return false
  if (EXPLICIT_MEMORY_MARKERS.some(marker => normalized.includes(marker))) return true
  if (!normalized.includes('want')) return false
  return PROJECT_WORKFLOW_MARKERS.some(marker => normalized.includes(marker))
    && PROJECT_MEMORY_INTENTS.some(intent => normalized.includes(intent))
}

/**
 * Deterministic topic path for one instruction.
 *
 * The digest makes the same instruction resolve to the same file, so a repeat
 * request overwrites rather than accumulating near-duplicate topics.
 */
export function workflowMemoryTopicPath(message: string): string {
  const normalized = message.trim().toLowerCase()
  const slug = normalized
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .split('-')
    .filter(Boolean)
    .slice(0, 6)
    .join('-') || 'note'
  const digest = createHash('sha256').update(normalized, 'utf8').digest('hex').slice(0, 8)
  return `${WORKFLOW_TOPIC_DIRECTORY}/workflow-${slug}-${digest}.md`
}

/**
 * Format one user-provided workflow instruction as a topic file.
 *
 * The frontmatter is what the memory index renders: without a description the
 * topic shows up in the manifest as a bare filename and is effectively lost.
 */
export function formatWorkflowMemoryNote(message: string, options: CaptureWorkflowMemoryOptions = {}): string {
  const instruction = message.trim()
  if (!instruction) throw new ValidationError('message', 'must be a non-empty workflow instruction', message)
  const now = options.clock?.() ?? new Date()
  if (Number.isNaN(now.valueOf())) throw new ValidationError('clock', 'must return a valid date')
  const root = options.projectRoot?.trim()
  const projectLine = root ? `\n- **Project root:** \`${resolve(root)}\`` : ''
  return [
    '---',
    `name: ${frontmatterScalar(workflowTopicName(instruction))}`,
    `description: ${frontmatterScalar(workflowDescription(instruction))}`,
    'type: workflow',
    '---',
    '',
    '# User workflow note',
    '',
    `- **Recorded:** ${now.toISOString()}${projectLine}`,
    '',
    `**Instruction:** ${instruction}`,
    '',
  ].join('\n')
}

/** Whether this instruction is already stored, in its topic file or the legacy single file. */
async function workflowMemoryAlreadyRecorded(
  memory: AgentMemory,
  scope: AgentMemoryScope,
  path: string,
  message: string,
): Promise<boolean> {
  for (const candidate of [path, WORKFLOW_MEMORY_FILE]) {
    try {
      if ((await memory.read(scope, candidate)).includes(message)) return true
    } catch (error) {
      if (!(error instanceof ValidationError)) throw error
    }
  }
  return false
}

function workflowTopicName(instruction: string): string {
  return `workflow: ${firstSentence(instruction, 60)}`
}

function workflowDescription(instruction: string): string {
  return `User workflow instruction: ${firstSentence(instruction, 160)}`
}

function firstSentence(instruction: string, maximum: number): string {
  const single = instruction.replace(/\s+/g, ' ').trim()
  const sentence = single.split(/(?<=[.!?])\s/)[0] ?? single
  return sentence.length <= maximum ? sentence : sentence.slice(0, maximum - 3).trimEnd() + '...'
}

/** Keep a one-line frontmatter value from breaking the block it lives in. */
function frontmatterScalar(value: string): string {
  return value.replace(/[\r\n]+/g, ' ').trim()
}
