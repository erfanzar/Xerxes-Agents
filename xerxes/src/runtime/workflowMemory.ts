// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { AgentMemory, AgentMemoryScope } from '../memory/agentMemory.js'

/** Durable memory file used only for explicit user workflow instructions. */
export const WORKFLOW_MEMORY_FILE = 'WORKFLOW.md'

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
  readonly reason?: 'duplicate' | 'empty' | 'memory_unavailable' | 'no_signal' | 'write_rejected'
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
 * This is deliberately narrow: normal conversation is never written
 * automatically, and it complements rather than replaces the agent-memory
 * tools. The note is written before prompt construction so an explicit
 * instruction can be visible to the current turn.
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

  const scope = memory.hasProjectScope() ? AgentMemoryScope.PROJECT : AgentMemoryScope.GLOBAL
  let existing: string
  try {
    existing = await memory.read(scope, WORKFLOW_MEMORY_FILE)
  } catch (error) {
    if (!(error instanceof ValidationError)) throw error
    existing = '# Workflow Memory\n\nDurable user workflow instructions and project operating notes.\n'
  }
  if (existing.includes(message)) return { captured: false, reason: 'duplicate', scope }

  const note = formatWorkflowMemoryNote(message, {
    ...(options.clock === undefined ? {} : { clock: options.clock }),
    ...(options.projectRoot === undefined ? {} : { projectRoot: options.projectRoot }),
  })
  try {
    const result = await memory.write(scope, WORKFLOW_MEMORY_FILE, appendNote(existing, note))
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

/** Format one user-provided workflow instruction as a dated Markdown entry. */
export function formatWorkflowMemoryNote(message: string, options: CaptureWorkflowMemoryOptions = {}): string {
  const instruction = message.trim()
  if (!instruction) throw new ValidationError('message', 'must be a non-empty workflow instruction', message)
  const now = options.clock?.() ?? new Date()
  if (Number.isNaN(now.valueOf())) throw new ValidationError('clock', 'must return a valid date')
  const root = options.projectRoot?.trim()
  const projectLine = root ? `\n**Project root:** \`${resolve(root)}\`` : ''
  return `## ${now.toISOString()} - user workflow note${projectLine}\n\n**Instruction:** ${instruction}`
}

function appendNote(existing: string, note: string): string {
  return existing.trimEnd() ? existing.trimEnd() + '\n\n' + note + '\n' : note + '\n'
}
