// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry } from '../executors/toolRegistry.js'
import { LessonError, LessonStore, type LessonEvidenceInput, type ResolveLessonToolCall } from '../memory/lessons.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export interface LessonToolsOptions {
  /** Resolve a governed lesson store for this project root, not a global chat. */
  readonly store: (workspace: string) => LessonStore | Promise<LessonStore>
  readonly resolveToolCall?: ResolveLessonToolCall
  readonly activeWorkspace: () => string
}

export const LESSON_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'lesson_record',
      description: 'Record an evidence-backed lesson for this workspace. Supply a short retrieval cue, the general lesson, and confidence. '
        + 'Usually omit evidence_tool_call_id: the host cites the latest completed successful tool call in this session, or verifies the explicit id. '
        + 'Alternatively supply test_evidence with an actual command, integer exit_code, and result summary; these are labelled reported, not host-verified. '
        + 'Missing or failed tool-call evidence is refused. Lessons start pending review; never invent evidence.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['cue', 'lesson', 'confidence'],
        properties: {
          cue: { type: 'string', minLength: 1, maxLength: 240, description: 'Short trigger phrase for future retrieval, e.g. Bun fixture directories.' },
          lesson: { type: 'string', minLength: 1, description: 'General actionable lesson supported by the cited evidence, not task-specific answers.' },
          confidence: { type: 'number', minimum: 0, maximum: 1 },
          evidence_tool_call_id: { type: 'string', minLength: 1, description: 'Optional exact call id; omit to resolve the latest successful session call. Do not combine with test_evidence.' },
          test_evidence: {
            type: 'object', additionalProperties: false, required: ['command', 'exit_code', 'summary'],
            description: 'Alternative reported test evidence. Failed tests may teach lessons too; report the actual exit code.',
            properties: {
              command: { type: 'string', minLength: 1 },
              exit_code: { type: 'integer' },
              summary: { type: 'string', minLength: 1, description: 'Observed test result and what it establishes.' },
            },
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'lesson_recall',
      description: 'Recall evidence-backed lessons from the active workspace only. Matches cue tokens case-insensitively, then lesson text; approved lessons rank before pending. Rejected, expired, and superseded lessons are excluded.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['cue'], properties: {
          cue: { type: 'string', minLength: 1, description: 'Trigger words describing the current situation.' },
          limit: { type: 'integer', minimum: 1, maximum: 100, description: 'Maximum returned lessons; defaults to 10.' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'lesson_review',
      description: 'Approve, reject, or roll back a lesson from the active workspace with human review and the normal permission gate. Automatic turns cannot approve their own lessons. Review uses governed memory; rollback expires the lesson while retaining the evidence and reason in the append-only audit log. Copy an exact id from lesson_recall and explain the decision.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['id', 'action', 'reason'], properties: {
          id: { type: 'string', minLength: 1 },
          action: { type: 'string', enum: ['approve', 'reject', 'rollback'] },
          reason: { type: 'string', minLength: 1, description: 'Concrete explanation for this review or rollback, retained in the governed log.' },
        },
      },
    },
  },
]

export function registerLessonTools(registry: ToolRegistry, options: LessonToolsOptions, agentId = 'default'): void {
  for (const definition of LESSON_TOOL_DEFINITIONS) {
    const name = definition.function.name
    registry.register(definition, (inputs, context) => refusalResult(async () => {
      const workspace = options.activeWorkspace()
      const store = await options.store(workspace)
      const sessionId = context.sessionId
      switch (name) {
        case 'lesson_record':
          return { ok: true, lesson: await store.record({
            cue: text(inputs.cue, 'cue'), lesson: text(inputs.lesson, 'lesson'), confidence: confidence(inputs.confidence),
            evidence: evidenceInput(inputs), workspace, agentId: context.agentId ?? agentId,
            ...(sessionId === undefined ? {} : { sessionId }),
          }, options.resolveToolCall) }
        case 'lesson_recall':
          return { ok: true, lessons: await store.recall(text(inputs.cue, 'cue'), workspace, inputs.limit === undefined ? {} : { limit: limit(inputs.limit) }) }
        case 'lesson_review': {
          const id = text(inputs.id, 'id')
          const reason = text(inputs.reason, 'reason')
          const action = text(inputs.action, 'action')
          if (action === 'approve' && context.metadata.goal_turn_human === false) {
            throw new LessonError('Approving a lesson requires a direct human turn and explicit human review; automatic turns cannot approve guidance for future turns.', 'LESSON_HUMAN_APPROVAL_REQUIRED')
          }
          if (action === 'rollback') {
            await store.rollback(id, reason, workspace)
            return { ok: true, id, action }
          }
          if (action !== 'approve' && action !== 'reject') {
            throw new LessonError('Lesson review action must be approve, reject, or rollback; no other mutations are supported.', 'LESSON_INVALID_INPUT')
          }
          return { ok: true, lesson: await store.review(id, action === 'approve' ? 'approved' : 'rejected', reason, workspace) }
        }
        default:
          throw new Error(`Unknown lesson tool: ${name}`)
      }
    }), agentId, {
      concurrencySafe: name === 'lesson_recall', defer: true, destructive: name === 'lesson_review', openWorld: false,
      readOnly: name === 'lesson_recall', interruptBehavior: name === 'lesson_recall' ? 'cancel' : 'block',
    })
  }
}

async function refusalResult(operation: () => Promise<unknown>): Promise<unknown> {
  try { return await operation() } catch (error) {
    if (!(error instanceof LessonError)) throw error
    return { ok: false, error: error.message, code: error.code }
  }
}

function evidenceInput(inputs: JsonObject): LessonEvidenceInput {
  if (inputs.test_evidence !== undefined) {
    if (inputs.evidence_tool_call_id !== undefined) {
      throw new LessonError('Choose one evidence source: test_evidence or evidence_tool_call_id, not both.', 'LESSON_INVALID_EVIDENCE')
    }
    const test = inputs.test_evidence
    if (!test || typeof test !== 'object' || Array.isArray(test) || typeof test.command !== 'string'
      || typeof test.exit_code !== 'number' || typeof test.summary !== 'string') {
      throw new LessonError('test_evidence needs a command, integer exit_code, and non-empty summary of the actual result.', 'LESSON_INVALID_EVIDENCE')
    }
    return { kind: 'test', command: test.command, exitCode: test.exit_code, summary: test.summary }
  }
  if (inputs.evidence_tool_call_id !== undefined && typeof inputs.evidence_tool_call_id !== 'string') {
    throw new LessonError('evidence_tool_call_id must be text; omit it to resolve the most recent successful session call.', 'LESSON_INVALID_EVIDENCE')
  }
  return { kind: 'tool_call', ...(inputs.evidence_tool_call_id === undefined ? {} : { toolCallId: inputs.evidence_tool_call_id }) }
}

function text(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new LessonError(`Lesson ${field} must be non-empty text; supply a concrete value before retrying.`, 'LESSON_INVALID_INPUT')
  }
  return value.trim()
}

function confidence(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new LessonError('Lesson confidence must be a finite number between 0 and 1, inclusive.', 'LESSON_INVALID_INPUT')
  }
  return value
}

function limit(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > 100) {
    throw new LessonError('Lesson recall limit must be an integer between 1 and 100.', 'LESSON_INVALID_INPUT')
  }
  return value
}

