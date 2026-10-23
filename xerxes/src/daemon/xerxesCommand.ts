// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { IMPROVE_CLI_USAGE, runImproveCli } from '../../playground/improveCli.js'
import { renderLessons, type LessonStore } from '../memory/lessons.js'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'
import type { ReflectionApplier, ReflectionProposalStore, StoredReflectionProposal } from '../runtime/reflection.js'
import { runWithActiveSession } from '../runtime/sessionContext.js'
import type { DaemonSession } from './runtime.js'

export interface LearningServices {
  readonly store: ReflectionProposalStore
  readonly applier: ReflectionApplier
  readonly lessons: LessonStore
}

export const XERXES_SUBCOMMANDS = ['help', 'proposals', 'show', 'approve', 'reject', 'lessons', 'improve'] as const
export const LESSON_REVIEW_ACTIONS = ['approve', 'reject', 'rollback'] as const
export const XERXES_USAGE = [
  'Review what Xerxes learned: reflection proposals and lessons',
  'Usage:',
  '  /xerxes [help]',
  '  /xerxes proposals',
  '  /xerxes show <id> [--json]',
  '  /xerxes approve <id> [reason]',
  '  /xerxes reject <id> <reason>',
  '  /xerxes lessons [cue]',
  '  /xerxes lessons approve <id> [reason]',
  '  /xerxes lessons reject <id> <reason>',
  '  /xerxes lessons rollback <id> <reason>',
  '  /xerxes improve --proposals <file> --max-candidates N --max-task-runs N [options]',
  '',
  'Proposals never change files until a human approves. Plugin ideas only suggest /plugin-creator.',
  'Only approved matching lessons are automatically attached to future turns as reviewed guidance.',
  'improve always previews (--dry-run); execution requires an explicit transport in the CLI.',
].join('\n')

const UNAVAILABLE = 'Learning review is not configured. Start the native Xerxes daemon or inject its shared reflection store, applier and lesson store.'

function refusal(error: unknown): JsonRpcPayload {
  const message = error instanceof Error ? error.message : String(error)
  return { ok: false, report: message, error: message }
}

/** Read-only snapshot. Both clients and slash lists use the same visibility and ordering rules. */
export async function listLearning(workspace: string, services?: LearningServices): Promise<JsonRpcPayload> {
  if (!services) return { ok: false, error: UNAVAILABLE }
  try {
    const [records, lessons] = await Promise.all([services.store.list(workspace), services.lessons.list(workspace)])
    const proposals = [...records].sort((a, b) => b.createdAt - a.createdAt || b.id.localeCompare(a.id))
    return { ok: true, workspace, proposals, lessons, counts: {
      pending_proposals: proposals.filter(record => record.status === 'pending').length,
      pending_lessons: lessons.filter(lesson => lesson.reviewStatus === 'pending').length,
    } }
  } catch (error) { return refusal(error) }
}

/** RPC and typed slash review are human boundaries; model tools retain their own authority gates. */
export async function reviewLearning(params: Readonly<Record<string, unknown>>, workspace: string, services?: LearningServices, session?: DaemonSession, caller: { readonly principalKind: 'user' | 'agent' } = { principalKind: 'user' }): Promise<JsonRpcPayload> {
  // An agent principal (the XERXES supervisor) can never review what Xerxes learned, whoever asked it to.
  if (caller.principalKind === 'agent') return refusal('Learning review is a human-only action; agent principals cannot approve, reject or roll back.')
  if (!services) return refusal(UNAVAILABLE)
  const { target, id, action, reason } = params
  if (target !== 'proposal' && target !== 'lesson') return refusal('target must be proposal or lesson.')
  if (action !== 'approve' && action !== 'reject' && action !== 'rollback') return refusal('action must be approve, reject, or rollback.')
  if (typeof id !== 'string' || !id.trim()) return refusal('id must be non-empty text.')
  if (reason !== undefined && (typeof reason !== 'string' || !reason.trim())) return refusal('reason must be non-empty text when supplied.')
  if ((action === 'reject' || action === 'rollback') && reason === undefined) return refusal(`A reason is required to ${action}.`)
  if (target === 'proposal' && action === 'rollback') return refusal('Rollback is only valid for lessons.')
  const explanation = typeof reason === 'string' ? reason.trim() : undefined
  try {
    if (target === 'proposal') {
      const proposal = action === 'approve'
        ? await runWithActiveSession(session, () => services.store.approve(id.trim(), services.applier, { workspace, ...(explanation ? { reason: explanation } : {}) }))
        : await services.store.reject(id.trim(), explanation!, workspace)
      const report = [`Reflection proposal ${proposal.id}: ${proposal.status}.`, proposal.hint ?? '', proposal.outcome ? JSON.stringify(proposal.outcome, null, 2) : ''].filter(Boolean).join('\n')
      return { ok: proposal.status !== 'failed', proposal, report,
        ...(proposal.status === 'failed' ? { error: typeof proposal.outcome?.error === 'string' ? proposal.outcome.error : 'The approved proposal could not be applied; review its outcome.' } : {}) }
    }
    const current = await services.lessons.get(id.trim(), workspace)
    if (!current) return refusal('No active lesson with this id exists in this workspace.')
    if (action === 'approve' && current.reviewStatus !== 'pending') return refusal(`Lesson is ${current.reviewStatus}, not pending.`)
    if (action === 'rollback') {
      await services.lessons.rollback(current.id, explanation!, workspace)
      return { ok: true, lesson: current, report: `Lesson ${current.id}: rolled back.\nReason: ${explanation}` }
    }
    const lesson = await services.lessons.review(current.id, action === 'approve' ? 'approved' : 'rejected', explanation, workspace)
    return { ok: true, lesson, report: renderLessons([lesson], workspace) }
  } catch (error) { return refusal(error) }
}

export function renderReflectionProposal(record: StoredReflectionProposal): string {
  const proposal = record.proposal
  const lines = [
    `${proposal.kind} ${proposal.target}`, `ID: ${record.id}`, `Status: ${record.status}`,
    `Trigger: ${record.trigger ?? 'not recorded'}`, `Objective: ${record.objective ?? 'not recorded'}`,
    '', 'Rationale:', proposal.rationale, '', 'Evidence:', ...proposal.evidence.map(line => `> ${line}`),
    '', 'Exact proposed content (nothing is written until approval):',
  ]
  if (proposal.kind === 'skill_create' || proposal.kind === 'skill_update') {
    lines.push('Skill description:', proposal.skill.description, '', 'Skill instructions:', proposal.skill.instructions, '', 'Bundle files:')
    const files = Object.entries(proposal.skill.files ?? {})
    if (!files.length) lines.push('(no additional files)')
    for (const [path, content] of files) lines.push(`${path} (${new TextEncoder().encode(content).byteLength} bytes)`, content)
  } else if (proposal.kind === 'agent_update') lines.push('Agent YAML:', proposal.agent.yaml)
  else if (proposal.kind === 'plugin_idea') lines.push('Plugin idea (description only; no plugin is written):', proposal.plugin.description)
  if (record.reason) lines.push('', `Review reason: ${record.reason}`)
  if (record.outcome) lines.push('', 'Outcome:', JSON.stringify(record.outcome, null, 2))
  if (record.hint) lines.push('', `Hint: ${record.hint}`)
  return lines.join('\n')
}

export async function xerxesCommand(args: string, workspace: string, services?: LearningServices, session?: DaemonSession): Promise<JsonRpcPayload> {
  const [action = 'help', ...rest] = args.trim().split(/\s+/u).filter(Boolean)
  const id = rest[0]
  const reason = rest.slice(1).join(' ')
  try {
    if (action === 'help') return { ok: true, report: XERXES_USAGE }
    if (action === 'improve') {
      const improveArgs = tokenizeArguments(args.trim().slice('improve'.length))
      if (!improveArgs.length) return { ok: true, report: IMPROVE_CLI_USAGE }
      const output: string[] = []
      const code = await runImproveCli([...improveArgs, '--dry-run'], {
        cwd: workspace, write: text => { output.push(text) }, writeError: text => { output.push(text) },
      })
      const realArgs = improveArgs.filter(arg => arg !== '--dry-run')
      const transport = realArgs.includes('--transport') ? '' : ' --transport <module>'
      const command = `bun run --cwd xerxes playground:improve --${transport} ${realArgs.map(quoteArgument).join(' ')}`
      return { ok: code === 0, dry_run: true, report: `${output.join('').trim()}\n\nNothing was executed. Run from the same workspace (adjust --cwd xerxes to the package location if needed):\n${command}`, command }
    }
    if (!services) return refusal(UNAVAILABLE)
    switch (action) {
      case 'proposals': {
        const result = await listLearning(workspace, services)
        if (result.ok !== true) return result
        const proposals = result.proposals as readonly StoredReflectionProposal[]
        return { ...result, report: proposals.length ? proposals.map(record => `${record.id}  ${record.status}  ${record.proposal.kind} ${record.proposal.target}\n  ${record.proposal.rationale.split('\n')[0]}  (${record.proposal.evidence.length} evidence)`).join('\n') : 'No reflection proposals in this workspace.' }
      }
      case 'show': {
        if (!id || (rest.length !== 1 && !(rest.length === 2 && rest[1] === '--json'))) return refusal('Usage: /xerxes show <id> [--json]')
        const proposal = await services.store.get(id, workspace)
        return proposal ? { ok: true, proposal, report: rest[1] === '--json' ? JSON.stringify(proposal, null, 2) : renderReflectionProposal(proposal) } : refusal('Reflection proposal was not found in this workspace.')
      }
      case 'approve':
      case 'reject':
        return reviewLearning({ target: 'proposal', id, action, ...(reason ? { reason } : {}) }, workspace, services, session)
      case 'lessons': {
        if (LESSON_REVIEW_ACTIONS.some(value => value === rest[0])) {
          const explanation = rest.slice(2).join(' ')
          return reviewLearning({ target: 'lesson', action: rest[0], id: rest[1], ...(explanation ? { reason: explanation } : {}) }, workspace, services, session)
        }
        const lessons = rest.length ? await services.lessons.recall(rest.join(' '), workspace) : await services.lessons.list(workspace)
        return { ok: true, lessons, report: renderLessons(lessons, workspace) }
      }
      default: return refusal(XERXES_USAGE)
    }
  } catch (error) { return refusal(error) }
}

/** Parse quoted filenames without a shell, substitution, expansion or execution. */
function tokenizeArguments(text: string): string[] {
  const result: string[] = []
  let token = ''; let quote = ''; let started = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!
    if (char === '\\' && quote !== "'") {
      if (++index >= text.length) throw new Error('Incomplete escaped argument')
      token += text[index]; started = true
    } else if (quote) {
      if (char === quote) quote = ''
      else token += char
    } else if (char === '"' || char === "'") { quote = char; started = true }
    else if (/\s/u.test(char)) { if (started) { result.push(token); token = ''; started = false } }
    else { token += char; started = true }
  }
  if (quote) throw new Error('Unclosed quoted argument')
  if (started) result.push(token)
  return result
}

function quoteArgument(value: string): string {
  return /^[A-Za-z0-9_./:=+-]+$/u.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`
}
