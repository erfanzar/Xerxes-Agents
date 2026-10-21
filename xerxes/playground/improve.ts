// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute, join, relative, resolve, sep } from 'node:path'
import { runHardSuite, type NativeHardTask } from './hard.js'
import type { EvaluationAgent, EvaluationReport, EvaluationScoreRow, EvaluationTurnResult } from './types.js'

/** One full-file textual replacement, applied only to a private candidate workspace. */
export interface HarnessEdit {
  readonly id: string
  readonly scope: 'INV' | 'DOM'
  readonly target: 'agent_yaml' | 'skill' | 'prompt_addendum'
  readonly description: string
  readonly apply: { readonly path: string; readonly content: string }
}

export interface HarnessCandidate {
  readonly id: string
  readonly edits: readonly HarnessEdit[]
}

/** Explicit ceilings; the optional token ceiling uses reported contextTokens, not billing usage. */
export interface ImproveBudget {
  readonly maxCandidates: number
  readonly maxTaskRuns: number
  readonly maxTokens?: number
  readonly maxWallMs?: number
}

export interface ImprovementPlan {
  readonly baseline: HarnessCandidate
  readonly candidates: readonly HarnessCandidate[]
  readonly budget: ImproveBudget
  readonly taskNames: readonly string[]
  /** Includes the conditional composed candidate, excludes baseline. */
  readonly candidateCount: number
  readonly composeReserved: boolean
  /** Exact reservation: (candidateCount + 1 baseline) * taskNames.length. */
  readonly taskRuns: number
}

export interface CandidateWorkspacePort {
  /** Owns cleanup even if allocation fails before a handle can be returned. */
  create(candidateId: string, edits: readonly HarnessEdit[]): Promise<{
    readonly directory: string
    dispose(): Promise<void>
  }>
}

export interface ImprovementProposalContext {
  readonly baseline: HarnessCandidate
  readonly baselineReport: EvaluationReport
  readonly proposals: readonly HarnessEdit[]
  readonly taskNames: readonly string[]
  readonly budget: ImproveBudget
  readonly signal: AbortSignal
}

export type ImprovementProgress =
  | { readonly type: 'candidate_started'; readonly candidateId: string }
  | { readonly type: 'task_completed'; readonly candidateId: string; readonly row: EvaluationScoreRow }
  | { readonly type: 'candidate_completed'; readonly candidateId: string }

export interface ImprovementPorts {
  /** Explicitly opt in to replacing the declared proposal slots after baseline evaluation. */
  readonly proposer?: { propose(context: ImprovementProposalContext): Promise<HarnessEdit[]> }
  readonly workspaces: CandidateWorkspacePort
  readonly agentFor: (candidateDirectory: string) => EvaluationAgent | Promise<EvaluationAgent>
  readonly tasks: readonly NativeHardTask[]
  readonly signal?: AbortSignal
  readonly onProgress?: (progress: ImprovementProgress) => void
}

export interface ImprovementCandidateResult {
  readonly candidate: HarnessCandidate
  readonly report: EvaluationReport
  readonly complete: boolean
  readonly passed: number
  readonly regressions: readonly string[]
  readonly improves: boolean
}

export interface ImprovementResult {
  readonly status: 'completed' | 'budget_exhausted' | 'cancelled'
  readonly stopReason: string | undefined
  readonly plan: ImprovementPlan
  readonly candidates: readonly ImprovementCandidateResult[]
  readonly budgetUsed: { readonly candidates: number; readonly taskRuns: number; readonly tokens: number; readonly wallMs: number }
  readonly recommendation: { readonly candidate: HarnessCandidate; readonly reason: string }
}

/** Validate and declare all spending before constructing a workspace, transport, or agent. */
export function planImprovement(input: {
  readonly baseline: HarnessCandidate
  readonly proposals: readonly HarnessCandidate[] | readonly HarnessEdit[]
  readonly tasks: readonly NativeHardTask[]
  readonly budget: ImproveBudget
}): ImprovementPlan {
  validateBudget(input.budget)
  const baseline = copyCandidate(input.baseline)
  const candidates = singleCandidates(input.proposals)
  const ids = new Set([baseline.id, 'composed'])
  if (baseline.id === 'composed') throw new Error('baseline id cannot be the reserved id composed')
  for (const candidate of candidates) {
    if (ids.has(candidate.id)) throw new Error(`duplicate or reserved candidate id: ${candidate.id}`)
    ids.add(candidate.id)
  }
  const taskNames = input.tasks.map(task => {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(task.name) || task.steps.length === 0) {
      throw new Error(`task ${task.name} needs a safe name and at least one step`)
    }
    return task.name
  })
  if (taskNames.length === 0) throw new Error('select at least one hard task before planning improvement')
  if (new Set(taskNames).size !== taskNames.length) throw new Error('duplicate task names are not allowed')
  const composeReserved = candidates.length >= 2
  const candidateCount = candidates.length + Number(composeReserved)
  const taskRuns = (candidateCount + 1) * taskNames.length
  if (candidateCount > input.budget.maxCandidates || taskRuns > input.budget.maxTaskRuns) {
    throw new Error(`Improvement requires ${candidateCount} candidates (including ${Number(composeReserved)} conditional composition slot) and ${taskRuns} task runs = (${candidateCount} + 1 baseline) × ${taskNames.length} tasks; declared maxCandidates=${input.budget.maxCandidates}, maxTaskRuns=${input.budget.maxTaskRuns}. Reduce proposals/tasks or explicitly raise --max-candidates and --max-task-runs before running.`)
  }
  return Object.freeze({
    baseline, candidates: Object.freeze(candidates), budget: Object.freeze({ ...input.budget }),
    taskNames: Object.freeze(taskNames), candidateCount, composeReserved, taskRuns,
  })
}

/** Run only in supplied private workspaces. Returned recommendations never modify the source harness. */
export async function runImprovement(plan: ImprovementPlan, ports: ImprovementPorts): Promise<ImprovementResult> {
  // Revalidate before any side effects, even when a caller deserialized or mutated its own plan.
  const checked = planImprovement({ baseline: plan.baseline, proposals: plan.candidates, tasks: ports.tasks, budget: plan.budget })
  if (checked.taskNames.join('\0') !== plan.taskNames.join('\0') || checked.taskRuns !== plan.taskRuns || checked.candidateCount !== plan.candidateCount) {
    throw new Error('runtime tasks or candidate counts differ from the declared tasks/plan; create and approve a new plan')
  }
  const startedAt = Date.now()
  const controller = new AbortController()
  const cancel = () => controller.abort(new StopRun('cancelled', 'caller aborted the improvement run'))
  ports.signal?.addEventListener('abort', cancel, { once: true })
  if (ports.signal?.aborted) cancel()
  const deadline = plan.budget.maxWallMs === undefined ? undefined : startedAt + plan.budget.maxWallMs
  const timer = plan.budget.maxWallMs === undefined ? undefined : setTimeout(() => {
    controller.abort(new StopRun('budget_exhausted', `maxWallMs=${plan.budget.maxWallMs} deadline reached`))
  }, plan.budget.maxWallMs)
  const used = { candidates: 0, taskRuns: 0, tokens: 0, wallMs: 0 }
  const results: ImprovementCandidateResult[] = []
  const directories = new Set<string>()
  let stop: StopRun | undefined

  function guard(spending = false): void {
    if (controller.signal.aborted) throw controller.signal.reason
    if (deadline !== undefined && Date.now() >= deadline) throw new StopRun('budget_exhausted', `maxWallMs=${plan.budget.maxWallMs} deadline reached`)
    if (spending && plan.budget.maxTokens !== undefined && used.tokens >= plan.budget.maxTokens) {
      throw new StopRun('budget_exhausted', `reported tokens ${used.tokens}/${plan.budget.maxTokens}; no further turns started`)
    }
  }

  async function bounded<T>(operation: () => Promise<T>): Promise<T> {
    guard()
    let abort: (() => void) | undefined
    try {
      return await Promise.race([
        operation(),
        new Promise<never>((_resolve, reject) => {
          abort = () => reject(controller.signal.reason)
          controller.signal.addEventListener('abort', abort, { once: true })
          if (controller.signal.aborted) abort()
        }),
      ])
    } finally {
      if (abort !== undefined) controller.signal.removeEventListener('abort', abort)
    }
  }

  async function evaluate(candidate: HarnessCandidate, isBaseline: boolean): Promise<void> {
    guard(true)
    if (!isBaseline && used.candidates >= plan.budget.maxCandidates) throw new StopRun('budget_exhausted', 'candidate budget exhausted')
    // Allocation is awaited (rather than raced) so a late workspace handle cannot leak on abort.
    const workspace = await ports.workspaces.create(candidate.id, isBaseline ? candidate.edits : [...plan.baseline.edits, ...candidate.edits])
    let agent: EvaluationAgent | undefined
    const rows: EvaluationScoreRow[] = []
    const sandboxDirectory = join(resolve(workspace.directory), '.improve-tasks')
    let enteredSuite = false
    try {
      const directory = resolve(workspace.directory)
      if (directories.has(directory)) throw new Error('CandidateWorkspacePort must return a separate directory for every candidate')
      directories.add(directory)
      guard(true)
      if (!isBaseline) used.candidates += 1
      ports.onProgress?.({ type: 'candidate_started', candidateId: candidate.id })
      guard(true)
      agent = await ports.agentFor(directory)
      guard(true)
      const underlying = agent
      let stepIndex = 0
      const wrapped: EvaluationAgent = {
        get model() { return underlying.model },
        start: () => bounded(() => underlying.start()),
        freshSession: () => { guard(true); return bounded(() => underlying.freshSession()) },
        close: () => underlying.close(),
        turn: async (prompt, options) => {
          guard(true)
          if (stepIndex === 0) {
            if (used.taskRuns >= plan.budget.maxTaskRuns || used.taskRuns >= plan.taskRuns) throw new StopRun('budget_exhausted', 'task-run budget exhausted')
            used.taskRuns += 1
          }
          const remaining = deadline === undefined ? undefined : Math.max(1, deadline - Date.now())
          const result = await bounded(() => underlying.turn(prompt, { ...options, retries: 0, ...(remaining === undefined ? {} : { timeoutMs: remaining }) }))
          used.tokens += observedTokens(result)
          stepIndex += 1
          return result
        },
      }
      const tasks = ports.tasks.map(task => ({
        ...task,
        steps: task.steps.map(step => ({ ...step, prompt: step.prompt.replaceAll('{dir}', join(sandboxDirectory, task.name)) })),
        check: (context: Parameters<NativeHardTask['check']>[0]) => bounded(() => Promise.resolve(task.check({ ...context, signal: AbortSignal.any([context.signal, controller.signal]) }))),
      }))
      enteredSuite = true
      await runHardSuite({
        agent: wrapped, sandboxDirectory, tasks,
        onRow: row => {
          guard()
          rows.push(row)
          stepIndex = 0
          ports.onProgress?.({ type: 'task_completed', candidateId: candidate.id, row })
        },
      })
      ports.onProgress?.({ type: 'candidate_completed', candidateId: candidate.id })
    } finally {
      try {
        const baselineRows = results[0]?.report.rows ?? []
        const complete = rows.length === ports.tasks.length
        const regressions = baselineRows.filter(row => row.ok && rows.some(other => other.name === row.name && !other.ok)).map(row => row.name)
        const passed = rows.filter(row => row.ok).length
        results.push({
          candidate, complete, passed, regressions,
          improves: !isBaseline && complete && results[0]?.complete === true && regressions.length === 0 && passed > (results[0]?.passed ?? 0),
          report: { kind: 'hard', model: agent?.model ?? '?', sandboxDirectory, rows, totalLatencyMs: rows.reduce((total, row) => total + row.latencyMs, 0) },
        })
        if (agent !== undefined && !enteredSuite) await agent.close()
      } finally { await workspace.dispose() }
    }
  }

  try {
    await evaluate(plan.baseline, true)
    let candidates = plan.candidates
    if (ports.proposer !== undefined) {
      guard(true)
      const edits = await bounded(() => ports.proposer!.propose({
        baseline: plan.baseline, baselineReport: results[0]!.report,
        proposals: candidates.flatMap(candidate => candidate.edits), taskNames: plan.taskNames,
        budget: plan.budget, signal: controller.signal,
      }))
      if (edits.length > candidates.length) throw new StopRun('budget_exhausted', `proposer returned ${edits.length} edits for ${candidates.length} declared single-edit slots; re-plan before running`)
      const proposed = planImprovement({ baseline: plan.baseline, proposals: edits, tasks: ports.tasks, budget: plan.budget })
      candidates = proposed.candidates
    }
    for (const candidate of candidates) await evaluate(candidate, false)
    const winners = results.filter(result => result.improves).flatMap(result => result.candidate.edits)
    if (winners.length >= 2) {
      const edits = [...winners.filter(edit => edit.scope === 'INV'), ...winners.filter(edit => edit.scope === 'DOM')]
      await evaluate({ id: 'composed', edits }, false)
    }
    guard()
  } catch (error) {
    if (!(error instanceof StopRun)) throw error
    stop = error
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    ports.signal?.removeEventListener('abort', cancel)
  }
  used.wallMs = Math.max(0, Date.now() - startedAt)
  const best = results.filter(result => result.improves).sort((left, right) => right.passed - left.passed)[0]
  return {
    status: stop?.status ?? 'completed', stopReason: stop?.message, plan, candidates: results, budgetUsed: used,
    recommendation: best === undefined
      ? { candidate: plan.baseline, reason: 'no candidate beat baseline without regressions on the full task set; keep baseline' }
      : { candidate: best.candidate, reason: `${best.passed}/${ports.tasks.length} tasks passed, strictly above baseline ${results[0]?.passed ?? 0}/${ports.tasks.length}, with no baseline-pass regressions; ties keep the earlier candidate` },
  }
}

/** Human-readable declaration suitable for explicit approval or a CLI dry run. */
export function formatImprovementPlan(plan: ImprovementPlan): string {
  return [
    '# Harness improvement plan (opt-in)',
    `(${plan.candidateCount} candidates + 1 baseline) × ${plan.taskNames.length} tasks = ${plan.taskRuns} task runs`,
    `Tasks: ${plan.taskNames.join(', ')}`,
    `Candidates: ${plan.candidates.map(candidate => `${candidate.id} [${candidate.edits[0]?.scope}]`).join(', ') || 'none'}`,
    `Conditional composition slot: ${plan.composeReserved ? '1 reserved; skipped if fewer than two individual winners' : 'none'}`,
    `Declared budget: maxCandidates=${plan.budget.maxCandidates}, maxTaskRuns=${plan.budget.maxTaskRuns}, maxTokens=${plan.budget.maxTokens ?? 'not set'}, maxWallMs=${plan.budget.maxWallMs ?? 'not set'}`,
    'Tokens use reported contextTokens (not billing usage); an in-flight turn may exceed the ceiling.',
    'Only private candidate workspaces are edited. Recommendations require human application.',
  ].join('\n')
}

/** Markdown scorecard includes partial rows, regressions, declared ceilings, and unused reservations. */
export function formatImprovementReport(result: ImprovementResult): string {
  const cell = (value: string) => value.replaceAll('|', '\\|').replaceAll(/[\r\n]/g, ' ')
  const used = result.budgetUsed
  return [
    '# Harness improvement report', `Status: ${result.status}${result.stopReason === undefined ? '' : ` — ${result.stopReason}`}`, '',
    `| Candidate | ${result.plan.taskNames.map(cell).join(' | ')} | Passes | Regressions |`,
    `| --- | ${result.plan.taskNames.map(() => '---').join(' | ')} | --- | --- |`,
    ...result.candidates.map(candidate => `| ${cell(candidate.candidate.id)}${candidate.complete ? '' : ' (partial)'} | ${result.plan.taskNames.map(name => {
      const row = candidate.report.rows.find(row => row.name === name)
      return row === undefined ? '—' : row.ok ? 'PASS' : 'FAIL'
    }).join(' | ')} | ${candidate.passed}/${result.plan.taskNames.length} | ${candidate.regressions.map(cell).join(', ') || 'none'} |`),
    '', `Candidates: ${used.candidates}/${result.plan.budget.maxCandidates} (reserved ${result.plan.candidateCount}, baseline excluded)`,
    `Task runs: ${used.taskRuns}/${result.plan.budget.maxTaskRuns} (reserved ${result.plan.taskRuns})`,
    `Reported tokens: ${used.tokens}/${result.plan.budget.maxTokens ?? 'not set'} (sum of contextTokens, not billing usage)`,
    `Wall time: ${used.wallMs}/${result.plan.budget.maxWallMs ?? 'not set'} ms`, '',
    `Recommendation: ${cell(result.recommendation.candidate.id)} — ${result.recommendation.reason}`,
    ...result.recommendation.candidate.edits.map(edit => `- ${edit.scope} ${edit.target}: ${cell(edit.apply.path)} — ${cell(edit.description)}`),
    'No edits were applied to the source repository; apply recommended replacements only after human review.',
  ].join('\n')
}

class StopRun extends Error {
  constructor(readonly status: 'budget_exhausted' | 'cancelled', message: string) { super(message) }
}

function singleCandidates(proposals: readonly HarnessCandidate[] | readonly HarnessEdit[]): HarnessCandidate[] {
  const seen = new Set<string>()
  return proposals.flatMap(proposal => {
    const edits = 'edits' in proposal ? copyCandidate(proposal).edits : [copyEdit(proposal)]
    return edits.map(edit => {
      if (seen.has(edit.id)) throw new Error(`duplicate edit id: ${edit.id}`)
      seen.add(edit.id)
      const id = 'edits' in proposal ? edits.length === 1 ? proposal.id : `${proposal.id}-${edit.id}` : edit.id
      return Object.freeze({ id, edits: Object.freeze([edit]) })
    })
  })
}

function copyCandidate(candidate: HarnessCandidate): HarnessCandidate {
  requiredText(candidate.id, 'candidate id')
  if (!Array.isArray(candidate.edits)) throw new Error('candidate edits must be an array')
  return Object.freeze({ id: candidate.id, edits: Object.freeze(candidate.edits.map(copyEdit)) })
}

function copyEdit(edit: HarnessEdit): HarnessEdit {
  requiredText(edit.id, 'edit id')
  requiredText(edit.description, 'edit description')
  if (edit.scope !== 'INV' && edit.scope !== 'DOM') throw new Error('edit scope must be INV or DOM')
  if (!['agent_yaml', 'skill', 'prompt_addendum'].includes(edit.target)) throw new Error('edit target must be agent_yaml, skill or prompt_addendum')
  if (!edit.apply || typeof edit.apply.path !== 'string' || typeof edit.apply.content !== 'string') throw new Error('edit apply requires { path, content } full replacement strings')
  const path = edit.apply.path
  const child = relative(resolve('/harness'), resolve('/harness', path)).split(sep).join('/')
  if (!path.trim() || isAbsolute(path) || path.includes('\\') || /^[A-Za-z]:/.test(path) || path.includes('\0') || !child || child === '..' || child.startsWith('../') || isAbsolute(child) || path.split('/').some(part => part === '..' || part.startsWith('.improve-'))) {
    throw new Error(`edit path must remain inside the candidate harness and outside reserved .improve-* paths: ${path}`)
  }
  return Object.freeze({ id: edit.id, scope: edit.scope, target: edit.target, description: edit.description, apply: Object.freeze({ path: child, content: edit.apply.content }) })
}

function validateBudget(budget: ImproveBudget): void {
  if (typeof budget !== 'object' || budget === null) throw new Error('an explicit positive improvement budget is required')
  positiveInteger(budget.maxCandidates, 'maxCandidates')
  positiveInteger(budget.maxTaskRuns, 'maxTaskRuns')
  if (budget.maxTokens !== undefined) positiveInteger(budget.maxTokens, 'maxTokens')
  if (budget.maxWallMs !== undefined) positiveInteger(budget.maxWallMs, 'maxWallMs')
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`)
}

function requiredText(value: string, name: string): void {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${name} must be non-empty`)
}

function observedTokens(result: EvaluationTurnResult): number {
  if (!Number.isSafeInteger(result.contextTokens) || result.contextTokens < 0) throw new Error('agent contextTokens must be a non-negative integer')
  return result.contextTokens
}
