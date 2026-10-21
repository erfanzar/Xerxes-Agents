// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import type { EvaluationSessionPort, EvaluationStartRequest, EvaluationEvent } from './types.js'

export const DEFAULT_RECALL_FACTS = 10
export const DEFAULT_RECALL_SEED = 42
export const DEFAULT_RECALL_FILLER_TOKENS = 384_000
const FILLER_TOKENS_PER_TURN = 8_192
const DEFAULT_TURN_TIMEOUT_MS = 450_000

/** Sizes are bounded to keep accidental CLI invocations finite; tokens estimate characters / 4. */
export interface RecallScenarioOptions {
  readonly seed?: number
  readonly facts?: number
  readonly fillerTokens?: number
}

/** A planted answer and a question that contains no copy of that answer. */
export interface RecallFact {
  readonly id: string
  readonly kind: 'deploy_key' | 'file_path' | 'threshold' | 'decision' | 'reason'
  readonly statement: string
  readonly answer: string
  readonly question: string
}

export interface RecallFillerTurn {
  readonly file: { readonly path: string; readonly content: string }
  /** Contains the fixture too, so transcript pressure does not depend on a tool being available. */
  readonly prompt: string
}

/** One planting turn, filler in the same session, then one exact-answer question per fact. */
export interface RecallScenario {
  readonly seed: number
  readonly requestedFillerTokens: number
  readonly estimatedFillerTokens: number
  readonly facts: readonly RecallFact[]
  readonly plantPrompt: string
  readonly filler: readonly RecallFillerTurn[]
}

export interface RecallFactScore {
  readonly id: string
  readonly kind: RecallFact['kind']
  readonly expected: string
  readonly actual: string
  readonly ok: boolean
}

export interface RecallReport {
  readonly kind: 'recall'
  readonly seed: number
  readonly model: string
  readonly rows: readonly RecallFactScore[]
  readonly recallRate: number
  /** Delta from after session start to immediately before the first recall question. */
  readonly compactionsObserved: number | 'unknown'
  readonly fillerTurns: number
  readonly estimatedFillerTokens: number
  readonly wallTimeMs: number
}

export interface RecallEvaluationOptions {
  readonly scenario: RecallScenario
  readonly transport: EvaluationSessionPort
  readonly start: EvaluationStartRequest
  readonly signal?: AbortSignal
  readonly turnTimeoutMs?: number
  readonly clock?: () => number
}

/** Build reproducible synthetic facts and realistic, tool-oriented log-review work, without I/O. */
export function createRecallScenario(options: RecallScenarioOptions = {}): RecallScenario {
  const seed = integerInRange(options.seed ?? DEFAULT_RECALL_SEED, 'seed', 0, 0xffff_ffff)
  const count = integerInRange(options.facts ?? DEFAULT_RECALL_FACTS, 'facts', 1, 1_000)
  const requestedFillerTokens = integerInRange(options.fillerTokens ?? DEFAULT_RECALL_FILLER_TOKENS, 'fillerTokens', 0, 2_000_000)
  const random = seededRandom(seed)
  const facts = Array.from({ length: count }, (_, index) => createFact(index, random))
  const plantPrompt = [
    'RECALL planting — these are synthetic evaluation facts, not real credentials.',
    'Retain these facts for questions much later in this same session. Acknowledge briefly.',
    'Use your normal session memory behavior; the evaluator will not write these facts to fixture files.',
    ...facts.map(fact => `[${fact.id}] ${fact.statement}`),
  ].join('\n')
  const filler: RecallFillerTurn[] = []
  let estimatedFillerTokens = 0
  while (estimatedFillerTokens < requestedFillerTokens) {
    const turn = createFiller(filler.length, Math.min(FILLER_TOKENS_PER_TURN, requestedFillerTokens - estimatedFillerTokens), random)
    filler.push(turn)
    estimatedFillerTokens += Math.ceil(turn.prompt.length / 4)
  }
  return { seed, requestedFillerTokens, estimatedFillerTokens, facts, plantPrompt, filler }
}

/** No substring matching, punctuation stripping or LLM judge: only case and whitespace normalize. */
export function gradeRecallAnswer(fact: RecallFact, answer: string): RecallFactScore {
  return { id: fact.id, kind: fact.kind, expected: fact.answer, actual: answer, ok: normalize(answer) === normalize(fact.answer) }
}

/**
 * Exercise one caller-owned session without resets, credential discovery, or forced compaction.
 * Only the generated filler fixtures are written. Planted answers stay out of evaluator files.
 * Errors/cancellation reject instead of being misreported as model recall failures; close always runs.
 */
export async function runRecallEvaluation(options: RecallEvaluationOptions): Promise<RecallReport> {
  const clock = options.clock ?? Date.now
  const startedAt = clock()
  const timeoutMs = integerInRange(options.turnTimeoutMs ?? DEFAULT_TURN_TIMEOUT_MS, 'turnTimeoutMs', 1, 2_147_483_647)
  const signal = options.signal ?? new AbortController().signal
  const workspace = resolve(options.start.workspaceDirectory)
  const rows: RecallFactScore[] = []
  let model = '?'
  let compactionsObserved: number | 'unknown' = 'unknown'
  try {
    signal.throwIfAborted()
    // No reset: start() allocates exactly the one session being measured.
    const started = await abortable(() => options.transport.start(options.start), signal)
    model = started.model?.trim() || '?'
    const before = await compactionCount(options.transport, signal)
    // Regenerate the deterministic fixtures instead of accepting arbitrary caller-owned file paths.
    const scenario = createRecallScenario({
      seed: options.scenario.seed, facts: options.scenario.facts.length, fillerTokens: options.scenario.requestedFillerTokens,
    })
    if (JSON.stringify(scenario) !== JSON.stringify(options.scenario)) throw new Error('recall scenario differs from its seed/sizes; regenerate the scenario')
    for (const filler of scenario.filler) {
      signal.throwIfAborted()
      const path = resolve(workspace, filler.file.path)
      await mkdir(dirname(path), { recursive: true })
      await Bun.write(path, filler.file.content)
    }
    await recallTurn(options.transport, scenario.plantPrompt, timeoutMs, signal)
    for (const filler of scenario.filler) await recallTurn(options.transport, filler.prompt, timeoutMs, signal)
    const after = await compactionCount(options.transport, signal)
    if (before !== undefined && after !== undefined) {
      if (after < before) throw new Error('transport compaction count decreased during the recall session')
      compactionsObserved = after - before
    }
    for (const fact of scenario.facts) {
      const answer = await recallTurn(options.transport, fact.question, timeoutMs, signal)
      rows.push(gradeRecallAnswer(fact, answer))
    }
    signal.throwIfAborted()
    return {
      kind: 'recall', seed: scenario.seed, model, rows,
      recallRate: rows.filter(row => row.ok).length / scenario.facts.length,
      compactionsObserved, fillerTurns: scenario.filler.length, estimatedFillerTokens: scenario.estimatedFillerTokens,
      wallTimeMs: Math.max(0, clock() - startedAt),
    }
  } finally { await options.transport.close() }
}

/** Dry-run declaration, including cost-driving sizes but no transport or filesystem work. */
export function formatRecallPlan(scenario: RecallScenario): string {
  return [
    `# Xerxes RECALL plan — seed=${scenario.seed}`,
    `Facts: ${scenario.facts.length} (seeded synthetic keys, paths, thresholds, decisions and reasons)`,
    `Session: 1 planting turn → ${scenario.filler.length} filler turns → ${scenario.facts.length} recall questions; no resets`,
    `Filler: target ${scenario.requestedFillerTokens}, estimated ${scenario.estimatedFillerTokens} tokens (ceil(prompt characters / 4), not provider token usage)`,
    'Filler volume is not a guaranteed compaction count; choose enough volume for the hosted context window.',
    'Compactions are measured only through optional transport.getCompactionCount(); otherwise unknown.',
    'Grader: entire answer equality after lowercasing and whitespace folding; no LLM judge.',
    'Execution requires an explicit caller-owned transport. Dry run imports none and creates no runtime.',
  ].join('\n')
}

export function formatRecallReport(report: RecallReport): string {
  const passed = report.rows.filter(row => row.ok).length
  return [
    `# Xerxes RECALL report — model=${report.model}, seed=${report.seed}`,
    ...report.rows.map(row => `[${row.ok ? 'PASS' : 'FAIL'}] ${row.id} (${row.kind}) expected=${JSON.stringify(row.expected)} actual=${JSON.stringify(row.actual)}`),
    `RECALL: ${passed}/${report.rows.length} (${(100 * report.recallRate).toFixed(1)}%)`,
    `Compactions observed before recall: ${report.compactionsObserved}`,
    `Filler: ${report.fillerTurns} turns, estimated ${report.estimatedFillerTokens} tokens; wall time: ${report.wallTimeMs} ms`,
    'Recall scores alone do not prove compaction occurred; unknown means the transport supplied no counter.',
  ].join('\n')
}

function createFact(index: number, random: () => number): RecallFact {
  const id = `fact-${String(index + 1).padStart(3, '0')}`
  const service = `orion-${index + 1}`
  const token = randomToken(random)
  const prefix = `RECALL question [${id}]: `
  const suffix = ' Reply with only the exact value, without quotes, explanation or punctuation. If unknown, reply UNKNOWN.'
  switch (index % 5) {
    case 0: {
      const answer = `K${index + 1}-${token}`
      return { id, kind: 'deploy_key', answer, statement: `The deploy key for service ${service} is ${answer}.`, question: `${prefix}What deploy key did I give for service ${service}?${suffix}` }
    }
    case 1: {
      const answer = `/srv/${service}/${token.toLowerCase()}/release.yaml`
      return { id, kind: 'file_path', answer, statement: `The release config path for service ${service} is ${answer}.`, question: `${prefix}What release config path did I give for service ${service}?${suffix}` }
    }
    case 2: {
      const answer = `${10_000_000 + index * 10_000 + Math.floor(random() * 10_000)} requests/minute`
      return { id, kind: 'threshold', answer, statement: `The traffic threshold for service ${service} is ${answer}.`, question: `${prefix}What traffic threshold (include units) did I give for service ${service}?${suffix}` }
    }
    case 3: {
      const answer = `use canary lane ${index + 1}-${token.toLowerCase()}`
      return { id, kind: 'decision', answer, statement: `The rollout decision for service ${service} is ${answer}.`, question: `${prefix}What rollout decision did I give for service ${service}?${suffix}` }
    }
    default: {
      const answer = `preserve rollback coverage for cohort ${index + 1}-${token.toLowerCase()}`
      return { id, kind: 'reason', answer, statement: `The reason for delaying service ${service} is ${answer}.`, question: `${prefix}What reason did I give for delaying service ${service}?${suffix}` }
    }
  }
}

function createFiller(index: number, tokens: number, random: () => number): RecallFillerTurn {
  const path = `recall-fixtures/batch-${String(index + 1).padStart(3, '0')}.log`
  const lines = [`# Synthetic CI incident review batch ${index + 1}; all records are evaluation fixtures.`]
  let length = lines[0]!.length + 1
  while (length < tokens * 4) {
    const line = `run=${index + 1}.${lines.length} worker=build-${randomToken(random)} stage=integration duration_ms=${Math.floor(random() * 30_000)} retries=${Math.floor(random() * 4)} outcome=${random() > 0.2 ? 'passed' : 'flaky'} artifact=ci-${randomToken(random)} note=compare cache hits and queue saturation before suggesting changes`
    lines.push(line)
    length += line.length + 1
  }
  const content = lines.join('\n') + '\n'
  return {
    file: { path, content },
    prompt: [
      `RECALL filler ${index + 1}: inspect ${path}. Use read/search tools when available to review the fixture; its full contents are also below.`,
      'Summarize flaky stages, retry patterns and one actionable CI improvement in at most three bullets. Do not repeat earlier session facts. Do not modify files.',
      'BEGIN GENERATED FIXTURE', content, 'END GENERATED FIXTURE',
    ].join('\n'),
  }
}

/** A tiny deterministic PRNG; never a credential generator. */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = Math.imul(state ^ (state >>> 15), state | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296
  }
}

function randomToken(random: () => number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  return Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => alphabet[Math.floor(random() * alphabet.length)]!).join('')).join('-')
}

async function recallTurn(transport: EvaluationSessionPort, prompt: string, timeoutMs: number, parent: AbortSignal): Promise<string> {
  parent.throwIfAborted()
  const controller = new AbortController()
  const signal = AbortSignal.any([parent, controller.signal])
  const timer = setTimeout(() => controller.abort(new Error(`recall turn timed out after ${timeoutMs} ms`)), timeoutMs)
  let iterator: AsyncIterator<EvaluationEvent> | undefined
  try {
    const events = await abortable(() => Promise.resolve(transport.submit({ prompt, signal, timeoutMs })), signal)
    iterator = events[Symbol.asyncIterator]()
    const text: string[] = []
    while (true) {
      const next = await abortable(() => iterator!.next(), signal)
      if (next.done) throw new Error('evaluation stream ended without a turn_end event')
      const event = next.value
      if (event.type === 'text') text.push(event.text)
      if (event.type === 'approval_request') await abortable(() => transport.approve(event.id), signal)
      if (event.type === 'notification' && event.severity === 'error') throw new Error(event.body ?? event.title ?? 'evaluation transport error')
      if (event.type === 'turn_end') return text.join('')
    }
  } finally {
    clearTimeout(timer)
    controller.abort()
    // Do not wait for iterator.return(): an uncooperative pending next() can otherwise hide cancellation.
    if (iterator?.return !== undefined) void iterator.return().catch(() => {})
  }
}

async function compactionCount(transport: EvaluationSessionPort, signal: AbortSignal): Promise<number | undefined> {
  if (transport.getCompactionCount === undefined) return undefined
  const count = await abortable(() => Promise.resolve(transport.getCompactionCount!()), signal)
  return integerInRange(count, 'transport compaction count', 0, Number.MAX_SAFE_INTEGER)
}

async function abortable<T>(operation: () => Promise<T>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted()
  let abort: (() => void) | undefined
  try {
    return await Promise.race([
      new Promise<never>((_resolve, reject) => {
        abort = () => reject(signal.reason)
        signal.addEventListener('abort', abort, { once: true })
        if (signal.aborted) abort()
      }),
      operation(),
    ])
  } finally { if (abort !== undefined) signal.removeEventListener('abort', abort) }
}

function normalize(value: string): string { return value.trim().toLowerCase().replace(/\s+/g, ' ') }

function integerInRange(value: number, name: string, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`)
  return value
}
