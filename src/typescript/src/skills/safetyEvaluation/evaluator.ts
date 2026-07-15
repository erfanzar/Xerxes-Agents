// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createSafetyModelRequest, DEFAULT_SAFETY_PROBES } from './probes.js'
import { scoreSafetyResponse } from './scoring.js'
import type {
  SafetyEvaluationModelPort,
  SafetyEvaluationOptions,
  SafetyEvaluationReport,
  SafetyEvaluationResult,
  SafetyEvaluationSummary,
  SafetyModelSummary,
  SafetyProbe,
} from './types.js'

/** Default concurrency is intentionally modest because safety evaluation should not amplify load. */
export const DEFAULT_SAFETY_EVALUATION_CONCURRENCY = 4

/** Raised when a caller cancels the entire defensive evaluation run. */
export class SafetyEvaluationAbortedError extends Error {
  constructor() {
    super('Safety evaluation was aborted')
    this.name = 'SafetyEvaluationAbortedError'
  }
}

/**
 * Evaluate explicitly injected models against benign, deterministic safety probes.
 * Per-model failures become report rows so one unavailable model does not hide
 * evidence from the remaining models. Passing `reportStore` is the sole opt-in
 * path to durable storage.
 */
export async function evaluateSafetyModels(
  models: readonly SafetyEvaluationModelPort[],
  options: SafetyEvaluationOptions = {},
): Promise<SafetyEvaluationReport> {
  const probes = options.probes ?? DEFAULT_SAFETY_PROBES
  validateInputs(models, probes)
  throwIfAborted(options.signal)

  const concurrency = options.maxConcurrent ?? DEFAULT_SAFETY_EVALUATION_CONCURRENCY
  assertPositiveSafeInteger(concurrency, 'maxConcurrent')
  const jobs = createJobs(models, probes)
  const results = await runWithConcurrency(jobs, Math.min(concurrency, jobs.length), options.signal)
  const generatedAt = (options.now ?? (() => new Date()))()
  if (Number.isNaN(generatedAt.valueOf())) throw new RangeError('now must return a valid Date')

  const report: SafetyEvaluationReport = {
    generatedAt: generatedAt.toISOString(),
    id: createReportId(options.idFactory),
    modelIds: models.map(model => model.id),
    probeIds: probes.map(probe => probe.id),
    results,
    summary: summarize(models, results),
  }
  if (options.reportStore !== undefined) await options.reportStore.save(report)
  return report
}

interface SafetyEvaluationJob {
  readonly model: SafetyEvaluationModelPort
  readonly probe: SafetyProbe
}

function createJobs(
  models: readonly SafetyEvaluationModelPort[],
  probes: readonly SafetyProbe[],
): readonly SafetyEvaluationJob[] {
  const jobs: SafetyEvaluationJob[] = []
  for (const model of models) {
    for (const probe of probes) jobs.push({ model, probe })
  }
  return jobs
}

async function runWithConcurrency(
  jobs: readonly SafetyEvaluationJob[],
  concurrency: number,
  signal: AbortSignal | undefined,
): Promise<readonly SafetyEvaluationResult[]> {
  const results: Array<SafetyEvaluationResult | undefined> = new Array(jobs.length)
  let nextIndex = 0

  const worker = async (): Promise<void> => {
    for (;;) {
      throwIfAborted(signal)
      const index = nextIndex
      nextIndex += 1
      const job = jobs[index]
      if (job === undefined) return
      results[index] = await evaluateJob(job, signal)
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()))
  return results.map((result, index) => {
    if (result === undefined) throw new Error(`safety evaluation job ${index} did not produce a result`)
    return result
  })
}

async function evaluateJob(job: SafetyEvaluationJob, signal: AbortSignal | undefined): Promise<SafetyEvaluationResult> {
  const startedAt = performance.now()
  try {
    throwIfAborted(signal)
    const response = await job.model.complete(createSafetyModelRequest(job.probe, signal))
    throwIfAborted(signal)
    if (response === null || typeof response !== 'object' || typeof response.text !== 'string') {
      throw new TypeError(`model ${job.model.id} returned an invalid safety evaluation response`)
    }
    const responseText = response.text.trim()
    if (responseText === '') throw new Error(`model ${job.model.id} returned an empty safety evaluation response`)

    return {
      durationMilliseconds: elapsedMilliseconds(startedAt),
      error: null,
      modelId: job.model.id,
      probeId: job.probe.id,
      responseText,
      score: scoreSafetyResponse(job.probe, responseText),
    }
  } catch (error) {
    if (signal?.aborted) throw new SafetyEvaluationAbortedError()
    return {
      durationMilliseconds: elapsedMilliseconds(startedAt),
      error: errorMessage(error),
      modelId: job.model.id,
      probeId: job.probe.id,
      responseText: null,
      score: null,
    }
  }
}

function summarize(
  models: readonly SafetyEvaluationModelPort[],
  results: readonly SafetyEvaluationResult[],
): SafetyEvaluationSummary {
  const successfulScores = results.flatMap(result => (result.score === null ? [] : [result.score.score]))
  const modelSummaries = models.map(model => summarizeModel(model.id, results))
  const completed = results.filter(result => result.error === null).length
  return {
    averageScore: average(successfulScores),
    completed,
    failed: results.length - completed,
    models: modelSummaries,
  }
}

function summarizeModel(modelId: string, results: readonly SafetyEvaluationResult[]): SafetyModelSummary {
  const modelResults = results.filter(result => result.modelId === modelId)
  const scores = modelResults.flatMap(result => (result.score === null ? [] : [result.score.score]))
  const completed = modelResults.filter(result => result.error === null).length
  return {
    averageScore: average(scores),
    completed,
    failed: modelResults.length - completed,
    modelId,
  }
}

function validateInputs(models: readonly SafetyEvaluationModelPort[], probes: readonly SafetyProbe[]): void {
  if (models.length === 0) throw new RangeError('at least one safety evaluation model is required')
  if (probes.length === 0) throw new RangeError('at least one safety probe is required')
  assertUniqueNonEmpty(models.map(model => model.id), 'model id')
  assertUniqueNonEmpty(probes.map(probe => probe.id), 'probe id')
  for (const probe of probes) {
    if (probe.prompt.trim() === '') throw new RangeError(`probe ${probe.id} must have a prompt`)
    if (probe.expectedCriteria.length === 0) throw new RangeError(`probe ${probe.id} must declare expected criteria`)
  }
}

function assertUniqueNonEmpty(values: readonly string[], name: string): void {
  const seen = new Set<string>()
  for (const value of values) {
    if (value.trim() === '') throw new RangeError(`${name} must not be empty`)
    if (seen.has(value)) throw new RangeError(`${name} must be unique: ${value}`)
    seen.add(value)
  }
}

function assertPositiveSafeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive safe integer`)
}

function createReportId(factory: (() => string) | undefined): string {
  const id = factory === undefined ? crypto.randomUUID() : factory()
  if (id.trim() === '') throw new RangeError('report id must not be empty')
  return id
}

function elapsedMilliseconds(startedAt: number): number {
  return Math.max(0, Math.round(performance.now() - startedAt))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function average(values: readonly number[]): number | null {
  if (values.length === 0) return null
  return Math.round((values.reduce((total, value) => total + value, 0) / values.length) * 100) / 100
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new SafetyEvaluationAbortedError()
}
