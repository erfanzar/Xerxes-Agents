// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mapGsm8kExamples } from './dataset.js'
import { parseGrpoMetric, validateGrpoCheckpoint, validateGrpoModelArtifact } from './metrics.js'
import { BASIC_GRPO_REWARD_PROGRAMS } from './rewards.js'
import type {
  GrpoDatasetExample,
  GrpoHostRunHandle,
  GrpoLoraConfig,
  GrpoTrainingConfig,
  GrpoTrainingDependencies,
  GrpoTrainingExecution,
  GrpoTrainingMetric,
  GrpoTrainingReportEvent,
  GrpoTrainingRequest,
  Gsm8kSourceExample,
} from './types.js'

/** Default model used by the original reference template. */
export const BASIC_GRPO_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

/** Default output reference owned by the configured training host. */
export const BASIC_GRPO_OUTPUT_DIRECTORY = 'outputs/grpo-model'

/** Build a fresh copy of the reference GRPO configuration. */
export function defaultBasicGrpoTrainingConfig(): GrpoTrainingConfig {
  return {
    modelName: BASIC_GRPO_MODEL,
    outputDirectory: BASIC_GRPO_OUTPUT_DIRECTORY,
    runName: 'grpo-training',
    learningRate: 5e-6,
    adamBeta1: 0.9,
    adamBeta2: 0.99,
    weightDecay: 0.1,
    warmupRatio: 0.1,
    scheduler: 'cosine',
    perDeviceTrainBatchSize: 1,
    gradientAccumulationSteps: 4,
    numGenerations: 8,
    maxPromptLength: 256,
    maxCompletionLength: 512,
    numTrainEpochs: 1,
    maxGradNorm: 0.1,
    loggingSteps: 1,
    saveSteps: 100,
    reportTo: 'wandb',
    accelerator: {
      precision: 'bf16',
      attentionImplementation: 'flash_attention_2',
      devicePlacement: 'host-owned',
      modelLoading: 'host-owned',
      optimizerExecution: 'host-owned',
    },
  }
}

/** Build a fresh copy of the LoRA adapter configuration from the reference template. */
export function defaultBasicGrpoLoraConfig(): GrpoLoraConfig {
  return {
    rank: 16,
    alpha: 32,
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    taskType: 'CAUSAL_LM',
    dropout: 0.05,
  }
}

/** Override surface for a native GRPO request while preserving explicit accelerator ownership. */
export interface BasicGrpoTrainingOptions {
  readonly config?: Partial<Omit<GrpoTrainingConfig, 'accelerator'>> & {
    readonly accelerator?: Partial<GrpoTrainingConfig['accelerator']>
  }
  readonly lora?: Partial<GrpoLoraConfig>
}

/** Create a validated request that a host accelerator/provider can execute. */
export function createBasicGrpoTrainingRequest(
  dataset: readonly GrpoDatasetExample[],
  options: BasicGrpoTrainingOptions = {},
): GrpoTrainingRequest {
  if (!Array.isArray(dataset)) throw new TypeError('dataset must be an array')
  const baseConfig = defaultBasicGrpoTrainingConfig()
  const config: GrpoTrainingConfig = {
    ...baseConfig,
    ...options.config,
    accelerator: { ...baseConfig.accelerator, ...options.config?.accelerator },
  }
  const lora = { ...defaultBasicGrpoLoraConfig(), ...options.lora }
  validateGrpoTrainingConfig(config)
  validateGrpoLoraConfig(lora)
  validateGrpoDataset(dataset)
  return {
    config,
    dataset: dataset.map(example => ({
      answer: example.answer,
      prompt: example.prompt.map((message: GrpoDatasetExample['prompt'][number]) => ({
        role: message.role,
        content: message.content,
      })),
    })),
    lora: { ...lora, targetModules: [...lora.targetModules] },
    rewardPrograms: BASIC_GRPO_REWARD_PROGRAMS,
  }
}

/** Build a request directly from a caller-provided GSM8K source slice. */
export function createBasicGrpoTrainingRequestFromGsm8k(
  examples: readonly Gsm8kSourceExample[],
  options: BasicGrpoTrainingOptions = {},
): GrpoTrainingRequest {
  return createBasicGrpoTrainingRequest(mapGsm8kExamples(examples), options)
}

/** Run the checkpoint/report lifecycle through explicit host accelerator and storage ports. */
export async function runBasicGrpoTraining(
  request: GrpoTrainingRequest,
  dependencies: GrpoTrainingDependencies,
  options: { readonly clock?: () => Date; readonly signal?: AbortSignal } = {},
): Promise<GrpoTrainingExecution> {
  validateGrpoTrainingConfig(request.config)
  validateGrpoLoraConfig(request.lora)
  validateGrpoDataset(request.dataset)
  if (!dependencies?.accelerator || !dependencies.storage) {
    throw new TypeError('runBasicGrpoTraining requires explicit accelerator and storage ports')
  }

  const clock = options.clock ?? (() => new Date())
  const handle = await dependencies.accelerator.start(request, options.signal)
  const runId = requireText(handle?.id, 'host run id')
  const startedAt = clock().toISOString()
  await dependencies.storage.writeRun({ runId, request, startedAt, status: 'running' })
  await report(dependencies, { kind: 'started', runId, startedAt })

  try {
    const result = await waitForHostRun(handle, dependencies, options.signal)
    const finalModel = validateGrpoModelArtifact(result.finalModel)
    await dependencies.storage.writeFinalModel(runId, finalModel)
    const completedAt = clock().toISOString()
    await dependencies.storage.writeRun({
      runId,
      request,
      startedAt,
      completedAt,
      status: 'succeeded',
    })
    await report(dependencies, {
      kind: 'succeeded',
      runId,
      completedAt,
      ...(result.summary === undefined ? {} : { summary: result.summary }),
    })
    return {
      runId,
      startedAt,
      completedAt,
      finalModel,
      ...(result.summary === undefined ? {} : { summary: result.summary }),
    }
  } catch (error) {
    const completedAt = clock().toISOString()
    const message = errorMessage(error)
    await dependencies.storage.writeRun({ runId, request, startedAt, completedAt, status: 'failed', error: message })
    await report(dependencies, { kind: 'failed', runId, completedAt, error: message })
    throw new GrpoTrainingRunError(`host GRPO run ${runId} failed: ${message}`, { cause: error })
  }
}

/** Error with the host run identifier and original accelerator/storage cause retained. */
export class GrpoTrainingRunError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'GrpoTrainingRunError'
  }
}

/** Produce a JSON-safe description for the safe CLI without serializing reward functions. */
export function describeBasicGrpoTrainingRequest(request: GrpoTrainingRequest): Readonly<Record<string, unknown>> {
  return {
    kind: 'xerxes.grpo-training-request.v1',
    datasetExamples: request.dataset.length,
    config: request.config,
    lora: request.lora,
    rewards: request.rewardPrograms.map(program => ({ id: program.id, description: program.description })),
    hostBoundary: {
      accelerator: 'A caller-owned Bun/TypeScript provider must load the model and execute optimization.',
      storage: 'A caller-owned storage port must persist checkpoint and final-model references.',
    },
  }
}

async function waitForHostRun(
  handle: GrpoHostRunHandle,
  dependencies: GrpoTrainingDependencies,
  signal: AbortSignal | undefined,
): Promise<Awaited<ReturnType<GrpoHostRunHandle['wait']>>> {
  return handle.wait({
    onMetric: async rawMetric => {
      const metric = parseGrpoMetric(rawMetric)
      await persistMetric(handle.id, metric, dependencies)
    },
    onCheckpoint: async rawCheckpoint => {
      const checkpoint = validateGrpoCheckpoint(rawCheckpoint)
      await dependencies.storage.writeCheckpoint(handle.id, checkpoint)
      await report(dependencies, { kind: 'checkpoint', runId: handle.id, checkpoint })
    },
  }, signal)
}

async function persistMetric(
  runId: string,
  metric: GrpoTrainingMetric,
  dependencies: GrpoTrainingDependencies,
): Promise<void> {
  await dependencies.storage.writeMetric(runId, metric)
  await report(dependencies, { kind: 'metric', runId, metric })
}

async function report(
  dependencies: GrpoTrainingDependencies,
  event: GrpoTrainingReportEvent,
): Promise<void> {
  if (dependencies.reporter !== undefined) await dependencies.reporter.report(event)
}

function validateGrpoTrainingConfig(config: GrpoTrainingConfig): void {
  requireText(config.modelName, 'modelName')
  requireText(config.outputDirectory, 'outputDirectory')
  requireText(config.runName, 'runName')
  requirePositive(config.learningRate, 'learningRate')
  requireUnitInterval(config.adamBeta1, 'adamBeta1')
  requireUnitInterval(config.adamBeta2, 'adamBeta2')
  requireNonNegative(config.weightDecay, 'weightDecay')
  requireUnitInterval(config.warmupRatio, 'warmupRatio')
  requirePositiveInteger(config.perDeviceTrainBatchSize, 'perDeviceTrainBatchSize')
  requirePositiveInteger(config.gradientAccumulationSteps, 'gradientAccumulationSteps')
  requirePositiveInteger(config.numGenerations, 'numGenerations')
  requirePositiveInteger(config.maxPromptLength, 'maxPromptLength')
  requirePositiveInteger(config.maxCompletionLength, 'maxCompletionLength')
  requirePositive(config.numTrainEpochs, 'numTrainEpochs')
  requireNonNegative(config.maxGradNorm, 'maxGradNorm')
  requirePositiveInteger(config.loggingSteps, 'loggingSteps')
  requirePositiveInteger(config.saveSteps, 'saveSteps')
  if (config.reportTo !== 'none' && config.reportTo !== 'wandb') {
    throw new TypeError('reportTo must be "none" or "wandb"')
  }
  if (config.scheduler !== 'cosine' && config.scheduler !== 'linear' && config.scheduler !== 'constant') {
    throw new TypeError('scheduler must be "cosine", "linear", or "constant"')
  }
  if (
    config.accelerator.devicePlacement !== 'host-owned'
    || config.accelerator.modelLoading !== 'host-owned'
    || config.accelerator.optimizerExecution !== 'host-owned'
  ) {
    throw new TypeError('accelerator model loading, device placement, and optimizer execution must remain host-owned')
  }
  if (!['bf16', 'fp16', 'fp32'].includes(config.accelerator.precision)) {
    throw new TypeError('accelerator precision must be bf16, fp16, or fp32')
  }
  requireText(config.accelerator.attentionImplementation, 'accelerator attentionImplementation')
}

function validateGrpoLoraConfig(lora: GrpoLoraConfig): void {
  requirePositiveInteger(lora.rank, 'lora rank')
  requirePositive(lora.alpha, 'lora alpha')
  requireUnitInterval(lora.dropout, 'lora dropout')
  if (lora.taskType !== 'CAUSAL_LM') throw new TypeError('lora taskType must be CAUSAL_LM')
  if (!Array.isArray(lora.targetModules) || lora.targetModules.length === 0) {
    throw new TypeError('lora targetModules must contain at least one module')
  }
  for (const target of lora.targetModules) requireText(target, 'lora target module')
}

function validateGrpoDataset(dataset: readonly GrpoDatasetExample[]): void {
  for (const [index, example] of dataset.entries()) {
    if (!example || typeof example !== 'object') throw new TypeError(`dataset example ${index} must be an object`)
    if (example.answer !== null && typeof example.answer !== 'string') {
      throw new TypeError(`dataset example ${index} answer must be text or null`)
    }
    if (!Array.isArray(example.prompt) || example.prompt.length === 0) {
      throw new TypeError(`dataset example ${index} prompt must contain messages`)
    }
    for (const message of example.prompt) {
      if (message.role !== 'system' && message.role !== 'user') {
        throw new TypeError(`dataset example ${index} has an unsupported role`)
      }
      requireText(message.content, `dataset example ${index} message content`)
    }
  }
}

function requireText(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty string`)
  return value.trim()
}

function requirePositive(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive finite number`)
  }
  return value
}

function requireNonNegative(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new TypeError(`${name} must be a non-negative finite number`)
  }
  return value
}

function requirePositiveInteger(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
    throw new TypeError(`${name} must be a positive safe integer`)
  }
  return value
}

function requireUnitInterval(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${name} must be a number from 0 through 1`)
  }
  return value
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
