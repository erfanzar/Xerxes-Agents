// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A value returned synchronously in tests or asynchronously by a host integration. */
export type GrpoMaybePromise<T> = T | Promise<T>

/** The chat roles used by the original GSM8K GRPO prompt. */
export type GrpoChatRole = 'system' | 'user'

/** One chat message sent to a host-owned GRPO provider. */
export interface GrpoChatMessage {
  readonly content: string
  readonly role: GrpoChatRole
}

/** The source shape of an example from the GSM8K `main` subset. */
export interface Gsm8kSourceExample {
  readonly answer: string
  readonly question: string
}

/** A GSM8K row normalized to the prompt/answer shape consumed by GRPO hosts. */
export interface GrpoDatasetExample {
  readonly answer: string | null
  readonly prompt: readonly GrpoChatMessage[]
}

/** A completion accepted by the native reward functions. */
export type GrpoCompletion = string | GrpoCompletionMessage | readonly GrpoCompletionMessage[]

/** A completion message, matching the first content item used by the Python template. */
export interface GrpoCompletionMessage {
  readonly content: string
}

/** Inputs supplied to one GRPO reward program. */
export interface GrpoRewardInput {
  readonly answers?: readonly (string | null | undefined)[]
  readonly completions: readonly GrpoCompletion[]
  readonly prompts?: readonly (readonly GrpoChatMessage[])[]
}

/** A named reward program a JavaScript-native or remote GRPO host can execute. */
export interface GrpoRewardProgram {
  readonly description: string
  readonly evaluate: (input: GrpoRewardInput) => readonly number[]
  readonly id: string
}

/** The LoRA adapter configuration from the bundled reference template. */
export interface GrpoLoraConfig {
  readonly alpha: number
  readonly dropout: number
  readonly rank: number
  readonly targetModules: readonly string[]
  readonly taskType: 'CAUSAL_LM'
}

/** Requirements delegated to the accelerator/provider integration, never emulated by Bun. */
export interface GrpoHostAcceleratorRequirements {
  readonly attentionImplementation: string
  readonly devicePlacement: 'host-owned'
  readonly modelLoading: 'host-owned'
  readonly optimizerExecution: 'host-owned'
  readonly precision: 'bf16' | 'fp16' | 'fp32'
}

/** Training settings that preserve the meaningful configuration of the reference GRPO script. */
export interface GrpoTrainingConfig {
  readonly accelerator: GrpoHostAcceleratorRequirements
  readonly adamBeta1: number
  readonly adamBeta2: number
  readonly gradientAccumulationSteps: number
  readonly learningRate: number
  readonly loggingSteps: number
  readonly maxCompletionLength: number
  readonly maxGradNorm: number
  readonly maxPromptLength: number
  readonly modelName: string
  readonly numGenerations: number
  readonly numTrainEpochs: number
  readonly outputDirectory: string
  readonly perDeviceTrainBatchSize: number
  readonly reportTo: 'none' | 'wandb'
  readonly runName: string
  readonly saveSteps: number
  readonly scheduler: 'cosine' | 'linear' | 'constant'
  readonly warmupRatio: number
  readonly weightDecay: number
}

/** Immutable request handed to an explicitly configured host accelerator/provider. */
export interface GrpoTrainingRequest {
  readonly config: GrpoTrainingConfig
  readonly dataset: readonly GrpoDatasetExample[]
  readonly lora: GrpoLoraConfig
  readonly rewardPrograms: readonly GrpoRewardProgram[]
}

/** A numeric metric emitted by a host-owned training execution. */
export interface GrpoTrainingMetric {
  readonly learningRate?: number
  readonly loss?: number
  readonly reward?: number
  readonly step: number
  readonly timestamp?: string
  readonly values?: Readonly<Record<string, number>>
}

/** A checkpoint reference emitted by the host that owns model bytes and accelerator storage. */
export interface GrpoCheckpoint {
  readonly id: string
  readonly location?: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
  readonly step: number
}

/** A final model reference emitted by the host that materialized the trained artifact. */
export interface GrpoModelArtifact {
  readonly id: string
  readonly location?: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
}

/** Final result returned after a host completes a GRPO run. */
export interface GrpoHostRunResult {
  readonly finalModel: GrpoModelArtifact
  readonly summary?: Readonly<Record<string, number>>
}

/** Lifecycle callback surface the host uses while running a requested GRPO job. */
export interface GrpoHostRunEvents {
  onCheckpoint(checkpoint: GrpoCheckpoint): GrpoMaybePromise<void>
  onMetric(metric: unknown): GrpoMaybePromise<void>
}

/** A running host-owned GRPO job. Model execution remains entirely behind this boundary. */
export interface GrpoHostRunHandle {
  readonly id: string
  cancel?(): GrpoMaybePromise<void>
  wait(events: GrpoHostRunEvents, signal?: AbortSignal): GrpoMaybePromise<GrpoHostRunResult>
}

/**
 * Host-owned accelerator/provider boundary. Implement it with a Bun-native
 * accelerator SDK, a remote training service, or a JavaScript/WASM backend.
 * This skill intentionally does not load PyTorch, Transformers, or Python.
 */
export interface GrpoHostAcceleratorProviderPort {
  start(request: GrpoTrainingRequest, signal?: AbortSignal): GrpoMaybePromise<GrpoHostRunHandle>
}

/** Persisted lifecycle state for an externally-owned GRPO run. */
export interface GrpoTrainingRunRecord {
  readonly completedAt?: string
  readonly error?: string
  readonly request: GrpoTrainingRequest
  readonly runId: string
  readonly startedAt: string
  readonly status: 'failed' | 'running' | 'succeeded'
}

/**
 * Explicit durable-storage boundary. The implementation decides how to retain
 * checkpoint/model references; it never receives or writes Python model state.
 */
export interface GrpoTrainingStoragePort {
  writeCheckpoint(runId: string, checkpoint: GrpoCheckpoint): GrpoMaybePromise<void>
  writeFinalModel(runId: string, model: GrpoModelArtifact): GrpoMaybePromise<void>
  writeMetric(runId: string, metric: GrpoTrainingMetric): GrpoMaybePromise<void>
  writeRun(record: GrpoTrainingRunRecord): GrpoMaybePromise<void>
}

/** A lifecycle event suitable for W&B, logs, audit sinks, or a user interface. */
export type GrpoTrainingReportEvent =
  | { readonly kind: 'started'; readonly runId: string; readonly startedAt: string }
  | { readonly kind: 'metric'; readonly metric: GrpoTrainingMetric; readonly runId: string }
  | { readonly checkpoint: GrpoCheckpoint; readonly kind: 'checkpoint'; readonly runId: string }
  | {
    readonly completedAt: string
    readonly kind: 'succeeded'
    readonly runId: string
    readonly summary?: Readonly<Record<string, number>>
  }
  | { readonly completedAt: string; readonly error: string; readonly kind: 'failed'; readonly runId: string }

/** Optional reporting boundary, distinct from durable checkpoint storage. */
export interface GrpoTrainingReporterPort {
  report(event: GrpoTrainingReportEvent): GrpoMaybePromise<void>
}

/** Dependencies required to launch and persist a GRPO training lifecycle. */
export interface GrpoTrainingDependencies {
  readonly accelerator: GrpoHostAcceleratorProviderPort
  readonly reporter?: GrpoTrainingReporterPort
  readonly storage: GrpoTrainingStoragePort
}

/** A completed native orchestration result. */
export interface GrpoTrainingExecution {
  readonly completedAt: string
  readonly finalModel: GrpoModelArtifact
  readonly runId: string
  readonly startedAt: string
  readonly summary?: Readonly<Record<string, number>>
}

/** Explicit dataset-loading boundary for the public GSM8K source. */
export interface Gsm8kDatasetPort {
  loadGsm8k(input: { readonly config: 'main'; readonly split: string }): GrpoMaybePromise<readonly Gsm8kSourceExample[]>
}
