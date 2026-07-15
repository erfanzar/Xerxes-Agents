// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A value that may be produced synchronously by a host integration or asynchronously by a model. */
export type MaybePromise<T> = T | Promise<T>

/** The defensive concern exercised by a benign safety probe. */
export type SafetyProbeCategory =
  | 'benign-helpfulness'
  | 'hidden-instructions'
  | 'instruction-integrity'
  | 'obfuscation-awareness'
  | 'privacy-boundary'

/** A transparent behavior the deterministic scorer can look for in a response. */
export type SafetyCriterion =
  | 'answer-benign-request'
  | 'identify-obfuscation'
  | 'identify-untrusted-content'
  | 'maintain-instruction-boundary'
  | 'offer-safe-alternative'
  | 'protect-hidden-instructions'
  | 'protect-private-data'

/** A safe, pre-authored probe for evaluating model robustness. */
export interface SafetyProbe {
  readonly category: SafetyProbeCategory
  readonly description: string
  readonly expectedCriteria: readonly SafetyCriterion[]
  readonly id: string
  readonly prompt: string
  /** Optional content treated as untrusted data, never as executable instructions. */
  readonly untrustedContent?: string
}

/** A non-executing indicator that untrusted text may be visually or structurally altered. */
export type ObfuscationIndicator =
  | 'control-characters'
  | 'format-characters'
  | 'mixed-script-token'
  | 'opaque-token'
  | 'unusual-whitespace'

/** A normalized representation of caller-supplied untrusted text. */
export interface NormalizedUntrustedText {
  readonly indicators: readonly ObfuscationIndicator[]
  readonly text: string
}

/** A defensive signal found without decoding or executing untrusted content. */
export type PromptInjectionSignal =
  | 'authority-impersonation'
  | 'instruction-override'
  | 'privileged-action-request'
  | 'protected-information-request'

/** The coarse risk level used to annotate, rather than execute, untrusted content. */
export type SafetyRiskLevel = 'high' | 'low' | 'medium' | 'none'

/** Defensive prompt-injection assessment with labels only, never raw pattern captures. */
export interface PromptInjectionAssessment extends NormalizedUntrustedText {
  readonly risk: SafetyRiskLevel
  readonly signals: readonly PromptInjectionSignal[]
}

/** Input delivered to an explicitly injected model implementation. */
export interface SafetyModelRequest {
  readonly injectionAssessment: PromptInjectionAssessment
  readonly modelInstruction: string
  readonly probe: SafetyProbe
  readonly prompt: string
  readonly signal?: AbortSignal
  readonly untrustedContent?: string
}

/** A model response used only for defensive evaluation. */
export interface SafetyModelResponse {
  readonly text: string
}

/** Explicit boundary for any local, remote, or test model used by the evaluator. */
export interface SafetyEvaluationModelPort {
  complete(request: SafetyModelRequest): MaybePromise<SafetyModelResponse>
  readonly id: string
}

/** Explainable result for a single expected safety behavior. */
export interface SafetyCriterionAssessment {
  readonly criterion: SafetyCriterion
  readonly passed: boolean
  readonly rationale: string
  readonly weight: number
}

/** Deterministic, inspectable score for one model response. */
export interface SafetyResponseScore {
  readonly criteria: readonly SafetyCriterionAssessment[]
  readonly earnedWeight: number
  readonly possibleWeight: number
  readonly score: number
}

/** Result from one model/probe pair. Failures are represented without hiding other results. */
export interface SafetyEvaluationResult {
  readonly durationMilliseconds: number
  readonly error: string | null
  readonly modelId: string
  readonly probeId: string
  readonly responseText: string | null
  readonly score: SafetyResponseScore | null
}

/** Aggregate status for one explicitly injected model. */
export interface SafetyModelSummary {
  readonly averageScore: number | null
  readonly completed: number
  readonly failed: number
  readonly modelId: string
}

/** Overall evaluation accounting, kept separate from per-response evidence. */
export interface SafetyEvaluationSummary {
  readonly averageScore: number | null
  readonly completed: number
  readonly failed: number
  readonly models: readonly SafetyModelSummary[]
}

/** The completed report. It is persisted only when the caller passes a report store. */
export interface SafetyEvaluationReport {
  readonly generatedAt: string
  readonly id: string
  readonly modelIds: readonly string[]
  readonly probeIds: readonly string[]
  readonly results: readonly SafetyEvaluationResult[]
  readonly summary: SafetyEvaluationSummary
}

/** Explicit opt-in persistence boundary for completed safety reports. */
export interface SafetyEvaluationReportStore {
  save(report: SafetyEvaluationReport): MaybePromise<void>
}

/** Dependencies for durable JSONL report storage. */
export interface SafetyReportFilesystemPort {
  appendText(path: string, text: string): MaybePromise<void>
  dirname(path: string): string
  ensureDirectory(path: string): MaybePromise<void>
}

/** Configuration for a single safety-evaluation run. */
export interface SafetyEvaluationOptions {
  readonly idFactory?: () => string
  readonly maxConcurrent?: number
  readonly now?: () => Date
  readonly probes?: readonly SafetyProbe[]
  readonly reportStore?: SafetyEvaluationReportStore
  readonly signal?: AbortSignal
}
