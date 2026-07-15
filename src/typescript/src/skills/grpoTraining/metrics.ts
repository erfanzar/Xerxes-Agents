// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { GrpoCheckpoint, GrpoModelArtifact, GrpoTrainingMetric } from './types.js'

const KNOWN_GRPO_METRIC_KEYS = new Set([
  'step',
  'global_step',
  'loss',
  'reward',
  'learningRate',
  'learning_rate',
  'timestamp',
  'metrics',
])

/** Normalize a provider metric payload while preserving extra numeric measurements. */
export function parseGrpoMetric(input: unknown): GrpoTrainingMetric {
  if (!isRecord(input)) throw new TypeError('GRPO metric must be an object')
  const step = requiredStep(input.step ?? input.global_step)
  const loss = optionalNumber(input.loss, 'loss')
  const reward = optionalNumber(input.reward, 'reward')
  const learningRate = optionalNumber(input.learningRate ?? input.learning_rate, 'learning rate')
  const timestamp = optionalText(input.timestamp, 'timestamp')
  const values = numericValues(input)
  return {
    step,
    ...(loss === undefined ? {} : { loss }),
    ...(reward === undefined ? {} : { reward }),
    ...(learningRate === undefined ? {} : { learningRate }),
    ...(timestamp === undefined ? {} : { timestamp }),
    ...(Object.keys(values).length === 0 ? {} : { values }),
  }
}

/** Validate a checkpoint reference before it enters caller-owned storage. */
export function validateGrpoCheckpoint(input: GrpoCheckpoint): GrpoCheckpoint {
  if (!input || typeof input !== 'object') throw new TypeError('GRPO checkpoint must be an object')
  const id = requiredText(input.id, 'checkpoint id')
  const step = requiredStep(input.step)
  const location = optionalText(input.location, 'checkpoint location')
  const metadata = validateMetadata(input.metadata, 'checkpoint metadata')
  return {
    id,
    step,
    ...(location === undefined ? {} : { location }),
    ...(metadata === undefined ? {} : { metadata }),
  }
}

/** Validate a final model reference before it enters caller-owned storage. */
export function validateGrpoModelArtifact(input: GrpoModelArtifact): GrpoModelArtifact {
  if (!input || typeof input !== 'object') throw new TypeError('GRPO final model must be an object')
  const id = requiredText(input.id, 'final model id')
  const location = optionalText(input.location, 'final model location')
  const metadata = validateMetadata(input.metadata, 'final model metadata')
  return {
    id,
    ...(location === undefined ? {} : { location }),
    ...(metadata === undefined ? {} : { metadata }),
  }
}

function numericValues(input: Record<string, unknown>): Record<string, number> {
  const values: Record<string, number> = {}
  for (const [name, value] of Object.entries(input)) {
    if (KNOWN_GRPO_METRIC_KEYS.has(name)) continue
    if (typeof value === 'number' && Number.isFinite(value)) values[name] = value
  }
  if (isRecord(input.metrics)) {
    for (const [name, value] of Object.entries(input.metrics)) {
      if (typeof value === 'number' && Number.isFinite(value)) values[name] = value
    }
  }
  return values
}

function optionalNumber(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError(`${name} must be a finite number`)
  return value
}

function optionalText(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined
  return requiredText(value, name)
}

function requiredStep(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new TypeError('GRPO metric/checkpoint step must be a non-negative safe integer')
  }
  return value
}

function requiredText(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty string`)
  return value.trim()
}

function validateMetadata(
  value: unknown,
  name: string,
): Readonly<Record<string, string | number | boolean | null>> | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value)) throw new TypeError(`${name} must be an object`)
  const metadata: Record<string, string | number | boolean | null> = {}
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== 'string' && typeof item !== 'number' && typeof item !== 'boolean' && item !== null) {
      throw new TypeError(`${name}.${key} must be a JSON scalar`)
    }
    metadata[key] = item
  }
  return metadata
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
