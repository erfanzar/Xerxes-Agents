// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ValidationError } from '../core/errors.js'
import type { JsonObject, JsonValue } from '../types/toolCalls.js'
import type { ImageInspectionResult } from './types.js'

/** Read-only tool names that may be dispatched by the parallel operator tool. */
export const SAFE_PARALLEL_TOOL_NAMES: ReadonlySet<string> = new Set([
  'APIClient',
  'CSVProcessor',
  'Calculator',
  'DateTimeProcessor',
  'DuckDuckGoSearch',
  'GlobTool',
  'GrepTool',
  'JSONProcessor',
  'ListDir',
  'MathematicalFunctions',
  'RSSReader',
  'ReadFile',
  'StatisticalAnalyzer',
  'SystemInfo',
  'TextProcessor',
  'URLAnalyzer',
  'UnitConverter',
  'agent_memory_list',
  'agent_memory_read',
  'agent_memory_search',
  'agent_memory_status',
  'get_memory_statistics',
  'search_memory',
  'session_search',
  'skill_view',
  'skills_list',
])

export interface OperatorPatchRequest {
  readonly check: boolean
  readonly patch: string
  readonly workdir?: string
}

export interface OperatorPatchResult {
  readonly applied: boolean
  readonly checked: boolean
  readonly stdout?: string
  readonly workdir?: string
}

/** Host-owned capability that applies a validated unified patch. */
export interface OperatorPatchApplier {
  applyPatch(request: OperatorPatchRequest, signal?: AbortSignal): Promise<OperatorPatchResult>
}

export interface ParallelReadonlyCall {
  readonly input: JsonObject
  readonly name: string
}

/**
 * Explicit dispatcher for inspection-only calls. The state layer intersects
 * this declaration with {@link SAFE_PARALLEL_TOOL_NAMES} before dispatching.
 */
export interface ParallelReadonlyToolPort {
  readonly toolNames: ReadonlySet<string>
  execute(call: ParallelReadonlyCall, signal?: AbortSignal): Promise<JsonValue>
}

export interface OperatorImageInspection extends ImageInspectionResult {
  /** Trusted-only data URL used to form a subsequent multimodal user message. */
  readonly imageDataUrl?: string
}

export interface OperatorImageInspector {
  inspectImage(
    request: { readonly detail: ImageInspectionResult['detail']; readonly path: string },
    signal?: AbortSignal,
  ): Promise<OperatorImageInspection>
}

export type OperatorWebSearchKind = 'image' | 'text'
export type OperatorWebSearchRecency = 'day' | undefined

export interface OperatorWebSearchRequest {
  readonly domains: readonly string[]
  readonly kind: OperatorWebSearchKind
  readonly maxResults: number
  readonly query: string
  readonly recency: OperatorWebSearchRecency
}

export interface OperatorWebSearchResponse {
  readonly engine?: string
  readonly results: readonly JsonObject[]
}

export interface OperatorWeatherRequest {
  readonly location: string
}

export interface OperatorFinanceRequest {
  readonly kind: string
  readonly market?: string
  readonly ticker: string
}

export interface OperatorSportsRequest {
  readonly dateFrom?: string
  readonly dateTo?: string
  readonly fn: 'schedule' | 'standings'
  readonly league: string
  readonly numGames?: number
  readonly opponent?: string
  readonly team?: string
}

/**
 * Host-owned public-web capability. It deliberately has no default network
 * implementation, so embedding applications control credentials, routing,
 * rate limits, and source policy.
 */
export interface OperatorWebPort {
  finance(request: OperatorFinanceRequest, signal?: AbortSignal): Promise<JsonObject>
  search(request: OperatorWebSearchRequest, signal?: AbortSignal): Promise<OperatorWebSearchResponse>
  sports(request: OperatorSportsRequest, signal?: AbortSignal): Promise<JsonObject>
  weather(request: OperatorWeatherRequest, signal?: AbortSignal): Promise<JsonObject>
}

export interface OperatorClock {
  now(): Date
}

export interface ParallelToolCallResult {
  readonly error?: string
  readonly index: number
  readonly name?: string
  readonly ok: boolean
  readonly result?: JsonValue
}

/** Reject patch text before it crosses the host boundary. */
export function validateUnifiedPatch(patch: string): string {
  const normalized = patch.trim()
  if (!normalized) throw new ValidationError('patch', 'must be non-empty', patch)
  const hasHeaders = (normalized.includes('--- ') && normalized.includes('+++ ')) || normalized.includes('diff --git ')
  const hasHunk = /^@@ /m.test(normalized)
  if (!hasHeaders || !hasHunk) {
    throw new ValidationError('patch', 'must be a unified diff with ---/+++ headers and @@ hunks', patch)
  }
  return patch
}

/** Run independent inspection calls concurrently while preserving input order. */
export async function runParallelReadonlyCalls(
  calls: readonly JsonValue[],
  maxWorkers: number,
  port: ParallelReadonlyToolPort | undefined,
  signal?: AbortSignal,
): Promise<{ readonly maxWorkers: number; readonly results: readonly ParallelToolCallResult[] }> {
  const workers = normalizeWorkers(maxWorkers)
  if (port === undefined) {
    throw new ConfigurationError('operator.parallelReadonlyToolPort', 'must be injected before parallel_tools can dispatch calls')
  }
  const results = new Array<ParallelToolCallResult>(calls.length)
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (cursor < calls.length) {
      const index = cursor
      cursor += 1
      results[index] = await runReadonlyCall(index, calls[index], port, signal)
    }
  }
  await Promise.all(Array.from({ length: Math.min(workers, calls.length) }, () => worker()))
  return Object.freeze({ maxWorkers: workers, results: Object.freeze(results) })
}

/** Render offset time from an explicit clock without a network dependency. */
export function offsetTime(utcOffset: string, clock: OperatorClock): Record<string, string> {
  const minutes = offsetMinutes(utcOffset)
  const shifted = new Date(clock.now().getTime() + minutes * 60_000)
  const iso = shifted.toISOString().replace(/\.\d{3}Z$/, '+00:00')
  return Object.freeze({
    utc_offset: utcOffset,
    iso,
    time: iso.slice(11, 19),
    date: iso.slice(0, 10),
  })
}

export function requirePatchApplier(value: OperatorPatchApplier | undefined): OperatorPatchApplier {
  if (value === undefined) {
    throw new ConfigurationError('operator.patchApplier', 'must be injected before apply_patch can modify a workspace')
  }
  return value
}

export function requireImageInspector(value: OperatorImageInspector | undefined): OperatorImageInspector {
  if (value === undefined) {
    throw new ConfigurationError('operator.imageInspector', 'must be injected before view_image can inspect a local image')
  }
  return value
}

export function requireWebPort(value: OperatorWebPort | undefined): OperatorWebPort {
  if (value === undefined) {
    throw new ConfigurationError('operator.webPort', 'must be injected before public-web operator tools can make requests')
  }
  return value
}

function normalizeWorkers(value: number): number {
  if (!Number.isInteger(value)) throw new ValidationError('max_workers', 'must be an integer', value)
  return Math.max(1, Math.min(value, 16))
}

async function runReadonlyCall(
  index: number,
  value: JsonValue | undefined,
  port: ParallelReadonlyToolPort,
  signal: AbortSignal | undefined,
): Promise<ParallelToolCallResult> {
  if (!isPlainRecord(value)) return failedReadonlyCall(index, undefined, 'call spec must be an object')
  const name = typeof value.name === 'string' ? value.name.trim() : ''
  const candidateInput = value.input ?? value.arguments ?? {}
  if (!isPlainRecord(candidateInput)) return failedReadonlyCall(index, name || undefined, 'input must be an object')
  if (!name || !SAFE_PARALLEL_TOOL_NAMES.has(name) || !port.toolNames.has(name)) {
    return failedReadonlyCall(index, name || undefined, `parallel_tools only allows read-only safe tools; rejected ${JSON.stringify(name)}`)
  }
  try {
    const result = await port.execute({ name, input: candidateInput }, signal)
    return Object.freeze({ index, name, ok: true, result })
  } catch (error) {
    return failedReadonlyCall(index, name, errorMessage(error))
  }
}

function failedReadonlyCall(index: number, name: string | undefined, error: string): ParallelToolCallResult {
  return Object.freeze({ index, ...(name === undefined ? {} : { name }), ok: false, error })
}

function offsetMinutes(value: string): number {
  const match = /^([+-])(0\d|1[0-4]):([0-5]\d)$/.exec(value)
  if (match === null) throw new ValidationError('utc_offset', 'must use a UTC offset from -14:00 through +14:00', value)
  const sign = match[1] === '+' ? 1 : -1
  const hours = Number(match[2])
  const minutes = Number(match[3])
  if (hours === 14 && minutes !== 0) {
    throw new ValidationError('utc_offset', 'must use a UTC offset from -14:00 through +14:00', value)
  }
  return sign * (hours * 60 + minutes)
}

function isPlainRecord(value: JsonValue | undefined): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
