// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { Usage } from '../types.js'

export const ZERO: Usage = { cache_read: 0, cache_write: 0, calls: 0, input: 0, output: 0, total: 0 }

/** Per-provider-round timing carried by a live daemon status_update. */
export interface UsageTelemetryDelta {
  llm_ms?: number
  ttft_ms?: number
}

/**
 * Merge cumulative counters while accumulating explicitly marked live timing.
 *
 * Session payloads expose cumulative `llm_duration_ms`/TTFT telemetry, whereas
 * usage_update events carry one provider round under the same wire-era names.
 * A plain object spread replaced the cumulative LLM time with the final round
 * while retaining the old TTFT average, producing impossible status lines.
 */
export function mergeLiveUsage(
  base: Usage,
  incoming: Usage | undefined,
  delta: UsageTelemetryDelta | undefined
): Usage {
  const next: Usage = incoming ? { ...base, ...incoming } : { ...base }
  if (!delta) return next

  if (delta.llm_ms !== undefined && Number.isFinite(delta.llm_ms)) {
    next.llm_ms = (base.llm_ms ?? 0) + Math.max(0, delta.llm_ms)
    next.llm_steps = (base.llm_steps ?? 0) + 1
  }
  if (delta.ttft_ms !== undefined && Number.isFinite(delta.ttft_ms)) {
    const fallbackSamples = base.ttft_avg_ms
      ? Math.max(1, base.llm_steps ?? 1)
      : 0
    const samples = base.ttft_samples ?? fallbackSamples
    const total = base.ttft_total_ms ?? (base.ttft_avg_ms ?? 0) * samples
    next.ttft_samples = samples + 1
    next.ttft_total_ms = total + Math.max(0, delta.ttft_ms)
    next.ttft_avg_ms = next.ttft_total_ms / next.ttft_samples
  }
  return next
}
