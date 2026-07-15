// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SkillTelemetry, type SkillTelemetryEvent } from './lifecycle.js'

export {
  SkillTelemetry,
  SkillLifecycleManager,
  SkillVariant,
  SkillVariantPicker,
  feedbackScore,
  percentile,
  successRate,
} from './lifecycle.js'
export type {
  DeprecationAction,
  DeprecationDecision,
  SkillAuthoredEvent,
  SkillFeedbackEvent,
  SkillLifecycleManagerOptions,
  SkillRetirementPort,
  SkillStats,
  SkillTelemetryEvent,
  SkillUsageEvent,
  SkillVariantOptions,
} from './lifecycle.js'

/** Returned by a host-owned telemetry source when it supports structured cleanup. */
export interface SkillTelemetrySubscription {
  unsubscribe(): void
}

/**
 * Explicit audit/telemetry source boundary.
 *
 * A daemon or audit subsystem chooses how events are delivered; the authoring
 * layer only consumes the normalized, in-memory telemetry event shape.
 */
export interface SkillTelemetrySourcePort {
  subscribe(listener: (event: SkillTelemetryEvent) => void): (() => void) | SkillTelemetrySubscription
}

/**
 * Attach a telemetry aggregator to a caller-owned event source.
 *
 * The returned function is idempotent and stops subsequent source events from
 * updating the aggregator. No audit subscription is created unless a host calls
 * this function with an explicit source port.
 */
export function attachSkillTelemetry(telemetry: SkillTelemetry, source: SkillTelemetrySourcePort): () => void {
  let active = true
  const subscription = source.subscribe(event => {
    if (active) {
      telemetry.record(event)
    }
  })
  const unsubscribe = typeof subscription === 'function'
    ? subscription
    : () => subscription.unsubscribe()
  return () => {
    if (!active) {
      return
    }
    active = false
    unsubscribe()
  }
}
