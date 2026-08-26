// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readTranscriptEventRecords } from './transcriptEventLog.js'

export interface TranscriptEventSequenceGap { readonly from: number; readonly to: number }
export interface TranscriptEventInspection {
  readonly duplicateEventIds: readonly string[]
  readonly duplicateSequences: readonly number[]
  readonly eventCount: number
  readonly firstSequence?: number
  readonly gaps: readonly TranscriptEventSequenceGap[]
  readonly lastSequence?: number
  readonly malformedLines: number
  readonly partialTail: boolean
  readonly sessionId: string
}

/** Inspect immutable event bytes without mutating or attempting repair. */
export function inspectTranscriptEventLog(bytes: Uint8Array, sessionId: string): TranscriptEventInspection {
  const decoded = readTranscriptEventRecords(bytes, sessionId)
  const ids = new Set<string>()
  const duplicateIds = new Set<string>()
  const sequenceCounts = new Map<number, number>()
  for (const event of decoded.events) {
    if (event.eventId !== undefined) {
      if (ids.has(event.eventId)) duplicateIds.add(event.eventId)
      ids.add(event.eventId)
    }
    if (event.sequence !== undefined) sequenceCounts.set(event.sequence, (sequenceCounts.get(event.sequence) ?? 0) + 1)
  }
  const sequences = [...sequenceCounts.keys()].sort((left, right) => left - right)
  const gaps: TranscriptEventSequenceGap[] = []
  for (let index = 1; index < sequences.length; index += 1) {
    const previous = sequences[index - 1]!
    const current = sequences[index]!
    if (current > previous + 1) gaps.push({ from: previous + 1, to: current - 1 })
  }
  return {
    duplicateEventIds: [...duplicateIds].sort(),
    duplicateSequences: [...sequenceCounts].filter(([, count]) => count > 1).map(([sequence]) => sequence),
    eventCount: decoded.events.length,
    ...(sequences.length === 0 ? {} : { firstSequence: sequences[0]!, lastSequence: sequences[sequences.length - 1]! }),
    gaps,
    malformedLines: decoded.malformedLines,
    partialTail: decoded.partialTail,
    sessionId,
  }
}
