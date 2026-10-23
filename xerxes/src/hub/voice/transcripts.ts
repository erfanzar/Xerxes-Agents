// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceMode } from './protocol.js'

/**
 * What voice persists: final transcripts, who said them, which model heard
 * them, and interruptions. There is deliberately no field that could hold
 * audio, and nothing in this module reads or writes audio bytes.
 */
export type VoiceTranscriptRecord =
  | {
      readonly kind: 'transcript' | 'submitted'
      readonly conversationId: string
      readonly utteranceId: string
      readonly userId: string
      readonly deviceId: string
      readonly displayName: string
      readonly mode: VoiceMode
      readonly text: string
      readonly provider?: string
      readonly model?: string
      /** True when the submitted text differs from the recognized text. */
      readonly edited?: boolean
      readonly at: string
    }
  | {
      readonly kind: 'interruption'
      readonly conversationId: string
      readonly responseId: string
      readonly userId: string
      readonly deviceId: string
      readonly playedMs?: number
      readonly at: string
    }

export interface VoiceTranscriptStore {
  append(record: VoiceTranscriptRecord): Promise<void>
}

/** Bounded in-memory store: the oldest records are dropped first. */
export class MemoryVoiceTranscriptStore implements VoiceTranscriptStore {
  readonly records: VoiceTranscriptRecord[] = []
  constructor(private readonly maxRecords = 1000) {}

  async append(record: VoiceTranscriptRecord): Promise<void> {
    this.records.push(record)
    while (this.records.length > this.maxRecords) this.records.shift()
  }
}
