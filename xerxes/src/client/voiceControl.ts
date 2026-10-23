// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceStateStore } from './voiceState.js'

/**
 * Push-to-talk seam. `mobile/voice` implements it (microphone capture, the
 * Hub's speech-to-text port). The app only ever uses the transcript to fill an
 * editable draft: voice never sends a message, approves, or confirms anything.
 * When no implementation is injected, the talk button is not rendered.
 */
export interface VoiceControl {
  /** Begin recording. Rejects with a readable Error when the microphone is denied or unavailable. */
  start(): Promise<void>
  /** Stop recording and resolve with the transcript, or undefined when nothing was heard. */
  stop(): Promise<string | undefined>
  /** Abandon the recording without a transcript. */
  cancel(): void
  /** Live voice state for the orb and the interrupt control. */
  readonly state?: VoiceStateStore
  /** Flush playback and abort synthesis. Never cancels agents or the supervisor. */
  interrupt?(): void
  /**
   * Ask the Hub to speak the supervisor's next reply on this device. Await it
   * before sending a voice message so the Hub is listening when the reply lands.
   * Never rejects: a failure just means the reply is shown, not spoken.
   */
  armReply?(enabled: boolean): Promise<void>
  /** Optional hands-free session. Present only when the Hub reports live voice. */
  readonly live?: LiveVoiceControl
}

export interface LiveVoiceControl {
  readonly active: boolean
  /** Open a live session. Rejects with a readable Error. */
  start(): Promise<void>
  /** End the live session and release the microphone. */
  stop(): void
}
