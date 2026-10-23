// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceControl } from '../../client/voiceControl.js'
import { hubVoiceUrl, VoiceLink, type VoiceSocketFactory } from '../../client/voiceLink.js'
import { createVoiceStateStore } from '../../client/voiceState.js'
import type { VoiceCapabilities } from '../../hub/voice/protocol.js'
import { browserMicrophone, type MicrophoneDeps } from './capture.js'
import { VoiceClientCore } from './core.js'
import { createLiveVoice } from './live.js'
import { browserLiveCaptureDeps, type LiveCaptureDeps } from './liveCapture.js'
import { browserAudioOutput, type AudioOutput } from './playback.js'
import { probeVoiceCapabilities, supportsLive, supportsPushToTalk } from './probe.js'
import { createPushToTalk } from './pushToTalk.js'

export interface VoiceSetup {
  readonly location: { readonly protocol: string; readonly host: string }
  /** The supervisor thread id once the XERXES tab opened it. */
  readonly conversationId: () => string | undefined
  readonly socketFactory?: VoiceSocketFactory
  readonly mic?: MicrophoneDeps | undefined
  readonly output?: AudioOutput | undefined
  readonly liveCapture?: LiveCaptureDeps | undefined
  readonly probeTimeoutMs?: number
}

/**
 * Probe the Hub and build the voice control. Resolves undefined (so no talk
 * button is shown) when the browser cannot record, the Hub is unreachable, or
 * the Hub reports no usable voice mode. Nothing here opens the microphone.
 */
export async function createVoiceControl(setup: VoiceSetup): Promise<VoiceControl | undefined> {
  const mic = 'mic' in setup ? setup.mic : browserMicrophone()
  if (mic === undefined) return undefined
  const url = hubVoiceUrl(setup.location)
  const createLink = (): VoiceLink => new VoiceLink({ url, ...(setup.socketFactory === undefined ? {} : { socketFactory: setup.socketFactory }) })
  const capabilities: VoiceCapabilities | undefined = await probeVoiceCapabilities(createLink, setup.probeTimeoutMs)
  if (!supportsPushToTalk(capabilities)) return undefined
  const state = createVoiceStateStore()
  const output = 'output' in setup ? setup.output : browserAudioOutput()
  const core = new VoiceClientCore({ createLink, output, state })
  const base = createPushToTalk({ core, mic, capabilities, conversationId: setup.conversationId })
  const capture = 'liveCapture' in setup ? setup.liveCapture : browserLiveCaptureDeps()
  if (capture === undefined || !supportsLive(capabilities)) return base
  return { ...base, live: createLiveVoice({ core, capture, capabilities, conversationId: setup.conversationId }) }
}
