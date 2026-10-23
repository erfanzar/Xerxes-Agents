// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceLink } from '../../client/voiceLink.js'
import type { VoiceCapabilities } from '../../hub/voice/protocol.js'

/**
 * Ask the Hub what voice can do. Resolves undefined when the endpoint is
 * unreachable, refuses the session, or does not answer in time: the talk
 * button is then not shown. It never guesses that voice is available.
 */
export async function probeVoiceCapabilities(createLink: () => VoiceLink, timeoutMs = 4000): Promise<VoiceCapabilities | undefined> {
  const link = createLink()
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await new Promise<VoiceCapabilities | undefined>(resolve => {
      const off = link.onEvent(event => {
        if (event.kind === 'control' && event.message.type === 'capabilities') resolve(event.message.capabilities)
        else if (event.kind === 'closed') resolve(undefined)
      })
      timer = setTimeout(() => resolve(undefined), timeoutMs)
      link.connect().then(
        () => { if (!link.sendControl({ type: 'capabilities' })) resolve(undefined) },
        () => resolve(undefined),
      ).finally(() => { void off })
    })
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    link.close(1000, 'probe done')
  }
}

/** Whether this capability reply allows the push-to-talk button. */
export function supportsPushToTalk(capabilities: VoiceCapabilities | undefined): capabilities is VoiceCapabilities {
  return capabilities !== undefined && capabilities.enabled && capabilities.modes.includes('push_to_talk') && capabilities.stt.available
}

export function supportsLive(capabilities: VoiceCapabilities | undefined): capabilities is VoiceCapabilities {
  return capabilities !== undefined && capabilities.enabled && capabilities.modes.includes('live') && capabilities.realtime.available
}
