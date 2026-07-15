// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  PromptContextBuilder,
  type PromptContextHost,
} from '../src/runtime/promptContext.js'
import {
  PromptProfile,
  getPromptProfileConfig,
} from '../src/runtime/promptProfiles.js'

test('prompt-context memory parity omits absent providers and caps injected memory at the selected profile limit', async () => {
  const noProvider = new PromptContextBuilder({ host: promptHost() })
  const empty = await noProvider.build()
  expect(empty.memorySection).toBe('')
  expect(empty.userProfileSection).toBe('')

  const calls: Array<readonly [string | undefined, number]> = []
  const builder = new PromptContextBuilder({
    host: promptHost(),
    memoryProvider: async (agentId, maximum) => {
      calls.push([agentId, maximum])
      return Array.from({ length: 20 }, (_, index) => `memory ${index}`)
    },
    userProfileProvider: async agentId => `expertise: native TypeScript for ${agentId}`,
  })
  const profile = {
    ...getPromptProfileConfig(PromptProfile.FULL),
    maxMemoriesInjected: 3,
  }
  const context = await builder.build({ agentId: 'coder', profile })

  expect(calls).toEqual([['coder', 3]])
  expect(context.memorySection).toContain('[Relevant Memories]')
  expect(context.memorySection.match(/memory /g)).toHaveLength(3)
  expect(context.userProfileSection).toContain('[User Profile]')
  expect(context.userProfileSection).toContain('native TypeScript for coder')
  expect(await builder.assembleSystemPromptPrefix({ agentId: 'coder', profile })).toContain('[Relevant Memories]')

  const callsBeforeMinimal = calls.length
  const minimal = await builder.build({ agentId: 'coder', profile: PromptProfile.MINIMAL })
  expect(minimal.memorySection).toBe('')
  expect(minimal.userProfileSection).toBe('')
  expect(calls.length).toBe(callsBeforeMinimal)
})

function promptHost(): PromptContextHost {
  return {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace/xerxes',
      workspaceName: 'xerxes',
      xerxesVersion: '0.2.0',
    }),
  }
}
