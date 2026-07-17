// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ProfileStore } from '../src/bridge/profiles.js'
import {
  PROVIDER_FLOW_ADD_LABEL,
  PROVIDER_FLOW_CUSTOM_MODEL_LABEL,
  PROVIDER_FLOW_EDIT_LABEL,
  PROVIDER_FLOW_REMOVE_LABEL,
  ProviderProfileFlow,
  canonicalProviderType,
  type ProviderFlowPrompt,
  type ProviderFlowTransition,
} from '../src/daemon/providerFlow.js'

function requirePrompt(transition: ProviderFlowTransition): ProviderFlowPrompt {
  if (!transition.prompt) {
    throw new Error('Expected a provider-flow question')
  }
  return transition.prompt
}

async function respond(
  flow: ProviderProfileFlow,
  transition: ProviderFlowTransition,
  answer: string,
): Promise<ProviderFlowTransition> {
  const prompt = requirePrompt(transition)
  const next = await flow.answer(prompt.requestId, { [prompt.question.questionId ?? 'answer']: answer })
  if (!next) {
    throw new Error('Provider flow rejected its expected response')
  }
  return next
}

function profileChoice(transition: ProviderFlowTransition, name: string): string {
  const choice = requirePrompt(transition).question.options?.find(option => option.startsWith(`${name}  (`))
  if (!choice) {
    throw new Error(`No main-menu option for ${name}`)
  }
  return choice
}

test('provider flow uses an injected model port and never places credentials in transitions', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-provider-flow-'))
  const store = new ProfileStore(join(directory, 'profiles.json'))
  const discoveries: Array<{ apiKey: string; baseUrl: string; provider: string }> = []
  const flow = new ProviderProfileFlow({
    profileStore: store,
    modelDiscovery: {
      async discover(input) {
        discoveries.push(input)
        return ['remote-model', 'gpt-4o']
      },
    },
  })
  try {
    let transition = await flow.start()
    expect(requirePrompt(transition).question.options).toContain(PROVIDER_FLOW_ADD_LABEL)

    transition = await respond(flow, transition, PROVIDER_FLOW_ADD_LABEL)
    expect(requirePrompt(transition).question.questionId).toBe('name')
    transition = await respond(flow, transition, 'production')
    expect(requirePrompt(transition).question.questionId).toBe('provider_type')
    transition = await respond(flow, transition, 'openai')
    expect(requirePrompt(transition).question.questionId).toBe('base_url')
    transition = await respond(flow, transition, '')
    expect(requirePrompt(transition).question.questionId).toBe('api_key')
    transition = await respond(flow, transition, 'top-secret-key')
    expect(requirePrompt(transition).question.options).toEqual(expect.arrayContaining([
      'gpt-4o',
      'remote-model',
      PROVIDER_FLOW_CUSTOM_MODEL_LABEL,
    ]))

    transition = await respond(flow, transition, 'remote-model')
    expect(transition).toMatchObject({ finished: true, reload: true })
    expect(JSON.stringify(transition)).not.toContain('top-secret-key')
    expect(transition.notice?.body).not.toContain('top-secret-key')
    expect(discoveries).toEqual([{
      apiKey: 'top-secret-key',
      baseUrl: 'https://api.openai.com/v1',
      provider: 'openai',
    }])
    expect(store.active()).toMatchObject({
      api_key: 'top-secret-key',
      base_url: 'https://api.openai.com/v1',
      model: 'remote-model',
      name: 'production',
      provider: 'openai',
    })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('provider flow redacts a model-discovery failure that includes the submitted credential', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-provider-flow-redact-'))
  const store = new ProfileStore(join(directory, 'profiles.json'))
  const flow = new ProviderProfileFlow({
    profileStore: store,
    modelDiscovery: {
      async discover(input) {
        throw new Error(`upstream rejected ${input.apiKey}`)
      },
    },
  })
  try {
    let transition = await flow.start()
    transition = await respond(flow, transition, PROVIDER_FLOW_ADD_LABEL)
    transition = await respond(flow, transition, 'redacted-failure')
    transition = await respond(flow, transition, 'openai')
    transition = await respond(flow, transition, '')
    transition = await respond(flow, transition, 'failure-secret')

    const prompt = requirePrompt(transition)
    expect(prompt.question.question).toContain('catalogue lookup was unavailable')
    expect(prompt.question.options).not.toContain('gpt-4o')
    expect(prompt.question.options).toContain(PROVIDER_FLOW_CUSTOM_MODEL_LABEL)
    expect(JSON.stringify(prompt)).not.toContain('failure-secret')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('provider flow handles aliases, switching, editing, removal, and cancellation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-provider-flow-actions-'))
  const store = new ProfileStore(join(directory, 'profiles.json'))
  try {
    const addFlow = new ProviderProfileFlow({ profileStore: store })
    let transition = await addFlow.start()
    transition = await respond(addFlow, transition, PROVIDER_FLOW_ADD_LABEL)
    transition = await respond(addFlow, transition, 'native-claude')
    transition = await respond(addFlow, transition, 'claude_code')
    expect(requirePrompt(transition).question.questionId).toBe('model')
    transition = await respond(addFlow, transition, 'sonnet')
    expect(transition.finished).toBeTrue()
    expect(store.active()).toMatchObject({
      api_key: '',
      base_url: 'claude-code://local',
      model: 'claude-code/sonnet',
      name: 'native-claude',
      provider: 'claude-code',
    })
    expect(canonicalProviderType('claude_code')).toBe('claude-code')
    expect(canonicalProviderType('')).toBe('auto')

    store.save({
      apiKey: 'old-key',
      baseUrl: 'https://provider.example/v1',
      model: 'old-model',
      name: 'other',
      provider: 'openai',
      setActive: false,
    })

    const switchFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await switchFlow.start()
    transition = await respond(switchFlow, transition, profileChoice(transition, 'other'))
    expect(transition).toMatchObject({ finished: true, reload: true })
    expect(store.active()?.name).toBe('other')

    const editProviderFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await editProviderFlow.start()
    transition = await respond(editProviderFlow, transition, PROVIDER_FLOW_EDIT_LABEL)
    transition = await respond(editProviderFlow, transition, 'other')
    transition = await respond(editProviderFlow, transition, 'provider_type')
    transition = await respond(editProviderFlow, transition, 'claude_code')
    expect(transition.finished).toBeTrue()
    expect(store.active()?.provider).toBe('claude-code')

    const redactFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await redactFlow.start()
    transition = await respond(redactFlow, transition, PROVIDER_FLOW_EDIT_LABEL)
    transition = await respond(redactFlow, transition, 'other')
    transition = await respond(redactFlow, transition, 'api_key')
    transition = await respond(redactFlow, transition, 'new-top-secret')
    expect(transition.notice?.body).toContain('***redacted***')
    expect(JSON.stringify(transition)).not.toContain('new-top-secret')
    expect(store.active()?.api_key).toBe('new-top-secret')

    const declineFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await declineFlow.start()
    transition = await respond(declineFlow, transition, PROVIDER_FLOW_REMOVE_LABEL)
    transition = await respond(declineFlow, transition, 'other')
    transition = await respond(declineFlow, transition, 'no')
    expect(transition.finished).toBeTrue()
    expect(store.list().some(profile => profile.name === 'other')).toBeTrue()

    const removeFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await removeFlow.start()
    transition = await respond(removeFlow, transition, PROVIDER_FLOW_REMOVE_LABEL)
    transition = await respond(removeFlow, transition, 'other')
    transition = await respond(removeFlow, transition, 'yes')
    expect(transition).toMatchObject({ finished: true, reload: true })
    expect(store.list().some(profile => profile.name === 'other')).toBeFalse()

    const beforeCancellation = store.list().map(profile => profile.name)
    const cancelFlow = new ProviderProfileFlow({ profileStore: store })
    transition = await cancelFlow.start()
    transition = await respond(cancelFlow, transition, PROVIDER_FLOW_ADD_LABEL)
    transition = await respond(cancelFlow, transition, 'Cancel')
    expect(transition.notice?.body).toBe('Cancelled.')
    expect(store.list().map(profile => profile.name)).toEqual(beforeCancellation)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
