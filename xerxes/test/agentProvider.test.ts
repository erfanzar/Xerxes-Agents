// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { agentProviderResolver, agentProviderRouteResolver, providerRouteIdentity } from '../src/daemon/agentProvider.js'
import type { ProviderProfile } from '../src/bridge/profiles.js'

test('missing child profiles suggest exact host names before routing or client creation', () => {
  const profile: ProviderProfile = {name:'zai-glm',provider:'zhipu',api_key:'never-expose',base_url:'https://private-endpoint.invalid',model:'glm-5.3-flash',sampling:{}}
  const profiles={get:(name:string)=>name===profile.name?profile:undefined,list:()=>[{...profile,active:false}]}
  let clients=0
  const resolver=agentProviderResolver(profiles,()=>{clients++;throw Error('Must not construct a client')})
  for (const resolve of [resolver,agentProviderRouteResolver(profiles)]) {
    try {resolve('zai','glm-5.3-flash');throw Error('Expected rejection')}
    catch(error){
      const message=error instanceof Error?error.message:''
      expect(message).toContain('Configured profiles for this model on the execution host: "zai-glm"')
      expect(message).toContain('No fallback provider was selected')
      expect(message).not.toContain(profile.api_key)
      expect(message).not.toContain(profile.base_url)
    }
  }
  expect(clients).toBe(0)
})

test('child provider resolver isolates credentials and limits for each selected profile', () => {
  const profile: ProviderProfile = { name: 'child', provider: 'openai', api_key: 'fixture-child-key', base_url: 'https://child.invalid/v1', model: 'fixture', sampling: {}, model_overrides: { fixture: { context_limit: 8192, max_output_tokens: 1024 } } }
  const calls: unknown[] = []
  const resolver = agentProviderResolver({ get: name => name === 'child' ? profile : undefined }, (model, overrides) => {
    calls.push({ model, overrides })
    return { async *stream() { yield { content: 'ready' } } }
  })
  const selected = resolver('child', 'fixture')
  expect(calls).toEqual([{ model: 'fixture', overrides: { provider: 'openai', api_key: 'fixture-child-key', base_url: 'https://child.invalid/v1' } }])
  expect(selected.contextLimit?.('fixture')).toBe(8192)
  expect(selected.maxOutputTokens?.('fixture')).toBe(1024)
  expect(() => resolver('missing', 'fixture')).toThrow('unavailable')
  expect(calls).toHaveLength(1)
})

test('saved routes permit credential rotation but reject endpoint changes before client creation', () => {
  let profile: ProviderProfile = { name: 'child', provider: 'openai', api_key: 'old-key', base_url: 'https://one.invalid/v1', model: 'fixture', sampling: {} }
  const store = { get: () => profile }
  const route = agentProviderRouteResolver(store)('child', 'fixture')
  const credentials: unknown[] = []
  const resolver = agentProviderResolver(store, (_model, overrides) => {
    credentials.push(overrides?.api_key)
    return { async *stream() { yield { content: 'ready' } } }
  })
  profile = { ...profile, api_key: 'rotated-key' }
  resolver('child', 'fixture', route)
  expect(credentials).toEqual(['rotated-key'])
  expect(route).toMatch(/^[a-f0-9]{64}$/)
  profile = { ...profile, base_url: 'https://two.invalid/v1' }
  expect(() => resolver('child', 'fixture', route)).toThrow('route changed')
  expect(credentials).toHaveLength(1)
})

test('routing identity distinguishes providers and transport while honoring inline connections', () => {
  const original = providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1' })
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1' })).toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://profile.invalid/v1' })).not.toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai-codex', baseUrl: 'https://inline.invalid/v1' })).not.toBe(original)
  expect(providerRouteIdentity('fixture', { provider: 'openai', baseUrl: 'https://inline.invalid/v1', responsesApi: true })).not.toBe(original)
})
