// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {parseLocalProviderCapabilities} from '../src/protocol/localProviderCapabilities.js'
import {boundedLocalCapabilities,localCapabilitySnapshot,localReasoningLevels} from '../src/daemon/localReasoningCapabilities.js'
import {catalogReasoningLevels,fallbackReasoningLevels,providerReasoningLevels,selectableEfforts} from '../src/llms/reasoningLevels.js'
const snapshot=()=>({version:1,model:'local-model',reasoning:{shape:'effort',efforts:['low','ultra'],canDisable:false,provenance:'provider_reported'}})

test('capability metadata is model-scoped, immutable, bounded and excludes arbitrary configuration',()=>{
  const source=snapshot(),parsed=parseLocalProviderCapabilities(source,'local-model')!
  source.reasoning.efforts.push('changed')
  expect(parsed.reasoning.efforts).toEqual(['low','ultra'])
  expect(Object.isFrozen(parsed.reasoning.efforts)).toBe(true)
  expect(()=>parseLocalProviderCapabilities(source,'another-model')).toThrow('Invalid local provider')
  for(const value of [
    {...snapshot(),api_key:'synthetic-private-key'},
    {...snapshot(),reasoning:{...snapshot().reasoning,description:'synthetic-private-key'}},
    {...snapshot(),reasoning:{...snapshot().reasoning,efforts:['low','low']}},
    {...snapshot(),reasoning:{...snapshot().reasoning,efforts:Array(17).fill('high')}},
    {...snapshot(),reasoning:{...snapshot().reasoning,efforts:['\u001b[31m']}},
    {...snapshot(),reasoning:{...snapshot().reasoning,shape:'inherent'}},
  ]) expect(()=>parseLocalProviderCapabilities(value,'local-model')).toThrow('Invalid local provider')
  expect(selectableEfforts(localReasoningLevels(parsed))).toEqual(['low','ultra'])
  expect(localReasoningLevels(parsed).defaultEffort).toBeUndefined()
})

test('a local snapshot preserves the actual control shape and catalog provenance',()=>{
  for(const model of ['gpt-4o','gpt-5','gpt-5.1']){
    const levels=catalogReasoningLevels(model,'openai')!,snapshot=localCapabilitySnapshot(model,levels)
    expect(snapshot.reasoning.provenance).toBe('bundled_catalog')
    expect(selectableEfforts(localReasoningLevels(snapshot))).toEqual(selectableEfforts(levels))
  }
})

test('a bounded local catalog lookup falls back locally on failure or timeout and cancels its port',async()=>{
  const fallback=fallbackReasoningLevels('anthropic')
  let cancelled=false
  const start=performance.now()
  const timeout=await boundedLocalCapabilities('local-model',signal=>{
    signal.addEventListener('abort',()=>{cancelled=true},{once:true})
    return new Promise(()=>{})
  },fallback,15)
  expect(performance.now()-start).toBeLessThan(1000)
  expect(cancelled).toBe(true)
  expect(timeout.reasoning.provenance).toBe('provider_fallback')
  const failed=await boundedLocalCapabilities('local-model',async()=>{throw new Error('synthetic-provider-secret')},fallback)
  expect(JSON.stringify(failed)).not.toContain('synthetic')
  const live=await boundedLocalCapabilities('local-model',async()=>providerReasoningLevels([{effort:'low',description:'private arbitrary description'},{effort:'ultra'}],'low'),fallback)
  expect(live.reasoning.provenance).toBe('provider_reported')
  expect(JSON.stringify(live)).not.toContain('description')
})
