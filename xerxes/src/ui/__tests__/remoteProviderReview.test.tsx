// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { RemoteProviderReview } from '../opentui/remoteProviderReview.js'
import { DEFAULT_THEME } from '../theme.js'
import type { RemoteTaskReview } from '../lib/remoteTaskSetup.js'
const value: RemoteTaskReview = {destination:'me@reviewed-host',workspace:'/remote/project',sessionId:'abcd1234',remoteModel:'remote-model',remoteProfile:'remote',localRequirement:'',running:false,localBindingSupported:true,inventoryError:'',profiles:[{name:'local-profile',model:'local-model',credentialSource:'local saved profile',supported:true,providerControlledOutput:false,setup:''}]}

for (const [width,height] of [[80,24],[40,18]]) it(`requires a separate authorization action at ${width}x${height}`,async()=>{
  const choose=vi.fn(),cancel=vi.fn()
  const screen=await testRender(<RemoteProviderReview value={value} t={DEFAULT_THEME} onChoose={choose} onCancel={cancel}/>,{width:width!,height:height!})
  const press=async(key:string)=>{await act(async()=>{screen.mockInput.pressKey(key);await Bun.sleep(key==='ESCAPE'?80:0)});await screen.flush()}
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Choose provider location')
    await press('ARROW_RIGHT');await press('RETURN')
    expect(choose).not.toHaveBeenCalled()
    expect(screen.captureCharFrame()).toContain('Authorize local provider')
    await press('RETURN')
    expect(choose).not.toHaveBeenCalled()
    await press('A')
    expect(choose).toHaveBeenCalledWith({kind:'local',profile:'local-profile',durationMinutes:60,maxRequests:200,maxOutputTokens:16384,maxConcurrent:4,consentProviderControlledOutput:false})
    expect(cancel).not.toHaveBeenCalled()
  } finally {await act(async()=>screen.renderer.destroy())}
})
it('defaults to retaining current setup without sharing local authority',async()=>{
  const choose=vi.fn()
  const screen=await testRender(<RemoteProviderReview value={value} t={DEFAULT_THEME} onChoose={choose} onCancel={()=>{}}/>,{width:80,height:24})
  try {await screen.flush();await act(async()=>screen.mockInput.pressKey('RETURN'));expect(choose).toHaveBeenCalledWith({kind:'remote'})}
  finally {await act(async()=>screen.renderer.destroy())}
})
it('requires explicit provider-controlled-output consent in addition to approval',async()=>{
  const choose=vi.fn()
  const screen=await testRender(<RemoteProviderReview value={{...value,profiles:[{...value.profiles[0]!,providerControlledOutput:true}]}} t={DEFAULT_THEME} onChoose={choose} onCancel={()=>{}}/>,{width:80,height:24})
  const press=async(key:string)=>{await act(async()=>screen.mockInput.pressKey(key));await screen.flush()}
  try {await screen.flush();await press('ARROW_RIGHT');await press('RETURN');await press('A');expect(choose).not.toHaveBeenCalled();await press('C');await press('A');expect(choose).toHaveBeenCalledWith(expect.objectContaining({maxOutputTokens:null,consentProviderControlledOutput:true}))}
  finally {await act(async()=>screen.renderer.destroy())}
})
it('running tasks cannot be rebound and Escape preserves their setup',async()=>{
  const choose=vi.fn(),cancel=vi.fn()
  const screen=await testRender(<RemoteProviderReview value={{...value,running:true}} t={DEFAULT_THEME} onChoose={choose} onCancel={cancel}/>,{width:80,height:24})
  try {await screen.flush();await act(async()=>screen.mockInput.pressKey('ARROW_RIGHT'));await act(async()=>screen.mockInput.pressKey('RETURN'));expect(choose).not.toHaveBeenCalled();await act(async()=>{screen.mockInput.pressKey('ESCAPE');await Bun.sleep(80)});expect(cancel).toHaveBeenCalledOnce()}
  finally {await act(async()=>screen.renderer.destroy())}
})

for(const [width,height] of [[80,24],[40,18]])it(`shows the existing local requirement without implying remote credentials at ${width}x${height}`,async()=>{
  const choose=vi.fn()
  const screen=await testRender(<RemoteProviderReview value={{...value,remoteProfile:'remote-fallback',localRequirement:'Local provider: chosen · requires local access'}} t={DEFAULT_THEME} onChoose={choose} onCancel={()=>{}}/>,{width:width!,height:height!})
  try {
    await screen.flush()
    const top=screen.captureCharFrame()
    expect(top).toContain('Local provider: chosen')
    expect(top).not.toContain('remote-fallback')
    expect(top).not.toContain('remote configuration')
    await act(async()=>screen.mockInput.pressKey('END'));await screen.flush()
    const bottom=screen.captureCharFrame()
    expect(bottom).not.toContain('existing remote credentials')
    expect(bottom).toContain('explicitly')
    expect(choose).not.toHaveBeenCalled()
  }finally{await act(async()=>screen.renderer.destroy())}
})
