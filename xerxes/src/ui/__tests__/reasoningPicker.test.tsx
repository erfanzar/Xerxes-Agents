// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import { ReasoningPicker } from '../opentui/reasoningPicker.js'
import { DEFAULT_THEME } from '../theme.js'

describe('OpenTUI reasoning picker', () => {
  it('keeps long local capability recovery instructions keyboard-accessible at 40 columns',async()=>{
    const onCancel=vi.fn(),onSelect=vi.fn()
    const services={gw:{} as GatewayClient,rpc:vi.fn(async()=>({levels:[],note:'Local reasoning capabilities are unavailable. '+ 'Details about this local provider. '.repeat(18)+'Reopen this SSH task and authorize its local provider.',shape:'unknown'}))} as unknown as GatewayServices
    const setup=await testRender(<GatewayProvider value={services}><ReasoningPicker onCancel={onCancel} onSelect={onSelect} t={DEFAULT_THEME}/></GatewayProvider>,{height:18,width:40})
    try {
      await act(async()=>{await Bun.sleep(0)});await setup.flush()
      expect(setup.captureCharFrame()).toContain('PgUp/PgDn scroll details')
      for(let i=0;i<8;i++){act(()=>setup.renderer.keyInput.processParsedKey({name:'pagedown',sequence:'\u001b[6~',raw:'\u001b[6~',ctrl:false,meta:false,shift:false,option:false,eventType:'press',source:'raw'}));await setup.flush()}
      expect(setup.captureCharFrame().replace(/\s+/g,' ')).toContain('authorize its local provider.')
      act(()=>setup.renderer.keyInput.processParsedKey({name:'escape',sequence:'\u001b',raw:'\u001b',ctrl:false,meta:false,shift:false,option:false,eventType:'press',source:'raw'}))
      expect(onCancel).toHaveBeenCalledTimes(1)
      expect(onSelect).not.toHaveBeenCalled()
    } finally {act(()=>setup.renderer.destroy())}
  })

  it('retains the highlighted level below wrapped capability details in a narrow terminal',async()=>{
    const onSelect=vi.fn()
    const services={gw:{} as GatewayClient,rpc:vi.fn(async()=>({current:'level-0',levels:Array.from({length:12},(_,i)=>({effort:'level-'+i})),note:'Capability snapshot from the local bundled model catalog. Changes apply only to this task; local profile defaults stay local.'}))} as unknown as GatewayServices
    const setup=await testRender(<GatewayProvider value={services}><ReasoningPicker onSelect={onSelect} t={DEFAULT_THEME}/></GatewayProvider>,{height:18,width:40})
    try {
      await act(async()=>{await Bun.sleep(0)});await setup.flush()
      act(()=>{for(let i=0;i<10;i++)setup.renderer.keyInput.processParsedKey({name:'down',sequence:'\u001b[B',raw:'\u001b[B',ctrl:false,meta:false,shift:false,option:false,eventType:'press',source:'raw'})})
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('● level-10')
      act(()=>setup.renderer.keyInput.processParsedKey({name:'return',sequence:'\r',raw:'\r',ctrl:false,meta:false,shift:false,option:false,eventType:'press',source:'raw'}))
      expect(onSelect).toHaveBeenCalledWith('level-10')
    } finally {act(()=>setup.renderer.destroy())}
  })
  it('keeps the highlighted effort visible after moving beyond the first page', async () => {
    const levels = Array.from({ length: 12 }, (_, index) => ({
      description: `description ${index}`,
      effort: `effort-${index}`
    }))
    let resolveLevels!: (value: unknown) => void
    const levelsResponse = new Promise(resolve => (resolveLevels = resolve))
    const services = {
      gw: {} as GatewayClient,
      rpc: vi.fn(() => levelsResponse)
    } as unknown as GatewayServices
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ReasoningPicker onSelect={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await act(async () => {
        resolveLevels({ current: 'effort-0', default: 'effort-0', levels })
        await Bun.sleep(0)
      })
      await setup.flush()

      act(() => {
        for (let index = 0; index < 10; index += 1) {
          setup.renderer.keyInput.processParsedKey({
            ctrl: false,
            eventType: 'press',
            meta: false,
            name: 'down',
            option: false,
            raw: '\u001b[B',
            sequence: '\u001b[B',
            shift: false,
            source: 'raw'
          })
        }
      })
      await setup.flush()

      const frame = setup.captureCharFrame()
      expect(frame).toContain('effort-10')
      expect(frame).toContain('● effort-10')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
