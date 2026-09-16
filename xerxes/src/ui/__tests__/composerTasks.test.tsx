/** @jsxImportSource @opentui/react */
// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, test } from 'vitest'
import { ComposerTaskSummary } from '../opentui/appLayout.js'
import { $uiState } from '../app/uiStore.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'
afterEach(()=>{resetTurnState();$uiState.set({...$uiState.get(),info:null})})
for(const width of [42,100,160])test(`task context is absent when empty and stays above input at ${width} columns`,async()=>{
 resetTurnState();$uiState.set({...$uiState.get(),info:null})
 const screen=await testRender(<box flexDirection="column" height="100%"><box flexGrow={1}><text>Transcript</text></box><ComposerTaskSummary/><text>Prompt input</text></box>,{width,height:22})
 try{
  await screen.flush();expect(screen.captureCharFrame()).not.toContain('Tasks');expect(screen.captureCharFrame()).not.toContain('Goal')
  act(()=>{$uiState.set({...$uiState.get(),info:{goal:'Review the project',goal_phase:'active'} as never})})
  await screen.flush();let frame=screen.captureCharFrame();expect(frame).toContain('Goal');expect(frame).not.toContain('0/0');expect(frame.indexOf('Review the project')).toBeLessThan(frame.indexOf('Prompt input'))
  act(()=>patchTurnState({todos:[{id:'one',content:'Inspect dependencies',status:'in_progress'},{id:'two',content:'Read instructions',status:'completed'}]}))
  await screen.flush();frame=screen.captureCharFrame();expect(frame).toContain('Tasks 1/2');expect(frame).toContain('Inspect dependencies')
  act(()=>{patchTurnState({todos:[]});$uiState.set({...$uiState.get(),info:null})})
  await screen.flush();frame=screen.captureCharFrame();expect(frame).not.toContain('Tasks');expect(frame).not.toContain('Goal');expect(frame).toContain('Prompt input')
 }finally{act(()=>screen.renderer.destroy())}
})
