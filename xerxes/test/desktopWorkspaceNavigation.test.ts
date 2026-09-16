// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {activateWorkspaceView, type WorkspaceView} from '../src/desktop/main/workspaceNavigation.js'
const view=(workspace:string,sessionId:string,calls:string[],extra:Partial<WorkspaceView>={}):WorkspaceView=>({workspace,sessionId,remote:null,isDestroyed:()=>false,isMinimized:()=>false,restore:()=>calls.push('restore'),show:()=>calls.push('show'),focus:()=>calls.push('focus'),...extra})
test('returning to an open workspace restores and focuses its existing view',()=>{
 const calls:string[]=[];const target=view('/alpha','running',calls,{isMinimized:()=>true})
 expect(activateWorkspaceView([target],'/alpha')).toBe(true)
 expect(calls).toEqual(['restore','show','focus'])
 expect(target.sessionId).toBe('running')
})
test('exact sessions, remote identity and closed windows are not interchangeable',()=>{
 const calls:string[]=[]
 const views=[view('/alpha','one',calls),view('/alpha','two',calls,{remote:{target:'server'}}),view('/beta','three',calls,{isDestroyed:()=>true})]
 expect(activateWorkspaceView(views,'/alpha','two')).toBe(false)
 expect(activateWorkspaceView(views,'/beta')).toBe(false)
 expect(activateWorkspaceView(views,'/missing')).toBe(false)
 expect(calls).toEqual([])
 expect(activateWorkspaceView(views,'/alpha','one')).toBe(true)
 expect(calls).toEqual(['show','focus'])
})
