// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {monitorFormSettings} from '../src/desktop/renderer/Monitors.js'
const form=(source:string)=>{const data=new FormData();for(const [k,v]of Object.entries({source,target:'source',trigger:'output',match:'done',duration:'3600',attempts:'3',timeout:'60',tokens:'1000'}))data.set(k,v);return data}
test('file watches omit terminal and match fields',()=>{expect(monitorFormSettings(form('file'))).toMatchObject({source_kind:'file',file_path:'source',trigger:'change',react:false});expect(monitorFormSettings(form('file'))).not.toHaveProperty('match');expect(monitorFormSettings(form('file'))).not.toHaveProperty('max_total_tokens')})
test('completion watches omit match and automatic reactions require opt-in',()=>{const data=form('terminal');data.set('trigger','completion');data.set('react','on');const result=monitorFormSettings(data);expect(result).not.toHaveProperty('match');expect(result).toMatchObject({terminal_id:'source',trigger:'completion',react:true,max_total_tokens:1000})})
test('web sources map only their matching source field',()=>{expect(monitorFormSettings(form('webhook'))).toMatchObject({source_kind:'webhook',webhook_name:'source',match:'done'});expect(monitorFormSettings(form('websocket'))).toMatchObject({source_kind:'websocket',websocket_url:'source',match:'done'})})
test('missing target, missing match and invalid duration fail before sending',()=>{for(const [key,value]of [['target',''],['match',''],['duration','0']]){const data=form('terminal');data.set(key!,value!);expect(()=>monitorFormSettings(data)).toThrow()}})
