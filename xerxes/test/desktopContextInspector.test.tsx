// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { contextSections, parseContextPage } from '../src/desktop/renderer/ContextInspector.js'
const page = () => ({ ok:true, generation:'gen-1', controls_revision:1, note:'Local estimate', section:'memory', offset:0, next_offset:20, sections:contextSections.map(id=>({id,count:21,available:true,estimated_tokens:200,provenance:'Saved session'})), entries:[{index:0,title:'Project notes',text:'Retained source text',estimated_tokens:10,truncated:false,control:{scope:'project',path:'notes.md',pinned:false,excluded:false}}] })
test('context inspector preserves paging generation, source identity and revision',()=>{const result=parseContextPage(page(),'memory',0);expect(result.generation).toBe('gen-1');expect(result.next_offset).toBe(20);expect(result.controls_revision).toBe(1);expect(result.entries[0]?.control?.path).toBe('notes.md')})
test('rejects stale section and offset responses',()=>{expect(()=>parseContextPage(page(),'tools',0)).toThrow();expect(()=>parseContextPage(page(),'memory',20)).toThrow()})
test('rejects malformed controls and non advancing pagination',()=>{for(const update of [{controls_revision:-1},{next_offset:0},{sections:[]},{entries:[{...page().entries[0],control:{scope:'arbitrary'}}]}])expect(()=>parseContextPage({...page(),...update},'memory',0)).toThrow()})
