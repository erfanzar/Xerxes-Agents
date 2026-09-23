// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { contextSections, parseContextPage } from '../src/desktop/renderer/ContextInspector.js'
const page = () => ({ ok:true, generation:'gen-1', controls_revision:1, note:'Local estimate', section:'memory', offset:0, next_offset:20, sections:contextSections.map(id=>({id,count:21,available:true,estimated_tokens:200,provenance:'Saved session'})), entries:[{index:0,title:'Project notes',text:'Retained source text',estimated_tokens:10,truncated:false,control:{scope:'project',path:'notes.md',pinned:false,excluded:false}}] })
test('context inspector preserves paging generation, source identity and revision',()=>{const result=parseContextPage(page(),'memory',0);expect(result.generation).toBe('gen-1');expect(result.next_offset).toBe(20);expect(result.controls_revision).toBe(1);expect(result.entries[0]?.control?.path).toBe('notes.md')})
test('rejects stale section and offset responses',()=>{expect(()=>parseContextPage(page(),'tools',0)).toThrow();expect(()=>parseContextPage(page(),'memory',20)).toThrow()})
test('rejects malformed controls and non advancing pagination',()=>{for(const update of [{controls_revision:-1},{next_offset:0},{sections:[]},{entries:[{...page().entries[0],control:{scope:'arbitrary'}}]}])expect(()=>parseContextPage({...page(),...update},'memory',0)).toThrow()})

import { contextTitle, reflowContextText } from '../src/desktop/renderer/ContextInspector.js'

test('identifier titles read as words; paths and prose stay as written', () => {
  expect(contextTitle('goal_policy')).toBe('Goal policy')
  expect(contextTitle('mode_hint')).toBe('Mode hint')
  expect(contextTitle('~/.xerxes/memory/user.md')).toBe('~/.xerxes/memory/user.md')
  expect(contextTitle('Project notes')).toBe('Project notes')
})

test('hard-wrapped prompt text reflows into paragraphs without flattening structure', () => {
  const wrapped = 'Mark complete only when the objective\nis actually achieved and THIS turn ran\nthe check.\n\nNo goal is set.'
  expect(reflowContextText(wrapped)).toBe('Mark complete only when the objective is actually achieved and THIS turn ran the check.\n\nNo goal is set.')
  const list = 'Rules:\n- one\n- two\n1. first\n  indented code'
  expect(reflowContextText(list)).toBe('Rules:\n- one\n- two\n1. first\n  indented code')
  const fenced = 'Run:\n```\nbun test\nbun run check\n```'
  expect(reflowContextText(fenced)).toBe(fenced)
})
