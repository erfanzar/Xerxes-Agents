// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { StructuredResult, structuredOutput } from '../src/desktop/renderer/StructuredResult.js'

test('goal results promote objective, milestone and criteria without inventing completion',()=>{
  const html=renderToStaticMarkup(createElement(StructuredResult,{value:{goal:{objective:'Preserve runtime behavior',phase:'paused',currentMilestone:'Verify cancellation',criteria:[{id:'security',description:'Keep certificate verification'}],revision:3},error:'Review required'}}))
  for(const text of ['Preserve runtime behavior','paused','Current milestone','Success criteria','Keep certificate verification','Goal metadata','Review required'])expect(html).toContain(text)
  expect(html).not.toContain('Completed')
})
test('unknown structured output stays navigable and text is escaped',()=>{
  const html=renderToStaticMarkup(createElement(StructuredResult,{value:{ok:false,items:[{message:'<script>bad</script>'}],missing:null}}))
  expect(html).toContain('false');expect(html).toContain('1 items');expect(html).toContain('&lt;script&gt;');expect(html).not.toContain('<script>')
  expect(structuredOutput('plain output')).toBeNull();expect(structuredOutput('{broken')).toBeNull();expect(structuredOutput('[]')).toEqual([])
})
