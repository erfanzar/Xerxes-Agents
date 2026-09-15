// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {createElement} from 'react'
import {renderToStaticMarkup} from 'react-dom/server'
import {DiffPreview} from '../src/desktop/renderer/DiffPreview.js'

test('review diff exposes changed lines, real old/new numbers and keyboard scrolling',()=>{
  const html=renderToStaticMarkup(createElement(DiffPreview,{label:'Snapshot restore diff',diff:'diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts\n@@ -41,2 +41,2 @@\n context\n-before\n+after\n'}))
  expect(html).toContain('aria-label="Snapshot restore diff"')
  expect(html).toContain('tabindex="0"')
  expect(html).toContain('aria-hidden="true">42</span>')
  expect(html).toContain('studio-diff-add')
  expect(html).toContain('studio-diff-del')
  expect(html).toContain('+after')
  expect(html).toContain('-before')
})
