// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { trustAllSkills } from '../src/desktop/renderer/catalogBulk.js'

test('trust all uses the fixed skill snapshot sequentially and continues after failures', async () => {
  const calls: string[] = []
  const result = await trustAllSkills(['alpha', 'beta', 'gamma'], async name => {
    calls.push(name)
    if (name === 'beta') throw new Error('scan refused')
    return { ok: true }
  })
  expect(calls).toEqual(['alpha', 'beta', 'gamma'])
  expect(result.trusted).toEqual(['alpha', 'gamma'])
  expect(result.failed).toEqual([{ name: 'beta', error: 'scan refused' }])
})

test('trust all reports daemon refusals and an empty catalog does nothing', async () => {
  const refused = await trustAllSkills(['unsafe'], async () => ({ ok: false, error: 'Instructions changed during review' }))
  expect(refused.trusted).toEqual([])
  expect(refused.failed).toEqual([{ name: 'unsafe', error: 'Instructions changed during review' }])
  expect(await trustAllSkills([], async () => { throw new Error('must not be called') })).toEqual({ trusted: [], failed: [] })
})

test('catalog actions have a separate compact toolbar and viewport confirmation', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/DesktopPanels.tsx', import.meta.url)).text()
  expect(source.includes('aria-label="Catalog views"')).toBe(true)
  expect(source.includes('className="catalog-actions"')).toBe(true)
  expect(source.includes('title={`Trust ${trustReview.length} skills?`}')).toBe(true)
})
