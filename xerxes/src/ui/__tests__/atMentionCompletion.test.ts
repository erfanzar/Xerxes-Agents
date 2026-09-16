// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { completionToApplyOnSubmit } from '../domain/slash.js'
import { completionRequestForInput } from '../hooks/useCompletion.js'

describe('OpenTUI @ file completion', () => {
  it('waits for a query character and requests the active mention token', () => {
    expect(completionRequestForInput('remove @')).toBeNull()
    expect(completionRequestForInput('remove @tmp-files')).toEqual({
      method: 'complete.path',
      params: { word: '@tmp-files' },
      replaceFrom: 7
    })
  })

  it('accepts a selected mention path without losing the surrounding prompt', () => {
    expect(completionToApplyOnSubmit('remove @tmp', '@tmp-files/', 7)).toBe('remove @tmp-files/')
    expect(completionToApplyOnSubmit('inspect @src/op', '@src/opentui/', 8)).toBe('inspect @src/opentui/')
  })

  it('falls through to submission when the selected mention already matches the draft', () => {
    expect(completionToApplyOnSubmit('remove @tmp-files/', '@tmp-files/', 7)).toBeNull()
  })

  it('completes path arguments without replacing the slash command name', () => {
    expect(completionRequestForInput('/image ./screens/shot')).toEqual({
      method: 'complete.path',
      params: { word: './screens/shot' },
      replaceFrom: 7
    })
    expect(completionToApplyOnSubmit('/image ./screens/shot', './screens/shot.png', 7)).toBe(
      '/image ./screens/shot.png'
    )
    expect(completionRequestForInput('/model ./not-a-path')).toBeNull()
  })
})

describe('skill argument completion requests', () => {
  it('requests full daemon command arguments without losing the leading command', () => {
    for (const input of ['/forge inspect gre', '/preset manage', '/config m', '/plugins enable test', '/skills inspect git']) {
      expect(completionRequestForInput(input)).toEqual({ method: 'complete.slash', params: { text: input }, replaceFrom: 1 })
    }
    expect(completionToApplyOnSubmit('/forge inspect gre', '/forge inspect greeting 1.0.0 ', 1)).toBe('/forge inspect greeting 1.0.0 ')
    expect(completionRequestForInput('/file evidence')).toEqual({ method: 'complete.path', params: { word: 'evidence', path_prefix: 'evidence' }, replaceFrom: 6 })
    expect(completionRequestForInput('/plugins install ./source')).toEqual({ method: 'complete.path', params: { word: './source' }, replaceFrom: 17 })
  })
  it('routes /skill <prefix> to daemon skill suggestions with the right replace point', () => {
    expect(completionRequestForInput('/skill git')).toEqual({
      method: 'skill_suggestions',
      params: { prefix: 'git' },
      replaceFrom: 7
    })
    expect(completionRequestForInput('/skill ')).toEqual({
      method: 'skill_suggestions',
      params: { prefix: '' },
      replaceFrom: 7
    })
  })

  it('leaves the bare /skill name and other commands on the catalog path', () => {
    expect(completionRequestForInput('/skill')).toEqual({
      method: 'complete.slash',
      params: { text: '/skill' },
      replaceFrom: 1
    })
    expect(completionRequestForInput('/skills list')).toEqual({ method: 'complete.slash', params: { text: '/skills list' }, replaceFrom: 1 })
  })
})
