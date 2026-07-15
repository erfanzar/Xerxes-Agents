// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { slashExecText } from './slashOutput.js'

describe('slashExecText', () => {
  it('renders nothing for empty successful slash responses', () => {
    expect(slashExecText({})).toBeNull()
    expect(slashExecText({ output: '' })).toBeNull()
  })

  it('keeps real output', () => {
    expect(slashExecText({ output: 'done' })).toBe('done')
  })

  it('keeps warnings even without output', () => {
    expect(slashExecText({ warning: 'careful' })).toBe('warning: careful')
  })

  it('combines warnings with output', () => {
    expect(slashExecText({ output: 'done', warning: 'careful' })).toBe('warning: careful\ndone')
  })
})
