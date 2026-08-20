// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { rebasePasteResult } from '../opentui/pasteRebase.js'

describe('asynchronous paste rebasing', () => {
  it('keeps text typed while the clipboard read is pending', () => {
    expect(
      rebasePasteResult(
        { cursor: 0, value: '' },
        { cursor: 6, value: 'pasted' },
        { cursor: 5, value: 'later' }
      )
    ).toEqual({ cursor: 11, value: 'pastedlater' })
  })

  it('rebases an insertion in the middle without dropping the original suffix', () => {
    expect(
      rebasePasteResult(
        { cursor: 2, value: 'abCD' },
        { cursor: 3, value: 'abXCD' },
        { cursor: 3, value: 'abYCD' }
      )
    ).toEqual({ cursor: 4, value: 'abXYCD' })
  })

  it('uses the resolved result directly when the textarea did not change', () => {
    expect(
      rebasePasteResult(
        { cursor: 1, value: 'ab' },
        { cursor: 2, value: 'aXb' },
        { cursor: 1, value: 'ab' }
      )
    ).toEqual({ cursor: 2, value: 'aXb' })
  })
})
