// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { EMPTY_PULSE } from '../lib/repoPulse.js'

import { chipKey, startWithChips } from './startWith.js'

const base = { agentsNeedingInput: 0, agentsWorking: 0, hasModel: true, pulse: EMPTY_PULSE }

describe('startWithChips', () => {
  it('always offers one way in, even with nothing else true to say', () => {
    expect(startWithChips(base).map(chip => chip.id)).toEqual(['map'])
  })

  it('never shows a chip that has nothing true to say', () => {
    const ids = startWithChips(base).map(chip => chip.id)

    expect(ids).not.toContain('diff')
    expect(ids).not.toContain('agents')
    expect(ids).not.toContain('provider')
  })

  it('carries each chip’s consequence, not just its name', () => {
    const chips = startWithChips({
      ...base,
      agentsNeedingInput: 2,
      agentsWorking: 4,
      pulse: { ...EMPTY_PULSE, additions: 418, changedFiles: 4, deletions: 96, dirty: 4 }
    })

    expect(chips.find(chip => chip.id === 'agents')?.consequence).toBe('4 working · 2 need you')
    expect(chips.find(chip => chip.id === 'diff')?.consequence).toBe('+418 −96 · 4 files')
  })

  it('orders by what you have to do: setup, then the fleet, then the tree', () => {
    const chips = startWithChips({
      agentsNeedingInput: 1,
      agentsWorking: 0,
      hasModel: false,
      pulse: { ...EMPTY_PULSE, additions: 1, changedFiles: 1, dirty: 1 }
    })

    expect(chips.map(chip => chip.id)).toEqual(['provider', 'agents', 'diff', 'map'])
  })

  it('wears amber only when a human is actually required', () => {
    expect(startWithChips({ ...base, hasModel: false })[0]!.tone).toBe('needsInput')
    expect(startWithChips({ ...base, agentsWorking: 3 })[0]!.tone).toBe('working')
    expect(startWithChips({ ...base, agentsNeedingInput: 1 })[0]!.tone).toBe('needsInput')
  })

  it('offers a read-through of unpushed work only when the tree is otherwise clean', () => {
    const clean = startWithChips({ ...base, pulse: { ...EMPTY_PULSE, ahead: 3 } })
    const dirty = startWithChips({
      ...base,
      pulse: { ...EMPTY_PULSE, ahead: 3, additions: 1, changedFiles: 1, dirty: 1 }
    })

    expect(clean.map(chip => chip.id)).toContain('unpushed')
    expect(dirty.map(chip => chip.id)).not.toContain('unpushed')
  })
})

describe('chipKey', () => {
  it('numbers the first nine chips and leaves the rest to the mouse', () => {
    expect(chipKey(0)).toBe('1')
    expect(chipKey(8)).toBe('9')
    expect(chipKey(9)).toBe('')
  })
})
