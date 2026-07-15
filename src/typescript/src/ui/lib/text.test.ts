// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  buildToolTrailLine,
  formatAbandonedClarify,
  formatToolCall,
  inlineToolDisplay,
  parseToolTrailResultLine,
  stripTrailingPasteNewlines
} from './text.js'

describe('stripTrailingPasteNewlines', () => {
  it('removes trailing newline runs from pasted text', () => {
    expect(stripTrailingPasteNewlines('alpha\n')).toBe('alpha')
    expect(stripTrailingPasteNewlines('alpha\nbeta\n\n')).toBe('alpha\nbeta')
  })

  it('preserves interior newlines', () => {
    expect(stripTrailingPasteNewlines('alpha\nbeta\ngamma')).toBe('alpha\nbeta\ngamma')
  })

  it('preserves newline-only pastes', () => {
    expect(stripTrailingPasteNewlines('\n\n')).toBe('\n\n')
  })
})

describe('formatAbandonedClarify', () => {
  it('renders the question, numbered options, and reason', () => {
    const out = formatAbandonedClarify('How do you want to scope?', ['Option A', 'Option B', 'Option C'], 'timed out')

    expect(out).toBe(
      [
        'ask How do you want to scope?',
        '  1. Option A',
        '  2. Option B',
        '  3. Option C',
        '  (timed out — no selection)'
      ].join('\n')
    )
  })

  it('handles a prompt with no choices (free-text clarify)', () => {
    const out = formatAbandonedClarify('What is the target branch?', null, 'cancelled')

    expect(out).toBe(['ask What is the target branch?', '  (cancelled — no selection)'].join('\n'))
  })

  it('trims surrounding whitespace on the question', () => {
    const out = formatAbandonedClarify('  trailing space  ', [], 'timed out')

    expect(out.split('\n')[0]).toBe('ask trailing space')
  })

  it('numbers options 1-based to match the live ClarifyPrompt', () => {
    const out = formatAbandonedClarify('q', ['first'], 'timed out')

    expect(out).toContain('  1. first')
    expect(out).not.toContain('  0.')
  })
})

describe('inlineToolDisplay', () => {
  it('renders persisted calls as compact Grok-style rows', () => {
    expect(inlineToolDisplay('Exec Command("ls -la") (0.1s)')).toBe('Exec Command ls -la')
    expect(inlineToolDisplay('Read File("src/app.ts") (0.2s)')).toBe('Read File src/app.ts')
    expect(inlineToolDisplay('Tool (12.4s)')).toBe('Tool')
    expect(inlineToolDisplay(formatToolCall('ReadFile', 'package.json'))).toBe('Read File package.json')
  })
})

describe('parseToolTrailResultLine', () => {
  it('does not mistake a separator-shaped tool argument for result detail', () => {
    const line = buildToolTrailLine('exec_command', 'printf "left :: right"', true, 'command failed', 0.2)

    expect(parseToolTrailResultLine(line)).toEqual({
      call: 'Exec Command("printf "left :: right"") (0.2s)',
      detail: 'command failed',
      mark: '✗'
    })
  })
})
