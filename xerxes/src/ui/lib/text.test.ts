// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  buildToolTrailLine,
  compactPreview,
  formatAbandonedClarify,
  formatToolCall,
  inlineToolDisplay,
  parseToolTrailResultLine,
  stripTrailingPasteNewlines,
  toolTrailParts
} from './text.js'
import { copyPreview } from './copyText.js'

describe('compactPreview', () => {
  it('collapses whitespace onto one line', () => {
    expect(compactPreview('  alpha \n beta\tgamma  ', 40)).toBe('alpha beta gamma')
  })

  it('returns empty for blank input', () => {
    expect(compactPreview('   \n\t ', 10)).toBe('')
  })

  it('leaves text that already fits untouched', () => {
    expect(compactPreview('alpha', 5)).toBe('alpha')
  })

  // Regression: clamping used to land mid-gap and render as "alpha …".
  it('strips the trailing space before the ellipsis', () => {
    expect(compactPreview('alpha beta', 7)).toBe('alpha…')
  })

  // Regression: a very narrow column sliced to the empty string, so a picker
  // row rendered as a bare ellipsis with no content at all.
  it('keeps at least one character in a very narrow column', () => {
    expect(compactPreview('alpha', 1)).toBe('a…')
    expect(compactPreview('alpha', 0)).toBe('a…')
  })

  it('is the single implementation behind copyPreview', () => {
    expect(copyPreview).toBe(compactPreview)
  })
})

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

  it('keeps a bounded Spawn Agents roster intact for live cube parsing', () => {
    const names = Array.from({ length: 8 }, (_, index) => `specialist-${index}-analyzer`)
    const line = formatToolCall('SpawnAgents', `8 agents: ${names.join(', ')}`)

    expect(line).toContain(names.at(-1))
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

describe('toolTrailParts', () => {
  it('separates the tool name, its arguments, and how long it took', () => {
    const line = buildToolTrailLine('read_file', 'src/one.ts', false, '', 0.2)
    const parsed = parseToolTrailResultLine(line)!

    expect(toolTrailParts(parsed.call)).toEqual({
      args: 'src/one.ts',
      duration: '0.2s',
      name: 'Read File'
    })
  })

  it('handles a call with no context and no duration', () => {
    expect(toolTrailParts('Task Output Tool')).toEqual({ args: '', duration: '', name: 'Task Output Tool' })
  })

  it('keeps arguments that themselves contain parentheses', () => {
    const line = buildToolTrailLine('exec', 'bun test (unit)', false, '', 1.5)
    const parsed = parseToolTrailResultLine(line)!
    const parts = toolTrailParts(parsed.call)

    expect(parts.name).toBe('Exec')
    expect(parts.args).toBe('bun test (unit)')
    expect(parts.duration).toBe('1.5s')
  })
})

it('keeps the reason visible on a failed tool row', () => {
  const line = buildToolTrailLine(
    'ReadFile',
    'xerxes/src/daemon/server.ts',
    true,
    'Tool execution failed: Function ReadFile: Validation error for file_path: must refer to an existing regular file',
    0.1,
  )
  // The framing the row already shows is dropped, so the budget is spent on
  // the reason instead of being consumed before it starts.
  expect(line).toContain('must refer to an existing regular file')
  expect(line).not.toContain('Tool execution failed')
  expect(line).not.toContain('Function ReadFile:')
  expect(line.endsWith('✗')).toBe(true)
})
