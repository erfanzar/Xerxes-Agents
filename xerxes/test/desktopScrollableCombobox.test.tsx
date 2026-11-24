// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  ScrollableCombobox,
  computeComboboxPlacement,
  filterComboboxOptions,
} from '../src/desktop/renderer/ScrollableCombobox.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')

describe('ScrollableCombobox', () => {
  test('filterComboboxOptions deduplicates, keeps full list when browsing, and ranks prefix matches first', () => {
    const hosts = [
      'caltech-large',
      'eserver-v4-8-1',
      'caltech-large',
      'prism-v5p-1024_1',
      'RTX-3090',
      'RTX-3090-Local',
      'softnu-local',
    ]

    expect(filterComboboxOptions(hosts, 'caltech-large', false)).toEqual([
      'caltech-large',
      'eserver-v4-8-1',
      'prism-v5p-1024_1',
      'RTX-3090',
      'RTX-3090-Local',
      'softnu-local',
    ])

    expect(filterComboboxOptions(hosts, 'local', true)).toEqual([
      'RTX-3090-Local',
      'softnu-local',
    ])

    expect(filterComboboxOptions(hosts, 'rtx', true)).toEqual([
      'RTX-3090',
      'RTX-3090-Local',
    ])

    expect(filterComboboxOptions(['my-prism', 'prism-1', 'other'], 'prism', true)).toEqual([
      'prism-1',
      'my-prism',
    ])

    expect(filterComboboxOptions(hosts, 'no-such-host', true)).toEqual([])
  })

  test('computeComboboxPlacement opens below when space fits and flips above with clamped height near window bottom', () => {
    const below = computeComboboxPlacement(
      { left: 140, top: 200, bottom: 242, width: 460 },
      { width: 1200, height: 800 },
      264,
    )
    expect(below.placement).toBe('below')
    if (below.placement === 'below') {
      expect(below.top).toBe(248)
      expect(below.maxHeight).toBe(264)
      expect(below.width).toBe(460)
      expect(below.left).toBe(140)
    }

    // Reproduces the SSH host field near the bottom of the Workspace modal:
    // flips above using a `bottom` anchor and bounds `maxHeight` so the list
    // stays inside the window and scrolls instead of clipping off the top.
    const above = computeComboboxPlacement(
      { left: 80, top: 640, bottom: 682, width: 460 },
      { width: 620, height: 740 },
      264,
    )
    expect(above.placement).toBe('above')
    if (above.placement === 'above') {
      expect(above.bottom).toBe(106)
      expect(above.maxHeight).toBe(264)
      // Top edge of popup = viewport.height - bottom - maxHeight >= 12px margin
      expect(740 - above.bottom - above.maxHeight).toBeGreaterThanOrEqual(12)
    }

    // Short viewport clamps maxHeight to available space above without overflowing.
    const tightAbove = computeComboboxPlacement(
      { left: 40, top: 180, bottom: 222, width: 460 },
      { width: 520, height: 260 },
      264,
    )
    expect(tightAbove.placement).toBe('above')
    expect(tightAbove.maxHeight).toBe(162)
    expect(tightAbove.width).toBe(460)
    expect(tightAbove.left).toBe(40)
  })

  test('renders accessible combobox markup with toggle chevron when options are provided', () => {
    const withOptions = renderToStaticMarkup(
      createElement(ScrollableCombobox, {
        id: 'ws-ssh-host-input',
        variant: 'workspace',
        value: '',
        options: ['caltech-large', 'GPU-H100'],
        onChange: () => {},
        placeholder: 'SSH alias, user@host or user@host:port',
        listLabel: 'SSH config hosts',
        toggleLabel: 'Show SSH config hosts',
      }),
    )
    expect(withOptions).toContain('role="combobox"')
    expect(withOptions).toContain('aria-autocomplete="list"')
    expect(withOptions).toContain('aria-label="Show SSH config hosts"')
    expect(withOptions).toContain('combo-field__toggle')
    expect(withOptions).not.toContain('<datalist')

    const empty = renderToStaticMarkup(
      createElement(ScrollableCombobox, {
        value: '',
        options: [],
        onChange: () => {},
        listLabel: 'SSH config hosts',
      }),
    )
    expect(empty).toContain('role="combobox"')
    expect(empty).not.toContain('combo-field__toggle')
  })

  test('replaces native non-scrolling <datalist> popups in WorkspacePanel and AgentIntelligenceCard with scrollable listbox CSS', () => {
    const panelsSource = readFileSync(join(rendererDir, 'DesktopPanels.tsx'), 'utf8')
    const intelligenceSource = readFileSync(join(rendererDir, 'AgentIntelligenceCard.tsx'), 'utf8')
    const atelierCss = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')

    expect(panelsSource).not.toContain('<datalist')
    expect(panelsSource).toContain('<ScrollableCombobox')
    expect(panelsSource).toContain('listLabel="SSH config hosts"')

    expect(intelligenceSource).not.toContain('<datalist')
    expect(intelligenceSource).toContain('<ScrollableCombobox')

    expect(atelierCss).toContain('.picker-layer>.combo-pop{pointer-events:auto;z-index:2}')
    expect(atelierCss).toContain('.combo-pop__list{flex:1 1 auto;min-height:0;overflow-y:auto;overscroll-behavior:contain')
  })
})
