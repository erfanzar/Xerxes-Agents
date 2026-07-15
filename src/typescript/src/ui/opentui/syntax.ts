// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Theme-aware native SyntaxStyle for <markdown>/<code>. Styles are created
// lazily after the renderer has started, then shared by palette signature so
// a long transcript does not allocate one native style handle per message.
import { RGBA, SyntaxStyle } from '@opentui/core'
import type { ColorInput } from '@opentui/core'

import type { Theme } from '../theme.js'

const cache = new Map<string, SyntaxStyle>()

function nativeColor(color: string): ColorInput {
  const rgb = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i.exec(color)
  if (rgb) {
    return RGBA.fromInts(Number(rgb[1]), Number(rgb[2]), Number(rgb[3]))
  }

  const indexed = /^ansi256\((\d+)\)$/i.exec(color)
  if (indexed) {
    return RGBA.fromIndex(Number(indexed[1]))
  }

  return color
}

function paletteKey(t: Theme): string {
  const c = t.color

  return [
    c.text,
    c.primary,
    c.accent,
    c.muted,
    c.system,
    c.toolName,
    c.completionBg,
    c.border,
    c.ok,
    c.warn,
    c.diffAddedWord,
    c.diffRemovedWord
  ].join('|')
}

export function getSyntaxStyle(t: Theme): SyntaxStyle {
  const key = paletteKey(t)
  const existing = cache.get(key)
  if (existing) {
    return existing
  }

  const c = t.color
  const style = SyntaxStyle.fromStyles({
    default: { fg: nativeColor(c.text) },
    'markup.heading': { bold: true, fg: nativeColor(c.accent) },
    'markup.heading.1': { bold: true, fg: nativeColor(c.accent) },
    'markup.heading.2': { bold: true, fg: nativeColor(c.primary) },
    'markup.heading.3': { bold: true, fg: nativeColor(c.primary) },
    'markup.bold': { bold: true, fg: nativeColor(c.primary) },
    'markup.italic': { fg: nativeColor(c.system), italic: true },
    'markup.raw': { fg: nativeColor(c.warn) },
    'markup.link': { fg: nativeColor(c.system), underline: true },
    'markup.link.label': { fg: nativeColor(c.primary) },
    'markup.link.url': { fg: nativeColor(c.system), underline: true },
    'markup.list': { fg: nativeColor(c.accent) },
    'markup.quote': { dim: true, fg: nativeColor(c.muted), italic: true },
    'markup.separator': { fg: nativeColor(c.border) },
    code: { bg: nativeColor(c.completionBg), fg: nativeColor(c.text) },
    comment: { dim: true, fg: nativeColor(c.muted), italic: true },
    string: { fg: nativeColor(c.ok) },
    constant: { fg: nativeColor(c.warn) },
    number: { fg: nativeColor(c.warn) },
    keyword: { fg: nativeColor(c.system) },
    function: { fg: nativeColor(c.accent) },
    constructor: { fg: nativeColor(c.accent) },
    type: { fg: nativeColor(c.toolName) },
    property: { fg: nativeColor(c.primary) },
    operator: { fg: nativeColor(c.primary) },
    punctuation: { fg: nativeColor(c.muted) },
    variable: { fg: nativeColor(c.text) },
    inserted: { fg: nativeColor(c.diffAddedWord) },
    deleted: { fg: nativeColor(c.diffRemovedWord) }
  })

  cache.set(key, style)
  return style
}
