// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * One syntax highlighter for the whole renderer: chat code blocks and the
 * file editor both use lowlight (highlight.js grammars). The editor needs the
 * result split into lines, with a token that spans lines (a docstring, a block
 * comment, a template string) keeping its class on every line it covers.
 */

import { common, createLowlight } from 'lowlight'

export const lowlight = createLowlight(common)
/** Above this, highlighting costs more than it gives; the code stays plain. */
export const HIGHLIGHT_LIMIT = 100_000

export interface HastNode {
  readonly type: string
  readonly value?: string
  readonly properties?: { readonly className?: unknown }
  readonly children?: readonly HastNode[]
}

export function classOf(value: unknown): string | undefined {
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === 'string').join(' ') || undefined
  return typeof value === 'string' ? value : undefined
}

export interface LineSpan { readonly text: string; readonly className?: string }

/**
 * The code as lines of classed spans, or null when the language is unknown,
 * the file is too large, or the grammar throws (the caller then falls back).
 * The spans of line N concatenate back to exactly line N of the input.
 */
export function highlightLines(code: string, language: string): LineSpan[][] | null {
  const lang = language.toLowerCase()
  if (!lang || code.length > HIGHLIGHT_LIMIT || !lowlight.registered(lang)) return null
  let nodes: readonly HastNode[]
  try { nodes = lowlight.highlight(lang, code).children } catch { return null }
  const lines: LineSpan[][] = [[]]
  const walk = (list: readonly HastNode[], classes: string): void => {
    for (const node of list) {
      if (node.type === 'text') {
        const parts = (node.value ?? '').split('\n')
        parts.forEach((part, index) => {
          if (index > 0) lines.push([])
          if (part) lines[lines.length - 1]!.push(classes ? { text: part, className: classes } : { text: part })
        })
      } else if (node.type === 'element') {
        const own = classOf(node.properties?.className)
        walk(node.children ?? [], own ? (classes ? `${classes} ${own}` : own) : classes)
      }
    }
  }
  walk(nodes, '')
  return lines
}
