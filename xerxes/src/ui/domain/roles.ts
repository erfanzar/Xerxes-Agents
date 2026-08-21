// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The transcript voice table — the single place that answers "who is
// speaking" in colour.
//
// The boot emblem animates a lapis → violet → gold gradient
// (`derafshGradientPalette`). That gradient is the legend for this table:
//
//   gold   → the human
//   lapis  → tools
//   violet → the system, and its own quieter thinking
//   none   → the model
//
// Two rules keep this from becoming noise:
//
//  1. Exactly one hued element per row. A voice earns either a bar or a
//     glyph, never both, and body prose stays neutral so it never fights the
//     markdown syntax colours (which already claim `accent` and `primary`).
//  2. The assistant has no marker at all. The model's voice is the absence
//     of chrome — that is precisely what gives the other markers meaning.
import type { Theme } from '../theme.js'
import type { Role } from '../types.js'

export interface Voice {
  /** Left rule colour for a banded voice; '' for voices that carry no bar. */
  bar: string
  /** Body text colour. */
  body: string
  /** Leading glyph; '' for voices that carry none. */
  glyph: string
  /** Glyph colour — the hued element for glyph-marked voices. */
  glyphColor: string
}

export const VOICE: Record<Role, (t: Theme) => Voice> = {
  // No bar, no glyph, neutral body. Deliberately unmarked.
  assistant: t => ({ bar: '', body: t.color.text, glyph: '', glyphColor: '' }),
  // Rare enough that a full-line hue is affordable and instantly identifiable.
  system: t => ({ bar: '', body: t.color.system, glyph: '·', glyphColor: t.color.system }),
  // Lapis glyph and name; arguments and timing stay muted. The glyph stays
  // '→' rather than an angle bracket: U+27E9 and friends are East-Asian
  // Ambiguous width and thinly covered by terminal fonts, so they risk
  // desyncing column math. Colour carries this voice, not glyph shape.
  tool: t => ({ bar: '', body: t.color.muted, glyph: '→', glyphColor: t.color.toolName }),
  // The transcript's strongest anchor: a gold rule beside a filled band.
  user: t => ({ bar: t.color.userBar, body: t.color.userText, glyph: '', glyphColor: '' })
}
