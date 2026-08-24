// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The transcript voice table — the single place that answers "who is
// speaking" in colour.
//
// The boot emblem animates a lapis → violet → gold gradient
// (`derafshGradientPalette`). That gradient is the legend for this table:
//
//   accent → the model, opening each of its turns
//   gold   → the human
//   lapis  → tools
//   violet → the system, and its own quieter thinking
//
// Two rules keep this from becoming noise:
//
//  1. Exactly one hued element per row. A voice earns either a bar or a
//     glyph, never both, and body prose stays neutral so it never fights the
//     markdown syntax colours (which already claim `accent` and `primary`).
//  2. The assistant's marker is small and singular: a ✦ that OPENS a turn
//     (redesign spec, transcript anatomy element ②) and nothing else. The
//     earlier "no marker at all" rule is superseded by the v3 redesign; the
//     model's body prose still carries no chrome, so the ✦ reads as a turn
//     boundary, not a per-line flag. The dim turn rail continues to span
//     multi-paragraph answers beside it.
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
  // No bar; a small accent ✦ opens each turn while the prose stays neutral.
  // The glyph renders once, at the head of the turn body — see TurnGlyph in
  // opentui/messageLine.tsx, which is the only consumer.
  // Prose, not titles. The canvas gives the model's answer the ramp's `prose`
  // step and keeps `title` for the user's own words and for row headings, so
  // scrolling fast the human's sentences sit a shade brighter than the
  // machine's — the same job the filled user band does, one step quieter.
  assistant: t => ({ bar: '', body: t.ds.prose, glyph: '✦', glyphColor: t.color.accent }),
  // Rare enough that a full-line hue is affordable and instantly identifiable.
  system: t => ({ bar: '', body: t.color.system, glyph: '·', glyphColor: t.color.system }),
  // ⏺ outcome glyph, tinted per row by the renderer (faint for quiet
  // read-only calls, ok-green on success, error-red on failure — anatomy
  // element ④); the NAME stays this lapis. The disc shape rather than '→'
  // keeps the outcome legible before the eye reaches the words. Colour
  // carries this voice, not glyph shape.
  // The disc is a STATE mark and wears the working blue; the verb beside it
  // wears `toolName`, the ramp step reserved for verbs. Those were the same
  // colour before the design system assigned the ramp by role, which is why
  // this now says `accent` explicitly rather than borrowing the verb's step.
  tool: t => ({ bar: '', body: t.color.muted, glyph: '⏺', glyphColor: t.color.accent }),
  // The transcript's strongest anchor: a blue rule beside a filled band.
  user: t => ({ bar: t.color.userBar, body: t.color.userText, glyph: '', glyphColor: '' })
}
