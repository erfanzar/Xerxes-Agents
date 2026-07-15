// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export const PASTE_SNIPPET_RE = /\[\[[^\n]*?\]\]/g

/** Normalized paste payload passed from the OpenTUI composer to the controller. */
export interface PasteEvent {
  bracketed?: boolean
  cursor: number
  hotkey?: boolean
  text: string
  value: string
}
