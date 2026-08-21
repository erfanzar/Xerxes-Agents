// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Type bridge for @opentui/core 0.4.3: `MarkdownRenderable` inherits the
// runtime `selectable` property from `Renderable` (see Renderable.d.ts), but
// `MarkdownOptions` — the type the React `<markdown>` JSX props derive from —
// omits it, so the prop fails type-check even though the reconciler applies
// it through the inherited setter at runtime. This augmentation restores the
// prop on the options type; drop the file once OpenTUI adds `selectable` to
// `MarkdownOptions` upstream.
import '@opentui/core'

declare module '@opentui/core' {
  interface MarkdownOptions {
    /** Inherited from Renderable at runtime; missing from MarkdownOptions in @opentui/core 0.4.3. */
    selectable?: boolean
  }
}

export {}
