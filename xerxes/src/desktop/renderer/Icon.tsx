// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

const paths = {
  sidebar: 'M3 3h14v14H3z M8 3v14',
  expand: 'M7 3H3v4 M13 3h4v4 M17 13v4h-4 M7 17H3v-4',
  collapse: 'M3 7h4V3 M13 3v4h4 M17 13h-4v4 M7 17v-4H3',
  search: 'M14 14l4 4 M15 9a6 6 0 1 1-12 0 6 6 0 0 1 12 0',
  plus: 'M10 4v12 M4 10h12',
  close: 'M5 5l10 10 M15 5L5 15',
  tools: 'M4 4h5v5H4z M12 4h4v5h-4z M4 12h5v4H4z M12 12h4v4h-4z',
  folder: 'M2 5h6l2 2h8v10H2z',
  file: 'M5 2h6l4 4v12H5z M11 2v5h4',
  chevron: 'M8 5l5 5-5 5',
  terminal: 'M2 4h16v12H2z M5 7l3 3-3 3 M10 13h4',
  shield: 'M10 2l7 3v5c0 4-4 7-7 8-3-1-7-4-7-8V5z M7 10l2 2 4-4',
  plan: 'M7 5h10 M7 10h10 M7 15h10 M3 5h.01 M3 10h.01 M3 15h.01',
  clock: 'M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0 M10 5v5l3 2',
  settings: 'M10 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M10 2v2 M10 16v2 M2 10h2 M16 10h2 M4 4l2 2 M14 14l2 2 M4 16l2-2 M14 6l2-2',
  activity: 'M2 11h4l2-7 4 13 2-6h4',
  changes: 'M5 3v10a3 3 0 0 0 3 3h5 M5 3l-2 2 M5 3l2 2 M15 17V7a3 3 0 0 0-3-3h-2',
  agent: 'M4 6h12v10H4z M10 3v3 M7 10h.01 M13 10h.01 M8 13h4',
  arrow: 'M4 10h12 M11 5l5 5-5 5',
} as const

export function Icon({ name, size = 18 }: { name: keyof typeof paths; size?: number }): ReactElement {
  return <svg className="app-icon" width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>
}
