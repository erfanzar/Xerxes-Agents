// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Generates Xerxes Agents' standalone modern developer-tool Brand SVG &
 * GIF89a asset pack into `assets/brand/`.
 *
 * Pure Bun TypeScript with zero native canvas or Python dependencies:
 * - SVGs use clean geometric paths and system font stacks.
 * - GIFs render smooth anti-aliased vector geometry directly into
 *   `Uint8ClampedArray` RGBA buffers and encode via `encodeAnimatedGif`
 *   (`xerxes/src/desktop/renderer/gifEncoder.ts`).
 *
 * Usage:
 *   bun xerxes/scripts/generateBrandAssets.ts
 */

import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'

import { encodeAnimatedGif, type GifFrame } from '../src/desktop/renderer/gifEncoder.js'
import {
  SIMURGH_ROLES,
  XERXES_PRISM_PRIMARY_PATHS,
  XERXES_PRISM_SECONDARY_PATHS,
  type SimurghSpriteRole,
  type SimurghSpriteState,
} from '../src/desktop/renderer/XerxesSprites.js'

const BRAND_DIR = join(import.meta.dir, '..', '..', 'assets', 'brand')
const SANS_FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
const MONO_FONT = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

// Clean Linear / Vercel / Claude Code dark neutral palette
const UI_BG = '#101010'
const UI_CARD = '#141414'
const UI_CHIP = '#1d1d1d'
const UI_HAIRLINE = '#262626'
const UI_SEPARATOR = '#323232'
const UI_TITLE = '#ededed'
const UI_SECONDARY = '#a3a3a3'
const UI_META = '#737373'
const UI_ACCENT = '#5b8def'
const UI_DONE = '#3fb950'
const UI_NEEDS = '#d7ba7d'
const UI_FAILED = '#f14c4c'

// Canonical 48×48 Faceted Architectural X-Prism polygons (matching XerxesSprites.tsx)
const PRISM_PRIMARY_POLYS: ReadonlyArray<ReadonlyArray<readonly [number, number]>> = [
  [
    [6.5, 8],
    [16.5, 8],
    [25.6, 19.65],
    [20.6, 26.05],
  ],
  [
    [41.5, 40],
    [31.5, 40],
    [22.4, 28.35],
    [27.4, 21.95],
  ],
]

const PRISM_SECONDARY_POLYS: ReadonlyArray<ReadonlyArray<readonly [number, number]>> = [
  [
    [31.5, 8],
    [41.5, 8],
    [30.8, 21.7],
    [25.8, 15.3],
  ],
  [
    [16.5, 40],
    [6.5, 40],
    [17.2, 26.3],
    [22.2, 32.7],
  ],
]

// ---------------------------------------------------------------------------
// Shared Monochrome 24×24 Role Icon SVG Helper
// ---------------------------------------------------------------------------

interface RoleColorSpec {
  readonly stroke: string
  readonly opacity: number
}

function resolveRoleColor(role: SimurghSpriteRole, state: SimurghSpriteState): RoleColorSpec {
  if (state === 'killed' || state === 'stopped') {
    return { stroke: UI_META, opacity: 0.62 }
  }
  if (state === 'error') {
    return { stroke: UI_FAILED, opacity: 1 }
  }
  if (state === 'waiting') {
    return role === 'coordinator'
      ? { stroke: UI_TITLE, opacity: 0.86 }
      : { stroke: UI_META, opacity: 0.92 }
  }
  if (role === 'verifier') {
    return { stroke: UI_DONE, opacity: 1 }
  }
  return { stroke: UI_TITLE, opacity: 1 }
}

function renderRoleIconSvg(
  role: SimurghSpriteRole,
  state: SimurghSpriteState,
  x: number,
  y: number,
  size: number,
): string {
  const scale = size / 24
  const c = resolveRoleColor(role, state)
  let body = ''
  if (role === 'coordinator') {
    body = [
      '<circle cx="12" cy="5.5" r="2.5" />',
      '<circle cx="6" cy="18.5" r="2.5" />',
      '<circle cx="18" cy="18.5" r="2.5" />',
      '<path d="M 12 8 V 12" />',
      '<path d="M 6 16 V 13.5 C 6 12.67 6.67 12 7.5 12 H 16.5 C 17.33 12 18 12.67 18 13.5 V 16" />',
    ].join('')
  } else if (role === 'worker') {
    body = '<path d="M 4.5 7 L 10.5 12 L 4.5 17" /><path d="M 12.5 17 H 19.5" />'
  } else if (role === 'verifier') {
    body = '<circle cx="12" cy="12" r="8.5" /><path d="M 8.5 12.2 L 10.8 14.5 L 15.8 9.5" />'
  } else if (role === 'narrator') {
    body = '<path d="M 4.5 7 H 19.5" /><path d="M 4.5 12 H 16.5" /><path d="M 4.5 17 H 12.5" />'
  } else {
    body = '<path d="M 3 12 H 7 L 9.5 6 L 14.5 18 L 17 12 H 21" />'
  }
  return `<g transform="translate(${x.toFixed(1)},${y.toFixed(1)}) scale(${scale.toFixed(4)})" fill="none" stroke="${c.stroke}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="${c.opacity.toFixed(2)}">${body}</g>`
}

// ---------------------------------------------------------------------------
// 1. Standalone SVGs: Clean Faceted Architectural X-Prism Mark (260x300 & 256x256)
// ---------------------------------------------------------------------------

function buildBrandCardSvg(options: {
  readonly width: number
  readonly height: number
  readonly prismSize: number
  readonly offsetY: number
  readonly showCaption: boolean
  readonly ariaLabel: string
}): string {
  const { width, height, prismSize, offsetY, showCaption, ariaLabel } = options
  const scale = prismSize / 48
  const originX = (width - prismSize) / 2
  const secondaryPaths = XERXES_PRISM_SECONDARY_PATHS.map(
    (d) => `<path d="${d}" fill="${UI_TITLE}" fill-opacity="0.42" />`,
  ).join('\n    ')
  const primaryPaths = XERXES_PRISM_PRIMARY_PATHS.map(
    (d) => `<path d="${d}" fill="${UI_TITLE}" />`,
  ).join('\n    ')

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${String(width)} ${String(height)}" width="${String(width)}" height="${String(height)}" role="img" aria-label="${ariaLabel}">
  <rect x="1" y="1" width="${String(width - 2)}" height="${String(height - 2)}" rx="20" fill="${UI_BG}" stroke="${UI_HAIRLINE}" stroke-width="1" />
  <g transform="translate(${originX.toFixed(1)},${offsetY.toFixed(1)}) scale(${scale.toFixed(4)})">
    ${secondaryPaths}
    ${primaryPaths}
  </g>${
    showCaption
      ? `
  <text x="${(width / 2).toFixed(1)}" y="${String(height - 32)}" text-anchor="middle" fill="${UI_TITLE}" font-family="${SANS_FONT}" font-size="13" font-weight="600" letter-spacing="4">XERXES</text>
  <text x="${(width / 2).toFixed(1)}" y="${String(height - 16)}" text-anchor="middle" fill="${UI_META}" font-family="${MONO_FONT}" font-size="8.5" letter-spacing="1.6">MULTI-AGENT RUNTIME</text>`
      : ''
  }
</svg>
`
}

function buildDerafshKavianiSvg(): string {
  return buildBrandCardSvg({
    width: 260,
    height: 300,
    prismSize: 148,
    offsetY: 46,
    showCaption: true,
    ariaLabel: 'Xerxes Agents Architectural X-Prism Mark',
  })
}

function buildSimurghEmblemSvg(): string {
  return buildBrandCardSvg({
    width: 256,
    height: 256,
    prismSize: 144,
    offsetY: 56,
    showCaption: false,
    ariaLabel: 'Xerxes Agents Brand Emblem',
  })
}

// ---------------------------------------------------------------------------
// 2. Standalone SVG: Clean Linear-Style Topology Diagram (760x340)
// ---------------------------------------------------------------------------

function buildSwarmConstellationSvg(): string {
  const nodes: ReadonlyArray<{
    readonly id: string
    readonly role: SimurghSpriteRole
    readonly name: string
    readonly subtitle: string
    readonly cx: number
    readonly cy: number
    readonly w: number
    readonly h: number
    readonly dot: string
  }> = [
    { id: 'coordinator', role: 'coordinator', name: 'Coordinator', subtitle: '4 active · orchestrating', cx: 380, cy: 172, w: 176, h: 72, dot: UI_TITLE },
    { id: 'worker', role: 'worker', name: 'implementer', subtitle: 'Active · building', cx: 146, cy: 112, w: 168, h: 64, dot: UI_ACCENT },
    { id: 'narrator', role: 'narrator', name: 'narrator', subtitle: 'Active · summarizing', cx: 146, cy: 232, w: 168, h: 64, dot: UI_SECONDARY },
    { id: 'verifier', role: 'verifier', name: 'verifier', subtitle: 'Completed · verified', cx: 614, cy: 112, w: 168, h: 64, dot: UI_DONE },
    { id: 'monitor', role: 'monitor', name: 'monitor', subtitle: 'Waiting · idle', cx: 614, cy: 232, w: 168, h: 64, dot: UI_META },
  ]

  const connectors: ReadonlyArray<{
    readonly d: string
    readonly stroke: string
    readonly dashed?: boolean
  }> = [
    { d: 'M 292 152 Q 262 126 230 118', stroke: UI_ACCENT },
    { d: 'M 230 100 Q 380 62 530 100', stroke: UI_SEPARATOR },
    { d: 'M 530 120 Q 498 134 468 152', stroke: UI_DONE },
    { d: 'M 292 192 Q 262 218 230 226', stroke: UI_SEPARATOR },
    { d: 'M 530 226 Q 498 214 468 192', stroke: UI_META, dashed: true },
  ]

  const connectorMarkup = connectors
    .map(
      (c) =>
        `\n    <path d="${c.d}" fill="none" stroke="${c.stroke}" stroke-width="1.3" stroke-opacity="0.7"${c.dashed ? ' stroke-dasharray="4 4"' : ''} stroke-linecap="round" />`,
    )
    .join('')

  const nodeMarkup = nodes
    .map((n) => {
      const x = n.cx - n.w / 2
      const y = n.cy - n.h / 2
      return `
    <g>
      <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${String(n.w)}" height="${String(n.h)}" rx="10" fill="${UI_CARD}" stroke="${UI_HAIRLINE}" stroke-width="1" />
      ${renderRoleIconSvg(n.role, 'working', x + 14, y + 16, 15)}
      <text x="${(x + 36).toFixed(1)}" y="${(y + 24).toFixed(1)}" fill="${UI_TITLE}" font-family="${SANS_FONT}" font-size="12.5" font-weight="600" dominant-baseline="central">${n.name}</text>
      <circle cx="${(x + n.w - 16).toFixed(1)}" cy="${(y + 24).toFixed(1)}" r="3" fill="${n.dot}" />
      <text x="${(x + 36).toFixed(1)}" y="${(y + 44).toFixed(1)}" fill="${UI_META}" font-family="${SANS_FONT}" font-size="11" dominant-baseline="central">${n.subtitle}</text>
    </g>`
    })
    .join('')

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" width="760" height="340" role="img" aria-label="Xerxes Immortals Topology">
  <rect x="1" y="1" width="758" height="338" rx="14" fill="${UI_BG}" stroke="${UI_HAIRLINE}" stroke-width="1" />
  <g>
    <text x="20" y="26" font-family="${SANS_FONT}"><tspan fill="${UI_TITLE}" font-size="12" font-weight="600">XERXES IMMORTALS</tspan><tspan dx="10" fill="${UI_META}" font-size="11" font-weight="400">· Multi-Agent Topology</tspan></text>
    <circle cx="702" cy="22" r="3" fill="${UI_DONE}" />
    <text x="712" y="26" fill="${UI_META}" font-family="${SANS_FONT}" font-size="11" font-weight="500">Live</text>
  </g>
  <line x1="1" y1="42" x2="759" y2="42" stroke="${UI_HAIRLINE}" stroke-width="1" />

  <g>${connectorMarkup}
  </g>

  <g>${nodeMarkup}
  </g>

  <line x1="1" y1="302" x2="759" y2="302" stroke="${UI_HAIRLINE}" stroke-width="1" />
  <g font-family="${SANS_FONT}" font-size="11" fill="${UI_META}">
    <circle cx="28" cy="321" r="3" fill="${UI_TITLE}" />
    <text x="38" y="324.5">Coordinator</text>

    <circle cx="132" cy="321" r="3" fill="${UI_ACCENT}" />
    <text x="142" y="324.5">Active</text>

    <circle cx="212" cy="321" r="3" fill="${UI_DONE}" />
    <text x="222" y="324.5">Completed</text>

    <circle cx="312" cy="321" r="3" fill="${UI_META}" />
    <text x="322" y="324.5">Waiting</text>
  </g>
</svg>
`
}

// ---------------------------------------------------------------------------
// 3. Standalone SVG: Clean Open-Arc Thinking Spinner (120x120)
// ---------------------------------------------------------------------------

function buildThinkingHaloSvg(): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120" role="img" aria-label="Xerxes Progress Spinner">
  <rect x="1" y="1" width="118" height="118" rx="22" fill="${UI_BG}" stroke="${UI_HAIRLINE}" stroke-width="1" />
  <circle cx="60" cy="60" r="28" fill="none" stroke="${UI_HAIRLINE}" stroke-width="5.5" />
  <g>
    <animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="1.4s" repeatCount="indefinite" />
    <path d="M 60 32 A 28 28 0 0 1 88 60" fill="none" stroke="${UI_TITLE}" stroke-width="5.5" stroke-linecap="round" />
  </g>
</svg>
`
}

// ---------------------------------------------------------------------------
// 4. Standalone SVG: Monochrome Dev-Tool Role Icons Sheet (760x280)
// ---------------------------------------------------------------------------

function buildSpritesSheetSvg(): string {
  const columns: ReadonlyArray<{
    readonly role: SimurghSpriteRole
    readonly title: string
    readonly subtitle: string
    readonly cx: number
  }> = [
    { role: 'coordinator', title: 'Coordinator', subtitle: 'Hub network', cx: 176 },
    { role: 'worker', title: 'Worker', subtitle: 'Terminal', cx: 298 },
    { role: 'verifier', title: 'Verifier', subtitle: 'Circle check', cx: 420 },
    { role: 'narrator', title: 'Narrator', subtitle: 'Summary', cx: 542 },
    { role: 'monitor', title: 'Monitor', subtitle: 'Activity', cx: 664 },
  ]

  const rows: ReadonlyArray<{
    readonly label: string
    readonly state: SimurghSpriteState
    readonly dot: string
    readonly y: number
  }> = [
    { label: 'Active', state: 'working', dot: UI_TITLE, y: 78 },
    { label: 'Waiting', state: 'waiting', dot: UI_META, y: 142 },
    { label: 'Completed', state: 'killed', dot: UI_DONE, y: 206 },
  ]

  const headerCols = columns
    .map(
      (c) => `
    <text x="${String(c.cx)}" y="53" text-anchor="middle" fill="${UI_TITLE}" font-family="${SANS_FONT}" font-size="11.5" font-weight="600">${c.title}</text>
    <text x="${String(c.cx)}" y="66" text-anchor="middle" fill="${UI_META}" font-family="${SANS_FONT}" font-size="9.5">${c.subtitle}</text>`,
    )
    .join('')

  const gridCells: string[] = []
  for (const row of rows) {
    const labelY = row.y + 30
    gridCells.push(`
    <circle cx="28" cy="${String(row.y + 26)}" r="3" fill="${row.dot}" />
    <text x="40" y="${String(labelY)}" fill="${UI_SECONDARY}" font-family="${SANS_FONT}" font-size="11.5" font-weight="500">${row.label}</text>`)

    for (const col of columns) {
      const tileX = col.cx - 50
      const iconSize = 22
      const iconX = col.cx - iconSize / 2
      const iconY = row.y + (52 - iconSize) / 2
      gridCells.push(`
    <rect x="${String(tileX)}" y="${String(row.y)}" width="100" height="52" rx="10" fill="${UI_CARD}" stroke="${UI_HAIRLINE}" stroke-width="1" />
    ${renderRoleIconSvg(col.role, row.state, iconX, iconY, iconSize)}`)
    }
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 280" width="760" height="280" role="img" aria-label="Xerxes Role Icons Sheet">
  <rect x="1" y="1" width="758" height="278" rx="14" fill="${UI_BG}" stroke="${UI_HAIRLINE}" stroke-width="1" />
  <text x="20" y="24"><tspan fill="${UI_TITLE}" font-family="${SANS_FONT}" font-size="12" font-weight="600">Xerxes Role Icons</tspan><tspan dx="10" fill="${UI_META}" font-family="${SANS_FONT}" font-size="11">· 24×24 · 1.5px stroke · 5 roles × 3 states</tspan></text>
  <line x1="1" y1="36" x2="759" y2="36" stroke="${UI_HAIRLINE}" stroke-width="1" />

  <g>${headerCols}
  </g>

  <g>${gridCells.join('')}
  </g>
</svg>
`
}

// ---------------------------------------------------------------------------
// Pure TypeScript Anti-Aliased Vector RGBA Rasterizer for GIF89a Assets
// ---------------------------------------------------------------------------

interface RgbColor {
  readonly r: number
  readonly g: number
  readonly b: number
}

function hexToRgb(hex: string): RgbColor {
  const clean = hex.startsWith('#') ? hex.slice(1) : hex
  const val = Number.parseInt(clean, 16)
  return {
    r: (val >> 16) & 0xff,
    g: (val >> 8) & 0xff,
    b: val & 0xff,
  }
}

const FONT_5X7: Readonly<Record<string, readonly string[]>> = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
  C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
  G: ['01111', '10000', '10000', '10111', '10001', '10001', '01111'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  I: ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
  J: ['00111', '00010', '00010', '00010', '10010', '10010', '01100'],
  K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
  V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
  W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
  X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
  Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
  Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
  '0': ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  '2': ['01110', '10001', '00001', '00110', '01000', '10000', '11111'],
  '3': ['11110', '00001', '00001', '01110', '00001', '00001', '11110'],
  '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  '6': ['01110', '10000', '11110', '10001', '10001', '10001', '01110'],
  '7': ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  '8': ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  '9': ['01110', '10001', '10001', '01111', '00001', '00001', '01110'],
  '+': ['00000', '00100', '00100', '11111', '00100', '00100', '00000'],
  '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
  '.': ['00000', '00000', '00000', '00000', '00000', '01100', '01100'],
  '·': ['00000', '00000', '00100', '01110', '00100', '00000', '00000'],
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
}

class RgbaSurface {
  readonly width: number
  readonly height: number
  readonly rgba: Uint8ClampedArray

  constructor(width: number, height: number, bgHex = UI_BG) {
    this.width = width
    this.height = height
    this.rgba = new Uint8ClampedArray(width * height * 4)
    this.fillSolid(bgHex)
  }

  fillSolid(hex: string): void {
    const { r, g, b } = hexToRgb(hex)
    for (let i = 0; i < this.rgba.length; i += 4) {
      this.rgba[i] = r
      this.rgba[i + 1] = g
      this.rgba[i + 2] = b
      this.rgba[i + 3] = 255
    }
  }

  blendPixel(x: number, y: number, r: number, g: number, b: number, alpha: number): void {
    const ix = Math.round(x)
    const iy = Math.round(y)
    if (ix < 0 || iy < 0 || ix >= this.width || iy >= this.height || alpha <= 0) return
    const a = alpha >= 1 ? 1 : alpha
    const idx = (iy * this.width + ix) * 4
    if (a >= 0.999) {
      this.rgba[idx] = r
      this.rgba[idx + 1] = g
      this.rgba[idx + 2] = b
      this.rgba[idx + 3] = 255
      return
    }
    const inv = 1 - a
    this.rgba[idx] = Math.round((this.rgba[idx] ?? 0) * inv + r * a)
    this.rgba[idx + 1] = Math.round((this.rgba[idx + 1] ?? 0) * inv + g * a)
    this.rgba[idx + 2] = Math.round((this.rgba[idx + 2] ?? 0) * inv + b * a)
    this.rgba[idx + 3] = 255
  }

  drawFilledCircle(cx: number, cy: number, radius: number, hex: string, alpha = 1): void {
    const { r, g, b } = hexToRgb(hex)
    const minX = Math.max(0, Math.floor(cx - radius - 1))
    const maxX = Math.min(this.width - 1, Math.ceil(cx + radius + 1))
    const minY = Math.max(0, Math.floor(cy - radius - 1))
    const maxY = Math.min(this.height - 1, Math.ceil(cy + radius + 1))
    for (let y = minY; y <= maxY; y += 1) {
      const dy = y - cy
      for (let x = minX; x <= maxX; x += 1) {
        const dx = x - cx
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist <= radius - 0.45) {
          this.blendPixel(x, y, r, g, b, alpha)
        } else if (dist < radius + 0.65) {
          const cov = Math.max(0, Math.min(1, radius + 0.65 - dist))
          this.blendPixel(x, y, r, g, b, alpha * cov)
        }
      }
    }
  }

  drawRing(
    cx: number,
    cy: number,
    radius: number,
    thickness: number,
    hex: string,
    alpha = 1,
  ): void {
    const { r, g, b } = hexToRgb(hex)
    const half = thickness / 2
    const minX = Math.max(0, Math.floor(cx - radius - half - 1))
    const maxX = Math.min(this.width - 1, Math.ceil(cx + radius + half + 1))
    const minY = Math.max(0, Math.floor(cy - radius - half - 1))
    const maxY = Math.min(this.height - 1, Math.ceil(cy + radius + half + 1))

    for (let y = minY; y <= maxY; y += 1) {
      const dy = y - cy
      for (let x = minX; x <= maxX; x += 1) {
        const dx = x - cx
        const dist = Math.sqrt(dx * dx + dy * dy)
        const delta = Math.abs(dist - radius)
        if (delta > half + 0.75) continue
        const cov = delta <= half ? 1 : Math.max(0, 1 - (delta - half) / 0.75)
        this.blendPixel(x, y, r, g, b, alpha * cov)
      }
    }
  }

  drawArc(
    cx: number,
    cy: number,
    radius: number,
    startRad: number,
    spanRad: number,
    thickness: number,
    hex: string,
    alpha = 1,
  ): void {
    const segs = Math.max(18, Math.ceil(radius * spanRad * 0.7))
    for (let i = 0; i < segs; i += 1) {
      const a0 = startRad + (spanRad * i) / segs
      const a1 = startRad + (spanRad * (i + 1)) / segs
      this.drawLine(
        cx + radius * Math.cos(a0),
        cy + radius * Math.sin(a0),
        cx + radius * Math.cos(a1),
        cy + radius * Math.sin(a1),
        hex,
        alpha,
        thickness,
      )
    }
  }

  drawLine(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    hex: string,
    alpha = 1,
    thickness = 1.4,
    dashed = false,
  ): void {
    if (dashed) {
      const dx = x1 - x0
      const dy = y1 - y0
      const len = Math.hypot(dx, dy)
      if (len < 0.01) return
      const dashLen = 4.0
      const gapLen = 4.0
      let pos = 0
      while (pos < len) {
        const end = Math.min(len, pos + dashLen)
        this.drawLine(
          x0 + (dx * pos) / len,
          y0 + (dy * pos) / len,
          x0 + (dx * end) / len,
          y0 + (dy * end) / len,
          hex,
          alpha,
          thickness,
          false,
        )
        pos += dashLen + gapLen
      }
      return
    }

    const { r, g, b } = hexToRgb(hex)
    const half = Math.max(0.35, thickness / 2)
    const pad = Math.ceil(half + 1.5)
    const minX = Math.max(0, Math.floor(Math.min(x0, x1) - pad))
    const maxX = Math.min(this.width - 1, Math.ceil(Math.max(x0, x1) + pad))
    const minY = Math.max(0, Math.floor(Math.min(y0, y1) - pad))
    const maxY = Math.min(this.height - 1, Math.ceil(Math.max(y0, y1) + pad))

    const vx = x1 - x0
    const vy = y1 - y0
    const vLen2 = vx * vx + vy * vy

    for (let py = minY; py <= maxY; py += 1) {
      for (let px = minX; px <= maxX; px += 1) {
        let t = 0
        if (vLen2 > 1e-6) {
          t = Math.max(0, Math.min(1, ((px - x0) * vx + (py - y0) * vy) / vLen2))
        }
        const dist = Math.hypot(px - (x0 + vx * t), py - (y0 + vy * t))
        if (dist <= half - 0.4) {
          this.blendPixel(px, py, r, g, b, alpha)
        } else if (dist < half + 0.7) {
          const cov = Math.max(0, Math.min(1, (half + 0.7 - dist) / 1.1))
          this.blendPixel(px, py, r, g, b, alpha * cov)
        }
      }
    }
  }

  strokePolyline(
    pts: ReadonlyArray<readonly [number, number]>,
    hex: string,
    alpha: number,
    thickness: number,
    closed = false,
  ): void {
    const count = closed ? pts.length : pts.length - 1
    for (let i = 0; i < count; i += 1) {
      const p0 = pts[i]
      const p1 = pts[(i + 1) % pts.length]
      if (!p0 || !p1) continue
      this.drawLine(p0[0], p0[1], p1[0], p1[1], hex, alpha, thickness)
    }
  }

  drawQuadBezier(
    x0: number,
    y0: number,
    cx: number,
    cy: number,
    x1: number,
    y1: number,
    hex: string,
    alpha = 1,
    thickness = 1.4,
    dashed = false,
  ): void {
    const approxLen = Math.hypot(cx - x0, cy - y0) + Math.hypot(x1 - cx, y1 - cy)
    const segs = Math.max(18, Math.ceil(approxLen / 3))
    const pts: Array<readonly [number, number]> = []
    for (let i = 0; i <= segs; i += 1) {
      const t = i / segs
      const inv = 1 - t
      pts.push([
        inv * inv * x0 + 2 * inv * t * cx + t * t * x1,
        inv * inv * y0 + 2 * inv * t * cy + t * t * y1,
      ])
    }
    for (let i = 0; i < segs; i += 1) {
      if (dashed && Math.floor((i / segs) * (approxLen / 7)) % 2 === 1) continue
      const p0 = pts[i]!
      const p1 = pts[i + 1]!
      this.drawLine(p0[0], p0[1], p1[0], p1[1], hex, alpha, thickness)
    }
  }

  drawRoundedRect(
    x: number,
    y: number,
    w: number,
    h: number,
    radius: number,
    fillHex: string | null,
    fillAlpha = 1,
    strokeHex: string | null = null,
    strokeWidth = 1.0,
    strokeAlpha = 1,
  ): void {
    const cx = x + w / 2
    const cy = y + h / 2
    const hw = w / 2 - radius
    const hh = h / 2 - radius
    const pad = Math.ceil(strokeWidth + 2)
    const minX = Math.max(0, Math.floor(x - pad))
    const maxX = Math.min(this.width - 1, Math.ceil(x + w + pad))
    const minY = Math.max(0, Math.floor(y - pad))
    const maxY = Math.min(this.height - 1, Math.ceil(y + h + pad))

    const fillRgb = fillHex !== null ? hexToRgb(fillHex) : null
    const strokeRgb = strokeHex !== null ? hexToRgb(strokeHex) : null
    const halfStroke = strokeWidth / 2

    for (let py = minY; py <= maxY; py += 1) {
      const qy = Math.abs(py - cy) - hh
      for (let px = minX; px <= maxX; px += 1) {
        const qx = Math.abs(px - cx) - hw
        const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0))
        const inside = Math.min(Math.max(qx, qy), 0)
        const sdf = outside + inside - radius

        if (fillRgb !== null && sdf < 0.7) {
          const cov = Math.max(0, Math.min(1, 0.7 - sdf))
          this.blendPixel(px, py, fillRgb.r, fillRgb.g, fillRgb.b, fillAlpha * cov)
        }

        if (strokeRgb !== null) {
          const edgeDist = Math.abs(sdf) - halfStroke
          if (edgeDist < 0.7) {
            const cov = Math.max(0, Math.min(1, 0.7 - edgeDist))
            this.blendPixel(px, py, strokeRgb.r, strokeRgb.g, strokeRgb.b, strokeAlpha * cov)
          }
        }
      }
    }
  }

  fillPolygonAA(
    points: ReadonlyArray<readonly [number, number]>,
    hex: string,
    alpha = 1,
  ): void {
    if (points.length < 3 || alpha <= 0) return
    const col = hexToRgb(hex)
    let minX = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY
    for (const [px, py] of points) {
      if (px < minX) minX = px
      if (px > maxX) maxX = px
      if (py < minY) minY = py
      if (py > maxY) maxY = py
    }
    const ix0 = Math.max(0, Math.floor(minX - 1))
    const ix1 = Math.min(this.width - 1, Math.ceil(maxX + 1))
    const iy0 = Math.max(0, Math.floor(minY - 1))
    const iy1 = Math.min(this.height - 1, Math.ceil(maxY + 1))

    const n = points.length
    const subOffsets = [-0.25, 0.25] as const

    const pointInPoly = (sx: number, sy: number): boolean => {
      let inside = false
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const pi = points[i]!
        const pj = points[j]!
        const intersect =
          pi[1] > sy !== pj[1] > sy &&
          sx < ((pj[0] - pi[0]) * (sy - pi[1])) / (pj[1] - pi[1]) + pi[0]
        if (intersect) inside = !inside
      }
      return inside
    }

    for (let py = iy0; py <= iy1; py += 1) {
      for (let px = ix0; px <= ix1; px += 1) {
        let hits = 0
        for (const oy of subOffsets) {
          for (const ox of subOffsets) {
            if (pointInPoly(px + ox, py + oy)) hits += 1
          }
        }
        if (hits === 0) continue
        this.blendPixel(px, py, col.r, col.g, col.b, alpha * (hits / 4))
      }
    }
  }

  drawXerxesPrism(cx: number, cy: number, size: number, pulseAlpha = 1): void {
    const scale = size / 48
    const ox = cx - size / 2
    const oy = cy - size / 2
    const mapPoly = (
      poly: ReadonlyArray<readonly [number, number]>,
    ): ReadonlyArray<readonly [number, number]> =>
      poly.map(([u, v]) => [ox + u * scale, oy + v * scale] as const)

    for (const poly of PRISM_SECONDARY_POLYS) {
      this.fillPolygonAA(mapPoly(poly), UI_TITLE, 0.42 * pulseAlpha)
    }
    for (const poly of PRISM_PRIMARY_POLYS) {
      this.fillPolygonAA(mapPoly(poly), UI_TITLE, pulseAlpha)
    }
  }

  /**
   * Smooth anti-aliased vector renderer for the 5 canonical monochrome 24×24
   * role icons (`1.5px` stroke, transparent background, zero medallion ring).
   */
  drawRoleIcon(
    role: SimurghSpriteRole,
    state: SimurghSpriteState,
    cx: number,
    cy: number,
    size: number,
  ): void {
    const c = resolveRoleColor(role, state)
    const scale = size / 24
    const tx = (u: number): number => cx + (u - 12) * scale
    const ty = (v: number): number => cy + (v - 12) * scale
    const lw = Math.max(1.1, 1.5 * scale)

    if (role === 'coordinator') {
      this.drawRing(tx(12), ty(5.5), 2.5 * scale, lw, c.stroke, c.opacity)
      this.drawRing(tx(6), ty(18.5), 2.5 * scale, lw, c.stroke, c.opacity)
      this.drawRing(tx(18), ty(18.5), 2.5 * scale, lw, c.stroke, c.opacity)
      this.drawLine(tx(12), ty(8), tx(12), ty(12), c.stroke, c.opacity, lw)
      this.strokePolyline(
        [
          [tx(6), ty(16)],
          [tx(6), ty(12)],
          [tx(18), ty(12)],
          [tx(18), ty(16)],
        ],
        c.stroke,
        c.opacity,
        lw,
      )
    } else if (role === 'worker') {
      this.strokePolyline(
        [
          [tx(4.5), ty(7)],
          [tx(10.5), ty(12)],
          [tx(4.5), ty(17)],
        ],
        c.stroke,
        c.opacity,
        lw,
      )
      this.drawLine(tx(12.5), ty(17), tx(19.5), ty(17), c.stroke, c.opacity, lw)
    } else if (role === 'verifier') {
      this.drawRing(cx, cy, 8.5 * scale, lw, c.stroke, c.opacity)
      this.strokePolyline(
        [
          [tx(8.5), ty(12.2)],
          [tx(10.8), ty(14.5)],
          [tx(15.8), ty(9.5)],
        ],
        c.stroke,
        c.opacity,
        lw,
      )
    } else if (role === 'narrator') {
      this.drawLine(tx(4.5), ty(7), tx(19.5), ty(7), c.stroke, c.opacity, lw)
      this.drawLine(tx(4.5), ty(12), tx(16.5), ty(12), c.stroke, c.opacity, lw)
      this.drawLine(tx(4.5), ty(17), tx(12.5), ty(17), c.stroke, c.opacity, lw)
    } else {
      this.strokePolyline(
        [
          [tx(3), ty(12)],
          [tx(7), ty(12)],
          [tx(9.5), ty(6)],
          [tx(14.5), ty(18)],
          [tx(17), ty(12)],
          [tx(21), ty(12)],
        ],
        c.stroke,
        c.opacity,
        lw,
      )
    }
  }

  drawText(
    text: string,
    x: number,
    y: number,
    hex: string,
    scale = 1,
    align: 'left' | 'center' | 'right' = 'left',
  ): void {
    const upper = text.toUpperCase()
    const charW = 5 * scale
    const gap = scale
    const totalWidth = upper.length > 0 ? upper.length * charW + (upper.length - 1) * gap : 0
    let cursorX = x
    if (align === 'center') cursorX = Math.round(x - totalWidth / 2)
    if (align === 'right') cursorX = Math.round(x - totalWidth)
    const { r, g, b } = hexToRgb(hex)

    for (let i = 0; i < upper.length; i += 1) {
      const ch = upper.charAt(i)
      const glyph = FONT_5X7[ch] ?? FONT_5X7[' ']
      if (glyph !== undefined) {
        for (let gy = 0; gy < glyph.length; gy += 1) {
          const line = glyph[gy] ?? ''
          for (let gx = 0; gx < line.length; gx += 1) {
            if (line.charAt(gx) !== '1') continue
            for (let sy = 0; sy < scale; sy += 1) {
              for (let sx = 0; sx < scale; sx += 1) {
                this.blendPixel(cursorX + gx * scale + sx, y + gy * scale + sy, r, g, b, 1)
              }
            }
          }
        }
      }
      cursorX += charW + gap
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Animated GIFs: Monochrome Role Icons (Coordinator & Worker, 160x160)
// ---------------------------------------------------------------------------

function buildRoleIconGif(role: 'coordinator' | 'worker', subtitle: string): Uint8Array {
  const width = 160
  const height = 160
  const frameCount = 12
  const frames: GifFrame[] = []

  for (let f = 0; f < frameCount; f += 1) {
    const surface = new RgbaSurface(width, height, UI_BG)
    surface.drawRoundedRect(4, 4, width - 8, height - 8, 16, UI_CARD, 1, UI_HAIRLINE, 1, 1)
    surface.drawRoleIcon(role, 'working', 80, 72, 60)
    surface.drawText(subtitle, 80, 134, UI_META, 1, 'center')
    frames.push({ width, height, rgba: surface.rgba, delayMs: 110 })
  }

  return encodeAnimatedGif(frames, { loop: 0, defaultDelayMs: 110 })
}

// ---------------------------------------------------------------------------
// 6. Animated GIFs: Faceted Architectural X-Prism Mark (280x320 & 280x280)
// ---------------------------------------------------------------------------

function buildBrandGif(
  width: number,
  height: number,
  frameCount: number,
  prismSize: number,
  prismCy: number,
  footerText: string,
): Uint8Array {
  const frames: GifFrame[] = []

  for (let f = 0; f < frameCount; f += 1) {
    const phase = (f / frameCount) * Math.PI * 2
    const pulse = 0.92 + 0.08 * (0.5 + 0.5 * Math.cos(phase))
    const surface = new RgbaSurface(width, height, UI_BG)
    surface.drawRoundedRect(3, 3, width - 6, height - 6, 20, UI_BG, 1, UI_HAIRLINE, 1, 1)
    surface.drawXerxesPrism(width / 2, prismCy, prismSize, pulse)
    surface.drawText(footerText, width / 2, height - 24, UI_META, 1, 'center')
    frames.push({ width, height, rgba: surface.rgba, delayMs: 110 })
  }

  return encodeAnimatedGif(frames, { loop: 0, defaultDelayMs: 110 })
}

// ---------------------------------------------------------------------------
// 7. Animated GIF: Clean Linear-Style Map Topology (640x400, 24 frames)
// ---------------------------------------------------------------------------

function buildImmortalsMapGif(): Uint8Array {
  const width = 640
  const height = 400
  const frameCount = 24
  const frames: GifFrame[] = []

  const completed = { x: 204, y: 58, w: 232, h: 72 }
  const active = { x: 32, y: 158, w: 212, h: 88 }
  const waiting = { x: 396, y: 158, w: 212, h: 88 }
  const coord = { x: 184, y: 274, w: 272, h: 78 }

  const vectors: ReadonlyArray<{
    readonly x0: number
    readonly y0: number
    readonly cx: number
    readonly cy: number
    readonly x1: number
    readonly y1: number
    readonly stroke: string
    readonly phaseOffset: number
    readonly dashed?: boolean
  }> = [
    { x0: 320, y0: 132, cx: 320, cy: 202, x1: 320, y1: 270, stroke: UI_DONE, phaseOffset: 0.1 },
    { x0: 214, y0: 272, cx: 182, cy: 264, x1: 168, y1: 248, stroke: UI_ACCENT, phaseOffset: 0.0 },
    { x0: 392, y0: 202, cx: 320, cy: 194, x1: 248, y1: 202, stroke: UI_META, phaseOffset: 0.35, dashed: true },
    { x0: 168, y0: 154, cx: 182, cy: 124, x1: 202, y1: 108, stroke: UI_DONE, phaseOffset: 0.65 },
  ]

  for (let f = 0; f < frameCount; f += 1) {
    const t = f / frameCount
    const phase = t * Math.PI * 2
    const surface = new RgbaSurface(width, height, UI_BG)

    // Clean top bar
    surface.drawText('XERXES IMMORTALS · MAP', 20, 16, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(570, 19, 3, UI_DONE, 1)
    surface.drawText('LIVE', 580, 16, UI_META, 1, 'left')
    surface.drawLine(0, 38, width, 38, UI_HAIRLINE, 1, 1)

    // Subtle connector edges
    for (const vec of vectors) {
      surface.drawQuadBezier(
        vec.x0,
        vec.y0,
        vec.cx,
        vec.cy,
        vec.x1,
        vec.y1,
        vec.stroke,
        0.65,
        1.4,
        vec.dashed ?? false,
      )
      const u = (t + vec.phaseOffset) % 1
      const inv = 1 - u
      const sx = inv * inv * vec.x0 + 2 * inv * u * vec.cx + u * u * vec.x1
      const sy = inv * inv * vec.y0 + 2 * inv * u * vec.cy + u * u * vec.y1
      surface.drawFilledCircle(sx, sy, 2.2, vec.stroke, 0.92)
    }

    // Completed card
    surface.drawRoundedRect(completed.x, completed.y, completed.w, completed.h, 10, UI_CARD, 1, UI_HAIRLINE, 1, 1)
    surface.drawFilledCircle(completed.x + 14, completed.y + 15, 3, UI_DONE, 1)
    surface.drawText('COMPLETED', completed.x + 24, completed.y + 12, UI_META, 1, 'left')
    surface.drawText('3 DONE', completed.x + completed.w - 14, completed.y + 12, UI_META, 1, 'right')
    surface.drawLine(completed.x, completed.y + 28, completed.x + completed.w, completed.y + 28, UI_HAIRLINE, 1, 1)
    surface.drawRoleIcon('worker', 'killed', completed.x + 22, completed.y + 49, 14)
    surface.drawText('SWARM-GRAPH', completed.x + 36, completed.y + 46, UI_SECONDARY, 1, 'left')
    surface.drawFilledCircle(completed.x + completed.w - 14, completed.y + 49, 3, UI_DONE, 1)

    // Active card
    surface.drawRoundedRect(active.x, active.y, active.w, active.h, 10, UI_CARD, 1, UI_HAIRLINE, 1, 1)
    surface.drawFilledCircle(active.x + 14, active.y + 15, 3, UI_TITLE, 1)
    surface.drawText('ACTIVE', active.x + 24, active.y + 12, UI_META, 1, 'left')
    surface.drawText('2 ACTIVE', active.x + active.w - 14, active.y + 12, UI_META, 1, 'right')
    surface.drawLine(active.x, active.y + 28, active.x + active.w, active.y + 28, UI_HAIRLINE, 1, 1)
    surface.drawRoleIcon('worker', 'working', active.x + 22, active.y + 46, 14)
    surface.drawText('MINIMAL-UI', active.x + 36, active.y + 43, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(active.x + active.w - 14, active.y + 46, 3, UI_ACCENT, 1)
    surface.drawRoleIcon('narrator', 'working', active.x + 22, active.y + 70, 14)
    surface.drawText('NARRATOR', active.x + 36, active.y + 67, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(active.x + active.w - 14, active.y + 70, 3, UI_ACCENT, 1)

    // Waiting card
    surface.drawRoundedRect(waiting.x, waiting.y, waiting.w, waiting.h, 10, UI_CARD, 1, UI_HAIRLINE, 1, 1)
    surface.drawFilledCircle(waiting.x + 14, waiting.y + 15, 3, UI_META, 1)
    surface.drawText('WAITING', waiting.x + 24, waiting.y + 12, UI_META, 1, 'left')
    surface.drawText('2 WAITING', waiting.x + waiting.w - 14, waiting.y + 12, UI_META, 1, 'right')
    surface.drawLine(waiting.x, waiting.y + 28, waiting.x + waiting.w, waiting.y + 28, UI_HAIRLINE, 1, 1)
    surface.drawRoleIcon('verifier', 'waiting', waiting.x + 22, waiting.y + 46, 14)
    surface.drawText('VERIFIER', waiting.x + 36, waiting.y + 43, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(waiting.x + waiting.w - 14, waiting.y + 46, 3, UI_META, 1)
    surface.drawRoleIcon('monitor', 'waiting', waiting.x + 22, waiting.y + 70, 14)
    surface.drawText('MONITOR', waiting.x + 36, waiting.y + 67, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(waiting.x + waiting.w - 14, waiting.y + 70, 3, UI_META, 1)

    // Coordinator card
    surface.drawRoundedRect(coord.x, coord.y, coord.w, coord.h, 10, UI_CARD, 1, UI_HAIRLINE, 1, 1)
    surface.drawRoleIcon('coordinator', 'working', coord.x + 22, coord.y + 22, 14)
    surface.drawText('COORDINATOR', coord.x + 36, coord.y + 19, UI_TITLE, 1, 'left')
    surface.drawFilledCircle(coord.x + coord.w - 14, coord.y + 22, 3, UI_ACCENT, 1)
    surface.drawText('3 OF 7 COMPLETED', coord.x + 36, coord.y + 39, UI_META, 1, 'left')

    const barX = coord.x + 14
    const barY = coord.y + 60
    const barW = coord.w - 28
    surface.drawRoundedRect(barX, barY, barW, 4, 2, UI_CHIP, 1)
    const doneFill = Math.round(barW * 0.43)
    const activeFill = Math.round(barW * (0.24 + 0.03 * Math.sin(phase)))
    surface.drawRoundedRect(barX, barY, doneFill, 4, 2, UI_DONE, 1)
    surface.drawRoundedRect(barX + doneFill, barY, activeFill, 4, 1, UI_ACCENT, 0.72)

    // Clean legend row
    surface.drawLine(0, 368, width, 368, UI_HAIRLINE, 1, 1)
    surface.drawLine(24, 384, 38, 384, UI_DONE, 1, 1.5)
    surface.drawText('COMPLETED', 46, 381, UI_META, 1, 'left')

    surface.drawLine(156, 384, 170, 384, UI_ACCENT, 1, 1.5)
    surface.drawText('DISPATCH SENT', 178, 381, UI_META, 1, 'left')

    surface.drawLine(316, 384, 334, 384, UI_META, 1, 1.5, true)
    surface.drawText('AWAITING HANDOFF', 342, 381, UI_META, 1, 'left')

    frames.push({ width, height, rgba: surface.rgba, delayMs: 100 })
  }

  return encodeAnimatedGif(frames, { loop: 0, defaultDelayMs: 100 })
}

// ---------------------------------------------------------------------------
// 8. Animated GIF: Minimal Open-Arc Progress Spinner (96x96, 12 frames)
// ---------------------------------------------------------------------------

function buildWorkingBadgeGif(): Uint8Array {
  const width = 96
  const height = 96
  const frameCount = 12
  const frames: GifFrame[] = []
  const cx = 48
  const cy = 48

  for (let f = 0; f < frameCount; f += 1) {
    const startRad = (f / frameCount) * Math.PI * 2 - Math.PI / 2
    const surface = new RgbaSurface(width, height, UI_BG)
    surface.drawRoundedRect(2, 2, width - 4, height - 4, 18, UI_BG, 1, UI_HAIRLINE, 1, 1)
    surface.drawRing(cx, cy, 24, 4.2, UI_HAIRLINE, 1)
    surface.drawArc(cx, cy, 24, startRad, Math.PI * 0.55, 4.2, UI_TITLE, 1)
    frames.push({ width, height, rgba: surface.rgba, delayMs: 80 })
  }

  return encodeAnimatedGif(frames, { loop: 0, defaultDelayMs: 80 })
}

// ---------------------------------------------------------------------------
// 9. Animated GIF: Clean Completion Check (128x128, 16 frames)
// ---------------------------------------------------------------------------

function buildCelebrateGif(): Uint8Array {
  const width = 128
  const height = 128
  const frameCount = 16
  const frames: GifFrame[] = []
  const cx = 64
  const cy = 64

  for (let f = 0; f < frameCount; f += 1) {
    const t = f / frameCount
    const surface = new RgbaSurface(width, height, UI_BG)
    surface.drawRoundedRect(4, 4, width - 8, height - 8, 18, UI_CARD, 1, UI_HAIRLINE, 1, 1)

    const radius = 26 + t * 22
    const alpha = Math.sin(t * Math.PI) * 0.35
    surface.drawRing(cx, cy, radius, 1.2, UI_DONE, alpha)
    surface.drawRoleIcon('verifier', 'working', cx, cy, 48)

    frames.push({ width, height, rgba: surface.rgba, delayMs: 95 })
  }

  return encodeAnimatedGif(frames, { loop: 0, defaultDelayMs: 95 })
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  await mkdir(BRAND_DIR, { recursive: true })

  if (SIMURGH_ROLES.length !== 5) {
    throw new Error(`Expected 5 canonical roles, found ${String(SIMURGH_ROLES.length)}.`)
  }

  const svgAssets: ReadonlyArray<readonly [string, string]> = [
    ['xerxes-derafsh-kaviani.svg', buildDerafshKavianiSvg()],
    ['xerxes-simurgh-emblem.svg', buildSimurghEmblemSvg()],
    ['xerxes-swarm-constellation.svg', buildSwarmConstellationSvg()],
    ['xerxes-thinking-halo.svg', buildThinkingHaloSvg()],
    ['xerxes-sprites-sheet.svg', buildSpritesSheetSvg()],
  ]

  const mapGif = buildImmortalsMapGif()
  const gifAssets: ReadonlyArray<readonly [string, Uint8Array]> = [
    ['xerxes-derafsh-kaviani.gif', buildBrandGif(280, 320, 20, 152, 134, 'XERXES AGENTS')],
    ['xerxes-simurgh-coordinator.gif', buildRoleIconGif('coordinator', 'COORDINATOR')],
    ['xerxes-simurgh-worker.gif', buildRoleIconGif('worker', 'WORKER')],
    ['xerxes-simurgh-emblem.gif', buildBrandGif(280, 280, 16, 146, 128, 'XERXES')],
    ['xerxes-immortals-battlefield.gif', mapGif],
    ['xerxes-swarm-hive.gif', mapGif],
    ['xerxes-working.gif', buildWorkingBadgeGif()],
    ['xerxes-celebrate.gif', buildCelebrateGif()],
  ]

  for (const [filename, svg] of svgAssets) {
    const target = join(BRAND_DIR, filename)
    await Bun.write(target, svg)
    console.log(`wrote ${filename} (${String(Buffer.byteLength(svg, 'utf8'))} bytes)`)
  }

  for (const [filename, bytes] of gifAssets) {
    const target = join(BRAND_DIR, filename)
    await Bun.write(target, bytes)
    console.log(`wrote ${filename} (${String(bytes.byteLength)} bytes)`)
  }
}

await main()
