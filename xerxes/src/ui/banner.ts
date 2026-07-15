// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Code-native terminal art for Xerxes. The default mark is the supplied
// Xerxes Derafsh Kaviani glyph. Keep the original Braille-pixel payload intact
// so the visual identity is not silently replaced by a generic ornament.

import type { ThemeColors } from './theme.js'

export type ArtLine = [color: string, text: string]

const RICH_RE = /\[(?:bold\s+)?(?:dim\s+)?(#(?:[0-9a-fA-F]{3,8}))\]([\s\S]*?)(\[\/\])/g

export function parseRichMarkup(markup: string): ArtLine[] {
  const lines: ArtLine[] = []
  for (const raw of markup.split('\n')) {
    const trimmed = raw.trimEnd()
    if (!trimmed) {
      lines.push(['', ' '])
      continue
    }
    const matches = [...trimmed.matchAll(RICH_RE)]
    if (!matches.length) {
      lines.push(['', trimmed])
      continue
    }
    let cursor = 0
    for (const match of matches) {
      const before = trimmed.slice(cursor, match.index)
      if (before) {
        lines.push(['', before])
      }
      lines.push([match[1]!, match[2]!])
      cursor = match.index! + match[0].length
    }
    if (cursor < trimmed.length) {
      lines.push(['', trimmed.slice(cursor)])
    }
  }
  return lines
}

const DERAFSH_KAVIANI_RAW_ART = [
  '⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⣿⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠘⢿⣿⣷⣾⣿⣷⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠽⣿⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⢴⣿⡀⠀⠀⠀⠀⠀⣀⠀⠐⠿⠀⠐⡀⠀⠀⠀⠀⠀⢰⣿⡦',
  '⠀⢸⣟⣻⣿⡿⠿⢻⣿⣟⣻⣿⣿⣿⣿⡟⠿⢿⣿⣿⣿⡇⠀',
  '⠀⢸⣿⡽⣏⣿⣶⣄⠀⠀⠀⠤⡄⠀⠀⣠⣶⣿⣹⢿⣿⡇⠀',
  '⠀⢸⣿⡇⠹⣧⣬⣿⣷⡀⠀⠂⠁⢀⣾⣿⣤⣾⠏⢰⣿⡇⠀',
  '⠀⢸⡷⡇⠀⠈⠻⣿⣍⣿⡄⣉⣠⣿⣹⣿⠟⠁⠀⣼⣿⡇⠀',
  '⠀⢸⡿⢿⢠⣶⡄⢀⡉⢻⣿⠛⣿⡟⢩⡀⢠⣤⡄⠿⢿⡇⠀',
  '⠀⢸⡿⡿⠈⠉⠁⣀⣤⣾⣿⣶⣿⣧⣤⣀⠈⠋⠁⢿⣿⡇⠀',
  '⠀⢸⣿⡇⠀⣠⣾⣿⣤⡿⠁⠶⠈⢿⣼⣿⣷⣄⠀⢙⣿⡇⠀',
  '⠀⢸⣯⡁⣼⣏⣩⣿⠟⠀⢠⠒⡄⠀⠻⣿⣉⣻⣧⢸⣿⡇⠀',
  '⠀⢸⣿⣿⣧⠾⠋⣁⣀⡀⣀⣀⡀⢀⣀⡈⠛⢿⣿⣿⣿⡇⠀',
  '⣠⣼⠷⠾⣿⡿⠿⠿⠿⠷⢾⣿⡷⠾⢿⡿⠿⠿⠿⠷⢾⣧⣄',
  '⠙⠛⠀⠀⡾⠀⠀⠀⠀⠀⠀⣿⠀⠀⢸⡇⠀⠀⠀⠀⠈⠛⠃',
  '⠀⠀⠀⣀⡴⠀⠀⠀⠀⠀⢀⣿⡀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⡾⠋⠀⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠳⡄⠀⠀⠀⠀⠀⠀',
  '⠀⢀⡴⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⢠⡇⠀⠀⠀⠀⠀⠀',
  '⠀⠻⠁⠀⠀⠀⢀⡄⠀⠀⠀⣿⠀⠀⠰⡟⠀⠀⠀⠀⠰⡦⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠛⠀⠀⠀⠀⠀⠀⠀'
] as const

export const DERAFSH_KAVIANI_ART = Object.freeze([...DERAFSH_KAVIANI_RAW_ART])

export const DERAFSH_KAVIANI_GLYPH = '✦'
export const DERAFSH_KAVIANI_WIDTH = Math.max(...DERAFSH_KAVIANI_ART.map(line => line.length))
// 12.5 native-only colour updates per second feels lively without asking React
// to reconcile the welcome layout or making idle terminals work unnecessarily.
export const DERAFSH_ANIMATION_FRAME_MS = 80
export const DERAFSH_ANIMATION_FRAME_COUNT = 80

const BRAILLE_BASE = 0x2800
const BRAILLE_LAST = 0x28ff
const BRAILLE_COLUMN_BITS = [
  [0, 1, 2, 6],
  [3, 4, 5, 7]
] as const

const brailleMaskAt = (line: string, column: number): number => {
  const codePoint = line.codePointAt(column) ?? BRAILLE_BASE
  return codePoint >= BRAILLE_BASE && codePoint <= BRAILLE_LAST ? codePoint - BRAILLE_BASE : 0
}

/**
 * Collapse each pair of four-pixel-high Braille rows into one row.
 * Adjacent vertical pixels are ORed, retaining the mark's silhouette while
 * halving its terminal height for the common 24-row viewport.
 */
export function compactBrailleRows(lines: readonly string[]): string[] {
  const compact: string[] = []

  for (let row = 0; row < lines.length; row += 2) {
    const upper = lines[row] ?? ''
    const lower = lines[row + 1] ?? ''
    const width = Math.max(upper.length, lower.length)
    let output = ''

    for (let column = 0; column < width; column += 1) {
      const sourceMasks = [brailleMaskAt(upper, column), brailleMaskAt(lower, column)] as const
      let outputMask = 0

      for (let sourceRow = 0; sourceRow < sourceMasks.length; sourceRow += 1) {
        const sourceMask = sourceMasks[sourceRow]!
        for (const bits of BRAILLE_COLUMN_BITS) {
          for (let pair = 0; pair < 2; pair += 1) {
            const sourceBitA = bits[pair * 2]!
            const sourceBitB = bits[pair * 2 + 1]!
            if (sourceMask & ((1 << sourceBitA) | (1 << sourceBitB))) {
              outputMask |= 1 << bits[sourceRow * 2 + pair]!
            }
          }
        }
      }

      output += String.fromCodePoint(BRAILLE_BASE + outputMask)
    }

    compact.push(output)
  }

  return compact
}

export const DERAFSH_KAVIANI_COMPACT_ART = Object.freeze(compactBrailleRows(DERAFSH_KAVIANI_ART))

const HEX_COLOR_RE = /^#[0-9a-f]{6}$/i
const TRUE_RE = /^(?:1|true|yes|on)$/i
const FALSE_RE = /^(?:0|false|no|off)$/i

const DARK_DERAFSH_BLUE = '#6ea8fe'
const DARK_DERAFSH_PURPLE = '#b18be8'
const DARK_DERAFSH_GOLD = '#d8ae58'
const DARK_DERAFSH_BRIDGE = '#9fb8d8'
const LIGHT_DERAFSH_BLUE = '#1f64b5'
const LIGHT_DERAFSH_PURPLE = '#7047b5'
const LIGHT_DERAFSH_GOLD = '#8a6200'
const LIGHT_DERAFSH_BRIDGE = '#466c91'

const positiveModulo = (value: number, divisor: number): number => ((value % divisor) + divisor) % divisor

const rgb = (color: string): readonly [number, number, number] | null => {
  if (!HEX_COLOR_RE.test(color)) {
    return null
  }

  return [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16)
  ]
}

const mixHex = (from: string, to: string, amount: number): string => {
  const a = rgb(from)
  const b = rgb(to)

  if (!a || !b) {
    return amount < 0.5 ? from : to
  }

  const channel = (index: 0 | 1 | 2) => Math.round(a[index] + (b[index] - a[index]) * amount)

  return `#${[channel(0), channel(1), channel(2)].map(value => value.toString(16).padStart(2, '0')).join('')}`
}

const isLightSurface = (color: string): boolean => {
  const value = rgb(color)

  if (!value) {
    return false
  }

  const linearize = (channel: number) => {
    const normalized = channel / 255
    return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4
  }
  const [red, green, blue] = value

  return 0.2126 * linearize(red) + 0.7152 * linearize(green) + 0.0722 * linearize(blue) >= 0.6
}

const colorOr = (color: string, fallback: string): string => (HEX_COLOR_RE.test(color) ? color : fallback)

/** A cyclic, theme-aware blue → purple → gold palette for the default mark. */
export function derafshGradientPalette(colors: ThemeColors): readonly string[] {
  const light = isLightSurface(colors.statusBg)
  const blue = light ? LIGHT_DERAFSH_BLUE : DARK_DERAFSH_BLUE
  const purple = colorOr(colors.system, light ? LIGHT_DERAFSH_PURPLE : DARK_DERAFSH_PURPLE)
  const gold = colorOr(colors.warn, light ? LIGHT_DERAFSH_GOLD : DARK_DERAFSH_GOLD)
  const bridge = light ? LIGHT_DERAFSH_BRIDGE : DARK_DERAFSH_BRIDGE

  return [blue, purple, gold, bridge]
}

const gradientColor = (palette: readonly string[], position: number): string => {
  const scaled = positiveModulo(position, 1) * palette.length
  const index = Math.floor(scaled) % palette.length
  const next = (index + 1) % palette.length

  return mixHex(palette[index]!, palette[next]!, scaled - Math.floor(scaled))
}

const DERAFSH_WAVE_PRIMARY_TURNS = 1.35
const DERAFSH_WAVE_SECONDARY_TURNS = 2.7
const DERAFSH_WAVE_PRIMARY_AMPLITUDE = 0.085
const DERAFSH_WAVE_SECONDARY_AMPLITUDE = 0.025
const TAU = Math.PI * 2

/**
 * Return a stable colour-space position for one row of the travelling wave.
 *
 * Only colour phase moves: the Braille payload and its terminal cells stay
 * fixed, avoiding the layout jitter that horizontal padding animation causes.
 */
export function derafshWavePosition(row: number, rowCount: number, frame: number): number {
  const phase = positiveModulo(frame, DERAFSH_ANIMATION_FRAME_COUNT) / DERAFSH_ANIMATION_FRAME_COUNT
  const verticalPosition = Math.max(0, Math.min(row, Math.max(0, rowCount - 1))) / Math.max(1, rowCount - 1)
  const primaryWave =
    Math.sin(TAU * (verticalPosition * DERAFSH_WAVE_PRIMARY_TURNS - phase)) * DERAFSH_WAVE_PRIMARY_AMPLITUDE
  const secondaryWave =
    Math.sin(TAU * (verticalPosition * DERAFSH_WAVE_SECONDARY_TURNS + phase * 2)) *
    DERAFSH_WAVE_SECONDARY_AMPLITUDE

  return phase + verticalPosition + primaryWave + secondaryWave
}

const derafshGradientFrameFor = (art: readonly string[], colors: ThemeColors, frame: number): ArtLine[] => {
  const palette = derafshGradientPalette(colors)

  return art.map((text, row) => [gradientColor(palette, derafshWavePosition(row, art.length, frame)), text])
}

/** Build one low-cost travelling colour-wave frame while preserving every glyph. */
export function derafshGradientFrame(colors: ThemeColors, frame: number): ArtLine[] {
  return derafshGradientFrameFor(DERAFSH_KAVIANI_ART, colors, frame)
}

/** Build an animated gradient frame for the half-height Derafsh. */
export function derafshCompactGradientFrame(colors: ThemeColors, frame: number): ArtLine[] {
  return derafshGradientFrameFor(DERAFSH_KAVIANI_COMPACT_ART, colors, frame)
}

/** Respect explicit/reduced-motion terminal policy without changing the art. */
export function derafshAnimationEnabled(
  env: Readonly<Record<string, string | undefined>> = process.env,
  stdoutIsTty = Boolean(process.stdout.isTTY)
): boolean {
  if (!stdoutIsTty) {
    return false
  }

  const override = (env.XERXES_TUI_ANIMATIONS ?? '').trim()
  if (FALSE_RE.test(override)) {
    return false
  }
  if (TRUE_RE.test(override)) {
    return true
  }

  return env.TERM !== 'dumb' && env.NODE_ENV !== 'test' && !TRUE_RE.test((env.CI ?? '').trim())
}

/**
 * Render the default Xerxes terminal mark or a skin-provided replacement.
 * Custom skin artwork keeps the existing rich-markup input path.
 */
export function derafshKaviani(colors: ThemeColors, customMark?: string): ArtLine[] {
  if (customMark) {
    return parseRichMarkup(customMark)
  }

  return DERAFSH_KAVIANI_ART.map(text => [colors.warn, text])
}

export const artWidth = (lines: readonly ArtLine[]): number =>
  lines.reduce((width, [, text]) => Math.max(width, text.length), 0)
