// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Xerxes banner art + rich-markup parser. Original Xerxes wordmark/emblem; the
// markup parser and gradient colorizer are generic utilities. Matches the
// banner export surface consumed by components/branding.tsx.

import type { ThemeColors } from './theme.js'

type Line = [string, string]

const RICH_RE = /\[(?:bold\s+)?(?:dim\s+)?(#(?:[0-9a-fA-F]{3,8}))\]([\s\S]*?)(\[\/\])/g

export function parseRichMarkup(markup: string): Line[] {
  const lines: Line[] = []
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
    for (const m of matches) {
      const before = trimmed.slice(cursor, m.index)
      if (before) {
        lines.push(['', before])
      }
      lines.push([m[1]!, m[2]!])
      cursor = m.index! + m[0].length
    }
    if (cursor < trimmed.length) {
      lines.push(['', trimmed.slice(cursor)])
    }
  }
  return lines
}

const LOGO_ART = [
  '██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗',
  '╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝',
  ' ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗',
  ' ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║',
  '██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║',
  '╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝'
]

const EMBLEM_ART = [
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
]

const LOGO_PALETTE = ['#61a7ff', '#5b8dff', '#6575ee', '#665fd6', '#6258b8', '#544c96'] as const
const LOGO_FALLBACK_COLOR = LOGO_PALETTE[LOGO_PALETTE.length - 1]
const EMBLEM_GRADIENT = EMBLEM_ART.map(() => 4)

const colorize = (art: readonly string[], gradient: readonly number[], c: ThemeColors): Line[] => {
  const p = [c.primary, c.accent, c.border, c.muted, c.warn]
  return art.map((text, i) => [p[gradient[i]!] ?? c.muted, text])
}

const colorizeLogo = (art: readonly string[]): Line[] =>
  art.map((text, i) => [LOGO_PALETTE[i] ?? LOGO_FALLBACK_COLOR, text])

export const LOGO_WIDTH = Math.max(...LOGO_ART.map(line => line.length))
export const CADUCEUS_WIDTH = Math.max(...EMBLEM_ART.map(line => line.length))

export const logo = (_c: ThemeColors, customLogo?: string): Line[] =>
  customLogo ? parseRichMarkup(customLogo) : colorizeLogo(LOGO_ART)

// Named `caduceus` to match the branding.tsx import surface; renders the
// Xerxes emblem.
export const caduceus = (c: ThemeColors, customHero?: string): Line[] =>
  customHero ? parseRichMarkup(customHero) : colorize(EMBLEM_ART, EMBLEM_GRADIENT, c)

export const artWidth = (lines: Line[]): number => lines.reduce((m, [, t]) => Math.max(m, t.length), 0)
