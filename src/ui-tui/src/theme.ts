// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Theme + palette for the Xerxes TUI. The shipped palette is "Persepolis
// Lapis" (mirrors src/python/xerxes/tui/skin_engine.py::_DEFAULT_ROLES): a
// lapis-blue hero with a gold-leaf rule, turquoise highlights, a royal-violet
// system voice and a cool-slate neutral.
//
// The colour math handles light/dark detection plus ANSI down-conversion for
// terminals without truecolor. `fromSkin` lets the daemon push live skin
// overrides later.

export interface ThemeColors {
  primary: string
  accent: string
  border: string
  text: string
  muted: string

  // Xerxes three-voice roles (skin_engine.py)
  toolName: string
  system: string

  completionBg: string
  completionCurrentBg: string
  completionMetaBg: string
  completionMetaCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

  prompt: string
  sessionLabel: string
  sessionBorder: string

  statusBg: string
  statusFg: string
  statusGood: string
  statusWarn: string
  statusBad: string
  statusCritical: string
  selectionBg: string

  diffAdded: string
  diffRemoved: string
  diffAddedWord: string
  diffRemovedWord: string

  shellDollar: string
}

export interface ThemeBrand {
  name: string
  icon: string
  prompt: string
  welcome: string
  goodbye: string
  tool: string
  helpHeader: string
}

export interface Theme {
  color: ThemeColors
  brand: ThemeBrand
  bannerLogo: string
  bannerHero: string
}

// ── Colour math (ported from Xerxes theme.ts) ───────────────────────────

function parseHex(h: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(h)
  if (!m) {
    return null
  }
  const n = Number.parseInt(m[1]!, 16)
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

function mix(a: string, b: string, t: number): string {
  const pa = parseHex(a)
  const pb = parseHex(b)
  if (!pa || !pb) {
    return a
  }
  const lerp = (i: 0 | 1 | 2) => Math.round(pa[i] + (pb[i] - pa[i]) * t)
  return '#' + ((1 << 24) | (lerp(0) << 16) | (lerp(1) << 8) | lerp(2)).toString(16).slice(1)
}

const XTERM_6_LEVELS = [0, 95, 135, 175, 215, 255] as const
const ANSI_LIGHT_MAX_LUMINANCE = 0.72
const ANSI_LIGHT_TARGET_LUMINANCE = 0.34
const ANSI_LIGHT_MIN_SATURATION = 0.22
const ANSI_MUTED_BUCKET = 245

const ANSI_NORMALIZED_FOREGROUNDS: readonly (keyof ThemeColors)[] = [
  'text',
  'label',
  'ok',
  'error',
  'warn',
  'prompt',
  'primary',
  'accent',
  'toolName',
  'system',
  'statusFg',
  'statusGood',
  'statusWarn',
  'statusBad',
  'statusCritical',
  'shellDollar'
]

const ANSI_MUTED_FOREGROUNDS: readonly (keyof ThemeColors)[] = ['muted', 'sessionLabel', 'sessionBorder']

function xtermEightBitRgb(colorNumber: number): [number, number, number] {
  if (colorNumber >= 232) {
    const value = 8 + (colorNumber - 232) * 10
    return [value, value, value]
  }
  if (colorNumber >= 16) {
    const offset = colorNumber - 16
    return [
      XTERM_6_LEVELS[Math.floor(offset / 36) % 6]!,
      XTERM_6_LEVELS[Math.floor(offset / 6) % 6]!,
      XTERM_6_LEVELS[offset % 6]!
    ]
  }
  return [0, 0, 0]
}

function channelLuminance(value: number): number {
  const n = value / 255
  return n <= 0.03928 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4
}

function relativeLuminance(r: number, g: number, b: number): number {
  return 0.2126 * channelLuminance(r) + 0.7152 * channelLuminance(g) + 0.0722 * channelLuminance(b)
}

function rgbToHsl(red: number, green: number, blue: number): [number, number, number] {
  const rn = red / 255
  const gn = green / 255
  const bn = blue / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const lightness = (max + min) / 2
  if (max === min) {
    return [0, 0, lightness]
  }
  const delta = max - min
  const saturation = lightness > 0.5 ? delta / (2 - max - min) : delta / (max + min)
  const hue =
    max === rn ? (gn - bn) / delta + (gn < bn ? 6 : 0) : max === gn ? (bn - rn) / delta + 2 : (rn - gn) / delta + 4
  return [hue / 6, saturation, lightness]
}

function circularDistance(a: number, b: number): number {
  const d = Math.abs(a - b)
  return Math.min(d, 1 - d)
}

function richEightBitColorNumber(red: number, green: number, blue: number): number {
  const [, saturation, lightness] = rgbToHsl(red, green, blue)
  if (saturation < 0.15) {
    const gray = Math.round(lightness * 25)
    return gray === 0 ? 16 : gray === 25 ? 231 : 231 + gray
  }
  const sixRed = red < 95 ? red / 95 : 1 + (red - 95) / 40
  const sixGreen = green < 95 ? green / 95 : 1 + (green - 95) / 40
  const sixBlue = blue < 95 ? blue / 95 : 1 + (blue - 95) / 40
  return 16 + 36 * Math.round(sixRed) + 6 * Math.round(sixGreen) + Math.round(sixBlue)
}

function bestReadableAnsiColor(red: number, green: number, blue: number): number {
  const [hue, saturation, lightness] = rgbToHsl(red, green, blue)
  let bestColor = richEightBitColorNumber(red, green, blue)
  let bestScore = Number.POSITIVE_INFINITY
  for (let colorNumber = 16; colorNumber <= 255; colorNumber += 1) {
    const [cr, cg, cb] = xtermEightBitRgb(colorNumber)
    if (relativeLuminance(cr, cg, cb) > ANSI_LIGHT_MAX_LUMINANCE) {
      continue
    }
    const [ch, cs, cl] = rgbToHsl(cr, cg, cb)
    const saturationFloorPenalty = cs < ANSI_LIGHT_MIN_SATURATION ? (ANSI_LIGHT_MIN_SATURATION - cs) * 3 : 0
    const score =
      circularDistance(ch, hue) * 4 +
      Math.abs(cs - Math.max(ANSI_LIGHT_MIN_SATURATION, saturation)) * 0.8 +
      Math.abs(cl - Math.min(lightness, ANSI_LIGHT_TARGET_LUMINANCE)) * 2 +
      saturationFloorPenalty
    if (score < bestScore) {
      bestColor = colorNumber
      bestScore = score
    }
  }
  return bestColor
}

function normalizeAnsiForeground(color: string): string {
  const rgb = parseHex(color)
  if (!rgb) {
    return color
  }
  const richAnsi = richEightBitColorNumber(rgb[0], rgb[1], rgb[2])
  const richRgb = xtermEightBitRgb(richAnsi)
  const ansi =
    relativeLuminance(richRgb[0], richRgb[1], richRgb[2]) > ANSI_LIGHT_MAX_LUMINANCE
      ? bestReadableAnsiColor(rgb[0], rgb[1], rgb[2])
      : richAnsi
  return `ansi256(${ansi})`
}

// ── Persepolis Lapis defaults ───────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'Xerxes-Agents',
  icon: '♔', // the King's authority
  prompt: '❯',
  welcome: 'The court awaits your word.',
  goodbye: 'The lamps are dimmed. Until the court reconvenes.',
  tool: '┊',
  helpHeader: 'Royal decrees'
}

const cleanPromptSymbol = (s: string | undefined, fallback: string) => {
  const cleaned = String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()
  return cleaned || fallback
}

export const DARK_THEME: Theme = {
  color: {
    primary: '#4f86ff', // lapis blue
    accent: '#2fd4c4', // faience turquoise
    border: '#2a4a7f', // deep lapis rule
    text: '#dbe6f5', // cool near-white
    muted: '#7b97b5', // cool slate
    toolName: '#a9c7ff', // pale azure
    system: '#c77dff', // royal violet

    completionBg: '#0e1b2e',
    completionCurrentBg: '#1d3a5f',
    completionMetaBg: '#0e1b2e',
    completionMetaCurrentBg: '#1d3a5f',

    label: '#a9c7ff',
    ok: '#3fb950',
    error: '#e0556b', // carmine
    warn: '#f0b429', // gold leaf

    prompt: '#dbe6f5',
    sessionLabel: '#7b97b5',
    sessionBorder: '#7b97b5',

    statusBg: '#0e1b2e',
    statusFg: '#b8cae0',
    statusGood: '#3fb950',
    statusWarn: '#f0b429',
    statusBad: '#e0556b',
    statusCritical: '#ff5e57',
    selectionBg: '#1d3a5f',

    diffAdded: 'rgb(214,244,221)',
    diffRemoved: 'rgb(250,219,225)',
    diffAddedWord: 'rgb(63,185,80)',
    diffRemovedWord: 'rgb(224,85,107)',

    shellDollar: '#2fd4c4'
  },
  brand: BRAND,
  bannerLogo: '',
  bannerHero: ''
}

// Light-terminal palette: darker lapis/teal that stays legible on white.
export const LIGHT_THEME: Theme = {
  color: {
    primary: '#1f5fd6',
    accent: '#0d8f84',
    border: '#1f4a86',
    text: '#11233a',
    muted: '#476284',
    toolName: '#2b5fb0',
    system: '#7a3fc0',

    completionBg: '#eef3fb',
    completionCurrentBg: mix('#eef3fb', '#1f5fd6', 0.22),
    completionMetaBg: '#eef3fb',
    completionMetaCurrentBg: mix('#eef3fb', '#1f5fd6', 0.22),

    label: '#2b5fb0',
    ok: '#1a7f37',
    error: '#c0344b',
    warn: '#9a6a00',

    prompt: '#11233a',
    sessionLabel: '#476284',
    sessionBorder: '#476284',

    statusBg: '#eef3fb',
    statusFg: '#28384f',
    statusGood: '#1a7f37',
    statusWarn: '#9a6a00',
    statusBad: '#c0344b',
    statusCritical: '#b21f2d',
    selectionBg: '#d2e2f7',

    diffAdded: 'rgb(200,240,208)',
    diffRemoved: 'rgb(244,206,213)',
    diffAddedWord: 'rgb(26,127,55)',
    diffRemovedWord: 'rgb(178,31,45)',

    shellDollar: '#0d8f84'
  },
  brand: BRAND,
  bannerLogo: '',
  bannerHero: ''
}

// ── Light/dark detection (ported from Xerxes) ───────────────────────────

const TRUE_RE = /^(?:1|true|yes|on)$/
const FALSE_RE = /^(?:0|false|no|off)$/
const LIGHT_DEFAULT_TERM_PROGRAMS = new Set<string>(['Apple_Terminal'])
const LUMA_LIGHT_THRESHOLD = 0.6
const HEX_3_RE = /^[0-9a-f]{3}$/
const HEX_6_RE = /^[0-9a-f]{6}$/

function backgroundLuminance(raw: string): null | number {
  const v = raw.trim().toLowerCase()
  if (!v) {
    return null
  }
  const hex = v.startsWith('#') ? v.slice(1) : v
  const rgb = HEX_6_RE.test(hex)
    ? [Number.parseInt(hex.slice(0, 2), 16), Number.parseInt(hex.slice(2, 4), 16), Number.parseInt(hex.slice(4, 6), 16)]
    : HEX_3_RE.test(hex)
      ? [
          Number.parseInt(hex[0]! + hex[0]!, 16),
          Number.parseInt(hex[1]! + hex[1]!, 16),
          Number.parseInt(hex[2]! + hex[2]!, 16)
        ]
      : null
  if (!rgb) {
    return null
  }
  return (0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!) / 255
}

export function detectLightMode(
  env: NodeJS.ProcessEnv = process.env,
  lightDefaultTermPrograms: ReadonlySet<string> = LIGHT_DEFAULT_TERM_PROGRAMS
): boolean {
  const lightFlag = (env.XERXES_TUI_LIGHT ?? '').trim().toLowerCase()
  if (TRUE_RE.test(lightFlag)) {
    return true
  }
  if (FALSE_RE.test(lightFlag)) {
    return false
  }
  const themeFlag = (env.XERXES_TUI_THEME ?? '').trim().toLowerCase()
  if (themeFlag === 'light') {
    return true
  }
  if (themeFlag === 'dark') {
    return false
  }
  const bgHint = backgroundLuminance(env.XERXES_TUI_BACKGROUND ?? '')
  if (bgHint !== null) {
    return bgHint >= LUMA_LIGHT_THRESHOLD
  }
  const colorfgbg = (env.COLORFGBG ?? '').trim()
  if (colorfgbg) {
    const lastField = colorfgbg.split(';').at(-1) ?? ''
    if (/^\d+$/.test(lastField)) {
      const bg = Number(lastField)
      if (bg === 7 || bg === 15) {
        return true
      }
      if (bg >= 0 && bg < 16) {
        return false
      }
    }
  }
  const termProgram = (env.TERM_PROGRAM ?? '').trim()
  return lightDefaultTermPrograms.has(termProgram)
}

function shouldNormalizeAnsiLightTheme(env: NodeJS.ProcessEnv = process.env, isLight = detectLightMode(env)): boolean {
  const colorTerm = (env.COLORTERM ?? '').trim().toLowerCase()
  const termProgram = (env.TERM_PROGRAM ?? '').trim()
  return termProgram === 'Apple_Terminal' && colorTerm !== 'truecolor' && colorTerm !== '24bit' && isLight
}

export function normalizeThemeForAnsiLightTerminal(
  theme: Theme,
  env: NodeJS.ProcessEnv = process.env,
  isLight = detectLightMode(env)
): Theme {
  if (!shouldNormalizeAnsiLightTheme(env, isLight)) {
    return theme
  }
  const color = { ...theme.color }
  for (const key of ANSI_NORMALIZED_FOREGROUNDS) {
    color[key] = normalizeAnsiForeground(color[key])
  }
  for (const key of ANSI_MUTED_FOREGROUNDS) {
    color[key] = `ansi256(${ANSI_MUTED_BUCKET})`
  }
  return { ...theme, color }
}

const DEFAULT_LIGHT_MODE = detectLightMode()

export const DEFAULT_THEME: Theme = normalizeThemeForAnsiLightTerminal(
  DEFAULT_LIGHT_MODE ? LIGHT_THEME : DARK_THEME,
  process.env,
  DEFAULT_LIGHT_MODE
)

// ── Skin → Theme (daemon wire override; wired in a later phase) ──────────

/**
 * Merge a skin payload (skin_engine.py roles/branding) over DEFAULT_THEME.
 * `roles` uses skin_engine keys: primary/accent/warn/error/tool_name/system/
 * muted/diff_add/diff_del.
 */
export function fromSkin(
  roles: Record<string, string>,
  branding: Record<string, string> = {},
  bannerLogo = '',
  bannerHero = '',
  toolPrefix = '',
  helpHeader = ''
): Theme {
  const d = DEFAULT_THEME
  const r = (k: string) => roles[k]
  const primary = r('primary') ?? d.color.primary
  const accent = r('accent') ?? d.color.accent
  const muted = r('muted') ?? d.color.muted
  const error = r('error') ?? d.color.error
  const warn = r('warn') ?? d.color.warn
  const completionBg = d.color.completionBg

  return normalizeThemeForAnsiLightTerminal(
    {
      color: {
        primary,
        accent,
        border: r('border') ?? mix(primary, '#000000', 0.45),
        text: r('text') ?? d.color.text,
        muted,
        toolName: r('tool_name') ?? d.color.toolName,
        system: r('system') ?? d.color.system,

        completionBg,
        completionCurrentBg: mix(completionBg, primary, 0.25),
        completionMetaBg: completionBg,
        completionMetaCurrentBg: mix(completionBg, primary, 0.25),

        label: r('tool_name') ?? d.color.label,
        ok: r('diff_add') ?? d.color.ok,
        error,
        warn,

        prompt: r('text') ?? d.color.prompt,
        sessionLabel: muted,
        sessionBorder: muted,

        statusBg: d.color.statusBg,
        statusFg: d.color.statusFg,
        statusGood: r('diff_add') ?? d.color.statusGood,
        statusWarn: warn,
        statusBad: error,
        statusCritical: d.color.statusCritical,
        selectionBg: mix(completionBg, primary, 0.3),

        diffAdded: d.color.diffAdded,
        diffRemoved: d.color.diffRemoved,
        diffAddedWord: d.color.diffAddedWord,
        diffRemovedWord: d.color.diffRemovedWord,

        shellDollar: accent
      },
      brand: {
        name: branding.agent_name ?? d.brand.name,
        icon: d.brand.icon,
        prompt: cleanPromptSymbol(branding.prompt_symbol, d.brand.prompt),
        welcome: branding.welcome ?? d.brand.welcome,
        goodbye: branding.goodbye ?? d.brand.goodbye,
        tool: cleanPromptSymbol(toolPrefix || branding.tool_prefix, d.brand.tool),
        helpHeader: helpHeader || branding.help_header || d.brand.helpHeader
      },
      bannerLogo,
      bannerHero
    },
    process.env,
    DEFAULT_LIGHT_MODE
  )
}
