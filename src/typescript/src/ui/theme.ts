// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Theme + palette for the Xerxes TUI. The shipped palette is "Night
// Standard": neutral graphite surfaces, an amber brand signal, and
// high-contrast text that keeps the keyboard-first interface readable.
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

  // Xerxes three-voice roles from the daemon skin payload.
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

// ── Night Standard defaults ─────────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'XERXES',
  prompt: '❯',
  welcome: 'Ready for your next command.',
  goodbye: 'Session closed.',
  tool: '│',
  helpHeader: 'Keyboard'
}

const cleanPromptSymbol = (s: string | undefined, fallback: string) => {
  const cleaned = String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()
  return cleaned || fallback
}

export const DARK_THEME: Theme = {
  color: {
    primary: '#e6e6e6',
    accent: '#d8ae58',
    border: '#333333',
    text: '#e9e9e9',
    muted: '#737373',
    toolName: '#c9c9c9',
    system: '#a98ad4',

    completionBg: '#111111',
    completionCurrentBg: '#1a1a1a',
    completionMetaBg: '#111111',
    completionMetaCurrentBg: '#1a1a1a',

    label: '#c9c9c9',
    ok: '#83c99d',
    error: '#dd7c88',
    warn: '#d8ae58',

    prompt: '#f4f4f4',
    sessionLabel: '#858585',
    sessionBorder: '#595959',

    statusBg: '#101010',
    statusFg: '#d6d6d6',
    statusGood: '#83c99d',
    statusWarn: '#d8ae58',
    statusBad: '#dd7c88',
    statusCritical: '#e35d6e',
    selectionBg: '#2a2a2a',

    diffAdded: 'rgb(190,232,204)',
    diffRemoved: 'rgb(245,202,210)',
    diffAddedWord: 'rgb(131,201,157)',
    diffRemovedWord: 'rgb(221,124,136)',

    shellDollar: '#d8ae58'
  },
  brand: BRAND,
  bannerLogo: '',
  bannerHero: ''
}

// Light-terminal palette: the same hierarchy with darker foreground roles.
export const LIGHT_THEME: Theme = {
  color: {
    primary: '#172533',
    accent: '#006f94',
    border: '#92a4b7',
    text: '#172533',
    muted: '#526579',
    toolName: '#31526f',
    system: '#6b46b5',

    completionBg: '#f4f7fb',
    completionCurrentBg: mix('#f4f7fb', '#006f94', 0.18),
    completionMetaBg: '#f4f7fb',
    completionMetaCurrentBg: mix('#f4f7fb', '#006f94', 0.18),

    label: '#31526f',
    ok: '#197a4f',
    error: '#b4233f',
    warn: '#825a00',

    prompt: '#172533',
    sessionLabel: '#526579',
    sessionBorder: '#73869a',

    statusBg: '#f4f7fb',
    statusFg: '#26384c',
    statusGood: '#197a4f',
    statusWarn: '#825a00',
    statusBad: '#b4233f',
    statusCritical: '#9f1239',
    selectionBg: '#dbeaf3',

    diffAdded: 'rgb(187,231,199)',
    diffRemoved: 'rgb(250,207,216)',
    diffAddedWord: 'rgb(25,122,79)',
    diffRemovedWord: 'rgb(180,35,63)',

    shellDollar: '#006f94'
  },
  brand: BRAND,
  bannerLogo: '',
  bannerHero: ''
}

// ── Light/dark detection (ported from Xerxes) ───────────────────────────

const TRUE_RE = /^(?:1|true|yes|on)$/
const FALSE_RE = /^(?:0|false|no|off)$/
// Previously defaulted Apple_Terminal to light mode on the assumption its
// stock profile is white-on-black-text. That's wrong for any dark Apple
// Terminal profile (e.g. the built-in "Pro" profile, a common choice) —
// confirmed by a real repro: LIGHT_THEME's dark-navy/dark-amber text
// (meant for a white background) rendering nearly invisible against an
// actual black background. No TERM_PROGRAM value gets a light default now;
// explicit XERXES_TUI_LIGHT/XERXES_TUI_THEME/XERXES_TUI_BACKGROUND/COLORFGBG
// still override for anyone who genuinely runs a light terminal.
const LIGHT_DEFAULT_TERM_PROGRAMS = new Set<string>([])
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

export type InteractionPaletteMode = 'code' | 'objective' | 'plan' | 'researcher'

const DARK_MODE_ACCENTS: Record<InteractionPaletteMode, string> = {
  code: '#aeb4bb',
  researcher: '#6ea8fe',
  plan: '#d8ae58',
  objective: '#b18be8'
}

const LIGHT_MODE_ACCENTS: Record<InteractionPaletteMode, string> = {
  code: '#45515e',
  researcher: '#1f64b5',
  plan: '#8a6200',
  objective: '#7047b5'
}

const interactionPaletteMode = (mode?: string): InteractionPaletteMode =>
  mode === 'researcher' || mode === 'plan' || mode === 'objective' ? mode : 'code'

/**
 * Overlay the interaction mode's visual identity without mutating the base
 * skin. Code is deliberately neutral gray; researcher, plan, and objective
 * use blue, gold, and purple respectively. Semantic colors (success, warning,
 * error, and the amber Derafsh brand) stay stable across mode changes.
 */
export function themeForMode(theme: Theme, mode?: string): Theme {
  const light = (backgroundLuminance(theme.color.statusBg) ?? 0) >= LUMA_LIGHT_THRESHOLD
  const paletteMode = interactionPaletteMode(mode)
  const accent = (light ? LIGHT_MODE_ACCENTS : DARK_MODE_ACCENTS)[paletteMode]
  const backgroundSurface = light ? '#f7f8fa' : '#101010'
  const panelSurface = light ? '#f1f3f5' : '#111111'
  const elementSurface = light ? '#e6e9ed' : '#1a1a1a'
  const activeSurface = paletteMode === 'code' ? elementSurface : mix(elementSurface, accent, light ? 0.1 : 0.12)
  const border = mix(panelSurface, accent, light ? 0.34 : 0.3)
  const selection = mix(activeSurface, accent, light ? 0.14 : 0.18)

  return {
    ...theme,
    color: {
      ...theme.color,
      primary: accent,
      accent,
      border,
      completionBg: panelSurface,
      completionCurrentBg: activeSurface,
      completionMetaBg: panelSurface,
      completionMetaCurrentBg: activeSurface,
      label: accent,
      sessionBorder: border,
      statusBg: backgroundSurface,
      selectionBg: selection,
      shellDollar: accent
    }
  }
}

// ── Skin → Theme daemon wire override ───────────────────────────────────

/**
 * Merge a daemon skin payload over DEFAULT_THEME.
 * `roles` uses the wire keys: primary/accent/warn/error/tool_name/system/
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
