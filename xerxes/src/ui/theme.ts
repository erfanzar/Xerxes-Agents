// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Theme + palette for the Xerxes TUI. The shipped palette is "Night
// Standard": neutral graphite surfaces, high-contrast text, and one brand
// signal that keeps the keyboard-first interface readable.
//
// The v2 "Persepolis" retune shifted the dark surfaces a touch blue (#0f0f12
// panel over #1d1d24 elevated), cooled the hairline to #32323b, and added a
// warm user-band tint so input reads as input.
//
// The v3 pivot moves the brand family from amber to lapis blue: every role
// that was gold/accent (header mark, active tab, composer ring, user band,
// turn rail, leader dots) is now the blue family, matching the tool voice's
// lapis. `warn` keeps its amber VALUE purely as a semantic warning colour —
// it no longer carries any brand duty.
//
// The colour math handles light/dark detection plus ANSI down-conversion for
// terminals without truecolor. `fromSkin` lets the daemon push live skin
// overrides later.

// ── Derafsh Kaviani emblem palette ──────────────────────────────────────
//
// The boot mark animates a blue → purple → deep-blue gradient. These are the
// source constants for it, owned here so the transcript can reuse the same
// hues: the emblem is meant to read as a legend for the voices below it.
// `banner.ts` imports them rather than keeping a private copy. (The former
// *_GOLD constants are named *_AZURE since the v3 pivot.)
export const DARK_DERAFSH_BLUE = '#6ea8fe'
export const DARK_DERAFSH_PURPLE = '#b18be8'
export const DARK_DERAFSH_AZURE = '#4a82d8'
export const DARK_DERAFSH_BRIDGE = '#9fb8d8'
export const LIGHT_DERAFSH_BLUE = '#1f64b5'
export const LIGHT_DERAFSH_PURPLE = '#7047b5'
export const LIGHT_DERAFSH_AZURE = '#275e9e'
export const LIGHT_DERAFSH_BRIDGE = '#466c91'

// ── Nocturne: the Xerxes terminal design system ─────────────────────────
//
// Source of truth: the "Xerxes Terminal UI Design" canvas, screen 10. The
// whole vocabulary is six voice colours, six surfaces, three edges and an
// eight-step text ramp — every screen is assembled from four row patterns
// (leader row, caption, card, footer) using nothing else.
//
// Two rules the canvas is emphatic about, and which the roles below encode:
//
//  1. Each voice colour owns exactly ONE meaning on every screen. Amber is
//     never emphasis — it means "a human is required", so a screen where
//     nothing is blocked carries no amber at all. Cyan appears only on hunk
//     headers, so a fold boundary is never read as a state dot.
//  2. The text ramp does the work colour usually does. Eight greys assigned
//     by role mean a screen can be fully legible while only two or three
//     things are actually coloured.
//
// These are raw tokens. Screens read them through the `ThemeColors` roles
// below, never directly, so a skin can repaint the product.
export interface NocturnePalette {
  /** Voice — same dot, same meaning, every screen. */
  working: string
  done: string
  failed: string
  needsInput: string
  activity: string
  structure: string

  /** Softened voice text, for prose that carries a state colour. */
  needsInputText: string
  failedText: string

  /** Surfaces — layered, never gradient. */
  backdrop: string
  sunken: string
  screen: string
  chrome: string
  card: string
  selected: string

  /** Edges. */
  hairline: string
  divider: string
  focusEdge: string

  /** Text ramp — quiet by default. */
  strong: string
  title: string
  prose: string
  secondary: string
  meta: string
  numeric: string
  caption: string
  separator: string
  /** The statusbar's `│` section break — quieter than any separator. */
  rule: string

  /** Dotted leaders: the loud one on active rows, the quiet one elsewhere. */
  leader: string
  leaderQuiet: string

  /** Tinted card grounds, one per state. */
  workingCardBg: string
  needsInputCardBg: string
  needsInputCardBorder: string
  doneCardBg: string
  doneCardBorder: string
  failedCardBg: string
  failedCardBorder: string

  /** Diff surfaces: context text, row tints, word-level tints. */
  diffContext: string
  diffAddRow: string
  diffDelRow: string
  diffAddWordBg: string
  diffDelWordBg: string
  diffAddWordFg: string
  diffDelWordFg: string
  diffHunkBg: string
  diffFoldBg: string
}

export const NOCTURNE_DARK: NocturnePalette = {
  working: '#6ea8fe',
  done: '#57ca85',
  failed: '#f47067',
  needsInput: '#d8ae58',
  activity: '#b39cf0',
  structure: '#56c8d8',

  needsInputText: '#e4c37e',
  failedText: '#e39a94',

  backdrop: '#000000',
  sunken: '#08080c',
  screen: '#0a0a0d',
  chrome: '#0d0d12',
  card: '#101017',
  selected: '#101725',

  hairline: '#1c2733',
  divider: '#161d27',
  focusEdge: '#223244',

  strong: '#e9e9ed',
  title: '#d7dce3',
  prose: '#c8cfd8',
  secondary: '#8b949e',
  meta: '#6b7280',
  numeric: '#565f6b',
  caption: '#4d5561',
  separator: '#3f4753',
  rule: '#26303c',

  leader: '#232b36',
  leaderQuiet: '#1e252e',

  workingCardBg: '#0d1017',
  needsInputCardBg: '#12100c',
  needsInputCardBorder: '#3a3018',
  doneCardBg: '#0c1210',
  doneCardBorder: '#1b2b22',
  failedCardBg: '#120d0d',
  failedCardBorder: '#3a1f1d',

  diffContext: '#7d8794',
  // The canvas paints these as 7% alpha over the screen ground; a terminal
  // cell has no alpha, so they are pre-composited here against #0a0a0d.
  diffAddRow: '#0d1512',
  diffDelRow: '#150e0e',
  diffAddWordBg: '#1d3628',
  diffDelWordBg: '#33191a',
  diffAddWordFg: '#b7f0cd',
  diffDelWordFg: '#ffc3bd',
  diffHunkBg: '#0b1214',
  diffFoldBg: '#0c0c11'
}

/**
 * The same vocabulary on a light terminal.
 *
 * Roles are preserved, not hues: the ramp inverts (strong is darkest), the
 * surfaces climb rather than fall, and each voice colour is darkened until it
 * clears 4.5:1 on the paper ground. A light terminal is not a dark one with
 * the background swapped, so the tinted card grounds are near-white washes
 * rather than the dark theme's near-black ones.
 */
export const NOCTURNE_LIGHT: NocturnePalette = {
  working: '#1f64b5',
  done: '#197a4f',
  failed: '#b4233f',
  needsInput: '#8a5b00',
  activity: '#7047b5',
  structure: '#0e7490',

  needsInputText: '#6d4700',
  failedText: '#8f1c33',

  backdrop: '#ffffff',
  sunken: '#eef1f6',
  screen: '#f7f8fa',
  chrome: '#f1f3f5',
  card: '#eef1f6',
  selected: '#e3ecf9',

  hairline: '#c8d1dc',
  divider: '#dbe2ea',
  focusEdge: '#9dbbe4',

  strong: '#0d1620',
  title: '#172533',
  prose: '#22303e',
  secondary: '#3f5162',
  meta: '#526579',
  numeric: '#647486',
  caption: '#73869a',
  separator: '#93a3b3',
  rule: '#b6c2ce',

  leader: '#cdd6e0',
  leaderQuiet: '#dde3ea',

  workingCardBg: '#eff4fb',
  needsInputCardBg: '#faf3e2',
  needsInputCardBorder: '#e0c98e',
  doneCardBg: '#eaf6ef',
  doneCardBorder: '#a9d6bd',
  failedCardBg: '#fbedef',
  failedCardBorder: '#e5b3bd',

  diffContext: '#4c5c6c',
  diffAddRow: '#eaf6ee',
  diffDelRow: '#fceef0',
  diffAddWordBg: '#c7ecd5',
  diffDelWordBg: '#f7ced6',
  diffAddWordFg: '#0d5233',
  diffDelWordFg: '#7c1329',
  diffHunkBg: '#e6f2f5',
  diffFoldBg: '#eceff3'
}

export interface ThemeColors {
  primary: string
  accent: string
  border: string
  text: string
  muted: string

  // Xerxes three-voice roles from the daemon skin payload.
  toolName: string
  system: string

  // Transcript voices. Deliberately outside `themeForMode`'s override list:
  // interaction mode answers "what am I doing", voice answers "who is
  // speaking". Smearing the mode accent over these is what flattened the
  // whole transcript to one gray.
  userBar: string
  userText: string
  thinking: string
  /**
   * The turn rail. Dimmer than `userBar` on purpose: the rail is painted as
   * a filled column so it can span a whole answer, and a filled cell inks
   * ~10x more than the `│` glyph it replaces. At full saturation a long
   * reply would be a solid gold stripe.
   */
  turnRail: string

  // Brand signal for chrome (panel frames, titles, the selected completion).
  // Same reason as above — these must survive every interaction mode.
  brandGold: string
  brandLapis: string

  completionBg: string
  completionCurrentBg: string
  completionMetaBg: string
  completionMetaCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

  /**
   * Warm surface behind the user transcript band. The gold bar marks whose
   * row it is; this tint makes the whole band read as "input" without
   * bordering or boxing it.
   */
  userBandBg: string

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
  /** Row backgrounds behind +/- lines in the F7 diff viewer. */
  diffAddedBg: string
  diffRemovedBg: string
  /**
   * Word-level backgrounds. Distinct from the row tints on purpose: the row
   * says the line changed, the word says WHICH substring moved, and a line
   * tint alone makes you re-read the whole line to find one renamed symbol.
   */
  diffAddedWordBg: string
  diffRemovedWordBg: string
  /** Hunk-header accent in the F7 diff viewer. */
  diffHunk: string

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
  /**
   * The raw Nocturne tokens `color` is derived from.
   *
   * `color` names the roles the product had before the design system existed
   * (accent, muted, ok…). `ds` carries the ones it has no name for: the six
   * surfaces, the eight-step text ramp, the dotted leaders and the tinted
   * state cards. Screens read `ds` for those and `color` for everything else,
   * so nothing has to hard-code a hex.
   */
  ds: NocturnePalette
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
  'userBar',
  'userText',
  'brandGold',
  'brandLapis',
  'statusFg',
  'statusGood',
  'statusWarn',
  'statusBad',
  'statusCritical',
  'shellDollar'
]

const ANSI_MUTED_FOREGROUNDS: readonly (keyof ThemeColors)[] = ['muted', 'sessionLabel', 'sessionBorder', 'thinking']

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
  // The tagline under the wordmark. It says what the product IS, because
  // the home screen's job is to answer "where am I" before it offers a way
  // in — 'Ready for your next command.' answered a question nobody asked on
  // a screen where nothing has happened yet.
  welcome: 'Many agents, one terminal.',
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

/**
 * Build the shipped palette for one ground from the Nocturne tokens.
 *
 * Every role below is a token, never a literal. That is the whole point of
 * the design system: the six voice colours and the eight-step ramp are
 * decided once, on screen 10 of the canvas, and each product role says which
 * step it borrows rather than re-picking a hex near it.
 */
function nocturneTheme(ds: NocturnePalette): Theme {
  return {
    color: {
      // Markdown's h1 rides `accent` and h2/h3 + bold ride `primary`, so
      // `primary` takes the ramp's brightest step — the one the canvas
      // reserves for "the thing you must read first".
      primary: ds.strong,
      accent: ds.working,
      border: ds.hairline,
      text: ds.title,
      muted: ds.meta,
      // The tool VERB, not the tool glyph. The canvas keeps verbs on the
      // secondary step and lets the coloured ⏺ carry the state, so a column
      // of tool rows reads as one quiet block with a few lit dots.
      toolName: ds.secondary,
      system: ds.activity,

      userBar: ds.working,
      userText: ds.title,
      turnRail: ds.focusEdge,
      thinking: ds.meta,

      brandGold: ds.working,
      brandLapis: ds.working,

      completionBg: ds.chrome,
      completionCurrentBg: ds.selected,
      completionMetaBg: ds.chrome,
      completionMetaCurrentBg: ds.selected,

      label: ds.secondary,
      ok: ds.done,
      error: ds.failed,
      warn: ds.needsInput,

      userBandBg: ds.workingCardBg,
      prompt: ds.strong,
      sessionLabel: ds.caption,
      sessionBorder: ds.hairline,

      statusBg: ds.screen,
      statusFg: ds.title,
      statusGood: ds.done,
      statusWarn: ds.needsInput,
      statusBad: ds.failed,
      // Deliberately the same red. The canvas allows six colours and each
      // owns one meaning; a seventh hue for "worse than bad" would be a hue
      // chosen because it looks urgent, not because it means something new.
      statusCritical: ds.failed,
      selectionBg: ds.selected,

      // Changed lines keep the prose step and let the +/- sign and the row
      // tint carry the state, so a diff reads as code with marks on it
      // rather than as two blocks of coloured text.
      diffAdded: ds.prose,
      diffRemoved: ds.prose,
      diffAddedWord: ds.diffAddWordFg,
      diffRemovedWord: ds.diffDelWordFg,
      diffAddedBg: ds.diffAddRow,
      diffRemovedBg: ds.diffDelRow,
      diffAddedWordBg: ds.diffAddWordBg,
      diffRemovedWordBg: ds.diffDelWordBg,
      diffHunk: ds.structure,

      shellDollar: ds.working
    },
    ds,
    brand: BRAND,
    bannerLogo: '',
    bannerHero: ''
  }
}

export const DARK_THEME: Theme = nocturneTheme(NOCTURNE_DARK)

/** The same roles on paper: see `NOCTURNE_LIGHT` for how the ramp inverts. */
export const LIGHT_THEME: Theme = nocturneTheme(NOCTURNE_LIGHT)

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

/**
 * The user's explicit palette choice, or null when nothing was pinned.
 *
 * Only the XERXES_TUI_* variables count as explicit: they are set deliberately
 * and must outrank the terminal's own answer. COLORFGBG and TERM_PROGRAM are
 * guesses, so the live probe is allowed to overrule them.
 */
export function explicitLightMode(env: NodeJS.ProcessEnv = process.env): boolean | null {
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
  return null
}

export function detectLightMode(
  env: NodeJS.ProcessEnv = process.env,
  lightDefaultTermPrograms: ReadonlySet<string> = LIGHT_DEFAULT_TERM_PROGRAMS
): boolean {
  const pinned = explicitLightMode(env)
  if (pinned !== null) {
    return pinned
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

const themeForLightMode = (isLight: boolean, env: NodeJS.ProcessEnv = process.env): Theme =>
  normalizeThemeForAnsiLightTerminal(isLight ? LIGHT_THEME : DARK_THEME, env, isLight)

const DEFAULT_LIGHT_MODE = detectLightMode()

/**
 * Pre-probe seed. The environment can only be sniffed, so this is the palette
 * used for the frames rendered before the terminal answers the OSC background
 * query; `subscribeTerminalThemeMode` replaces it with the real answer.
 */
export const DEFAULT_THEME: Theme = themeForLightMode(DEFAULT_LIGHT_MODE)

export type TerminalThemeMode = 'dark' | 'light'

/**
 * The renderer's live theme-mode surface (structurally satisfied by OpenTUI's
 * `CliRenderer`), kept minimal so this module never imports the renderer.
 */
export interface TerminalThemeModeSource {
  on(event: 'theme_mode', listener: (mode: TerminalThemeMode) => void): unknown
  off(event: 'theme_mode', listener: (mode: TerminalThemeMode) => void): unknown
  readonly themeMode: TerminalThemeMode | null
  waitForThemeMode(timeoutMs?: number): Promise<TerminalThemeMode | null>
}

const THEME_PROBE_TIMEOUT_MS = 250

let liveLightMode = DEFAULT_LIGHT_MODE
let liveBaseTheme = DEFAULT_THEME

/** The palette every later derivation (skins, mode overlays) builds on. */
export const currentBaseTheme = (): Theme => liveBaseTheme

export const currentLightMode = (): boolean => liveLightMode

/**
 * Adopt a terminal-reported light/dark mode. Returns the new base theme when
 * the mode actually flipped, and null when nothing changed — so callers never
 * push a redundant re-render.
 */
export function applyTerminalThemeMode(mode: TerminalThemeMode, env: NodeJS.ProcessEnv = process.env): null | Theme {
  const isLight = mode === 'light'

  if (isLight === liveLightMode) {
    return null
  }

  liveLightMode = isLight
  liveBaseTheme = themeForLightMode(isLight, env)

  return liveBaseTheme
}

/** Re-seed from the environment. Exists so tests start from a known palette. */
export function resetTerminalThemeMode(env: NodeJS.ProcessEnv = process.env): Theme {
  liveLightMode = detectLightMode(env)
  liveBaseTheme = themeForLightMode(liveLightMode, env)

  return liveBaseTheme
}

/**
 * Consume the renderer's probed background colour and keep following it.
 *
 * Without this the palette is whatever the environment implied at module load,
 * frozen for the process: switching the terminal to a light profile mid-session
 * leaves dark-on-light text unreadable until restart. `apply` receives the new
 * base theme; the returned function unsubscribes.
 */
export function subscribeTerminalThemeMode(
  source: TerminalThemeModeSource,
  apply: (theme: Theme) => void,
  { env = process.env, probeTimeoutMs = THEME_PROBE_TIMEOUT_MS }: TerminalThemeModeOptions = {}
): () => void {
  // An explicitly pinned palette is a user decision; the probe must not fight it.
  if (explicitLightMode(env) !== null) {
    return () => void 0
  }

  const adopt = (mode: null | TerminalThemeMode) => {
    if (!mode) {
      return
    }

    const next = applyTerminalThemeMode(mode, env)

    if (next) {
      apply(next)
    }
  }

  const onThemeMode = (mode: TerminalThemeMode) => adopt(mode)

  source.on('theme_mode', onThemeMode)

  const probed = source.themeMode

  if (probed) {
    adopt(probed)
  } else {
    // The query may still be in flight when the app mounts; adopting late is
    // fine because `applyTerminalThemeMode` is idempotent per mode.
    void Promise.resolve(source.waitForThemeMode(probeTimeoutMs)).then(adopt, () => void 0)
  }

  return () => {
    source.off('theme_mode', onThemeMode)
  }
}

export interface TerminalThemeModeOptions {
  env?: NodeJS.ProcessEnv
  probeTimeoutMs?: number
}

export type InteractionPaletteMode = 'code' | 'objective' | 'plan' | 'researcher'

/**
 * Interaction mode reads off the same six voice colours as everything else —
 * researcher borrows `working`, plan borrows `structure`, objective borrows
 * `activity`. Code is deliberately un-hued: it sits between the ramp's title
 * and secondary steps, so the default mode adds no colour to the screen at
 * all.
 */
const modeAccents = (ds: NocturnePalette): Record<InteractionPaletteMode, string> => ({
  code: mix(ds.title, ds.secondary, 0.5),
  researcher: ds.working,
  plan: ds.structure,
  objective: ds.activity
})

const interactionPaletteMode = (mode?: string): InteractionPaletteMode =>
  mode === 'researcher' || mode === 'plan' || mode === 'objective' ? mode : 'code'

/**
 * Overlay the interaction mode's visual identity without mutating the base
 * skin. Code is deliberately neutral gray; researcher, plan, and objective
 * use blue, teal, and purple respectively. Semantic colors (success, warning,
 * error) stay stable across mode changes.
 *
 * Two roles are deliberately NOT overridden here, and adding them back would
 * undo the whole voice system:
 *
 * - `primary` carries markdown's h2/h3 and bold, while `accent` carries h1.
 *   Setting both to the mode accent collapsed three heading levels into one
 *   flat gray in `code` mode.
 * - `label` is a panel-header role; the user transcript band reads `userText`.
 *
 * The voice roles (`userBar`, `userText`, `toolName`, `system`, `thinking`)
 * and the brand roles (`brandGold`, `brandLapis`) must never be listed below.
 * Mode identity lives in `accent`, `border`, and the surfaces.
 */
export function themeForMode(theme: Theme, mode?: string): Theme {
  const ds = theme.ds
  const light = (backgroundLuminance(ds.screen) ?? 0) >= LUMA_LIGHT_THRESHOLD
  const paletteMode = interactionPaletteMode(mode)
  const accent = modeAccents(ds)[paletteMode]
  // Mode identity is a tint ON the Nocturne surfaces, never a replacement for
  // them: the screen stays the screen, the selected row stays the selected
  // row, and the mode only says how much of the accent they carry.
  const activeSurface = paletteMode === 'code' ? ds.selected : mix(ds.selected, accent, light ? 0.1 : 0.12)
  const border = mix(ds.hairline, accent, light ? 0.34 : 0.3)
  const selection = mix(activeSurface, accent, light ? 0.14 : 0.18)

  return {
    ...theme,
    color: {
      ...theme.color,
      accent,
      border,
      completionBg: ds.chrome,
      completionCurrentBg: activeSurface,
      completionMetaBg: ds.chrome,
      completionMetaCurrentBg: activeSurface,
      sessionBorder: border,
      statusBg: ds.screen,
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
  // Live base, not the boot seed: a skin merged after a theme flip must inherit
  // the palette the terminal is actually showing.
  const d = currentBaseTheme()
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

        // Voice + brand roles are derived rather than given their own wire
        // keys: `ROLE_NAMES` is a daemon contract and should not grow for a
        // client-side concern. Brand roles follow the skin's `accent` (NOT
        // `warn` — amber is a semantic warning colour and must never repaint
        // the chrome). Deriving also keeps the `mono` skin genuinely
        // monochrome for free — it supplies gray roles, so the voices go gray
        // with it, which is exactly what that accessibility skin promises.
        userBar: r('accent') ?? d.color.userBar,
        turnRail: r('border') ?? d.color.turnRail,
        userText: r('text') ?? d.color.userText,
        thinking: mix(muted, r('system') ?? d.color.system, 0.5),

        brandGold: r('accent') ?? d.color.brandGold,
        brandLapis: r('system') ?? d.color.brandLapis,

        completionBg,
        completionCurrentBg: mix(completionBg, primary, 0.25),
        completionMetaBg: completionBg,
        completionMetaCurrentBg: mix(completionBg, primary, 0.25),

        label: r('tool_name') ?? d.color.label,
        ok: r('diff_add') ?? d.color.ok,
        error,
        warn,

        // Derive the cool band tint from whatever the skin says the accent
        // is, so a re-branded skin keeps its own tint instead of Xerxes' blue.
        userBandBg: mix(d.color.statusBg, accent, 0.09),
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
        diffAddedBg: d.color.diffAddedBg,
        diffRemovedBg: d.color.diffRemovedBg,
        diffAddedWordBg: d.color.diffAddedWordBg,
        diffRemovedWordBg: d.color.diffRemovedWordBg,
        diffHunk: d.color.diffHunk,

        shellDollar: accent
      },
      // Surfaces, the ramp and the leaders are structural — a skin recolours
      // the product, it does not restack it — so only the voice colours
      // follow the wire roles. That is also what keeps the `mono` skin
      // genuinely monochrome for free: it supplies gray roles, so the voices
      // go gray with it.
      ds: {
        ...d.ds,
        working: accent,
        done: r('diff_add') ?? d.ds.done,
        failed: error,
        needsInput: warn,
        activity: r('system') ?? d.ds.activity
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
    currentLightMode()
  )
}
