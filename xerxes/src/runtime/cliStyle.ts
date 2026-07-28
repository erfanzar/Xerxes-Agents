// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Presentation layer for the non-TUI command line (`xerxes doctor`, `update`,
// `--help`, usage errors). The TUI owns its own renderer; this is for the plain
// commands, which until now emitted undifferentiated `console.log` text.
//
// Two rules shape everything here:
//
//   1. **Colour is a decoration, never information.** Every status also carries a
//      glyph and a word, so a run with colour stripped loses nothing. A reader on
//      a monochrome terminal, a screen reader, or a CI log gets the same content.
//   2. **Non-interactive output stays plain.** `NO_COLOR`, a non-TTY stdout, a
//      CI environment, or `TERM=dumb` all disable styling. This is not politeness:
//      the update command's output is asserted on by tests and read by scripts, so
//      escape sequences leaking into a pipe would be a correctness bug.
//
// The palette is the TUI's Persepolis Lapis so the two surfaces look related. It
// is duplicated rather than imported because the TUI bundle compiles standalone
// under `rootDir: src/ui`; see `core/hostPlatform.ts` for the same constraint.

/** Depth of colour a terminal can render. */
export type ColorDepth = 'none' | 'ansi16' | 'ansi256' | 'truecolor'

export interface StyleEnvironment {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly isTTY?: boolean
}

/** Persepolis Lapis, matching `ui/lib/skinEngine.ts`. */
const PALETTE = Object.freeze({
  primary: [0x4f, 0x86, 0xff],
  accent: [0x2f, 0xd4, 0xc4],
  warn: [0xf0, 0xb4, 0x29],
  error: [0xe0, 0x55, 0x6b],
  ok: [0x3f, 0xb9, 0x50],
  muted: [0x7b, 0x97, 0xb5],
  heading: [0xa9, 0xc7, 0xff],
}) satisfies Readonly<Record<string, readonly [number, number, number]>>

export type PaletteRole = keyof typeof PALETTE

/** Nearest basic-ANSI foreground for each role, used at `ansi16`. */
const ANSI16: Readonly<Record<PaletteRole, number>> = Object.freeze({
  primary: 94,
  accent: 96,
  warn: 93,
  error: 91,
  ok: 92,
  muted: 90,
  heading: 96,
})

const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on'])

/**
 * Decide how much colour to emit.
 *
 * Order matters: `NO_COLOR` wins over everything (it is a user's explicit opt-out,
 * and the convention is that mere presence counts regardless of value), then an
 * explicit `FORCE_COLOR`, then terminal capability. A non-TTY is plain even on a
 * capable terminal, because the consumer is a pipe.
 */
export function detectColorDepth(source: StyleEnvironment = {}): ColorDepth {
  const env = source.env ?? process.env
  const isTTY = source.isTTY ?? Boolean(process.stdout.isTTY)

  if ('NO_COLOR' in env) return 'none'

  const forced = env.FORCE_COLOR?.trim()
  if (forced !== undefined && forced !== '') {
    if (forced === '0' || forced === 'false') return 'none'
    if (forced === '3') return 'truecolor'
    if (forced === '2') return 'ansi256'
    return 'ansi16'
  }

  const term = env.TERM?.trim().toLowerCase() ?? ''
  if (term === 'dumb') return 'none'
  if (!isTTY) return 'none'

  const colorTerm = env.COLORTERM?.trim().toLowerCase() ?? ''
  if (colorTerm === 'truecolor' || colorTerm === '24bit') return 'truecolor'
  if (term.includes('256')) return 'ansi256'
  if (term === '') return 'none'
  return 'ansi16'
}

/** Downsample a 24-bit colour to the xterm 256-colour cube. */
function to256(rgb: readonly [number, number, number]): number {
  const [red, green, blue] = rgb
  // The 6x6x6 colour cube starts at index 16; each axis is quantized to 0..5.
  const axis = (value: number): number => Math.round((value / 255) * 5)
  return 16 + 36 * axis(red) + 6 * axis(green) + axis(blue)
}

export interface CliStyle {
  readonly depth: ColorDepth
  /** True when any escape sequence is emitted at all. */
  readonly enabled: boolean
  bold(text: string): string
  color(role: PaletteRole, text: string): string
  dim(text: string): string
}

/** Build a styler for a given colour depth. */
export function createCliStyle(depth: ColorDepth = detectColorDepth()): CliStyle {
  if (depth === 'none') {
    const identity = (text: string): string => text
    return { depth, enabled: false, bold: identity, dim: identity, color: (_role, text) => text }
  }
  // Each wrapper closes only the attribute it opened — 22 for bold/dim, 39 for
  // foreground — rather than the blanket `0m` reset. With a blanket reset,
  // `bold('a' + color('ok', 'b') + 'c')` would lose its bold at 'c', because the
  // inner colour's reset clears every attribute on its way out. Spelling the CSI
  // out avoids a literal escape byte in source, which editors strip silently.
  const wrap = (open: string, close: string, text: string): string =>
    text.length === 0 ? text : `\u001B[${open}m${text}\u001B[${close}m`
  return {
    depth,
    enabled: true,
    bold: text => wrap('1', '22', text),
    dim: text => wrap('2', '22', text),
    color: (role, text) => {
      if (depth === 'ansi16') return wrap(String(ANSI16[role]), '39', text)
      if (depth === 'ansi256') return wrap(`38;5;${to256(PALETTE[role])}`, '39', text)
      const [red, green, blue] = PALETTE[role]
      return wrap(`38;2;${red};${green};${blue}`, '39', text)
    },
  }
}

export type StatusKind = 'fail' | 'info' | 'ok' | 'pending' | 'warn'

/**
 * Glyph and colour per status.
 *
 * ASCII-safe alternates are used when the terminal cannot be trusted with the
 * nicer glyphs — a box-drawing character rendered as a replacement box is worse
 * than a plain `-`.
 */
const STATUS: Readonly<Record<StatusKind, { readonly ascii: string; readonly glyph: string; readonly role: PaletteRole }>> =
  Object.freeze({
    ok: { glyph: '✓', ascii: 'ok', role: 'ok' },
    warn: { glyph: '!', ascii: '!', role: 'warn' },
    fail: { glyph: '✗', ascii: 'x', role: 'error' },
    info: { glyph: '•', ascii: '-', role: 'primary' },
    pending: { glyph: '…', ascii: '...', role: 'muted' },
  })

/**
 * Whether the terminal can be trusted to render non-ASCII glyphs.
 *
 * A mis-rendered glyph shifts every column after it, so the check is
 * conservative: it requires an explicitly UTF-8 locale rather than assuming one.
 */
export function supportsUnicode(source: StyleEnvironment = {}): boolean {
  const env = source.env ?? process.env
  if (TRUE_VALUES.has(env.XERXES_CLI_ASCII?.trim().toLowerCase() ?? '')) return false
  const locale = `${env.LC_ALL ?? ''}${env.LC_CTYPE ?? ''}${env.LANG ?? ''}`.toLowerCase()
  if (locale.includes('utf-8') || locale.includes('utf8')) return true
  // Windows Terminal and modern PowerShell hosts are UTF-8 capable without
  // setting a POSIX locale variable, which no Windows host does.
  return Boolean(env.WT_SESSION || env.TERM_PROGRAM)
}

export interface CliWriterOptions {
  readonly style?: CliStyle
  readonly unicode?: boolean
  /** Sink for finished lines. Defaults to stdout via console.log. */
  readonly write?: (line: string) => void
}

/**
 * Line-oriented renderer for CLI commands.
 *
 * Every method returns the rendered string as well as emitting it, so a caller
 * that needs to compose or capture output does not have to choose between
 * styling and control.
 */
export class CliWriter {
  readonly style: CliStyle
  readonly unicode: boolean
  private readonly sink: (line: string) => void

  constructor(options: CliWriterOptions = {}) {
    this.style = options.style ?? createCliStyle()
    this.unicode = options.unicode ?? supportsUnicode()
    this.sink = options.write ?? (line => console.log(line))
  }

  /** Emit a pre-rendered line unchanged. */
  line(text = ''): string {
    this.sink(text)
    return text
  }

  /** Section heading with a trailing rule, for grouping a command's output. */
  heading(text: string, width = 0): string {
    const label = this.style.bold(this.style.color('heading', text))
    const target = width || terminalWidth()
    const ruleWidth = Math.max(0, target - text.length - 1)
    const rule = ruleWidth > 0 ? ' ' + this.style.dim(this.ruleChar().repeat(ruleWidth)) : ''
    return this.line(`${label}${rule}`)
  }

  /** A status row: glyph, label, message, and an optional indented hint beneath. */
  status(kind: StatusKind, label: string, message: string, hint = ''): string {
    const spec = STATUS[kind]
    const glyph = this.style.color(spec.role, this.unicode ? spec.glyph : spec.ascii)
    const name = label ? `${this.style.color(spec.role === 'ok' ? 'muted' : spec.role, label)}: ` : ''
    const rendered = `${glyph} ${name}${message}`
    this.line(rendered)
    if (hint) {
      for (const hintLine of hint.split('\n')) {
        this.line('    ' + this.style.dim(`${this.unicode ? '→' : '->'} ${hintLine}`))
      }
    }
    return rendered
  }

  /** Aligned `label  value` row for reporting settled facts. */
  field(label: string, value: string, labelWidth = 0): string {
    const padded = labelWidth > 0 ? label.padEnd(labelWidth) : label
    const rendered = `  ${this.style.color('muted', padded)}  ${value}`
    return this.line(rendered)
  }

  /** A numbered or bulleted step inside a plan. */
  step(text: string, ordinal?: number): string {
    const marker = ordinal === undefined
      ? this.style.color('muted', this.unicode ? '·' : '-')
      : this.style.color('muted', `${ordinal}.`)
    return this.line(`  ${marker} ${text}`)
  }

  /** De-emphasized guidance; never carries information the reader must have. */
  hint(text: string): string {
    return this.line(this.style.dim(text))
  }

  /** Highlight a command the user can copy and run. */
  command(text: string): string {
    return this.style.color('accent', text)
  }

  /** Emphasize a value inside an otherwise plain sentence. */
  value(text: string): string {
    return this.style.color('primary', text)
  }

  private ruleChar(): string {
    return this.unicode ? '─' : '-'
  }
}

/** Usable output width, clamped so a very wide or unknown terminal stays readable. */
export function terminalWidth(columns: number | undefined = process.stdout.columns): number {
  const resolved = Number.isFinite(columns) && (columns ?? 0) > 0 ? (columns as number) : 80
  return Math.max(40, Math.min(100, resolved))
}
