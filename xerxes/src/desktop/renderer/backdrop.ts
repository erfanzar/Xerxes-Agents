// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A window background the whole theme follows (Hermes's seeded palette).
 *
 * The user picks a gradient or an image. Two seed colours come out of it —
 * the gradient's stops, or the image's top and bottom halves — and every
 * surface, the accent and light/dark are DERIVED from those seeds:
 *
 * - light or dark follows the background's own luminance, not a toggle;
 * - surfaces are the seeds' mean pulled toward black (dark) or white (light)
 *   and left translucent, so the background reads through every pane;
 * - the accent is the most colourful seed, shifted in lightness until it
 *   clears WCAG AA on the conversation surface (hue and chroma kept).
 *
 * Text keeps the stock ramp: it is already tuned for AA on both families of
 * surface, and tinting it is how themes get unreadable.
 */

import { trackBackdropImage } from './backdropLayers.js'
import { loadPalette, paletteTokens, paletteVariant, type Palette } from './palettes.js'

export const BACKDROP_KEY = 'xerxes.desktop.backdrop.v1'

/**
 * How solid the surfaces over the background are, 0–100. The conversation is
 * its own lever and starts see-through so the background lives there; the
 * sidebar, toolbar, rail and cards share the other and start near solid,
 * which keeps navigation calm.
 */
export interface BackdropOpacity {
  readonly chat?: number
  readonly panels?: number
  /** How strongly the picture is blurred under every surface, 0–100 (0 is sharp). */
  readonly blur?: number
}

export const DEFAULT_CHAT_OPACITY = 70
export const DEFAULT_PANEL_OPACITY = 75
export const DEFAULT_BLUR = 75

/**
 * A see-through window needs the native material under the page: macOS
 * windows are created with vibrancy (main.ts). Elsewhere the window is
 * opaque, so the option is not offered.
 */
export function transparencySupported(): boolean {
  return typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent)
}

export type Backdrop =
  | { readonly kind: 'none' }
  | ({ readonly kind: 'gradient'; readonly from: string; readonly to: string; readonly angle: number } & BackdropOpacity)
  | ({ readonly kind: 'image'; readonly src: string; readonly from: string; readonly to: string; readonly accent?: string } & BackdropOpacity)
  /**
   * The window itself: what is behind it shows through. `systemBlur` false
   * turns macOS's blur off for clear glass (default on).
   */
  | ({ readonly kind: 'transparent'; readonly systemBlur?: boolean } & BackdropOpacity)

/**
 * Artwork that ships with the app (`renderer/backgrounds/`). Each carries its
 * seeds, measured once from the image with the same sampling `backdropFromImage`
 * uses — reading pixels from a `file://` image at runtime would taint the
 * canvas in the packaged app, so they are not recomputed there.
 */
export const BUNDLED_BACKGROUNDS: ReadonlyArray<{ readonly name: string; readonly backdrop: Extract<Backdrop, { kind: 'image' }> }> = [
  { name: 'Persepolis Ablaze', backdrop: { kind: 'image', src: 'backgrounds/background-1.jpg', from: '#461e19', to: '#2b201b', accent: '#611c17' } },
  { name: 'Guardian of Pasargadae', backdrop: { kind: 'image', src: 'backgrounds/background-2.jpg', from: '#1b1b1b', to: '#1c1c1c', accent: '#1c1b1c' } },
  { name: 'Faravahar', backdrop: { kind: 'image', src: 'backgrounds/background-3.jpg', from: '#2b110d', to: '#20120e', accent: '#3f130f' } },
  { name: 'The Immortals', backdrop: { kind: 'image', src: 'backgrounds/background-4.jpg', from: '#322055', to: '#2c1d49', accent: '#1f0f40' } },
  { name: 'Artemisia', backdrop: { kind: 'image', src: 'backgrounds/background-5.jpg', from: '#691ede', to: '#5217b1', accent: '#7220ef' } },
  { name: 'Immortals by Moonlight', backdrop: { kind: 'image', src: 'backgrounds/background-6.jpg', from: '#302765', to: '#181244', accent: '#100b3a' } },
]

/** A bundled artwork path, as stored; anything else must be an embedded image. */
const BUNDLED_SRC = /^backgrounds\/background-\d+\.jpg$/

/** Starting points in the picker; any two colours (or an image) work. */
export const BACKDROP_PRESETS: ReadonlyArray<{ readonly name: string; readonly backdrop: Backdrop }> = [
  { name: 'Lapis', backdrop: { kind: 'gradient', from: '#0b1f4d', to: '#3b1a5a', angle: 135 } },
  { name: 'Dusk', backdrop: { kind: 'gradient', from: '#2b1055', to: '#d4418e', angle: 160 } },
  { name: 'Forest', backdrop: { kind: 'gradient', from: '#0f2027', to: '#2c5364', angle: 145 } },
  { name: 'Ember', backdrop: { kind: 'gradient', from: '#1f0b0b', to: '#8a3b12', angle: 150 } },
  { name: 'Dawn', backdrop: { kind: 'gradient', from: '#fdfbfb', to: '#e2d1f9', angle: 160 } },
  { name: 'Sea glass', backdrop: { kind: 'gradient', from: '#e0f7f4', to: '#b8d8f5', angle: 140 } },
]

type Rgb = readonly [number, number, number]

export function parseHex(hex: string): Rgb | undefined {
  const match = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim())
  if (!match) return undefined
  const value = match[1]!.length === 3 ? match[1]!.split('').map(c => c + c).join('') : match[1]!
  return [0, 2, 4].map(i => parseInt(value.slice(i, i + 2), 16)) as unknown as Rgb
}

export function toHex([r, g, b]: Rgb): string {
  return '#' + [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('')
}

const mixRgb = (a: Rgb, b: Rgb, t: number): Rgb => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]

/** WCAG relative luminance. */
export function luminance([r, g, b]: Rgb): number {
  const linear = (v: number) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4 }
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
}

export function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

function toHsl([r, g, b]: Rgb): [number, number, number] {
  const [rn, gn, bn] = [r / 255, g / 255, b / 255]
  const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn), l = (max + min) / 2
  if (max === min) return [0, 0, l]
  const d = max - min
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  const h = max === rn ? (gn - bn) / d + (gn < bn ? 6 : 0) : max === gn ? (bn - rn) / d + 2 : (rn - gn) / d + 4
  return [h / 6, s, l]
}

function fromHsl([h, s, l]: [number, number, number]): Rgb {
  if (s === 0) return [l * 255, l * 255, l * 255]
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q
  const channel = (t: number) => {
    const u = t < 0 ? t + 1 : t > 1 ? t - 1 : t
    return 255 * (u < 1 / 6 ? p + (q - p) * 6 * u : u < 1 / 2 ? q : u < 2 / 3 ? p + (q - p) * (2 / 3 - u) * 6 : p)
  }
  return [channel(h + 1 / 3), channel(h), channel(h - 1 / 3)]
}

/** HSL saturation, which ranks dark colours fairly (raw chroma calls every dark colour grey). */
const colourfulness = (color: Rgb) => { const [, s, l] = toHsl(color); return s * (1 - Math.abs(2 * l - 1) * 0.5) }

/**
 * Shift a colour's lightness — hue and saturation kept — until it clears
 * `ratio` against `surface`, so a deep lapis stays blue instead of washing
 * toward grey. An accent needs some colour to be one, so saturation is
 * floored.
 */
export function ensureContrast(color: Rgb, surface: Rgb, ratio: number): Rgb {
  const [h, s, l] = toHsl(color)
  // An accent needs colour — unless the background has none (a greyscale
  // image), where inventing a hue would be wrong.
  const saturation = s < 0.12 ? s : Math.max(s, 0.55)
  const lighter = luminance(surface) < 0.5
  for (let step = 0; step <= 40; step += 1) {
    const lightness = lighter ? l + (0.95 - l) * step / 40 : l - l * step / 40
    const candidate = fromHsl([h, saturation, lightness])
    if (contrast(candidate, surface) >= ratio) return candidate
  }
  return lighter ? [255, 255, 255] : [0, 0, 0]
}

export interface DerivedTheme {
  readonly mode: 'light' | 'dark'
  readonly paint: string
  readonly tokens: Readonly<Record<string, string>>
}

const alpha = (color: Rgb, opacity: number) => toHex(color) + Math.round(opacity * 255).toString(16).padStart(2, '0')

/** Every token the background drives, from its two seeds (plus an image's accent). */
export function deriveTheme(backdrop: Exclude<Backdrop, { kind: 'none' } | { kind: 'transparent' }>): DerivedTheme | undefined {
  const from = parseHex(backdrop.from), to = parseHex(backdrop.to)
  if (!from || !to) return undefined
  const mean = mixRgb(from, to, 0.5)
  const mode = luminance(mean) > 0.4 ? 'light' : 'dark'
  const ink: Rgb = mode === 'dark' ? [0, 0, 0] : [255, 255, 255]
  // Surfaces: the background's own colour, darkened (or lightened) enough to
  // carry text, and translucent so the background still shows through.
  const surface = (depth: number) => mixRgb(mean, ink, depth)
  const screen = surface(mode === 'dark' ? 0.72 : 0.82)
  const seeds = [from, to, ...(backdrop.kind === 'image' && backdrop.accent && parseHex(backdrop.accent) ? [parseHex(backdrop.accent)!] : [])]
  const vivid = seeds.reduce((best, seed) => colourfulness(seed) > colourfulness(best) ? seed : best)
  const accent = ensureContrast(vivid, screen, 4.5)
  const paint = backdrop.kind === 'image'
    ? `center / cover no-repeat url("${backdrop.src.replace(/"/g, '%22')}"), linear-gradient(160deg, ${backdrop.from}, ${backdrop.to})`
    : `linear-gradient(${backdrop.angle}deg, ${backdrop.from}, ${backdrop.to})`
  const glassEdge = mode === 'dark' ? '#ffffff1f' : '#ffffffcc'
  const clamp = (value: number | undefined, fallback: number) => Math.max(0, Math.min(100, value ?? fallback)) / 100
  const chat = clamp(backdrop.chat, DEFAULT_CHAT_OPACITY)
  const panels = clamp(backdrop.panels, DEFAULT_PANEL_OPACITY)
  // Its own lever: how solid a surface is and how blurred the picture under
  // it is are separate choices.
  const blur = clamp(backdrop.blur, DEFAULT_BLUR)
  // macOS's own glass: a wide blur with colour kept rich (180% saturation,
  // as the system materials do). The default level, 75%, is its 30px.
  const chatRadius = Math.round(blur * 40), panelRadius = Math.round(blur * 40)
  // Cards float on either family of surface, so they stay a touch more solid than the panels.
  const cards = Math.min(1, panels + 0.08)
  return {
    mode,
    paint,
    tokens: {
      '--x-backdrop': alpha(surface(mode === 'dark' ? 0.82 : 0.7), panels),
      '--x-sunken': alpha(surface(mode === 'dark' ? 0.8 : 0.75), Math.max(panels, 0.6)),
      '--x-screen': alpha(screen, chat),
      '--x-chrome': alpha(surface(mode === 'dark' ? 0.78 : 0.86), cards),
      '--x-card': alpha(surface(mode === 'dark' ? 0.62 : 0.92), cards),
      '--x-selected': alpha(mixRgb(screen, mode === 'dark' ? [255, 255, 255] : [0, 0, 0], 0.08), Math.max(cards, 0.9)),
      '--x-chip': alpha(surface(mode === 'dark' ? 0.66 : 0.9), cards),
      '--mac-toolbar': alpha(surface(mode === 'dark' ? 0.78 : 0.86), panels),
      '--glass-navigation': alpha(surface(mode === 'dark' ? 0.76 : 0.86), panels),
      '--glass-overlay': alpha(surface(mode === 'dark' ? 0.68 : 0.94), 0.94),
      // One lever: solidity and blur rise together. At 0 the conversation is
      // clear glass and the background is shown exactly as it is.
      '--glass-blur': panelRadius > 0 ? `blur(${panelRadius}px) saturate(180%)` : 'none',
      '--x-chat-blur': chatRadius > 0 ? `blur(${chatRadius}px) saturate(180%)` : 'none',
      // Blur radii for the static layers (see backdropLayers.ts): each layer
      // overhangs its surface by this much and is clipped back, so the blur
      // has real pixels at the edge instead of fading to transparent.
      '--x-chat-r': `${chatRadius}px`,
      '--x-panel-r': `${panelRadius}px`,
      '--glass-edge': glassEdge,
      // An OPAQUE surface for 1px seams: anything translucent that thin shows
      // the raw, unblurred background, which reads as a glittering dashed line.
      '--x-seam': toHex(surface(mode === 'dark' ? 0.7 : 0.8)),
      // An OPAQUE card colour for popovers that float over the conversation:
      // the translucent tokens let the transcript read through the list.
      // Depth sits between a card and the panel under it, so it matches both.
      '--x-popover': toHex(surface(mode === 'dark' ? 0.7 : 0.9)),
      '--x-accent': toHex(accent),
      '--x-focus': toHex(accent),
      '--x-working': toHex(accent),
      '--x-accent-soft': alpha(accent, 0.12),
      // Opaque ink on accent-filled buttons: the conversation's own colour, solid.
      '--x-on-accent': toHex(screen),
      '--x-inverse': toHex(screen),
    },
  }
}

export function parseBackdrop(value: unknown): Backdrop {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const hex = (v: unknown) => typeof v === 'string' && parseHex(v) ? v : undefined
  const from = hex(record.from), to = hex(record.to)
  const level = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(100, Math.round(v))) : undefined
  const chat = level(record.chat), panels = level(record.panels), blur = level(record.blur)
  const opacity = { ...(chat === undefined ? {} : { chat }), ...(panels === undefined ? {} : { panels }), ...(blur === undefined ? {} : { blur }) }
  if (record.kind === 'transparent') return { kind: 'transparent', ...(record.systemBlur === false ? { systemBlur: false } : {}), ...opacity }
  if (record.kind === 'gradient' && from && to) {
    const angle = typeof record.angle === 'number' && Number.isFinite(record.angle) ? record.angle % 360 : 160
    return { kind: 'gradient', from, to, angle, ...opacity }
  }
  if (record.kind === 'image' && from && to && typeof record.src === 'string' && (BUNDLED_SRC.test(record.src) || /^data:image\/(png|jpeg|webp);base64,/.test(record.src))) {
    const accent = hex(record.accent)
    return { kind: 'image', src: record.src, from, to, ...(accent ? { accent } : {}), ...opacity }
  }
  return { kind: 'none' }
}

const PAINTED = new Set<string>()

/** Paint the background and every token it drives; `none` restores the stock theme. */
export function applyBackdrop(backdrop: Backdrop, root: HTMLElement = document.documentElement, palette: Palette | undefined = loadPalette()): void {
  for (const name of PAINTED) root.style.removeProperty(name)
  PAINTED.clear()
  root.style.removeProperty('--x-backdrop-paint')
  // macOS's blur behind the window is only turned off for clear transparent
  // glass; every other background keeps the native material (the page covers it).
  if (typeof window !== 'undefined') window.xerxes?.setWindowBlur?.(!(backdrop.kind === 'transparent' && backdrop.systemBlur === false))
  const paint = (tokens: Readonly<Record<string, string>>) => {
    for (const [name, value] of Object.entries(tokens)) {
      root.style.setProperty(name, value)
      PAINTED.add(name)
    }
  }
  // The user's light/dark choice (or the system's) picks a palette's variant.
  const chosen = root.getAttribute('data-user-theme')
  const wanted: 'light' | 'dark' = chosen === 'light' || chosen === 'dark' ? chosen : window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  const variant = palette ? paletteVariant(palette, wanted) : undefined
  if (palette) root.setAttribute('data-palette', palette.name)
  else root.removeAttribute('data-palette')
  if (backdrop.kind === 'none') {
    root.removeAttribute('data-backdrop')
    trackBackdropImage(undefined)
    // Hand light/dark back to the user's choice (or the system), unless a
    // single-mode palette decides it.
    root.setAttribute('data-theme', variant?.mode ?? wanted)
    // No background: the palette's surfaces are solid.
    if (variant) paint(paletteTokens(variant.colors))
    return
  }
  if (backdrop.kind === 'transparent') {
    if (!transparencySupported()) { applyBackdrop(BUNDLED_BACKGROUNDS[1]!.backdrop, root, palette); return }
    // The desktop behind the window is the picture; macOS blurs it in the
    // compositor, so the page adds no blur of its own — only the tint the
    // sliders set, in a neutral Xerxes surface (or the theme's).
    const mode = variant?.mode ?? wanted
    const base = mode === 'light' ? '#f4f5f7' : '#15171c'
    const neutral = deriveTheme({ kind: 'gradient', from: base, to: base, angle: 0, ...(backdrop.chat === undefined ? {} : { chat: backdrop.chat }), ...(backdrop.panels === undefined ? {} : { panels: backdrop.panels }) })
    if (!neutral) return
    root.setAttribute('data-backdrop', 'transparent')
    root.setAttribute('data-theme', mode)
    root.style.setProperty('--x-backdrop-paint', 'transparent')
    trackBackdropImage(undefined)
    // A grey seed would derive a grey accent; keep the stock Xerxes accent.
    const { '--x-accent': _a, '--x-focus': _f, '--x-working': _w, '--x-accent-soft': _s, '--x-on-accent': _o, ...tokens } = neutral.tokens
    paint({ ...tokens, '--x-backdrop-image': 'none', '--x-backdrop-gradient': 'none', '--glass-blur': 'none', '--x-chat-blur': 'none', '--x-chat-r': '0px', '--x-panel-r': '0px' })
    if (variant) paint(paletteTokens(variant.colors, { chat: opacityOf(backdrop.chat, DEFAULT_CHAT_OPACITY), panels: opacityOf(backdrop.panels, DEFAULT_PANEL_OPACITY) }))
    return
  }
  const derived = deriveTheme(backdrop)
  if (!derived) return
  root.setAttribute('data-backdrop', backdrop.kind)
  root.setAttribute('data-theme', variant?.mode ?? derived.mode)
  root.style.setProperty('--x-backdrop-paint', derived.paint)
  // The same picture, split for the surfaces' blurred layers.
  const image = backdrop.kind === 'image' ? `url("${backdrop.src.replace(/"/g, '%22')}")` : 'none'
  const gradient = backdrop.kind === 'image' ? `linear-gradient(160deg, ${backdrop.from}, ${backdrop.to})` : `linear-gradient(${backdrop.angle}deg, ${backdrop.from}, ${backdrop.to})`
  for (const [name, value] of [['--x-backdrop-image', image], ['--x-backdrop-gradient', gradient]] as const) {
    root.style.setProperty(name, value)
    PAINTED.add(name)
  }
  trackBackdropImage(backdrop.kind === 'image' ? backdrop.src : undefined)
  paint(derived.tokens)
  // With a palette the glass takes the theme's colours instead of the
  // picture's, at the same opacities; blur, radii and seams stay derived.
  if (variant) paint(paletteTokens(variant.colors, { chat: opacityOf(backdrop.chat, DEFAULT_CHAT_OPACITY), panels: opacityOf(backdrop.panels, DEFAULT_PANEL_OPACITY) }))
}

const opacityOf = (value: number | undefined, fallback: number) => Math.max(0, Math.min(100, value ?? fallback)) / 100

/**
 * What a fresh install shows: no background — solid surfaces in the default
 * colour theme (Nous, see palettes.ts). Any choice, including None, is
 * remembered explicitly.
 */
export function defaultBackdrop(): Backdrop {
  return { kind: 'none' }
}

export function loadBackdrop(): Backdrop {
  try {
    const stored = localStorage.getItem(BACKDROP_KEY)
    return stored === null ? defaultBackdrop() : parseBackdrop(JSON.parse(stored))
  } catch { return defaultBackdrop() }
}

/** Persist; an image too large for local storage is reported rather than silently lost. */
export function saveBackdrop(backdrop: Backdrop): boolean {
  try {
    localStorage.setItem(BACKDROP_KEY, JSON.stringify(backdrop))
    return true
  } catch { return false }
}

/**
 * The image's accent: the hue that carries the most colour, averaged only
 * within that hue — averaging across hues (an orange sun on green) gives mud.
 */
export function dominantVivid(pixels: readonly Rgb[]): Rgb {
  const bins = new Map<number, { weight: number; members: Rgb[] }>()
  for (const pixel of pixels) {
    const weight = colourfulness(pixel)
    if (weight < 0.15) continue
    const bin = Math.floor(toHsl(pixel)[0] * 12) % 12
    const entry = bins.get(bin) ?? { weight: 0, members: [] }
    entry.weight += weight
    entry.members.push(pixel)
    bins.set(bin, entry)
  }
  const best = [...bins.values()].sort((a, b) => b.weight - a.weight)[0]
  // Within that hue, the most colourful quarter: its dark and grey pixels would dull it.
  const members = best ? [...best.members].sort((a, b) => colourfulness(b) - colourfulness(a)).slice(0, Math.max(1, Math.ceil(best.members.length / 4))) : [...pixels]
  return members.reduce<Rgb>((sum, p) => [sum[0] + p[0] / members.length, sum[1] + p[1] / members.length, sum[2] + p[2] / members.length], [0, 0, 0])
}

/**
 * Read an image file into a background: downscaled for storage, with its
 * seeds taken from the pixels — the top and bottom halves' averages for the
 * surfaces, the most colourful tenth for the accent.
 */
export async function backdropFromImage(file: Blob): Promise<Extract<Backdrop, { kind: 'image' }>> {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, 1920 / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(bitmap.width * scale))
  canvas.height = Math.max(1, Math.round(bitmap.height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('This window cannot read images.')
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  const src = canvas.toDataURL('image/jpeg', 0.82)
  const sample = document.createElement('canvas')
  sample.width = 24
  sample.height = 24
  const sampleContext = sample.getContext('2d')!
  sampleContext.drawImage(bitmap, 0, 0, 24, 24)
  const data = sampleContext.getImageData(0, 0, 24, 24).data
  const pixels: Rgb[] = []
  for (let i = 0; i < data.length; i += 4) pixels.push([data[i]!, data[i + 1]!, data[i + 2]!])
  const average = (list: Rgb[]): Rgb => list.reduce<Rgb>((sum, p) => [sum[0] + p[0] / list.length, sum[1] + p[1] / list.length, sum[2] + p[2] / list.length], [0, 0, 0])
  const half = pixels.length / 2
  return { kind: 'image', src, from: toHex(average(pixels.slice(0, half))), to: toHex(average(pixels.slice(half))), accent: toHex(dominantVivid(pixels)) }
}
