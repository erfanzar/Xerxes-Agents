// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Native skin storage and ANSI palette helpers for the OpenTUI client.
 *
 * Skin files intentionally use Xerxes' small, line-oriented format rather
 * than depending on a YAML parser: one `key: value` mapping per line, with
 * blank lines and whole-line comments ignored.  The format is shared with
 * existing Xerxes skin files while keeping the Bun UI self-contained.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, extname, join, resolve } from 'node:path'

const HEX_COLOR_RE = /^#?([0-9a-fA-F]{6})$/
const SAFE_SKIN_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/
const SAFE_SKIN_KEY_RE = /^[A-Za-z][A-Za-z0-9_-]*$/
const MAX_SKIN_NAME_LENGTH = 80
const ANSI_ESCAPE = '\u001b['
const DEFAULT_SKIN_NAME = 'default'
const SKIN_FILE_SUFFIXES = ['.yaml', '.skin'] as const

export const ROLE_NAMES = Object.freeze([
  'primary',
  'accent',
  'warn',
  'error',
  'tool_name',
  'system',
  'muted',
  'diff_add',
  'diff_del'
] as const)

export type SkinRoleName = (typeof ROLE_NAMES)[number]
export type RgbColor = readonly [red: number, green: number, blue: number]

/** Role colours for the Persepolis Lapis default skin. */
export const DEFAULT_ROLE_COLORS: Readonly<Record<SkinRoleName, string>> = Object.freeze({
  primary: '#4f86ff',
  accent: '#2fd4c4',
  warn: '#f0b429',
  error: '#e0556b',
  tool_name: '#a9c7ff',
  system: '#c77dff',
  muted: '#7b97b5',
  diff_add: '#3fb950',
  diff_del: '#e0556b'
})

export const DEFAULT_BRANDING: Readonly<Record<string, string>> = Object.freeze({
  agent_name: 'Xerxes-Agents',
  welcome: 'The court awaits your word.',
  goodbye: 'The lamps are dimmed. Until the court reconvenes.',
  response_label: 'xerxes',
  prompt_symbol: '❯',
  help_header: 'Royal decrees',
  spinner_verbs: 'inscribing,consulting,surveying,summoning,gilding,marshalling,decreeing,unrolling'
})

const CLASSIC_ROLE_COLORS: Readonly<Record<string, string>> = Object.freeze({
  primary: '#f7c948',
  accent: '#3ddc97',
  warn: '#ffb86c',
  error: '#ff6b6b',
  tool_name: '#6bb1ff',
  system: '#a695e7',
  muted: '#999999'
})

const CLASSIC_BRANDING: Readonly<Record<string, string>> = Object.freeze({
  agent_name: 'Xerxes',
  welcome: 'Welcome to Xerxes',
  goodbye: 'see you next session',
  response_label: 'xerxes',
  prompt_symbol: '›',
  help_header: 'Slash commands',
  spinner_verbs: 'thinking,planning,working,searching,reading,assembling'
})

/** Built-in role palettes.  User files may add further colour roles. */
export const BUILTIN_SKINS: Readonly<Record<string, Readonly<Record<string, string>>>> = Object.freeze({
  default: DEFAULT_ROLE_COLORS,
  classic: CLASSIC_ROLE_COLORS,
  'high-contrast': Object.freeze({ ...DEFAULT_ROLE_COLORS, primary: '#ffffff', accent: '#00ffff', muted: '#cccccc' }),
  dim: Object.freeze({ ...DEFAULT_ROLE_COLORS, primary: '#bcbcbc', accent: '#808080', muted: '#444444' }),
  ares: Object.freeze({
    ...DEFAULT_ROLE_COLORS,
    primary: '#ff5e57',
    accent: '#ff9f1a',
    warn: '#feca57',
    tool_name: '#ff7675'
  }),
  mono: Object.freeze({
    ...DEFAULT_ROLE_COLORS,
    primary: '#eeeeee',
    accent: '#bbbbbb',
    warn: '#bbbbbb',
    error: '#bbbbbb',
    tool_name: '#bbbbbb',
    system: '#bbbbbb',
    muted: '#666666',
    diff_add: '#dddddd',
    diff_del: '#888888'
  }),
  slate: Object.freeze({
    ...DEFAULT_ROLE_COLORS,
    primary: '#90a4ae',
    accent: '#80cbc4',
    warn: '#ffcc80',
    tool_name: '#82b1ff',
    system: '#b39ddb',
    muted: '#546e7a'
  }),
  daylight: Object.freeze({
    ...DEFAULT_ROLE_COLORS,
    primary: '#222831',
    accent: '#0f4c75',
    warn: '#fb7e21',
    error: '#c0392b',
    tool_name: '#0073e6',
    system: '#3742fa',
    muted: '#999999'
  })
})

export const BUILTIN_BRANDING: Readonly<Record<string, Readonly<Record<string, string>>>> = Object.freeze({
  classic: CLASSIC_BRANDING,
  ares: Object.freeze({
    ...DEFAULT_BRANDING,
    agent_name: 'Ares',
    response_label: 'ares',
    spinner_verbs: 'calculating,striking,charging,advancing'
  }),
  mono: Object.freeze({ ...DEFAULT_BRANDING, prompt_symbol: '>' })
})

export class SkinValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SkinValidationError'
  }
}

export class SkinNotFoundError extends Error {
  constructor(name: string) {
    super(`skin not found: ${name}`)
    this.name = 'SkinNotFoundError'
  }
}

/** Parse a #rrggbb or rrggbb colour into its red, green, and blue channels. */
export function hexToRgb(hexColor: string): RgbColor {
  if (typeof hexColor !== 'string') {
    throw new SkinValidationError(`invalid hex color: ${String(hexColor)}`)
  }

  const match = HEX_COLOR_RE.exec(hexColor.trim())
  if (!match) {
    throw new SkinValidationError(`invalid hex color: ${JSON.stringify(hexColor)}`)
  }

  const hex = match[1]!
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16)
  ]
}

/** Return the ANSI 24-bit foreground escape sequence for a hex colour. */
export function hexToAnsiForeground(hexColor: string): string {
  const [red, green, blue] = hexToRgb(hexColor)
  return `${ANSI_ESCAPE}38;2;${red};${green};${blue}m`
}

/** Return the ANSI 24-bit background escape sequence for a hex colour. */
export function hexToAnsiBackground(hexColor: string): string {
  const [red, green, blue] = hexToRgb(hexColor)
  return `${ANSI_ESCAPE}48;2;${red};${green};${blue}m`
}

export interface SkinSnapshot {
  readonly branding: Readonly<Record<string, string>>
  readonly name: string
  readonly roles: Readonly<Record<string, string>>
}

export interface SkinOptions {
  readonly branding?: Readonly<Record<string, string>>
  readonly name: string
  readonly roles?: Readonly<Record<string, string>>
}

/** One named theme containing colour roles and visible branding strings. */
export class Skin {
  readonly branding: Readonly<Record<string, string>>
  readonly name: string
  readonly roles: Readonly<Record<string, string>>

  constructor({ name, roles = DEFAULT_ROLE_COLORS, branding = DEFAULT_BRANDING }: SkinOptions) {
    this.name = validateSkinName(name)
    this.roles = freezeSkinRecord(roles, 'role')
    this.branding = freezeSkinRecord(branding, 'branding')
  }

  /** Get a colour role, falling back to the shipped default palette. */
  color(role: string): string {
    const skinColor = Object.hasOwn(this.roles, role) ? this.roles[role] : undefined
    const defaultColor = Object.hasOwn(DEFAULT_ROLE_COLORS, role)
      ? DEFAULT_ROLE_COLORS[role as SkinRoleName]
      : undefined
    return skinColor ?? defaultColor ?? '#ffffff'
  }

  /** Get this role's ANSI 24-bit foreground escape sequence. */
  foreground(role: string): string {
    return hexToAnsiForeground(this.color(role))
  }

  /** Get a branding label, falling back to the shipped default wording. */
  label(key: string): string {
    const skinLabel = Object.hasOwn(this.branding, key) ? this.branding[key] : undefined
    const defaultLabel = Object.hasOwn(DEFAULT_BRANDING, key) ? DEFAULT_BRANDING[key] : undefined
    return skinLabel ?? defaultLabel ?? ''
  }

  /** Return spinner words parsed from the comma-separated branding value. */
  spinnerVerbs(): string[] {
    const verbs = this.label('spinner_verbs')
      .split(',')
      .map(verb => verb.trim())
      .filter(Boolean)
    return verbs.length ? verbs : ['working']
  }

  /** Create an immutable snapshot suitable for IPC or persistence. */
  toObject(): SkinSnapshot {
    return Object.freeze({
      name: this.name,
      roles: Object.freeze({ ...this.roles }),
      branding: Object.freeze({ ...this.branding })
    })
  }
}

/**
 * Filesystem operations used by {@link SkinEngine}.
 *
 * The port keeps storage explicit so callers can provide a virtual filesystem
 * in an embedded TUI, while the default adapter uses Bun's Node-compatible
 * filesystem APIs and has no Python, YAML, or subprocess dependency.
 */
export interface SkinFileSystem {
  createDirectory(path: string): void
  isDirectory(path: string): boolean
  isFile(path: string): boolean
  readDirectory(path: string): readonly string[]
  readTextFile(path: string): string
  writeTextFile(path: string, contents: string): void
}

/** Bun-native synchronous adapter used by the local terminal UI. */
export const bunSkinFileSystem: SkinFileSystem = Object.freeze({
  createDirectory(path: string): void {
    mkdirSync(path, { recursive: true })
  },
  isDirectory(path: string): boolean {
    return existsSync(path) && statSync(path).isDirectory()
  },
  isFile(path: string): boolean {
    return existsSync(path) && statSync(path).isFile()
  },
  readDirectory(path: string): readonly string[] {
    return readdirSync(path)
  },
  readTextFile(path: string): string {
    return readFileSync(path, 'utf8')
  },
  writeTextFile(path: string, contents: string): void {
    writeFileSync(path, contents, 'utf8')
  }
})

export interface SkinEngineOptions {
  readonly baseDirectory?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fileSystem?: SkinFileSystem
  readonly homeDirectory?: string
}

/** Resolve the skin directory below $XERXES_HOME, without creating it. */
export function skinDirectory(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  homeDirectory = homedir()
): string {
  const configuredHome = environment.XERXES_HOME?.trim()
  const baseDirectory = configuredHome ? resolve(configuredHome) : join(homeDirectory, '.xerxes')
  return join(baseDirectory, 'skins')
}

/** Discover, load, save, and select native TUI skins. */
export class SkinEngine {
  readonly baseDirectory: string
  private activeName = DEFAULT_SKIN_NAME
  private readonly fileSystem: SkinFileSystem

  constructor({
    baseDirectory,
    environment = process.env,
    fileSystem = bunSkinFileSystem,
    homeDirectory
  }: SkinEngineOptions = {}) {
    this.baseDirectory = baseDirectory ? resolve(baseDirectory) : skinDirectory(environment, homeDirectory)
    this.fileSystem = fileSystem
  }

  /** Return sorted built-in and user skin names. */
  available(): string[] {
    const names = new Set(Object.keys(BUILTIN_SKINS))
    if (!this.fileSystem.isDirectory(this.baseDirectory)) {
      return [...names].sort()
    }

    for (const entry of this.fileSystem.readDirectory(this.baseDirectory)) {
      const suffix = extname(entry)
      if (!isSkinFileSuffix(suffix) || !this.fileSystem.isFile(join(this.baseDirectory, entry))) {
        continue
      }

      const name = basename(entry, suffix)
      if (isValidSkinName(name)) {
        names.add(name)
      }
    }

    return [...names].sort()
  }

  /** Load a built-in skin or a user .yaml/.skin file. */
  load(name: string): Skin {
    const normalizedName = validateSkinName(name)
    const builtInRoles = BUILTIN_SKINS[normalizedName]
    if (builtInRoles) {
      return new Skin({
        name: normalizedName,
        roles: builtInRoles,
        branding: BUILTIN_BRANDING[normalizedName] ?? DEFAULT_BRANDING
      })
    }

    for (const suffix of SKIN_FILE_SUFFIXES) {
      const path = join(this.baseDirectory, `${normalizedName}${suffix}`)
      if (this.fileSystem.isFile(path)) {
        return parseSkinDocument(normalizedName, this.fileSystem.readTextFile(path))
      }
    }

    throw new SkinNotFoundError(normalizedName)
  }

  /** Persist a user skin as <name>.yaml and return the written path. */
  save(skin: Skin): string {
    this.fileSystem.createDirectory(this.baseDirectory)
    const path = join(this.baseDirectory, `${validateSkinName(skin.name)}.yaml`)
    const lines = [
      ...Object.entries(skin.roles).map(([key, value]) => `${key}: ${value}`),
      ...Object.entries(skin.branding)
        .filter(([, value]) => Boolean(value))
        .map(([key, value]) => `${key}: ${value}`)
    ]
    this.fileSystem.writeTextFile(path, `${lines.join('\n')}\n`)
    return path
  }

  /** Make a skin active for this engine and return its loaded snapshot. */
  setActive(name: string): Skin {
    const skin = this.load(name)
    this.activeName = skin.name
    return skin
  }

  /** Return a fresh snapshot of this engine's selected skin. */
  active(): Skin {
    return this.load(this.activeName)
  }
}

/** Parse the small Xerxes line-oriented skin document format. */
export function parseSkinDocument(name: string, contents: string): Skin {
  if (typeof contents !== 'string') {
    throw new SkinValidationError('skin document must be text')
  }

  const roles: Record<string, string> = { ...DEFAULT_ROLE_COLORS }
  const branding: Record<string, string> = { ...DEFAULT_BRANDING }

  for (const rawLine of contents.split(/\r\n?|\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) {
      continue
    }

    const separator = line.indexOf(':')
    if (separator < 0) {
      continue
    }

    const key = line.slice(0, separator).trim()
    const value = stripSkinQuotes(line.slice(separator + 1).trim())
    if (!value) {
      continue
    }
    validateSkinKey(key)
    validateSkinValue(value, key)

    if (Object.hasOwn(DEFAULT_BRANDING, key)) {
      branding[key] = value
    } else {
      hexToRgb(value)
      roles[key] = value
    }
  }

  return new Skin({ name, roles, branding })
}

interface ActiveSkinOptions {
  readonly engine?: SkinEngine
  readonly environment?: Readonly<Record<string, string | undefined>>
}

let activeSkin: Skin | undefined

/** Return the process-wide active skin, resolved once from XERXES_SKIN. */
export function getActiveSkin({ engine, environment = process.env }: ActiveSkinOptions = {}): Skin {
  if (activeSkin) {
    return activeSkin
  }

  const requestedName = environment.XERXES_SKIN?.trim() || DEFAULT_SKIN_NAME
  const skinEngine = engine ?? new SkinEngine({ environment })
  try {
    activeSkin = skinEngine.load(requestedName)
  } catch (error) {
    if (!(error instanceof SkinNotFoundError) && !(error instanceof SkinValidationError)) {
      throw error
    }
    activeSkin = skinEngine.load(DEFAULT_SKIN_NAME)
  }
  return activeSkin
}

/** Set the process-wide skin from a snapshot or a name resolved by an engine. */
export function setActiveSkin(skin: Skin | string, engine = new SkinEngine()): Skin {
  if (typeof skin === 'string') {
    activeSkin = engine.load(skin)
    return activeSkin
  }
  if (!(skin instanceof Skin)) {
    throw new SkinValidationError('active skin must be a Skin or skin name')
  }
  activeSkin = skin
  return activeSkin
}

/** Return the active skin's 24-bit foreground escape sequence for a role. */
export function activeForeground(role: string): string {
  return getActiveSkin().foreground(role)
}

function freezeSkinRecord(
  values: Readonly<Record<string, string>>,
  kind: 'branding' | 'role'
): Readonly<Record<string, string>> {
  const normalized: Record<string, string> = {}
  for (const [key, value] of Object.entries(values)) {
    validateSkinKey(key)
    validateSkinValue(value, key)
    if (kind === 'role') {
      hexToRgb(value)
    }
    normalized[key] = value
  }
  return Object.freeze(normalized)
}

function isSkinFileSuffix(suffix: string): suffix is (typeof SKIN_FILE_SUFFIXES)[number] {
  return (SKIN_FILE_SUFFIXES as readonly string[]).includes(suffix)
}

function isValidSkinName(name: string): boolean {
  try {
    validateSkinName(name)
    return true
  } catch (error) {
    if (error instanceof SkinValidationError) {
      return false
    }
    throw error
  }
}

function stripSkinQuotes(value: string): string {
  return value.replace(/^"+|"+$/g, '').replace(/^'+|'+$/g, '')
}

function validateSkinKey(key: string): void {
  if (!SAFE_SKIN_KEY_RE.test(key)) {
    throw new SkinValidationError(`invalid skin key: ${JSON.stringify(key)}`)
  }
}

function validateSkinName(name: string): string {
  if (typeof name !== 'string') {
    throw new SkinValidationError('skin name must be a string')
  }
  const normalized = name.trim()
  if (!normalized || normalized.length > MAX_SKIN_NAME_LENGTH || !SAFE_SKIN_NAME_RE.test(normalized)) {
    throw new SkinValidationError(`invalid skin name: ${JSON.stringify(name)}`)
  }
  return normalized
}

function validateSkinValue(value: string, key: string): void {
  if (typeof value !== 'string' || value.includes('\u0000') || /[\r\n]/.test(value)) {
    throw new SkinValidationError(`invalid skin value for ${key}`)
  }
}
