// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'

export const CREATOR_TRACE_METADATA_KEY = 'xerxes_creator_trace_v1'
const DOCUMENT_VERSION = 1
const MAX_PACKAGES = 128
const MAX_PARAMETERS = 64
const MAX_TEMPLATE_CHARS = 16_384
const MAX_RENDERED_CHARS = 65_536
const MAX_TRACE_ROWS = 64
const TOOL_NAME = /^[a-z][a-z0-9_-]{1,63}$/
const PARAMETER_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/
const VERSION = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/
const PLACEHOLDER = /{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}/g

export interface DeclarativeForgeParameter {
  readonly name: string
  readonly description: string
  readonly required: boolean
  readonly defaultValue?: string
}

export interface DeclarativeForgePackage {
  readonly name: string
  readonly version: string
  readonly description: string
  readonly parameters: readonly DeclarativeForgeParameter[]
  readonly template: string
  readonly createdAt: string
}

export interface DeclarativeForgeDefinition {
  readonly name: string
  readonly version: string
  readonly description: string
  readonly parameters?: readonly Readonly<Record<string, unknown>>[]
  readonly template: string
}

interface ForgeDocument {
  readonly version: typeof DOCUMENT_VERSION
  readonly packages: readonly DeclarativeForgePackage[]
}

export interface CreatorTraceRow {
  readonly action: string
  readonly name: string
  readonly version: string
  readonly status: 'error' | 'ok'
  readonly detail: string
  readonly at: string
}

/**
 * Persistent, declarative creator-mode package store.
 *
 * A forged tool only renders a bounded text template from declared scalar
 * inputs. It cannot execute JavaScript, spawn a process, read a file, or use
 * the network, so defining a package does not smuggle a new privileged host
 * capability past the normal tool-policy boundary.
 */
export class DeclarativeToolForge {
  readonly filePath: string

  constructor(filePath = join(xerxesHome(), 'forged-tools.json')) {
    this.filePath = filePath
  }

  list(): DeclarativeForgePackage[] {
    return [...this.load().packages].sort((left, right) => (
      left.name.localeCompare(right.name) || compareVersions(right.version, left.version)
    ))
  }

  inspect(name: string, version?: string): DeclarativeForgePackage | undefined {
    const cleanName = validName(name)
    const candidates = this.list().filter(pkg => pkg.name === cleanName)
    if (!version) return candidates[0]
    const cleanVersion = validVersion(version)
    return candidates.find(pkg => pkg.version === cleanVersion)
  }

  define(input: DeclarativeForgeDefinition): DeclarativeForgePackage {
    const pkg = normalizePackage(input)
    const document = this.load()
    if (document.packages.some(existing => existing.name === pkg.name && existing.version === pkg.version)) {
      throw new ValidationError('forge.version', `${pkg.name}@${pkg.version} already exists; forged versions are immutable`)
    }
    if (document.packages.length >= MAX_PACKAGES) {
      throw new ValidationError('forge.packages', `cannot exceed ${MAX_PACKAGES} versions`)
    }
    this.write({ version: DOCUMENT_VERSION, packages: [...document.packages, pkg] })
    return pkg
  }

  undefine(name: string, version: string): boolean {
    const cleanName = validName(name)
    const cleanVersion = validVersion(version)
    const document = this.load()
    const packages = document.packages.filter(pkg => !(pkg.name === cleanName && pkg.version === cleanVersion))
    if (packages.length === document.packages.length) return false
    this.write({ version: DOCUMENT_VERSION, packages })
    return true
  }

  run(name: string, version: string | undefined, inputs: Readonly<Record<string, unknown>>): {
    readonly name: string
    readonly version: string
    readonly output: string
  } {
    const pkg = this.inspect(name, version)
    if (!pkg) throw new ValidationError('forge.name', `no forged package named ${name}${version ? `@${version}` : ''}`)
    const declarations = new Map(pkg.parameters.map(parameter => [parameter.name, parameter]))
    for (const key of Object.keys(inputs)) {
      if (!declarations.has(key)) throw new ValidationError(`forge.inputs.${key}`, 'is not a declared parameter')
    }
    const values = new Map<string, string>()
    for (const parameter of pkg.parameters) {
      const supplied = Object.hasOwn(inputs, parameter.name)
      const value = supplied ? scalarText(inputs[parameter.name], `forge.inputs.${parameter.name}`) : parameter.defaultValue
      if (value === undefined && parameter.required) {
        throw new ValidationError(`forge.inputs.${parameter.name}`, 'is required')
      }
      values.set(parameter.name, value ?? '')
    }
    const output = pkg.template.replace(PLACEHOLDER, (_match, parameter: string) => values.get(parameter) ?? '')
    if (output.length > MAX_RENDERED_CHARS) {
      throw new ValidationError('forge.output', `cannot exceed ${MAX_RENDERED_CHARS} characters`)
    }
    return { name: pkg.name, version: pkg.version, output }
  }

  private load(): ForgeDocument {
    let raw: string
    try {
      raw = readFileSync(this.filePath, 'utf8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { version: DOCUMENT_VERSION, packages: [] }
      throw error
    }
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new ValidationError('forge.document', 'must be an object')
    }
    const record = parsed as Readonly<Record<string, unknown>>
    if (record.version !== DOCUMENT_VERSION || !Array.isArray(record.packages)) {
      throw new ValidationError('forge.document', `must use version ${DOCUMENT_VERSION}`)
    }
    return {
      version: DOCUMENT_VERSION,
      packages: record.packages.map((pkg, index) => persistedPackage(pkg, index)),
    }
  }

  private write(document: ForgeDocument): void {
    mkdirSync(dirname(this.filePath), { recursive: true, mode: 0o700 })
    const temporary = `${this.filePath}.${process.pid}.${randomUUID()}.tmp`
    writeFileSync(temporary, `${JSON.stringify(document, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
    chmodSync(temporary, 0o600)
    renameSync(temporary, this.filePath)
  }
}

export function recordCreatorTrace(
  metadata: Record<string, unknown>,
  input: Omit<CreatorTraceRow, 'at'> & { readonly at?: string },
): void {
  const rows = creatorTraceValues(metadata)
  rows.push({
    action: bounded(input.action, 32),
    name: bounded(input.name, 64),
    version: bounded(input.version, 64),
    status: input.status,
    detail: bounded(input.detail, 500),
    at: input.at ?? new Date().toISOString(),
  })
  metadata[CREATOR_TRACE_METADATA_KEY] = rows.slice(-MAX_TRACE_ROWS)
}

export function creatorTraceValues(metadata: Readonly<Record<string, unknown>>): CreatorTraceRow[] {
  const value = metadata[CREATOR_TRACE_METADATA_KEY]
  if (!Array.isArray(value)) return []
  return value.slice(-MAX_TRACE_ROWS).flatMap(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const row = item as Readonly<Record<string, unknown>>
    const status = row.status === 'error' ? 'error' : row.status === 'ok' ? 'ok' : undefined
    if (!status) return []
    return [{
      action: text(row.action),
      name: text(row.name),
      version: text(row.version),
      status,
      detail: text(row.detail),
      at: text(row.at),
    }]
  })
}

function normalizePackage(input: DeclarativeForgeDefinition): DeclarativeForgePackage {
  const name = validName(input.name)
  const version = validVersion(input.version)
  const description = requiredText(input.description, 'forge.description', 1_000)
  const template = requiredText(input.template, 'forge.template', MAX_TEMPLATE_CHARS)
  const rawParameters = input.parameters ?? []
  if (rawParameters.length > MAX_PARAMETERS) {
    throw new ValidationError('forge.parameters', `cannot exceed ${MAX_PARAMETERS} entries`)
  }
  const parameters = rawParameters.map((parameter, index) => normalizeParameter(parameter, index))
  const names = new Set<string>()
  for (const parameter of parameters) {
    if (names.has(parameter.name)) throw new ValidationError(`forge.parameters.${parameter.name}`, 'is duplicated')
    names.add(parameter.name)
  }
  for (const match of template.matchAll(PLACEHOLDER)) {
    const parameter = match[1] ?? ''
    if (!names.has(parameter)) {
      throw new ValidationError('forge.template', `references undeclared parameter ${parameter}`)
    }
  }
  return { name, version, description, parameters, template, createdAt: new Date().toISOString() }
}

function normalizeParameter(value: Readonly<Record<string, unknown>>, index: number): DeclarativeForgeParameter {
  const name = text(value.name)
  if (!PARAMETER_NAME.test(name)) {
    throw new ValidationError(`forge.parameters.${index}.name`, 'must be a safe identifier')
  }
  const defaultValue = value.default === undefined
    ? undefined
    : scalarText(value.default, `forge.parameters.${index}.default`)
  return {
    name,
    description: bounded(text(value.description), 500),
    required: value.required === true,
    ...(defaultValue === undefined ? {} : { defaultValue }),
  }
}

function persistedPackage(value: unknown, index: number): DeclarativeForgePackage {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ValidationError(`forge.packages.${index}`, 'must be an object')
  }
  const row = value as Readonly<Record<string, unknown>>
  const rawParameters = Array.isArray(row.parameters)
    ? row.parameters.map(parameter => {
        if (!parameter || typeof parameter !== 'object' || Array.isArray(parameter)) return {}
        const record = parameter as Readonly<Record<string, unknown>>
        return {
          name: record.name,
          description: record.description,
          required: record.required,
          ...(record.defaultValue === undefined ? {} : { default: record.defaultValue }),
        }
      })
    : []
  const normalized = normalizePackage({
    name: text(row.name),
    version: text(row.version),
    description: text(row.description),
    template: text(row.template),
    parameters: rawParameters,
  })
  return { ...normalized, createdAt: text(row.createdAt) || normalized.createdAt }
}

function validName(value: string): string {
  const clean = text(value)
  if (!TOOL_NAME.test(clean)) {
    throw new ValidationError('forge.name', 'must be 2-64 lowercase letters, numbers, underscores, or hyphens')
  }
  return clean
}

function validVersion(value: string): string {
  const clean = text(value)
  if (!VERSION.test(clean)) throw new ValidationError('forge.version', 'must be semantic version x.y.z')
  return clean
}

function requiredText(value: unknown, field: string, max: number): string {
  const clean = text(value)
  if (!clean) throw new ValidationError(field, 'must not be empty')
  if (clean.length > max) throw new ValidationError(field, `cannot exceed ${max} characters`)
  return clean
}

function scalarText(value: unknown, field: string): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value === 'boolean' || value === null) return String(value)
  throw new ValidationError(field, 'must be a string, finite number, boolean, or null')
}

function compareVersions(left: string, right: string): number {
  const a = left.split('-', 1)[0]!.split('.').map(Number)
  const b = right.split('-', 1)[0]!.split('.').map(Number)
  for (let index = 0; index < 3; index += 1) {
    const delta = (a[index] ?? 0) - (b[index] ?? 0)
    if (delta !== 0) return delta
  }
  return left.localeCompare(right)
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function bounded(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}
