// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { normalizeOrigin } from './identity/core.js'
import type { HubPaths } from './paths.js'

/**
 * Settings `xerxes hub setup` saves so `hub run`, `hub pair` and a service
 * manager need no environment variables. An environment variable always wins
 * over the file. The file is never rewritten when it cannot be read.
 */
export interface HubSettings {
  readonly publicOrigin?: string
  readonly host?: string
  readonly port?: number
}

export class HubSettingsError extends Error {
  constructor(message: string) { super(message); this.name = 'HubSettingsError' }
}

export const HUB_SETTINGS_FILENAME = 'settings.json'
const KEYS: ReadonlySet<string> = new Set(['version', 'public_origin', 'host', 'port'])

export function hubSettingsPath(paths: Pick<HubPaths, 'stateDir'>): string {
  return join(paths.stateDir, HUB_SETTINGS_FILENAME)
}

function validate(value: unknown, where: string): HubSettings {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new HubSettingsError(`${where} must be a JSON object`)
  const row = value as Record<string, unknown>
  for (const key of Object.keys(row)) if (!KEYS.has(key)) throw new HubSettingsError(`${where} has an unknown setting ${JSON.stringify(key)}`)
  if (row.version !== 1) throw new HubSettingsError(`${where} has an unsupported version`)
  let publicOrigin: string | undefined
  if (row.public_origin !== undefined) {
    publicOrigin = typeof row.public_origin === 'string' ? normalizeOrigin(row.public_origin) : undefined
    if (publicOrigin === undefined) throw new HubSettingsError(`${where}: public_origin must be an exact origin such as https://hub.example.ts.net`)
  }
  if (row.port !== undefined && (typeof row.port !== 'number' || !Number.isInteger(row.port) || row.port < 1 || row.port > 65535)) {
    throw new HubSettingsError(`${where}: port must be an integer from 1 to 65535`)
  }
  if (row.host !== undefined && (typeof row.host !== 'string' || !/^[^\s]{1,255}$/.test(row.host))) throw new HubSettingsError(`${where}: host must be a host name or address`)
  return {
    ...(publicOrigin === undefined ? {} : { publicOrigin }),
    ...(typeof row.host === 'string' ? { host: row.host } : {}),
    ...(typeof row.port === 'number' ? { port: row.port } : {}),
  }
}

export async function loadHubSettings(paths: Pick<HubPaths, 'stateDir'>): Promise<HubSettings> {
  const path = hubSettingsPath(paths)
  let raw: string
  try {
    raw = await readFile(path, 'utf8')
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return {}
    throw error
  }
  let parsed: unknown
  try { parsed = JSON.parse(raw) } catch { throw new HubSettingsError(`${path} is not valid JSON. It was left untouched; fix or remove it.`) }
  return validate(parsed, path)
}

export async function saveHubSettings(paths: Pick<HubPaths, 'stateDir'>, settings: HubSettings): Promise<void> {
  const wire = {
    version: 1,
    ...(settings.publicOrigin === undefined ? {} : { public_origin: settings.publicOrigin }),
    ...(settings.host === undefined ? {} : { host: settings.host }),
    ...(settings.port === undefined ? {} : { port: settings.port }),
  }
  validate(wire, 'Hub settings')
  await mkdir(paths.stateDir, { recursive: true, mode: 0o700 })
  const path = hubSettingsPath(paths)
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
  await writeFile(temporary, `${JSON.stringify(wire, null, 2)}\n`, { mode: 0o600 })
  await chmod(temporary, 0o600)
  await rename(temporary, path)
}

/** The environment `hub run` and the identity commands see: variables first, then saved settings. */
export function settingsEnvironment(environment: Record<string, string | undefined>, settings: HubSettings): Record<string, string | undefined> {
  const unset = (name: string): boolean => !environment[name]?.trim()
  return {
    ...environment,
    ...(settings.publicOrigin !== undefined && unset('XERXES_HUB_PUBLIC_ORIGINS') ? { XERXES_HUB_PUBLIC_ORIGINS: settings.publicOrigin } : {}),
    ...(settings.host !== undefined && unset('XERXES_HUB_HOST') ? { XERXES_HUB_HOST: settings.host } : {}),
    ...(settings.port !== undefined && unset('XERXES_HUB_PORT') ? { XERXES_HUB_PORT: String(settings.port) } : {}),
  }
}
