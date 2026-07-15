// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from './paths.js'

/** A caller-supplied snapshot or resolver-backed view of environment values. */
export type DaemonEnvironment = Readonly<Record<string, string | undefined>>

/** Flexible nested settings retained from the daemon JSON document. */
export type DaemonSettings = Record<string, unknown>

/** Explicit synchronous text source for the daemon JSON configuration file. */
export type DaemonConfigFileReader = (path: string) => string | undefined

/** Nested daemon configuration consumed by the Bun runtime. */
export interface DaemonConfig {
  readonly channels: Record<string, DaemonSettings>
  readonly control: DaemonSettings
  readonly maxConcurrentTurns: number
  readonly projectDirectory: string
  readonly runtime: DaemonSettings
  readonly workspace: DaemonSettings
}

/** All host-dependent data required by the pure daemon config loader. */
export interface DaemonConfigLoadOptions {
  readonly environment: DaemonEnvironment
  readonly home: string
  readonly projectDirectory: string
  readonly readFile: DaemonConfigFileReader
}

/** Optional overrides for the explicit host-bound wrapper used by the Bun CLI. */
export interface SystemDaemonConfigLoadOptions {
  readonly environment?: DaemonEnvironment
  readonly home?: string
  readonly projectDirectory?: string
  readonly readFile?: DaemonConfigFileReader
}

/** Resolve the daemon JSON path from an explicit Xerxes home directory. */
export function daemonConfigPath(home: string): string {
  return join(requiredPath(home, 'home'), 'daemon', 'config.json')
}

/**
 * Load nested or historical flat daemon configuration without reading host globals.
 *
 * Callers supply the environment, home directory, project directory, and file reader
 * so tests and embedding hosts control every input. Unreadable or malformed JSON uses
 * defaults, matching the Python daemon's resilient configuration behavior.
 */
export function loadDaemonConfig(options: DaemonConfigLoadOptions): DaemonConfig {
  const inputs = normalizeLoadOptions(options)
  const config: MutableDaemonConfig = {
    runtime: {},
    control: {
      websocket_host: '127.0.0.1',
      websocket_port: 11996,
      unix_socket: join(inputs.home, 'daemon', 'xerxes.sock'),
      pid_file: join(inputs.home, 'daemon', 'daemon.pid'),
      log_dir: join(inputs.home, 'daemon', 'logs'),
    },
    workspace: {
      root: join(inputs.home, 'agents'),
      default_agent_id: 'default',
    },
    channels: {},
    projectDirectory: inputs.projectDirectory,
    maxConcurrentTurns: 8,
  }
  mergeDaemonConfig(config, readJsonObject(inputs.readFile, daemonConfigPath(inputs.home)))
  applyEnvironment(config, inputs.environment)
  return config
}

/**
 * Load daemon configuration from the Bun host process.
 *
 * The ambient environment, home directory, current directory, and filesystem access
 * are intentionally isolated in this CLI-oriented wrapper; use {@link loadDaemonConfig}
 * for embedding and tests.
 */
export function loadSystemDaemonConfig(options: SystemDaemonConfigLoadOptions = {}): DaemonConfig {
  const environment = options.environment ?? process.env
  return loadDaemonConfig({
    environment,
    home: options.home ?? xerxesHome(environment),
    projectDirectory: options.projectDirectory ?? process.cwd(),
    readFile: options.readFile ?? readSystemFile,
  })
}

/** Return a copy of runtime settings with explicit environment references resolved. */
export function daemonRuntime(config: DaemonConfig, environment: DaemonEnvironment): DaemonSettings {
  return resolveEnvironmentReferences(config.runtime, environment)
}

/** Return a copy of every channel setting with explicit environment references resolved. */
export function daemonChannels(
  config: DaemonConfig,
  environment: DaemonEnvironment,
): Record<string, DaemonSettings> {
  assertEnvironment(environment)
  const resolved: Record<string, DaemonSettings> = {}
  for (const [name, channel] of Object.entries(config.channels)) {
    const item = copyRecord(channel)
    const settings = ownValue(channel, 'settings')
    defineValue(item, 'settings', isRecord(settings) ? resolveEnvironmentReferences(settings, environment) : {})
    defineValue(resolved, name, item)
  }
  return resolved
}

/** Resolve `*_env` and `env:NAME` values from a caller-supplied environment. */
export function resolveEnvironmentReferences(
  settings: DaemonSettings,
  environment: DaemonEnvironment,
): DaemonSettings {
  if (!isRecord(settings)) {
    throw new ValidationError('daemonSettings', 'must be a settings object', settings)
  }
  assertEnvironment(environment)
  const resolved: DaemonSettings = {}
  for (const [key, value] of Object.entries(settings)) {
    if (key.endsWith('_env') && typeof value === 'string') {
      defineValue(resolved, key.slice(0, -4), environmentValue(environment, value) ?? '')
      continue
    }
    if (typeof value === 'string' && value.startsWith('env:')) {
      defineValue(resolved, key, environmentValue(environment, value.slice(4)) ?? '')
      continue
    }
    defineValue(resolved, key, value)
  }
  return resolved
}

interface MutableDaemonConfig {
  channels: Record<string, DaemonSettings>
  control: DaemonSettings
  maxConcurrentTurns: number
  projectDirectory: string
  runtime: DaemonSettings
  workspace: DaemonSettings
}

interface NormalizedDaemonConfigLoadOptions {
  readonly environment: DaemonEnvironment
  readonly home: string
  readonly projectDirectory: string
  readonly readFile: DaemonConfigFileReader
}

function normalizeLoadOptions(options: DaemonConfigLoadOptions): NormalizedDaemonConfigLoadOptions {
  if (!isRecord(options)) {
    throw new ValidationError('daemonConfigOptions', 'must provide explicit config loader inputs', options)
  }
  if (typeof options.readFile !== 'function') {
    throw new ValidationError('readFile', 'must be a synchronous daemon config file reader', options.readFile)
  }
  assertEnvironment(options.environment)
  return {
    environment: options.environment,
    home: requiredPath(options.home, 'home'),
    projectDirectory: requiredPath(options.projectDirectory, 'projectDirectory'),
    readFile: options.readFile,
  }
}

function mergeDaemonConfig(config: MutableDaemonConfig, raw: DaemonSettings): void {
  mergeRecord(config.runtime, ownValue(raw, 'runtime'))
  mergeRecord(config.control, ownValue(raw, 'control'))
  mergeRecord(config.workspace, ownValue(raw, 'workspace'))
  const channels = ownValue(raw, 'channels')
  if (isRecord(channels)) {
    for (const [name, value] of Object.entries(channels)) {
      if (isRecord(value)) {
        defineValue(config.channels, name, copyRecord(value))
      }
    }
  }

  const legacyMap: Readonly<Record<string, (value: unknown) => void>> = {
    ws_host: value => { defineValue(config.control, 'websocket_host', String(value)) },
    ws_port: value => { defineValue(config.control, 'websocket_port', integer(value, 11996)) },
    socket_path: value => { defineValue(config.control, 'unix_socket', String(value)) },
    pid_file: value => { defineValue(config.control, 'pid_file', String(value)) },
    log_dir: value => { defineValue(config.control, 'log_dir', String(value)) },
    auth_token: value => { defineValue(config.control, 'auth_token', String(value)) },
    model: value => { defineValue(config.runtime, 'model', String(value)) },
    base_url: value => { defineValue(config.runtime, 'base_url', String(value)) },
    api_key: value => { defineValue(config.runtime, 'api_key', String(value)) },
    max_concurrent_tasks: value => { config.maxConcurrentTurns = integer(value, config.maxConcurrentTurns) },
    max_concurrent_turns: value => { config.maxConcurrentTurns = integer(value, config.maxConcurrentTurns) },
    project_dir: value => { config.projectDirectory = String(value) },
  }
  for (const [key, apply] of Object.entries(legacyMap)) {
    if (hasOwn(raw, key)) {
      apply(raw[key])
    }
  }
}

function applyEnvironment(config: MutableDaemonConfig, environment: DaemonEnvironment): void {
  const daemonHost = configuredEnvironmentValue(environment, 'XERXES_DAEMON_HOST')
  if (daemonHost !== undefined) {
    defineValue(config.control, 'websocket_host', daemonHost)
  }
  const daemonPort = configuredEnvironmentValue(environment, 'XERXES_DAEMON_PORT')
  if (daemonPort !== undefined) {
    defineValue(config.control, 'websocket_port', integer(daemonPort, 11996))
  }
  const daemonSocket = configuredEnvironmentValue(environment, 'XERXES_DAEMON_SOCKET')
  if (daemonSocket !== undefined) {
    defineValue(config.control, 'unix_socket', daemonSocket)
  }
  const daemonToken = configuredEnvironmentValue(environment, 'XERXES_DAEMON_TOKEN')
  if (daemonToken !== undefined) {
    defineValue(config.control, 'auth_token', daemonToken)
  }
  const maxTasks = configuredEnvironmentValue(environment, 'XERXES_MAX_TASKS')
  if (maxTasks !== undefined) {
    config.maxConcurrentTurns = integer(maxTasks, config.maxConcurrentTurns)
  }
  const maxTurns = configuredEnvironmentValue(environment, 'XERXES_MAX_TURNS')
  if (maxTurns !== undefined) {
    config.maxConcurrentTurns = integer(maxTurns, config.maxConcurrentTurns)
  }
  for (const [name, key] of [
    ['XERXES_MODEL', 'model'],
    ['XERXES_BASE_URL', 'base_url'],
    ['XERXES_API_KEY', 'api_key'],
    ['XERXES_PERMISSION_MODE', 'permission_mode'],
  ] as const) {
    const value = configuredEnvironmentValue(environment, name)
    if (value !== undefined) {
      defineValue(config.runtime, key, value)
    }
  }
  if (configuredEnvironmentValue(environment, 'XERXES_DAEMON_ENABLE_TELEGRAM') !== undefined) {
    const settings = enableChannel(config, 'telegram', 'telegram')
    setDefault(settings, 'token_env', 'TELEGRAM_BOT_TOKEN')
  }
  if (configuredEnvironmentValue(environment, 'XERXES_DAEMON_ENABLE_DISCORD') !== undefined) {
    const settings = enableChannel(config, 'discord', 'discord')
    const discordBotToken = configuredEnvironmentValue(environment, 'DISCORD_BOT_TOKEN')
    setDefault(settings, 'token_env', discordBotToken === undefined ? 'DISCORD_TOKEN' : 'DISCORD_BOT_TOKEN')
    setDefault(settings, 'transport', 'gateway')
    setDefault(settings, 'require_mention', true)
    setDefaultFromEnvironment(settings, 'application_id', environment, 'XERXES_DISCORD_APPLICATION_ID')
    setDefaultFromEnvironment(settings, 'register_commands', environment, 'XERXES_DISCORD_REGISTER_COMMANDS')
    setDefaultFromEnvironment(settings, 'allowed_channel_names', environment, 'XERXES_DISCORD_CHANNEL_NAME')
    setDefaultFromEnvironment(settings, 'allowed_channel_ids', environment, 'XERXES_DISCORD_CHANNEL_ID')
    setDefaultFromEnvironment(settings, 'allowed_guild_ids', environment, 'XERXES_DISCORD_GUILD_ID')
    setDefault(settings, 'instance_name', firstConfiguredEnvironment(
      environment,
      'XERXES_DISCORD_INSTANCE_NAME',
      'XERXES_DISCORD_DEVICE_NAME',
    ))
    setDefault(settings, 'address_names', firstConfiguredEnvironment(
      environment,
      'XERXES_DISCORD_ADDRESS_NAME',
      'XERXES_DISCORD_WAKE_NAME',
    ))
  }
}

function enableChannel(config: MutableDaemonConfig, name: string, type: string): DaemonSettings {
  const existing = ownValue(config.channels, name)
  const channel = isRecord(existing) ? copyRecord(existing) : {}
  defineValue(channel, 'enabled', true)
  setDefault(channel, 'type', type)
  const existingSettings = ownValue(channel, 'settings')
  const settings = isRecord(existingSettings) ? copyRecord(existingSettings) : {}
  defineValue(channel, 'settings', settings)
  defineValue(config.channels, name, channel)
  return settings
}

function setDefaultFromEnvironment(
  target: DaemonSettings,
  key: string,
  environment: DaemonEnvironment,
  environmentName: string,
): void {
  setDefault(target, key, configuredEnvironmentValue(environment, environmentName))
}

function firstConfiguredEnvironment(
  environment: DaemonEnvironment,
  ...names: readonly string[]
): string | undefined {
  for (const name of names) {
    const value = configuredEnvironmentValue(environment, name)
    if (value !== undefined) {
      return value
    }
  }
  return undefined
}

function mergeRecord(target: DaemonSettings, source: unknown): void {
  if (!isRecord(source)) {
    return
  }
  for (const [key, value] of Object.entries(source)) {
    defineValue(target, key, value)
  }
}

function readJsonObject(readFile: DaemonConfigFileReader, path: string): DaemonSettings {
  try {
    const text = readFile(path)
    if (typeof text !== 'string') {
      return {}
    }
    const parsed: unknown = JSON.parse(text)
    return isRecord(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function readSystemFile(path: string): string | undefined {
  try {
    return readFileSync(path, 'utf8')
  } catch {
    return undefined
  }
}

function requiredPath(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.trim() === '' || value.includes('\0')) {
    throw new ValidationError(field, 'must be a non-empty path without NUL bytes', value)
  }
  return value
}

function assertEnvironment(environment: DaemonEnvironment): void {
  if (environment === null || typeof environment !== 'object' || Array.isArray(environment)) {
    throw new ValidationError('daemonEnvironment', 'must be an explicit environment map', environment)
  }
}

function configuredEnvironmentValue(environment: DaemonEnvironment, name: string): string | undefined {
  const value = environmentValue(environment, name)
  return value === '' ? undefined : value
}

function environmentValue(environment: DaemonEnvironment, name: string): string | undefined {
  if (!hasOwn(environment, name)) {
    return undefined
  }
  const value = environment[name]
  return typeof value === 'string' ? value : undefined
}

function integer(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value)
  }
  if (typeof value === 'string' && /^[+-]?\d+$/.test(value)) {
    const parsed = Number(value)
    return Number.isSafeInteger(parsed) ? parsed : fallback
  }
  return fallback
}

function copyRecord(source: DaemonSettings): DaemonSettings {
  const copied: DaemonSettings = {}
  mergeRecord(copied, source)
  return copied
}

function setDefault(target: DaemonSettings, key: string, value: unknown): void {
  if (value !== undefined && !hasOwn(target, key)) {
    defineValue(target, key, value)
  }
}

function ownValue(record: DaemonSettings, key: string): unknown {
  return hasOwn(record, key) ? record[key] : undefined
}

function hasOwn(record: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

function defineValue(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
