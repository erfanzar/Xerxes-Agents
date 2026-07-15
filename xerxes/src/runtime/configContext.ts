// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AsyncLocalStorage } from 'node:async_hooks'

export const INHERITABLE_CONFIG_KEYS = [
  'model',
  'base_url',
  'api_key',
  'temperature',
  'top_p',
  'top_k',
  'max_tokens',
  'min_p',
  'frequency_penalty',
  'presence_penalty',
  'repetition_penalty',
  'permission_mode',
] as const

export type RuntimeConfig = Readonly<Record<string, unknown>>
export type RuntimeEventCallback = (eventType: string, data: RuntimeConfig) => void

const inheritableKeys = new Set<string>(INHERITABLE_CONFIG_KEYS)
const activeConfigStorage = new AsyncLocalStorage<RuntimeConfig | undefined>()

let globalConfig: RuntimeConfig = {}
let eventCallback: RuntimeEventCallback | undefined

/** Replace the process-wide configuration snapshot with a defensive copy. */
export function setConfig(config: Readonly<Record<string, unknown>>): void {
  globalConfig = { ...config }
}

/** Return a fresh snapshot so consumers cannot mutate process-global state. */
export function getConfig(): Record<string, unknown> {
  return { ...globalConfig }
}

/**
 * Install the active turn configuration for the current async execution chain.
 *
 * New code should favor runWithActiveConfig because it restores context
 * automatically. This convenience is retained for synchronous daemon setup.
 */
export function setActiveConfig(config: Readonly<Record<string, unknown>> | undefined): void {
  activeConfigStorage.enterWith(config === undefined ? undefined : { ...config })
}

/** Return a copy of the configuration bound to this async turn, if any. */
export function getActiveConfig(): Record<string, unknown> | undefined {
  const value = activeConfigStorage.getStore()
  return value === undefined ? undefined : { ...value }
}

/** Execute work with an isolated active turn configuration. */
export function runWithActiveConfig<T>(
  config: Readonly<Record<string, unknown>> | undefined,
  operation: () => T,
): T {
  return activeConfigStorage.run(config === undefined ? undefined : { ...config }, operation)
}

/** Select non-empty values the spawned-agent surface is allowed to inherit. */
export function getInheritableConfig(): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(globalConfig).filter(([key, value]) => (
      inheritableKeys.has(key) && value !== undefined && value !== null && value !== ''
    )),
  )
}

/** Install or remove the optional process-level runtime event sink. */
export function setEventCallback(callback: RuntimeEventCallback | undefined): void {
  eventCallback = callback
}

/** Return the installed runtime event sink, if any. */
export function getEventCallback(): RuntimeEventCallback | undefined {
  return eventCallback
}

/**
 * Publish a lightweight event without allowing observer failures to break a turn.
 *
 * The callback boundary is intentionally isolated because telemetry and bridge
 * observers are not authorized to alter execution control flow.
 */
export function emitRuntimeEvent(eventType: string, data: Readonly<Record<string, unknown>>): void {
  const callback = eventCallback
  if (callback === undefined) return
  try {
    callback(eventType, { ...data })
  } catch {
    // Observability callbacks must never terminate a model/tool turn.
  }
}
