// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DaemonConfig, DaemonEnvironment } from './config.js'
import { WebhookMonitorHub } from '../runtime/webhookMonitorSource.js'

/** Secrets are resolved only in the host, never supplied by model calls or RPC. */
export function daemonMonitorWebhookHub(config: DaemonConfig, environment: DaemonEnvironment): WebhookMonitorHub | undefined {
  const value = config.runtime.monitor_webhooks
  if (value === undefined) return undefined
  if (!isRecord(value)) throw new TypeError('runtime.monitor_webhooks must be an object')
  if (Object.keys(value).some(key => !['host', 'port', 'sources'].includes(key))) throw new TypeError('Unknown runtime.monitor_webhooks setting')
  const host = value.host ?? '127.0.0.1'
  const port = value.port ?? 11998
  if (typeof host !== 'string' || !host.trim() || /[\s\0]/.test(host)) throw new TypeError('monitor_webhooks.host must be a hostname or IP address')
  if (typeof port !== 'number' || !Number.isInteger(port) || port < 0 || port > 65535) throw new TypeError('monitor_webhooks.port must be an integer from 0 to 65535')
  if (!Array.isArray(value.sources) || !value.sources.length || value.sources.length > 64) throw new TypeError('monitor_webhooks.sources must contain 1–64 configured sources')
  const seenNames = new Set<string>()
  const sources = value.sources.map((entry: unknown) => {
    if (!isRecord(entry) || Object.keys(entry).some(key => !['name', 'secret_env'].includes(key)) || typeof entry.name !== 'string' || !/^[a-zA-Z0-9_-]{1,64}$/.test(entry.name)
      || typeof entry.secret_env !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(entry.secret_env)) throw new TypeError('Each webhook source requires a valid name and secret_env')
    if (seenNames.has(entry.name)) throw new TypeError(`Duplicate webhook source name: ${entry.name}`)
    seenNames.add(entry.name)
    const secret = environment[entry.secret_env]
    if (typeof secret !== 'string' || Buffer.byteLength(secret, 'utf8') < 32) throw new TypeError(`Webhook source ${entry.name} requires at least 32 secret bytes in ${entry.secret_env}`)
    return { name: entry.name, secret }
  })
  return new WebhookMonitorHub({ host, port, sources })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
