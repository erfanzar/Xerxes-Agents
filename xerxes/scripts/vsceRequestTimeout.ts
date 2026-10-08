// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Preloaded into `vsce publish` by publishVscode.ts. vsce's Marketplace client
 * gives a request three minutes of socket silence and then fails it with
 * "Request timeout"; a platform package carries its own Bun (30–48 MB), so on
 * a slow uplink the upload plus the Marketplace's verification outlasts that
 * and every publish failed. This raises the client's default wait; a value
 * the client is given explicitly still wins.
 */
import { realpathSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'

const REQUEST_TIMEOUT_MS = 30 * 60_000

const vsce = realpathSync(join(import.meta.dir, '..', 'node_modules', '@vscode', 'vsce'))
const webApi = createRequire(join(vsce, 'package.json')).resolve('azure-devops-node-api/WebApi')
const { HttpClient } = createRequire(webApi)('typed-rest-client/HttpClient') as { HttpClient: { prototype: object } }
const explicit = new WeakMap<object, number | undefined>()
Object.defineProperty(HttpClient.prototype, '_socketTimeout', {
  configurable: true,
  get(this: object) { return explicit.get(this) ?? REQUEST_TIMEOUT_MS },
  set(this: object, value: number | undefined) { explicit.set(this, value) },
})
