// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseLocalConfig } from './catalog.js'
import { engineRecoveryFor, type RecoverableEngine, type RecoverEngine } from './engineRecovery.js'
import { createLmStudioEngine, systemLmStudioPorts } from './lmstudio.js'
import { createLocalServer, systemServerPorts } from './server.js'

/**
 * Recovery for the engines `xerxes local start` manages, with the same
 * binaries and state files. The local settings are read when a recovery runs,
 * so a mistake in them is reported in that turn, not at daemon start.
 */
export function systemEngineRecoveryFor(
  runtime: Record<string, unknown>,
  env: Readonly<Record<string, string | undefined>>,
): (route: { readonly baseUrl: string | undefined; readonly model: string }) => RecoverEngine | undefined {
  const engines: readonly RecoverableEngine[] = [
    {
      recover: async target => {
        const { lmsPath } = parseLocalConfig(runtime, env)
        return createLmStudioEngine(systemLmStudioPorts(), lmsPath === undefined ? {} : { lmsPath }).recover(target)
      },
    },
    {
      recover: async target => {
        const { llamaServerPath } = parseLocalConfig(runtime, env)
        return createLocalServer(systemServerPorts(), llamaServerPath === undefined ? {} : { llamaServerPath }).recover(target)
      },
    },
  ]
  return route => engineRecoveryFor(engines, route)
}
