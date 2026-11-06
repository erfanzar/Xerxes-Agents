// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProfileStore } from '../bridge/profiles.js'
import { loadSystemDaemonConfig } from '../daemon/config.js'
import { parseLocalConfig } from './catalog.js'
import { runLocalCommand, type SaveLocalProfileInput } from './command.js'
import { probeHardware, systemHardwarePorts } from './hardware.js'
import { createLocalServer, systemServerPorts } from './server.js'

/**
 * `xerxes local …`: production wiring of the managed llama.cpp runtime.
 *
 * Nothing here runs at startup; every action is an explicit subcommand. The
 * saved `bonsai-local` profile is marked `data_trust: local` and
 * `prompt_profile: lean`, and it is not made the active profile.
 */
export async function runLocalCli(argv: readonly string[]): Promise<number> {
  const config = loadSystemDaemonConfig()
  const runtime = config.runtime
  const local = parseLocalConfig(runtime, process.env)
  const controller = new AbortController()
  const onSignal = () => controller.abort()
  process.once('SIGINT', onSignal)
  try {
    return await runLocalCommand([...argv], {
      hardware: () => probeHardware(systemHardwarePorts()),
      fetch: (url, init) => fetch(url, init),
      server: createLocalServer(systemServerPorts(), local.llamaServerPath === undefined ? {} : { llamaServerPath: local.llamaServerPath }),
      saveProfile: async input => saveLocalProfile(new ProfileStore(), input),
      stdout: text => { process.stdout.write(text) },
      stderr: text => { process.stderr.write(text) },
      config: runtime,
      env: process.env,
      signal: controller.signal,
    })
  } finally {
    process.off('SIGINT', onSignal)
  }
}

/**
 * Save the `bonsai-local` profile with the window llama-server was launched
 * with. llama.cpp's model list reports no context size, so without this the
 * runtime never knows the window: auto-compaction does not run and a full
 * prompt overflows instead of being compacted.
 */
export function saveLocalProfile(store: ProfileStore, input: SaveLocalProfileInput): void {
  store.save({
    name: input.name,
    apiKey: '',
    baseUrl: input.baseUrl,
    model: input.model,
    provider: input.provider,
    dataTrust: input.dataTrust,
    promptProfile: input.promptProfile,
    setActive: false,
  })
  store.replaceModelCapabilities(input.name, { [input.model]: { context_limit: input.contextLimit } })
}
