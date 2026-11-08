// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProfileStore } from '../bridge/profiles.js'
import { loadSystemDaemonConfig } from '../daemon/config.js'
import { parseLocalConfig } from './catalog.js'
import { runLocalCommand, type SaveLocalProfileInput } from './command.js'
import { probeHardware, systemHardwarePorts } from './hardware.js'
import { createLmStudioEngine, systemLmStudioPorts } from './lmstudio.js'
import { probeLoadedWindows, systemWindowFetch } from './loadedWindow.js'
import { createLocalServer, systemServerPorts } from './server.js'

/**
 * `xerxes local …`: production wiring of the managed local engines
 * (llama.cpp's llama-server, or LM Studio through its `lms` CLI).
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
      lmstudio: createLmStudioEngine(systemLmStudioPorts(), local.lmsPath === undefined ? {} : { lmsPath: local.lmsPath }),
      loadedWindow: baseUrl => probeLoadedWindows({ baseUrl }, systemWindowFetch),
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
 * Save the `bonsai-local` profile with the window the engine loaded (LM
 * Studio's `loaded_context_length`, llama-server's `/props` n_ctx). Neither
 * model list reports it, so without this the runtime never knows the window:
 * auto-compaction does not run and a full prompt overflows instead of being
 * compacted. The daemon re-reads it from the server on each model discovery.
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
