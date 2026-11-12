// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProfileStore } from '../bridge/profiles.js'
import { loadSystemDaemonConfig } from '../daemon/config.js'
import { parseLocalConfig } from './catalog.js'
import { runLocalCommand, type LocalCommandPorts, type SaveLocalProfileInput } from './command.js'
import { ggufAttention, readGgufMetadata } from './ggufMetadata.js'
import { probeHardware, systemHardwarePorts } from './hardware.js'
import { createLmStudioEngine, systemLmStudioPorts } from './lmstudio.js'
import { probeLoadedWindows, systemWindowFetch } from './loadedWindow.js'
import { freeMemoryBytes } from './memoryFit.js'
import { readMlxAttention } from './mlxConfig.js'
import { createLocalServer, systemServerPorts } from './server.js'

/** What a caller of `runLocalCommand` supplies itself: where output goes and how to stop. */
export interface LocalCommandIo {
  stdout(text: string): void
  stderr(text: string): void
  readonly signal?: AbortSignal
}

/**
 * Production ports for `runLocalCommand`: the managed local engines (llama.cpp's
 * llama-server, mlx-lm's mlx_lm.server, or LM Studio through its `lms` CLI),
 * real hardware probes, downloads and the profile store. The daemon config is
 * read on each call, so a setup run from the desktop sees the current settings.
 */
export function systemLocalCommandPorts(io: LocalCommandIo): LocalCommandPorts {
  const runtime = loadSystemDaemonConfig().runtime
  const local = parseLocalConfig(runtime, process.env)
  return {
    hardware: () => probeHardware(systemHardwarePorts()),
    fetch: (url, init) => fetch(url, init),
    server: createLocalServer(systemServerPorts(), local.llamaServerPath === undefined ? {} : { binaryPath: local.llamaServerPath }),
    mlx: createLocalServer(systemServerPorts(), { engine: 'mlx', ...(local.mlxServerPath === undefined ? {} : { binaryPath: local.mlxServerPath }) }),
    lmstudio: createLmStudioEngine(systemLmStudioPorts(), local.lmsPath === undefined ? {} : { lmsPath: local.lmsPath }),
    loadedWindow: baseUrl => probeLoadedWindows({ baseUrl }, systemWindowFetch),
    memoryFit: {
      freeBytes: hw => freeMemoryBytes(hw, systemHardwarePorts().run),
      attention: async path => ggufAttention(path, await readGgufMetadata(path)),
      weightsBytes: path => Bun.file(path).size,
      mlxAttention: readMlxAttention,
    },
    saveProfile: async input => saveLocalProfile(new ProfileStore(), input),
    stdout: io.stdout,
    stderr: io.stderr,
    config: runtime,
    env: process.env,
    ...(io.signal === undefined ? {} : { signal: io.signal }),
  }
}

/**
 * `xerxes local …`. Nothing here runs at startup; every action is an explicit
 * subcommand. The saved `bonsai-local` profile is marked `data_trust: local`
 * and `prompt_profile: lean`, and it is not made the active profile: choosing
 * a mode (Private or Hybrid) in the desktop app, or `/provider`, does that.
 */
export async function runLocalCli(argv: readonly string[]): Promise<number> {
  const controller = new AbortController()
  const onSignal = () => controller.abort()
  process.once('SIGINT', onSignal)
  try {
    return await runLocalCommand([...argv], systemLocalCommandPorts({
      stdout: text => { process.stdout.write(text) },
      stderr: text => { process.stderr.write(text) },
      signal: controller.signal,
    }))
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
