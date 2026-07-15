// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

/** IDE-discovery metadata for the Bun-native `xerxes-acp` executable. */
export const ACP_REGISTRY_METADATA = {
  name: 'xerxes',
  display_name: 'Xerxes',
  description: 'Multi-agent coding assistant built on Xerxes-Agents.',
  version: '0.2.6',
  vendor: 'Erfan Zare Chavoshi',
  license: 'Apache-2.0',
  homepage: 'https://github.com/erfanzar/Xerxes-Agents',
  distribution: {
    type: 'command',
    command: 'xerxes-acp',
    args: [],
  },
  capabilities: {
    streaming: true,
    tools: true,
    permissions: true,
    sessions: true,
    fork: true,
    cancel: true,
    models: true,
  },
} as const

/** Compatibility name retained for callers that used the Python adapter's metadata export. */
export const REGISTRY_METADATA = ACP_REGISTRY_METADATA

export function defaultAcpRegistryDirectory(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  homeDirectory = homedir(),
): string {
  return join(environment.XDG_CONFIG_HOME ?? join(homeDirectory, '.config'), 'agent-registry')
}

/** Write the standard ACP `agent.json` manifest and return its absolute path. */
export async function writeAcpRegistryFile(targetDirectory?: string): Promise<string> {
  const directory = join(targetDirectory ?? defaultAcpRegistryDirectory(), 'xerxes')
  await mkdir(directory, { recursive: true })
  const output = join(directory, 'agent.json')
  await writeFile(output, `${JSON.stringify(ACP_REGISTRY_METADATA, null, 2)}\n`, 'utf8')
  return output
}
