// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { glob, stat, writeFile } from 'node:fs/promises'
import { runMachineCommand } from '../../daemon/machineCommand.js'
import { browseSshFolders, readSshHosts } from '../../daemon/machineDiscovery.js'

/** Electron supplies filesystem ports; validation, locking and SSH policy stay shared with Bun. */
export async function desktopMachineCommand(registryPath: string, command: string, home?: string): Promise<Record<string, unknown>> {
  return runMachineCommand(registryPath, command, {
    hosts: () => readSshHosts(home, async function* (pattern) {
      for await (const file of glob(pattern)) {
        if ((await stat(file)).isFile()) yield file
      }
    }),
    browse: browseSshFolders,
  }, {
    write: (path, content) => writeFile(path, content, { mode: 0o600 }),
  })
}
