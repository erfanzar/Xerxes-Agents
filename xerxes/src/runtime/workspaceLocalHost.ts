// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'

import type { LocalWorkspaceHostPort, WorkspaceExecResult } from './workspaceProvider.js'

export const localWorkspaceHost: LocalWorkspaceHostPort = {
  async spawn(command, options): Promise<WorkspaceExecResult> {
    const child = Bun.spawn([...command], {
      cwd: options.cwd,
      ...(options.env === undefined ? {} : { env: options.env }),
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const [stdout, stderr] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
    ])
    const exitCode = await child.exited
    return { exitCode, stdout, stderr }
  },
  async readFile(path) {
    return Bun.file(path).text()
  },
  async writeFile(path, content) {
    await writeFile(path, content, 'utf8')
  },
  async mkdir(path) {
    await mkdir(path, { recursive: true })
  },
}
