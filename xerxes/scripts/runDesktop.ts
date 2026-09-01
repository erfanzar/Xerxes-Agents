// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Launches the branded bundle on macOS and the Electron host elsewhere. */

import { execFileSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const bundleBinary = join(packageDirectory, 'dist', 'Xerxes Agents.app', 'Contents', 'MacOS', 'Xerxes Agents')

// A running instance holds the OLD renderer in memory: Electron's
// single-instance handoff means launching again just focuses the stale
// window, and the freshly built bundle (new daemon fingerprint, new UI)
// never loads. Terminate the old instance first so `bun run desktop` always
// opens what it just built. scoped to this exact bundle path — other apps
// named alike are untouched.
if (process.platform === 'darwin') {
  try {
    const listing = execFileSync('pgrep', ['-fl', bundleBinary], { encoding: 'utf8' })
    const pids = listing
      .split('\n')
      .map(line => Number.parseInt(line.split(' ', 1)[0] ?? '', 10))
      .filter(pid => Number.isFinite(pid) && pid !== process.pid)
    for (const pid of pids) {
      try {
        process.kill(pid, 'SIGTERM')
      } catch {
        // already gone
      }
    }
    if (pids.length) {
      const deadline = Date.now() + 3_000
      while (Date.now() < deadline) {
        const alive = pids.some(pid => {
          try {
            process.kill(pid, 0)
            return true
          } catch {
            return false
          }
        })
        if (!alive) break
        await new Promise(resolve => setTimeout(resolve, 100))
      }
    }
  } catch {
    // pgrep exits 1 when nothing matches — the common, fine case.
  }
}

const command = process.platform === 'darwin'
  ? [bundleBinary]
  : ['bunx', '--bun', '--no-install', 'electron', join(packageDirectory, 'dist', 'desktop')]

const desktop = Bun.spawn(command, {
  cwd: packageDirectory,
  stdin: 'inherit',
  stdout: 'inherit',
  stderr: 'inherit',
})
process.exitCode = await desktop.exited
