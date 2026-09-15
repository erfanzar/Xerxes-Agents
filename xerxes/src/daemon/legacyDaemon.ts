// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createConnection } from 'node:net'
import { legacyProjectDaemonPaths } from './paths.js'

/** Never load a second writable copy of a workspace still owned by an old daemon. */
export async function assertLegacyDaemonReleased(cwd: string): Promise<void> {
  const path = legacyProjectDaemonPaths(cwd, { ...process.env, XERXES_DAEMON_SOCKET: '' }).socketPath
  await new Promise<void>((resolve, reject) => {
    const socket = createConnection(path)
    const finish = (error?: Error) => {
      clearTimeout(timer)
      socket.destroy()
      if (error) reject(error)
      else resolve()
    }
    const timer = setTimeout(() => finish(new Error('Could not verify ownership of the previous project daemon; existing work was left untouched')), 2000)
    socket.once('connect', () => finish(new Error('This workspace is still open in a previous project daemon. Continue in its existing window or terminal; close that runtime after its work finishes before opening it in the global daemon.')))
    socket.once('error', (error: NodeJS.ErrnoException) => finish(error.code === 'ENOENT' || error.code === 'ECONNREFUSED' ? undefined : error))
  })
}
