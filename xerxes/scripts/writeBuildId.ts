// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Stamps the compiled runtime with the source-tree fingerprint. Daemons
 * launched from a bundled cli.js cannot rehash absent TypeScript sources, so
 * they read this sibling `build-id` file instead (see
 * src/daemon/sourceBuild.ts#daemonBuildIdForEntry). Keeping the fingerprint
 * identical between source daemons and bundled daemons of the same build is
 * what lets the desktop's compatibility check trust both equally.
 */

import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { sourceDaemonBuildId } from '../src/daemon/sourceBuild.js'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')

const buildId = await sourceDaemonBuildId(join(packageDirectory, 'src'))
if (!buildId) throw new Error('runtime build could not fingerprint daemon source')

await Bun.write(join(packageDirectory, 'dist', 'build-id'), `${buildId}\n`)
console.log(`runtime build id ${buildId}`)
