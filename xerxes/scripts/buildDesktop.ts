// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Build the desktop app into `dist/desktop`.
 *
 * The theme stylesheet is GENERATED here rather than committed, so the desktop
 * palette cannot be edited independently of `ui/theme.ts` — the split that let
 * every other pair of surfaces in this repo drift apart.
 */

import { cp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { DAEMON_PROTOCOL_VERSION } from '../src/daemon/fingerprint.js'
import { sourceDaemonBuildId } from '../src/daemon/sourceBuild.js'
import { themeStylesheet } from '../src/desktop/tokens.js'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outputDirectory = join(packageDirectory, 'dist', 'desktop')
const rendererOutput = join(outputDirectory, 'renderer')

async function buildDesktop(): Promise<void> {
  await mkdir(rendererOutput, { recursive: true })
  const packageRecord = JSON.parse(await readFile(join(packageDirectory, 'package.json'), 'utf8')) as {
    readonly version?: unknown
  }
  const desktopVersion = typeof packageRecord.version === 'string' ? packageRecord.version : ''
  if (!desktopVersion) throw new Error('desktop build needs package.json version')
  const expectedDaemonBuildId = await sourceDaemonBuildId(join(packageDirectory, 'src'))
  if (!expectedDaemonBuildId) throw new Error('desktop build could not fingerprint daemon source')

  const renderer = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'desktop', 'renderer', 'main.tsx')],
    outdir: rendererOutput,
    target: 'browser',
    format: 'esm',
    minify: true,
    define: {
      __XERXES_DESKTOP_VERSION__: JSON.stringify(desktopVersion),
      __XERXES_DESKTOP_PROTOCOL__: String(DAEMON_PROTOCOL_VERSION),
      __XERXES_EXPECTED_DAEMON_BUILD_ID__: JSON.stringify(expectedDaemonBuildId),
    },
  })
  if (!renderer.success) {
    for (const log of renderer.logs) console.error(log)
    throw new Error('desktop renderer build failed')
  }

  const main = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'desktop', 'main.ts')],
    outdir: outputDirectory,
    target: 'node',
    format: 'esm',
    // Electron supplies its own runtime; bundling it would ship a second copy.
    external: ['electron'],
  })
  if (!main.success) {
    for (const log of main.logs) console.error(log)
    throw new Error('desktop main-process build failed')
  }

  // The sandboxed preload must be CommonJS: package.json declares
  // "type": "module", so the bundler would otherwise emit ESM, which
  // sandboxed preloads do not support. format: 'cjs' pins module.exports.
  const preload = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'desktop', 'preload.ts')],
    outdir: outputDirectory,
    target: 'node',
    format: 'cjs',
    minify: true,
    external: ['electron'],
  })
  if (!preload.success) {
    for (const log of preload.logs) console.error(log)
    throw new Error('desktop preload build failed')
  }

  await writeFile(
    join(outputDirectory, 'package.json'),
    `${JSON.stringify({
      name: 'xerxes-agents-desktop',
      productName: 'Xerxes Agents',
      version: desktopVersion,
      private: true,
      type: 'module',
      main: 'main.js',
    }, null, 2)}\n`,
    'utf8',
  )
  await writeFile(join(rendererOutput, 'theme.css'), themeStylesheet(), 'utf8')
  for (const asset of ['index.html', 'app.css']) {
    await cp(join(packageDirectory, 'src', 'desktop', 'renderer', asset), join(rendererOutput, asset))
  }
  // The renderer's brand asset ships inside the bundle so CSP img-src 'self'
  // covers it; the dock icon reads the full-resolution source from assets/.
  await cp(
    join(packageDirectory, '..', 'assets', 'logo-128.png'),
    join(rendererOutput, 'logo-128.png'),
  )

  console.log(`built ${outputDirectory}`)
}

if (import.meta.main) {
  await buildDesktop()
}
