// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Build the desktop app into `dist/desktop`.
 *
 * The theme stylesheet is GENERATED here rather than committed, so the desktop
 * palette cannot be edited independently of `ui/theme.ts` — the split that let
 * every other pair of surfaces in this repo drift apart.
 */

import { cp, mkdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { themeStylesheet } from '../src/desktop/tokens.js'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outputDirectory = join(packageDirectory, 'dist', 'desktop')
const rendererOutput = join(outputDirectory, 'renderer')

async function buildDesktop(): Promise<void> {
  await mkdir(rendererOutput, { recursive: true })

  const renderer = await Bun.build({
    entrypoints: [join(packageDirectory, 'src', 'desktop', 'renderer', 'main.tsx')],
    outdir: rendererOutput,
    target: 'browser',
    format: 'esm',
    minify: true,
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

  await writeFile(join(rendererOutput, 'theme.css'), themeStylesheet(), 'utf8')
  for (const asset of ['index.html', 'base.css']) {
    await cp(join(packageDirectory, 'src', 'desktop', 'renderer', asset), join(rendererOutput, asset))
  }

  console.log(`built ${outputDirectory}`)
}

if (import.meta.main) {
  await buildDesktop()
}
