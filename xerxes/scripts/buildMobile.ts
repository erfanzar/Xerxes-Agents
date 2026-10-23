// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Build the phone app into `dist/mobile`:
 *
 *   index.html          shell, references the hashed bundle
 *   assets/*-<hash>.*   hashed JS and CSS (cacheable forever by the Hub)
 *   sw.js               service worker; the shell list is baked in
 *   manifest.webmanifest
 *   icons/*.png         generated from assets/logo-512.png with fast-png
 *
 * The Hub serves this directory as its static root.
 */

import { createHash } from 'node:crypto'
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { decode, encode } from 'fast-png'

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = join(packageDirectory, '..')
const sourceDirectory = join(packageDirectory, 'src', 'mobile')
const outputDirectory = join(packageDirectory, 'dist', 'mobile')

interface Raster {
  readonly width: number
  readonly height: number
  readonly channels: number
  readonly data: Uint8Array
}

/** Area-average downscale: good enough for app icons and needs no native dependency. */
export function downscale(source: Raster, size: number): Raster {
  const { width, height, channels, data } = source
  const out = new Uint8Array(size * size * channels)
  for (let y = 0; y < size; y += 1) {
    const y0 = Math.floor((y * height) / size)
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / size))
    for (let x = 0; x < size; x += 1) {
      const x0 = Math.floor((x * width) / size)
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / size))
      for (let c = 0; c < channels; c += 1) {
        let sum = 0
        let count = 0
        for (let yy = y0; yy < y1; yy += 1) {
          for (let xx = x0; xx < x1; xx += 1) {
            sum += data[(yy * width + xx) * channels + c] ?? 0
            count += 1
          }
        }
        out[(y * size + x) * channels + c] = Math.round(sum / count)
      }
    }
  }
  return { width: size, height: size, channels, data: out }
}

async function writeIcons(): Promise<void> {
  const iconDirectory = join(outputDirectory, 'icons')
  await mkdir(iconDirectory, { recursive: true })
  const logo = decode(new Uint8Array(await readFile(join(repositoryRoot, 'assets', 'logo-512.png'))))
  if (logo.depth !== 8 || (logo.channels !== 3 && logo.channels !== 4)) {
    throw new Error(`assets/logo-512.png must be 8-bit RGB or RGBA, got depth ${logo.depth} with ${logo.channels} channels`)
  }
  const raster: Raster = { width: logo.width, height: logo.height, channels: logo.channels, data: logo.data as Uint8Array }
  for (const [name, size] of [['icon-192.png', 192], ['icon-512.png', 512], ['apple-touch-icon.png', 180]] as const) {
    const scaled = size === raster.width ? raster : downscale(raster, size)
    await writeFile(
      join(iconDirectory, name),
      encode({ width: scaled.width, height: scaled.height, channels: scaled.channels, depth: 8, data: scaled.data }),
    )
  }
}

async function buildMobile(): Promise<void> {
  await rm(outputDirectory, { recursive: true, force: true })
  await mkdir(join(outputDirectory, 'assets'), { recursive: true })

  const app = await Bun.build({
    entrypoints: [join(sourceDirectory, 'main.tsx'), join(sourceDirectory, 'mobile.css')],
    outdir: join(outputDirectory, 'assets'),
    target: 'browser',
    format: 'esm',
    minify: true,
    naming: { entry: '[name]-[hash].[ext]', chunk: '[name]-[hash].[ext]', asset: '[name]-[hash].[ext]' },
    define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  })
  if (!app.success) {
    for (const log of app.logs) console.error(log)
    throw new Error('mobile bundle build failed')
  }
  // Minification alone does not select React's production entry point.
  for (const output of app.outputs) {
    if (output.path.endsWith('.js') && (await output.text()).includes('Download the React DevTools')) {
      throw new Error('Mobile bundle contains development React')
    }
  }

  const script = app.outputs.find(output => output.kind === 'entry-point' && output.path.endsWith('.js'))
  const style = app.outputs.find(output => output.path.endsWith('.css'))
  if (script === undefined) throw new Error('mobile build produced no JavaScript entry point')
  if (style === undefined) throw new Error('mobile build produced no stylesheet')
  const scriptName = `/assets/${basename(script.path)}`
  const styleName = `/assets/${basename(style.path)}`

  const html = (await readFile(join(sourceDirectory, 'index.html'), 'utf8')).replace('{{SCRIPT}}', scriptName).replace('{{STYLE}}', styleName)
  if (html.includes('{{')) throw new Error('index.html still has an unreplaced placeholder')
  await writeFile(join(outputDirectory, 'index.html'), html, 'utf8')

  // Shell = index.html plus content-hashed assets. Never an /api/ path.
  const shell = ['/index.html', scriptName, styleName]
  const version = createHash('sha256').update(shell.join('\n')).digest('hex').slice(0, 12)
  const worker = await Bun.build({
    entrypoints: [join(sourceDirectory, 'pwa', 'sw.ts')],
    outdir: outputDirectory,
    target: 'browser',
    format: 'esm',
    minify: true,
    naming: { entry: 'sw.js' },
    define: { __XERXES_SHELL__: JSON.stringify(shell), __XERXES_SHELL_VERSION__: JSON.stringify(version) },
  })
  if (!worker.success) {
    for (const log of worker.logs) console.error(log)
    throw new Error('service worker build failed')
  }

  await cp(join(sourceDirectory, 'pwa', 'manifest.webmanifest'), join(outputDirectory, 'manifest.webmanifest'))
  await writeIcons()
  console.log(`built ${outputDirectory} (shell ${version})`)
}

if (import.meta.main) {
  await buildMobile()
}
