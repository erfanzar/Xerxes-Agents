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
import { realpathSync } from 'node:fs'
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
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

  // Build AudioWorklet separately: it has its own global scope, and liveCapture loads this exact URL.
  for (const [entry, name] of [
    ['voice/worklet/pcmCapture.worklet.ts', 'voice-worklet.js'],
    ['voice/call/worker.ts', 'voice-call-worker.js'],
    ['voice/call/vadWorker.ts', 'voice-vad-worker.js'],
  ] as const) {
    const result = await Bun.build({
      entrypoints: [join(sourceDirectory, entry)], outdir: outputDirectory,
      target: 'browser', format: 'esm', minify: true, naming: { entry: name },
      define: { 'process.env.NODE_ENV': JSON.stringify('production') },
      plugins: [{
        name: 'browser-transformers-stub-metadata',
        setup(build) {
          build.onLoad({ filter: /transformers\.web\.js$/u }, async ({ path }) => ({
            loader: 'js',
            // The browser distribution contains empty Webpack native-module stubs. Remove only their generated
            // banners/import annotations, not licenses or code; the post-build check rejects native code references.
            contents: (await readFile(path, 'utf8')).replace(/\/\*![\s\S]*?\*\//gu, comment => /(?:onnxruntime-node|sharp) \(ignored\)|^\/\*!\s*(?:onnxruntime-node|sharp)\s*\*\/$/u.test(comment) ? '' : comment),
          }))
        },
      }],
    })
    if (!result.success) {
      for (const log of result.logs) console.error(log)
      throw new Error(`${name} build failed`)
    }
  }
  // Resolve the ONNX version actually owned by transformers, not an unrelated workspace runtime.
  const transformerEntry = import.meta.resolve('@huggingface/transformers')
  // Compare real files, not URLs: on Windows Bun links packages with junctions, so resolving from inside
  // kokoro-js returns the same file through kokoro's own node_modules link (and drive-letter case can differ).
  const realFile = (url: string): string => {
    const path = realpathSync(fileURLToPath(url))
    return process.platform === 'win32' ? path.toLowerCase() : path
  }
  if (realFile(import.meta.resolve('@huggingface/transformers', import.meta.resolve('kokoro-js'))) !== realFile(transformerEntry)) {
    throw new Error('Kokoro and call models must resolve one shared transformers.js runtime')
  }
  const ortDirectory = dirname(fileURLToPath(import.meta.resolve('onnxruntime-web', transformerEntry)))
  const ortOutput = join(outputDirectory, 'voice-ort')
  await mkdir(ortOutput, { recursive: true })
  for (const name of await readdir(ortDirectory)) {
    if (/^ort-wasm-simd-threaded(?:\.jsep)?\.(?:wasm|mjs)$/u.test(name)) await cp(join(ortDirectory, name), join(ortOutput, name))
  }
  const ortFiles = await readdir(ortOutput)
  if (ortFiles.length !== 4 || ortFiles.filter(name => name.endsWith('.wasm')).length !== 2) throw new Error('Call build did not copy both WASM / WebGPU ONNX backends')
  for (const name of ['voice-worklet.js', 'voice-call-worker.js', 'voice-vad-worker.js', ...ortFiles.map(name => `voice-ort/${name}`)]) {
    const file = Bun.file(join(outputDirectory, name))
    if (!(await file.exists()) || file.size === 0) throw new Error(`Missing call asset: ${name}`)
    console.log(`call asset ${name}: ${file.size} bytes`)
  }
  // Browser dependencies must never leak native adapters into the shipped phone bundle.
  for (const name of ['voice-worklet.js', 'voice-call-worker.js', 'voice-vad-worker.js', ...ortFiles.filter(name => name.endsWith('.mjs')).map(name => `voice-ort/${name}`), ...app.outputs.filter(output => output.path.endsWith('.js')).map(output => `assets/${basename(output.path)}`)]) {
    const text = await readFile(join(outputDirectory, name), 'utf8')
    const native = /onnxruntime-node|sharp/u.exec(text)
    if (native) throw new Error(`Native inference dependency leaked into ${name}: ${text.slice(Math.max(0, native.index - 100), native.index + 140)}`)
  }

  // Shell = index.html plus content-hashed assets. Models and inference assets are loaded only when a call starts.
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
