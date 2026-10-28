// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, readdir, stat } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'

it('builds the worklet, independent VAD/speech workers and same-origin ONNX assets without native dependencies', async () => {
  const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
  const proc = Bun.spawn([process.execPath, join(root, 'scripts/buildMobile.ts')], { cwd: root, stdout: 'pipe', stderr: 'pipe' })
  const [code, stdout, stderr] = await Promise.all([proc.exited, new Response(proc.stdout).text(), new Response(proc.stderr).text()])
  expect(code, `${stdout}\n${stderr}`).toBe(0)
  const output = join(root, 'dist/mobile')
  const required = ['voice-worklet.js', 'voice-call-worker.js', 'voice-vad-worker.js',
    'voice-ort/ort-wasm-simd-threaded.wasm', 'voice-ort/ort-wasm-simd-threaded.mjs',
    'voice-ort/ort-wasm-simd-threaded.jsep.wasm', 'voice-ort/ort-wasm-simd-threaded.jsep.mjs']
  for (const name of required) expect((await stat(join(output, name))).size, name).toBeGreaterThan(0)
  for (const directory of ['', 'assets', 'voice-ort']) {
    for (const name of await readdir(join(output, directory))) {
      if (!/\.(?:m?js)$/u.test(name)) continue
      const text = await readFile(join(output, directory, name), 'utf8')
      expect(text, `${directory}/${name}`).not.toMatch(/sharp|onnxruntime-node/u)
    }
  }
}, 30_000)
