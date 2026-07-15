// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { isAbsolute, resolve } from 'node:path'

const root = resolve(import.meta.dir, '..')
const outputOverride = process.env.XERXES_TUI_BUILD_OUT?.trim()
const output = outputOverride
  ? isAbsolute(outputOverride)
    ? outputOverride
    : resolve(root, outputOverride)
  : resolve(root, 'dist/ui/entry.js')
const bundle = Bun.file(output)

if (!(await bundle.exists())) {
  throw new Error(`missing ${output}; run bun run build first`)
}

const body = await bundle.text()
if (body.length === 0) {
  throw new Error('dist/ui/entry.js is empty')
}
if (body.startsWith('#!')) {
  throw new Error('dist/ui/entry.js must not retain the source shebang')
}
if (!body.includes('createRequire as __cr')) {
  throw new Error('dist/ui/entry.js is missing the CommonJS require bridge')
}
if (!body.includes('@opentui/core')) {
  throw new Error('dist/ui/entry.js is not the OpenTUI build')
}
if (!body.includes('Ready for your next command.')) {
  throw new Error('dist/ui/entry.js is not the current Xerxes UI build')
}
if (!body.includes('Choose a model with /provider')) {
  throw new Error('dist/ui/entry.js is missing the startup welcome screen')
}

console.log(`verified ${bundle.name} (${body.length} bytes)`)
