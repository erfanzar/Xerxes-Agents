// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { fileAgeOf, fileTypeOf, sortTreeRows } from '../src/desktop/renderer/WorkspaceFileTree.js'

test('file rows show a compact age, and nothing when the runtime sent no time', () => {
  const now = Date.parse('2026-11-14T12:00:00Z')
  expect(fileAgeOf('2026-11-13T19:00:00Z', now)).toBe('17h')
  expect(fileAgeOf('2026-11-10T12:00:00Z', now)).toBe('4d')
  expect(fileAgeOf('', now)).toBe('')
})
import { documentImages, htmlToMarkdown } from '../src/desktop/renderer/DesktopPanels.js'

test('README images: badges become links, remote images links, relative images resolve from the document folder', () => {
  const md = '[![npm](https://img.shields.io/npm/v/x)](https://npmjs.com/x)\n![logo](assets/logo.png)\n![shot](https://example.com/a.png)\n```\n![keep](a.png)\n```'
  const out = documentImages(md, './docs/README.md')
  expect(out).toContain('[npm](https://npmjs.com/x)')
  expect(out).toContain('![logo](docs/assets/logo.png)')
  expect(out).toContain('[shot](https://example.com/a.png)')
  expect(out).toContain('![keep](a.png)')
  expect(documentImages(htmlToMarkdown('<p align="center"><img src="assets/logo.png" alt="Xerxes logo"></p>'), 'README.md')).toContain('![Xerxes logo](assets/logo.png)')
})

test('the file tree lists folders first, then files, each by name ignoring case and a leading dot', () => {
  const rows = [
    { value: './README.md', meta: 'file' },
    { value: './xerxes/', meta: 'dir' },
    { value: './.gitignore', meta: 'file' },
    { value: './assets/', meta: 'dir' },
    { value: './.github/', meta: 'dir' },
    { value: './bun.lock', meta: 'file' },
    { value: './Dockerfile', meta: 'file' },
  ]
  expect(sortTreeRows(rows).map(row => row.value)).toEqual([
    './assets/', './.github/', './xerxes/',
    './bun.lock', './Dockerfile', './.gitignore', './README.md',
  ])
})

test('file icons are typed by name so each kind gets its own icon colour', () => {
  expect(fileTypeOf('App.tsx').type).toBe('ts')
  expect(fileTypeOf('index.js').type).toBe('js')
  expect(fileTypeOf('package.json').type).toBe('json')
  expect(fileTypeOf('docker-compose.yml').type).toBe('config')
  expect(fileTypeOf('Dockerfile').type).toBe('docker')
  expect(fileTypeOf('AGENTS.md').type).toBe('doc')
  expect(fileTypeOf('shading.css').type).toBe('css')
  expect(fileTypeOf('bun.lock').type).toBe('lock')
  expect(fileTypeOf('LICENSE').type).toBe('plain')
})
