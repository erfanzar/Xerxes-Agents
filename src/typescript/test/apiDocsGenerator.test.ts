// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  API_DOCS_MANIFEST_FILE,
  ApiDocsGenerationError,
  discoverTypeScriptModules,
  extractTypeScriptExports,
  formatTypeScriptSources,
  generateTypeScriptApiDocs,
} from '../src/maintenance/apiDocsGenerator.js'

test('TypeScript API docs discover modules, extract exports, and write deterministic nested indexes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-typescript-api-docs-'))
  const sourceDirectory = join(root, 'source')
  const outputDirectory = join(root, 'generated-docs')
  try {
    await writeSource(sourceDirectory, 'index.ts', [
      '/** Runtime entrypoint. */',
      "export { child } from './feature/child.js'",
      "export type { Child } from './feature/child.js'",
      '',
    ].join('\n'))
    await writeSource(sourceDirectory, 'feature/child.ts', [
      '/** Contract exposed by the feature module. */',
      'export interface Child {',
      '  readonly name: string',
      '}',
      '',
      '/** Construct the default child. */',
      "export const child: Child = { name: 'xerxes' }",
      '',
      'const privateValue = 1',
      '',
    ].join('\n'))
    await writeSource(sourceDirectory, 'feature/nested/view.tsx', [
      '/** Render a feature view. */',
      'export function View(): string { return "view" }',
      '',
    ].join('\n'))
    await writeSource(sourceDirectory, 'feature/ignored.d.ts', 'export interface Ignored { readonly value: string }\n')
    await writeSource(sourceDirectory, 'node_modules/foreign.ts', 'export const foreign = true\n')

    const discovered = await discoverTypeScriptModules(sourceDirectory, '@test/runtime')
    expect(discovered.map(module => module.sourceRelativePath)).toEqual([
      'feature/child.ts',
      'feature/nested/view.tsx',
      'index.ts',
    ])
    expect(discovered.map(module => module.documentationPath)).toEqual([
      'feature/child.md',
      'feature/nested/view.md',
      'index.api.md',
    ])
    expect(discovered.at(-1)).toMatchObject({ moduleName: '@test/runtime' })

    const result = await generateTypeScriptApiDocs({
      outputDirectory,
      packageName: '@test/runtime',
      sourceDirectory,
    })
    expect(result.changed).toBeTrue()
    expect(result.modules).toHaveLength(3)
    expect(result.changes.filter(change => change.action === 'created')).toHaveLength(7)

    const rootIndex = await readFile(join(outputDirectory, 'index.md'), 'utf8')
    expect(rootIndex).toContain('# @test/runtime API Reference')
    expect(rootIndex).toContain('[`feature`](feature/index.md)')
    expect(rootIndex).toContain('[`@test/runtime`](index.api.md)')

    const featureIndex = await readFile(join(outputDirectory, 'feature', 'index.md'), 'utf8')
    expect(featureIndex.indexOf('## Packages')).toBeLessThan(featureIndex.indexOf('## Modules'))
    expect(featureIndex).toContain('[`nested`](nested/index.md)')
    expect(featureIndex).toContain('[`@test/runtime/feature/child`](child.md)')

    const childPage = await readFile(join(outputDirectory, 'feature', 'child.md'), 'utf8')
    expect(childPage).toContain('Contract exposed by the feature module.')
    expect(childPage).toContain('### `Child` (interface)')
    expect(childPage).toContain('### `child` (const)')
    expect(childPage).not.toContain('privateValue')

    const manifest = JSON.parse(await readFile(join(outputDirectory, API_DOCS_MANIFEST_FILE), 'utf8')) as { files: string[] }
    expect(manifest.files).toEqual([
      'feature/child.md',
      'feature/index.md',
      'feature/nested/index.md',
      'feature/nested/view.md',
      'index.api.md',
      'index.md',
    ])
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('dry runs identify only manifest-owned stale pages and leave handwritten output untouched', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-typescript-api-docs-clean-'))
  const sourceDirectory = join(root, 'source')
  const outputDirectory = join(root, 'generated-docs')
  try {
    await writeSource(sourceDirectory, 'index.ts', 'export const root = true\n')
    await writeSource(sourceDirectory, 'nested/removed.ts', 'export const removed = true\n')
    await generateTypeScriptApiDocs({ outputDirectory, sourceDirectory })

    const stalePage = join(outputDirectory, 'nested', 'removed.md')
    const handwritten = join(outputDirectory, 'handwritten.md')
    await writeFile(handwritten, '# Keep me\n', 'utf8')
    await rm(join(sourceDirectory, 'nested', 'removed.ts'))

    const preview = await generateTypeScriptApiDocs({ dryRun: true, outputDirectory, sourceDirectory })
    expect(preview.changed).toBeTrue()
    expect(preview.changes).toContainEqual({ action: 'deleted', path: stalePage })
    expect(await readFile(stalePage, 'utf8')).toContain('removed')
    expect(await readFile(handwritten, 'utf8')).toBe('# Keep me\n')

    const written = await generateTypeScriptApiDocs({ outputDirectory, sourceDirectory })
    expect(written.changes).toContainEqual({ action: 'deleted', path: stalePage })
    await expect(access(stalePage)).rejects.toThrow()
    expect(await readFile(handwritten, 'utf8')).toBe('# Keep me\n')
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('the generator refuses an unowned destination and keeps formatting behind an injected port', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-typescript-api-docs-boundary-'))
  const sourceDirectory = join(root, 'source')
  const unsafeOutputDirectory = join(root, 'unowned-docs')
  try {
    await writeSource(sourceDirectory, 'alpha.ts', 'export const alpha = true\n')
    await mkdir(unsafeOutputDirectory, { recursive: true })
    await writeFile(join(unsafeOutputDirectory, 'notes.md'), '# Handwritten docs\n', 'utf8')

    await expect(generateTypeScriptApiDocs({ outputDirectory: unsafeOutputDirectory, sourceDirectory }))
      .rejects.toBeInstanceOf(ApiDocsGenerationError)

    const calls: { fix: boolean; paths: readonly string[] }[] = []
    const formatted = await formatTypeScriptSources(sourceDirectory, {
      async format(paths, options) {
        calls.push({ fix: options.fix, paths })
      },
    }, false)
    expect(formatted.paths).toEqual([join(sourceDirectory, 'alpha.ts')])
    expect(calls).toEqual([{ fix: false, paths: [join(sourceDirectory, 'alpha.ts')] }])
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('TypeScript export extraction covers declarations, re-exports, defaults, and malformed source errors', () => {
  const exports = extractTypeScriptExports([
    '/** A constant. */',
    'export const value = 1',
    'export type Alias = string',
    'export { value as renamed }',
    'export * from "./other.js"',
    'export default class DefaultExport {}',
    '',
  ].join('\n'), 'sample.ts')
  expect(exports.map(symbol => [symbol.name, symbol.kind])).toEqual([
    ['*', 're-export'],
    ['Alias', 'type'],
    ['default', 'class'],
    ['renamed', 're-export'],
    ['value', 'const'],
  ])
  expect(exports.find(symbol => symbol.name === 'value')).toMatchObject({ documentation: 'A constant.' })
  expect(() => extractTypeScriptExports('export const =', 'broken.ts')).toThrow(ApiDocsGenerationError)
})

async function writeSource(root: string, relativePath: string, content: string): Promise<void> {
  const path = join(root, relativePath)
  await mkdir(join(path, '..'), { recursive: true })
  await writeFile(path, content, 'utf8')
}
