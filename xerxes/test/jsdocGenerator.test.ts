// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  generateJSDoc,
  generateJSDocForFiles,
  type SourceDocumentationFilePort,
} from '../src/maintenance/generateJSDoc.js'

test('JSDoc generation documents modules, declarations, parameters, inheritance, and inferred returns', () => {
  const source = [
    '// Copyright header stays first.',
    'export async function getUser(userId: string, limit = 10): Promise<User> { return {} as User }',
    '',
    'class Store extends BaseStore {',
    '  retries = 3',
    '  get currentUser(): User { return {} as User }',
    '  async saveUser(value: User): Promise<void> {}',
    '}',
    '',
    'export const buildMap = (input: string) => new Map([[input, input]])',
    '',
  ].join('\n')

  const result = generateJSDoc(source, 'agent-runtime.ts')

  expect(result).toMatchObject({ changed: true, declarationsDocumented: 6, moduleDocumented: true, valid: true })
  expect(result.text).toStartWith('// Copyright header stays first.\n/**')
  expect(result.text).toContain(' * Agent runtime module for Xerxes.')
  expect(result.text).toContain(' * @module agent-runtime')
  expect(result.text).toContain(' * @remarks Exports: getUser, buildMap.')
  expect(result.text).toContain(' * Asynchronously retrieve the user.')
  expect(result.text).toContain(' * @param userId - Input: user id.')
  expect(result.text).toContain(' * @param limit - Optional input: limit. Defaults to 10.')
  expect(result.text).toContain(' * @returns Resolves with the result of the operation.')
  expect(result.text).toContain(' * Store.')
  expect(result.text).toContain(' * @extends BaseStore')
  expect(result.text).toContain('  /**\n   * The retries.\n   */\n  retries = 3')
  expect(result.text).toContain(' * Return the current user.')
  expect(result.text).toContain(' * Asynchronously save user.')
  expect(result.text).not.toContain('Asynchronously save user.\n   *\n   * @returns')
  expect(result.text).toContain(' * Build map.')
  expect(result.text).toContain(' * @returns Result of the operation.')
})

test('JSDoc generation preserves existing comments, is idempotent, and rejects invalid TypeScript', () => {
  const source = [
    '// License comment',
    '/**',
    ' * Existing function documentation.',
    ' */',
    'export function keepExisting(): void {}',
    '',
    'function parsePayload(value: string) { return value }',
    '',
  ].join('\n')

  const first = generateJSDoc(source, 'parser.ts')
  expect(first).toMatchObject({ changed: true, declarationsDocumented: 1, moduleDocumented: true, valid: true })
  expect(first.text.indexOf('@module parser')).toBeLessThan(first.text.indexOf('Existing function documentation.'))
  expect(first.text.match(/Existing function documentation\./gu)).toHaveLength(1)
  expect(first.text).toContain(' * Parse payload.')

  const second = generateJSDoc(first.text, 'parser.ts')
  expect(second).toEqual({
    changed: false,
    declarationsDocumented: 0,
    moduleDocumented: false,
    text: first.text,
    valid: true,
  })

  const invalid = generateJSDoc('export function broken( {', 'broken.ts')
  expect(invalid).toEqual({
    changed: false,
    declarationsDocumented: 0,
    moduleDocumented: false,
    text: 'export function broken( {',
    valid: false,
  })
})

test('JSDoc file generation honors dry runs and exposes an injectable file port', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-jsdoc-generator-'))
  try {
    const path = join(root, 'sample.ts')
    const source = 'export const buildValue = (input: string) => input\n'
    await writeFile(path, source, 'utf8')

    const preview = await generateJSDocForFiles([path], { dryRun: true })
    expect(preview[0]).toMatchObject({ changed: true, valid: true })
    expect(await readFile(path, 'utf8')).toBe(source)

    const written = await generateJSDocForFiles([path])
    expect(written[0]).toMatchObject({ changed: true, valid: true })
    expect(await readFile(path, 'utf8')).toContain('@module sample')

    let virtualText = 'function createTask(id: string) { return id }\n'
    const writes: string[] = []
    const port: SourceDocumentationFilePort = {
      readText: async () => virtualText,
      writeText: async (_path, text) => {
        writes.push(text)
        virtualText = text
      },
    }
    const virtualPreview = await generateJSDocForFiles(['/virtual/task.ts'], { dryRun: true }, port)
    expect(virtualPreview[0]).toMatchObject({ changed: true, valid: true })
    expect(writes).toEqual([])

    await generateJSDocForFiles(['/virtual/task.ts'], { atomic: false }, port)
    expect(writes).toHaveLength(1)
    expect(virtualText).toContain(' * Create task.')
    await expect(generateJSDocForFiles(['/virtual/task.py'], {}, port)).rejects.toThrow(
      'Expected a TypeScript source file',
    )
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})
