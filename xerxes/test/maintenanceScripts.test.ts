// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CANONICAL_COPYRIGHT_LINE,
  LICENSE_TRAILER,
  fixLicenseHeaders,
  fixLicenseText,
} from '../src/maintenance/fixLicenseHeaders.js'
import {
  cleanupDuplicateDocstringFiles,
  cleanupDuplicateDocstrings,
} from '../src/maintenance/cleanupDuplicateDocstrings.js'
import {
  removeDocstringsAndComments,
  removeDocstringsFromFiles,
} from '../src/maintenance/removeDocstrings.js'

test('license-header maintenance normalizes targeted files, skips excluded directories, and honors dry runs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-license-maintenance-'))
  try {
    const source = [
      '# Copyright 2026 Xerxes-Agents Author.',
      '#',
      '# Licensed under the Apache License, Version 2.0 (the "License");',
      '# you may not use this file except in compliance with the License.',
      '# You may obtain a copy of the License at',
      '#',
      '#     https://www.apache.org/licenses/LICENSE-2.0',
      'print("hello")',
      '',
    ].join('\n')
    const packageDirectory = join(root, 'nested')
    const ignoredDirectory = join(root, 'node_modules', 'dependency')
    await mkdir(packageDirectory, { recursive: true })
    await mkdir(ignoredDirectory, { recursive: true })
    const pythonPath = join(packageDirectory, 'module.py')
    const dockerfilePath = join(root, 'Dockerfile')
    const ignoredPath = join(ignoredDirectory, 'ignored.py')
    await writeFile(pythonPath, source, 'utf8')
    await writeFile(dockerfilePath, source.replace('.py', '.docker'), 'utf8')
    await writeFile(ignoredPath, source, 'utf8')
    await writeFile(join(root, 'notes.txt'), source, 'utf8')

    const preview = await fixLicenseHeaders(root, { dryRun: true })
    expect(preview.map(change => change.path).sort()).toEqual([dockerfilePath, pythonPath].sort())
    expect(preview.every(change => change.changed === false)).toBeTrue()
    expect(await readFile(pythonPath, 'utf8')).toBe(source)

    const written = await fixLicenseHeaders(root)
    expect(written).toHaveLength(2)
    const normalized = await readFile(pythonPath, 'utf8')
    expect(normalized).toContain(CANONICAL_COPYRIGHT_LINE)
    expect(normalized).toContain(LICENSE_TRAILER)
    expect(await readFile(ignoredPath, 'utf8')).toBe(source)
    expect(fixLicenseText(normalized).fixes).toEqual([])
    expect((await readdir(packageDirectory)).some(name => name.endsWith('.tmp'))).toBeFalse()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('duplicate-docstring cleanup keeps official module and class docs while removing only extra literals', async () => {
  const source = [
    '"""module documentation"""',
    '',
    '"module attribute"',
    '',
    'if enabled:',
    '    "conditional literal stays"',
    '',
    'class Example:',
    '    """class documentation"""',
    '    "class attribute"',
    '',
    '    def method(self):',
    '        """function documentation"""',
    '        "function literal stays"',
    '        return "value"',
    '',
    'class Nested:',
    '    """nested documentation"""',
    '',
  ].join('\n')

  const result = cleanupDuplicateDocstrings(source)
  expect(result).toMatchObject({ changed: true, removed: 2, valid: true })
  expect(result.text).toContain('"""module documentation"""')
  expect(result.text).toContain('"""class documentation"""')
  expect(result.text).toContain('"""function documentation"""')
  expect(result.text).toContain('"function literal stays"')
  expect(result.text).toContain('"conditional literal stays"')
  expect(result.text).not.toContain('"module attribute"')
  expect(result.text).not.toContain('"class attribute"')

  const invalid = cleanupDuplicateDocstrings('"unterminated\n')
  expect(invalid).toEqual({ changed: false, removed: 0, text: '"unterminated\n', valid: false })
})

test('duplicate-docstring cleanup previews then atomically writes explicitly named files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-docstring-cleanup-'))
  try {
    const path = join(root, 'sample.py')
    const source = '"""docs"""\n"attribute"\n'
    await writeFile(path, source, 'utf8')

    const preview = await cleanupDuplicateDocstringFiles([path], { dryRun: true })
    expect(preview[0]).toMatchObject({ changed: true, removed: 1, valid: true })
    expect(await readFile(path, 'utf8')).toBe(source)

    const written = await cleanupDuplicateDocstringFiles([path])
    expect(written[0]).toMatchObject({ changed: true, removed: 1, valid: true })
    expect(await readFile(path, 'utf8')).toBe('"""docs"""\n')
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('docstring and comment removal preserves ordinary string values and syntactically nonempty suites', async () => {
  const source = [
    '#!/usr/bin/env python3',
    '# module comment',
    '"""module documentation"""',
    '"module attribute"',
    '',
    'class OnlyDocs:',
    '    """class documentation"""',
    '    "class attribute"',
    '',
    'class Example:',
    '    """class documentation"""',
    '    "class attribute"  # attribute comment',
    '',
    '    def method(self):',
    '        """function documentation"""',
    '        # function comment',
    '        "function literal stays"',
    '        value = "# not a comment"  # inline comment',
    '        return value',
    '',
    'if enabled:',
    '    "conditional literal stays"  # conditional comment',
    '',
  ].join('\n')

  const result = removeDocstringsAndComments(source)
  expect(result).toMatchObject({ changed: true, docstringsRemoved: 7, emptySuitesPreserved: 1, valid: true })
  expect(result.text).toContain('class OnlyDocs:\n    pass\n')
  expect(result.text).toContain('"function literal stays"')
  expect(result.text).toContain('"conditional literal stays"')
  expect(result.text).toContain('value = "# not a comment"')
  expect(result.text).not.toContain('module documentation')
  expect(result.text).not.toContain('class attribute')
  expect(result.text).not.toContain('function documentation')
  expect(result.text).not.toContain('inline comment')
  expect(result.text).not.toContain('conditional comment')

  const invalid = removeDocstringsAndComments("'''unterminated\n")
  expect(invalid.valid).toBeFalse()
  expect(invalid.text).toBe("'''unterminated\n")
})

test('docstring removal supports dry runs before an atomic write', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-docstring-removal-'))
  try {
    const path = join(root, 'only-docstring.py')
    const source = 'def documented():\n    """only documentation"""\n'
    await writeFile(path, source, 'utf8')

    const preview = await removeDocstringsFromFiles([path], { dryRun: true })
    expect(preview[0]).toMatchObject({ changed: true, emptySuitesPreserved: 1, valid: true })
    expect(await readFile(path, 'utf8')).toBe(source)

    await removeDocstringsFromFiles([path])
    expect(await readFile(path, 'utf8')).toBe('def documented():\n    pass\n')
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})
