// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  analyzeCodeStructure,
  applyDiff,
  copyFile,
  createDiff,
  deleteFile,
  detectLanguage,
  findAndReplace,
  gitAdd,
  gitApplyPatch,
  gitDiff,
  gitLog,
  gitStatus,
  listDirectory,
  moveFile,
  readFile,
  registerCodingTools,
  writeFile,
} from '../src/tools/codingTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-coding-tools-'))
  try {
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

async function git(cwd: string, arguments_: readonly string[]): Promise<string> {
  const child = Bun.spawn(['git', ...arguments_], { cwd, stdin: 'ignore', stderr: 'pipe', stdout: 'pipe' })
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ])
  if (code !== 0) {
    throw new Error(stderr)
  }
  return stdout
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('coding file operations stay in the workspace and preserve useful edit output', async () => {
  await inWorkspace(async (_workspace, paths) => {
    const write = await writeFile({ content: 'one\ntwo\nthree\n', file_path: 'src/example.txt' }, paths)
    expect(write).toContain('Successfully wrote')
    expect(write).toContain('@@')

    const chunk = await readFile({ end_line: 2, file_path: 'src/example.txt', start_line: 2 }, paths)
    expect(chunk).toContain('     2 | two')
    expect(chunk).toContain('Continue with start_line=3')
    const listing = await listDirectory({ directory: '.', pattern: '**/*.txt', recursive: true }, paths)
    expect(listing).toContain('src/example.txt')

    expect(await copyFile(
      { destination: 'copy.txt', source: 'src/example.txt' },
      paths,
    )).toContain('Successfully copied')
    expect(await moveFile({ destination: 'moved.txt', source: 'copy.txt' }, paths)).toContain('Successfully moved')
    expect(await findAndReplace({
      backup: true,
      file_path: 'moved.txt',
      replace: 'TWO',
      search: 'two',
    }, paths)).toContain('Replaced 1')
    expect(await Bun.file(await paths.resolve('moved.txt.bak')).text()).toContain('two')
    expect(await deleteFile({ path: 'moved.txt' }, paths)).toContain('Successfully deleted')

    await expect(readFile({ file_path: '../outside.txt' }, paths)).rejects.toThrow()
  })
})

test('unified diff creation round-trips additions, removals, and no-final-newline files', () => {
  const original = 'alpha\nbeta\ngamma'
  const modified = 'alpha\nBETA\ngamma\ndelta\n'
  const diff = createDiff({ file_name: 'sample.txt', modified, original })

  expect(diff).toContain('--- a/sample.txt')
  expect(diff).toContain('+++ b/sample.txt')
  expect(diff).toContain('\\ No newline at end of file')
  expect(applyDiff({ diff, original })).toBe(modified)

  const manyLines = Array.from({ length: 20 }, (_value, index) => 'line-' + index).join('\n') + '\n'
  const separatedChanges = manyLines.replace('line-1', 'changed-1').replace('line-18', 'changed-18')
  const separatedDiff = createDiff({ modified: separatedChanges, original: manyLines })
  expect(separatedDiff.match(/^@@/gm)).toHaveLength(2)
  expect(applyDiff({ diff: separatedDiff, original: manyLines })).toBe(separatedChanges)
})

test('find/replace and code analysis expose the legacy coding surface', async () => {
  await inWorkspace(async (_workspace, paths) => {
    await Bun.write(await paths.resolve('module.ts'), [
      'import { value } from "./value.js"',
      '',
      '// public model',
      'export class Worker {',
      '  run(): void {}',
      '}',
      'export const convert = (value: string) => value.toUpperCase()',
      '',
    ].join('\n'))
    const analysis = await analyzeCodeStructure({ file_path: 'module.ts' }, paths)
    expect(analysis).toContain('Language: TypeScript')
    expect(analysis).toContain('Worker')
    expect(analysis).toContain('convert')
    expect(detectLanguage('test.py')).toBe('Python')
    expect(detectLanguage('.xyz')).toBe('Unknown')
  })
})

test('git status, add, diff, and log run with direct argv inside a contained repository', async () => {
  await inWorkspace(async (workspace, paths) => {
    await git(workspace, ['init'])
    await git(workspace, ['config', 'user.email', 'test@example.invalid'])
    await git(workspace, ['config', 'user.name', 'Xerxes Test'])
    await Bun.write(join(workspace, 'tracked.txt'), 'before\n')
    expect(await gitAdd({ files: ['tracked.txt'] }, paths)).toContain('Successfully staged')
    await git(workspace, ['commit', '-m', 'initial'])

    const patch = createDiff({
      from_file: 'tracked.txt',
      modified: 'patched\n',
      original: 'before\n',
      to_file: 'tracked.txt',
    })
    expect(await gitApplyPatch({ check_only: true, patch_content: patch }, paths)).toBe('Patch can be applied cleanly')
    expect(await gitApplyPatch({ patch_content: patch }, paths)).toBe('Patch applied successfully')
    expect(await Bun.file(join(workspace, 'tracked.txt')).text()).toBe('patched\n')

    await Bun.write(join(workspace, 'tracked.txt'), 'after\n')
    expect(await gitStatus({}, paths)).toContain('Modified (unstaged)')
    expect(await gitDiff({ context_lines: 0 }, paths)).toContain('-before')
    expect(await gitLog({ max_commits: 1 }, paths)).toContain('initial')
  })
})

test('lower-case coding schemas register independently from the newer camel-case file tools', async () => {
  await inWorkspace(async (_workspace, paths) => {
    const registry = new ToolRegistry()
    registerCodingTools(registry, paths)
    expect(registry.definitions().map(definition => definition.function.name)).toContain('analyze_code_structure')
    expect(await registry.execute(
      call('write_file', { content: 'registered\n', file_path: 'registered.txt' }),
      { metadata: {} },
    )).toContain('Successfully wrote')
  })
})

test('write_file defaults to no overwrite and skips the diff preview for oversized inputs but still writes', async () => {
  await inWorkspace(async (workspace, paths) => {
    const original = Array.from({ length: 1_100 }, (_value, index) => 'old-line-' + index).join('\n') + '\n'
    const modified = Array.from({ length: 1_100 }, (_value, index) => 'new-line-' + index).join('\n') + '\n'
    await Bun.write(join(workspace, 'large.txt'), original)

    await expect(writeFile({ content: modified, file_path: 'large.txt' }, paths)).rejects.toThrow('overwrite=true')
    const written = await writeFile({ content: modified, file_path: 'large.txt', overwrite: true }, paths)
    expect(written).toContain('Successfully wrote')
    expect(written).toContain('diff skipped')
    expect(written).not.toContain('@@')
    expect(await Bun.file(join(workspace, 'large.txt')).text()).toBe(modified)

    expect(() => createDiff({ modified, original })).toThrow('diff limit')
  })
})

test('read_file rejects files beyond the byte cap with an actionable error', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'huge.txt'), 'x'.repeat(10_000_001))
    await expect(readFile({ file_path: 'huge.txt' }, paths)).rejects.toThrow('read_file limit')
  })
})

test('find_and_replace validates the regex before writing a backup and keeps $ sequences literal', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'target.txt'), 'alpha beta\n')
    await expect(findAndReplace({
      file_path: 'target.txt',
      regex: true,
      replace: 'x',
      search: '([',
    }, paths)).rejects.toThrow('valid JavaScript regular expression')
    expect(await Bun.file(join(workspace, 'target.txt.bak')).exists()).toBeFalse()
    expect(await Bun.file(join(workspace, 'target.txt')).text()).toBe('alpha beta\n')

    const replaced = await findAndReplace({
      backup: false,
      file_path: 'target.txt',
      regex: true,
      replace: '[$&][$1]',
      search: 'beta',
    }, paths)
    expect(replaced).toContain('Replaced 1')
    expect(await Bun.file(join(workspace, 'target.txt')).text()).toBe('alpha [$&][$1]\n')
  })
})

test('find_and_replace refuses regex mode beyond the subject-size cap but allows literal mode', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'subject.txt'), 'x'.repeat(1_000_001))
    await expect(findAndReplace({
      backup: false,
      file_path: 'subject.txt',
      regex: true,
      replace: 'y',
      search: 'x+',
    }, paths)).rejects.toThrow('regex subject limit')
    expect(await findAndReplace({
      backup: false,
      file_path: 'subject.txt',
      replace: 'yyy',
      search: 'xxx',
    }, paths)).toContain('Replaced')
  })
})
