// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync, utimesSync } from 'node:fs'
import { lstat, mkdir, mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

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
  runGit,
  writeFile,
} from '../src/tools/codingTools.js'
import { fileStateTracker } from '../src/tools/fileState.js'
import { DEFAULT_MAX_READ_FILE_BYTES, MAX_READ_WINDOW_CHARS } from '../src/tools/fileTools.js'
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

afterEach(() => {
  fileStateTracker.clear()
})

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
    // A Windows host default of autocrlf=true rewrites the applied file to
    // CRLF, which has nothing to do with what this test exercises.
    await git(workspace, ['config', 'core.autocrlf', 'false'])
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
    await Bun.write(join(workspace, 'huge.txt'), 'x'.repeat(DEFAULT_MAX_READ_FILE_BYTES + 1))
    await expect(readFile({ file_path: 'huge.txt' }, paths)).rejects.toThrow(
      String(DEFAULT_MAX_READ_FILE_BYTES) + '-byte read_file limit',
    )
  })
})

test('read_file refuses a window over the character ceiling instead of dumping a generated file', async () => {
  await inWorkspace(async (workspace, paths) => {
    // One line, so the line window cannot bound the return and no truncation notice fires.
    await Bun.write(join(workspace, 'bundle.min.js'), 'a'.repeat(MAX_READ_WINDOW_CHARS + 1))
    const minified = readFile({ file_path: 'bundle.min.js' }, paths)
    await expect(minified).rejects.toThrow(String(MAX_READ_WINDOW_CHARS) + '-character read_file window ceiling')
    await expect(minified).rejects.toThrow('GrepTool')

    const wide = ('y'.repeat(49) + '\n').repeat(2_000)
    await Bun.write(join(workspace, 'wide.txt'), wide)
    // The suggested end_line is absolute, so it must be offset by start_line.
    await expect(readFile({ end_line: -1, file_path: 'wide.txt', start_line: 101 }, paths))
      .rejects.toThrow('retry with end_line=')
    expect(await readFile({ file_path: 'wide.txt' }, paths)).toContain('Continue with start_line=401')
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

test('the coding surface enforces the same read-before-write rule as the upper-case tools', async () => {
  await inWorkspace(async (workspace, paths) => {
    const path = join(workspace, 'module.ts')
    await Bun.write(path, 'export const value = 1\n')
    const context = { sessionId: 'session-coding' }

    await expect(findAndReplace(
      { backup: false, file_path: 'module.ts', replace: '2', search: '1' },
      paths,
      context,
    )).rejects.toThrow('requires reading "module.ts" first')
    expect(await Bun.file(path).text()).toBe('export const value = 1\n')

    await readFile({ file_path: 'module.ts' }, paths, context)
    expect(await findAndReplace(
      { backup: false, file_path: 'module.ts', replace: '2', search: '1' },
      paths,
      context,
    )).toContain('Replaced 1 occurrence(s)')

    // Another writer lands between the read and the whole-file rewrite.
    await Bun.write(path, 'export const value = 99\nexport const extra = true\n')
    const future = new Date(Date.now() + 5_000)
    utimesSync(path, future, future)
    await expect(writeFile(
      { content: 'export const value = 3\n', file_path: 'module.ts', overwrite: true },
      paths,
      context,
    )).rejects.toThrow('a whole-file write would discard those changes')
    expect(await Bun.file(path).text()).toContain('export const extra = true')
  })
})

/**
 * Write a stub `git` shell script and hand back its absolute path, so tests can
 * point runGit's `executable` parameter at it. (Bun resolves bare executable
 * names against the PATH captured at startup, so PATH mutation cannot shadow
 * `git` for Bun.spawn.)
 */
async function writeStubGit(script: string): Promise<string> {
  const stubDirectory = await mkdtemp(join(tmpdir(), 'xerxes-stub-git-'))
  const stubPath = join(stubDirectory, 'git')
  await Bun.write(stubPath, `#!/bin/sh\n${script}`)
  chmodSync(stubPath, 0o755)
  return stubPath
}

test('runGit collects output even when a hook keeps the pipe open after git exits', async () => {
  // The bug this pins: awaiting `new Response(child.stdout).text()` waits for
  // EOF, and EOF requires every holder of the write end to close. A background
  // holder spawned by a git hook kept the pipe open long after git itself was
  // done, so the call hung forever even though the command had succeeded.
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-run-git-holder-'))
  const stub = await writeStubGit('(sleep 30) &\necho hook-output\nexit 0\n')
  try {
    const started = Date.now()
    const output = await runGit(workspace, ['status'], undefined, undefined, stub)
    expect(Date.now() - started).toBeLessThan(10_000)
    expect(output).toContain('hook-output')
  } finally {
    await rm(workspace, { force: true, recursive: true })
    await rm(dirname(stub), { force: true, recursive: true })
  }
})

test('runGit rejects with the timeout error instead of hanging when the child ignores SIGTERM', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-run-git-timeout-'))
  // Ignore SIGTERM like a wedged hook, hold the pipe with a holder, and
  // busy-wait so only SIGKILL can end it.
  const stub = await writeStubGit("trap '' TERM\n(sleep 3) &\necho out\nwhile :; do :; done\n")
  try {
    const started = Date.now()
    await expect(runGit(workspace, ['status'], undefined, 800, stub))
      .rejects.toThrow('command timed out after 800ms')
    const elapsed = Date.now() - started
    // The initial SIGTERM fired at 800ms; the SIGKILL escalation lands about
    // one grace period later. Anything longer means the escalation failed.
    expect(elapsed).toBeGreaterThan(700)
    expect(elapsed).toBeLessThan(15_000)
  } finally {
    await rm(workspace, { force: true, recursive: true })
    await rm(dirname(stub), { force: true, recursive: true })
  }
})

test('move_file refuses an existing destination without overwrite and never leaves residue', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'src.txt'), 'source\n')
    await Bun.write(join(workspace, 'dst.txt'), 'destination\n')

    await expect(moveFile({ destination: 'dst.txt', source: 'src.txt' }, paths))
      .rejects.toThrow('already exists; pass overwrite=true to replace it')
    // The refusal must not consume the source or disturb the destination.
    expect(await Bun.file(join(workspace, 'src.txt')).text()).toBe('source\n')
    expect(await Bun.file(join(workspace, 'dst.txt')).text()).toBe('destination\n')

    // overwrite=true keeps the documented replace semantics.
    expect(await moveFile({ destination: 'dst.txt', overwrite: true, source: 'src.txt' }, paths))
      .toContain('Successfully moved')
    expect(await Bun.file(join(workspace, 'dst.txt')).text()).toBe('source\n')
    expect(await Bun.file(join(workspace, 'src.txt')).exists()).toBeFalse()
    const residue = await Array.fromAsync(new Bun.Glob('*.tmp').scan({ cwd: workspace }))
    expect(residue).toEqual([])
  })
})

test('move_file moves a directory to a fresh destination with overwrite=false', async () => {
  // The bug this pins: the TOCTOU reservation created an empty FILE at the
  // destination unconditionally, so rename(dir → existing file) failed ENOTDIR
  // and no flag combination could move a directory anymore.
  await inWorkspace(async (workspace, paths) => {
    await mkdir(join(workspace, 'project', 'nested'), { recursive: true })
    await Bun.write(join(workspace, 'project', 'nested', 'leaf.txt'), 'inside\n')

    expect(await moveFile({ destination: 'moved/project', source: 'project' }, paths))
      .toContain('Successfully moved')
    expect(await Bun.file(join(workspace, 'moved', 'project', 'nested', 'leaf.txt')).text()).toBe('inside\n')
    // The source tree is gone and the destination is the directory itself, not
    // a reservation artifact.
    expect(await Bun.file(join(workspace, 'project', 'nested', 'leaf.txt')).exists()).toBeFalse()
    const stat = await lstat(join(workspace, 'moved', 'project'))
    expect(stat.isDirectory()).toBeTrue()
    // No stray reservation files or directories next to it.
    expect(await readdir(join(workspace, 'moved'))).toEqual(['project'])
  })
})

test('move_file still refuses an existing destination for directories without overwrite', async () => {
  await inWorkspace(async (workspace, paths) => {
    await mkdir(join(workspace, 'src'), { recursive: true })
    await mkdir(join(workspace, 'dst'), { recursive: true })
    await Bun.write(join(workspace, 'src', 'a.txt'), 'a\n')

    await expect(moveFile({ destination: 'dst', source: 'src' }, paths))
      .rejects.toThrow('already exists; pass overwrite=true to replace it')
    expect(await Bun.file(join(workspace, 'src', 'a.txt')).text()).toBe('a\n')
    expect(await readdir(join(workspace, 'dst'))).toEqual([])
  })
})

test.skipIf(process.platform === 'win32')('runGit takes down a hook helper forked during the kill window', async () => {
  // The bug this pins: only the direct git pid was ever signalled. A hook that
  // traps SIGTERM, forks a helper while dying, and then exits within the grace
  // window left that helper orphaned — the escalation was skipped because the
  // direct child looked finished.
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-run-git-late-fork-'))
  const logPath = join(workspace, 'ticks.log')
  const stub = await writeStubGit(
    `trap '(while :; do echo tick >> "${logPath}"; sleep 0.05; done) &' TERM\n`
      + 'echo out\n'
      + 'while :; do :; done\n',
  )
  try {
    const started = Date.now()
    await expect(runGit(workspace, ['status'], undefined, 800, stub))
      .rejects.toThrow('command timed out after 800ms')
    // TERM at 800ms (trapped, forks the helper), SIGKILL at ~2.8s, then the
    // post-exit group sweep.
    const elapsed = Date.now() - started
    expect(elapsed).toBeGreaterThan(2_500)
    expect(elapsed).toBeLessThan(15_000)

    await Bun.sleep(400)
    const file = Bun.file(logPath)
    const before = (await file.exists()) ? (await file.text()).length : -1
    await Bun.sleep(800)
    const after = (await file.exists()) ? (await file.text()).length : -1
    expect(after).toBe(before)
  } finally {
    await rm(workspace, { force: true, recursive: true })
    await rm(dirname(stub), { force: true, recursive: true })
  }
})
