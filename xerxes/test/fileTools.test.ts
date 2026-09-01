// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { utimesSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { fileStateTracker, setFileFreshnessEnforcement } from '../src/tools/fileState.js'
import {
  DEFAULT_MAX_READ_FILE_BYTES,
  appendFile,
  editFile,
  MAX_READ_WINDOW_CHARS,
  READ_FILE_DEFINITION,
  readFile,
  resolveMaxReadFileBytes,
  setMaxReadFileBytes,
  writeFile,
} from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-file-tools-'))
  try {
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

/** Simulate another writer: change the bytes and push the mtime past the recorded read. */
async function externalWrite(path: string, content: string): Promise<void> {
  await Bun.write(path, content)
  const future = new Date(Date.now() + 5_000)
  utimesSync(path, future, future)
}

afterEach(() => {
  setMaxReadFileBytes(undefined)
  setFileFreshnessEnforcement(undefined)
  fileStateTracker.clear()
})

test('AppendFile rechecks the destination immediately before mutation', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'log.txt'), 'before\n')
    let rechecked = false
    const original = paths.recheck.bind(paths)
    paths.recheck = async target => {
      rechecked = true
      return original(target)
    }
    await appendFile({ file_path: 'log.txt', lines: 'after' }, paths)
    expect(rechecked).toBeTrue()
  })
})

test('ReadFile refuses a one-line minified window that the line limit cannot bound', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'bundle.min.js'), 'a'.repeat(MAX_READ_WINDOW_CHARS + 1))

    // The default read looks complete: one line, no truncation notice, no max_chars.
    const rejection = readFile({ file_path: 'bundle.min.js' }, paths)
    await expect(rejection).rejects.toThrow(String(MAX_READ_WINDOW_CHARS) + '-character ReadFile window ceiling')
    await expect(rejection).rejects.toThrow('GrepTool')
    await expect(rejection).rejects.toThrow('max_chars')
    await expect(readFile({ file_path: 'bundle.min.js', limit: 1 }, paths)).rejects.toThrow('window ceiling')

    // max_chars is the documented escape hatch for lines too long to page by count.
    const capped = await readFile({ file_path: 'bundle.min.js', max_chars: 1_000 }, paths)
    expect(capped).toContain('truncated by max_chars')
    expect(capped.length).toBeLessThan(MAX_READ_WINDOW_CHARS)
  })
})

test('ReadFile refuses an oversized whole-file read and names the line limit that fits', async () => {
  await inWorkspace(async (workspace, paths) => {
    const line = 'x'.repeat(49) + '\n'
    await Bun.write(join(workspace, 'wide.txt'), line.repeat(2_000))

    await expect(readFile({ file_path: 'wide.txt', limit: -1 }, paths)).rejects.toThrow('retry with limit=800')
    await expect(readFile({ file_path: 'wide.txt', limit: 2_000 }, paths)).rejects.toThrow('window ceiling')
    const paged = await readFile({ file_path: 'wide.txt' }, paths)
    expect(paged).toContain('Continue with offset=400, limit=400')
  })
})

test('ReadFile byte ceiling defaults low and resolves environment over runtime setting', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'huge.txt'), 'x'.repeat(DEFAULT_MAX_READ_FILE_BYTES + 1))
    await expect(readFile({ file_path: 'huge.txt' }, paths)).rejects.toThrow(
      String(DEFAULT_MAX_READ_FILE_BYTES) + '-byte ReadFile limit',
    )

    expect(resolveMaxReadFileBytes({})).toBe(DEFAULT_MAX_READ_FILE_BYTES)
    setMaxReadFileBytes(1_024)
    expect(resolveMaxReadFileBytes({})).toBe(1_024)
    expect(resolveMaxReadFileBytes({ XERXES_MAX_READ_FILE_BYTES: '4096' })).toBe(4_096)
    expect(resolveMaxReadFileBytes({ XERXES_MAX_READ_FILE_BYTES: 'not-a-number' })).toBe(1_024)
    setMaxReadFileBytes(undefined)
    expect(resolveMaxReadFileBytes({ XERXES_MAX_READ_FILE_BYTES: '0' })).toBe(DEFAULT_MAX_READ_FILE_BYTES)
    expect(() => setMaxReadFileBytes(0)).toThrow('positive integer')
  })
})

test('ReadFile documents the ceiling, the paths it accepts, and the recoverable outcomes', () => {
  const description = READ_FILE_DEFINITION.function.description ?? ''
  expect(description).toContain(String(MAX_READ_WINDOW_CHARS))
  expect(description).toContain(String(DEFAULT_MAX_READ_FILE_BYTES))
  expect(description).toContain('GrepTool')
  expect(description).toContain('ListDir')
  expect(description).toContain('no line-number prefix')
})

test('FileEditTool retries a failed match against folded quotes and keeps the file typography', async () => {
  await inWorkspace(async (workspace, paths) => {
    const curly = 'const label = “ready”\n'
    await Bun.write(join(workspace, 'curly.ts'), curly)

    const result = await editFile(
      { file_path: 'curly.ts', old_string: 'const label = "ready"', new_string: 'const label = "done"' },
      paths,
    )
    expect(result).toContain('Applied 1 replacement(s)')
    expect(result).toContain('normalized quotes and spaces')
    expect(await Bun.file(join(workspace, 'curly.ts')).text()).toBe('const label = “done”\n')
  })
})

test('FileEditTool folds non-breaking spaces and leaves an unrelated span untouched', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'spaces.md'), 'alpha beta\nplain gamma\n')

    await editFile({ file_path: 'spaces.md', old_string: 'alpha beta', new_string: 'alpha delta' }, paths)
    // The replacement inherits the span's own space so the file does not end up mixed.
    expect(await Bun.file(join(workspace, 'spaces.md')).text()).toBe('alpha delta\nplain gamma\n')

    // A replacement needing more spaces than the span had must not have them turned invisible.
    const widened = await editFile(
      { file_path: 'spaces.md', new_string: 'alpha delta epsilon', old_string: 'alpha delta' },
      paths,
    )
    expect(widened).toContain('normalized quotes and spaces')
    expect(await Bun.file(join(workspace, 'spaces.md')).text()).toBe('alpha delta epsilon\nplain gamma\n')
  })
})

test('FileEditTool matches an LF edit against a CRLF file and writes CRLF back', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'crlf.txt'), 'one\r\ntwo\r\nthree\r\n')

    const result = await editFile(
      { file_path: 'crlf.txt', old_string: 'one\ntwo', new_string: 'ONE\nTWO' },
      paths,
    )
    expect(result).toContain('CRLF line endings')
    expect(await Bun.file(join(workspace, 'crlf.txt')).text()).toBe('ONE\r\nTWO\r\nthree\r\n')
  })
})

test('FileEditTool keeps the ambiguity guard and the not-found error on the forgiving retry', async () => {
  await inWorkspace(async (workspace, paths) => {
    const twice = '“a”\n“a”\n'
    await Bun.write(join(workspace, 'twice.txt'), twice)

    await expect(editFile(
      { file_path: 'twice.txt', old_string: '"a"', new_string: '"b"' },
      paths,
    )).rejects.toThrow('appears 2 times; provide more context or set replace_all=true')
    expect(await Bun.file(join(workspace, 'twice.txt')).text()).toBe(twice)

    expect(await editFile(
      { file_path: 'twice.txt', old_string: '"a"', new_string: '"b"', replace_all: true },
      paths,
    )).toContain('Applied 2 replacement(s)')
    expect(await Bun.file(join(workspace, 'twice.txt')).text()).toBe('“b”\n“b”\n')

    await expect(editFile(
      { file_path: 'twice.txt', old_string: 'never present', new_string: 'x' },
      paths,
    )).rejects.toThrow('was not found exactly')
  })
})

test('FileEditTool leaves an exact match on the fast path and does not fold intentional characters', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'exact.ts'), 'const straight = "keep"\n')

    const result = await editFile(
      { file_path: 'exact.ts', old_string: '"keep"', new_string: '“quoted”' },
      paths,
    )
    expect(result).not.toContain('normalized quotes')
    expect(await Bun.file(join(workspace, 'exact.ts')).text()).toBe('const straight = “quoted”\n')
  })
})

test('FileEditTool refuses to edit a file this session never read, and proceeds after one read', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'edit.ts'), 'const a = 1\n')
    const context = { sessionId: 'session-edit' }
    const edit = { file_path: 'edit.ts', new_string: 'const a = 2', old_string: 'const a = 1' }

    await expect(editFile(edit, paths, context)).rejects.toThrow('requires reading "edit.ts" first')
    expect(await Bun.file(join(workspace, 'edit.ts')).text()).toBe('const a = 1\n')

    await readFile({ file_path: 'edit.ts' }, paths, context)
    expect(await editFile(edit, paths, context)).toContain('Applied 1 replacement(s)')
    expect(await Bun.file(join(workspace, 'edit.ts')).text()).toBe('const a = 2\n')
  })
})

test('a targeted edit on a file changed since the read applies and reports the drift', async () => {
  await inWorkspace(async (workspace, paths) => {
    const path = join(workspace, 'notes.md')
    await Bun.write(path, 'alpha\nbeta\ngamma\n')
    const context = { sessionId: 'session-drift' }
    await readFile({ file_path: 'notes.md' }, paths, context)
    await externalWrite(path, 'alpha\nbeta edited elsewhere\ngamma\n')

    const result = await editFile(
      { file_path: 'notes.md', new_string: 'GAMMA', old_string: 'gamma' },
      paths,
      context,
    )
    expect(result).toContain('[stale-read] notes.md changed on disk')
    expect(result).toContain('+beta edited elsewhere')
    expect(result).toContain('Applied 1 replacement(s)')
    // The other writer's line is still there: the edit ran against the current bytes.
    expect(await Bun.file(path).text()).toBe('alpha\nbeta edited elsewhere\nGAMMA\n')
  })
})

test('WriteFile refuses to overwrite a file changed since the read but still creates new ones', async () => {
  await inWorkspace(async (workspace, paths) => {
    const path = join(workspace, 'config.json')
    await Bun.write(path, '{"a":1}\n')
    const context = { sessionId: 'session-write' }
    await readFile({ file_path: 'config.json' }, paths, context)
    await externalWrite(path, '{"a":2}\n')

    await expect(writeFile(
      { content: '{"a":3}\n', file_path: 'config.json', overwrite: true },
      paths,
      context,
    )).rejects.toThrow('a whole-file write would discard those changes')
    expect(await Bun.file(path).text()).toBe('{"a":2}\n')

    // Creating a file nobody has read is not a lost update, so it stays allowed.
    expect(await writeFile({ content: 'new\n', file_path: 'fresh.txt' }, paths, context)).toContain('(created)')
  })
})

test('a ranged read is a partial view, so drift under it cannot be summarised away', async () => {
  await inWorkspace(async (workspace, paths) => {
    const path = join(workspace, 'long.txt')
    await Bun.write(path, 'one\ntwo\nthree\nfour\n')
    const context = { sessionId: 'session-partial' }
    await readFile({ file_path: 'long.txt', limit: 2, offset: 2 }, paths, context)
    await externalWrite(path, 'one\ntwo\nthree\nfour\nfive\n')

    await expect(editFile(
      { file_path: 'long.txt', new_string: 'THREE', old_string: 'three' },
      paths,
      context,
    )).rejects.toThrow('your read covered only part of the file')
  })
})

test('the freshness kill switch restores the previous read-nothing-first behaviour', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'blind.txt'), 'before\n')
    setFileFreshnessEnforcement(false)

    expect(await editFile(
      { file_path: 'blind.txt', new_string: 'after', old_string: 'before' },
      paths,
      { sessionId: 'session-off' },
    )).toContain('Applied 1 replacement(s)')
    expect(await Bun.file(join(workspace, 'blind.txt')).text()).toBe('after\n')
  })
})

test('max_chars truncation lands under the window ceiling instead of contradicting it', async () => {
  await inWorkspace(async (workspace, paths) => {
    // 60k characters on one line: the whole-file read cannot fit the ceiling,
    // and max_chars=40000 is exactly the documented remedy.
    const text = 'z'.repeat(60_000)
    await Bun.write(join(workspace, 'generated.txt'), text)

    const capped = await readFile({ file_path: 'generated.txt', limit: -1, max_chars: MAX_READ_WINDOW_CHARS }, paths)
    // Previously this threw a ValidationError telling the caller to raise the
    // very ceiling they had just set: the marker was appended past 40000 and
    // then rejected by enforceReadWindowCeiling.
    expect(capped).toContain('truncated by max_chars')
    expect(capped.length).toBeLessThanOrEqual(MAX_READ_WINDOW_CHARS)

    // A small max_chars still truncates to the requested size (marker included
    // in the response but the slice honors the cap).
    const small = await readFile({ file_path: 'generated.txt', max_chars: 100 }, paths)
    expect(small.startsWith('z'.repeat(100))).toBe(true)
    expect(small).toContain('truncated by max_chars')
    expect(small.length).toBeLessThanOrEqual(MAX_READ_WINDOW_CHARS)
  })
})

test('a max_chars above the window ceiling is clamped instead of self-contradicting', async () => {
  await inWorkspace(async (workspace, paths) => {
    const text = 'w'.repeat(60_000)
    await Bun.write(join(workspace, 'wide.txt'), text)

    // max_chars larger than the ceiling used to skip truncation and then be
    // rejected by the ceiling — with a remedy advising the already-supplied
    // parameter.
    const huge = await readFile({ file_path: 'wide.txt', limit: -1, max_chars: 1_000_000 }, paths)
    expect(huge).toContain('truncated by max_chars')
    expect(huge.length).toBeLessThanOrEqual(MAX_READ_WINDOW_CHARS)
    // The slice honours the caller's cap where it is under the ceiling.
    const mid = await readFile({ file_path: 'wide.txt', limit: -1, max_chars: 500 }, paths)
    expect(mid.startsWith('w'.repeat(500))).toBe(true)

    // limit=-1 without any char cap keeps its documented semantics: an
    // oversized whole-file read is an error that names the ceiling, not a cut.
    await expect(readFile({ file_path: 'wide.txt', limit: -1 }, paths))
      .rejects.toThrow(String(MAX_READ_WINDOW_CHARS) + '-character ReadFile window ceiling')
  })
})
