// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { GLOB_TOOL_DEFINITION, GREP_TOOL_DEFINITION, globFiles, grepFiles } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'

async function inWorkspace(
  run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>,
  repository = true,
): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-ignore-search-'))
  try {
    if (repository) await mkdir(join(workspace, '.git'))
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

async function files(workspace: string, names: readonly string[]): Promise<void> {
  for (const name of names) await Bun.write(join(workspace, name), 'search-needle\n')
}

/** Exercise the same ignore decisions through both model-facing search implementations. */
async function visibility(
  paths: WorkspacePathResolver,
  visible: readonly string[],
  ignored: readonly string[],
  options: { path?: string; include_ignored?: boolean } = {},
): Promise<void> {
  const globbed = await globFiles({ pattern: '**/*', include_hidden: true, ...options }, paths)
  const grepped = (await grepFiles({ pattern: 'search-needle', ...options }, paths)).split('\n')
  for (const name of visible) {
    expect(globbed).toContain(name)
    expect(grepped).toContain(name)
  }
  for (const name of ignored) {
    expect(globbed).not.toContain(name)
    expect(grepped).not.toContain(name)
  }
}

test('both search schemas expose include_ignored with a false default', () => {
  for (const definition of [GLOB_TOOL_DEFINITION, GREP_TOOL_DEFINITION]) {
    expect(definition.function.parameters).toMatchObject({
      properties: { include_ignored: { type: 'boolean', default: false } },
    })
    expect(definition.function.description).not.toContain('.gitignore is not honored')
  }
})

test('ignored dependency/build directories are pruned by default and restored by include_ignored', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'node_modules/\nbuild/\n')
    await files(workspace, ['src/main.txt', 'node_modules/dependency.txt', 'build/output.txt', '.git/internal.txt'])
    await visibility(paths, ['src/main.txt'], ['node_modules/dependency.txt', 'build/output.txt', '.git/internal.txt'])
    expect(await globFiles({ pattern: '**/*' }, paths)).not.toContain('node_modules')
    await visibility(paths, ['src/main.txt', 'node_modules/dependency.txt', 'build/output.txt'], ['.git/internal.txt'], { include_ignored: true })
  })
})

test('nested .gitignore files are relative to their own directory and override parent patterns', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), '*.txt\n')
    await Bun.write(join(workspace, 'src/.gitignore'), '!keep.txt\n/only-here.txt\n')
    await files(workspace, ['root.txt', 'src/keep.txt', 'src/only-here.txt', 'src/deep/keep.txt', 'other/keep.txt'])
    await visibility(paths, ['src/keep.txt', 'src/deep/keep.txt'], ['root.txt', 'src/only-here.txt', 'other/keep.txt'])
    await visibility(paths, ['src/keep.txt', 'src/deep/keep.txt'], ['src/only-here.txt'], { path: 'src' })
  })
})

test('negation uses the last matching rule but cannot re-include a child of an excluded directory', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'cache/*\n!cache/keep.txt\nblocked/\n!blocked/keep.txt\n*.log.txt\n!keep.log.txt\nagain.log.txt\n')
    await Bun.write(join(workspace, 'blocked/.gitignore'), '!keep.txt\n')
    await files(workspace, ['cache/keep.txt', 'cache/drop.txt', 'blocked/keep.txt', 'keep.log.txt', 'again.log.txt'])
    await visibility(paths, ['cache/keep.txt', 'keep.log.txt'], ['cache/drop.txt', 'blocked/keep.txt', 'again.log.txt'])
    await visibility(paths, [], ['blocked/keep.txt'], { path: 'blocked' })
    expect(await grepFiles({ pattern: 'search-needle', path: 'blocked/keep.txt' }, paths)).toBe('No matches found.')
    expect(await grepFiles({ pattern: 'search-needle', path: 'blocked/keep.txt', include_ignored: true }, paths)).toBe('blocked/keep.txt')
  })
})

test('re-including a parent directory makes subsequent child negations effective', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'blocked/\n!blocked/\nblocked/*\n!blocked/keep.txt\n')
    await files(workspace, ['blocked/keep.txt', 'blocked/drop.txt'])
    await visibility(paths, ['blocked/keep.txt'], ['blocked/drop.txt'])
  })
})

test('leading or middle slashes anchor rules while basename rules apply at any depth', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), '/root.txt\nloose.txt\nassets/fixed.txt\n')
    await files(workspace, ['root.txt', 'deep/root.txt', 'loose.txt', 'deep/loose.txt', 'assets/fixed.txt', 'deep/assets/fixed.txt'])
    await visibility(paths, ['deep/root.txt', 'deep/assets/fixed.txt'], ['root.txt', 'loose.txt', 'deep/loose.txt', 'assets/fixed.txt'])
  })
})

test('directory-only rules do not exclude a regular file with the same basename', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'artifact/\n')
    await files(workspace, ['artifact', 'other/artifact/output.txt', 'other/visible.txt'])
    await visibility(paths, ['artifact', 'other/visible.txt'], ['other/artifact/output.txt'])
  })
})

test('.git/info/exclude applies below .gitignore in precedence', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.git/info/exclude'), '*.txt\n')
    await Bun.write(join(workspace, '.gitignore'), '!keep.txt\n')
    await files(workspace, ['drop.txt', 'deep/drop.txt', 'keep.txt'])
    await visibility(paths, ['keep.txt'], ['drop.txt', 'deep/drop.txt'])
  })
})

test('a .git file identifies a work tree and reads an in-workspace gitdir exclude file', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'tree/.git'), 'gitdir: ../metadata\n')
    await Bun.write(join(workspace, 'metadata/info/exclude'), 'excluded.txt\n')
    await Bun.write(join(workspace, 'tree/.gitignore'), 'ignored.txt\n')
    await files(workspace, ['tree/visible.txt', 'tree/ignored.txt', 'tree/excluded.txt'])
    await visibility(paths, ['tree/visible.txt'], ['tree/ignored.txt', 'tree/excluded.txt'], { path: 'tree' })
  }, false)
})

test('without a repository .gitignore files do not change either search', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'build/\n*.txt\n')
    await files(workspace, ['build/output.txt', 'visible.txt'])
    await visibility(paths, ['build/output.txt', 'visible.txt'], [])
  }, false)
})

test('wildcards cover *, ?, ranges, negated classes, and globstars at all three positions', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), [
      'star*.txt', 'single?.txt', 'range[0-2].txt', 'negated[!a].txt',
      '**/leaf.txt', 'tree/**', 'a/**/z.txt',
    ].join('\n'))
    const ignored = ['star.txt', 'star-more.txt', 'single1.txt', 'range2.txt', 'negatedb.txt', 'leaf.txt', 'deep/leaf.txt', 'tree/a.txt', 'tree/deep/b.txt', 'a/z.txt', 'a/b/c/z.txt']
    const visible = ['single12.txt', 'range3.txt', 'negateda.txt', 'deep/visible.txt', 'other/a/z.txt']
    await files(workspace, [...ignored, ...visible])
    await visibility(paths, visible, ignored)
    expect(await globFiles({ pattern: 'tree', include_hidden: true }, paths)).toEqual(['tree'])
  })
})

test('comments, CRLF, escaped leading symbols, escaped wildcards, and trailing spaces follow gitignore syntax', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), '\r\n# comment\r\nignored.txt   \r\n\\#literal.txt\r\n\\!literal.txt\r\nliteral\\*.txt\r\ntrailing.txt\\ \r\n')
    const ignored = ['ignored.txt', '#literal.txt', '!literal.txt', 'literal*.txt', 'trailing.txt ']
    const visible = ['comment', 'literal-other.txt', 'trailing.txt', 'ignored.txt ']
    await files(workspace, [...ignored, ...visible])
    await visibility(paths, visible, ignored)
  })
})

test('ignore-file and gitdir symlinks cannot cause reads outside the workspace', async () => {
  await inWorkspace(async (workspace, paths) => {
    const outside = await mkdtemp(join(tmpdir(), 'xerxes-ignore-outside-'))
    try {
      await Bun.write(join(outside, 'rules'), '*\n')
      await symlink(join(outside, 'rules'), join(workspace, '.gitignore'))
      await symlink(outside, join(workspace, '.git/info'))
      await files(workspace, ['visible.txt'])
      await visibility(paths, ['visible.txt'], [])
      await expect(globFiles({ pattern: '**/*', path: '../' }, paths)).rejects.toThrow('escapes workspace')
      await expect(grepFiles({ pattern: 'x', path: outside }, paths)).rejects.toThrow('escapes workspace')
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })
})

test('ignore caches last for only one search, so changed rules affect the next walk', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'target.txt\n')
    await files(workspace, ['target.txt'])
    await visibility(paths, [], ['target.txt'])
    await Bun.write(join(workspace, '.gitignore'), '')
    await visibility(paths, ['target.txt'], [])
  })
})

test('Bun glob syntax, directory matches, hidden filtering, symlinks, and result limits are preserved', async () => {
  await inWorkspace(async (workspace, paths) => {
    await files(workspace, ['src/a.txt', 'src/b.md', '.hidden.txt', '.hidden/deep.txt'])
    await symlink(join(workspace, 'src'), join(workspace, 'linked'))
    expect(await globFiles({ pattern: 'src/*.{txt,md}' }, paths)).toEqual(['src/a.txt', 'src/b.md'])
    expect(await globFiles({ pattern: 'src/' }, paths)).toEqual(['src'])
    expect(await globFiles({ pattern: '.hidden.txt' }, paths)).toEqual(['.hidden.txt'])
    expect(await globFiles({ pattern: '**/*.txt' }, paths)).toEqual(['src/a.txt'])
    expect(await globFiles({ pattern: '**/*.txt', include_hidden: true }, paths)).toEqual(['.hidden.txt', '.hidden/deep.txt', 'src/a.txt'])
    expect(await grepFiles({ pattern: 'search-needle' }, paths)).toBe('src/a.txt\nsrc/b.md')
    expect(await globFiles({ pattern: '**/*', max_results: 1 }, paths)).toHaveLength(2)
    expect(await grepFiles({ pattern: 'search-needle', max_results: 1 }, paths)).toContain('result limit reached')
  })
})

test('the ignore-aware walk retains Bun glob matching when there are no ignore rules', async () => {
  await inWorkspace(async (workspace, paths) => {
    await files(workspace, ['src/a.txt', 'src/b.md', 'src/deep/c.txt', 'src/.hidden.txt', 'other/b.txt', '.hidden.txt', '.hidden/deep.txt'])
    const patterns = [
      '**/*', '**/*.txt', '**/{a,b}.txt', '{src,other}/**/*.txt', '**/', '*',
      'src/', 'src', 'src/*', 'src/**', 'src/**/', './src/*.txt', '.hidden.txt',
      '.hidden/*', '**/.hidden.txt', 'src/.*', '**/{a,.hidden}.txt', 'src/{a,b}.*', '!src', 'src/[ab].txt',
    ]
    for (const include_hidden of [false, true]) {
      for (const pattern of patterns) {
        const options = { pattern, include_hidden }
        expect({ pattern, include_hidden, matches: await globFiles(options, paths) }).toEqual({
          pattern, include_hidden, matches: await globFiles({ ...options, include_ignored: true }, paths),
        })
      }
    }
  })
})

test('repository discovery and ignore-file reads stop at the workspace boundary', async () => {
  await inWorkspace(async workspace => {
    await Bun.write(join(workspace, '.gitignore'), '*.txt\n')
    await Bun.write(join(workspace, 'inner/.gitignore'), '*.txt\n')
    await files(workspace, ['inner/visible.txt'])
    const innerPaths = new WorkspacePathResolver(join(workspace, 'inner'))
    await visibility(innerPaths, ['visible.txt'], [])
  })
})

test('a nested worktree marker takes precedence over the enclosing repository', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), '*.txt\n')
    await mkdir(join(workspace, 'nested/.git'), { recursive: true })
    await Bun.write(join(workspace, 'nested/.gitignore'), 'drop.txt\n')
    await files(workspace, ['nested/keep.txt', 'nested/drop.txt'])
    await visibility(paths, ['nested/keep.txt'], ['nested/drop.txt'], { path: 'nested' })
  })
})

test('linked worktrees use a workspace-contained common directory for info/exclude', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'tree/.git'), 'gitdir: ../metadata/worktrees/tree\n')
    await Bun.write(join(workspace, 'metadata/worktrees/tree/commondir'), '../..\n')
    await Bun.write(join(workspace, 'metadata/info/exclude'), 'drop.txt\n')
    await files(workspace, ['tree/keep.txt', 'tree/drop.txt'])
    await visibility(paths, ['tree/keep.txt'], ['tree/drop.txt'], { path: 'tree' })
  }, false)
})

test('bracket classes include literal closing brackets, ranges, and POSIX ASCII classes', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'class[[:digit:]].txt\nclosing[]a].txt\nrange[a-c].txt\ndescending[z-a].txt\n')
    await files(workspace, ['class1.txt', 'classa.txt', 'closing].txt', 'closinga.txt', 'closingb.txt', 'rangeb.txt', 'rangez.txt', 'descendingz.txt', 'descendinga.txt'])
    await visibility(paths, ['classa.txt', 'closingb.txt', 'rangez.txt', 'descendinga.txt'], ['class1.txt', 'closing].txt', 'closinga.txt', 'rangeb.txt', 'descendingz.txt'])
  })
})

test('git wildcards count UTF-8 bytes while non-ASCII literal patterns still match', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, '.gitignore'), 'single?.txt\ntwo??.txt\nliteralé.txt\nclass[é].txt\n')
    await files(workspace, ['singleé.txt', 'singlea.txt', 'twoé.txt', 'literalé.txt', 'classé.txt'])
    await visibility(paths, ['singleé.txt', 'classé.txt'], ['singlea.txt', 'twoé.txt', 'literalé.txt'])
  })
})

test('GrepTool searches files larger than the former 1 MB ceiling and still skips binary files', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'large.txt'), 'x'.repeat(2_000_000) + '\nsearch-needle\n')
    await Bun.write(join(workspace, 'binary.txt'), 'search-needle\0\n')
    expect(await grepFiles({ pattern: 'search-needle', output_mode: 'content' }, paths)).toBe('large.txt:2:search-needle')
  })
})

test('GrepTool reports the count and byte limit for oversized skips even when no files match', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'huge.txt'), 'x'.repeat(8_388_609))
    await Bun.write(join(workspace, 'huge2.txt'), 'x'.repeat(8_388_609))
    const noMatches = await grepFiles({ pattern: 'absent' }, paths)
    expect(noMatches).toContain('No matches found.')
    expect(noMatches).toContain('Skipped 2 files')
    expect(noMatches).toContain('8388608-byte')
    await files(workspace, ['visible.txt'])
    const matches = await grepFiles({ pattern: 'search-needle' }, paths)
    expect(matches).toContain('visible.txt')
    expect(matches).toContain('Skipped 2 files')
    const direct = await grepFiles({ pattern: 'absent', path: 'huge.txt' }, paths)
    expect(direct).toContain('Skipped 1 file')
  })
})
