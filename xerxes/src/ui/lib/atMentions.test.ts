// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it, vi } from 'vitest'

import {
  AT_MENTION_TRIGGERS,
  atMentionTokens,
  completeAtMention,
  expandAtMention,
  expandAtMentionsInText,
  parseAtMentionToken,
  type AtMentionDirectoryEntry,
  type AtMentionFileSystemPort,
  type AtMentionGitPort,
  type AtMentionOptions,
  type AtMentionPathInfo
} from './atMentions.js'

const ROOT = '/workspace'

const pathInfo: Readonly<Record<string, AtMentionPathInfo>> = {
  [ROOT]: { path: ROOT, kind: 'directory' },
  '/workspace/README.md': { path: '/workspace/README.md', kind: 'file' },
  '/workspace/docs': { path: '/workspace/docs', kind: 'directory' },
  '/workspace/docs/guide.md': { path: '/workspace/docs/guide.md', kind: 'file' },
  '/workspace/docs/zebra.txt': { path: '/workspace/docs/zebra.txt', kind: 'file' },
  '/workspace/link': { path: '/outside', kind: 'directory' },
  '/workspace/src': { path: '/workspace/src', kind: 'directory' },
  '/workspace/src/main.ts': { path: '/workspace/src/main.ts', kind: 'file' },
  '/workspace/src/nested': { path: '/workspace/src/nested', kind: 'directory' }
}

const directoryEntries: Readonly<Record<string, readonly AtMentionDirectoryEntry[]>> = {
  [ROOT]: [
    { name: 'src', path: '/workspace/src' },
    { name: 'README.md', path: '/workspace/README.md' },
    { name: 'link', path: '/workspace/link' },
    { name: '..', path: '/outside' }
  ],
  '/workspace/docs': [
    { name: 'zebra.txt', path: '/workspace/docs/zebra.txt' },
    { name: 'guide.md', path: '/workspace/docs/guide.md' }
  ],
  '/workspace/src': [
    { name: 'nested', path: '/workspace/src/nested' },
    { name: 'main.ts', path: '/workspace/src/main.ts' }
  ]
}

function createFileSystem(): AtMentionFileSystemPort {
  return {
    inspect: async candidate => pathInfo[candidate],
    readDirectory: async candidate => directoryEntries[candidate] ?? [],
    readTextFile: async candidate => {
      if (candidate === '/workspace/src/main.ts') {
        return 'export const main = true\n'
      }
      if (candidate === '/workspace/README.md') {
        return '# Xerxes\n'
      }
      if (candidate === '/workspace/docs/guide.md') {
        return '# Guide\n'
      }
      throw new Error(`not readable: ${candidate}`)
    }
  }
}

function createGit(): AtMentionGitPort {
  return {
    diff: vi.fn(async (_root: string, mode: 'staged' | 'unstaged') => {
      return mode === 'staged' ? 'staged diff' : 'unstaged diff'
    }),
    logOne: vi.fn(async (_root: string, ref: string) => `abc123 ${ref}`),
    listRefs: vi.fn(async () => ['feature/mentions', 'main'])
  }
}

function options(git: AtMentionGitPort = createGit()): AtMentionOptions {
  return { workspaceRoot: ROOT, fileSystem: createFileSystem(), git, maxResults: 10 }
}

describe('parseAtMentionToken', () => {
  it('finds the active token before a cursor and exposes a bare trigger token', () => {
    expect(parseAtMentionToken('open @file:src/ma')).toEqual({
      cursor: 17,
      trigger: '@file:',
      remainder: 'src/ma',
      start: 5,
      textBeforeToken: 'open '
    })
    expect(parseAtMentionToken('open @gi')).toMatchObject({ trigger: '@', remainder: 'gi', start: 5 })
    expect(parseAtMentionToken('open @file:src/main.ts then')).toBeUndefined()
    expect(parseAtMentionToken('open @file:src/main.ts', -1)).toBeUndefined()
  })

  it('extracts only supported expansion tokens in document order', () => {
    expect(atMentionTokens('see @file:src/main.ts @diff @staged @git:main @url:https://example.test a@b.test')).toEqual(
      ['@file:src/main.ts', '@diff', '@staged', '@git:main', '@url:https://example.test']
    )
  })
})

describe('completeAtMention', () => {
  it('completes triggers, contained file/folder paths, and host-provided git refs', async () => {
    const configured = options()

    expect(await completeAtMention('say @fi', configured)).toEqual(
      AT_MENTION_TRIGGERS.map(trigger => ({
        kind: 'trigger',
        display: trigger,
        replacement: trigger,
        replaceStart: 4
      }))
    )
    expect(await completeAtMention('open @file:src/m', configured)).toEqual([
      {
        kind: 'file',
        display: 'main.ts',
        replacement: 'src/main.ts',
        replaceStart: 11
      }
    ])
    expect(await completeAtMention('open @folder:s', configured)).toEqual([
      {
        kind: 'folder',
        display: 'src/',
        replacement: 'src/',
        replaceStart: 13
      }
    ])
    expect(await completeAtMention('@git:fe', configured)).toEqual([
      {
        kind: 'git',
        display: 'feature/mentions',
        replacement: 'feature/mentions',
        replaceStart: 5
      }
    ])
  })

  it('does not complete lexical escapes or symlink-resolved paths outside the workspace', async () => {
    const configured = options()

    expect(await completeAtMention('open @file:../', configured)).toEqual([])
    expect(await completeAtMention('open @folder:l', configured)).toEqual([])
  })
})

describe('expandAtMention', () => {
  it('expands file, folder, git, staged, diff, and URL mentions through injected ports', async () => {
    const git = createGit()
    const configured = options(git)

    await expect(expandAtMention('@file:src/main.ts', configured)).resolves.toEqual({
      token: '@file:src/main.ts',
      kind: 'file',
      payload: 'export const main = true\n',
      error: ''
    })
    await expect(expandAtMention('@folder:docs', configured)).resolves.toEqual({
      token: '@folder:docs',
      kind: 'folder',
      payload: 'guide.md\nzebra.txt',
      error: ''
    })
    await expect(expandAtMention('@diff', configured)).resolves.toMatchObject({
      kind: 'diff',
      payload: 'unstaged diff',
      error: ''
    })
    await expect(expandAtMention('@staged', configured)).resolves.toMatchObject({
      kind: 'staged',
      payload: 'staged diff',
      error: ''
    })
    await expect(expandAtMention('@git:feature/mentions', configured)).resolves.toMatchObject({
      kind: 'git',
      payload: 'abc123 feature/mentions',
      error: ''
    })
    await expect(expandAtMention('@url:https://example.test/a?q=1', configured)).resolves.toEqual({
      token: '@url:https://example.test/a?q=1',
      kind: 'url',
      payload: 'https://example.test/a?q=1',
      error: ''
    })

    expect(git.diff).toHaveBeenNthCalledWith(1, ROOT, 'unstaged')
    expect(git.diff).toHaveBeenNthCalledWith(2, ROOT, 'staged')
    expect(git.logOne).toHaveBeenCalledWith(ROOT, 'feature/mentions')
  })

  it('returns explicit containment and unavailable-port errors without reading outside content', async () => {
    const fileSystem = createFileSystem()
    const readTextFile = vi.spyOn(fileSystem, 'readTextFile')
    const configured = { ...options(), fileSystem }

    await expect(expandAtMention('@file:../secret.txt', configured)).resolves.toMatchObject({
      kind: 'file',
      payload: '',
      error: 'escapes workspace root'
    })
    await expect(expandAtMention('@folder:link', configured)).resolves.toMatchObject({
      kind: 'folder',
      payload: '',
      error: 'escapes workspace root'
    })
    await expect(expandAtMention('@diff', { ...configured, git: undefined })).resolves.toMatchObject({
      kind: 'diff',
      payload: '',
      error: 'git port not configured'
    })
    expect(readTextFile).not.toHaveBeenCalled()
  })

  it('expands every supported token in order and leaves unrelated @ text alone', async () => {
    const expanded = await expandAtMentionsInText(
      'mail a@b.test then @url:https://example.test then @diff then @file:src/main.ts',
      options()
    )

    expect(expanded.map(item => item.token)).toEqual(['@url:https://example.test', '@diff', '@file:src/main.ts'])
    expect(expanded.map(item => item.kind)).toEqual(['url', 'diff', 'file'])
  })
})
