// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, readdirSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { expandSnips } from '../app/useSubmission.js'
import { pasteTokenLabel } from '../lib/text.js'
import {
  pasteTokenHash,
  prunePasteStore,
  readPasteSnippet,
  storePasteSnippet,
  tagPasteToken
} from '../lib/pasteStore.js'

let home = ''
const previousHome = process.env.XERXES_HOME

const tokenFor = (text: string) => {
  const hash = storePasteSnippet(text)

  expect(hash).not.toBeNull()

  return tagPasteToken(pasteTokenLabel(text, text.split('\n').length), hash!)
}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'xerxes-paste-home-'))
  process.env.XERXES_HOME = home
})

afterEach(() => {
  if (previousHome === undefined) {
    delete process.env.XERXES_HOME
  } else {
    process.env.XERXES_HOME = previousHome
  }

  rmSync(home, { force: true, recursive: true })
})

describe('paste store', () => {
  it('content-addresses a snippet and reads it back', () => {
    const text = 'Traceback\n  line 1\n  line 2'
    const hash = storePasteSnippet(text)

    expect(hash).toMatch(/^[0-9a-f]{8}$/)
    expect(readPasteSnippet(hash!)).toBe(text)
  })

  it('reuses one file for identical content', () => {
    expect(storePasteSnippet('same')).toBe(storePasteSnippet('same'))
    expect(readdirSync(join(home, 'pastes'))).toHaveLength(1)
  })

  it('lengthens the key instead of returning a colliding one', () => {
    const text = 'colliding paste'
    const short = storePasteSnippet(text)!

    // Squat the short key with foreign bytes, exactly as a truncated-digest
    // collision would: the store must not hand that content back for `text`.
    writeFileSync(join(home, 'pastes', `${short}.txt`), 'not the paste')

    const longer = storePasteSnippet(text)

    expect(longer).not.toBe(short)
    expect(longer).toMatch(/^[0-9a-f]{16}$/)
    expect(readPasteSnippet(longer!)).toBe(text)
  })

  it('refuses hashes that are not bare digests', () => {
    expect(readPasteSnippet('../../etc/passwd')).toBeNull()
    expect(readPasteSnippet('')).toBeNull()
  })

  it('returns null for an unknown hash', () => {
    expect(readPasteSnippet('deadbeef')).toBeNull()
  })

  it('evicts the least recently used entries past the file ceiling', () => {
    const dir = join(home, 'pastes')
    const hashes = ['old one', 'middle one', 'new one'].map((text, index) => {
      const hash = storePasteSnippet(text)!
      const stamp = new Date(Date.now() + index * 1000)

      utimesSync(join(dir, `${hash}.txt`), stamp, stamp)

      return hash
    })

    prunePasteStore(process.env, { maxFiles: 2 })

    expect(readPasteSnippet(hashes[0]!)).toBeNull()
    expect(readPasteSnippet(hashes[1]!)).toBe('middle one')
    expect(readPasteSnippet(hashes[2]!)).toBe('new one')
  })

  it('evicts past the byte ceiling', () => {
    const dir = join(home, 'pastes')
    const older = storePasteSnippet('x'.repeat(2048))!
    const newer = storePasteSnippet('y'.repeat(64))!
    const past = new Date(Date.now() - 60_000)

    utimesSync(join(dir, `${older}.txt`), past, past)
    prunePasteStore(process.env, { maxBytes: 1024 })

    expect(readPasteSnippet(older)).toBeNull()
    expect(readPasteSnippet(newer)).toHaveLength(64)
  })
})

describe('paste tokens', () => {
  it('stamps the hash inside the closing brackets and parses it back', () => {
    const token = tagPasteToken('[[ stack.. [12 lines] ]]', 'a1b2c3d4')

    expect(token.startsWith('[[ ')).toBe(true)
    expect(token.endsWith(' ]]')).toBe(true)
    expect(pasteTokenHash(token)).toBe('a1b2c3d4')
  })

  it('leaves a non-token label alone', () => {
    expect(tagPasteToken('plain', 'a1b2c3d4')).toBe('plain')
    expect(pasteTokenHash('[[ no hash here ]]')).toBeNull()
  })
})

describe('expandSnips', () => {
  it('expands a live composer snippet', () => {
    const text = 'line a\nline b'
    const token = tokenFor(text)

    expect(expandSnips([{ label: token, text }])(`see ${token}`)).toBe(`see ${text}`)
  })

  it('expands from the token alone once the snippet is gone', () => {
    // History recall / queue replay / snippet eviction all land here: without
    // the stamped hash the model received the literal placeholder.
    const text = 'evicted paste body'
    const token = tokenFor(text)

    expect(expandSnips([])(`see ${token}`)).toBe(`see ${text}`)
  })

  it('leaves an unresolvable token verbatim', () => {
    const token = '[[ mystery [3 lines] ]]'

    expect(expandSnips([])(token)).toBe(token)
  })

  it('expands repeated tokens of the same content', () => {
    const text = 'twice'
    const token = tokenFor(text)

    expect(expandSnips([{ label: token, text }])(`${token} ${token}`)).toBe(`${text} ${text}`)
  })
})
