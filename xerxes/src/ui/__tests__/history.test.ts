// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { appendFileSync, mkdtempSync, readFileSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { append, HistoryCursor, load, projectKey } from '../lib/history.js'

let home = ''
let projectA = ''
let projectB = ''
const previousHome = process.env.XERXES_HOME
const previousCwd = process.cwd()

const historyPath = () => join(home, '.xerxes_history')

beforeEach(() => {
  // realpath: process.cwd() reports the resolved path, and the workspace key
  // hashes it — a symlinked /tmp would otherwise key writes and reads apart.
  home = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-history-home-')))
  projectA = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-project-a-')))
  projectB = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-project-b-')))
  process.env.XERXES_HOME = home
  process.chdir(projectA)
})

afterEach(() => {
  process.chdir(previousCwd)

  if (previousHome === undefined) {
    delete process.env.XERXES_HOME
  } else {
    process.env.XERXES_HOME = previousHome
  }

  for (const dir of [home, projectA, projectB]) {
    rmSync(dir, { force: true, recursive: true })
  }
})

describe('prompt history', () => {
  it('tags entries with the workspace key and hides other workspaces', () => {
    append('prompt from A')

    expect(readFileSync(historyPath(), 'utf8')).toContain(`project:${projectKey(projectA)}`)

    process.chdir(projectB)
    append('prompt from B')

    expect(load()).toEqual(['prompt from B'])

    process.chdir(projectA)
    expect(load()).toEqual(['prompt from A'])
  })

  it('keeps untagged legacy entries visible in every workspace', () => {
    appendFileSync(historyPath(), '\n# 2026-01-01 00:00:00.000\n+legacy prompt\n')
    append('tagged prompt')

    expect(load()).toEqual(['legacy prompt', 'tagged prompt'])

    process.chdir(projectB)
    expect(load()).toEqual(['legacy prompt'])
  })

  it('re-reads when another process grew the file', () => {
    append('mine')
    expect(load()).toEqual(['mine'])

    // A second TUI appending to the shared file: the module cache used to hide
    // this for the lifetime of the process.
    appendFileSync(historyPath(), `\n# 2026-01-01 00:00:00.000 project:${projectKey(projectA)}\n+theirs\n`)

    expect(load()).toEqual(['mine', 'theirs'])
  })

  it('refills the same array instance so mounted refs see new entries', () => {
    const first = load()

    append('later')

    expect(load()).toBe(first)
    expect(first).toContain('later')
  })

  it('round-trips multi-line prompts with their newlines', () => {
    append('line one\nline two')

    expect(load()).toEqual(['line one\nline two'])
  })
})

describe('HistoryCursor', () => {
  it('recalls multi-line prompts unchanged', () => {
    const cursor = new HistoryCursor(['single', 'first\n  second'])

    expect(cursor.prev()).toBe('first\n  second')
    expect(cursor.prev()).toBe('single')
  })

  it('drops blank entries', () => {
    const cursor = new HistoryCursor(['  ', 'kept'])

    expect(cursor.prev()).toBe('kept')
    expect(cursor.prev()).toBe('kept')
  })
})
