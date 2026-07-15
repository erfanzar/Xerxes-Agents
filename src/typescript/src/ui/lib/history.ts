// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const MAX = 1000
const dir = process.env.XERXES_HOME ?? join(homedir(), '.xerxes')
const file = join(dir, '.xerxes_history')

let cache: string[] | null = null

function normalizeHistoryLine(line: string) {
  return line.trim().replace(/\s+/g, ' ')
}

export function load() {
  if (cache) {
    return cache
  }

  try {
    if (!existsSync(file)) {
      cache = []

      return cache
    }

    const entries: string[] = []
    let current: string[] = []

    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (line.startsWith('+')) {
        current.push(line.slice(1))
      } else if (current.length) {
        entries.push(current.join('\n'))
        current = []
      }
    }

    if (current.length) {
      entries.push(current.join('\n'))
    }

    cache = entries.slice(-MAX)
  } catch {
    cache = []
  }

  return cache
}

export function append(line: string) {
  const trimmed = line.trim()

  if (!trimmed) {
    return
  }

  const items = load()

  if (items.at(-1) === trimmed) {
    return
  }

  items.push(trimmed)

  if (items.length > MAX) {
    items.splice(0, items.length - MAX)
  }

  try {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true })
    }

    const ts = new Date().toISOString().replace('T', ' ').replace('Z', '')

    const encoded = trimmed
      .split('\n')
      .map(l => `+${l}`)
      .join('\n')

    appendFileSync(file, `\n# ${ts}\n${encoded}\n`)
  } catch {
    void 0
  }
}

export function loadHistory(path = file): string[] {
  try {
    if (!existsSync(path)) {
      return []
    }

    return readFileSync(path, 'utf8').split('\n').map(normalizeHistoryLine).filter(Boolean).slice(-MAX)
  } catch {
    return []
  }
}

export function appendHistory(line: string, path = file): void {
  const trimmed = normalizeHistoryLine(line)

  if (!trimmed) {
    return
  }

  const directory = path.slice(0, Math.max(0, path.lastIndexOf('/')))

  try {
    if (directory && !existsSync(directory)) {
      mkdirSync(directory, { recursive: true })
    }

    appendFileSync(path, `${trimmed}\n`)
  } catch {
    void 0
  }
}

export class HistoryCursor {
  private items: string[]
  private index: number

  constructor(items: string[] = []) {
    this.items = items.map(normalizeHistoryLine).filter(Boolean).slice(-MAX)
    this.index = this.items.length
  }

  atLive(): boolean {
    return this.index >= this.items.length
  }

  prev(): string {
    if (!this.items.length) {
      return ''
    }

    this.index = Math.max(0, this.index - 1)

    return this.items[this.index] ?? ''
  }

  next(): string {
    if (!this.items.length) {
      return ''
    }

    this.index = Math.min(this.items.length, this.index + 1)

    return this.atLive() ? '' : (this.items[this.index] ?? '')
  }

  push(line: string): void {
    const trimmed = normalizeHistoryLine(line)

    if (!trimmed) {
      return
    }

    if (this.items.at(-1) !== trimmed) {
      this.items.push(trimmed)

      if (this.items.length > MAX) {
        this.items.splice(0, this.items.length - MAX)
      }
    }

    this.index = this.items.length
  }
}
