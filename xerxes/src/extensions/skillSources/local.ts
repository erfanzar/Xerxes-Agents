// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, readdir } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'

import {
  normalizeSkillSearchLimit,
  requireSkillSearchQuery,
  SkillSourceError,
  SkillSourceNotFoundError,
  type SkillBundle,
  type SkillSearchHit,
  type SkillSource,
} from './base.js'

const SKILL_MARKDOWN_FILE = 'SKILL.md'

export interface LocalSkillSourceOptions {
  /** Root directory containing nested skill directories. */
  readonly root: string
}

/** Read SKILL.md files from a local directory tree without using a package or remote fallback. */
export class LocalSkillSource implements SkillSource {
  readonly name = 'local'
  readonly root: string

  constructor(options: LocalSkillSourceOptions) {
    if (!options.root.trim()) {
      throw new SkillSourceError('local skill source root must be a non-empty path')
    }
    this.root = resolve(options.root)
  }

  async search(query: string, limit = 20): Promise<readonly SkillSearchHit[]> {
    const normalizedQuery = requireSkillSearchQuery(query).toLowerCase()
    const maximum = normalizeSkillSearchLimit(limit)
    if (maximum === 0) return []

    const hits: SkillSearchHit[] = []
    for (const skillPath of await localSkillMarkdownPaths(this.root)) {
      let body: string
      try {
        body = await readFile(skillPath, 'utf8')
      } catch {
        // One unreadable third-party skill must not hide the rest of a local catalogue.
        continue
      }
      const name = basename(dirname(skillPath))
      if (body.toLowerCase().includes(normalizedQuery) || name.toLowerCase().includes(normalizedQuery)) {
        hits.push({
          name,
          description: firstDescriptionLine(body),
          sourceName: this.name,
          version: extractSkillVersion(body),
          tags: [],
        })
        if (hits.length >= maximum) break
      }
    }
    return hits
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    const name = validateLocalSkillIdentifier(identifier)
    const candidates = await localSkillMarkdownPaths(this.root)
    const skillPath = candidates.find(path => basename(dirname(path)) === name)
    if (skillPath === undefined) {
      throw new SkillSourceNotFoundError(this.root, name)
    }

    let body: string
    try {
      body = await readFile(skillPath, 'utf8')
    } catch (error) {
      throw new SkillSourceError(`cannot read local skill ${name}: ${errorMessage(error)}`, { cause: error })
    }
    return {
      name,
      version: extractSkillVersion(body),
      bodyMarkdown: body,
      metadata: {},
      sourceName: this.name,
    }
  }
}

/** Extract `version:` from the first 20 lines, matching the Python source's fallback contract. */
export function extractSkillVersion(body: string): string {
  for (const line of body.split(/\r?\n/).slice(0, 20)) {
    const match = /^version:\s*(.*)$/i.exec(line)
    if (match?.[1] !== undefined) {
      return match[1].trim().replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, '$1$2')
    }
  }
  return '0.0.1'
}

/** Return the first human-readable non-frontmatter, non-heading line in a SKILL.md document. */
export function firstDescriptionLine(body: string): string {
  const lines = body.split(/\r?\n/)
  let frontmatter = lines[0]?.trim() === '---'
  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index]
    if (rawLine === undefined) continue
    const line = rawLine.trim()
    if (frontmatter) {
      if (index > 0 && line === '---') frontmatter = false
      continue
    }
    if (line && !line.startsWith('#')) return line.slice(0, 200)
  }
  return ''
}

function validateLocalSkillIdentifier(identifier: string): string {
  if (
    typeof identifier !== 'string'
    || !identifier
    || identifier === '.'
    || identifier === '..'
    || identifier.includes('/')
    || identifier.includes('\\')
    || identifier.includes('\0')
  ) {
    throw new SkillSourceError('local skill identifier must be a single directory name')
  }
  return identifier
}

async function localSkillMarkdownPaths(root: string): Promise<readonly string[]> {
  const found: string[] = []
  const directories = [root]
  while (directories.length) {
    const directory = directories.pop()
    if (directory === undefined) continue
    try {
      const entries = await readdir(directory, { encoding: 'utf8', withFileTypes: true })
      entries.sort((left, right) => left.name.localeCompare(right.name))
      for (const entry of entries) {
        const path = join(directory, entry.name)
        if (entry.isDirectory()) {
          directories.push(path)
        } else if (entry.isFile() && entry.name === SKILL_MARKDOWN_FILE) {
          found.push(path)
        }
      }
    } catch {
      continue
    }
  }
  return found.sort((left, right) => left.localeCompare(right))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
