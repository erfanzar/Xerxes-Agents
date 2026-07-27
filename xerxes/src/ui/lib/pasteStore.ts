// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Content-addressed spill file for collapsed pastes.
//
// The composer replaces a big paste with a short display token and keeps the
// real text in React state. That state is cleared on submit and trimmed when
// too many snippets pile up, so any later re-use of the token — history recall,
// queue replay, resume — used to submit the literal placeholder to the model
// with no error at all. Writing the text to a file named after its own hash and
// stamping that hash into the token makes the token self-resolving: expansion
// only ever needs the token itself.
import { createHash } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  unlinkSync,
  utimesSync,
  writeFileSync
} from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

// A truncated digest can collide. Short keys keep the visible token compact;
// the longer ones are only reached when a shorter key already holds different
// bytes, so a collision degrades the token's length instead of handing the
// model somebody else's paste.
const HASH_LENGTHS = [8, 16, 32, 64] as const
const HASH_RE = /^[0-9a-f]{8,64}$/
const TOKEN_HASH_RE = /#([0-9a-f]{8,64})\s*\]\]\s*$/
const MAX_FILES = 512
const MAX_TOTAL_BYTES = 32 * 1024 * 1024

const pasteDirectory = (env: NodeJS.ProcessEnv = process.env) =>
  join((env.XERXES_HOME ?? '').trim() || join(homedir(), '.xerxes'), 'pastes')

const snippetPath = (hash: string, env: NodeJS.ProcessEnv = process.env) => join(pasteDirectory(env), `${hash}.txt`)

/**
 * Persist `text` and return the hash to stamp into its token, or null when the
 * store is unusable (read-only home, full disk). Callers must treat null as
 * "keep the in-memory snippet only" rather than as a failure to paste.
 */
export function storePasteSnippet(text: string, env: NodeJS.ProcessEnv = process.env): null | string {
  if (!text) {
    return null
  }

  try {
    const dir = pasteDirectory(env)

    if (!existsSync(dir)) {
      mkdirSync(dir, { mode: 0o700, recursive: true })
    }

    const digest = createHash('sha256').update(text).digest('hex')

    for (const length of HASH_LENGTHS) {
      const hash = digest.slice(0, length)
      const file = join(dir, `${hash}.txt`)

      if (!existsSync(file)) {
        writeFileSync(file, text, { mode: 0o600 })
        prunePasteStore(env)

        return hash
      }

      if (readFileSync(file, 'utf8') === text) {
        // Re-pasting the same content keeps it young so pruning evicts the
        // snippets nobody has referenced instead of the hot one.
        touch(file)

        return hash
      }
    }

    return null
  } catch {
    return null
  }
}

export function readPasteSnippet(hash: string, env: NodeJS.ProcessEnv = process.env): null | string {
  // The hash reaches this from prompt text, so it is untrusted input on a path
  // join: anything but a bare digest is refused rather than escaped.
  if (!HASH_RE.test(hash)) {
    return null
  }

  try {
    const file = snippetPath(hash, env)

    return existsSync(file) ? readFileSync(file, 'utf8') : null
  } catch {
    return null
  }
}

/** Stamp `hash` inside the token's closing brackets so it survives copy/recall. */
export function tagPasteToken(label: string, hash: string): string {
  if (!label.endsWith(']]')) {
    return label
  }

  return `${label.replace(/\s*\]\]$/, '')} #${hash} ]]`
}

export const pasteTokenHash = (token: string): null | string => TOKEN_HASH_RE.exec(token)?.[1] ?? null

export interface PasteStoreLimits {
  maxBytes?: number
  maxFiles?: number
}

/** Bound the spill directory by count and bytes, evicting least-recently-used. */
export function prunePasteStore(
  env: NodeJS.ProcessEnv = process.env,
  { maxBytes = MAX_TOTAL_BYTES, maxFiles = MAX_FILES }: PasteStoreLimits = {}
): void {
  try {
    const dir = pasteDirectory(env)

    if (!existsSync(dir)) {
      return
    }

    const files = readdirSync(dir)
      .filter(name => name.endsWith('.txt'))
      .flatMap(name => {
        try {
          const stats = statSync(join(dir, name))

          return [{ mtime: stats.mtimeMs, path: join(dir, name), size: stats.size }]
        } catch {
          return []
        }
      })
      .sort((a, b) => b.mtime - a.mtime)

    let total = 0

    for (const [index, file] of files.entries()) {
      total += file.size

      if (index < maxFiles && total <= maxBytes) {
        continue
      }

      try {
        unlinkSync(file.path)
      } catch {
        void 0
      }
    }
  } catch {
    void 0
  }
}

function touch(file: string) {
  try {
    const now = new Date()

    utimesSync(file, now, now)
  } catch {
    void 0
  }
}
