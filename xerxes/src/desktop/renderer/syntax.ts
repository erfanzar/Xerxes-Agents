// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A small line-by-line highlighter for the file editor: comments, strings,
 * numbers and keywords, by file extension. It is deliberately shallow — no
 * grammar, no state across lines (a block comment colours only the lines
 * that open or close it) — so it stays instant on every keystroke.
 */

export type TokenKind = 'comment' | 'string' | 'number' | 'keyword' | 'plain'
export interface Token { readonly kind: TokenKind; readonly text: string }

interface Syntax { readonly comment: readonly string[]; readonly keywords: ReadonlySet<string> }

const words = (text: string): ReadonlySet<string> => new Set(text.split(/\s+/).filter(Boolean))
const C_LIKE = words('if else for while do switch case break continue return function const let var class extends implements interface type enum import export from as default new this super try catch finally throw async await yield of in instanceof typeof void null undefined true false static public private protected readonly abstract declare namespace module keyof satisfies package func go defer struct map chan select fn impl trait mut pub use crate match where loop self Self None Some Ok Err int long float double char bool boolean string')
const PYTHON = words('def class if elif else for while return import from as with try except finally raise pass break continue lambda yield async await in is not and or None True False self global nonlocal assert del')
const SHELL = words('if then else elif fi for while do done case esac function in return export local set unset echo exit')
const LANGUAGES: Record<string, Syntax> = {
  c: { comment: ['//'], keywords: C_LIKE },
  hash: { comment: ['#'], keywords: PYTHON },
  shell: { comment: ['#'], keywords: SHELL },
  data: { comment: ['#'], keywords: words('true false null') },
  none: { comment: [], keywords: new Set() },
}
const EXTENSIONS: Record<string, keyof typeof LANGUAGES> = {
  ts: 'c', tsx: 'c', js: 'c', jsx: 'c', mjs: 'c', cjs: 'c', go: 'c', rs: 'c', c: 'c', h: 'c', cc: 'c', cpp: 'c', hpp: 'c', java: 'c', kt: 'c', swift: 'c', cs: 'c', css: 'c', scss: 'c', json: 'c', jsonc: 'c',
  py: 'hash', pyi: 'hash', rb: 'hash',
  sh: 'shell', bash: 'shell', zsh: 'shell',
  toml: 'data', yaml: 'data', yml: 'data', ini: 'data', cfg: 'data', conf: 'data', env: 'data',
}

export function syntaxFor(path: string): Syntax {
  const name = path.split('/').pop() ?? ''
  if (/^(Dockerfile|Makefile|\.gitignore|\.dockerignore)$/.test(name)) return LANGUAGES.shell!
  const extension = name.includes('.') ? name.split('.').pop()!.toLowerCase() : ''
  return LANGUAGES[EXTENSIONS[extension] ?? 'none']!
}

/** Tokens of one line; their texts always concatenate back to the line. */
export function tokenize(line: string, syntax: Syntax): Token[] {
  const out: Token[] = []
  let plain = ''
  const flush = () => { if (plain) { out.push({ kind: 'plain', text: plain }); plain = '' } }
  const push = (kind: TokenKind, text: string) => { flush(); out.push({ kind, text }) }
  let index = 0
  while (index < line.length) {
    const rest = line.slice(index)
    if (syntax.comment.some(marker => rest.startsWith(marker)) || rest.startsWith('/*') && syntax.comment.includes('//')) { push('comment', rest); break }
    const quote = rest[0]
    if (quote === '"' || quote === "'" || quote === '`') {
      let end = 1
      while (end < rest.length && rest[end] !== quote) end += rest[end] === '\\' ? 2 : 1
      push('string', rest.slice(0, Math.min(end + 1, rest.length)))
      index += Math.min(end + 1, rest.length)
      continue
    }
    const word = /^[A-Za-z_$][\w$]*/.exec(rest)
    if (word) {
      if (syntax.keywords.has(word[0])) push('keyword', word[0]); else plain += word[0]
      index += word[0].length
      continue
    }
    const number = /^(0x[\da-fA-F_]+|\d[\d_]*(\.\d+)?([eE][+-]?\d+)?)/.exec(rest)
    if (number && !/[\w$]$/.test(plain)) { push('number', number[0]); index += number[0].length; continue }
    plain += rest[0]
    index += 1
  }
  flush()
  return out
}
