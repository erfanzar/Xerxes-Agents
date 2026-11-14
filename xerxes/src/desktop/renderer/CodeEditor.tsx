// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The file editor (Claude Code's file pane): the file is always editable in
 * place — no separate edit mode. A transparent textarea lies exactly over a
 * highlighted copy of the same text, laid out identically (same font, same
 * wrapping), so the caret, selection and undo are the browser's own while
 * the colours come from the copy underneath. ⌘S saves.
 */

import { memo, useDeferredValue, useMemo, type ReactElement } from 'react'
import { highlightLines, type LineSpan } from './codeHighlight.js'
import { grammarFor, syntaxFor, tokenize } from './syntax.js'

/** One row: lowlight's spans when the file's grammar is known, else the small line tokenizer. */
const Line = memo(function Line({ number, text, path, spans }: { number: number; text: string; path: string; spans: readonly LineSpan[] | undefined }): ReactElement {
  const tokens = useMemo(() => spans ? [] : tokenize(text, syntaxFor(path)), [spans, text, path])
  const body = spans
    ? spans.map((span, index) => span.className ? <span key={index} className={span.className}>{span.text}</span> : span.text)
    : tokens.map((token, index) => token.kind === 'plain' ? token.text : <span key={index} className={`tok-${token.kind}`}>{token.text}</span>)
  return <div className="code-editor__row">
    <span className="code-editor__num" aria-hidden="true">{number}</span>
    <code>{body.length ? body : ' '}</code>
  </div>
})

/** Indentation the file already uses: a tab, or its smallest space step. */
export function indentUnit(text: string, path: string): string {
  if (/^\t/m.test(text)) return '\t'
  const steps = [...text.matchAll(/^( +)\S/gm)].map(match => match[1]!.length)
  const smallest = steps.length ? Math.min(...steps) : 0
  return ' '.repeat(smallest === 2 || smallest === 4 ? smallest : /\.py$/.test(path) ? 4 : 2)
}

export function CodeEditor({ path, value, onChange, onSave, readOnly = false }: {
  path: string
  value: string
  onChange: (value: string) => void
  onSave: () => void
  readOnly?: boolean
}): ReactElement {
  // The coloured copy may trail a fast typist by a frame; the text itself never does.
  const painted = useDeferredValue(value)
  const lines = useMemo(() => painted.split('\n'), [painted])
  // Whole-file grammar, so a docstring or block comment keeps its colour on every line it covers.
  const highlighted = useMemo(() => highlightLines(painted, grammarFor(path)), [painted, path])
  const indent = useMemo(() => indentUnit(value, path), [value, path])
  return <div className="code-editor">
    <div className="code-editor__body">
      <div className="code-editor__paint" aria-hidden="true">
        {lines.map((line, index) => <Line key={index} number={index + 1} text={line} path={path} spans={highlighted?.[index]} />)}
      </div>
      <textarea
        className="code-editor__input"
        aria-label={`Contents of ${path}`}
        value={value}
        readOnly={readOnly}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        onChange={event => onChange(event.target.value)}
        onKeyDown={event => {
          if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (!readOnly) onSave(); return }
          if (event.key === 'Tab' && !event.shiftKey && !readOnly) {
            // Insert through the editing command so undo still works.
            event.preventDefault()
            document.execCommand('insertText', false, indent)
          }
        }}
      />
    </div>
  </div>
}
