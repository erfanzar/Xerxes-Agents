// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The file editor (Claude Code's file pane): the file is always editable in
 * place — no separate edit mode. A transparent textarea lies exactly over a
 * highlighted copy of the same text, laid out identically (same font, same
 * wrapping), so the caret, selection and undo are the browser's own while
 * the colours come from the copy underneath. ⌘S saves. With a completer
 * (Python files), typing opens a completion popup at the caret.
 */

import { memo, useDeferredValue, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { highlightLines, type LineSpan } from './codeHighlight.js'
import { completionEdit, completionKindTag, completionPrefix, cursorPosition, filterCompletions, newlineIndent, shouldComplete, type Completer, type CompletionItemView } from './codeCompletion.js'
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

interface Popup { readonly items: readonly CompletionItemView[]; readonly top: number; readonly left: number }

/** Where the caret's line sits inside the editor body, for a monospace textarea. */
function caretPoint(element: HTMLTextAreaElement, offset: number): { top: number; left: number } {
  const style = getComputedStyle(element)
  const lineHeight = Number.parseFloat(style.lineHeight) || 20
  let width = 7.5
  try {
    const context = document.createElement('canvas').getContext('2d')
    if (context) { context.font = style.font || `${style.fontSize} ${style.fontFamily}`; width = context.measureText('M').width || width }
  } catch { /* Measurement is best effort; the default width keeps the popup near the caret. */ }
  const { line, character } = cursorPosition(element.value, offset)
  return { top: element.offsetTop + (line + 1) * lineHeight + 2, left: (Number.parseFloat(style.paddingLeft) || 0) + character * width }
}

export function CodeEditor({ path, value, onChange, onSave, readOnly = false, complete, onFormat }: {
  /** Format the whole file (Shift+Option+F); omitted means no formatter for this file. */
  onFormat?: () => void
  path: string
  value: string
  onChange: (value: string) => void
  onSave: () => void
  readOnly?: boolean
  /** Completion source for this file; omitted means no popup. */
  complete?: Completer
}): ReactElement {
  // The coloured copy may trail a fast typist by a frame; the text itself never does.
  const painted = useDeferredValue(value)
  const lines = useMemo(() => painted.split('\n'), [painted])
  // Whole-file grammar, so a docstring or block comment keeps its colour on every line it covers.
  const highlighted = useMemo(() => highlightLines(painted, grammarFor(path)), [painted, path])
  const indent = useMemo(() => indentUnit(value, path), [value, path])
  const input = useRef<HTMLTextAreaElement>(null)
  const request = useRef(0)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const inserting = useRef(false)
  const [popup, setPopup] = useState<Popup | null>(null)
  const [caret, setCaret] = useState(0)
  const [active, setActive] = useState(0)
  useEffect(() => () => clearTimeout(timer.current), [])
  useEffect(() => { request.current++; setPopup(null) }, [path])
  const shown = useMemo(() => popup ? filterCompletions(popup.items, completionPrefix(value, caret)).slice(0, 50) : [], [popup, value, caret])
  const open = shown.length > 0
  const close = (): void => { request.current++; clearTimeout(timer.current); setPopup(null) }
  const schedule = (text: string, offset: number): void => {
    clearTimeout(timer.current)
    if (!complete || readOnly || !shouldComplete(text, offset)) { if (popup) close(); return }
    const id = ++request.current
    timer.current = setTimeout(() => {
      const { line, character } = cursorPosition(text, offset)
      complete(text, line, character).then(items => {
        if (request.current !== id || !input.current) return
        setActive(0)
        setPopup(items.length ? { items, ...caretPoint(input.current, offset) } : null)
      }, () => { if (request.current === id) setPopup(null) })
    }, 120)
  }
  const accept = (item: CompletionItemView): void => {
    const element = input.current
    if (!element) return
    const edit = completionEdit(element.value, element.selectionStart, item)
    close()
    element.setSelectionRange(edit.start, edit.end)
    // Insert through the editing command so undo still works; the resulting change must not reopen the popup.
    inserting.current = true
    try { document.execCommand('insertText', false, edit.insert) } finally { inserting.current = false }
  }
  const chosen = shown[Math.min(active, shown.length - 1)]
  return <div className="code-editor">
    <div className="code-editor__body">
      <div className="code-editor__paint" aria-hidden="true">
        {lines.map((line, index) => <Line key={index} number={index + 1} text={line} path={path} spans={highlighted?.[index]} />)}
      </div>
      <textarea
        ref={input}
        className="code-editor__input"
        aria-label={`Contents of ${path}`}
        aria-autocomplete={complete ? 'list' : undefined}
        aria-expanded={complete ? open : undefined}
        value={value}
        readOnly={readOnly}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        onChange={event => {
          const text = event.target.value, offset = event.target.selectionStart
          onChange(text)
          setCaret(offset)
          if (!inserting.current) schedule(text, offset)
        }}
        onSelect={event => {
          const offset = event.currentTarget.selectionStart
          setCaret(offset)
          // Moving the caret off the word being completed closes the popup.
          if (popup && (event.currentTarget.selectionEnd !== offset || !completionPrefix(event.currentTarget.value, offset) && event.currentTarget.value[offset - 1] !== '.' && event.currentTarget.value[offset - 1] !== ' ')) close()
        }}
        onBlur={() => close()}
        onKeyDown={event => {
          if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (!readOnly) onSave(); return }
          if (open && chosen) {
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault()
              setActive(index => (Math.min(index, shown.length - 1) + (event.key === 'ArrowDown' ? 1 : shown.length - 1)) % shown.length)
              return
            }
            if ((event.key === 'Enter' || event.key === 'Tab') && !event.shiftKey) { event.preventDefault(); accept(chosen); return }
            if (event.key === 'Escape') { event.preventDefault(); close(); return }
          }
          if (onFormat && !readOnly && event.altKey && event.shiftKey && event.code === 'KeyF') {
            // Shift+Option+F formats the file (Ruff for Python).
            event.preventDefault()
            onFormat()
            return
          }
          if (event.key === 'Enter' && !event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey && !readOnly) {
            // Keep the line's indentation (one level deeper after `:`), inserted so undo still works.
            event.preventDefault()
            const element = event.currentTarget
            document.execCommand('insertText', false, '\n' + newlineIndent(element.value, element.selectionStart, indent))
            return
          }
          if (complete && !readOnly && event.code === 'Space' && (event.ctrlKey || event.altKey)) {
            // Option+Space (or Ctrl+Space) asks explicitly, wherever the caret is.
            event.preventDefault()
            const element = event.currentTarget, offset = element.selectionStart, id = ++request.current
            const { line, character } = cursorPosition(element.value, offset)
            complete(element.value, line, character).then(items => {
              if (request.current === id && input.current) { setActive(0); setPopup(items.length ? { items, ...caretPoint(input.current, offset) } : null) }
            }, () => {})
            return
          }
          if (event.key === 'Tab' && !event.shiftKey && !readOnly) {
            // Insert through the editing command so undo still works.
            event.preventDefault()
            document.execCommand('insertText', false, indent)
          }
        }}
      />
      {open && popup && <div className="code-completion" role="listbox" aria-label="Completions" style={{ top: popup.top, left: popup.left }}>
        <ul>
          {shown.map((item, index) => <li key={item.label + index} role="option" aria-selected={item === chosen}
            className={item === chosen ? 'is-active' : undefined}
            onMouseDown={event => { event.preventDefault(); accept(item) }}
            onMouseEnter={() => setActive(index)}>
            <span className="code-completion__kind" aria-hidden="true">{completionKindTag(item.kind)}</span>
            <span className="code-completion__label">{item.label}</span>
            {item.detail && <span className="code-completion__detail">{item.detail}</span>}
          </li>)}
        </ul>
        {chosen?.documentation && <p className="code-completion__doc">{chosen.documentation}</p>}
      </div>}
    </div>
  </div>
}
