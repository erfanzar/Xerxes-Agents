// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState } from 'react'

import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { isUsableClipboardText, readClipboardText } from '../lib/clipboard.js'
import { resolveEditor } from '../lib/editor.js'
import { readOsc52Clipboard } from '../lib/osc52.js'
import { useStdin, withTerminalSuspended } from '../lib/terminalRuntime.opentui.js'
import { isRemoteShellSession } from '../lib/terminalSetup.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'
import type { PasteEvent } from '../protocol/paste.js'

import type { MaybePromise, PasteSnippet, UseComposerStateOptions, UseComposerStateResult } from './interfaces.js'
import { $isBlocked } from './overlayStore.js'
import { getUiState } from './uiStore.js'

const PASTE_SNIP_MAX_COUNT = 32
const PASTE_SNIP_MAX_TOTAL_BYTES = 4 * 1024 * 1024

const trimSnips = (snips: PasteSnippet[]): PasteSnippet[] => {
  let total = 0
  const out: PasteSnippet[] = []

  for (let i = snips.length - 1; i >= 0; i--) {
    const snip = snips[i]!
    const size = snip.text.length

    if (out.length >= PASTE_SNIP_MAX_COUNT || total + size > PASTE_SNIP_MAX_TOTAL_BYTES) {
      break
    }

    total += size
    out.unshift(snip)
  }

  return out.length === snips.length ? snips : out
}

/** Insert text at the cursor position, adding spacing to separate from adjacent non-whitespace. */
function insertAtCursor(value: string, cursor: number, text: string): { cursor: number; value: string } {
  const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
  const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
  const insert = `${lead}${text}${tail}`

  return {
    cursor: cursor + insert.length,
    value: value.slice(0, cursor) + insert + value.slice(cursor)
  }
}

/**
 * Quick client-side heuristic to detect text that looks like a dropped file path.
 * Native Bun currently treats dropped paths as ordinary prompt text. Keep the
 * narrow helper available for a future native attachment port without sending
 * the retired `input.detect_drop` compatibility RPC.
 */
export function looksLikeDroppedPath(text: string): boolean {
  const trimmed = text.trim()

  if (!trimmed || trimmed.includes('\n')) {
    return false
  }

  // file:// URIs, relative, home-relative, quoted, and Windows drive paths
  if (
    trimmed.startsWith('file://') ||
    trimmed.startsWith('~/') ||
    trimmed.startsWith('./') ||
    trimmed.startsWith('../') ||
    trimmed.startsWith('"/') ||
    trimmed.startsWith("'/") ||
    trimmed.startsWith('"~') ||
    trimmed.startsWith("'~") ||
    /^[A-Za-z]:[/\\]/.test(trimmed) ||
    /^["'][A-Za-z]:[/\\]/.test(trimmed)
  ) {
    return true
  }

  // Bare absolute paths (start with /) — require a second '/' or a '.' to avoid
  // false positives on short strings like "/api" or "/help" which would trigger
  // unnecessary RPC round-trips.
  if (trimmed.startsWith('/')) {
    const rest = trimmed.slice(1)

    return rest.includes('/') || rest.includes('.')
  }

  return false
}

export function useComposerState({
  catalog,
  gw,
  onClipboardPaste,
  onDroppedPath,
  submitRef
}: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [pasteSnips, setPasteSnips] = useState<PasteSnippet[]>([])
  const isBlocked = useStore($isBlocked)
  const { querier } = useStdin() as { querier: Parameters<typeof readOsc52Clipboard>[0] }

  const {
    queueRef,
    queueEditRef,
    queuedDisplay,
    queueEditIdx,
    enqueue,
    dequeue,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue
  } = useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace, dismissCompletions } = useCompletion(
    input,
    isBlocked,
    gw,
    catalog
  )

  const clearIn = useCallback(() => {
    setInput('')
    setInputBuf([])
    setPasteSnips([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setQueueEdit, setHistoryIdx])

  const handleResolvedPaste = useCallback(
    async ({
      bracketed,
      cursor,
      text,
      value
    }: Omit<PasteEvent, 'hotkey'>): Promise<null | { cursor: number; value: string }> => {
      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        if (bracketed) {
          void onClipboardPaste(true)
        }

        return null
      }

      if (looksLikeDroppedPath(cleanedText)) {
        onDroppedPath?.(cleanedText)
      }

      const lineCount = cleanedText.split('\n').length
      const pasteCollapseLines = getUiState().pasteCollapseLines
      const pasteCollapseChars = getUiState().pasteCollapseChars
      const linesHit = pasteCollapseLines > 0 && lineCount >= pasteCollapseLines
      const charsHit = pasteCollapseChars > 0 && cleanedText.length >= pasteCollapseChars

      if (!linesHit && !charsHit) {
        return {
          cursor: cursor + cleanedText.length,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const inserted = insertAtCursor(value, cursor, label)

      setPasteSnips(prev => trimSnips([...prev, { label, text: cleanedText }]))

      return inserted
    },
    [onClipboardPaste, onDroppedPath]
  )

  const handleTextPaste = useCallback(
    ({
      bracketed,
      cursor,
      hotkey,
      text,
      value
    }: PasteEvent): MaybePromise<null | { cursor: number; value: string }> => {
      if (hotkey) {
        const preferOsc52 = isRemoteShellSession(process.env)

        const readPreferredText = preferOsc52
          ? readOsc52Clipboard(querier).then(async osc52Text => {
              if (isUsableClipboardText(osc52Text)) {
                return osc52Text
              }

              return readClipboardText()
            })
          : readClipboardText().then(async clipText => {
              if (isUsableClipboardText(clipText)) {
                return clipText
              }

              return readOsc52Clipboard(querier)
            })

        return readPreferredText.then(async preferredText => {
          if (isUsableClipboardText(preferredText)) {
            return handleResolvedPaste({ bracketed: false, cursor, text: preferredText, value })
          }

          void onClipboardPaste(false)

          return null
        })
      }

      return handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value })
    },
    [handleResolvedPaste, onClipboardPaste, querier]
  )

  const openEditor = useCallback(async () => {
    const dir = mkdtempSync(join(tmpdir(), 'xerxes-'))
    const file = join(dir, 'prompt.md')
    const [cmd, ...args] = resolveEditor()

    writeFileSync(file, [...inputBuf, input].join('\n'))

    let exitCode: null | number = null

    await withTerminalSuspended(async () => {
      exitCode = spawnSync(cmd!, [...args, file], { stdio: 'inherit' }).status
    })

    try {
      if (exitCode !== 0) {
        return
      }

      const text = readFileSync(file, 'utf8').trimEnd()

      if (!text) {
        return
      }

      setInput('')
      setInputBuf([])
      submitRef.current(text)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  }, [input, inputBuf, submitRef])

  const actions = useMemo(
    () => ({
      clearIn,
      dequeue,
      dismissCompletions,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      removeQueue: removeQ,
      replaceQueue: replaceQ,
      setCompIdx,
      setHistoryIdx,
      setInput,
      setInputBuf,
      setPasteSnips,
      setQueueEdit,
      syncQueue
    }),
    [
      clearIn,
      dequeue,
      dismissCompletions,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      removeQ,
      replaceQ,
      setCompIdx,
      setHistoryIdx,
      setQueueEdit,
      syncQueue
    ]
  )

  const refs = useMemo(
    () => ({
      historyDraftRef,
      historyRef,
      queueEditRef,
      queueRef,
      submitRef
    }),
    [historyDraftRef, historyRef, queueEditRef, queueRef, submitRef]
  )

  const state = useMemo(
    () => ({
      compIdx,
      compReplace,
      completions,
      historyIdx,
      input,
      inputBuf,
      pasteSnips,
      queueEditIdx,
      queuedDisplay
    }),
    [compIdx, compReplace, completions, historyIdx, input, inputBuf, pasteSnips, queueEditIdx, queuedDisplay]
  )

  return {
    actions,
    refs,
    state
  }
}
