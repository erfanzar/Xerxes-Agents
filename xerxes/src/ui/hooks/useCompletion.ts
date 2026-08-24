// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useCallback, useEffect, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import { looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { CompletionResponse } from '../gatewayTypes.js'
import { rankCompletionItems } from '../lib/completion.js'
import { asRpcResult } from '../lib/rpc.js'
import type { SlashCatalog } from '../types.js'

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

const commandToken = (input: string) => input.slice(1).split(/\s+/, 1)[0]?.toLowerCase() ?? ''

const completionKey = (item: CompletionItem) => (item.text || item.display).replace(/^\/+/, '').toLowerCase()

export function slashCompletionsFromCatalog(input: string, catalog: null | SlashCatalog): CompletionItem[] {
  if (!catalog || !looksLikeSlashCommand(input) || /\s/.test(input.slice(1))) {
    return []
  }

  const prefix = commandToken(input)

  // Category per command name, so the menu can rank by group. Skills are no
  // longer hoisted to the front here — `rankCompletionItems` owns ordering,
  // and a hoist would only fight it.
  const groupOf = new Map<string, string>()
  const skillPairs = catalog.categories.find(category => category.name === 'project skills')?.pairs ?? []

  for (const category of catalog.categories) {
    const group = category.name === 'project skills' ? 'skills' : category.name

    for (const [name] of category.pairs) {
      groupOf.set(name, group)
    }
  }

  const seen = new Set(catalog.pairs.map(([name]) => name))
  const pairs = [...catalog.pairs, ...skillPairs.filter(([name]) => !seen.has(name))]

  return pairs
    .filter(([name, meta]) => {
      // Local candidates follow the same ladder as rankCompletionItems:
      // name prefix → name substring → description ("skill body") hit. The
      // ranker owns ordering; this filter only decides what reaches it.
      const bare = name.replace(/^\/+/, '').toLowerCase()
      if (!prefix || bare.startsWith(prefix)) {
        return true
      }
      return bare.includes(prefix) || meta.toLowerCase().includes(prefix)
    })
    .map(([name, meta]) => {
      const group = groupOf.get(name)

      return {
        display: name.replace(/^\/+/, ''),
        ...(group ? { group } : {}),
        meta,
        text: name.startsWith('/') ? name : `/${name}`
      }
    })
}

export function mergeCompletionItems(primary: CompletionItem[], secondary: CompletionItem[]): CompletionItem[] {
  const seen = new Set<string>()
  const merged: CompletionItem[] = []

  for (const item of [...primary, ...secondary]) {
    const key = completionKey(item)

    if (!key || seen.has(key)) {
      continue
    }

    seen.add(key)
    merged.push(item)
  }

  return merged
}

export function completionRequestForInput(
  input: string
):
  | { method: 'complete.path'; params: { word: string }; replaceFrom: number }
  | { method: 'complete.slash'; params: { text: string }; replaceFrom: number }
  | null {
  const isSlashCommand = looksLikeSlashCommand(input)
  const isSlashName = isSlashCommand && !/\s/.test(input.slice(1))
  const pathWord = isSlashName ? null : (input.match(TAB_PATH_RE)?.[1] ?? null)

  if (!isSlashName && !pathWord) {
    return null
  }

  // Match Grok's file picker: a mention starts searching after the first
  // query character, so a lone `@` never opens an unranked project dump.
  if (pathWord === '@') {
    return null
  }

  // `/model` uses the two-step ModelPicker (real curated IDs).
  // Slash completion here only showed short aliases + vendor/family meta.
  if (isSlashCommand && /^\/model(?:\s|$)/.test(input)) {
    return null
  }

  if (isSlashName) {
    return { method: 'complete.slash', params: { text: input }, replaceFrom: 1 }
  }

  return {
    method: 'complete.path',
    params: { word: pathWord! },
    replaceFrom: input.length - pathWord!.length
  }
}

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient, catalog: null | SlashCatalog = null) {
  const [completions, setCompletions] = useState<CompletionItem[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')
  const catalogRef = useRef<SlashCatalog | null>(null)
  const dismissedInputRef = useRef<null | string>(null)

  const dismissCompletions = useCallback(() => {
    dismissedInputRef.current = input
    setCompletions([])
    setCompIdx(0)
    setCompReplace(0)
  }, [input])

  useEffect(() => {
    const clear = () => {
      setCompletions(prev => (prev.length ? [] : prev))
      setCompIdx(prev => (prev ? 0 : prev))
      setCompReplace(prev => (prev ? 0 : prev))
    }

    if (blocked) {
      ref.current = ''
      clear()

      return
    }

    const catalogChanged = catalog !== catalogRef.current

    if (input === ref.current && !catalogChanged) {
      return
    }

    ref.current = input
    catalogRef.current = catalog

    if (dismissedInputRef.current !== null && dismissedInputRef.current !== input) {
      dismissedInputRef.current = null
    }

    if (dismissedInputRef.current === input) {
      clear()

      return
    }

    const request = completionRequestForInput(input)
    if (!request) {
      clear()

      return
    }

    const initialLocal = request.method === 'complete.slash' ? slashCompletionsFromCatalog(input, catalog) : []

    if (initialLocal.length) {
      setCompletions(rankCompletionItems(initialLocal, commandToken(input)))
      setCompIdx(0)
      setCompReplace(request.replaceFrom)
    } else {
      // A request for a different path/slash token must not leave the prior
      // token's rows actionable during the debounce/network window. Without
      // this clear, a quick Tab or Enter could apply a stale completion to the
      // new draft before the matching daemon response arrived.
      clear()
    }

    const t = setTimeout(() => {
      if (ref.current !== input) {
        return
      }

      gw.request<CompletionResponse>(request.method, request.params)
        .then(raw => {
          if (ref.current !== input) {
            return
          }

          const r = asRpcResult<CompletionResponse>(raw)

          const remote = r?.items ?? []
          const local = request.method === 'complete.slash' ? slashCompletionsFromCatalog(input, catalog) : []

          setCompletions(
            request.method === 'complete.slash'
              ? rankCompletionItems(mergeCompletionItems(local, remote), commandToken(input))
              : remote
          )
          setCompIdx(0)
          setCompReplace(request.method === 'complete.slash' ? (r?.replace_from ?? 1) : request.replaceFrom)
        })
        .catch((e: unknown) => {
          if (ref.current !== input) {
            return
          }

          const local = request.method === 'complete.slash' ? slashCompletionsFromCatalog(input, catalog) : []

          if (local.length) {
            setCompletions(rankCompletionItems(local, commandToken(input)))
            setCompIdx(0)
            setCompReplace(request.replaceFrom)
            return
          }

          setCompletions([
            {
              text: '',
              display: 'completion unavailable',
              meta: e instanceof Error && e.message ? e.message : 'unavailable'
            }
          ])
          setCompIdx(0)
          setCompReplace(request.replaceFrom)
        })
    }, 60)

    return () => clearTimeout(t)
  }, [blocked, catalog, gw, input])

  return { completions, compIdx, setCompIdx, compReplace, dismissCompletions }
}
