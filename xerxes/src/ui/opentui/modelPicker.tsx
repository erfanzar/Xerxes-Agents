// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme } from '../app/uiStore.js'
import { providerDisplayNames } from '../domain/providers.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { ModelModelsResponse, ModelOptionProvider, ModelOptionsResponse } from '../gatewayTypes.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

import { GLYPH } from '../domain/nocturne.js'
import { Box, Span, Text } from './primitives.js'
import { windowItems } from './overlayLayout.js'
import { InfoRow, ModalShell } from './pickerChrome.js'

/**
 * Row cap for the picker's list panes.
 *
 * This used to be 12 regardless of the terminal, so a provider with 14 models
 * on a 90-row screen showed three or four of them under a mostly empty panel.
 * The cap now only exists to stop the modal swallowing a very tall terminal;
 * the real budget is the height.
 */
const MAX_VISIBLE = 40
const MIN_PANEL_WIDTH = 40
// Match the other primary overlays: wide enough for two readable panes, but
// still bounded so the transcript remains recognizable behind the scrim.
const MAX_PANEL_WIDTH = 110
/** At this width the picker switches to mockup 09's side-by-side layout. */
const TWO_PANE_MIN_WIDTH = 84

type Stage = 'provider' | 'model'

interface ProviderRow {
  name: string
  provider: ModelOptionProvider
}

interface ModelDiscovery {
  error?: string
  models: string[]
  requestId: number
  source?: string
  status: 'error' | 'loading' | 'partial' | 'ready'
  warning?: string
}

interface ModelRow {
  custom: boolean
  model: string
}

export interface ModelPickerProps {
  allowPersistGlobal?: boolean
  onCancel?: () => void
  onSelect: (value: string) => void
  sessionId?: null | string
  t?: Theme
}

const consume = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

const uniqueModels = (values: readonly (null | string | undefined)[]) => [
  ...new Set(values.map(value => value?.trim()).filter((value): value is string => Boolean(value)))
]

/** Mockup 09 groups models by family (`opus`, `sonnet`, …) derived from the ID. */
/**
 * The group a model belongs to.
 *
 * Vendor-prefixed ids (`meta/llama-3.3-70b`, `deepseek/deepseek-v4`) group by
 * VENDOR — that is the only grouping that means anything across a 400-model
 * OpenRouter catalogue. Bare ids fall back to stripping a trailing version
 * segment, which is what turns `k3` and `k3-256k` into one family.
 *
 * This used to strip the version segment only, so `kimi-for-coding` and
 * `meta/muse-spark-1.2-contributor` were each their own family of one — and
 * every one of them printed a caption naming the single row underneath it.
 */
const familyOf = (model: string): string => {
  // A leading `~` marks the profile's configured model; it is a badge on the
  // row, not part of its vendor, and leaving it in gave that one model a
  // family of its own next to the vendor it actually belongs to.
  const id = model.startsWith('~') ? model.slice(1) : model
  const slash = id.indexOf('/')

  if (slash > 0) {
    return id.slice(0, slash)
  }

  const parts = id.split('-')

  return parts.length > 1 && /^\d/.test(parts[parts.length - 1] ?? '')
    ? parts.slice(0, -1).join('-')
    : id
}

/**
 * Sort rows so each family is contiguous.
 *
 * The captions are emitted whenever the family CHANGES, so an unsorted list
 * produces a caption almost every row: a 422-model OpenRouter catalogue came
 * out with `qwen · 51` printed twice over two rows, `deepseek · 14` three
 * times, and a `~z-ai/…` model sitting under `tencent`. The counts were
 * right and the groups were fiction.
 *
 * Families keep the order they FIRST appear in, and members keep their order
 * within a family, so whatever ranking the provider returned still decides
 * what you see first — it is grouped, not re-sorted.
 */
const groupByFamily = (rows: readonly ModelRow[]): ModelRow[] => {
  const families = new Map<string, ModelRow[]>()

  for (const row of rows) {
    const key = row.model ? familyOf(row.model) : ''
    const bucket = families.get(key)

    if (bucket) {
      bucket.push(row)
    } else {
      families.set(key, [row])
    }
  }

  return [...families.values()].flat()
}

/** Window entries with family captions interleaved, per the mockup's mgroups. */
const familyDisplayRows = (
  rows: readonly ModelRow[]
): Array<{ caption?: string; row?: ModelRow }> => {
  const counts = new Map<string, number>()

  for (const row of rows) {
    if (row.model) counts.set(familyOf(row.model), (counts.get(familyOf(row.model)) ?? 0) + 1)
  }

  const display: Array<{ caption?: string; row?: ModelRow }> = []
  let previousFamily: null | string = null

  for (const row of rows) {
    if (!row.model) {
      display.push({ row })
      continue
    }

    const family = familyOf(row.model)
    const size = counts.get(family) ?? 1

    // A caption for a family of ONE is the model's own name printed twice,
    // once with `· 1` under it. It cost a row per model, which is why a
    // provider with four models filled the pane with two of them.
    if (family !== previousFamily && size > 1) {
      display.push({ caption: `${family} · ${size}` })
      previousFamily = family
    }

    display.push({ row })
  }

  return display
}

export function ModelPicker({
  allowPersistGlobal = false,
  onCancel,
  onSelect,
  sessionId,
  t: suppliedTheme
}: ModelPickerProps) {
  const { gw } = useGateway()
  const storeSessionId = useStore($uiSessionId)
  const storeTheme = useStore($uiTheme)
  const { height, width } = useTerminalDimensions()
  const t = suppliedTheme ?? storeTheme
  const effectiveSessionId = sessionId === undefined ? storeSessionId : sessionId

  // Mockup 09: on a wide terminal both stages sit side by side. Narrow
  // terminals keep the sequential wizard.
  const twoPane = width >= TWO_PANE_MIN_WIDTH

  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [discoveryVersion, setDiscoveryVersion] = useState(0)
  const [filter, setFilter] = useState('')
  const [modelIdx, setModelIdx] = useState(0)
  const [modelProviderSlug, setModelProviderSlug] = useState<null | string>(null)
  const [optionsError, setOptionsError] = useState('')
  const [optionsLoading, setOptionsLoading] = useState(true)
  const [optionsRequest, setOptionsRequest] = useState(0)
  const [persistGlobal, setPersistGlobal] = useState(false)
  const [providerIdx, setProviderIdx] = useState(0)
  const [stage, setStage] = useState<Stage>('provider')
  const discoveries = useRef(new Map<string, ModelDiscovery>())
  const nextDiscoveryRequest = useRef(0)

  useEffect(
    () => () => {
      nextDiscoveryRequest.current += 1
      discoveries.current.clear()
    },
    []
  )

  const close = useCallback(() => {
    patchOverlayState({ modelPicker: false })
    onCancel?.()
  }, [onCancel])

  const select = useCallback(
    (value: string) => {
      patchOverlayState({ modelPicker: false })
      onSelect(value)
    },
    [onSelect]
  )

  useEffect(() => {
    let active = true

    setOptionsLoading(true)
    setOptionsError('')

    gw.request<ModelOptionsResponse>('model.options', effectiveSessionId ? { session_id: effectiveSessionId } : {})
      .then(raw => {
        if (!active) {
          return
        }

        const result = asRpcResult<ModelOptionsResponse>(raw)

        if (!result) {
          setOptionsError('invalid response: model.options')
          setOptionsLoading(false)

          return
        }

        const next = result.providers ?? []
        discoveries.current.clear()
        setDiscoveryVersion(version => version + 1)
        setProviders(next)
        setCurrentModel(String(result.model ?? ''))
        setProviderIdx(
          Math.max(
            0,
            next.findIndex(provider => provider.is_current)
          )
        )
        setModelIdx(0)
        setModelProviderSlug(null)
        setStage('provider')
        setFilter('')
        setOptionsLoading(false)
      })
      .catch((reason: unknown) => {
        if (!active) {
          return
        }

        setOptionsError(rpcErrorMessage(reason))
        setOptionsLoading(false)
      })

    return () => {
      active = false
    }
  }, [effectiveSessionId, gw, optionsRequest])

  const discoverModels = useCallback(
    (selected: ModelOptionProvider, refresh = false) => {
      const cached = discoveries.current.get(selected.slug)
      if (!refresh && cached && (cached.status === 'loading' || cached.status === 'ready')) {
        return
      }

      const requestId = ++nextDiscoveryRequest.current
      discoveries.current.set(selected.slug, {
        models: cached?.models ?? [],
        requestId,
        status: 'loading'
      })
      setDiscoveryVersion(version => version + 1)

      void gw
        .request<ModelModelsResponse>('model.models', {
          profile_name: selected.slug
        })
        .then(raw => {
          if (discoveries.current.get(selected.slug)?.requestId !== requestId) {
            return
          }
          const result = asRpcResult<ModelModelsResponse>(raw)
          if (!result) {
            throw new Error('invalid response: model.models')
          }
          const warning = result.warning?.trim()
          discoveries.current.set(selected.slug, {
            models: uniqueModels(result.models ?? []),
            requestId,
            source: result.source,
            status: warning ? 'partial' : 'ready',
            ...(warning ? { warning } : {})
          })
          setDiscoveryVersion(version => version + 1)
        })
        .catch((reason: unknown) => {
          if (discoveries.current.get(selected.slug)?.requestId !== requestId) {
            return
          }
          discoveries.current.set(selected.slug, {
            error: rpcErrorMessage(reason),
            models: cached?.models ?? [],
            requestId,
            status: 'error'
          })
          setDiscoveryVersion(version => version + 1)
        })
    },
    [gw]
  )

  const providerNames = useMemo(() => providerDisplayNames(providers), [providers])
  const providerRows = useMemo<ProviderRow[]>(
    () =>
      providers.map((provider, index) => ({
        name: providerNames[index] ?? provider.name ?? provider.slug,
        provider
      })),
    [providerNames, providers]
  )

  const filteredProviderRows = useMemo(() => {
    if (stage !== 'provider' || !filter.trim()) {
      return providerRows
    }

    return fuzzyRank(
      providerRows,
      filter,
      row =>
        `${row.name} ${row.provider.slug} ${row.provider.provider_type ?? ''} ${row.provider.configured_model ?? ''}`
    ).map(result => result.item)
  }, [filter, providerRows, stage])

  const provider = useMemo(() => {
    if (stage === 'model') {
      return providers.find(candidate => candidate.slug === modelProviderSlug)
    }

    return filteredProviderRows[providerIdx]?.provider
  }, [filteredProviderRows, modelProviderSlug, providerIdx, providers, stage])

  const providerName = useMemo(() => {
    if (!provider) {
      return '(unknown provider)'
    }

    const index = providers.findIndex(candidate => candidate.slug === provider.slug)

    return providerNames[index] ?? provider.name ?? provider.slug
  }, [provider, providerNames, providers])

  const discovery = useMemo(
    () => (modelProviderSlug ? discoveries.current.get(modelProviderSlug) : undefined),
    [discoveryVersion, modelProviderSlug]
  )
  const allModels = useMemo(
    () =>
      uniqueModels([
        ...(provider?.is_current ? [currentModel] : []),
        ...(discovery?.models ?? []),
        provider?.configured_model
      ]),
    [currentModel, discovery, provider]
  )
  const filteredModels = useMemo(() => {
    if (stage !== 'model' || !filter.trim()) {
      return allModels
    }

    return fuzzyRank(allModels, filter, model => model).map(result => result.item)
  }, [allModels, filter, stage])
  const modelRows = useMemo<ModelRow[]>(() => {
    if (filteredModels.length > 0) {
      return groupByFamily(filteredModels.map(model => ({ custom: false, model })))
    }
    const custom = filter.trim()

    return custom ? [{ custom: true, model: custom }] : []
  }, [filter, filteredModels])

  useEffect(() => {
    setProviderIdx(index => Math.max(0, Math.min(index, Math.max(0, filteredProviderRows.length - 1))))
  }, [filteredProviderRows.length])

  useEffect(() => {
    setModelIdx(index => Math.max(0, Math.min(index, Math.max(0, modelRows.length - 1))))
  }, [modelRows.length])

  const back = useCallback(() => {
    if (filter.trim()) {
      setFilter('')
      setModelIdx(0)

      if (stage === 'provider') {
        setProviderIdx(0)
      }

      return
    }

    if (stage === 'model') {
      const fullIndex = providerRows.findIndex(row => row.provider.slug === modelProviderSlug)
      setProviderIdx(Math.max(0, fullIndex))
      setModelIdx(0)
      setModelProviderSlug(null)
      setStage('provider')

      return
    }

    close()
  }, [close, filter, modelProviderSlug, providerRows, stage])

  const handleKey = useCallback(
    (event: KeyEvent) => {
      const name = event.name.toLowerCase()
      const sequence = event.sequence ?? ''
      const isEscape = name === 'escape'
      const isReturn = name === 'return' || name === 'enter' || name === 'kpenter'
      const isClose = event.ctrl && name === 'c'
      const isRefresh = event.ctrl && name === 'r'

      if (isClose) {
        consume(event)
        close()

        return
      }

      if (optionsLoading) {
        if (isEscape) {
          consume(event)
          close()
        }

        return
      }

      // Two-pane keys: ←→ moves between the provider rail and the model pane;
      // each pane keeps its own selection. Enter on the rail dives right.
      if (twoPane && providers.length > 0) {
        const focusedProvider = filteredProviderRows[providerIdx]?.provider

        if (isEscape) {
          consume(event)

          if (filter.trim()) {
            setFilter('')
            setModelIdx(0)
          } else {
            close()
          }

          return
        }

        if (name === 'right' && stage === 'provider' && focusedProvider) {
          consume(event)
          setModelProviderSlug(focusedProvider.slug)
          discoverModels(focusedProvider)
          setStage('model')

          return
        }

        if (name === 'left' && stage === 'model') {
          consume(event)
          setStage('provider')

          return
        }

        if (name === 'up' || name === 'down') {
          consume(event)
          const delta = name === 'up' ? -1 : 1

          if (stage === 'provider') {
            setProviderIdx(index => Math.max(0, Math.min(filteredProviderRows.length - 1, index + delta)))
          } else {
            setModelIdx(index => Math.max(0, Math.min(modelRows.length - 1, index + delta)))
          }

          return
        }

        if (isReturn) {
          consume(event)

          if (stage === 'provider' && focusedProvider) {
            setModelProviderSlug(focusedProvider.slug)
            discoverModels(focusedProvider)
            setStage('model')

            return
          }

          const model = event.ctrl && filter.trim() ? filter.trim() : modelRows[modelIdx]?.model

          if (modelProviderSlug && model) {
            select(
              `${model} --provider ${modelProviderSlug}${
                allowPersistGlobal && persistGlobal ? ' --global' : ` ${TUI_SESSION_MODEL_FLAG}`
              }`
            )
          }

          return
        }
      }

      // With no provider profiles (initial load failed or none configured)
      // Ctrl+R retries model.options. Every other key keeps its normal
      // meaning — the error stays inline and never becomes a modal takeover
      // where Esc destroys the whole picker.
      if (providers.length === 0 && isRefresh) {
        consume(event)
        setOptionsRequest(request => request + 1)

        return
      }

      if (isEscape) {
        consume(event)
        back()

        return
      }

      if (isRefresh && stage === 'model' && provider) {
        consume(event)
        if (discovery?.status !== 'loading') {
          discoverModels(provider, true)
        }

        return
      }

      if (name === 'up') {
        consume(event)

        if (stage === 'provider') {
          setProviderIdx(index => Math.max(0, index - 1))
        } else {
          setModelIdx(index => Math.max(0, index - 1))
        }

        return
      }

      if (name === 'down') {
        consume(event)

        if (stage === 'provider') {
          setProviderIdx(index => Math.min(Math.max(0, filteredProviderRows.length - 1), index + 1))
        } else {
          setModelIdx(index => Math.min(Math.max(0, modelRows.length - 1), index + 1))
        }

        return
      }

      if (isReturn) {
        consume(event)

        if (stage === 'provider') {
          const selected = filteredProviderRows[providerIdx]?.provider

          if (!selected) {
            return
          }

          setModelProviderSlug(selected.slug)
          setModelIdx(0)
          setStage('model')
          setFilter('')
          discoverModels(selected)

          return
        }

        const model = event.ctrl && filter.trim() ? filter.trim() : modelRows[modelIdx]?.model

        if (!provider || !model) {
          return
        }

        select(
          `${model} --provider ${provider.slug}${
            allowPersistGlobal && persistGlobal ? ' --global' : ` ${TUI_SESSION_MODEL_FLAG}`
          }`
        )

        return
      }

      if (name === 'backspace' || name === 'delete') {
        consume(event)
        setFilter(value => value.slice(0, -1))

        if (stage === 'provider') {
          setProviderIdx(0)
        } else {
          setModelIdx(0)
        }

        return
      }

      if (event.ctrl && name === 'u') {
        consume(event)
        setFilter('')

        if (stage === 'provider') {
          setProviderIdx(0)
        } else {
          setModelIdx(0)
        }

        return
      }

      // Mockup 09 footer: "a set as default". 'a' is an alias for the
      // ctrl+g scope toggle so committing the picked model as the durable
      // default is one key; ctrl+g keeps working for discoverability.
      if (
        allowPersistGlobal &&
        ((event.ctrl && name === 'g') ||
          (!event.ctrl &&
            !event.meta &&
            !event.super &&
            sequence.length === 1 &&
            sequence >= ' ' &&
            sequence === 'a'))
      ) {
        consume(event)
        setPersistGlobal(value => !value)

        return
      }

      if (!event.ctrl && !event.meta && !event.super && sequence.length === 1 && sequence >= ' ') {
        consume(event)
        setFilter(value => value + sequence)

        if (stage === 'provider') {
          setProviderIdx(0)
        } else {
          setModelIdx(0)
        }
      }
    },
    [
      allowPersistGlobal,
      back,
      close,
      discoverModels,
      discovery,
      filter,
      filteredProviderRows,
      modelIdx,
      modelProviderSlug,
      twoPane,
      modelRows,
      optionsLoading,
      persistGlobal,
      provider,
      providerIdx,
      providers.length,
      select,
      stage
    ]
  )

  useKeyboard(handleKey)

  const panelWidth = Math.max(
    1,
    Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, width - 6), Math.max(1, width - 2))
  )
  const visible = Math.max(1, Math.min(MAX_VISIBLE, height - 16))
  const panelHeight = Math.min(height, visible + 12 + (optionsError ? 1 : 0))

  // Mockup 09: on a wide terminal both stages sit side by side — profiles and
  // providers in the left rail, the focused profile's models on the right —
  // so browsing models never loses your provider place. Narrow terminals
  // keep the sequential wizard.

  // Keep the right pane pointed at whichever provider row is highlighted.
  useEffect(() => {
    if (!twoPane || stage !== 'provider' || optionsLoading) {
      return
    }

    const focused = filteredProviderRows[providerIdx]?.provider

    if (focused && focused.slug !== modelProviderSlug) {
      setModelProviderSlug(focused.slug)
      discoverModels(focused)
      setModelIdx(0)
    }
  }, [discoverModels, filteredProviderRows, modelProviderSlug, optionsLoading, providerIdx, stage, twoPane])

  if (optionsLoading) {
    return (
      <ModalShell height={height} panelHeight={5} panelWidth={panelWidth} t={t} title="Select model" width={width}>
        <InfoRow color={t.color.muted}>loading provider profiles…</InfoRow>
        <InfoRow color={t.color.muted}>Esc close</InfoRow>
      </ModalShell>
    )
  }

  // Mockup 09's wide layout: provider rail left, focused profile's models
  // right, both visible at once. The rail is `stage === 'provider'`.
  if (twoPane && providers.length > 0) {
    const focused = filteredProviderRows[providerIdx]?.provider
    const rightSlug = modelProviderSlug ?? focused?.slug ?? null
    const rightProvider = providers.find(candidate => candidate.slug === rightSlug)
    const rightDiscovery = discoveryVersion >= 0 && rightSlug ? discoveries.current.get(rightSlug) : undefined
    const rightAllModels = uniqueModels([
      ...(rightProvider?.is_current ? [currentModel] : []),
      ...(rightDiscovery?.models ?? []),
      rightProvider?.configured_model
    ])
    const rightFiltered =
      stage === 'model' && filter.trim()
        ? fuzzyRank(rightAllModels, filter, model => model).map(result => result.item)
        : rightAllModels
    const rightRows: ModelRow[] =
      rightFiltered.length > 0
        ? groupByFamily(rightFiltered.map(model => ({ custom: false, model })))
        : filter.trim()
          ? [{ custom: true, model: filter.trim() }]
          : []
    let cursor = 0
    const rightDisplay = familyDisplayRows(rightRows).map(entry => {
      if (entry.caption) {
        return { ...entry, index: -1 }
      }

      const index = cursor

      cursor += 1

      return { ...entry, index }
    })
    // Rows the panel's own chrome costs, counted rather than guessed:
    // frame 2 + padding 2 + header 1 + footer 2 (its row plus its marginTop)
    // + 1 for the panes' marginTop. The old figure was 5, so the frame was
    // three rows shorter than its contents and the provider rail painted its
    // last entry BELOW the footer and outside the border.
    const WIDE_PANEL_CHROME = 8
    // One shared budget, derived from what the panel can actually show, minus
    // the pane's own STEP caption.
    const paneRows = Math.max(3, Math.min(visible, height - 2 - WIDE_PANEL_CHROME) - 1)
    const providerRowsVisible = Math.min(filteredProviderRows.length, paneRows)
    // Budgeted in DISPLAY entries — family captions included, because they
    // occupy rows too.
    const modelRowsVisible = Math.min(rightDisplay.length, paneRows)
    // The pane must FOLLOW the selection, not show the first N rows forever.
    // It used to `.slice(0, modelRowsVisible)`, so a provider with fourteen
    // models showed the same three or four no matter how far down you
    // arrowed — the selection moved, the list did not, and the picker read as
    // broken rather than as scrolled.
    const rightSelected = Math.max(
      0,
      rightDisplay.findIndex(entry => entry.index === modelIdx && !entry.caption)
    )
    const rightStart = Math.max(
      0,
      Math.min(rightSelected - Math.floor(modelRowsVisible / 2), rightDisplay.length - modelRowsVisible)
    )
    const rightWindow = rightDisplay.slice(rightStart, rightStart + modelRowsVisible)
    const rightHidden = rightDisplay.length - rightStart - rightWindow.length
    // Content-sized like the other redesigned overlays. The previous fixed
    // `visible + 12` budget left a large empty black slab under short model
    // lists in tall terminals, which made this picker feel unrelated to F6–F8.
    const widePanelHeight = Math.min(
      Math.max(1, height - 2),
      Math.max(12, Math.max(providerRowsVisible, modelRowsVisible) + 1 + WIDE_PANEL_CHROME)
    )
    const paneLabel = stage === 'provider' ? 'profiles' : 'models'
    const filterLabel = filter ? `search: ${filter}▎` : `${paneLabel} · type to filter`

    return (
      <ModalShell
        footer={
          <Box flexDirection="row" width="100%">
            <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
              <Text color={t.color.muted} wrap="truncate-end">
                {`←→ pane · ↑↓ select · Enter use${allowPersistGlobal ? ' · a set as default' : ''}`}
              </Text>
            </Box>
            <Box flexShrink={0}>
              <Text color={t.color.muted} wrap="truncate-end">
                {`current ${compactPreview(currentModel || 'unknown', 18)} · ${persistGlobal ? 'default' : 'session'} · Esc cancel`}
              </Text>
            </Box>
          </Box>
        }
        headerRight={<Text color={filter ? t.color.accent : t.color.muted}>{filterLabel}</Text>}
        height={height}
        panelHeight={widePanelHeight}
        panelWidth={panelWidth}
        t={t}
        title="Model"
        titleDetail={providerName}
        width={width}
      >
        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0} marginTop={1} paddingLeft={2} paddingRight={2}>
          {/* Provider rail: mockup's left column. */}
          {/* overflow:hidden is load-bearing: without it the rail paints its
              overflowing rows straight through the panel's border and past
              the footer, which is exactly what a 9-provider list did. */}
          <Box flexDirection="column" flexShrink={0} minHeight={0} overflow="hidden" paddingRight={1} width={32}>
            {/* Both stages stay on screen. A wizard that replaces the
                provider list with the model list makes you remember which
                provider you picked, and back out to check. */}
            <Text wrap="truncate-end">
              <Span color={t.ds.caption}>STEP 1 </Span>
              <Span color={stage === 'provider' ? t.color.accent : t.ds.secondary}>provider</Span>
              <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
              <Span color={t.ds.caption}>{filteredProviderRows.length}</Span>
            </Text>
            {windowItems(filteredProviderRows, providerIdx, providerRowsVisible).items.map(row => {
              const selected = row.provider.slug === focused?.slug
              // Mockup 09: health before commitment. A profile whose discovery
              // already failed shows red ✗ plus an offline tag and a grayed
              // name BEFORE you select it. Only what discovery already knows
              // is surfaced here — no reachability probes are invented.
              // "No models" is usually a key problem, not an empty
              // catalogue — so the row carries a reachability dot and where
              // the credential came from, before you commit to opening it.
              const known = discoveries.current.get(row.provider.slug)
              const offline = known?.status === 'error'
              const nameColor = offline
                ? t.ds.secondary
                : selected
                  ? t.color.accent
                  : row.provider.is_current
                    ? t.ds.title
                    : t.ds.secondary
              const providerMeta = offline
                ? 'offline'
                : row.provider.is_current
                  ? 'active'
                  : row.provider.provider_type?.trim() || row.provider.auth_type?.trim() || ''

              return (
                <Box
                  backgroundColor={selected ? t.color.completionCurrentBg : undefined}
                  flexDirection="row"
                  flexShrink={0}
                  key={row.provider.slug}
                  paddingLeft={1}
                >
                  <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
                    <Text wrap="truncate-end">
                      {/* One dot, three meanings, the same three everywhere
                          else in the product: red is broken, green is
                          reachable, blue is the one in use. */}
                      <Span color={offline ? t.ds.failed : row.provider.is_current ? t.ds.working : t.ds.done}>
                        {`${selected ? '›' : ' '} ${GLYPH.state} `}
                      </Span>
                      <Span color={nameColor}>{compactPreview(row.name, providerMeta ? 18 : 24)}</Span>
                    </Text>
                  </Box>
                  {providerMeta ? (
                    <Box flexShrink={0}>
                      <Text color={offline ? t.color.error : t.color.muted} dimColor={!offline && !row.provider.is_current}>
                        {compactPreview(providerMeta, 8)}
                      </Text>
                    </Box>
                  ) : null}
                </Box>
              )
            })}
          </Box>
          <Box backgroundColor={t.color.border} flexShrink={0} width={1} />
          {/* Model pane: the highlighted profile's models, family-grouped. */}
          <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} overflow="hidden" paddingLeft={2}>
            <Text wrap="truncate-end">
              <Span color={t.ds.caption}>STEP 2 </Span>
              <Span color={stage === 'model' ? t.color.accent : t.ds.secondary}>model</Span>
              <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
              <Span color={t.ds.caption}>{rightRows.length}</Span>
            </Text>
            {rightDiscovery?.status === 'loading' ? (
              <Text color={t.color.muted} wrap="truncate-end">
                discovering models…
              </Text>
            ) : null}
            {rightDiscovery?.status === 'error' ? (
              <Text color={t.color.error} wrap="truncate-end">
                discovery failed — type a full model ID
              </Text>
            ) : null}
            {rightWindow.map((entry, index) =>
              entry.caption ? (
                <Text color={t.color.muted} dimColor key={`family-${index}-${entry.caption}`} wrap="truncate-end">
                  {`  ${entry.caption}`}
                </Text>
              ) : (
                <Box
                  backgroundColor={
                    entry.index === modelIdx && stage === 'model' ? t.color.completionCurrentBg : undefined
                  }
                  flexShrink={0}
                  key={`${rightSlug}:${entry.row?.model ?? index}`}
                  paddingLeft={1}
                >
                  <Text
                    color={entry.index === modelIdx && stage === 'model' ? t.color.accent : t.color.text}
                    wrap="truncate-end"
                  >
                    {`${entry.index === modelIdx && stage === 'model' ? '›' : ' '} ${
                      entry.row?.model === currentModel ? '*' : ' '
                    } ${entry.row?.custom ? `Use "${entry.row.model}"` : entry.row?.model ?? ' '}`}
                  </Text>
                </Box>
              )
            )}
            {/* How much is off-screen, in both directions — otherwise a
                windowed list is indistinguishable from a truncated one. */}
            {rightStart > 0 || rightHidden > 0 ? (
              <Text color={t.ds.caption} wrap="truncate-end">
                {`${rightStart > 0 ? `↑ ${rightStart}` : ''}${rightStart > 0 && rightHidden > 0 ? '  ' : ''}${
                  rightHidden > 0 ? `↓ ${rightHidden}` : ''
                }  ${modelIdx + 1}/${rightRows.length}`}
              </Text>
            ) : null}
            {rightDiscovery?.status === 'partial' ? (
              <Text color={t.color.warn} wrap="truncate-end">
                {rightDiscovery.warning}
              </Text>
            ) : null}
          </Box>
        </Box>
      </ModalShell>
    )
  }

  // A failed model.options load (or a genuinely empty profile list) renders
  // through the normal provider stage below: the error shows inline and the
  // picker keeps its usual browsing keys instead of taking over the screen.
  if (stage === 'provider') {
    const rows = filteredProviderRows.map(({ name, provider: item }) => {
      const cached = discoveries.current.get(item.slug)
      const mark = item.is_current ? '*' : '●'
      const suffix =
        cached?.status === 'ready'
          ? `${cached.models.length} available`
          : cached?.status === 'loading'
            ? 'discovering…'
            : cached?.status === 'partial'
              ? `incomplete · ${cached.warning ?? 'retry available'}`
              : cached?.status === 'error'
                ? 'discovery failed'
                : 'discover models'

      return {
        discovery: cached,
        id: item.slug,
        item,
        label: `${mark} ${name} · ${suffix}`
      }
    })
    const { items, offset } = windowItems(rows, providerIdx, visible)

    return (
      <ModalShell
        height={height}
        panelHeight={panelHeight}
        panelWidth={panelWidth}
        t={t}
        title="Model"
        width={width}
      >
        <InfoRow color={t.color.muted}>step 1/2 · provider · Enter to continue</InfoRow>
        <InfoRow color={t.color.muted}>Current: {currentModel || '(unknown)'}</InfoRow>
        {optionsError ? <InfoRow color={t.color.error}>error: {optionsError}</InfoRow> : null}
        <InfoRow color={filter ? t.color.accent : t.color.muted}>
          {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
        </InfoRow>
        <InfoRow color={t.color.warn}>{provider?.warning ? `warning: ${provider.warning}` : ' '}</InfoRow>
        <InfoRow color={t.color.muted}>{offset > 0 ? `↑ ${offset} more` : ' '}</InfoRow>

        {items.length === 0 ? (
          <InfoRow color={t.color.muted}>{filter.trim() ? 'no providers match' : 'no providers available'}</InfoRow>
        ) : (
          Array.from({ length: visible }, (_, index) => {
            const row = items[index]
            const absoluteIndex = offset + index
            const selected = absoluteIndex === providerIdx

            return (
              <box
                backgroundColor={selected ? t.color.completionCurrentBg : undefined}
                flexShrink={0}
                height={1}
                key={row?.id ?? `provider-pad-${index}`}
                paddingLeft={2}
                paddingRight={2}
                width="100%"
              >
                <text
                  fg={
                    selected
                      ? t.color.accent
                      : row?.discovery?.status === 'partial'
                        ? t.color.warn
                        : row?.discovery?.status === 'error'
                          ? t.color.error
                          : t.color.text
                  }
                  flexShrink={0}
                  truncate
                  width="100%"
                  wrapMode="none"
                >
                  {row ? `${selected ? '›' : ' '} ${absoluteIndex + 1}. ${row.label}` : ' '}
                </text>
              </box>
            )
          })
        )}

        <InfoRow color={t.color.muted}>
          {offset + visible < rows.length ? `↓ ${rows.length - offset - visible} more` : ' '}
        </InfoRow>
        <InfoRow color={t.color.muted}>
          {allowPersistGlobal
            ? `persist: ${persistGlobal ? 'global' : 'live runtime'} · a set as default · ctrl+g toggle`
            : 'scope: live runtime'}
        </InfoRow>
        <InfoRow color={t.color.muted}>
          {providers.length === 0
            ? 'Ctrl+R retry · Esc close'
            : allowPersistGlobal
              ? '↑/↓ select · Enter discover · a set as default · Esc clear/back'
              : '↑/↓ select · Enter discover · Esc clear/back · Ctrl+C close'}
        </InfoRow>
      </ModalShell>
    )
  }

  const { items, offset } = windowItems(modelRows, modelIdx, visible)
  // Mockup 09's grouped model pane: family captions interleaved with rows,
  // derived from IDs only — no invented context windows or pricing badges.
  let sequentialIndex = offset
  const displayRows = familyDisplayRows(items).map(entry => {
    if (entry.caption) {
      return { ...entry, absoluteIndex: -1 }
    }

    const absoluteIndex = sequentialIndex

    sequentialIndex += 1

    return { ...entry, absoluteIndex }
  })
  const discoveryMessage =
    discovery?.status === 'loading'
      ? 'discovering models from this profile…'
      : discovery?.status === 'error'
        ? `discovery failed: ${discovery.error ?? 'unknown error'}`
        : discovery?.status === 'partial'
          ? `warning: ${discovery.warning}`
          : discovery?.source
            ? `source: ${discovery.source}`
            : ' '
  const discoveryColor =
    discovery?.status === 'error' ? t.color.error : discovery?.warning ? t.color.warn : t.color.muted

  return (
    <ModalShell
      height={height}
      panelHeight={panelHeight}
      panelWidth={panelWidth}
      t={t}
      title={`Model › ${providerName}`}
      width={width}
    >
      <InfoRow color={t.color.muted}>step 2/2 · Esc back</InfoRow>
      <InfoRow color={filter ? t.color.accent : t.color.muted}>
        {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
      </InfoRow>
      <InfoRow color={provider?.warning ? t.color.warn : discoveryColor}>
        {provider?.warning ? `warning: ${provider.warning}` : discoveryMessage}
      </InfoRow>
      <InfoRow color={t.color.muted}>{offset > 0 ? `↑ ${offset} more` : ' '}</InfoRow>

      {Array.from({ length: visible }, (_, index) => {
        const entry = displayRows[index]
        const row = entry?.row
        const model = row?.model
        const absoluteIndex = entry?.absoluteIndex ?? offset + index
        const selected = absoluteIndex === modelIdx

        if (entry?.caption) {
          return (
            <box flexShrink={0} height={1} key={`family-${entry.caption}`} paddingLeft={2} paddingRight={2} width="100%">
              <text fg={t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
                {entry.caption}
              </text>
            </box>
          )
        }

        return (
          <box
            backgroundColor={selected ? t.color.completionCurrentBg : undefined}
            flexShrink={0}
            height={1}
            key={model ? `${provider?.slug ?? 'provider'}:${model}` : `model-pad-${index}`}
            paddingLeft={2}
            paddingRight={2}
            width="100%"
          >
            <text fg={selected ? t.color.accent : t.color.text} flexShrink={0} truncate width="100%" wrapMode="none">
              {model
                ? `${selected ? '›' : model === currentModel ? '*' : ' '} ${absoluteIndex + 1}. ${
                    row.custom ? `Use "${model}"` : model
                  }`
                : index === 0 && items.length === 0
                  ? filter.trim()
                    ? `Use Ctrl+Enter to select "${filter.trim()}"`
                    : discovery?.status === 'loading'
                      ? 'discovering models…'
                      : 'no models discovered · type a full model ID'
                  : ' '}
            </text>
          </box>
        )
      })}

      <InfoRow color={t.color.muted}>
        {offset + visible < modelRows.length ? `↓ ${modelRows.length - offset - visible} more` : ' '}
      </InfoRow>
      <InfoRow color={t.color.muted}>
        {allowPersistGlobal
          ? `persist: ${persistGlobal ? 'global' : 'live runtime'} · ctrl+g toggle`
          : 'scope: live runtime'}
      </InfoRow>
      <InfoRow color={t.color.muted}>
        {discovery?.status === 'loading'
          ? 'Enter fallback · type full ID · Esc back · Ctrl+C close'
          : discovery?.status === 'error'
            ? 'type full ID · Ctrl+R retry · Esc clear/back'
            : discovery?.status === 'partial'
              ? 'fallback available · Ctrl+R retry · Esc clear/back'
              : 'Enter switch · Ctrl+Enter typed ID · Ctrl+R refresh · Esc clear/back'}
      </InfoRow>
    </ModalShell>
  )
}
