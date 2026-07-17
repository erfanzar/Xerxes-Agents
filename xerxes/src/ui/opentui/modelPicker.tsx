// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
import type { KeyEvent } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme } from '../app/uiStore.js'
import { providerDisplayNames } from '../domain/providers.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { ModelModelsResponse, ModelOptionProvider, ModelOptionsResponse } from '../gatewayTypes.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

const MAX_VISIBLE = 12
const MIN_PANEL_WIDTH = 40
const MAX_PANEL_WIDTH = 90

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

const windowItems = <T,>(items: readonly T[], selected: number, visible: number) => {
  const offset = Math.max(0, Math.min(selected - Math.floor(visible / 2), items.length - visible))

  return { items: items.slice(offset, offset + visible), offset }
}

const uniqueModels = (values: readonly (null | string | undefined)[]) => [
  ...new Set(values.map(value => value?.trim()).filter((value): value is string => Boolean(value)))
]

function ModalShell({
  children,
  height,
  panelHeight,
  panelWidth,
  t,
  title,
  width
}: {
  children: ReactNode
  height: number
  panelHeight: number
  panelWidth: number
  t: Theme
  title: string
  width: number
}) {
  const top = Math.max(0, Math.floor((height - panelHeight) / 2))

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height={height}
      left={0}
      paddingTop={top}
      position="absolute"
      top={0}
      width={width}
      zIndex={200}
    >
      <box
        backgroundColor={t.color.statusBg}
        flexDirection="column"
        flexShrink={0}
        height={panelHeight}
        paddingBottom={1}
        paddingTop={1}
        width={panelWidth}
      >
        <box flexDirection="row" flexShrink={0} justifyContent="space-between" paddingLeft={2} paddingRight={2}>
          <text fg={t.color.accent} flexShrink={0}>
            <b>{title}</b>
          </text>
          <text fg={t.color.muted} flexShrink={0}>
            esc
          </text>
        </box>
        {children}
      </box>
    </box>
  )
}

function InfoRow({ children, color, pad = true }: { children: ReactNode; color: string; pad?: boolean }) {
  return (
    <box flexShrink={0} paddingLeft={pad ? 2 : 0} paddingRight={pad ? 2 : 0}>
      <text fg={color} flexShrink={0} truncate width="100%" wrapMode="none">
        {children}
      </text>
    </box>
  )
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
      return filteredModels.map(model => ({ custom: false, model }))
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

      if (optionsError || providers.length === 0) {
        if (isEscape) {
          consume(event)
          close()
        } else if (isRefresh) {
          consume(event)
          setOptionsRequest(request => request + 1)
        }

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

      if (allowPersistGlobal && event.ctrl && name === 'g') {
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
      modelRows,
      optionsError,
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
  const panelHeight = Math.min(height, visible + 12)

  if (optionsLoading) {
    return (
      <ModalShell height={height} panelHeight={5} panelWidth={panelWidth} t={t} title="Select model" width={width}>
        <InfoRow color={t.color.muted}>loading provider profiles…</InfoRow>
        <InfoRow color={t.color.muted}>Esc close</InfoRow>
      </ModalShell>
    )
  }

  if (optionsError) {
    return (
      <ModalShell height={height} panelHeight={7} panelWidth={panelWidth} t={t} title="Select model" width={width}>
        <box flexShrink={0} paddingLeft={2} paddingRight={2} paddingTop={1}>
          <text fg={t.color.error} flexShrink={0} wrapMode="word">
            error: {optionsError}
          </text>
        </box>
        <InfoRow color={t.color.muted}>Ctrl+R retry · Esc close</InfoRow>
      </ModalShell>
    )
  }

  if (providers.length === 0) {
    return (
      <ModalShell height={height} panelHeight={6} panelWidth={panelWidth} t={t} title="Select model" width={width}>
        <InfoRow color={t.color.muted}>no providers available</InfoRow>
        <InfoRow color={t.color.muted}>Ctrl+R retry · Esc close</InfoRow>
      </ModalShell>
    )
  }

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
        title="Select provider · step 1/2"
        width={width}
      >
        <InfoRow color={t.color.muted}>Full model IDs on the next step · Enter to continue</InfoRow>
        <InfoRow color={t.color.muted}>Current: {currentModel || '(unknown)'}</InfoRow>
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
            ? `persist: ${persistGlobal ? 'global' : 'live runtime'} · ctrl+g toggle`
            : 'scope: live runtime'}
        </InfoRow>
        <InfoRow color={t.color.muted}>↑/↓ select · Enter discover · Esc clear/back · Ctrl+C close</InfoRow>
      </ModalShell>
    )
  }

  const { items, offset } = windowItems(modelRows, modelIdx, visible)
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
      title="Select model · step 2/2"
      width={width}
    >
      <InfoRow color={t.color.muted}>{providerName} · Esc back</InfoRow>
      <InfoRow color={filter ? t.color.accent : t.color.muted}>
        {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
      </InfoRow>
      <InfoRow color={provider?.warning ? t.color.warn : discoveryColor}>
        {provider?.warning ? `warning: ${provider.warning}` : discoveryMessage}
      </InfoRow>
      <InfoRow color={t.color.muted}>{offset > 0 ? `↑ ${offset} more` : ' '}</InfoRow>

      {Array.from({ length: visible }, (_, index) => {
        const row = items[index]
        const model = row?.model
        const absoluteIndex = offset + index
        const selected = absoluteIndex === modelIdx

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
