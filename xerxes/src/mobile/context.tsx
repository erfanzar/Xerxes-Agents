// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createContext, useContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactElement, type ReactNode } from 'react'

import type { DraftStore } from '../client/drafts.js'
import type { HubController } from '../client/hubController.js'
import type { HubState, HubStore } from '../client/hubStore.js'
import type { ImageEngine } from '../client/imagePrep.js'
import type { VoiceControl } from '../client/voiceControl.js'
import type { HubHttp } from './api.js'
import type { PairingRead } from './pairing.js'
import type { PushService } from './pwa/push.js'
import { parseRoute, type Route, type Router } from './routes.js'
import type { VoiceCallService } from './voice/call/service.js'

/** Everything the pair page needs; the fragment was already read and cleared before render. */
export interface PairEnv {
  readonly read: PairingRead
  readonly http: Pick<HubHttp, 'consumePairing' | 'refresh'>
  readonly deviceLabel: string
  /** Called once the Hub accepted this device (the session refresh succeeded). */
  readonly onPaired: () => void
  readonly pollMs?: number
  readonly maxWaitMs?: number
}

export interface ScrollMemory {
  get(key: string): number | undefined
  set(key: string, top: number): void
}

export function createScrollMemory(): ScrollMemory {
  const positions = new Map<string, number>()
  return { get: key => positions.get(key), set: (key, top) => { positions.set(key, top) } }
}

export interface AppServices {
  readonly store: HubStore
  readonly controller: HubController
  readonly router: Router
  readonly drafts: DraftStore
  readonly scroll: ScrollMemory
  readonly now: () => number
  /** Injected by the voice phase. When absent the talk button is not rendered. */
  readonly voice?: VoiceControl | undefined
  /** App-scoped phone call. MobileApp supplies it when the embedding host does not inject one. */
  readonly call?: VoiceCallService
  readonly push?: PushService | undefined
  readonly pair?: PairEnv | undefined
  /** Browser image engine; tests inject a fake. */
  readonly imageEngine?: ImageEngine | undefined
}

const ServicesContext = createContext<AppServices | null>(null)

export function ServicesProvider({ services, children }: { services: AppServices; children: ReactNode }): ReactElement {
  return <ServicesContext.Provider value={services}>{children}</ServicesContext.Provider>
}

export function useServices(): AppServices {
  const services = useContext(ServicesContext)
  if (services === null) throw new Error('useServices must be used inside ServicesProvider')
  return services
}

export function useCallService(): VoiceCallService {
  const { call } = useServices()
  if (!call) throw new Error('Voice UI requires the app-scoped call service.')
  return call
}

export function useHubState(): HubState {
  const { store } = useServices()
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
}

export function useRoute(): Route {
  const { router } = useServices()
  const path = useSyncExternalStore(router.subscribe, router.path, router.path)
  return useMemo(() => parseRoute(path), [path])
}

/** A clock that re-renders on an interval so relative times stay fresh. */
export function useNow(intervalMs = 30_000): number {
  const { now } = useServices()
  const [value, setValue] = useState(() => now())
  useEffect(() => {
    const id = setInterval(() => setValue(now()), intervalMs)
    return () => clearInterval(id)
  }, [now, intervalMs])
  return value
}
