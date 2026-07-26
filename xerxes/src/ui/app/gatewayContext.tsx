// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createContext, useContext } from 'react'

import type { GatewayProviderProps, GatewayServices } from './interfaces.js'

const GatewayContext = createContext<GatewayServices | null>(null)

export function GatewayProvider({ children, value }: GatewayProviderProps) {
  return <GatewayContext.Provider value={value}>{children}</GatewayContext.Provider>
}

export function useGateway() {
  const value = useContext(GatewayContext)

  if (!value) {
    throw new Error('GatewayContext missing')
  }

  return value
}

/**
 * Gateway access for components that must also render outside a provider
 * (tests, embedded previews). Returns null instead of throwing; callers
 * degrade honestly by hiding gateway-dependent actions.
 */
export function useOptionalGateway(): GatewayServices | null {
  return useContext(GatewayContext)
}
