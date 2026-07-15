// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// OpenTUI app root. The mature controller keeps its session, streaming,
// slash-command, and submission logic while terminal-facing compatibility
// imports resolve to the OpenTUI runtime shim at build time.
import type { GatewayClient } from '../gatewayClient.js'
import { GatewayProvider } from '../app/gatewayContext.js'
import { useMainApp } from '../app/useMainApp.js'

import { AppLayout } from './appLayout.js'

export function AppOpenTui({ gw }: { gw: GatewayClient }) {
  const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw)

  return (
    <GatewayProvider value={gateway}>
      <AppLayout
        actions={appActions}
        composer={appComposer}
        progress={appProgress}
        status={appStatus}
        transcript={appTranscript}
      />
    </GatewayProvider>
  )
}
