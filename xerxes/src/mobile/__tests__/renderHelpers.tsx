// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'

import { createDraftStore } from '../../client/drafts.js'
import { createHubStore, type HubState } from '../../client/hubStore.js'
import { createScrollMemory, type AppServices } from '../context.js'
import { MobileApp } from '../MobileApp.js'
import { createMemoryRouter } from '../routes.js'
import { NOW, stubController } from './helpers.js'

export function makeServices(state: HubState, path: string, extra: Partial<AppServices> = {}): AppServices {
  return {
    store: createHubStore(state),
    controller: stubController(),
    router: createMemoryRouter(path),
    drafts: createDraftStore(),
    scroll: createScrollMemory(),
    now: () => NOW,
    ...extra,
  }
}

export function render(state: HubState, path: string, extra: Partial<AppServices> = {}): string {
  return renderToStaticMarkup(<MobileApp services={makeServices(state, path, extra)} />)
}
