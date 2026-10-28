// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { tierDownloadBytes } from '../../client/voiceModelManifest.js'
import { CallView } from '../components/CallView.js'
import { ServicesProvider } from '../context.js'
import { liveState } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'

describe('call view', () => {
  it('offers a fullscreen keyboard-accessible call without a Hub voice control/provider', () => {
    const services = makeServices(liveState(), '/xerxes')
    const html = renderToStaticMarkup(<ServicesProvider services={services}><CallView onClose={() => undefined} /></ServicesProvider>)
    expect(html).toContain('role="dialog"')
    expect(html).toContain('aria-modal="true"')
    expect(html).toContain('sheet-fullscreen')
    expect(html).toContain('Call XERXES')
    expect(html).toContain('downloads ~160 MB once')
    expect(html).toContain(`Light · ${Math.round(tierDownloadBytes('light') / 1_000_000)} MB`)
    expect(html).toContain('>Mute</button>')
    expect(html).toContain('>Hang up</button>')
    expect(html).toContain('aria-label="Live call captions"')
    expect(html).toContain('No Hub voice provider is needed')
  })
  it('keeps the regular Xerxes composer and talk button behavior alongside the Call entry', () => {
    const state = liveState()
    const html = render({ ...state, supervisor: { ...state.supervisor, phase: 'ready' } }, '/xerxes')
    expect(html).toContain('>Call</button>')
    expect(html).toContain('aria-label="Message to XERXES"')
    expect(html).toContain('>Send</button>')
  })
})
