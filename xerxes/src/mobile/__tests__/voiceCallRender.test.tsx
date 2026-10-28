// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { tierDownloadBytes } from '../../client/voiceModelManifest.js'
import { createDebugCapture } from '../voice/call/debugCapture.js'
import type { CallSnapshot } from '../voice/call/engine.js'
import { MobileApp } from '../MobileApp.js'
import { voiceStateLine } from '../voice/call/presentation.js'
import { liveState, withSupervisorConfig } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'
import { voiceServiceFixture } from './voiceCallServiceFakes.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")
const ready = () => withSupervisorConfig(liveState(), { status: 'configured' })

describe('app-level voice scene', () => {
  it('puts the prominent waveform voice button next to Send inside the composer, not in the model toolbar', () => {
    const html = plain(render(ready(), '/xerxes'))
    const composer = html.slice(html.indexOf('<form class="composer"'), html.indexOf('</form>'))
    expect(composer).toContain('aria-label="Start voice mode"')
    expect(composer).toContain('class="composer-voice"')
    expect(composer).toContain('<svg')
    expect(composer).toContain('>Send</button>')
    expect(html).not.toContain('>Call</button>')
    expect(html).not.toContain('voice-screen')
  })
  it('has a dark full-screen orb scene, accessible controls, settings tiers and first-download notice, not a sheet', async () => {
    const f = voiceServiceFixture()
    await f.service.loadStatus()
    f.service.open()
    const services = makeServices(ready(), '/xerxes', { call: f.service })
    const html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html).toContain('role="dialog" aria-modal="true"')
    expect(html).toContain('class="voice-screen')
    expect(html).toContain('class="voice-orb"')
    expect(html).not.toContain('sheet-fullscreen')
    expect(html).toContain('aria-label="Voice settings"')
    // Sizes come from the manifest, so a model change can never leave stale text behind.
    expect(html).toContain(`This call downloads ~${Math.round(tierDownloadBytes('standard') / 1_000_000)} MB once`)
    expect(html).toContain('Moonshine Streaming small + Kokoro')
    expect(html).toContain(`Light · ${Math.round(tierDownloadBytes('light') / 1_000_000)} MB`)
    expect(html).toContain('>Mute</span>')
    expect(html).toContain('>Chat</span>')
    expect(html).toContain('aria-label="End voice call"')
    expect(html).toContain('aria-label="Live call captions"')
    expect(html).toContain('No Hub voice provider is needed')
    f.service.dispose()
  })
  it.each([
    ['loading', 'Downloading voice (80 MB of 162 MB)'], ['listening', 'Listening'], ['hearing', 'Listening'],
    ['transcribing', 'Thinking'], ['thinking', 'Thinking'], ['speaking', 'Speaking'],
  ] satisfies readonly [CallSnapshot['phase'], string][])('renders %s with short captions and the appropriate state', async (phase, expected) => {
    const f = voiceServiceFixture()
    await f.service.loadStatus()
    await f.service.start()
    f.emit({ phase, progress: { loaded: 80_000_000, total: 162_000_000 }, heard: 'What needs my attention?', spokenSentence: phase === 'speaking' ? 'Your build is ready.' : '', inputLevel: 0.2, outputLevel: 0.7 })
    const html = plain(renderToStaticMarkup(<MobileApp services={makeServices(ready(), '/xerxes', { call: f.service })} />))
    expect(html).toContain(`>${expected}</h2>`)
    expect(html).toContain('What needs my attention?')
    if (phase === 'speaking') {
      expect(html).toContain('Your build is ready.')
      expect(html).toContain('aria-label="Interrupt XERXES"')
      expect(html).toContain('--voice-level:0.7')
    }
    f.service.dispose()
  })
  it('minimizes to a pill with a return control, survives navigation, returns to the orb and only End hangs up', async () => {
    const f = voiceServiceFixture()
    await f.service.loadStatus()
    await f.service.start()
    const services = makeServices(ready(), '/xerxes', { call: f.service })
    f.service.minimize()
    let html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html).not.toContain('class="voice-screen')
    expect(html).toContain('aria-label="Voice call in progress"')
    expect(html).toContain('aria-label="Return to voice mode — Listening"')
    expect(html).toContain('voice-live-dot')
    expect(html).toContain('aria-label="Message to XERXES"')
    expect(html).toContain('>Send</button>')
    expect(f.engine.hangup).not.toHaveBeenCalled()
    services.router.navigate({ name: 'usage' })
    renderToStaticMarkup(<MobileApp services={services} />)
    expect(f.engine.hangup).not.toHaveBeenCalled()
    services.router.navigate({ name: 'xerxes' })
    f.service.open()
    html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html).toContain('class="voice-screen')
    expect(html).not.toContain('aria-label="Voice call in progress"')
    expect(f.engine.start).toHaveBeenCalledOnce()
    expect(f.engine.hangup).not.toHaveBeenCalled()
    f.service.end()
    html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html).not.toContain('class="voice-screen')
    expect(html).not.toContain('aria-label="Voice call in progress"')
    expect(f.engine.hangup).toHaveBeenCalledOnce()
    f.service.dispose()
  })
  it('renders the debug capture preference off by default, with local-only retention text and no capture panel', async () => {
    const f = voiceServiceFixture()
    await f.service.loadStatus()
    f.service.open()
    const html = plain(renderToStaticMarkup(<MobileApp services={makeServices(ready(), '/xerxes', { call: f.service })} />))
    expect(html).toContain('Keep last utterance for debugging')
    expect(html.match(/<input[^>]*type="checkbox"[^>]*>/u)?.[0]).not.toContain('checked')
    expect(html).toContain('Audio is never uploaded')
    expect(html).not.toContain('aria-label="Last utterance capture"')
    expect(html).not.toContain('>Save WAV</button>')
    f.service.dispose()
  })
  it('renders the enabled capture transcript, precise duration/levels, VAD offsets and local Play/Save controls, then removes it when disabled', async () => {
    const f = voiceServiceFixture()
    f.service.setDebugCaptureEnabled(true)
    await f.service.loadStatus()
    await f.service.start()
    const capture = { ...createDebugCapture(new Float32Array(16000).fill(0.25), { startSample: 3200, endSample: 12000, preRollSamples: 3200 }), transcript: 'Check <this> sentence.' }
    f.emit({ debugCapture: capture })
    const services = makeServices(ready(), '/xerxes', { call: f.service })
    let html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html.match(/<input[^>]*type="checkbox"[^>]*>/u)?.[0]).toContain('checked')
    expect(html).toContain('aria-label="Last utterance capture"')
    expect(html).toContain('Check &lt;this&gt; sentence.')
    expect(html).toContain('1.000 s · 16 kHz mono · 16,000 samples')
    expect(html).toContain('Peak 0.2500 · RMS 0.2500')
    expect(html).toContain('VAD start 0.200 s · end 0.750 s · pre-roll 0.200 s')
    expect(html).toContain('>Play</button>')
    expect(html).toContain('>Save WAV</button>')
    f.service.setDebugCaptureEnabled(false)
    html = plain(renderToStaticMarkup(<MobileApp services={services} />))
    expect(html).not.toContain('aria-label="Last utterance capture"')
    expect(f.service.getSnapshot().call.debugCapture).toBeUndefined()
    expect(f.engine.hangup).not.toHaveBeenCalled()
    f.service.dispose()
  })
  it('renders Connecting without ending or disposing the call', async () => {
    const f = voiceServiceFixture()
    await f.service.loadStatus()
    await f.service.start()
    f.service.setConnected(false)
    const html = plain(renderToStaticMarkup(<MobileApp services={makeServices(ready(), '/xerxes', { call: f.service })} />))
    expect(voiceStateLine(f.service.getSnapshot())).toBe('Connecting')
    expect(html).toContain('>Connecting</h2>')
    expect(f.engine.hangup).not.toHaveBeenCalled()
    f.service.dispose()
  })
})
