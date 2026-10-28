// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { InstallHint } from '../components/InstallHint.js'
import { liveState } from './helpers.js'
import { render } from './renderHelpers.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '')

describe('install hint', () => {
  it('tells an iPhone browser tab to Add to Home Screen and pair from the installed app', () => {
    const html = plain(renderToStaticMarkup(<InstallHint env={{ standalone: false, ios: true, dismissed: false }} />))
    expect(html).toContain('Install Xerxes as an app.')
    expect(html).toContain('<strong>Add to Home Screen</strong>')
    expect(html).toContain('Safari keeps a separate sign-in')
    expect(html).toContain('>Not now</button>')
  })
  it('uses the browser install wording off iOS', () => {
    const html = plain(renderToStaticMarkup(<InstallHint env={{ standalone: false, ios: false, dismissed: false }} />))
    expect(html).toContain('<strong>Install app</strong>')
  })
  it('never shows inside the installed app or after it was dismissed', () => {
    expect(renderToStaticMarkup(<InstallHint env={{ standalone: true, ios: true, dismissed: false }} />)).toBe('')
    expect(renderToStaticMarkup(<InstallHint env={{ standalone: false, ios: true, dismissed: true }} />)).toBe('')
  })
  it('is part of the app shell when running in a browser tab', () => {
    const html = plain(render(liveState(), '/', { install: { standalone: false, ios: true, dismissed: false } }))
    expect(html).toContain('aria-label="Install Xerxes"')
    const installed = plain(render(liveState(), '/', { install: { standalone: true, ios: true, dismissed: false } }))
    expect(installed).not.toContain('aria-label="Install Xerxes"')
  })
})
