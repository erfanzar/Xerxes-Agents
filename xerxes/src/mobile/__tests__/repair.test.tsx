// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import jsQR from 'jsqr'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { encodeQr } from '../../hub/identity/qr.js'
import { SignedOut } from '../components/SignedOut.js'
import { checkPairLink, PAIR_LINK_MAX_LENGTH } from '../repairLink.js'
import { cameraErrorMessage } from '../repairScanner.js'
import { liveState } from './helpers.js'
import { render } from './renderHelpers.js'

const ORIGIN = 'https://xerxes-hub.example.ts.net'
const SECRET = 'c0de-SECRET-value-123'
const ticket = (payload: Record<string, string>): string => Buffer.from(JSON.stringify(payload)).toString('base64url')
const GOOD = `${ORIGIN}/pair#${ticket({ hub_id: 'hubid', pairing_id: 'pair_abc', code: SECRET })}`
const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'").replaceAll('&rsquo;', '’')

describe('checkPairLink', () => {
  it('accepts this Hub’s pairing link and rebuilds it from the trusted origin', () => {
    const check = checkPairLink(`  ${GOOD}\n`, ORIGIN)
    expect(check).toEqual({ ok: true, href: GOOD })
  })
  it('accepts the query-style fragment the pair page also understands', () => {
    expect(checkPairLink(`${ORIGIN}/pair#pairing_id=p1&code=${SECRET}`, ORIGIN).ok).toBe(true)
  })
  it('refuses a link for another host without repeating the secret', () => {
    const check = checkPairLink(GOOD.replace(ORIGIN, 'https://evil.example'), ORIGIN)
    expect(check.ok).toBe(false)
    if (!check.ok) {
      expect(check.message).toContain('different Hub (evil.example)')
      expect(check.message).not.toContain(SECRET)
    }
  })
  it.each([
    ['empty', '   '],
    ['not a URL', 'hello there'],
    ['javascript URL', 'javascript:alert(1)'],
    ['credentials in URL', GOOD.replace('https://', 'https://user:pw@')],
    ['wrong path', GOOD.replace('/pair#', '/invite#')],
    ['path with suffix', GOOD.replace('/pair#', '/pair/x#')],
    ['no ticket', `${ORIGIN}/pair`],
    ['broken ticket', `${ORIGIN}/pair#${ticket({ pairing_id: 'p1' })}`],
    ['too long', `${ORIGIN}/pair#${'a'.repeat(PAIR_LINK_MAX_LENGTH)}`],
    ['http on an https Hub', GOOD.replace('https://', 'http://')],
  ])('refuses %s', (_name, raw) => {
    const check = checkPairLink(raw, ORIGIN)
    expect(check.ok).toBe(false)
    if (!check.ok) expect(check.message).not.toContain(SECRET)
  })
})

describe('QR round trip', () => {
  it('decodes a QR code made by the Hub’s own encoder into a link the phone accepts', () => {
    const qr = encodeQr(GOOD)
    const scale = 4
    const quiet = 4
    const side = (qr.size + quiet * 2) * scale
    const pixels = new Uint8ClampedArray(side * side * 4).fill(255)
    for (let y = 0; y < qr.size; y += 1) {
      for (let x = 0; x < qr.size; x += 1) {
        if (qr.modules[y]?.[x] !== true) continue
        for (let dy = 0; dy < scale; dy += 1) {
          for (let dx = 0; dx < scale; dx += 1) {
            const index = (((y + quiet) * scale + dy) * side + (x + quiet) * scale + dx) * 4
            pixels[index] = 0; pixels[index + 1] = 0; pixels[index + 2] = 0
          }
        }
      }
    }
    const decoded = jsQR(pixels, side, side)
    expect(decoded?.data).toBe(GOOD)
    expect(checkPairLink(decoded?.data ?? '', ORIGIN)).toEqual({ ok: true, href: GOOD })
  })
})

describe('camera errors', () => {
  it.each([
    ['NotAllowedError', 'Allow the camera'],
    ['NotFoundError', 'No camera was found'],
    ['NotReadableError', 'busy in another app'],
    ['NotSupportedError', 'not available here'],
  ])('%s gives an actionable message with the paste fallback', (name, expected) => {
    const message = cameraErrorMessage(new DOMException('x', name))
    expect(message).toContain(expected)
    if (name !== 'NotReadableError') expect(message).toContain('aste')
  })
})

describe('signed-out screen', () => {
  it('offers in-app scanning, a paste fallback and explains why not to use the Camera app', () => {
    const html = plain(renderToStaticMarkup(<SignedOut reason="Signed out. Pair this device again." openLink={() => undefined} />))
    expect(html).toContain('>Signed out</h1>')
    expect(html).toContain('Signed out. Pair this device again.')
    expect(html).toContain('>Scan pairing QR code</button>')
    expect(html).toContain('Paste the pairing link instead')
    expect(html).toContain('>Open link</button>')
    expect(html).toContain('Camera app opens Safari')
    expect(html).toContain('<strong>Pair a phone</strong>')
    expect(html).not.toContain('<video')
  })
  it('replaces the tabs’ content when the connection is revoked', () => {
    const html = plain(render({ ...liveState(), connection: { state: 'revoked', reason: 'Signed out. Pair this device again.' } }, '/'))
    expect(html).toContain('>Scan pairing QR code</button>')
    expect(html).toContain('will not reconnect on its own')
  })
})
