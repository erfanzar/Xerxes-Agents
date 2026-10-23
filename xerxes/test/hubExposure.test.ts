// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import {
  HubExposureError, exposureConfigFromEnvironment, isLoopbackAddress, isTailnetAddress, isWildcardAddress, originAllowed, resolveHubExposure,
} from '../src/hub/exposure.js'

const TLS = { cert: 'CERT', key: 'KEY' }

describe('Hub exposure', () => {
  test('defaults to loopback and derives exact local origins from the bound port', () => {
    const exposure = resolveHubExposure()
    expect(exposure).toMatchObject({ kind: 'loopback', hostname: '127.0.0.1', port: 8737 })
    expect(exposure.allowedOrigins(4321)).toEqual(['http://127.0.0.1:4321', 'http://[::1]:4321', 'http://localhost:4321'])
  })

  test('refuses wildcard bind addresses', () => {
    for (const host of ['0.0.0.0', '::', '[::]', '*', '', '  ']) {
      expect(() => resolveHubExposure({ host })).toThrow(HubExposureError)
      expect(isWildcardAddress(host)).toBe(true)
    }
    expect(() => resolveHubExposure({ host: '0.0.0.0', tls: TLS })).toThrow(/wildcard/)
  })

  test('refuses public, private-LAN and look-alike addresses even with TLS', () => {
    for (const host of ['8.8.8.8', '192.168.1.5', '10.0.0.1', '100.63.255.255', '100.128.0.1', '2001:db8::1', 'example.com', '::ffff:0.0.0.0', 'fd7b:115c:a1e0::1']) {
      expect(() => resolveHubExposure({ host, tls: TLS })).toThrow(HubExposureError)
    }
  })

  test('accepts the whole loopback family without TLS', () => {
    for (const host of ['127.0.0.1', '127.0.0.2', '::1', 'localhost', '::ffff:127.0.0.1']) {
      expect(isLoopbackAddress(host)).toBe(true)
      expect(resolveHubExposure({ host, port: 0 }).kind).toBe('loopback')
    }
    expect(resolveHubExposure({ host: 'localhost' }).hostname).toBe('127.0.0.1')
  })

  test('accepts a tailnet address only with TLS', () => {
    for (const host of ['100.64.0.1', '100.127.255.254', 'fd7a:115c:a1e0::1']) {
      expect(isTailnetAddress(host)).toBe(true)
      expect(() => resolveHubExposure({ host })).toThrow(/TLS/)
      const exposure = resolveHubExposure({ host, tls: TLS, port: 0 })
      expect(exposure.kind).toBe('tailnet')
      expect(exposure.tls).toEqual(TLS)
      const origins = exposure.allowedOrigins(9000)
      expect(origins.every(origin => origin.startsWith('https://'))).toBe(true)
      expect(origins).toHaveLength(1)
    }
  })

  test('computes the exact public origin list from configuration', () => {
    const exposure = resolveHubExposure({ tailnetHostname: 'Box.Tail1234.ts.net', publicOrigins: ['https://hub.example.com:8443/'], port: 0 })
    expect(exposure.publicOrigins).toEqual(['https://box.tail1234.ts.net', 'https://hub.example.com:8443'])
    const origins = exposure.allowedOrigins(5555)
    expect(origins).toContain('https://box.tail1234.ts.net')
    expect(origins).toContain('http://127.0.0.1:5555')
    expect(originAllowed('https://box.tail1234.ts.net', origins)).toBe(true)
    expect(originAllowed('https://box.tail1234.ts.net.evil.com', origins)).toBe(false)
    expect(originAllowed('http://box.tail1234.ts.net', origins)).toBe(false)
    expect(originAllowed(null, origins)).toBe(false)
    expect(originAllowed('', origins)).toBe(false)
  })

  test('rejects malformed, plaintext, wildcard and path-bearing public origins', () => {
    for (const origin of ['http://hub.example.com', 'https://*.example.com', 'https://hub.example.com/app', 'https://user:pw@hub.example.com', 'not a url', 'https://hub.example.com/?x=1']) {
      expect(() => resolveHubExposure({ publicOrigins: [origin] })).toThrow(HubExposureError)
    }
    expect(() => resolveHubExposure({ tailnetHostname: 'box.example.com' })).toThrow(/ts\.net/)
    expect(() => resolveHubExposure({ port: 70000 })).toThrow(HubExposureError)
  })

  test('reads the environment contract without trusting malformed input', () => {
    expect(exposureConfigFromEnvironment({ XERXES_HUB_HOST: '127.0.0.1', XERXES_HUB_PORT: '9000', XERXES_HUB_PUBLIC_ORIGINS: 'https://a.example.com, https://b.example.com', XERXES_HUB_TAILNET_HOSTNAME: 'x.ts.net' }, TLS)).toEqual({
      host: '127.0.0.1', port: 9000, tls: TLS, publicOrigins: ['https://a.example.com', 'https://b.example.com'], tailnetHostname: 'x.ts.net',
    })
    expect(exposureConfigFromEnvironment({})).toEqual({})
    expect(() => exposureConfigFromEnvironment({ XERXES_HUB_PORT: '80abc' })).toThrow(HubExposureError)
  })
})
