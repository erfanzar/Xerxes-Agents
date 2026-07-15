// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  HOME_ASSISTANT_TOOL_DEFINITIONS,
  HomeAssistantClient,
  registerHomeAssistantTools,
} from '../src/tools/homeAssistantTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

function result(value: string): JsonObject {
  return JSON.parse(value) as JsonObject
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    headers: { 'Content-Type': 'application/json' },
    status,
  })
}

test('Home Assistant tools use an explicit authenticated client and preserve entity/service filtering', async () => {
  const calls: Array<{
    readonly body: string | undefined
    readonly headers: Headers
    readonly method: string
    readonly url: string
  }> = []
  const states = [
    { entity_id: 'light.kitchen_main', state: 'off', attributes: { area_id: 'kitchen' } },
    { entity_id: 'light.living_room', state: 'on', attributes: { area_id: 'living_room' } },
    { entity_id: 'switch.coffee', state: 'off', attributes: { area_id: 'kitchen' } },
  ]
  const services = [
    { domain: 'light', services: { turn_on: { description: 'Turn on' }, turn_off: {} } },
    { domain: 'switch', services: { turn_on: {}, turn_off: {} } },
  ]
  const client = new HomeAssistantClient({
    baseUrl: 'https://hass.test/nested',
    token: 'explicit-token',
    fetchImplementation: async (input, init) => {
      const url = input.toString()
      calls.push({
        body: typeof init?.body === 'string' ? init.body : undefined,
        headers: new Headers(init?.headers),
        method: init?.method ?? 'GET',
        url,
      })
      if (url.endsWith('/api/states')) return jsonResponse(states)
      if (url.endsWith('/api/services')) return jsonResponse(services)
      if (url.endsWith('/api/states/light.kitchen_main')) {
        return jsonResponse({
          ...states[0],
          last_changed: '2026-07-13T12:00:00Z',
          last_updated: '2026-07-13T12:01:00Z',
        })
      }
      if (url.endsWith('/api/services/light/turn_on')) {
        return jsonResponse([{ entity_id: 'light.kitchen_main', state: 'on' }])
      }
      return jsonResponse({ error: 'not found' }, 404)
    },
  })
  const registry = new ToolRegistry()
  registerHomeAssistantTools(registry, { client })
  const context = { metadata: {} }

  const listed = result(await registry.execute(call('ha_list_entities', {
    area: 'kitchen',
    domain: 'light',
  }), context))
  expect(listed).toEqual({
    count: 1,
    entities: [{
      attributes: { area_id: 'kitchen' },
      entity_id: 'light.kitchen_main',
      state: 'off',
    }],
  })

  const listedServices = result(await registry.execute(call('ha_list_services', { domain: 'light' }), context))
  expect(listedServices.domains).toEqual([services[0]!])

  const state = result(await registry.execute(call('ha_get_state', {
    entity_id: 'light.kitchen_main',
  }), context))
  expect(state).toEqual({
    attributes: { area_id: 'kitchen' },
    entity_id: 'light.kitchen_main',
    last_changed: '2026-07-13T12:00:00Z',
    last_updated: '2026-07-13T12:01:00Z',
    state: 'off',
  })

  const changed = result(await registry.execute(call('ha_call_service', {
    data: { entity_id: 'light.kitchen_main' },
    domain: 'light',
    service: 'turn_on',
  }), context))
  expect(changed).toEqual({
    changed: [{ entity_id: 'light.kitchen_main', state: 'on' }],
    domain: 'light',
    ok: true,
    service: 'turn_on',
  })

  expect(calls).toHaveLength(4)
  expect(calls.every(entry => entry.url.startsWith('https://hass.test/nested/api/'))).toBeTrue()
  expect(calls.every(entry => entry.headers.get('Authorization') === 'Bearer explicit-token')).toBeTrue()
  const serviceRequest = calls.find(entry => entry.url.endsWith('/api/services/light/turn_on'))
  expect(serviceRequest?.method).toBe('POST')
  expect(serviceRequest?.body).toBe(JSON.stringify({ entity_id: 'light.kitchen_main' }))
})

test('Home Assistant tools return body-safe errors for not-found and service failures', async () => {
  const client = new HomeAssistantClient({
    baseUrl: 'https://hass.test',
    token: 'safe-token',
    fetchImplementation: async (input) => {
      const path = new URL(input.toString()).pathname
      if (path.includes('/api/states/')) {
        return jsonResponse({ message: 'entity does not exist' }, 404)
      }
      if (path.includes('/api/services/')) {
        return new Response('the secret token is safe-token', { status: 503 })
      }
      return jsonResponse([])
    },
  })
  const registry = new ToolRegistry()
  registerHomeAssistantTools(registry, { client })
  const context = { metadata: {} }

  const missing = result(await registry.execute(call('ha_get_state', {
    entity_id: 'light.does_not_exist',
  }), context))
  expect(missing).toEqual({ entity_id: 'light.does_not_exist', error: 'not_found' })

  const failed = result(await registry.execute(call('ha_call_service', {
    domain: 'light',
    service: 'turn_on',
  }), context))
  expect(failed).toEqual({
    domain: 'light',
    error: 'request_failed',
    ok: false,
    service: 'turn_on',
  })
  expect(JSON.stringify(failed)).not.toContain('safe-token')
})

test('Home Assistant credentials are always explicit and registry definitions remain host-bound', async () => {
  const previousToken = process.env.HASS_TOKEN
  const previousBaseUrl = process.env.HASS_BASE_URL
  process.env.HASS_TOKEN = 'environment-token'
  process.env.HASS_BASE_URL = 'https://environment-hass.test'
  try {
    expect(() => new HomeAssistantClient({
      baseUrl: 'https://hass.test',
      fetchImplementation: async () => jsonResponse([]),
    })).toThrow(ConfigurationError)

    let authorization: string | null = null
    const localClient = new HomeAssistantClient({
      allowUnauthenticated: true,
      baseUrl: 'http://localhost:8123',
      fetchImplementation: async (_input, init) => {
        authorization = new Headers(init?.headers).get('Authorization')
        return jsonResponse([])
      },
    })
    await localClient.listStates()
    expect(authorization).toBeNull()

    const registry = new ToolRegistry()
    registerHomeAssistantTools(registry, { client: localClient })
    expect(registry.definitions().map(definition => definition.function.name).sort()).toEqual([
      'ha_call_service',
      'ha_get_state',
      'ha_list_entities',
      'ha_list_services',
    ])
    expect(HOME_ASSISTANT_TOOL_DEFINITIONS.map(definition => definition.function.name)).toEqual([
      'ha_list_entities',
      'ha_list_services',
      'ha_get_state',
      'ha_call_service',
    ])
  } finally {
    if (previousToken === undefined) delete process.env.HASS_TOKEN
    else process.env.HASS_TOKEN = previousToken
    if (previousBaseUrl === undefined) delete process.env.HASS_BASE_URL
    else process.env.HASS_BASE_URL = previousBaseUrl
  }
})
