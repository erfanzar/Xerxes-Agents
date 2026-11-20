// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createSlackBroker } from './broker.js'

if (import.meta.main) {
  try {
    const env = Bun.env
    const portText = env.XERXES_SLACK_BROKER_PORT ?? '8796'
    const port = Number(portText)
    if (!/^[0-9]{1,5}$/u.test(portText) || port < 1 || port > 65535) throw new Error()
    const broker = createSlackBroker({
      publicOrigin: env.XERXES_SLACK_BROKER_ORIGIN ?? '',
      clientId: env.XERXES_SLACK_BROKER_CLIENT_ID ?? '',
      clientSecret: env.XERXES_SLACK_BROKER_CLIENT_SECRET ?? '',
      enrollmentToken: env.XERXES_SLACK_BROKER_ENROLLMENT_TOKEN ?? '',
      fetch: (url, init) => fetch(url, init),
    })
    const server = Bun.serve({
      hostname: '127.0.0.1', port, idleTimeout: 30, maxRequestBodySize: 1024,
      fetch: request => broker.handle(request),
      error: () => new Response('Slack broker unavailable.', { status: 500, headers: { 'cache-control': 'no-store' } }),
    })
    setInterval(broker.sweep, 15_000).unref()
    console.info(`Slack OAuth broker listening on loopback port ${server.port}.`)
  } catch {
    console.error('Slack OAuth broker could not start. Check the HTTPS origin, enrollment token, Slack app credentials, and loopback port.')
    process.exitCode = 1
  }
}
