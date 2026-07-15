---
name: add-channel-adapter
description: Add a Bun-native messaging channel adapter with lifecycle, webhook, and configuration coverage.
version: 2.0.0
tags: [channels, adapter, messaging, typescript, bun, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when adding a native messaging platform adapter or webhook relay
to Xerxes. An adapter normalizes inbound data into `ChannelMessage`, sends
outbound messages, and has an explicit asynchronous lifecycle.

Do not use this skill for a one-off web client, a provider adapter, or a
frontend-only integration.

# How to use

## 1. Inspect the native channel contracts

Read these files before writing an adapter:

- `xerxes/src/channels/base.ts` for `Channel` and `InboundHandler`.
- `xerxes/src/channels/types.ts` for `ChannelMessage` and
  `createChannelMessage()`.
- `xerxes/src/channels/webhooks.ts` for `WebhookChannel` and raw-byte
  webhook handling.
- A comparable adapter such as `telegram.ts`, `slack.ts`, or
  `genericWebhook.ts`.

Every adapter implements this lifecycle:

```ts
interface Channel {
  readonly name: string
  start(onInbound: InboundHandler): Promise<void>
  stop(): Promise<void>
  send(message: ChannelMessage): Promise<void>
}
```

Use `WebhookChannel` when the provider pushes raw HTTP callbacks. It preserves
the original headers and bytes for signature validation, normalizes errors into
safe HTTP responses, and dispatches parsed messages through the installed
inbound handler.

## 2. Implement a concrete TypeScript adapter

Create a descriptive camel-case module under `xerxes/src/channels/`,
for example `myPlatform.ts`. Start it with the Apache-2.0 header used by nearby
native modules.

```ts
import type { Channel, InboundHandler } from './base.js'
import type { ChannelMessage } from './types.js'

export class MyPlatformChannel implements Channel {
  readonly name = 'my_platform'
  #onInbound: InboundHandler | undefined

  async start(onInbound: InboundHandler): Promise<void> {
    this.#onInbound = onInbound
    // Start only the configured platform transport.
  }

  async stop(): Promise<void> {
    this.#onInbound = undefined
    // Close owned sockets or timers.
  }

  async send(message: ChannelMessage): Promise<void> {
    // Send a normalized outbound message with an explicit HTTP/WebSocket port.
  }
}
```

Use injected `fetch`, WebSocket, REST, or SDK ports where tests need a
deterministic fake. Validate provider payloads at the adapter boundary. Do not
discover credentials, silently start an unrelated daemon, or turn an unconfigured
transport into a success response.

## 3. Register configuration deliberately

For a built-in configured channel, add its constructor to the `switch` in
`xerxes/src/channels/configured.ts`, then export the module from
`xerxes/src/channels/index.ts`. Use the `ConfiguredChannelManager`
contract so `channel.list`, `channel.enable`, and `channel.disable` expose
configuration and lifecycle errors accurately.

If the adapter needs host-owned non-serializable ports, extend
`ConfiguredChannelTransportPorts` or supply a `ConfiguredChannelFactory` from
`xerxes/src/daemon/channels.ts`. Keep credentials in the resolved
daemon configuration or explicitly supplied host ports, never in source or
tests.

## 4. Add focused Bun tests

Place tests under `xerxes/test/` alongside the closest channel suite.
Cover observable behavior:

- `start()` and `stop()` lifecycle,
- inbound normalization and `ChannelMessage` ownership,
- outbound serialization and error propagation,
- webhook signature/raw-body behavior when applicable, and
- configured manager enable/disable behavior if the adapter is built in.

Mock every external HTTP or WebSocket call. Do not contact a real platform from
the test suite.

Run the relevant suite, for example:

```bash
bun test xerxes/test/channelAdapters.test.ts
bun test xerxes/test/channels.test.ts
bun run --cwd xerxes check
```

## Common pitfalls

- Adapter `name` must be stable and match the configured channel type or
  registry name that users select.
- Never block `start()` while waiting for a remote message. Start a managed
  asynchronous loop and stop it cleanly.
- Preserve raw webhook bytes for verification; parsing before signature checks
  can invalidate the provider contract.
- Copy message metadata and attachments rather than sharing mutable input
  objects across turns.
- A configured but unavailable transport must be listed with its real error;
  it must not be replaced with a fabricated relay or fallback.
