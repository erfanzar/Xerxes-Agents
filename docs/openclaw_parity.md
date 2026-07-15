# Capability parity roadmap

Xerxes evaluates adjacent agent platforms by observable capabilities, not by copying their internal
architecture or branding. A native capability is considered complete only when it has a typed
implementation, focused tests, and a verified entrypoint.

| Capability | Native direction | Status rule |
| --- | --- | --- |
| Streaming agent turns | Async stream events and tool loop | Verify cancellation, tool pairing, and terminal errors. |
| Durable sessions | Native store, search, replay, export | Verify persisted format and recovery. |
| Skills and plugins | Markdown skill registry and typed hooks | Verify discovery, guardrails, and lifecycle. |
| Multi-agent orchestration | Cortex topology engine | Verify delegation and result aggregation. |
| Chat channels | Configured adapter lifecycle | Verify each direct transport separately; relay/host ports are not direct parity. |
| Browser and computer use | Explicit provider contracts | Require an injected real adapter; never simulate an action. |
| External memory, media, training | Typed integration ports | Require explicit credentials, hardware, or provider client. |

The roadmap intentionally rejects a broad “supported” label for an unimplemented remote transport.
When a host boundary is not configured, the native API must return an explicit unavailable result or
skip an opt-in integration test.
