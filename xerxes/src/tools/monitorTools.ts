// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ToolRegistry, ToolExecutionContext } from '../executors/toolRegistry.js'
import type { TerminalMonitors } from '../runtime/terminalMonitors.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, requiredString } from './inputs.js'

function owner(context: ToolExecutionContext): string {
  if (!context.sessionId?.trim()) throw new Error('Monitor tools require an authenticated session context')
  return context.sessionId
}
export function registerMonitorTools(registry: ToolRegistry, monitors: TerminalMonitors): void {
  const definitions: ToolDefinition[] = [
    { type: 'function', function: { name: 'monitor_terminal', description: 'Watch a terminal owned by this session. Use trigger=completion to receive its exit result and final output once (including if it just finished); match is then unnecessary. Otherwise watch future output for a case-insensitive literal match. Matching lines are deduplicated, recorded in Runs, and announced to attached users without polling the model. Set react=true only when the user requested automatic investigation or action on matches. This starts bounded background model turns; notification-only is the default. The watch expires or stops after max_events; stopping it does not kill the command. A watch on a success marker stays silent if the process crashes, and silence is not success: to learn how a run ended, add a trigger=completion watch.', parameters: { type: 'object', additionalProperties: false, properties: {
      terminal_id: { type: 'string' }, trigger: { type: 'string', enum: ['output', 'completion'], default: 'output' }, match: { type: 'string', minLength: 1, maxLength: 1024 },
      duration_seconds: { type: 'integer', minimum: 1, maximum: 86400, default: 3600 },
      max_events: { type: 'integer', minimum: 1, maximum: 1000, default: 50 },
      react: { type: 'boolean', default: false },
      max_total_tokens: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER, description: 'Lifetime measured-token admission threshold across reactions, including child and auxiliary calls. Requires react=true. In-flight calls may overshoot; unknown usage blocks further reactions.' },
      max_reactions: { type: 'integer', minimum: 1, maximum: 10, default: 3 },
      reaction_timeout_seconds: { type: 'integer', minimum: 1, maximum: 120, default: 60 },
    }, required: ['terminal_id'] } } },
    { type: 'function', function: { name: 'monitor_file', description: 'Watch metadata changes to one existing regular file inside this session workspace. Records bounded changed/deleted/recreated events without reading contents or polling the model. Atomic file replacement is observed; rapid changes may coalesce. Files outside the workspace and symlink escapes are rejected. After daemon restart, the interrupted watch reports an observation gap; create a new watch to continue. Set react=true only for user-requested automatic investigation, subject to bounded reaction budgets.', parameters: { type: 'object', additionalProperties: false, properties: {
      file_path: { type: 'string', minLength: 1, maxLength: 8192 },
      duration_seconds: { type: 'integer', minimum: 1, maximum: 86400, default: 3600 }, max_events: { type: 'integer', minimum: 1, maximum: 1000, default: 50 },
      react: { type: 'boolean', default: false }, max_reactions: { type: 'integer', minimum: 1, maximum: 10, default: 3 },
      reaction_timeout_seconds: { type: 'integer', minimum: 1, maximum: 120, default: 60 }, max_total_tokens: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
    }, required: ['file_path'] } } },
    { type: 'function', function: { name: 'monitor_websocket', description: 'Watch a server-pushed WebSocket text feed for a case-insensitive literal match. Matching messages are deduplicated and recorded in Runs without polling the model. Supports wss:// or loopback ws://; URL credentials, queries and fragments are rejected. No authentication headers or application subscription messages are sent. Frames are bounded; binary messages fail visibly. Reconnects are bounded and record observation gaps, because missed messages cannot be recovered generically. Set react=true only for user-requested automatic investigation, subject to reaction limits.', parameters: { type: 'object', additionalProperties: false, properties: {
      websocket_url: { type: 'string', minLength: 1, maxLength: 4096 }, match: { type: 'string', minLength: 1, maxLength: 1024 },
      duration_seconds: { type: 'integer', minimum: 1, maximum: 86400, default: 3600 }, max_events: { type: 'integer', minimum: 1, maximum: 1000, default: 50 },
      react: { type: 'boolean', default: false }, max_reactions: { type: 'integer', minimum: 1, maximum: 10, default: 3 },
      reaction_timeout_seconds: { type: 'integer', minimum: 1, maximum: 120, default: 60 }, max_total_tokens: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
    }, required: ['websocket_url', 'match'] } } },
    { type: 'function', function: { name: 'list_monitor_sources', description: 'List named webhook sources configured by this host. Returns names only, never authentication secrets. Use a configured name with monitor_webhook.', parameters: { type: 'object', additionalProperties: false, properties: {} } } },
    { type: 'function', function: { name: 'monitor_webhook', description: 'Watch a named, host-configured authenticated webhook source for a case-insensitive literal match. Call list_monitor_sources first to discover names. Only signed deliveries are accepted; matching events are recorded without polling the model. Deliveries while detached or offline are not recovered. Secrets are configured through host environment variables, never tool arguments. Set react=true only for user-requested automatic investigation; notification-only is the default.', parameters: { type: 'object', additionalProperties: false, properties: {
      webhook_name: { type: 'string', pattern: '^[a-zA-Z0-9_-]{1,64}$' }, match: { type: 'string', minLength: 1, maxLength: 1024 },
      duration_seconds: { type: 'integer', minimum: 1, maximum: 86400, default: 3600 }, max_events: { type: 'integer', minimum: 1, maximum: 1000, default: 50 },
      react: { type: 'boolean', default: false }, max_reactions: { type: 'integer', minimum: 1, maximum: 10, default: 3 },
      reaction_timeout_seconds: { type: 'integer', minimum: 1, maximum: 120, default: 60 }, max_total_tokens: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
    }, required: ['webhook_name', 'match'] } } },
    { type: 'function', function: { name: 'list_monitors', description: 'Inspect this session’s terminal, file, WebSocket and webhook watches and retained events. Do not poll this in a loop: monitors react to source events automatically.', parameters: { type: 'object', additionalProperties: false, properties: {} } } },
    { type: 'function', function: { name: 'stop_monitor', description: 'Stop one watch owned by this session without stopping its source process.', parameters: { type: 'object', additionalProperties: false, properties: { monitor_id: { type: 'string' } }, required: ['monitor_id'] } } },
  ]
  for (const definition of definitions) registry.register(definition, async (args, context) => {
    const session = owner(context)
    if (args.max_total_tokens !== undefined && args.react !== true) throw new Error('Token threshold requires automatic reactions')
    if (args.react === true && context.metadata.goal_turn_human !== true) throw new Error('Automatic reactions require a direct user turn')
    const trigger = (optionalString(args, 'trigger') ?? 'output')
    if (trigger !== 'output' && trigger !== 'completion') throw new Error('Invalid monitor trigger')
    switch (definition.function.name) {
      case 'list_monitor_sources': return JSON.stringify({ webhooks: monitors.webhookSources(session) })
      case 'monitor_webhook': return JSON.stringify(await monitors.startWebhook(session, { name: requiredString(args, 'webhook_name'), match: requiredString(args, 'match'),
        durationMs: optionalInteger(args, 'duration_seconds', 3600) * 1000, maxEvents: optionalInteger(args, 'max_events', 50),
        ...(args.react === true ? { reaction: { ...(args.max_total_tokens === undefined ? {} : { maxTotalTokens: optionalInteger(args, 'max_total_tokens', 1) }), maxReactions: optionalInteger(args, 'max_reactions', 3), maxDurationMs: optionalInteger(args, 'reaction_timeout_seconds', 60) * 1000 } } : {}) }))
      case 'list_monitors': return JSON.stringify(monitors.list(session))
      case 'stop_monitor': return JSON.stringify(monitors.stop(session, requiredString(args, 'monitor_id')))
      case 'monitor_websocket': return JSON.stringify(await monitors.startWebSocket(session, { url: requiredString(args, 'websocket_url'), match: requiredString(args, 'match'),
        durationMs: optionalInteger(args, 'duration_seconds', 3600) * 1000, maxEvents: optionalInteger(args, 'max_events', 50),
        ...(args.react === true ? { reaction: { ...(args.max_total_tokens === undefined ? {} : { maxTotalTokens: optionalInteger(args, 'max_total_tokens', 1) }), maxReactions: optionalInteger(args, 'max_reactions', 3), maxDurationMs: optionalInteger(args, 'reaction_timeout_seconds', 60) * 1000 } } : {}) }))
      case 'monitor_file': return JSON.stringify(await monitors.startFile(session, { path: requiredString(args, 'file_path'),
        durationMs: optionalInteger(args, 'duration_seconds', 3600) * 1000, maxEvents: optionalInteger(args, 'max_events', 50),
        ...(args.react === true ? { reaction: { ...(args.max_total_tokens === undefined ? {} : { maxTotalTokens: optionalInteger(args, 'max_total_tokens', 1) }), maxReactions: optionalInteger(args, 'max_reactions', 3), maxDurationMs: optionalInteger(args, 'reaction_timeout_seconds', 60) * 1000 } } : {}) }))
      default: return JSON.stringify(monitors.start(session, { terminalId: requiredString(args, 'terminal_id'), trigger, match: (optionalString(args, 'match') ?? ''),
        durationMs: optionalInteger(args, 'duration_seconds', 3600) * 1000, maxEvents: optionalInteger(args, 'max_events', 50),
        ...(args.react === true ? { reaction: { ...(args.max_total_tokens === undefined ? {} : { maxTotalTokens: optionalInteger(args, 'max_total_tokens', 1) }), maxReactions: optionalInteger(args, 'max_reactions', 3), maxDurationMs: optionalInteger(args, 'reaction_timeout_seconds', 60) * 1000 } } : {}) }))
    }
  }, 'default', { concurrencySafe: true, destructive: false, openWorld: ['monitor_websocket', 'monitor_webhook'].includes(definition.function.name), readOnly: ['list_monitors', 'list_monitor_sources'].includes(definition.function.name), maxResultBytes: 64_000 })
}
