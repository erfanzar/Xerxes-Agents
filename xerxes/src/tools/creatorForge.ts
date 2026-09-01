// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolRegistry } from '../executors/toolRegistry.js'
import {
  DeclarativeToolForge,
  recordCreatorTrace,
  type DeclarativeForgeDefinition,
} from '../extensions/declarativeForge.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export const CREATOR_FORGE_TOOL_NAME = 'CreatorForgeTool'

export const CREATOR_FORGE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: CREATOR_FORGE_TOOL_NAME,
    description: [
      'Define, inspect, run, or remove an immutable declarative text tool.',
      'Forged tools only render bounded templates from declared scalar inputs;',
      'they cannot execute code or access files, processes, or the network.',
    ].join(' '),
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['list', 'inspect', 'define', 'run', 'undefine'],
        },
        name: { type: 'string', description: 'Lowercase forged package name.' },
        version: { type: 'string', description: 'Immutable semantic version, e.g. 0.1.0.' },
        description: { type: 'string', description: 'What the forged package does.' },
        template: { type: 'string', description: 'Text template with {{parameter}} placeholders.' },
        parameters: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              description: { type: 'string' },
              required: { type: 'boolean' },
              default: { type: ['string', 'number', 'boolean', 'null'] },
            },
            required: ['name'],
            additionalProperties: false,
          },
        },
        input: {
          type: 'object',
          description: 'Scalar values supplied when action is run.',
          additionalProperties: true,
        },
      },
      required: ['action'],
      additionalProperties: false,
    },
  },
}

export function registerCreatorForgeTool(
  registry: ToolRegistry,
  forge: DeclarativeToolForge,
  agentId = 'default',
): void {
  registry.register(
    CREATOR_FORGE_DEFINITION,
    (inputs, context) => creatorForgeTool(inputs, context.metadata, forge),
    agentId,
    {
      concurrencySafe: false,
      defer: true,
      destructive: true,
      interruptBehavior: 'cancel',
      maxResultBytes: 65_536,
      openWorld: false,
      readOnly: false,
    },
    [
      'Creator mode is declarative and version-immutable.',
      'Use define only after the user asks to create a reusable tool.',
      'Never claim a forged package can perform host side effects; it only renders text.',
      'Define and undefine are policy-gated tool calls and remain disallowed in plan mode.',
    ].join('\n'),
  )
}

export function creatorForgeTool(
  inputs: JsonObject,
  metadata: Record<string, unknown>,
  forge: DeclarativeToolForge,
): unknown {
  const action = text(inputs.action)
  const name = text(inputs.name)
  const version = text(inputs.version)
  try {
    if ((action === 'define' || action === 'undefine') && metadata.permission_mode === 'plan') {
      throw new Error(`CreatorForgeTool ${action} is disabled in plan mode`)
    }
    let result: unknown
    switch (action) {
      case 'list':
        result = { ok: true, packages: forge.list().map(summary) }
        break
      case 'inspect': {
        const pkg = forge.inspect(name, version || undefined)
        result = pkg ? { ok: true, package: pkg } : { ok: false, error: 'forged package not found' }
        break
      }
      case 'define': {
        const definition: DeclarativeForgeDefinition = {
          name,
          version,
          description: text(inputs.description),
          template: typeof inputs.template === 'string' ? inputs.template : '',
          parameters: parameterRecords(inputs.parameters),
        }
        const pkg = forge.define(definition)
        result = { ok: true, package: summary(pkg) }
        break
      }
      case 'run':
        result = { ok: true, ...forge.run(name, version || undefined, record(inputs.input)) }
        break
      case 'undefine':
        result = forge.undefine(name, version)
          ? { ok: true, removed: `${name}@${version}` }
          : { ok: false, error: 'forged package not found' }
        break
      default:
        throw new TypeError('CreatorForgeTool action must be list, inspect, define, run, or undefine')
    }
    recordCreatorTrace(metadata, {
      action,
      name,
      version: version || ('version' in (result as object) ? text((result as Record<string, unknown>).version) : ''),
      status: isOk(result) ? 'ok' : 'error',
      detail: traceDetail(result),
    })
    return result
  } catch (error) {
    recordCreatorTrace(metadata, {
      action: action || 'invalid',
      name,
      version,
      status: 'error',
      detail: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}

function parameterRecords(value: unknown): readonly Readonly<Record<string, unknown>>[] {
  if (!Array.isArray(value)) return []
  return value.map(item => record(item))
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
}

function summary(pkg: ReturnType<DeclarativeToolForge['list']>[number]): Record<string, unknown> {
  return {
    name: pkg.name,
    version: pkg.version,
    description: pkg.description,
    parameters: pkg.parameters.map(parameter => ({
      name: parameter.name,
      required: parameter.required,
      ...(parameter.description ? { description: parameter.description } : {}),
      ...(parameter.defaultValue === undefined ? {} : { default: parameter.defaultValue }),
    })),
    created_at: pkg.createdAt,
  }
}

function isOk(value: unknown): boolean {
  return !value || typeof value !== 'object' || Array.isArray(value)
    ? true
    : (value as Readonly<Record<string, unknown>>).ok !== false
}

function traceDetail(value: unknown): string {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  const row = value as Readonly<Record<string, unknown>>
  return text(row.error ?? row.output ?? row.removed)
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}
