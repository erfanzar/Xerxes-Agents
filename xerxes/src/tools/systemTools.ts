// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { statfs } from 'node:fs/promises'
import {
  arch,
  availableParallelism,
  cpus,
  freemem,
  hostname,
  loadavg,
  networkInterfaces,
  platform,
  release,
  totalmem,
  type,
  uptime,
  version,
} from 'node:os'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalString, requiredString } from './inputs.js'

const SAFE_ENVIRONMENT_KEYS = new Set([
  'BUN_VERSION',
  'CI',
  'COLORTERM',
  'HOME',
  'LANG',
  'NODE_ENV',
  'PATH',
  'PWD',
  'SHELL',
  'TERM',
  'TMPDIR',
  'TZ',
  'USER',
])
const SYSTEM_INFO_TYPES = ['all', 'os', 'cpu', 'memory', 'disk', 'network', 'process'] as const

export const SYSTEM_INFO_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'SystemInfo',
    description: 'Inspect local operating-system, CPU, memory, filesystem, network, and Bun process metadata.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        info_type: { type: 'string', enum: [...SYSTEM_INFO_TYPES], default: 'all' },
      },
    },
  },
}

export const ENVIRONMENT_MANAGER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'EnvironmentManager',
    description: 'Inspect non-sensitive process environment settings. Sensitive values are always redacted.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        operation: { type: 'string', enum: ['get', 'list'] },
        key: {
          type: 'string',
          description: 'Variable name for get, or an optional prefix for the safe list.',
        },
      },
      required: ['operation'],
    },
  },
}

export const SYSTEM_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  SYSTEM_INFO_DEFINITION,
  ENVIRONMENT_MANAGER_DEFINITION,
]

/**
 * Register safe inspection-only system tools.
 *
 * Filesystem mutation, arbitrary process control, and environment mutation stay
 * with their dedicated permission-gated tool modules.
 */
export function registerSystemTools(registry: ToolRegistry): void {
  registry.register(SYSTEM_INFO_DEFINITION, systemInfo)
  registry.register(ENVIRONMENT_MANAGER_DEFINITION, inspectEnvironment)
}

export async function systemInfo(inputs: JsonObject): Promise<JsonObject> {
  const infoType = optionalString(inputs, 'info_type') ?? 'all'
  if (!SYSTEM_INFO_TYPES.includes(infoType as (typeof SYSTEM_INFO_TYPES)[number])) {
    throw new ValidationError('info_type', 'must be all, os, cpu, memory, disk, network, or process', infoType)
  }

  if (infoType === 'os') return { os: osInfo() }
  if (infoType === 'cpu') return { cpu: cpuInfo() }
  if (infoType === 'memory') return { memory: memoryInfo() }
  if (infoType === 'disk') return { disk: await diskInfo() }
  if (infoType === 'network') return { network: networkInfo() }
  if (infoType === 'process') return { process: processInfo() }

  return {
    cpu: cpuInfo(),
    disk: await diskInfo(),
    memory: memoryInfo(),
    network: networkInfo(),
    os: osInfo(),
    process: processInfo(),
  }
}

export function inspectEnvironment(inputs: JsonObject): JsonObject {
  const operation = requiredString(inputs, 'operation')

  if (operation === 'get') {
    const key = requiredString(inputs, 'key')
    const value = process.env[key]
    const redacted = value !== undefined && !isInspectableEnvironmentKey(key)
    return {
      exists: value !== undefined,
      key,
      redacted,
      value: value === undefined ? null : redacted ? '[REDACTED]' : value,
    }
  }

  if (operation === 'list') {
    const prefix = optionalString(inputs, 'key')
    const environment: JsonObject = {}
    for (const key of Object.keys(process.env).sort((left, right) => left.localeCompare(right))) {
      if (!isInspectableEnvironmentKey(key) || (prefix !== undefined && !key.startsWith(prefix))) {
        continue
      }
      const value = process.env[key]
      if (value !== undefined) environment[key] = value
    }
    return { count: Object.keys(environment).length, environment }
  }

  throw new ValidationError('operation', 'must be get or list; set and remove are not inspection operations', operation)
}

function osInfo(): JsonObject {
  return {
    architecture: arch(),
    hostname: hostname(),
    platform: platform(),
    release: release(),
    system: type(),
    version: version(),
  }
}

function cpuInfo(): JsonObject {
  const processors = cpus()
  return {
    available_parallelism: availableParallelism(),
    frequency_mhz: processors[0]?.speed ?? null,
    load_average: loadavg(),
    logical_cores: processors.length,
    model: processors[0]?.model ?? null,
    uptime: uptime(),
  }
}

function memoryInfo(): JsonObject {
  const total = totalmem()
  const available = freemem()
  const used = total - available
  return {
    available,
    available_gb: gigabytes(available),
    percent: total === 0 ? 0 : round((used / total) * 100),
    total,
    total_gb: gigabytes(total),
    used,
    used_gb: gigabytes(used),
  }
}

async function diskInfo(): Promise<JsonObject> {
  try {
    const filesystem = await statfs(process.cwd())
    const blockSize = Number(filesystem.bsize)
    const total = blockSize * Number(filesystem.blocks)
    const free = blockSize * Number(filesystem.bavail)
    const used = total - free
    return {
      available: true,
      free,
      free_gb: gigabytes(free),
      percent: total === 0 ? 0 : round((used / total) * 100),
      total,
      total_gb: gigabytes(total),
      used,
      used_gb: gigabytes(used),
    }
  } catch (error) {
    return {
      available: false,
      error: errorMessage(error),
    }
  }
}

function networkInfo(): JsonObject {
  const interfaces: JsonObject[] = []
  for (const [name, addresses] of Object.entries(networkInterfaces())) {
    const values = (addresses ?? []).map(address => ({
      address: address.address,
      cidr: address.cidr ?? null,
      family: address.family,
      internal: address.internal,
      netmask: address.netmask,
    }))
    interfaces.push({ addresses: values, name })
  }
  return { hostname: hostname(), interfaces }
}

function processInfo(): JsonObject {
  const memory = process.memoryUsage()
  return {
    architecture: process.arch,
    bun_version: Bun.version,
    cwd: process.cwd(),
    node_compat_version: process.version,
    pid: process.pid,
    platform: process.platform,
    process_uptime: process.uptime(),
    memory: {
      array_buffers: memory.arrayBuffers,
      external: memory.external,
      heap_total: memory.heapTotal,
      heap_used: memory.heapUsed,
      resident_set: memory.rss,
    },
  }
}

function isInspectableEnvironmentKey(key: string): boolean {
  return SAFE_ENVIRONMENT_KEYS.has(key) || key.startsWith('LC_')
}

function gigabytes(value: number): number {
  return round(value / (1024 ** 3))
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
