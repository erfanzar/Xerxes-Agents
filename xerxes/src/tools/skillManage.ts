// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { scanContextContent } from '../security/promptScanner.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalString, requiredString } from './inputs.js'

export type SkillManageIntent = 'create' | 'delete' | 'edit' | 'list' | 'view'

export interface SkillManageOptions {
  /** Override only for an explicit tenant/workspace host or tests. */
  readonly authoredDirectory?: string
}

/** Create, edit, view, delete, and list skills authored through the agent tool. */
export class AgentAuthoredSkillStore {
  readonly authoredDirectory: string

  constructor(options: SkillManageOptions = {}) {
    this.authoredDirectory = resolve(options.authoredDirectory ?? join(xerxesHome(), 'skills', 'agent-authored'))
  }

  async manage(
    intent: string,
    options: { readonly body?: string; readonly description?: string; readonly name?: string; readonly version?: string } = {},
  ): Promise<JsonObject> {
    if (intent === 'list') {
      await mkdir(this.authoredDirectory, { recursive: true })
      const entries = await readdir(this.authoredDirectory, { withFileTypes: true })
      return {
        ok: true,
        intent,
        skills: entries
          .filter(entry => entry.isFile() && entry.name.endsWith('.md'))
          .map(entry => entry.name.slice(0, -'.md'.length))
          .sort((left, right) => left.localeCompare(right)),
      }
    }

    const name = options.name ?? ''
    const failure = validateName(name)
    if (failure) return { ok: false, intent, name, error: failure }
    const path = await this.pathFor(name)

    if (intent === 'view') {
      if (!(await exists(path))) return { ok: false, intent, name, error: 'not found' }
      return { ok: true, intent, name, path, body: await readFile(path, 'utf8') }
    }

    if (intent === 'delete') {
      if (!(await exists(path))) return { ok: false, intent, name, error: 'not found' }
      await rm(path)
      return { ok: true, intent, name, path }
    }

    if (intent === 'create' || intent === 'edit') {
      const body = options.body ?? ''
      if (!body.trim()) return { ok: false, intent, name, error: 'body must be non-empty' }
      const scanned = scanContextContent(body, path)
      if (!scanned.trim()) return { ok: false, intent, name, error: 'security scan stripped all content' }
      if (intent === 'create' && await exists(path)) {
        return { ok: false, intent, name, error: 'skill already exists; use intent=edit' }
      }
      const content = frontmatter(name, options.description ?? 'Agent-authored skill', options.version ?? '0.1.0') + scanned.trimStart()
      await atomicWrite(path, content)
      return { ok: true, intent, name, path }
    }

    return { ok: false, intent, name, error: 'unknown intent: ' + intent }
  }

  private async pathFor(name: string): Promise<string> {
    await mkdir(this.authoredDirectory, { recursive: true })
    return join(this.authoredDirectory, name + '.md')
  }
}

export const SKILL_MANAGE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'skill_manage',
    description: 'Create, edit, view, delete, or list security-scanned agent-authored Markdown skills.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        intent: { type: 'string', enum: ['list', 'view', 'create', 'edit', 'delete'] },
        name: { type: 'string' },
        body: { type: 'string' },
        description: { type: 'string' },
        version: { type: 'string', default: '0.1.0' },
      },
      required: ['intent'],
    },
  },
}

export function registerSkillManageTool(registry: ToolRegistry, options: SkillManageOptions = {}): AgentAuthoredSkillStore {
  const store = new AgentAuthoredSkillStore(options)
  registry.register(SKILL_MANAGE_DEFINITION, inputs => {
    const name = optionalString(inputs, 'name')
    const body = optionalString(inputs, 'body')
    const description = optionalString(inputs, 'description')
    const version = optionalString(inputs, 'version')
    return store.manage(requiredString(inputs, 'intent'), {
      ...(name === undefined ? {} : { name }),
      ...(body === undefined ? {} : { body }),
      ...(description === undefined ? {} : { description }),
      ...(version === undefined ? {} : { version }),
    })
  })
  return store
}

function frontmatter(name: string, description: string, version: string): string {
  return [
    '---',
    'name: ' + yamlScalar(name),
    'description: ' + yamlScalar(description),
    'version: ' + yamlScalar(version),
    'author: xerxes-agent',
    '---',
    '',
    '',
  ].join('\n')
}

function yamlScalar(value: string): string {
  return JSON.stringify(value)
}

function validateName(name: string): string | undefined {
  if (!name || name.includes('/') || name.includes('\\') || name.includes('..') || name.includes('\0')) {
    return 'invalid skill name: ' + JSON.stringify(name)
  }
  return undefined
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') return false
    throw error
  }
}

async function atomicWrite(path: string, body: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = join(dirname(path), '.' + crypto.randomUUID() + '.tmp')
  try {
    await Bun.write(temporary, body)
    await rename(temporary, path)
  } finally {
    await rm(temporary, { force: true })
  }
}
