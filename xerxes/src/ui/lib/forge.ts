// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export interface ForgeParameter { name: string; description: string; required: boolean; default?: string }
export interface ForgePackage { name: string; version: string; description: string; parameters: ForgeParameter[]; template?: string; createdAt: string }
export function forgePackage(value: unknown, detailed = false): ForgePackage {
  if (!value || typeof value !== 'object') throw new Error('Invalid Forge package')
  const row = value as Record<string, unknown>
  if (typeof row.name !== 'string' || !row.name || typeof row.version !== 'string' || typeof row.description !== 'string'
    || !Array.isArray(row.parameters) || typeof row.created_at !== 'string' || (detailed && typeof row.template !== 'string')) throw new Error('Invalid Forge package')
  const parameters = row.parameters.map((value: unknown): ForgeParameter => {
    if (!value || typeof value !== 'object') throw new Error('Invalid Forge parameter')
    const p = value as Record<string, unknown>
    if (typeof p.name !== 'string' || typeof p.description !== 'string' || typeof p.required !== 'boolean'
      || (p.default !== undefined && typeof p.default !== 'string')) throw new Error('Invalid Forge parameter')
    return { name: p.name, description: p.description, required: p.required, ...(p.default !== undefined ? { default: p.default } : {}) }
  })
  return { name: row.name, version: row.version, description: row.description, parameters, createdAt: row.created_at, ...(typeof row.template === 'string' ? { template: row.template } : {}) }
}
