// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, mkdir, realpath, rename, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { loadAgentSpecData } from '../agents/agentSpec.js'
import { parseYaml } from '../agents/yaml.js'
import type { SkillBundleStore } from '../tools/agentMetaTools.js'
import type { ReflectionApplier } from './reflection.js'

/** One host-owned applier, shared by model tools and human slash review. */
export function createReflectionApplier(options: {
  readonly skills: SkillBundleStore
  readonly activeWorkspace: () => string
}): ReflectionApplier {
  return {
    async applySkill(request) {
      await mkdir(options.skills.directory, { recursive: true })
      const existing = request.action === 'update' ? await options.skills.get(request.name) : undefined
      return options.skills.manage({ ...request, tags: existing?.metadata.tags ?? [], version: existing?.metadata.version ?? '0.1.0' })
    },
    async applyAgent(name, yaml) {
      let temporary: string | undefined
      try {
        if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/u.test(name)) throw new Error('Agent target must be a name without path separators or ..')
        const workspace = await realpath(options.activeWorkspace())
        const directory = join(workspace, '.xerxes', 'agents')
        const target = join(directory, `${name}.yaml`)
        const spec = loadAgentSpecData(target, parseYaml(yaml, target))
        if (spec.name !== name) throw new Error(`Agent YAML name must equal approved target '${name}'.`)
        // Refuse even internal symlinks: approvals name a project file, not an alias.
        for (const path of [join(workspace, '.xerxes'), directory]) {
          try {
            const info = await lstat(path)
            if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`Agent directory must be a regular project directory: ${path}`)
          } catch (error) { if (!isMissing(error)) throw error }
        }
        try {
          const info = await lstat(target)
          if (!info.isFile() || info.isSymbolicLink()) throw new Error('Agent target must be a regular project agent file or a new file.')
        } catch (error) { if (!isMissing(error)) throw error }
        await mkdir(directory, { recursive: true })
        if (await realpath(directory) !== directory) throw new Error('Agent directory escapes the project agent directory.')
        temporary = join(directory, `.${name}.${crypto.randomUUID()}.tmp`)
        await writeFile(temporary, yaml, { flag: 'wx', mode: 0o600 })
        await rename(temporary, target)
        temporary = undefined
        return { ok: true, name, path: target }
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : String(error) }
      } finally { if (temporary !== undefined) await unlink(temporary) }
    },
  }
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT'
}
