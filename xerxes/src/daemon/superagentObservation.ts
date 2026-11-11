// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { validateAbilityInput } from '../hub/supervisor/abilities/catalog.js'
import { WorkspaceObservations, type ObservationHosts } from '../hub/supervisor/abilities/observations.js'

/** Root is supplied by the daemon's verified project context, never by the model. */
export async function observeWorkspace(root: string, input: Readonly<Record<string, unknown>>, hosts: ObservationHosts = {}, signal?: AbortSignal): Promise<Readonly<Record<string, unknown>>> {
  const { ability, project_dir: _projectDir, session_key: _sessionKey, hub_auth: _hubAuth, ...fields } = input
  if (typeof ability !== 'string') throw new Error('An observation ability is required')
  const checked = validateAbilityInput(ability, { ...fields, workspace_id: 'native' })
  if (!checked.ok) throw new Error(checked.error)
  const args = checked.args
  const observer = new WorkspaceObservations(root, hosts)
  const text = (key: string, fallback = '') => typeof args[key] === 'string' ? args[key] : fallback
  const number = (key: string, fallback: number) => typeof args[key] === 'number' ? args[key] : fallback
  let result: Readonly<Record<string, unknown>>
  switch (ability) {
    case 'workspace.read': result = await observer.read(text('path'), number('offset', 0), number('max_bytes', 12000)); break
    case 'workspace.ls': result = await observer.list(text('path', '.'), number('offset', 0), number('limit', 100)); break
    case 'workspace.grep': result = await observer.search(text('query'), number('offset', 0), number('limit', 50), signal); break
    case 'workspace.git': {
      const view = text('view')
      if (view !== 'status' && view !== 'log' && view !== 'diff') throw new Error('Invalid git view')
      result = await observer.git(view, signal); break
    }
    case 'machine.health': result = await observer.health(signal); break
    case 'github.read': {
      const view = text('view')
      if (view !== 'issues' && view !== 'prs' && view !== 'ci') throw new Error('Invalid GitHub view')
      result = await observer.github(view, signal); break
    }
    case 'ui.screenshot': result = await observer.screenshot(number('width', 1280), signal); break
    default: throw new Error('Only Tier A observation abilities may run on this daemon route')
  }
  if (Buffer.byteLength(JSON.stringify(result)) > 60 * 1024) throw new Error('Observation exceeds 60 KiB; reduce max_bytes or limit and page the result')
  return result
}
