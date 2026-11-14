// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `artifacts.list`, `artifacts.read` and `artifacts.pin`: the daemon side of
 * the session artifacts the agent saves with ArtifactTool. The caller resolves
 * the session; this module only validates params and shapes replies.
 */
import { summaryOf, type Artifact, type ArtifactStore, type ArtifactSummary } from '../session/artifacts.js'

export type ArtifactsRpcMethod = 'artifacts.list' | 'artifacts.read' | 'artifacts.pin' | 'artifacts.update'

export type ArtifactsRpcReply = Record<string, unknown>

export async function artifactsRpc(
  store: ArtifactStore,
  method: ArtifactsRpcMethod,
  sessionId: string,
  params: Readonly<Record<string, unknown>>,
): Promise<ArtifactsRpcReply> {
  try {
    if (method === 'artifacts.list') {
      return { ok: true, artifacts: (await store.list(sessionId)).map(summaryWire) }
    }
    const id = typeof params.id === 'string' ? params.id.trim() : ''
    if (!id) return { ok: false, error: 'id is required' }
    if (method === 'artifacts.read') {
      const artifact = await store.read(sessionId, id)
      return artifact ? { ok: true, artifact: artifactWire(artifact) } : { ok: false, error: `no artifact ${id} in this session` }
    }
    if (method === 'artifacts.update') {
      // A person's edit from the viewer: new content (and optionally a title), version bumped.
      if (typeof params.content !== 'string') return { ok: false, error: 'content must be a string' }
      if (params.title !== undefined && typeof params.title !== 'string') return { ok: false, error: 'title must be a string' }
      const updated = await store.update(sessionId, id, { content: params.content, ...(typeof params.title === 'string' ? { title: params.title } : {}) })
      return { ok: true, artifact: artifactWire(updated) }
    }
    if (typeof params.pinned !== 'boolean') return { ok: false, error: 'pinned must be a boolean' }
    const pinned = await store.pin(sessionId, id, params.pinned)
    return pinned ? { ok: true, artifact: summaryWire(pinned) } : { ok: false, error: `no artifact ${id} in this session` }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

function summaryWire(summary: ArtifactSummary): ArtifactsRpcReply {
  return {
    id: summary.id,
    title: summary.title,
    kind: summary.kind,
    ...(summary.language === undefined ? {} : { language: summary.language }),
    size: summary.size,
    created_at: summary.created_at,
    updated_at: summary.updated_at,
    pinned: summary.pinned,
    version: summary.version,
  }
}

function artifactWire(artifact: Artifact): ArtifactsRpcReply {
  return { ...summaryWire(summaryOf(artifact)), content: artifact.content }
}
