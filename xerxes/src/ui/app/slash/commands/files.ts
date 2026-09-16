// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'
import { getTurnState } from '../../turnStore.js'

export const fileCommands: SlashCommand[] = [
  {
    name: 'tool-output', group: 'tools', help: 'read tool calls and full received output [list|last|number|id]',
    run: (argument, ctx) => {
      const records = Object.assign({}, ...ctx.local.getHistoryItems().map(message => message.toolRecords ?? {}), getTurnState().toolRecords) as ReturnType<typeof getTurnState>['toolRecords']
      const entries = Object.entries(records)
      const wanted = argument.trim()
      if (!entries.length) return ctx.transcript.sys('No retained tool output in this view. Saved legacy sessions may contain only tool summaries.')
      if (!wanted || wanted === 'list') {
        ctx.transcript.page(entries.map(([id, record], index) => `${index + 1}. ${record.name}${record.error ? ' · failed' : ''}\n   ${id}`).join('\n') + '\n\n/tool-output last or /tool-output <number> opens a result.', 'Tool outputs')
        return
      }
      const found = wanted === 'last' ? entries.at(-1) : /^\d+$/.test(wanted) ? entries[Number(wanted) - 1] : entries.find(([id]) => id === wanted)
      if (!found) return ctx.transcript.sys('Tool output not found. Use /tool-output list.')
      const [id, record] = found
      ctx.transcript.page([record.name, id, record.args ? `Call\n${record.args}` : '', record.reasoning ? `Context\n${record.reasoning}` : '', record.error ? `Error\n${record.error}` : '', `Output\n${record.result ?? 'No output was retained.'}`].filter(Boolean).join('\n\n'), 'Tool output')
    },
  },
  { name: 'search', group: 'session', help: 'search transcripts [--session <id>] [--limit <count>] <text>', run: (argument, ctx) => {
    let query = argument.trim(), sessionId: string | undefined, limit = 20
    while (/^--(?:session|limit)(?:\s|$)/.test(query)) {
      const option = /^--(session|limit)\s+(\S+)(?:\s+|$)/.exec(query)
      if (!option) { ctx.transcript.sys('Usage: /search [--session <id>] [--limit 1–500] <text>'); return }
      if (option[1] === 'session') sessionId = option[2]
      else limit = Number(option[2])
      query = query.slice(option[0].length).trim()
    }
    if (!query || !Number.isSafeInteger(limit) || limit < 1 || limit > 500) { ctx.transcript.sys('Usage: /search [--session <id>] [--limit 1–500] <text>'); return }
    ctx.transcript.sys('Searching saved conversations…')
    void ctx.gateway.rpc('session.search', { query, limit, ...(sessionId ? { session_id: sessionId } : {}) }).then(ctx.guarded(result => {
      if (result?.ok !== true || !Array.isArray(result.results)) throw new Error(String(result?.error ?? 'Search unavailable'))
      const rows = result.results.map((item: unknown) => {
        if (!item || typeof item !== 'object') throw new Error('Invalid search result')
        const hit = item as Record<string, unknown>
        return `${hit.title || hit.session_id} · ${hit.updated_at || 'date unavailable'}\n${hit.session_id} · message ${hit.message_index} · ${hit.role}\n${hit.excerpt}\n/resume ${hit.session_id}`
      })
      const stats = result.stats as Record<string, unknown> | undefined
      ctx.transcript.page([`${result.results.length} matches for ${query}${sessionId ? ` in ${sessionId}` : ''} · limit ${limit}`, ...rows,
        stats ? `Search coverage: ${stats.sessions ?? 0} sessions · ${stats.searchable_messages ?? 0} searchable / ${stats.indexed_messages ?? 0} indexed messages · ${stats.truncated_messages ?? 0} truncated` : '',
        stats?.unrecognized_messages ? `${stats.unrecognized_messages} messages could not be indexed.` : '',
        !rows.length ? 'No matching messages.' : 'Use /resume <session id> to open a conversation.'].filter(Boolean).join('\n\n'), 'Transcript search')
    })).catch(ctx.guardedErr)
  } },
  {
    name: 'file', group: 'tools', help: 'preview a workspace text file with line numbers', usage: '/file <path>',
    run: (argument, ctx) => {
      const path = argument.trim()
      if (!path) return ctx.transcript.sys('Usage: /file <workspace-relative path>')
      ctx.transcript.sys(`Reading ${path}…`)
      void ctx.gateway.rpc('workspace.filePreview', { path, session_id: ctx.sid }).then(ctx.guarded(result => {
        if (result?.ok !== true || typeof result.content !== 'string' || typeof result.path !== 'string') throw new Error(String(result?.error ?? 'Invalid file preview response'))
        const lines = result.content.split('\n')
        const width = String(lines.length).length
        ctx.transcript.page(`${result.path}${result.truncated ? '\nPreview limited to 128 KiB; file continues.' : ''}\n\n${lines.map((line, index) => `${String(index + 1).padStart(width)}  ${line}`).join('\n')}`, 'Workspace file')
      })).catch(ctx.guardedErr)
    },
  },
  {
    name: 'undo-edits', group: 'tools', help: 'reverse recorded agent edits after confirmation', usage: '/undo-edits <recorded path|--all>',
    run: (argument, ctx) => {
      const path = argument.trim()
      if (!path) return ctx.transcript.sys('Usage: /undo-edits <exact recorded path|--all>. Use /diff to review first.')
      patchOverlayState({ confirm: {
        title: 'Reverse recorded agent edits?', danger: true, confirmLabel: 'Reverse edits', cancelLabel: 'Cancel',
        detail: `${path === '--all' ? 'All files with reversible edits in this session' : path}. This changes files on disk. Only recorded text edits can be reversed; mismatching files are refused. Use /diff to review first.`,
        onConfirm: () => {
          if (ctx.stale()) return
          void ctx.gateway.rpc('changes.undo', { session_id: ctx.sid, path: path === '--all' ? '' : path }).then(ctx.guarded(result => {
            const rows = Array.isArray(result?.results) ? result.results : []
            const lines = rows.map((item: unknown) => {
              if (!item || typeof item !== 'object') return 'Invalid file result'
              const row = item as Record<string, unknown>
              return `${String(row.path ?? 'File')}: ${row.ok === true ? `${String(row.reverted ?? 0)} edits reversed` : String(row.error ?? 'not reversed')}`
            })
            ctx.transcript.page(lines.join('\n') || String(result?.error ?? 'No edits reversed.'), 'Recorded edit results')
          })).catch(ctx.guardedErr)
        },
      } })
    },
  },
]
