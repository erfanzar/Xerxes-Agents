// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import { plural } from '../../client/format.js'
import { toolFamily } from '../../client/taskTypes.js'
import { diffCounts, diffLines, diffRows } from './diffLines.js'
import { formatDuration, OUTPUT_PREVIEW_LINES, outputPreview, toolIcon, toolTarget, toolVerb, type ToolDiff, type ToolView } from './toolView.js'

/** A unified diff: `+` added and `-` removed lines with a tint, long unchanged runs folded. */
export function DiffView({ diff }: { diff: ToolDiff }): ReactElement {
  const ops = diffLines(diff.before, diff.after)
  const rows = diffRows(ops)
  const { added, removed } = diffCounts(ops)
  return (
    <figure className="diff" aria-label={`Changes to ${diff.path}`}>
      <figcaption className="diff-head">
        <code className="diff-path">{diff.path}</code>
        <span className="diff-count">+{added} −{removed}</span>
      </figcaption>
      {rows.length === 0 ? <p className="block-meta">No line changes.</p> : (
        <pre className="diff-body">
          {rows.map((row, index) => row.kind === 'gap'
            ? <span key={index} className="diff-line diff-gap">⋯ {plural(row.count, 'unchanged line')}</span>
            : (
              <span key={index} className={`diff-line diff-${row.kind}`}>
                <span className="diff-mark">{row.kind === 'add' ? '+' : row.kind === 'del' ? '-' : ' '}</span>{row.text}
              </span>
            ))}
        </pre>
      )}
    </figure>
  )
}

/**
 * One tool call as an expandable row: verb and target with state and duration
 * on the line, and the full command or arguments, the edit diff and the output
 * inside. Long output starts collapsed at OUTPUT_PREVIEW_LINES lines.
 */
export function ToolRow({ tool }: { tool: ToolView }): ReactElement {
  const [all, setAll] = useState(false)
  const target = toolTarget(tool)
  const command = toolFamily(tool.name) === 'command'
  const duration = formatDuration(tool.durationMs)
  const output = tool.output === undefined || tool.output === '' ? undefined : outputPreview(tool.output, all)
  // An edit's arguments repeat its diff; show them only when there is no diff.
  const args = tool.args !== undefined && tool.args.trim() !== '' && tool.diff === undefined ? tool.args : undefined
  const empty = args === undefined && tool.diff === undefined && output === undefined
  return (
    <details className="fold tool-row" data-state={tool.state}>
      <summary>
        <span className="tool-icon" aria-hidden="true">{toolIcon(tool.name)}</span>
        <span className="tool-verb">{toolVerb(tool.name, tool.state)}</span>
        {target === '' ? null : <code className="tool-target">{target}</code>}
        <span className="tool-meta">
          {tool.state === 'working'
            ? <span className="tool-spin" role="img" aria-label="running" />
            : tool.state === 'error'
              ? <span className="tool-state tool-error">failed</span>
              : <span className="tool-state tool-done" role="img" aria-label="done">✓</span>}
          {duration === '' ? null : <span className="tool-duration">{duration}</span>}
        </span>
      </summary>
      <div className="tool-body">
        {args === undefined ? null : <pre className={`block-code tool-args${command ? ' is-command' : ''}`} aria-label={command ? 'Command' : 'Arguments'}>{args}</pre>}
        {tool.diff === undefined ? null : <DiffView diff={tool.diff} />}
        {output === undefined ? null : (
          <>
            <pre className="block-code tool-output" aria-label="Output">{output.text}</pre>
            {output.clipped || (all && output.lines > OUTPUT_PREVIEW_LINES) ? (
              <button type="button" className="btn btn-quiet tool-more" onClick={() => setAll(!all)}>
                {all ? 'Show less' : `Show all ${output.lines} lines`}
              </button>
            ) : null}
          </>
        )}
        {empty ? <p className="block-meta">{tool.state === 'working' ? 'Running… no output yet.' : 'No details were recorded for this step.'}</p> : null}
      </div>
    </details>
  )
}
