// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ReactElement } from 'react'
import { parseUnifiedDiff } from '../../ui/lib/gitDiff.js'

export function DiffPreview({ diff, label }: { diff: string; label: string }): ReactElement {
  const parsed = parseUnifiedDiff(diff)
  if (!diff.trim()) return <p>No changes in this preview.</p>
  return <>
    <pre className="change-review__source unified-diff" tabIndex={0} aria-label={label}>
      {parsed.lines.map((line, index) => <span className={`studio-diff-${line.kind}`} key={index}>
        <span className="diff-gutter" aria-hidden="true">{line.oldLine ?? ''}</span>
        <span className="diff-gutter" aria-hidden="true">{line.newLine ?? ''}</span>
        <span className="diff-code">{line.text}</span>
      </span>)}
    </pre>
    {parsed.truncated && <p>Diff preview truncated.</p>}
  </>
}
