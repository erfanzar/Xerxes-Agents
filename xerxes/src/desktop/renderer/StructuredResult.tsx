// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

const isRecord = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value)
const labelOf = (key: string): string => key.replace(/([a-z])([A-Z])/g, '$1 $2').replaceAll('_', ' ')
function Value({ value, depth = 0 }: { value: unknown; depth?: number }): ReactElement {
  if (value === null) return <span className="result-muted">null</span>
  if (typeof value !== 'object') return <span className="result-value">{String(value)}</span>
  const entries = Array.isArray(value) ? value.map((item,index)=>[String(index + 1),item] as const) : Object.entries(value)
  if (!entries.length) return <span className="result-muted">{Array.isArray(value) ? 'Empty list' : 'Empty object'}</span>
  if (depth >= 4) return <span className="result-muted">{entries.length} {entries.length === 1 ? 'entry' : 'entries'} · available in Raw details</span>
  const overflow = entries.length - 50
  return <>
    <dl className="result-fields">{entries.slice(0,50).map(([key,item])=><div key={key}>{item !== null && typeof item === 'object' ? <><dt>{labelOf(key)}</dt><dd><details><summary className="result-muted">{Array.isArray(item) ? `${item.length} items` : `${Object.keys(item).length} fields`}</summary><Value value={item} depth={depth+1}/></details></dd></> : <><dt>{labelOf(key)}</dt><dd><Value value={item} depth={depth+1}/></dd></>}</div>)}</dl>
    {overflow > 0 && <p className="result-muted">{overflow} more {overflow === 1 ? 'entry' : 'entries'} in Raw details</p>}
  </>
}
export function StructuredResult({ value }: { value: unknown }): ReactElement {
  const goal = isRecord(value) && isRecord(value.goal) && typeof value.goal.objective === 'string' ? value.goal : null
  if (!goal) return <div className="structured-result"><Value value={value}/></div>
  const { objective, phase, currentMilestone, criteria, ...metadata } = goal
  const extra = Object.fromEntries(Object.entries(value as Record<string, unknown>).filter(([key])=>key!=='goal'))
  return <div className="structured-result goal-result">
    <header><strong>Goal</strong>{typeof phase==='string' && <span>{phase}</span>}</header>
    <p>{String(objective)}</p>
    {typeof currentMilestone==='string' && currentMilestone && <section><strong className="goal-result__subhead">Current milestone</strong><p>{currentMilestone}</p></section>}
    {Array.isArray(criteria) && <details><summary>Success criteria <span className="result-muted">{criteria.length}</span></summary><ul>{criteria.map((criterion,index)=><li key={index}>{isRecord(criterion) && typeof criterion.description==='string' ? <><p>{criterion.description}</p>{Object.keys(criterion).some(key=>key!=='description' && key!=='id') && <details><summary>Details</summary><Value value={Object.fromEntries(Object.entries(criterion).filter(([key])=>key!=='description'))}/></details>}</> : <Value value={criterion}/>}</li>)}</ul></details>}
    {Object.keys(metadata).length>0 && <details><summary>Goal metadata</summary><Value value={metadata}/></details>}
    {Object.keys(extra).length>0 && <Value value={extra}/>}
  </div>
}
export function structuredOutput(text: string): object | null {
  try { const value: unknown = JSON.parse(text); return value !== null && typeof value === 'object' ? value : null } catch { return null }
}
