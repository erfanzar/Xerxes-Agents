// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Which models the current task used, and how much: the worker on this Mac,
 * the cloud advisor and any subagents. In Hybrid the composer shows the split
 * as a small bar ("97% on this Mac"), and the Usage panel opens with the full
 * breakdown: each model, where it ran, its calls and tokens, what the cloud
 * cost, and the advisor's limits. Shares are by tokens processed (read and
 * written), the work each model actually did.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'

import {
  compactBytes, compactTokens, costLabel, modelRoleLabel, modelUsageSummary, processedTokens, taskModelsOf, taskSplit, usd,
  type AdvisorLimitsView, type TaskModelUsageView, type TaskSplit,
} from '../../auth/usageView.js'
import { desktopCall } from './desktopRpc.js'
import type { Snapshot } from './store.js'

export interface TaskModels {
  readonly models: readonly TaskModelUsageView[]
  readonly advisor?: AdvisorLimitsView
}

/** How often the numbers refresh while a turn runs, so an advisor call shows up as it happens. */
const LIVE_MS = 4_000

/**
 * This task's models from `session.usage`: after each turn, when the task
 * changes, and every few seconds while a turn runs. Null on a runtime that
 * does not record them, before the task has a session, and while offline.
 */
export function useTaskModels(snap: Snapshot): TaskModels | null {
  const [value, setValue] = useState<TaskModels | null>(null)
  const online = snap.connection === 'online'
  const task = `${snap.sessionKey}\u0000${snap.currentId}`
  const latest = useRef(task)
  latest.current = task
  const refresh = useCallback(async () => {
    if (!online) return
    const asked = task
    try {
      const parsed = taskModelsOf(await desktopCall(window.xerxes, snap.sessionKey, 'session.usage', {}))
      if (latest.current !== asked) return
      setValue(parsed.models ? { models: parsed.models, ...(parsed.advisor ? { advisor: parsed.advisor } : {}) } : null)
    } catch {
      // No session yet, or an older runtime: there is nothing to show, and the
      // composer must not grow an error for a figure it only summarizes.
      if (latest.current === asked) setValue(null)
    }
  }, [online, snap.sessionKey, task])
  // A different task starts empty rather than showing the previous one's split.
  useEffect(() => { setValue(null) }, [task])
  useEffect(() => { void refresh() }, [refresh, snap.turnActive])
  useEffect(() => {
    if (!snap.turnActive || !online) return
    const timer = setInterval(() => { if (document.visibilityState === 'visible') void refresh() }, LIVE_MS)
    return () => clearInterval(timer)
  }, [refresh, snap.turnActive, online])
  return value
}

const percentLabel = (split: TaskSplit): number => {
  const share = split.localShare ?? 0
  // Never "100%" while something ran in the cloud, never "0%" while something ran here.
  const rounded = Math.round(share * 100)
  if (split.cloudTokens > 0 && rounded >= 100) return 99
  if (split.localTokens > 0 && rounded <= 0) return 1
  return rounded
}

function cloudSpend(split: TaskSplit): string | null {
  if (split.cloudTokens === 0) return null
  if (split.cloudOnPlan) return 'included in plan'
  return split.cloudUsd === undefined ? 'price unknown' : usd(split.cloudUsd)
}

function where(model: TaskModelUsageView): 'local' | 'cloud' | 'unknown' {
  return model.local === true ? 'local' : model.local === false ? 'cloud' : 'unknown'
}

const WHERE_LABEL = { local: 'on this Mac', cloud: 'cloud', unknown: 'unknown location' } as const

/** "Main · ternary-bonsai-2-27b (on this Mac): 1.2M tokens in 14 calls". */
function modelLine(model: TaskModelUsageView): string {
  const place = where(model)
  const cost = place === 'cloud' ? `, ${costLabel(model.costUsd, model.costBasis)}` : ''
  return `${modelRoleLabel(model.role)} · ${model.model} (${WHERE_LABEL[place]}): ${compactTokens(processedTokens(model))} tokens in ${model.calls} ${model.calls === 1 ? 'call' : 'calls'}${cost}`
}

/** Two segments, on this Mac then cloud, sized by tokens; a 2px gap keeps them apart. */
function SplitBar({ split, className }: { split: TaskSplit; className: string }): ReactElement {
  return <span className={className} aria-hidden="true">
    {split.localTokens > 0 && <span data-where="local" style={{ flexGrow: split.localTokens }} />}
    {split.cloudTokens > 0 && <span data-where="cloud" style={{ flexGrow: split.cloudTokens }} />}
  </span>
}

/**
 * The composer's Hybrid indicator. Shown in Hybrid once the task has used a
 * model, and in any mode when a task ran both here and in the cloud.
 */
export function ModelSplitChip({ snap, onOpen }: { snap: Snapshot; onOpen: () => void }): ReactElement | null {
  return <ModelSplitChipView task={useTaskModels(snap)} hybrid={snap.runMode?.mode === 'hybrid'} onOpen={onOpen} />
}

export function ModelSplitChipView({ task, hybrid, onOpen }: { task: TaskModels | null; hybrid: boolean; onOpen: () => void }): ReactElement | null {
  if (!task?.models.length) return null
  const split = taskSplit(task.models)
  if (split.localShare === null || (!hybrid && (split.localTokens === 0 || split.cloudTokens === 0))) return null
  const percent = percentLabel(split)
  const spend = cloudSpend(split)
  const text = split.cloudTokens === 0 ? 'All on this Mac' : split.localTokens === 0 ? 'All in the cloud' : `${percent}% on this Mac`
  const title = [
    split.cloudTokens === 0
      ? 'This task ran entirely on this Mac.'
      : `This task: ${percent}% of tokens on this Mac, ${100 - percent}% in the cloud${spend ? ` (${spend})` : ''}.`,
    ...task.models.map(modelLine),
    'Open Usage for details.',
  ].join('\n')
  return <button className="cchip composer__text modelsplit" title={title} aria-label={`${text}${spend && split.cloudTokens > 0 ? `, cloud ${spend}` : ''}. Open usage.`} onClick={onOpen}>
    <SplitBar split={split} className="modelsplit__bar" />
    <span>{text}</span>
    {spend && split.cloudTokens > 0 && !split.cloudOnPlan && <span className="modelsplit__spend">{spend}</span>}
  </button>
}

/** The Usage panel's first section: this task, model by model. */
export function TaskModelsSection({ snap }: { snap: Snapshot }): ReactElement | null {
  return <TaskModelsView task={useTaskModels(snap)} hybrid={snap.runMode?.mode === 'hybrid'} />
}

export function TaskModelsView({ task, hybrid }: { task: TaskModels | null; hybrid: boolean }): ReactElement | null {
  if (!task) return null
  const split = taskSplit(task.models)
  const percent = percentLabel(split)
  const spend = cloudSpend(split)
  const showBar = split.localShare !== null && (hybrid || (split.localTokens > 0 && split.cloudTokens > 0))
  return <section className="usage__section" aria-labelledby="usage-task">
    <header className="usage__head">
      <h3 id="usage-task">This task</h3>
      <span className="usage__sub">{hybrid ? 'Hybrid · by tokens processed' : 'By tokens processed'}</span>
    </header>
    {!task.models.length && <p className="usage__empty">No model calls in this task yet.</p>}
    {showBar && <div className="tasksplit">
      <SplitBar split={split} className="tasksplit__bar" />
      <ul className="tasksplit__legend">
        <li data-where="local"><span className="tasksplit__swatch" aria-hidden="true" />On this Mac <b>{split.cloudTokens === 0 ? 100 : percent}%</b> · {compactTokens(split.localTokens)} tokens</li>
        <li data-where="cloud"><span className="tasksplit__swatch" aria-hidden="true" />Cloud <b>{split.cloudTokens === 0 ? 0 : 100 - percent}%</b> · {compactTokens(split.cloudTokens)} tokens{spend ? ` · ${spend}` : ''}</li>
        {split.unknownTokens > 0 && <li data-where="unknown">{compactTokens(split.unknownTokens)} tokens from a model whose location is unknown</li>}
      </ul>
    </div>}
    {task.models.length > 0 && <ul className="usage-models">{task.models.map(model => {
      const place = where(model)
      return <li key={`${model.role ?? ''}\u0000${model.model}\u0000${model.profile ?? ''}`} className="usage-model">
        <div className="usage-model__row">
          <span className="taskmodel__name">
            <span className="tasksplit__swatch" data-where={place} aria-hidden="true" />
            <strong className="usage-model__name" title={model.profile ? `${model.model} · ${model.profile}` : model.model}>{model.model}</strong>
          </span>
          <span className="usage-model__cost" data-unknown={(place === 'cloud' && model.costUsd === undefined && model.costBasis !== 'subscription') || undefined}>
            {place === 'local' ? 'On this Mac' : costLabel(model.costUsd, model.costBasis)}
          </span>
        </div>
        <div className="usage-model__meta">
          {modelRoleLabel(model.role)} · {place === 'unknown' ? 'location unknown' : place === 'local' ? 'on this Mac' : 'cloud'} · {modelUsageSummary(model)}{model.bytesSent ? ` · ${compactBytes(model.bytesSent)} sent` : ''}
        </div>
      </li>
    })}</ul>}
    {hybrid && task.advisor && <dl className="usage-plan__facts taskadvisor">
      <div><dt>Advisor</dt><dd>{task.advisor.profile}</dd></div>
      <div><dt>Advisor calls</dt><dd>{split.advisorCalls}</dd></div>
      <div><dt>Sent to the advisor</dt><dd>{split.bytesSent > 0 ? compactBytes(split.bytesSent) : 'Nothing yet'}</dd></div>
      <div><dt>Limits per task</dt><dd>{task.advisor.maxCalls} calls · {usd(task.advisor.maxUsd)}</dd></div>
    </dl>}
  </section>
}
