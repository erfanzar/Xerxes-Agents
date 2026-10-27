// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'
import { store, type Snapshot, type Store } from './store.js'
import { proposalKindLabel, proposalPreview, type LearningTarget, type Lesson, type ReflectionNotice, type StoredReflectionProposal } from './learning.js'
import { Icon } from './Icon.js'

type Actions = Pick<Store, 'refreshLearning' | 'reviewLearning' | 'openLearning'>
interface ReviewProps { readonly snap: Snapshot; readonly actions: Actions }

export function LearnedBadge({ snap }: { snap: Snapshot }): ReactElement | null {
  const count = (snap.learningCounts?.pending_proposals ?? 0) + (snap.learningCounts?.pending_lessons ?? 0)
  return count ? <span className="learned-badge" aria-label={`${count} pending learned items`}>{count}</span> : null
}

/** The click is the approval; reasons never travel through chat or slash text. */
export function LearningReviewControls({ target, id, approved = false, disabled = false, snap, actions }: ReviewProps & { target: LearningTarget; id: string; approved?: boolean; disabled?: boolean }): ReactElement {
  const busy = Boolean(disabled || snap.learningReviewBusy || snap.learningLoading || snap.connection !== 'online')
  const action = approved ? 'rollback' : 'reject'
  const label = approved ? 'Roll back' : 'Reject'
  return <div className="learning-review">
    {!approved && <button className="btn" disabled={busy} onClick={() => void actions.reviewLearning(target, id, 'approve')}>Approve</button>}
    <form className="learning-reason" onSubmit={event => {
      event.preventDefault()
      const input = event.currentTarget.elements.namedItem('reason') as HTMLInputElement | null
      void actions.reviewLearning(target, id, action, input?.value ?? '')
    }}>
      <label><span className="studio-muted">Reason to {approved ? 'roll back' : 'reject'} (required)</span><input name="reason" required disabled={busy} autoComplete="off" aria-label={`Reason to ${approved ? 'roll back' : 'reject'} ${target} ${id}`} /></label>
      <button type="submit" className="btn" disabled={busy}>{label}</button>
    </form>
  </div>
}

function ReviewResult({ target, id, snap }: { target: LearningTarget; id: string; snap: Snapshot }): ReactElement | null {
  const result = snap.learningResults?.[`${target}:${id}`]
  return result ? <>{result.report && <pre className="studio-source learning-report" role="status">{result.report}</pre>}{result.error && <p className="studio-error" role="alert">{result.error}</p>}</> : null
}

export function ProposalCard({ stored, snap, actions }: ReviewProps & { stored: StoredReflectionProposal }): ReactElement {
  const proposal = stored.proposal
  return <article className="railcard learned-card" tabIndex={-1} data-learning-proposal={stored.id} aria-label={`${proposalKindLabel(proposal.kind)}: ${proposal.target}`}>
    <div className="railcard__head"><strong className="railcard__title">{proposalKindLabel(proposal.kind)}</strong><span className="railcard__meta">{stored.status}</span></div>
    <h3>{proposal.target}</h3><p className="learning-prose">{proposal.rationale}</p>
    <h4>Quoted evidence</h4><pre className="studio-source">{proposal.evidence.join('\n')}</pre>
    <details className="learning-preview"><summary>Preview proposed content</summary>{proposalPreview(proposal).map(item => <section key={item.title}><h4>{item.title}</h4><pre className="studio-source">{item.content}</pre></section>)}</details>
    {stored.reason && <p className="learning-prose">Review reason: {stored.reason}</p>}
    {stored.outcome && <details className="learning-preview" open={stored.status === 'failed'}><summary>Review outcome</summary><pre className="studio-source">{JSON.stringify(stored.outcome, null, 2)}</pre></details>}
    {stored.hint && <p>{stored.hint}</p>}
    {stored.status === 'acknowledged' && !stored.hint && <p>Idea acknowledged. Use /plugin-creator to develop it; no plugin code was applied.</p>}
    <ReviewResult target="proposal" id={stored.id} snap={snap} />
    {stored.status === 'pending' && <LearningReviewControls target="proposal" id={stored.id} snap={snap} actions={actions} />}
  </article>
}

export function LessonCard({ lesson, snap, actions }: ReviewProps & { lesson: Lesson }): ReactElement {
  const evidence = lesson.evidence
  return <article className="railcard learned-card" aria-label={`Lesson: ${lesson.cue}`}>
    <div className="railcard__head"><strong className="railcard__title">{lesson.cue}</strong><span className="railcard__meta">Confidence {Math.round(lesson.confidence * 100)}%</span></div>
    <p className="learning-prose">{lesson.lesson}</p>
    <pre className="studio-source">{evidence.kind === 'shared' ? `Shared from ${evidence.host} via ${evidence.source} — ${evidence.summary}` : evidence.kind === 'tool_call' ? `${evidence.toolName} — ${evidence.summary}` : `Test (reported): ${evidence.command}\nexit code ${evidence.exitCode} — ${evidence.summary}`}</pre>
    <ReviewResult target="lesson" id={lesson.id} snap={snap} />
    {(lesson.reviewStatus === 'pending' || lesson.reviewStatus === 'approved') && <LearningReviewControls target="lesson" id={lesson.id} approved={lesson.reviewStatus === 'approved'} snap={snap} actions={actions} />}
  </article>
}

export function LearnedLessons({ snap, actions }: ReviewProps): ReactElement {
  return <section aria-label="Lessons">{(['pending', 'approved'] as const).map(status => {
    const lessons = snap.learning?.lessons.filter(lesson => lesson.reviewStatus === status) ?? []
    return <section key={status}><h3>{status === 'pending' ? 'Pending' : 'Approved'}</h3>{lessons.map(lesson => <LessonCard key={lesson.id} lesson={lesson} snap={snap} actions={actions} />)}{snap.learning && !lessons.length && <div className="studio-empty">{status === 'pending' ? 'No pending lessons. Evidence-backed cues and lessons recorded during work appear here for approval.' : 'No approved lessons. Approve a pending lesson to keep it; approved lessons can be rolled back with a reason.'}</div>}</section>
  })}</section>
}

export function LearnedPanel({ snap, actions = store }: { snap: Snapshot; actions?: Actions }): ReactElement {
  // The header badge sums both kinds; each sub-tab says its own share, and the
  // panel opens on the one that has something waiting (until the person picks).
  const pendingProposals = snap.learningCounts?.pending_proposals ?? 0
  const pendingLessons = snap.learningCounts?.pending_lessons ?? 0
  const [tab, setTab] = useState<'proposals' | 'lessons'>(() => !pendingProposals && pendingLessons ? 'lessons' : 'proposals')
  const root = useRef<HTMLDivElement>(null)
  const focusedRequest = useRef('')
  const picked = useRef(false)
  const choose = (next: 'proposals' | 'lessons') => { picked.current = true; setTab(next) }
  useEffect(() => {
    if (picked.current) return
    if (tab === 'proposals' && !pendingProposals && pendingLessons) setTab('lessons')
    if (tab === 'lessons' && !pendingLessons && pendingProposals) setTab('proposals')
  }, [pendingProposals, pendingLessons, tab])
  useEffect(() => { void actions.refreshLearning() }, [snap.cwd, snap.sessionKey, snap.learningOpenRevision, actions])
  useEffect(() => { if (snap.learningProposalId) setTab('proposals') }, [snap.learningOpenRevision])
  useEffect(() => {
    if (tab !== 'proposals' || !snap.learningProposalId || !snap.learning) return
    const request = `${snap.learningOpenRevision}:${snap.learningProposalId}`
    if (focusedRequest.current === request) return
    const card = Array.from(root.current?.querySelectorAll<HTMLElement>('[data-learning-proposal]') ?? []).find(element => element.dataset.learningProposal === snap.learningProposalId)
    if (card) {
      focusedRequest.current = request
      card.scrollIntoView({ block: 'center' }); card.focus({ preventScroll: true })
    }
  }, [snap.learningOpenRevision, snap.learning, tab])
  return <div ref={root} className="learned-panel">
    <p className="studio-muted">What Xerxes learned from its work. Proposals and lessons stay pending; nothing changes without your approval.</p>
    <div className="studio-tabs" role="group" aria-label="Learned views">
      <button className={tab === 'proposals' ? 'is-selected' : ''} aria-pressed={tab === 'proposals'} onClick={() => choose('proposals')}>Proposals{pendingProposals ? <span className="learned-badge" aria-label={`${pendingProposals} pending proposals`}>{pendingProposals}</span> : null}</button>
      <button className={tab === 'lessons' ? 'is-selected' : ''} aria-pressed={tab === 'lessons'} onClick={() => choose('lessons')}>Lessons{pendingLessons ? <span className="learned-badge" aria-label={`${pendingLessons} pending lessons`}>{pendingLessons}</span> : null}</button>
      <button disabled={snap.learningLoading || snap.learningReviewBusy || snap.connection !== 'online'} onClick={() => void actions.refreshLearning()}>Refresh</button>
    </div>
    {snap.learningLoading && <p className="studio-progress" role="status">Loading learned items…</p>}
    {snap.learningReviewBusy && <p className="studio-progress" role="status">Reviewing…</p>}
    {snap.learningError && <p className="studio-error" role="alert">{snap.learningError}</p>}
    {/* Reviewed/expired lessons leave learning.list; their reports must not vanish with the card. */}
    {Object.entries(snap.learningResults ?? {}).filter(([key]) => key.startsWith('lesson:') && !snap.learning?.lessons.some(lesson => key === `lesson:${lesson.id}`)).map(([key, result]) => <div key={key}>{result.report && <pre className="studio-source" role="status">{result.report}</pre>}{result.error && <p className="studio-error" role="alert">{result.error}</p>}</div>)}
    {tab === 'proposals' ? <section aria-label="Reflection proposals">
      {snap.learning?.proposals.map(stored => <ProposalCard key={stored.id} stored={stored} snap={snap} actions={actions} />)}
      {snap.learning && !snap.learning.proposals.length && <div className="studio-empty">No proposals yet. After a goal or substantial work, Xerxes may suggest a skill, agent update, or plugin idea with quoted evidence. Nothing is applied without approval.</div>}
    </section> : <LearnedLessons snap={snap} actions={actions} />}
  </div>
}

export function ReflectionCard({ reflection, snap, actions = store }: ReviewProps & { reflection: ReflectionNotice }): ReactElement {
  const stored = snap.learning?.proposals.find(row => row.id === reflection.id)
  // A notification says a proposal was created, not that it is still pending.
  const pending = stored?.status === 'pending'
  return <section className="railcard learned-card learned-notice" aria-label="What Xerxes learned">
    <div className="railcard__head"><Icon name="info" size={14} /><strong className="railcard__title">{proposalKindLabel(reflection.kind)}</strong>{stored && <span className="railcard__meta">{stored.status}</span>}</div>
    <h3>{reflection.target}</h3><p className="learning-prose">{reflection.rationale}</p><p className="studio-muted">{reflection.evidence_count} quoted evidence {reflection.evidence_count === 1 ? 'line' : 'lines'}. Nothing is applied without approval.</p>
    <button className="btn" onClick={() => actions.openLearning(reflection.id)}>Review</button>
    {(pending || !stored) && <LearningReviewControls target="proposal" id={reflection.id} snap={snap} actions={actions} disabled={!stored} />}
    {!stored && <p className="studio-muted">Open Review to load the proposal and its current status before deciding.</p>}
    {stored?.hint && <p>{stored.hint}</p>}
    <ReviewResult target="proposal" id={reflection.id} snap={snap} />
    {snap.learningError && <p className="studio-error" role="alert">{snap.learningError}</p>}
  </section>
}

export function ConnectedReflectionCard({ reflection }: { reflection: ReflectionNotice }): ReactElement {
  const snap = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  return <ReflectionCard reflection={reflection} snap={snap} actions={store} />
}
