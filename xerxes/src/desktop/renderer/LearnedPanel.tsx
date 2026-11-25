// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, useSyncExternalStore, type ReactElement, type ReactNode } from 'react'
import { isExternalAcpBinding } from './AcpRuntimePicker.js'
import { store, type Snapshot, type Store } from './store.js'
import { proposalKindLabel, proposalPreview, type LearningTarget, type Lesson, type ReflectionNotice, type StoredReflectionProposal } from './learning.js'
import { Icon } from './Icon.js'

type Actions = Pick<Store, 'refreshLearning' | 'reviewLearning' | 'reviewAllLearning' | 'openLearning'>
interface ReviewProps { readonly snap: Snapshot; readonly actions: Actions }

export interface LearningReviewControlsProps extends ReviewProps {
  readonly target: LearningTarget
  readonly id: string
  readonly approved?: boolean
  readonly disabled?: boolean
  readonly initialReasonOpen?: boolean
  /** Optional leading action (e.g. `Review` in inline ReflectionCard) placed on the same compact row. */
  readonly children?: ReactNode
}

export interface LearningReviewControlsViewProps extends Omit<LearningReviewControlsProps, 'initialReasonOpen'> {
  readonly reasonOpen: boolean
  readonly onOpenReason: () => void
  readonly onCancelReason: () => void
  readonly reasonError?: string
  readonly onReasonError?: (message: string) => void
}

export function LearnedBadge({ snap }: { snap: Snapshot }): ReactElement | null {
  if (isExternalAcpBinding(snap.acpBinding)) return null
  const count = (snap.learningCounts?.pending_proposals ?? 0) + (snap.learningCounts?.pending_lessons ?? 0)
  return count ? <span className="learned-badge" aria-label={`${count} pending learned items`}>{count}</span> : null
}

/** Pure presentational review bar: compact single action row by default; reject/rollback reason discloses on demand. */
export function LearningReviewControlsView({
  target,
  id,
  approved = false,
  disabled = false,
  snap,
  actions,
  reasonOpen,
  onOpenReason,
  onCancelReason,
  reasonError = '',
  onReasonError,
  children,
}: LearningReviewControlsViewProps): ReactElement {
  const busy = Boolean(disabled || snap.learningReviewBusy || snap.learningLoading || snap.connection !== 'online')
  const action = approved ? 'rollback' : 'reject'
  const label = approved ? 'Roll back' : 'Reject'
  const confirmLabel = approved ? 'Confirm roll back' : 'Confirm reject'
  const showReasonForm = reasonOpen && !disabled
  return (
    <div className={`learning-review${showReasonForm ? ' is-open' : ''}`}>
      {children}
      {!approved && (
        <button
          type="button"
          className="btn btn--solid learning-review__approve"
          disabled={busy}
          onClick={() => void actions.reviewLearning(target, id, 'approve')}
        >
          Approve
        </button>
      )}
      {!showReasonForm ? (
        <button
          type="button"
          className="btn"
          disabled={busy}
          aria-expanded={false}
          onClick={() => { onReasonError?.(''); onOpenReason() }}
        >
          {label}
        </button>
      ) : (
        <form
          className="learning-reason"
          onSubmit={event => {
            event.preventDefault()
            const input = event.currentTarget.elements.namedItem('reason') as HTMLInputElement | null
            const reason = input?.value ?? ''
            if (!reason.trim()) {
              onReasonError?.(`Enter a reason to ${approved ? 'roll back' : 'reject'}.`)
              return
            }
            onReasonError?.('')
            void actions.reviewLearning(target, id, action, reason)
          }}
        >
          <label>
            <span className="studio-muted">Reason to {approved ? 'roll back' : 'reject'} (required)</span>
            <input
              name="reason"
              required
              autoFocus
              disabled={busy}
              autoComplete="off"
              aria-invalid={reasonError ? true : undefined}
              placeholder={approved ? 'Why roll back this lesson…' : 'Why reject this proposal…'}
              aria-label={`Reason to ${approved ? 'roll back' : 'reject'} this ${target}`}
              onChange={() => { if (reasonError) onReasonError?.('') }}
              onKeyDown={event => {
                if (event.key === 'Escape') {
                  event.preventDefault()
                  event.stopPropagation()
                  onReasonError?.('')
                  onCancelReason()
                }
              }}
            />
          </label>
          <button type="submit" className="btn btn--danger" disabled={busy}>
            {confirmLabel}
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            disabled={busy}
            onClick={() => { onReasonError?.(''); onCancelReason() }}
          >
            Cancel
          </button>
          {reasonError && <p className="studio-error" role="alert">{reasonError}</p>}
        </form>
      )}
    </div>
  )
}

/** The click is the approval; reasons never travel through chat or slash text. */
export function LearningReviewControls({
  initialReasonOpen = false,
  ...props
}: LearningReviewControlsProps): ReactElement {
  const [reasonOpen, setReasonOpen] = useState(initialReasonOpen)
  const [reasonError, setReasonError] = useState('')
  useEffect(() => {
    if (props.disabled) {
      setReasonOpen(false)
      setReasonError('')
    }
  }, [props.disabled, props.id, props.approved])
  return (
    <LearningReviewControlsView
      {...props}
      reasonOpen={reasonOpen}
      reasonError={reasonError}
      onReasonError={setReasonError}
      onOpenReason={() => setReasonOpen(true)}
      onCancelReason={() => setReasonOpen(false)}
    />
  )
}

function ReviewResult({ target, id, snap }: { target: LearningTarget; id: string; snap: Snapshot }): ReactElement | null {
  const result = snap.learningResults?.[`${target}:${id}`]
  return result ? <>{result.report && <pre className="studio-source learning-report" role="status">{result.report}</pre>}{result.error && <p className="studio-error" role="alert">{result.error}</p>}</> : null
}

const OUTCOME_KEY_LABELS: Readonly<Record<string, string>> = {
  ok: 'Result',
  error: 'Error',
  report: 'Report',
  path: 'Path',
  filePath: 'Path',
  file_path: 'Path',
  written: 'Written files',
  findings: 'Findings',
  warnings: 'Warnings',
  message: 'Message',
  status: 'Status',
  target: 'Target',
}

function outcomeFieldLabel(key: string): string {
  const mapped = OUTCOME_KEY_LABELS[key]
  if (mapped) return mapped
  const words = key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim()
  return words ? words[0]!.toUpperCase() + words.slice(1) : key
}

function formatOutcomeScalar(key: string, value: unknown): string {
  if (key === 'ok' && typeof value === 'boolean') return value ? 'Applied' : 'Failed'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return 'None'
  if (Array.isArray(value)) {
    if (value.length === 0) return 'None'
    return value.map(item => formatOutcomeScalar('', item)).join('\n')
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined)
    if (entries.length === 0) return 'None'
    return entries.map(([k, v]) => `${outcomeFieldLabel(k)}: ${formatOutcomeScalar(k, v)}`).join('\n')
  }
  return String(value)
}

/** Format a proposal's structured review outcome into human-readable key/value rows instead of raw JSON. */
export function formatOutcomeEntries(outcome: Readonly<Record<string, unknown>>): Array<{ readonly label: string; readonly value: string }> {
  return Object.entries(outcome)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => ({ label: outcomeFieldLabel(key), value: formatOutcomeScalar(key, value) }))
}

export function ProposalCard({ stored, snap, actions }: ReviewProps & { stored: StoredReflectionProposal }): ReactElement {
  const proposal = stored.proposal
  const outcomeRows = stored.outcome ? formatOutcomeEntries(stored.outcome) : []
  const statusLabel = stored.status ? stored.status[0]!.toUpperCase() + stored.status.slice(1) : ''
  return <article className="railcard learned-card" tabIndex={-1} data-learning-proposal={stored.id} aria-label={`${proposalKindLabel(proposal.kind)}: ${proposal.target}`}>
    <div className="railcard__head"><strong className="railcard__title">{proposalKindLabel(proposal.kind)}</strong><span className="railcard__meta">{statusLabel}</span></div>
    <h4>{proposal.target}</h4><p className="learning-prose">{proposal.rationale}</p>
    <h5>Quoted evidence</h5><pre className="studio-source">{proposal.evidence.join('\n')}</pre>
    <details className="learning-preview"><summary>Preview proposed content</summary>{proposalPreview(proposal).map(item => <section key={item.title}><h5>{item.title}</h5><pre className="studio-source">{item.content}</pre></section>)}</details>
    {stored.reason && <p className="learning-prose">Review reason: {stored.reason}</p>}
    {outcomeRows.length > 0 && <details className="learning-preview" open={stored.status === 'failed'}><summary>Review outcome</summary><dl className="learning-outcome">{outcomeRows.map(row => <div key={row.label} className="learning-outcome__row"><dt>{row.label}</dt><dd>{row.value}</dd></div>)}</dl></details>}
    {stored.hint && <p>{stored.hint}</p>}
    {stored.status === 'acknowledged' && !stored.hint && <p>Idea acknowledged. Use <code>/plugin-creator</code> to develop it; no plugin code was applied.</p>}
    <ReviewResult target="proposal" id={stored.id} snap={snap} />
    {stored.status === 'pending' && <LearningReviewControls target="proposal" id={stored.id} snap={snap} actions={actions} />}
  </article>
}

export function LessonCard({ lesson, snap, actions }: ReviewProps & { lesson: Lesson }): ReactElement {
  const evidence = lesson.evidence
  return <article className="railcard learned-card" aria-label={`Lesson: ${lesson.cue}`}>
    <div className="railcard__head"><strong className="railcard__title">{lesson.cue}</strong><span className="railcard__meta">Confidence {Math.round(lesson.confidence * 100)}%</span></div>
    <p className="learning-prose">{lesson.lesson}</p>
    <h4>Evidence</h4>
    <pre className="studio-source">{evidence.kind === 'shared' ? `Shared from ${evidence.host} via ${evidence.source} — ${evidence.summary}` : evidence.kind === 'tool_call' ? `${evidence.toolName} — ${evidence.summary}` : `Test (reported): ${evidence.command}\nexit code ${evidence.exitCode} — ${evidence.summary}`}</pre>
    <ReviewResult target="lesson" id={lesson.id} snap={snap} />
    {(lesson.reviewStatus === 'pending' || lesson.reviewStatus === 'approved') && <LearningReviewControls target="lesson" id={lesson.id} approved={lesson.reviewStatus === 'approved'} snap={snap} actions={actions} />}
  </article>
}

export function LearnedLessons({ snap, actions }: ReviewProps): ReactElement {
  const [confirmApproveAll, setConfirmApproveAll] = useState(false)
  const pendingCount = snap.learning?.lessons.filter(lesson => lesson.reviewStatus === 'pending').length ?? 0
  const approveDisabled = snap.learningReviewBusy || snap.learningLoading || snap.connection !== 'online'
  useEffect(() => {
    if (pendingCount === 0 || snap.learningReviewBusy) setConfirmApproveAll(false)
  }, [pendingCount, snap.learningReviewBusy])
  return <section className="learned-lessons" aria-label="Lessons">{(['pending', 'approved'] as const).map(status => {
    const lessons = snap.learning?.lessons.filter(lesson => lesson.reviewStatus === status) ?? []
    const title = status === 'pending' ? 'Pending' : 'Approved'
    const headingId = `learned-${status}-title`
    return <section className="learned-group" key={status} aria-labelledby={headingId}>
      <div className="learned-group__head">
        <div className="learned-group__heading">
          <h3 id={headingId}>{title}</h3>
          <p>{lessons.length === 1 ? '1 lesson' : `${lessons.length} lessons`}{status === 'pending' ? ' awaiting review' : ' in this workspace'}</p>
        </div>
        {status === 'pending' && lessons.length > 0 && (
          confirmApproveAll ? (
            <div className="learned-group__confirm" role="group" aria-label={`Confirm approving all ${lessons.length} pending lessons`}>
              <button
                type="button"
                className="btn btn--solid learned-group__approve"
                disabled={approveDisabled}
                onClick={() => { setConfirmApproveAll(false); void actions.reviewAllLearning('lesson') }}
              >
                Confirm approve ({lessons.length})
              </button>
              <button type="button" className="btn btn--ghost" disabled={approveDisabled} onClick={() => setConfirmApproveAll(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <button type="button" className="btn btn--solid learned-group__approve" disabled={approveDisabled} onClick={() => setConfirmApproveAll(true)}>
              Approve all ({lessons.length})
            </button>
          )
        )}
      </div>
      <div className="learned-group__items">
        {lessons.map(lesson => <LessonCard key={lesson.id} lesson={lesson} snap={snap} actions={actions} />)}
        {snap.learning && !lessons.length && <div className="studio-empty">{status === 'pending' ? 'No pending lessons. Evidence-backed cues and lessons recorded during work appear here for approval.' : 'No approved lessons. Approve a pending lesson to keep it; approved lessons can be rolled back with a reason.'}</div>}
      </div>
    </section>
  })}</section>
}

export function LearnedPanel({ snap, actions = store }: { snap: Snapshot; actions?: Actions }): ReactElement {
  // The header badge sums both kinds; each sub-tab says its own share, and the
  // panel opens on the one that has something waiting (until the person picks).
  const pendingProposals = snap.learningCounts?.pending_proposals ?? 0
  const pendingLessons = snap.learningCounts?.pending_lessons ?? 0
  const [tab, setTab] = useState<'proposals' | 'lessons'>(() => !pendingProposals && pendingLessons ? 'lessons' : 'proposals')
  const [confirmApproveAllProposals, setConfirmApproveAllProposals] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const focusedRequest = useRef('')
  const picked = useRef(false)
  const choose = (next: 'proposals' | 'lessons') => { picked.current = true; setConfirmApproveAllProposals(false); setTab(next) }
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
  const pendingTotal = pendingProposals + pendingLessons
  const loadedPendingProposals = snap.learning?.proposals.filter(proposal => proposal.status === 'pending').length ?? 0
  const approveProposalsDisabled = snap.learningReviewBusy || snap.learningLoading || snap.connection !== 'online'
  useEffect(() => {
    if (loadedPendingProposals === 0 || snap.learningReviewBusy) setConfirmApproveAllProposals(false)
  }, [loadedPendingProposals, snap.learningReviewBusy])
  return <div ref={root} className="learned-panel">
    <div className="learned-panel__intro">
      <div>
        <h2>Learned</h2>
        <p className="studio-muted">Review what Xerxes learned from its work. Nothing changes without your approval.</p>
      </div>
      <span className="learned-panel__summary">{pendingTotal === 0 ? 'Nothing awaiting review' : `${pendingTotal} ${pendingTotal === 1 ? 'item' : 'items'} awaiting review`}</span>
    </div>
    <div className="learned-panel__toolbar">
      <div className="learned-panel__tabs" role="tablist" aria-label="Learned views">
        <button type="button" role="tab" aria-selected={tab === 'proposals'} className={tab === 'proposals' ? 'is-selected' : ''} onClick={() => choose('proposals')}>Proposals{pendingProposals ? <span className="learned-badge" aria-label={`${pendingProposals} pending proposals`}>{pendingProposals}</span> : null}</button>
        <button type="button" role="tab" aria-selected={tab === 'lessons'} className={tab === 'lessons' ? 'is-selected' : ''} onClick={() => choose('lessons')}>Lessons{pendingLessons ? <span className="learned-badge" aria-label={`${pendingLessons} pending lessons`}>{pendingLessons}</span> : null}</button>
      </div>
      <button type="button" className="btn btn--ghost learned-panel__refresh" disabled={snap.learningLoading || snap.learningReviewBusy || snap.connection !== 'online'} onClick={() => void actions.refreshLearning()}>Refresh</button>
    </div>
    {snap.learningLoading && <p className="studio-progress" role="status">Loading learned items…</p>}
    {snap.learningReviewBusy && <p className="studio-progress" role="status">Reviewing…</p>}
    {snap.learningError && <p className="studio-error" role="alert">{snap.learningError}</p>}
    {/* Reviewed items may leave learning.list; their outcome must not vanish with the card. */}
    {Object.entries(snap.learningResults ?? {}).filter(([key]) => key.startsWith('lesson:') ? !snap.learning?.lessons.some(lesson => key === `lesson:${lesson.id}`) : key.startsWith('proposal:') && !snap.learning?.proposals.some(proposal => key === `proposal:${proposal.id}`)).map(([key, result]) => <div key={key}>{result.report && <pre className="studio-source" role="status">{result.report}</pre>}{result.error && <p className="studio-error" role="alert">{result.error}</p>}</div>)}
    {tab === 'proposals' ? <section aria-label="Reflection proposals">
      <div className="learned-group__head">
        <div className="learned-group__heading"><h3>Proposals</h3><p>{loadedPendingProposals} awaiting review</p></div>
        {loadedPendingProposals > 0 && (
          confirmApproveAllProposals ? (
            <div className="learned-group__confirm" role="group" aria-label={`Confirm approving all ${loadedPendingProposals} pending proposals`}>
              <button
                type="button"
                className="btn btn--solid learned-group__approve"
                disabled={approveProposalsDisabled}
                onClick={() => { setConfirmApproveAllProposals(false); void actions.reviewAllLearning('proposal') }}
              >
                Confirm approve ({loadedPendingProposals})
              </button>
              <button type="button" className="btn btn--ghost" disabled={approveProposalsDisabled} onClick={() => setConfirmApproveAllProposals(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <button type="button" className="btn btn--solid learned-group__approve" disabled={approveProposalsDisabled} onClick={() => setConfirmApproveAllProposals(true)}>
              Approve all ({loadedPendingProposals})
            </button>
          )
        )}
      </div>
      {snap.learning?.proposals.map(stored => <ProposalCard key={stored.id} stored={stored} snap={snap} actions={actions} />)}
      {snap.learning && !snap.learning.proposals.length && <div className="studio-empty">No proposals yet. After a goal or substantial work, Xerxes may suggest a skill, agent update, or plugin idea with quoted evidence. Nothing is applied without approval.</div>}
    </section> : <LearnedLessons snap={snap} actions={actions} />}
  </div>
}

export function ReflectionCard({ reflection, snap, actions = store }: ReviewProps & { reflection: ReflectionNotice }): ReactElement | null {
  if (isExternalAcpBinding(snap.acpBinding)) return null
  const stored = snap.learning?.proposals.find(row => row.id === reflection.id)
  // A notification says a proposal was created, not that it is still pending.
  const pending = stored?.status === 'pending'
  const reviewButton = <button type="button" className="btn" onClick={() => actions.openLearning(reflection.id)}>Review</button>
  return <section className="railcard learned-card learned-notice" aria-label="What Xerxes learned">
    <div className="railcard__head"><Icon name="info" size={14} /><strong className="railcard__title">{proposalKindLabel(reflection.kind)}</strong>{stored && <span className="railcard__meta">{stored.status}</span>}</div>
    <h3>{reflection.target}</h3><p className="learning-prose">{reflection.rationale}</p><p className="studio-muted">{reflection.evidence_count} quoted evidence {reflection.evidence_count === 1 ? 'line' : 'lines'}. Nothing is applied without approval.</p>
    {(pending || !stored) ? (
      <LearningReviewControls target="proposal" id={reflection.id} snap={snap} actions={actions} disabled={!stored}>
        {reviewButton}
      </LearningReviewControls>
    ) : (
      <div className="learning-review">
        {reviewButton}
      </div>
    )}
    {!stored && <p className="studio-muted">Open Review to load the proposal and its current status before deciding.</p>}
    {stored?.hint && <p>{stored.hint}</p>}
    <ReviewResult target="proposal" id={reflection.id} snap={snap} />
    {snap.learningError && <p className="studio-error" role="alert">{snap.learningError}</p>}
  </section>
}

export function ConnectedReflectionCard({ reflection }: { reflection: ReflectionNotice }): ReactElement | null {
  const snap = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  if (isExternalAcpBinding(snap.acpBinding)) return null
  return <ReflectionCard reflection={reflection} snap={snap} actions={store} />
}
