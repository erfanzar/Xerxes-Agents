// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { parseGoalInspection, type GoalInspection } from '../../ui/lib/goalInspection.js'
import { desktopCall, desktopError } from './desktopRpc.js'
import type { Snapshot } from './store.js'

type Decision = { session_id: string; goal_id: string; revision: number; criterion_id: string; description: string }
export function GoalInspector({ snap }: { snap: Snapshot }): ReactElement {
  const [inspection, setInspection] = useState<GoalInspection | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [decision, setDecision] = useState<Decision | null>(null)
  const [draft, setDraft] = useState('')
  const epoch = useRef(0)
  const pending = useRef(false)
  const load = async (): Promise<void> => {
    if (pending.current) return
    const version=++epoch.current
    setLoading(true); setError('')
    try { const value=await desktopCall(window.xerxes,snap.sessionKey,'goal.inspect'); const parsed=parseGoalInspection(value,snap.currentId); if(version===epoch.current)setInspection(parsed) }
    catch(failure){if(version===epoch.current)setError(desktopError(failure))}
    finally{if(version===epoch.current)setLoading(false)}
  }
  useEffect(()=>{pending.current=false;setSaving(false);void load();return()=>{epoch.current++}},[snap.currentId,snap.connection])
  const submit = async (): Promise<void> => {
    if (!decision || !draft.trim() || pending.current || snap.turnActive) return
    pending.current=true; const version=++epoch.current;setSaving(true);setError('');setNotice('')
    try {
      // This strict human-only RPC rejects session_key; bind the exact inspected revision.
      const { description: _description, ...binding }=decision
      const value=await desktopCall(window.xerxes,'','goal.decision',{...binding,summary:draft.trim()})
      const parsed=parseGoalInspection(value,decision.session_id)
      if(version===epoch.current){setInspection(parsed);setDecision(null);setDraft('');setNotice('Criterion accepted by you.')}
    }catch(failure){if(version===epoch.current)setError(desktopError(failure)+' Your note is kept. Refresh and review the criterion before retrying.')}
    finally{if(version===epoch.current){pending.current=false;setSaving(false)}}
  }
  const goal=inspection?.goal
  return <div className="goal-inspector">
    <div className="goal-inspector__toolbar">
      <button type="button" className="btn" disabled={loading||saving} onClick={()=>void load()}>Refresh goal details</button>
    </div>
    {loading&&<p className="studio-muted" role="status">Loading goal details…</p>}
    {error&&<p className="studio-error" role="alert">{error}</p>}
    {notice&&<p className="goal-inspector__notice" role="status">{notice}</p>}
    {goal&&<>
      {goal.currentMilestone&&(
        <div className="goal-inspector__milestone">
          <div className="goal-inspector__label">Current milestone</div>
          <p className="goal-inspector__milestone-text">{goal.currentMilestone}</p>
        </div>
      )}
      {goal.blockedReason&&<p className="studio-error">{goal.blockedReason.message}</p>}
      <dl className="context-breakdown"><dt>Revision</dt><dd>{goal.revision}</dd><dt>Rounds</dt><dd>{goal.roundsStarted} / {goal.maxGoalRounds===Number.MAX_SAFE_INTEGER?'Unlimited':goal.maxGoalRounds}</dd>
      {goal.maxTotalTokens!==undefined&&<><dt>Token budget</dt><dd>{goal.maxTotalTokens.toLocaleString()}</dd></>}
      {goal.maxDurationMs!==undefined&&<><dt>Time limit</dt><dd>{Math.ceil(goal.maxDurationMs/60000)} min</dd></>}
      {inspection.tokenUsage&&<><dt>Measured tokens</dt><dd>{(inspection.tokenUsage.inputTokens+inspection.tokenUsage.outputTokens).toLocaleString()}{!inspection.tokenUsage.complete?' (incomplete)':''}</dd><dt>Pending calls</dt><dd>{inspection.tokenUsage.pendingCalls}</dd></>}
      {inspection.continuation&&<><dt>Continuation</dt><dd>{inspection.continuation.state}</dd></>}
      </dl>
      {inspection.continuation?.reason && inspection.continuation.reason !== goal.blockedReason?.message && (
        <div className="goal-inspector__continuation">
          <div className="goal-inspector__label">Continuation details</div>
          <p className="studio-muted">{inspection.continuation.reason}</p>
        </div>
      )}
      <h4 className="goal-inspector__heading">Success criteria</h4>
      {!goal.criteria.length&&<p className="studio-muted">No criteria recorded.</p>}
      {goal.criteria.length>0&&(
        <ul className="goal-criteria">
          {goal.criteria.map(criterion=>(
            <li className="goal-criterion" key={criterion.id} data-accepted={criterion.evidence ? 'true' : undefined}>
              <div className="goal-criterion__head">
                <span className="goal-criterion__desc">{criterion.description}</span>
                {criterion.evidence&&(
                  <span className="goal-criterion__badge" data-kind={criterion.evidence.kind}>
                    {criterion.evidence.kind==='user-decision'?'Accepted by you':'Tool evidence'}
                  </span>
                )}
              </div>
              {criterion.evidence?.summary&&<p className="goal-criterion__evidence">{criterion.evidence.summary}</p>}
              <div className="goal-criterion__actions">
                <button
                  type="button"
                  className="btn"
                  disabled={saving||loading||snap.turnActive||!!error}
                  onClick={()=>{setDecision({session_id:inspection.sessionId,goal_id:goal.id,revision:goal.revision,criterion_id:criterion.id,description:criterion.description});setNotice('')}}
                >
                  Record acceptance
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {snap.turnActive&&<p className="studio-muted">Wait for the turn to finish before recording an acceptance.</p>}
    </>}
    {inspection&&!goal&&<p className="studio-muted">No goal in this session.</p>}
    {decision&&(
      <form className="goal-decision" onSubmit={e=>{e.preventDefault();void submit()}}>
        <div className="goal-decision__head">
          <div className="goal-inspector__label">Accept this criterion</div>
          <p className="goal-decision__desc">{decision.description}</p>
        </div>
        <label className="field">Acceptance note<textarea aria-label="Acceptance note" required rows={3} value={draft} disabled={saving} onChange={e=>setDraft(e.target.value)}/></label>
        <div className="lsp-actions">
          <button type="submit" className="btn btn--solid" disabled={saving||loading||snap.turnActive||!!error||!draft.trim()}>Accept criterion</button>
          <button type="button" className="btn btn--ghost" disabled={saving} onClick={()=>{setDecision(null);setDraft('')}}>Cancel</button>
        </div>
      </form>
    )}
  </div>
}
export function GoalInspectorDisclosure({snap}:{snap:Snapshot}):ReactElement{
  const [open,setOpen]=useState(false)
  return <details className="goal-inspection" onToggle={e=>setOpen(e.currentTarget.open)}><summary>Criteria & evidence</summary>{open&&<GoalInspector key={snap.currentId} snap={snap}/>}</details>
}
