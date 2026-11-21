// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useRef, useState, type ReactElement, type ReactNode } from 'react'

import { HubRequestError } from '../../client/hubConnection.js'
import {
  isSlackClientValue, isSlackUserId, sameSlackContact, slackAppManifest, slackCreateAppUrl, slackUnavailable,
  type SlackContact, type SlackDm, type SlackDraft, type SlackIdentity, type SlackStatus, type SupervisorSlackClient,
} from '../../client/supervisorSlack.js'
import { copyToClipboard } from './CopyButton.js'

const SLACK_UNAVAILABLE = 'Managed Slack sign-in is currently unavailable.'
const HUB_TOO_OLD = 'Your Hub runs an older Xerxes that can’t connect Slack from this app. Update the Hub, then press Refresh.'
const NO_HTTPS = 'Slack needs this Hub’s HTTPS address to send you back after signing in, and the Hub has none. Give the Hub a public HTTPS address, then press Refresh.'

/** A message the owner can act on; Hub messages for these codes are written for people and never carry the secret. */
function setupError(reason: unknown): string {
  if (reason instanceof HubRequestError) {
    if (reason.code === 'unknown_method' || (reason.code === 'invalid_request' && reason.message.startsWith('unknown hub method'))) return HUB_TOO_OLD
    if (reason.code === 'invalid_request' || reason.code === 'conflict' || reason.code === 'unconfigured') return reason.message
    if (reason.code === 'unavailable' || reason.code === 'rate_limited' || reason.code === 'timeout') return `${reason.message} Try again in a moment.`
  }
  if (reason instanceof Error && reason.message.startsWith('Paste the Client ID')) return reason.message
  return 'Could not save the Slack app. Press Refresh to check the Hub, then try again.'
}

/** One-time setup, all in the app: create the owner's own Slack app (pre-filled), then hand its credentials to the Hub. */
export function SlackAppSetup({ redirectUrl, disabled, openApp, onSave }: {
  readonly redirectUrl: string
  readonly disabled: boolean
  readonly openApp: () => void
  readonly onSave: (app: { readonly clientId: string; readonly clientSecret: string }) => void
}): ReactElement {
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')
  const [copied, setCopied] = useState<string>()
  const id = clientId.trim()
  const value = secret.trim()
  const valid = isSlackClientValue(id) && isSlackClientValue(value)
  return <form className="xslack__card xslack__setup" aria-label="Set up your XERXES Slack app" onSubmit={event => {
    event.preventDefault()
    if (disabled || !valid) return
    // The secret never stays in renderer state after it is handed to the Hub.
    setSecret('')
    onSave({ clientId: id, clientSecret: value })
  }}>
    <p>One-time setup, about a minute. After this, connecting is one click.</p>
    <ol className="xslack__steps">
      <li>
        <strong>Create your XERXES app in Slack.</strong>
        <span>Slack opens with everything filled in. Pick your workspace, click <em>Next</em>, then <em>Create</em>.</span>
        <div className="xslack__actions">
          <button type="button" className="btn btn--solid" disabled={disabled} onClick={openApp}>Create XERXES app in Slack</button>
          <button type="button" className="btn" disabled={disabled}
            onClick={() => void copyToClipboard(slackAppManifest(redirectUrl)).then(ok => setCopied(ok ? 'Manifest copied.' : 'Could not copy the manifest.'))}>Copy manifest</button>
        </div>
        <span className="studio-muted">If Slack opens without the details, choose <em>From a manifest</em> and paste the copied manifest.</span>
        {copied ? <span className="studio-muted" role="status">{copied}</span> : null}
      </li>
      <li>
        <strong>Paste its Client ID and Client Secret.</strong>
        <span>They’re on the app’s <em>Basic Information</em> page under <em>App Credentials</em>. Click <em>Show</em> to see the secret.</span>
        <div className="xslack__fields">
          <input aria-label="Slack Client ID" value={clientId} placeholder="Client ID" autoComplete="off" spellCheck={false}
            disabled={disabled} onChange={event => setClientId(event.target.value)} />
          <input type="password" aria-label="Slack Client Secret" value={secret} placeholder="Client Secret" autoComplete="off" spellCheck={false}
            disabled={disabled} onChange={event => setSecret(event.target.value)} />
          <button type="submit" className="btn btn--solid" disabled={disabled || !valid}>Save</button>
        </div>
      </li>
    </ol>
    <p className="studio-muted">XERXES acts as you in Slack (no bot), only in direct messages with people you approve, and replies start as drafts for your review. The secret stays on your Hub.</p>
  </form>
}
const SLACK_EXPIRED = 'Slack sign-in expired. Start again to connect.'
const SLACK_DENIED = 'Slack sign-in was declined. Nothing was connected.'

/** A local, single-flight request. Ignore completions after this screen is left. */
function useSlackRequest(): {
  readonly busy: boolean
  readonly run: (work: (current: () => boolean) => Promise<void>, onError: (error: unknown) => void) => void
} {
  const scope = useRef<object | undefined>(undefined)
  const pending = useRef(false)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    scope.current = {}
    pending.current = false
    return () => { scope.current = undefined }
  }, [])
  const run = useCallback((work: (current: () => boolean) => Promise<void>, onError: (error: unknown) => void): void => {
    const token = scope.current
    if (token === undefined || pending.current) return
    const current = (): boolean => scope.current === token
    pending.current = true
    setBusy(true)
    void work(current).catch(error => { if (current()) onError(error) }).finally(() => {
      if (current()) { pending.current = false; setBusy(false) }
    })
  }, [])
  return { busy, run }
}

function ExactIds({ contact }: { readonly contact: SlackIdentity }): ReactElement {
  return <span className="xslack__ids">Workspace <code>{contact.team_id}</code> · Member <code>{contact.user_id}</code></span>
}

function ConfirmSlack({ title, children, action, disabled, onCancel, onConfirm }: {
  readonly title: string
  readonly children: ReactNode
  readonly action: string
  readonly disabled: boolean
  readonly onCancel: () => void
  readonly onConfirm: () => void
}): ReactElement {
  return <section className="xslack__card xslack__confirm" aria-label={title}>
    <h4>{title}</h4>
    {children}
    <div className="xslack__actions">
      <button type="button" className="btn" disabled={disabled} onClick={onCancel}>Cancel</button>
      <button type="button" className="btn btn--solid" disabled={disabled} onClick={onConfirm}>{action}</button>
    </div>
  </section>
}

type Change = { kind: 'disconnect' } | { kind: 'add'; contact: SlackIdentity } | { kind: 'remove' | 'auto'; contact: SlackContact }

interface SlackWorkspaceProps {
  readonly client: SupervisorSlackClient
  readonly openOAuth: (url: string) => Promise<boolean>
}

export function SupervisorSlack({ client, openOAuth }: SlackWorkspaceProps): ReactElement {
  const { run, busy } = useSlackRequest()
  const [status, setStatus] = useState<SlackStatus>()
  const [error, setError] = useState<string>()
  const [oauthUntil, setOAuthUntil] = useState(0)
  const [oauthNote, setOAuthNote] = useState<string>()
  const [brokerPairing, setBrokerPairing] = useState<{ origin: string; code: string }>()
  const [userId, setUserId] = useState('')
  const [change, setChange] = useState<Change>()
  const [selected, setSelected] = useState<SlackContact>()
  const account = status?.connection ?? null
  const pending = status?.pending_identity
  // Hubs that report `app` let the owner set up their own Slack app here; older Hubs only offer operator sign-in.
  const appSource = status?.app
  const redirectUrl = status?.redirect_url ?? undefined
  const browserSignIn = status?.browser_sign_in ?? status?.configured ?? false
  const contact = selected === undefined ? undefined : status?.contacts.find(row => sameSlackContact(row, selected))
  const accept = useCallback((next: SlackStatus): void => {
    setStatus(next)
    if (next.connection !== null || next.pending_identity !== undefined || !next.configured) {
      setOAuthUntil(0)
      setOAuthNote(undefined)
      setBrokerPairing(undefined)
    } else if (next.oauth_status === 'denied' || next.oauth_status === 'expired') {
      setOAuthUntil(0)
      setOAuthNote(next.oauth_status === 'denied' ? SLACK_DENIED : SLACK_EXPIRED)
      setBrokerPairing(undefined)
    }
  }, [])
  const refresh = useCallback((): void => {
    run(async current => {
      setChange(undefined)
      const next = await client.status()
      if (current()) { accept(next); setError(undefined) }
    }, reason => {
      setStatus(undefined)
      setError(slackUnavailable(reason) ? SLACK_UNAVAILABLE : 'Could not read Slack status from the Hub.')
    })
  }, [accept, client, run])
  useEffect(refresh, [refresh])
  useEffect(() => {
    if (oauthUntil === 0 || busy) return
    const timer = setTimeout(() => {
      if (Date.now() >= oauthUntil) {
        setOAuthUntil(0)
        setOAuthNote('Slack sign-in took too long. Refresh its status or start again.')
        setBrokerPairing(undefined)
      }
      // Read the Hub at the deadline as well: Slack may already be waiting for confirmation.
      refresh()
    }, Math.max(0, Math.min(3_000, oauthUntil - Date.now())))
    return () => clearTimeout(timer)
  }, [busy, oauthUntil, refresh])

  function saveApp(app: { readonly clientId: string; readonly clientSecret: string }): void {
    if (account !== null || pending !== undefined) return
    run(async current => {
      setError(undefined)
      setOAuthNote(undefined)
      const next = await client.saveApp(app)
      if (current()) accept(next)
    }, reason => setError(setupError(reason)))
  }
  function removeApp(): void {
    run(async current => {
      setError(undefined)
      const next = await client.removeApp()
      if (current()) accept(next)
    }, reason => setError(setupError(reason)))
  }
  function openCreateApp(): void {
    if (redirectUrl === undefined) return
    run(async () => {
      setError(undefined)
      if (!(await openOAuth(slackCreateAppUrl(redirectUrl)))) throw new Error('Browser not opened')
    }, () => setError('Could not open Slack in your browser. Use Copy manifest and create the app at api.slack.com/apps.'))
  }
  function connect(): void {
    if (!browserSignIn || account !== null || pending !== undefined || oauthUntil > 0) return
    run(async current => {
      setError(undefined)
      setOAuthNote(undefined)
      setBrokerPairing(undefined)
      const { url, expiresAt, pairingCode } = await client.connect()
      if (!current()) return
      const deadline = expiresAt ?? Date.now() + (pairingCode === undefined ? 120_000 : 10 * 60_000)
      if (deadline <= Date.now()) { setOAuthNote(SLACK_EXPIRED); return }
      if (!(await openOAuth(url))) throw new Error('Browser not opened')
      if (current()) {
        setOAuthUntil(deadline)
        if (pairingCode !== undefined) setBrokerPairing({ origin: new URL(url).origin, code: pairingCode })
        setOAuthNote('Finish signing in to Slack in your browser, then return here to confirm your account.')
      }
    }, reason => setError(slackUnavailable(reason)
      ? SLACK_UNAVAILABLE : 'Could not open Slack sign-in. Refresh to check its status before trying again.'))
  }
  function confirmConnection(): void {
    if (pending === undefined) return
    run(async current => {
      setError(undefined)
      const next = await client.confirmConnection(pending)
      if (current()) accept(next)
    }, () => {
      setStatus(undefined)
      setError('Could not confirm this Slack account. Refresh to check whether the request expired or the account is connected.')
    })
  }
  function lookup(): void {
    if (account === null || !isSlackUserId(userId.trim())) return
    const ids = { team_id: account.team_id, user_id: userId.trim() }
    run(async current => {
      setError(undefined)
      const found = await client.lookup(ids)
      if (current()) setChange({ kind: 'add', contact: found })
    }, () => setError('No exact match could be verified. Check the Slack workspace and member IDs.'))
  }
  function mutate(action: () => Promise<SlackStatus>): void {
    run(async current => {
      setChange(undefined)
      setError(undefined)
      const next = await action()
      if (current()) { accept(next); setUserId('') }
    }, () => {
      setStatus(undefined)
      setSelected(undefined)
      setError('The Hub did not confirm the change. Refresh to check its current state; nothing will be retried automatically.')
    })
  }
  if (contact !== undefined) return <SlackDirectMessage key={`${account?.user_id}:${contact.team_id}:${contact.user_id}`} contact={contact} client={client} onBack={() => { setSelected(undefined); refresh() }} />
  const matchListed = status?.contacts.some(row => row.user_id === userId.trim()) ?? false
  const locked = busy || change !== undefined
  return <section className="xslack" aria-label="Slack controls" aria-busy={busy}>
    <header className="xslack__head">
      <div><span className="cap">Hub owner · Private</span><h2>Slack</h2><p>Manage who XERXES can reply to in Slack DMs.</p></div>
      <button type="button" className="btn" onClick={refresh} disabled={busy}>Refresh</button>
    </header>
    {error ? <p className="notice is-error" role="alert">{error}</p> : null}
    {status === undefined ? (error ? null : <p role="status">Reading Slack status…</p>) : <>
      <div className="xslack__card xslack__row">
        <div>
          <h3>{account !== null ? account.team_name || 'Connected to Slack' : pending !== undefined ? 'Awaiting confirmation' : 'Not connected'}</h3>
          {account !== null ? <ExactIds contact={account} /> : null}
          {account === null && pending === undefined
            ? <p>{browserSignIn ? 'Sign in with Slack to connect XERXES as you.'
              : appSource === 'none' ? (redirectUrl === undefined ? NO_HTTPS : 'Set up your XERXES Slack app once. After that, connecting is one click.')
              : HUB_TOO_OLD}</p> : null}
        </div>
        {account !== null ? <button type="button" className="btn" disabled={locked} onClick={() => setChange({ kind: 'disconnect' })}>Disconnect</button>
          : pending === undefined && browserSignIn ? <button type="button" className="btn btn--solid" disabled={locked || oauthUntil > 0} onClick={connect}>Sign in with Slack</button> : null}
      </div>
      {account === null && pending === undefined && appSource === 'none' && redirectUrl !== undefined && oauthUntil === 0
        ? <SlackAppSetup redirectUrl={redirectUrl} disabled={locked} openApp={openCreateApp} onSave={saveApp} /> : null}
      {account === null && pending === undefined && appSource === 'saved' && oauthUntil === 0
        ? <p className="studio-muted xslack__appnote">Using your XERXES Slack app.{' '}
          <button type="button" className="btn btn--link" disabled={locked} onClick={removeApp}>Use a different app</button></p> : null}
      {pending !== undefined ? <section className="xslack__card xslack__confirm" aria-label="Confirm your Slack account">
        <h4>Confirm your Slack account</h4>
        <ExactIds contact={pending} />
        <p>Confirm only if these IDs match the Slack workspace and account you just authorized. If they don’t match, do not confirm.</p>
        <button type="button" className="btn btn--solid" disabled={busy} onClick={confirmConnection}>Confirm Slack account</button>
      </section> : null}
      {brokerPairing !== undefined ? <section className="xslack__card" aria-label="Slack pairing code">
        <h4>Slack pairing code</h4>
        <p>Enter this code only on <strong>{brokerPairing.origin}</strong> in the browser. Do not share it.</p>
        <code>{brokerPairing.code}</code>
      </section> : null}
      {oauthNote ? <p className="notice" role="status">{oauthNote}</p> : null}
      {account !== null && change?.kind === 'disconnect' ? <ConfirmSlack title="Disconnect this Slack account?" action="Disconnect Slack" disabled={busy}
        onCancel={() => setChange(undefined)} onConfirm={() => mutate(() => client.disconnect(account))}>
        <ExactIds contact={account} /><p>XERXES will stop replying and drafting in this workspace.</p>
      </ConfirmSlack> : null}
      {account === null ? null : <>
        <div className="xslack__list">
          <h3>Approved contacts</h3>
          {status.contacts.length === 0 ? <p className="studio-muted">No contacts yet. Add someone by their exact Slack member ID.</p> : null}
          {status.contacts.map(row => <div className="xslack__card xslack__row" key={row.user_id}>
            <div><strong>{row.display_name || row.user_id}</strong><ExactIds contact={row} /></div>
            <div className="xslack__actions">
              <select aria-label={`Reply mode for ${row.user_id}`} value={row.mode} disabled={locked} onChange={event => {
                if (event.target.value === 'auto') setChange({ kind: 'auto', contact: row })
                else mutate(() => client.mode(row, 'draft'))
              }}>
                <option value="draft">Drafts for review</option><option value="auto">Auto-reply</option>
              </select>
              <button type="button" className="btn" aria-label={`Open DM with ${row.user_id}`} disabled={locked} onClick={() => setSelected(row)}>Open DM</button>
              <button type="button" className="btn" aria-label={`Remove ${row.user_id}`} disabled={locked} onClick={() => setChange({ kind: 'remove', contact: row })}>Remove</button>
            </div>
          </div>)}
          {change?.kind === 'remove' || change?.kind === 'auto' ? <ConfirmSlack
            title={change.kind === 'remove' ? 'Remove this contact?' : 'Allow automatic Slack replies?'} action={change.kind === 'remove' ? 'Remove contact' : 'Enable auto-reply'} disabled={busy}
            onCancel={() => setChange(undefined)} onConfirm={() => mutate(() => change.kind === 'remove' ? client.remove(change.contact) : client.mode(change.contact, 'auto'))}>
            <strong>{change.contact.display_name || change.contact.user_id}</strong><ExactIds contact={change.contact} />
            <p>{change.kind === 'remove' ? 'XERXES will stop replying to this contact.' : 'XERXES can send replies to this contact without asking you to review them.'}</p>
          </ConfirmSlack> : null}
        </div>
        <form className="xslack__card" aria-label="Add Slack contact" onSubmit={event => { event.preventDefault(); if (!locked && !matchListed) lookup() }}>
          <h3>Add a contact</h3>
          <p>In Slack, open the person’s profile and copy their member ID. The workspace ID is <code>{account.team_id}</code>.</p>
          <label htmlFor="xslack-member">Slack member ID</label>
          <div className="xslack__actions">
            <input id="xslack-member" value={userId} placeholder="U… or W…" autoComplete="off" spellCheck={false} disabled={locked} onChange={event => setUserId(event.target.value)} />
            <button type="submit" className="btn" disabled={locked || matchListed || !isSlackUserId(userId.trim())}>Look up exact ID</button>
          </div>
          {matchListed ? <p className="studio-muted">This member is already on the list.</p> : null}
        </form>
        {change?.kind === 'add' ? <ConfirmSlack title="Confirm this Slack contact" action="Add contact" disabled={busy}
          onCancel={() => setChange(undefined)} onConfirm={() => mutate(() => client.confirm(change.contact))}>
          <strong>{change.contact.display_name || change.contact.user_id}</strong><ExactIds contact={change.contact} />
          <p>Verify both IDs before adding. Replies start as drafts for review.</p>
        </ConfirmSlack> : null}
      </>}
    </>}
  </section>
}

/** A direct message never becomes a shared supervisor message or composer destination. */
export function SlackDirectMessage({ client, contact, onBack }: {
  readonly client: SupervisorSlackClient
  readonly contact: SlackContact
  readonly onBack: () => void
}): ReactElement {
  const { run, busy } = useSlackRequest()
  const [dm, setDm] = useState<SlackDm>()
  const [review, setReview] = useState<SlackDraft>()
  const [error, setError] = useState<string>()
  const [sent, setSent] = useState(false)
  const refresh = useCallback(() => run(async current => {
    setReview(undefined)
    setSent(false)
    const result = await client.dm(contact)
    if (current()) { setDm(result); setError(undefined) }
  }, () => { setDm(undefined); setError('Could not read this Slack DM. Refresh to try again.') }), [client, contact, run])
  useEffect(refresh, [refresh])
  function send(draft: SlackDraft): void {
    if (!dm || !sameSlackContact(dm, contact) || !dm.drafts.some(row => row.id === draft.id && row.text === draft.text)) return
    run(async current => {
      setReview(undefined)
      setSent(false)
      const result = await client.send(contact, draft)
      if (current()) { setDm(result); setSent(true) }
    }, () => {
      setDm(undefined)
      setError('The Hub did not confirm whether this draft was sent. Refresh and check the DM before taking any further action. Nothing was retried.')
    })
  }
  return <section className="xslack" aria-label="Private Slack direct message" aria-busy={busy}>
    <header className="xslack__head">
      <div><span className="cap">Slack DM · Private</span><h2>{contact.display_name || contact.user_id}</h2><ExactIds contact={contact} /></div>
      <div className="xslack__actions">
        <button type="button" className="btn" onClick={onBack} disabled={busy}>Back to Slack</button>
        <button type="button" className="btn" onClick={refresh} disabled={busy}>Refresh DM</button>
      </div>
    </header>
    <p className="studio-muted">This conversation is separate from XERXES Chat and Activity.</p>
    {error ? <p role="alert" className="notice is-error">{error}</p> : null}
    {sent ? <p role="status" className="notice">Draft sent to Slack.</p> : null}
    {dm === undefined ? (error ? null : <p role="status">Opening Slack DM…</p>) : <>
      <div className="xslack__list" aria-label="Slack messages">
        <h3>Messages</h3>
        {dm.messages.length === 0 ? <p className="studio-muted">No messages yet.</p> : dm.messages.map(message => <article className="xslack__card" key={message.id}>
          <div className="xslack__message-meta"><strong>{message.direction === 'inbound' ? contact.display_name || contact.user_id : 'XERXES'}</strong>
            {Number.isFinite(Date.parse(message.created_at)) ? <time dateTime={message.created_at}>{new Date(message.created_at).toLocaleString()}</time> : null}</div>
          <p className="xslack__text">{message.text}</p>
        </article>)}
      </div>
      <div className="xslack__list" aria-label="Slack drafts">
        <h3>Drafts for review</h3>
        {dm.drafts.length === 0 ? <p className="studio-muted">No pending drafts.</p> : dm.drafts.map(draft => review?.id === draft.id
          ? <ConfirmSlack key={draft.id} title="Review before sending" action="Send to Slack" disabled={busy}
            onCancel={() => setReview(undefined)} onConfirm={() => send(review)}>
            <p>Send to <strong>{contact.display_name || contact.user_id}</strong></p><ExactIds contact={contact} /><p className="xslack__text">{review.text}</p>
          </ConfirmSlack>
          : <article className="xslack__card" key={draft.id}>
            <p className="xslack__text">{draft.text}</p>
            <button type="button" className="btn" disabled={busy || review !== undefined} onClick={() => setReview(draft)}>Review draft</button>
          </article>)}
      </div>
    </>}
  </section>
}
