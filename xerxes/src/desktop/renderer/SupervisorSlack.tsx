// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useId, useRef, useState, type ReactElement, type ReactNode } from 'react'

import { HubRequestError } from '../../client/hubConnection.js'
import {
  isSlackClientValue, isSlackConfigToken, isSlackUserId, sameSlackContact, SLACK_CONFIG_TOKEN_PAGE, slackAppManifest, slackCreateAppUrl, slackUnavailable,
  type SlackContact, type SlackDm, type SlackDraft, type SlackIdentity, type SlackStatus, type SupervisorSlackClient,
} from '../../client/supervisorSlack.js'
import { copyToClipboard } from './CopyButton.js'
import { SelectField } from './SelectField.js'

const SLACK_UNAVAILABLE = 'Managed Slack sign-in is currently unavailable.'
const HUB_TOO_OLD = 'Your Hub runs an older Xerxes that can’t connect Slack from this app. Update the Hub, then press Refresh.'
const NO_HTTPS = 'Slack needs this Hub’s HTTPS address to send you back after signing in, and the Hub has none. Give the Hub a public HTTPS address, then press Refresh.'

/** A message the owner can act on; Hub messages for these codes are written for people and never carry the secret. */
export function setupError(reason: unknown): string {
  if (reason instanceof HubRequestError) {
    if (reason.code === 'unknown_method' || (reason.code === 'invalid_request' && reason.message.startsWith('unknown hub method'))) return HUB_TOO_OLD
    if (reason.code === 'invalid_request' || reason.code === 'conflict' || reason.code === 'unconfigured') return reason.message
    if (reason.code === 'unavailable' || reason.code === 'rate_limited' || reason.code === 'timeout') {
      return /try again/iu.test(reason.message) ? reason.message : `${reason.message} Try again in a moment.`
    }
  }
  if (reason instanceof Error && (reason.message.startsWith('Paste the Client ID') || reason.message.startsWith('Paste the Access Token'))) return reason.message
  return 'Could not save the Slack app. Press Refresh to check the Hub, then try again.'
}

export class BrowserNotOpenedError extends Error {
  constructor() {
    super('Browser not opened')
    this.name = 'BrowserNotOpenedError'
  }
}

export function connectError(reason: unknown): string {
  if (reason instanceof BrowserNotOpenedError) return 'Could not open Slack in your browser. Click Sign in with Slack to try again.'
  if (slackUnavailable(reason)) return SLACK_UNAVAILABLE
  if (reason instanceof HubRequestError && reason.code === 'conflict') return reason.message
  return 'Could not open Slack sign-in. Refresh to check its status before trying again.'
}

export type SlackSignInOutcome =
  | { readonly kind: 'abandoned' }
  | { readonly kind: 'expired'; readonly note: string }
  | { readonly kind: 'opened'; readonly until: number; readonly note: string; readonly pairing?: { readonly origin: string; readonly code: string } }

export async function launchSlackSignIn(options: {
  readonly client: Pick<SupervisorSlackClient, 'connect'>
  readonly openOAuth: (url: string) => Promise<boolean>
  readonly current: () => boolean
  readonly now?: () => number
}): Promise<SlackSignInOutcome> {
  const { url, expiresAt, pairingCode } = await options.client.connect()
  if (!options.current()) return { kind: 'abandoned' }
  const now = (options.now ?? Date.now)()
  const deadline = expiresAt ?? now + (pairingCode === undefined ? 120_000 : 10 * 60_000)
  if (deadline <= now) return { kind: 'expired', note: SLACK_EXPIRED }
  let opened = false
  try { opened = await options.openOAuth(url) } catch { opened = false }
  if (!opened) throw new BrowserNotOpenedError()
  if (!options.current()) return { kind: 'abandoned' }
  return {
    kind: 'opened',
    until: deadline,
    note: 'Finish signing in to Slack in your browser, then return here to confirm your account.',
    ...(pairingCode === undefined ? {} : { pairing: { origin: new URL(url).origin, code: pairingCode } }),
  }
}

/**
 * One-time setup, all in the app. Automatic: the Hub creates the owner's XERXES app with a one-time
 * configuration token and starts Sign in with Slack. Manual (fallback): create it from the pre-filled
 * manifest, then paste its Client ID and Secret.
 */
export function SlackAppSetup({ redirectUrl, disabled, openApp, openTokenPage, onCreate, onSave }: {
  readonly redirectUrl: string
  readonly disabled: boolean
  readonly openApp: () => void
  readonly openTokenPage: () => void
  readonly onCreate: (configToken: string) => void
  readonly onSave: (app: { readonly clientId: string; readonly clientSecret: string }) => void
}): ReactElement {
  const [configToken, setConfigToken] = useState('')
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')
  const [copied, setCopied] = useState<string>()
  const baseId = useId()
  const token = configToken.trim()
  const tokenValid = isSlackConfigToken(token)
  const tokenInvalid = Boolean(token && !tokenValid)
  const id = clientId.trim()
  const value = secret.trim()
  const valid = isSlackClientValue(id) && isSlackClientValue(value)
  return <div className="xslack__card xslack__setup">
    <form className="xslack__auto" aria-label="Set up Slack automatically" onSubmit={event => {
      event.preventDefault()
      if (disabled || !tokenValid) return
      // The token is used once by the Hub and never kept in renderer state.
      setConfigToken('')
      onCreate(token)
    }}>
      <p>One-time setup. XERXES creates its Slack app for you, then you approve it in Slack.</p>
      <ol className="xslack__steps">
        <li>
          <strong>Get a configuration token from Slack.</strong>
          <span>On the page that opens, scroll to <em>Your App Configuration Tokens</em>, choose <em>Generate Token</em>, pick your workspace, then copy the <em>Access Token</em>.</span>
          <div className="xslack__actions">
            <button type="button" className="btn" disabled={disabled} onClick={openTokenPage}>Open Slack token page</button>
          </div>
        </li>
        <li>
          <strong>Paste it and let XERXES do the rest.</strong>
          <span>It starts with <code>xoxe.xoxp-</code>. It’s used once to create the app and isn’t stored.</span>
          <label className="sr-only" htmlFor={`${baseId}-token`}>Slack configuration token</label>
          <div className="xslack__fields">
            <input id={`${baseId}-token`} type="password" aria-label="Slack configuration token" value={configToken} placeholder="xoxe.xoxp-…" autoComplete="off" spellCheck={false}
              aria-invalid={tokenInvalid ? true : undefined} {...(tokenInvalid ? { 'aria-describedby': `${baseId}-token-err` } : {})}
              disabled={disabled} onChange={event => setConfigToken(event.target.value)} />
            <button type="submit" className="btn btn--solid" disabled={disabled || !tokenValid}>Set up automatically</button>
          </div>
          {tokenInvalid ? <span id={`${baseId}-token-err`} role="alert" className="studio-error">That isn’t a configuration token. Copy the <em>Access Token</em> (not the Refresh Token); it starts with <code>xoxe.xoxp-</code>.</span> : null}
        </li>
      </ol>
    </form>
    <details className="xslack__manual">
      <summary>Set it up by hand instead</summary>
    <form aria-label="Set up your XERXES Slack app by hand" onSubmit={event => {
    event.preventDefault()
    if (disabled || !valid) return
    // The secret never stays in renderer state after it is handed to the Hub.
    setSecret('')
    onSave({ clientId: id, clientSecret: value })
  }}>
    <ol className="xslack__steps">
      <li>
        <strong>Create your XERXES app in Slack.</strong>
        <span>Slack opens with everything filled in. Pick your workspace, choose <em>Next</em>, then <em>Create</em>.</span>
        <div className="xslack__actions">
          <button type="button" className="btn" disabled={disabled} onClick={openApp}>Create XERXES app in Slack</button>
          <button type="button" className="btn" disabled={disabled}
            onClick={() => void copyToClipboard(slackAppManifest(redirectUrl)).then(ok => setCopied(ok ? 'Manifest copied.' : 'Could not copy the manifest.'))}>Copy manifest</button>
        </div>
        <span className="studio-muted">If Slack opens without the details, choose <em>From a manifest</em> and paste the copied manifest.</span>
        {copied ? <span className="studio-muted" role="status">{copied}</span> : null}
      </li>
      <li>
        <strong>Paste its Client ID and Client Secret.</strong>
        <span>They’re on the app’s <em>Basic Information</em> page under <em>App Credentials</em>. Choose <em>Show</em> to see the secret.</span>
        <div className="xslack__fields">
          <label className="sr-only" htmlFor={`${baseId}-cid`}>Slack Client ID</label>
          <input id={`${baseId}-cid`} aria-label="Slack Client ID" value={clientId} placeholder="Client ID" autoComplete="off" spellCheck={false}
            disabled={disabled} onChange={event => setClientId(event.target.value)} />
          <label className="sr-only" htmlFor={`${baseId}-sec`}>Slack Client Secret</label>
          <input id={`${baseId}-sec`} type="password" aria-label="Slack Client Secret" value={secret} placeholder="Client Secret" autoComplete="off" spellCheck={false}
            disabled={disabled} onChange={event => setSecret(event.target.value)} />
          <button type="submit" className="btn btn--solid" disabled={disabled || !valid}>Save</button>
        </div>
      </li>
    </ol>
    </form>
    </details>
    <p className="studio-muted">XERXES acts as you in Slack (no bot), only in direct messages with people you approve, and replies start as drafts for your review. The app’s secret stays on your Hub.</p>
  </div>
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

function ConfirmSlack({ title, children, action, destructive = false, disabled, onCancel, onConfirm }: {
  readonly title: string
  readonly children: ReactNode
  readonly action: string
  readonly destructive?: boolean
  readonly disabled: boolean
  readonly onCancel: () => void
  readonly onConfirm: () => void
}): ReactElement {
  return <section className="xslack__card xslack__confirm" aria-label={title}>
    <h4>{title}</h4>
    {children}
    <div className="xslack__actions">
      <button type="button" className="btn" disabled={disabled} onClick={onCancel}>Cancel</button>
      <button type="button" className={destructive ? 'btn btn--danger' : 'btn btn--solid'} disabled={disabled} onClick={onConfirm}>{action}</button>
    </div>
  </section>
}

type Change = { kind: 'disconnect' } | { kind: 'remove_app' } | { kind: 'add'; contact: SlackIdentity } | { kind: 'remove' | 'auto'; contact: SlackContact }

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
    if (account !== null || pending !== undefined || oauthUntil > 0) return
    run(async current => {
      setError(undefined)
      setOAuthNote(undefined)
      const next = await client.saveApp(app)
      if (current()) accept(next)
    }, reason => setError(setupError(reason)))
  }
  function removeApp(): void {
    if (account !== null || pending !== undefined || oauthUntil > 0) return
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
      if (!(await openOAuth(slackCreateAppUrl(redirectUrl)))) throw new BrowserNotOpenedError()
    }, () => setError('Could not open Slack in your browser. Use Copy manifest and create the app at api.slack.com/apps.'))
  }
  async function beginSignIn(current: () => boolean): Promise<void> {
    setBrokerPairing(undefined)
    const outcome = await launchSlackSignIn({ client, openOAuth, current })
    if (!current() || outcome.kind === 'abandoned') return
    if (outcome.kind === 'expired') {
      setOAuthUntil(0)
      setOAuthNote(outcome.note)
      return
    }
    setOAuthUntil(outcome.until)
    setBrokerPairing(outcome.pairing)
    setOAuthNote(outcome.note)
  }
  function connect(): void {
    if (!browserSignIn || account !== null || pending !== undefined || oauthUntil > 0) return
    run(async current => {
      setError(undefined)
      setOAuthNote(undefined)
      await beginSignIn(current)
    }, reason => {
      setOAuthUntil(0)
      setBrokerPairing(undefined)
      setError(connectError(reason))
    })
  }
  /** Automatic setup: the Hub creates the XERXES app, then Sign in with Slack opens straight away. */
  function createApp(configToken: string): void {
    if (account !== null || pending !== undefined || oauthUntil > 0) return
    run(async current => {
      setError(undefined)
      setOAuthNote(undefined)
      const next = await client.createApp(configToken)
      if (!current()) return
      accept(next)
      try { await beginSignIn(current) } catch {
        if (current()) {
          setOAuthUntil(0)
          setBrokerPairing(undefined)
          setOAuthNote('Your XERXES Slack app is ready. Click Sign in with Slack to connect.')
        }
      }
    }, reason => setError(setupError(reason)))
  }
  function openTokenPage(): void {
    run(async () => {
      setError(undefined)
      if (!(await openOAuth(SLACK_CONFIG_TOKEN_PAGE))) throw new BrowserNotOpenedError()
    }, () => setError('Could not open Slack in your browser. Open api.slack.com/apps and generate a configuration token there.'))
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
        ? <SlackAppSetup redirectUrl={redirectUrl} disabled={locked} openApp={openCreateApp} openTokenPage={openTokenPage} onCreate={createApp} onSave={saveApp} /> : null}
      {account === null && pending === undefined && appSource === 'saved' && oauthUntil === 0
        ? <p className="studio-muted xslack__appnote">Using your XERXES Slack app.{' '}
          <button type="button" className="btn btn--link" disabled={locked} onClick={() => setChange({ kind: 'remove_app' })}>Use a different app</button></p> : null}
      {account === null && change?.kind === 'remove_app' ? <ConfirmSlack title="Remove saved Slack app?" action="Remove app" destructive disabled={busy}
        onCancel={() => setChange(undefined)} onConfirm={() => { setChange(undefined); removeApp() }}>
        <p>This removes the saved Slack app credentials from the Hub so you can configure a different app.</p>
      </ConfirmSlack> : null}
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
      {account !== null && change?.kind === 'disconnect' ? <ConfirmSlack title="Disconnect this Slack account?" action="Disconnect Slack" destructive disabled={busy}
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
              <SelectField
                ariaLabel={`Reply mode for ${row.user_id}`}
                listLabel={`Reply mode for ${row.user_id}`}
                value={row.mode}
                disabled={locked}
                options={[
                  { value: 'draft', label: 'Drafts for review' },
                  { value: 'auto', label: 'Auto-reply' },
                ]}
                onChange={value => {
                  if (value === 'auto') setChange({ kind: 'auto', contact: row })
                  else mutate(() => client.mode(row, 'draft'))
                }}
              />
              <button type="button" className="btn" aria-label={`Open DM with ${row.user_id}`} disabled={locked} onClick={() => setSelected(row)}>Open DM</button>
              <button type="button" className="btn" aria-label={`Remove ${row.user_id}`} disabled={locked} onClick={() => setChange({ kind: 'remove', contact: row })}>Remove</button>
            </div>
          </div>)}
          {change?.kind === 'remove' || change?.kind === 'auto' ? <ConfirmSlack
            title={change.kind === 'remove' ? 'Remove this contact?' : 'Allow automatic Slack replies?'} action={change.kind === 'remove' ? 'Remove contact' : 'Enable auto-reply'} destructive={change.kind === 'remove'} disabled={busy}
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
            <input id="xslack-member" value={userId} placeholder="U… or W…" autoComplete="off" spellCheck={false} disabled={locked}
              aria-invalid={matchListed ? true : undefined} {...(matchListed ? { 'aria-describedby': 'xslack-member-note' } : {})}
              onChange={event => setUserId(event.target.value)} />
            <button type="submit" className="btn" disabled={locked || matchListed || !isSlackUserId(userId.trim())}>Look up exact ID</button>
          </div>
          {matchListed ? <p id="xslack-member-note" className="studio-muted">This member is already on the list.</p> : null}
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
            <button type="button" className="btn" title={review !== undefined ? 'Finish or cancel the open draft review first' : undefined} disabled={busy || review !== undefined} onClick={() => setReview(draft)}>Review draft</button>
          </article>)}
      </div>
    </>}
  </section>
}
