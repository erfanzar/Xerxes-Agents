// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** `xerxes auth` — manage subscription-backed provider sessions. */

import { CliWriter } from '../runtime/cliStyle.js'
import {
  CODEX_PROVIDER,
  CodexSession,
  codexClaims,
  codexCliHome,
  importCodexCliTokens,
} from './codexAuth.js'
import { codexRedirectUri, loginWithChatGpt } from './codexLogin.js'

export const AUTH_HELP = `Usage: xerxes auth <command> [provider]

Commands:
  login codex     sign in to ChatGPT and authorize Codex for this machine
  status [codex]  report the stored session, its account, and its plan
  logout codex    forget the stored session

Provider aliases: codex, chatgpt, openai-codex

A ChatGPT Plus/Pro/Business plan authorizes the Codex backend through OAuth,
so no OPENAI_API_KEY is involved and turns are not metered per token. If the
Codex CLI is already signed in on this machine, its session is adopted
automatically and 'auth login' is not required.

Select the provider with a 'codex/' model prefix, for example:
  XERXES_MODEL=codex/gpt-5.3-codex xerxes "explain this repo"

or set 'model: codex/gpt-5.3-codex' in the runtime configuration. Routing is
explicit on purpose: a bare '-codex' model name still goes to the metered
OpenAI API, so switching who pays for a turn is always a deliberate act.`

/** Raised for a usage mistake so the caller can print help instead of a stack. */
export class AuthCommandError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AuthCommandError'
  }
}

const CODEX_ALIASES = new Set(['codex', 'chatgpt', 'openai-codex', 'openai_codex'])

export interface AuthCommandOptions {
  readonly session?: CodexSession
  readonly writer?: CliWriter
}

/** Run `xerxes auth ...`. Returns the intended process exit code. */
export async function runAuthCommand(
  argumentsAfterCommand: readonly string[],
  options: AuthCommandOptions = {},
): Promise<number> {
  const writer = options.writer ?? new CliWriter()
  const [subcommand, ...rest] = argumentsAfterCommand

  if (!subcommand || subcommand === '--help' || subcommand === '-h') {
    writer.line(AUTH_HELP)
    return 0
  }

  const session = options.session ?? new CodexSession()

  switch (subcommand) {
    case 'login':
      return runLogin(requireCodexProvider(rest[0], 'login'), session, writer)
    case 'status':
      return runStatus(rest[0] === undefined ? CODEX_PROVIDER : requireCodexProvider(rest[0], 'status'), session, writer)
    case 'logout':
      return runLogout(requireCodexProvider(rest[0], 'logout'), session, writer)
    default:
      throw new AuthCommandError(`Unknown auth command '${subcommand}'`)
  }
}

/**
 * Only Codex is wired today, so an unknown provider is a usage error rather
 * than a silent no-op that looks like it worked.
 */
function requireCodexProvider(provider: string | undefined, subcommand: string): string {
  if (provider === undefined) {
    throw new AuthCommandError(`auth ${subcommand} requires a provider, for example 'xerxes auth ${subcommand} codex'`)
  }
  if (!CODEX_ALIASES.has(provider.toLowerCase())) {
    throw new AuthCommandError(`Unknown auth provider '${provider}'; supported: codex`)
  }
  return CODEX_PROVIDER
}

async function runLogin(provider: string, session: CodexSession, writer: CliWriter): Promise<number> {
  writer.heading('Sign in to ChatGPT')
  writer.hint(`A browser window will open. Callback: ${codexRedirectUri()}`)
  writer.line()

  const { token } = await loginWithChatGpt()
  await session.store(token)

  const claims = codexClaims(token.accessToken)
  writer.status('ok', '', `Signed in and stored the ${provider} session.`)
  reportClaims(writer, claims)
  return 0
}

async function runStatus(provider: string, session: CodexSession, writer: CliWriter): Promise<number> {
  writer.heading('ChatGPT session')
  const stored = await session.stored()
  if (stored) {
    writer.status('ok', '', `Signed in (${provider}).`)
    reportClaims(writer, codexClaims(stored.accessToken))
    writer.field('expires', expiryText(stored.expiresAt))
    return 0
  }

  // An adoptable CLI session is not "signed out" — reporting it that way would
  // send the user through a browser flow they do not need.
  const imported = await importCodexCliTokens()
  if (imported) {
    writer.status('ok', '', 'Not signed in to Xerxes, but the Codex CLI session on this machine will be used.')
    reportClaims(writer, codexClaims(imported.accessToken))
    writer.field('source', codexCliHome())
    return 0
  }

  writer.status('warn', '', 'Not signed in.')
  writer.hint(`run '${writer.command('xerxes auth login codex')}' to authorize a ChatGPT plan.`)
  return 1
}

async function runLogout(provider: string, session: CodexSession, writer: CliWriter): Promise<number> {
  const removed = await session.logout()
  if (removed) {
    writer.status('ok', '', `Removed the stored ${provider} session.`)
    // The CLI's own credentials are not Xerxes' to delete; say so rather than
    // let the user believe the machine is fully signed out.
    writer.hint(`the Codex CLI session in ${codexCliHome()} is untouched and may still be adopted.`)
    return 0
  }
  writer.status('warn', '', 'No stored session to remove.')
  return 0
}

function reportClaims(
  writer: CliWriter,
  claims: { accountId: string | undefined; email: string | undefined; planType: string | undefined },
): void {
  if (claims.email) writer.field('account', claims.email)
  if (claims.planType) writer.field('plan', claims.planType)
  if (claims.accountId) writer.field('workspace', claims.accountId)
}

function expiryText(expiresAt: number | undefined): string {
  if (expiresAt === undefined) return 'unknown'
  const remainingSeconds = expiresAt - Date.now() / 1_000
  if (remainingSeconds <= 0) return 'expired (refreshes on next use)'
  const hours = Math.floor(remainingSeconds / 3_600)
  return hours >= 24 ? `${Math.floor(hours / 24)}d` : hours >= 1 ? `${hours}h` : '<1h'
}
