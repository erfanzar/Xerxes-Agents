// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** `xerxes auth` — manage subscription-backed provider sessions. */

import { CliWriter } from '../runtime/cliStyle.js'
import {
  AnthropicOAuthSession,
  isAnthropicOAuthToken,
} from './anthropicOAuth.js'
import {
  CODEX_PROVIDER,
  CodexSession,
  codexClaims,
  codexCliHome,
  importCodexCliTokens,
} from './codexAuth.js'
import { codexRedirectUri, loginWithChatGpt } from './codexLogin.js'
import { COPILOT_PROVIDER, CopilotSession } from './copilotAuth.js'
import { KimiCodingOAuthSession } from './kimiCodingOAuth.js'
import { OpenRouterOAuthSession } from './openrouterOAuth.js'
import { RadiusOAuthSession } from './radiusOAuth.js'
import { XaiOAuthSession } from './xaiOAuth.js'

export const AUTH_HELP = `Usage: xerxes auth <command> [provider] [arguments]

Commands:
  login codex        sign in to ChatGPT and authorize Codex for this machine
  login copilot      sign in to GitHub and authorize Copilot for this machine
  login anthropic [--code <paste>]   sign in with a Claude Pro/Max plan
  login kimi         sign in with the Kimi Code subscription (device code)
  login openrouter [--code <paste>]  sign in to OpenRouter (issues an API key)
  login xai          sign in with SuperGrok / X Premium (device code)
  login radius <gateway> [--device]  sign in to a Radius gateway
  status [provider]  report the stored session, its account, and its plan
  logout <provider>  forget the stored session

Provider aliases: codex, chatgpt, openai-codex, copilot, github-copilot,
anthropic, claude, kimi, kimi-code, openrouter, xai, grok, radius

Subscription plans authorize their providers through OAuth, so no per-request
API key is involved and turns are not metered per token. Browser flows print
the authorization URL (and open a browser); pass --code with the pasted
redirect URL or authorization code when the browser runs on another machine.
Device flows print a code to enter at the printed URL.

Select the provider with a model prefix, for example:
  XERXES_MODEL=codex/gpt-5.3-codex xerxes "explain this repo"

Routing is explicit on purpose: a bare '-codex' model name still goes to the
metered OpenAI API, so switching who pays for a turn is always a deliberate
act.`

/** Raised for a usage mistake so the caller can print help instead of a stack. */
export class AuthCommandError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AuthCommandError'
  }
}

const CODEX_ALIASES = new Set(['codex', 'chatgpt', 'openai-codex', 'openai_codex'])
const COPILOT_ALIASES = new Set(['copilot', 'github-copilot', 'github_copilot', 'gh-copilot'])
const ANTHROPIC_ALIASES = new Set(['anthropic', 'claude'])
const KIMI_ALIASES = new Set(['kimi', 'kimi-code', 'kimi_code', 'kimi-coding'])
const OPENROUTER_ALIASES = new Set(['openrouter'])
const XAI_ALIASES = new Set(['xai', 'grok'])
const RADIUS_ALIASES = new Set(['radius'])

type AuthProviderKind = 'anthropic' | 'codex' | 'copilot' | 'kimi' | 'openrouter' | 'radius' | 'xai'

export interface AuthCommandOptions {
  readonly session?: CodexSession
  readonly copilotSession?: CopilotSession
  readonly anthropicOAuthSession?: AnthropicOAuthSession
  readonly kimiOAuthSession?: KimiCodingOAuthSession
  readonly openrouterOAuthSession?: OpenRouterOAuthSession
  readonly xaiOAuthSession?: XaiOAuthSession
  readonly radiusOAuthSession?: RadiusOAuthSession
  readonly writer?: CliWriter
}

interface AuthArguments {
  readonly provider: string | undefined
  readonly positional: readonly string[]
  readonly flags: ReadonlyMap<string, string>
  readonly flagsPresent: ReadonlySet<string>
}

function parseAuthArguments(rest: readonly string[]): AuthArguments {
  const positional: string[] = []
  const flags = new Map<string, string>()
  const flagsPresent = new Set<string>()
  for (let index = 0; index < rest.length; index++) {
    const value = rest[index]
    if (value === undefined) break
    if (value.startsWith('--')) {
      const name = value.slice(2)
      const next = rest[index + 1]
      if (next !== undefined && !next.startsWith('--')) {
        flags.set(name, next)
        index += 1
      } else {
        flagsPresent.add(name)
      }
      continue
    }
    positional.push(value)
  }
  return { provider: positional[0], positional, flags, flagsPresent }
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

  switch (subcommand) {
    case 'login': {
      const args = parseAuthArguments(rest)
      const provider = requireProvider(args.provider, 'login')
      switch (provider) {
        case 'codex':
          return runLogin(provider, options.session ?? new CodexSession(), writer)
        case 'copilot':
          return runCopilotLogin(options.copilotSession ?? new CopilotSession(), writer)
        case 'anthropic':
          return runAnthropicLogin(
            options.anthropicOAuthSession ?? new AnthropicOAuthSession(),
            writer,
            args.flags.get('code'),
          )
        case 'kimi':
          return runKimiLogin(options.kimiOAuthSession ?? new KimiCodingOAuthSession(), writer)
        case 'openrouter':
          return runOpenRouterLogin(
            options.openrouterOAuthSession ?? new OpenRouterOAuthSession(),
            writer,
            args.flags.get('code'),
          )
        case 'xai':
          return runXaiLogin(options.xaiOAuthSession ?? new XaiOAuthSession(), writer)
        case 'radius':
          return runRadiusLogin(options.radiusOAuthSession ?? new RadiusOAuthSession(), writer, args)
      }
      return 0
    }
    case 'status': {
      const args = parseAuthArguments(rest)
      if (args.provider === undefined) {
        return runStatus('codex', options.session ?? new CodexSession(), writer)
      }
      const provider = requireProvider(args.provider, 'status')
      switch (provider) {
        case 'codex':
          return runStatus(provider, options.session ?? new CodexSession(), writer)
        case 'copilot':
          return runCopilotStatus(options.copilotSession ?? new CopilotSession(), writer)
        case 'anthropic':
          return runAnthropicStatus(options.anthropicOAuthSession ?? new AnthropicOAuthSession(), writer)
        case 'kimi':
          return runFlowStatus('Kimi Code session', options.kimiOAuthSession ?? new KimiCodingOAuthSession(), writer, 'xerxes auth login kimi')
        case 'openrouter':
          return runFlowStatus('OpenRouter session', options.openrouterOAuthSession ?? new OpenRouterOAuthSession(), writer, 'xerxes auth login openrouter')
        case 'xai':
          return runFlowStatus('xAI session', options.xaiOAuthSession ?? new XaiOAuthSession(), writer, 'xerxes auth login xai')
        case 'radius':
          return runRadiusStatus(options.radiusOAuthSession ?? new RadiusOAuthSession(), writer)
      }
      return 0
    }
    case 'logout': {
      const args = parseAuthArguments(rest)
      const provider = requireProvider(args.provider, 'logout')
      switch (provider) {
        case 'codex':
          return runLogout(provider, options.session ?? new CodexSession(), writer)
        case 'copilot':
          return runCopilotLogout(options.copilotSession ?? new CopilotSession(), writer)
        case 'anthropic':
          return runFlowLogout('anthropic', options.anthropicOAuthSession ?? new AnthropicOAuthSession(), writer)
        case 'kimi':
          return runFlowLogout('kimi-code', options.kimiOAuthSession ?? new KimiCodingOAuthSession(), writer)
        case 'openrouter':
          return runFlowLogout('openrouter', options.openrouterOAuthSession ?? new OpenRouterOAuthSession(), writer)
        case 'xai':
          return runFlowLogout('xai', options.xaiOAuthSession ?? new XaiOAuthSession(), writer)
        case 'radius':
          return runFlowLogout('radius', options.radiusOAuthSession ?? new RadiusOAuthSession(), writer)
      }
      return 0
    }
    default:
      throw new AuthCommandError(`Unknown auth command '${subcommand}'`)
  }
}

/**
 * Only wired providers are accepted, so an unknown provider is a usage error
 * rather than a silent no-op that looks like it worked.
 */
function requireProvider(provider: string | undefined, subcommand: string): AuthProviderKind {
  if (provider === undefined) {
    throw new AuthCommandError(`auth ${subcommand} requires a provider, for example 'xerxes auth ${subcommand} codex'`)
  }
  const normalized = provider.toLowerCase()
  if (CODEX_ALIASES.has(normalized)) return 'codex'
  if (COPILOT_ALIASES.has(normalized)) return 'copilot'
  if (ANTHROPIC_ALIASES.has(normalized)) return 'anthropic'
  if (KIMI_ALIASES.has(normalized)) return 'kimi'
  if (OPENROUTER_ALIASES.has(normalized)) return 'openrouter'
  if (XAI_ALIASES.has(normalized)) return 'xai'
  if (RADIUS_ALIASES.has(normalized)) return 'radius'
  throw new AuthCommandError(
    `Unknown auth provider '${provider}'; supported: codex, copilot, anthropic, kimi, openrouter, xai, radius`,
  )
}

async function runCopilotLogin(session: CopilotSession, writer: CliWriter): Promise<number> {
  writer.heading('Sign in to GitHub Copilot')
  const credential = await session.login((userCode, verificationUri) => {
    writer.line()
    writer.field('code', userCode)
    writer.field('open', verificationUri)
    writer.line()
    writer.hint('Enter the code at the URL above; this waits for GitHub to confirm.')
  })
  writer.status('ok', '', `Signed in and stored the ${COPILOT_PROVIDER} session.`)
  writer.field('expires', expiryText(credential.expires))
  if (credential.enterpriseUrl) writer.field('api', credential.enterpriseUrl)
  return 0
}

async function runCopilotStatus(session: CopilotSession, writer: CliWriter): Promise<number> {
  writer.heading('GitHub Copilot session')
  const stored = await session.stored()
  if (stored) {
    writer.status('ok', '', `Signed in (${COPILOT_PROVIDER}).`)
    writer.field('expires', expiryText(stored.expires))
    if (stored.enterpriseUrl) writer.field('api', stored.enterpriseUrl)
    if (stored.availableModelIds?.length) writer.field('models', String(stored.availableModelIds.length))
    return 0
  }
  const envToken = process.env.COPILOT_GITHUB_TOKEN?.trim()
    ?? process.env.GH_TOKEN?.trim()
    ?? process.env.GITHUB_TOKEN?.trim()
  if (envToken) {
    writer.status('ok', '', 'Not signed in to Xerxes, but a GitHub token from the environment will be used.')
    return 0
  }
  writer.status('warn', '', 'Not signed in.')
  writer.hint(`run '${writer.command('xerxes auth login copilot')}' to authorize GitHub Copilot.`)
  return 1
}

async function runCopilotLogout(session: CopilotSession, writer: CliWriter): Promise<number> {
  const removed = await session.logout()
  if (removed) {
    writer.status('ok', '', `Removed the stored ${COPILOT_PROVIDER} session.`)
    writer.hint('COPILOT_GITHUB_TOKEN, GH_TOKEN, and GITHUB_TOKEN in the environment are untouched.')
    return 0
  }
  writer.status('warn', '', 'No stored session to remove.')
  return 0
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

async function runAnthropicLogin(
  session: AnthropicOAuthSession,
  writer: CliWriter,
  manualCode: string | undefined,
): Promise<number> {
  writer.heading('Sign in with Claude (Pro/Max)')
  writer.line()
  const credential = await session.login({
    ...(manualCode !== undefined
      ? { manualInput: async () => manualCode }
      : {}),
    openUrl: url => {
      writer.field('open', url)
      writer.hint('Complete login in the opened browser. If it runs on another machine,')
      writer.hint(`re-run with --code and paste the final redirect URL (callback port ${callbackPortHint(url)}).`)
    },
  })
  writer.status('ok', '', 'Signed in and stored the anthropic session.')
  writer.field('expires', expiryText(credential.expires))
  return 0
}

function callbackPortHint(url: string): string {
  try {
    return new URL(url).searchParams.get('redirect_uri')?.split(':').at(-1) ?? '53692'
  } catch {
    return '53692'
  }
}

async function runAnthropicStatus(session: AnthropicOAuthSession, writer: CliWriter): Promise<number> {
  writer.heading('Anthropic subscription session')
  const stored = await session.stored()
  if (stored) {
    writer.status('ok', '', 'Signed in (anthropic).')
    writer.field('kind', isAnthropicOAuthToken(stored.access) ? 'Claude subscription OAuth' : 'bearer token')
    writer.field('expires', expiryText(stored.expires))
    return 0
  }
  const ambient = process.env.ANTHROPIC_AUTH_TOKEN?.trim() ?? process.env.ANTHROPIC_OAUTH_TOKEN?.trim()
  if (ambient) {
    writer.status('ok', '', 'Not signed in to Xerxes, but ANTHROPIC_AUTH_TOKEN from the environment will be used.')
    return 0
  }
  writer.status('warn', '', 'Not signed in.')
  writer.hint(`run '${writer.command('xerxes auth login anthropic')}' to authorize a Claude plan.`)
  return 1
}

async function runKimiLogin(session: KimiCodingOAuthSession, writer: CliWriter): Promise<number> {
  writer.heading('Sign in with Kimi Code')
  const credential = await session.login((userCode, verificationUri) => {
    writer.line()
    writer.field('code', userCode)
    writer.field('open', verificationUri)
    writer.line()
    writer.hint('Enter the code at the URL above; this waits for Kimi to confirm.')
  })
  writer.status('ok', '', 'Signed in and stored the kimi-code session.')
  writer.field('expires', expiryText(credential.expires))
  return 0
}

async function runOpenRouterLogin(
  session: OpenRouterOAuthSession,
  writer: CliWriter,
  manualCode: string | undefined,
): Promise<number> {
  writer.heading('Sign in to OpenRouter')
  writer.line()
  const credential = await session.login({
    ...(manualCode !== undefined ? { manualInput: async () => manualCode } : {}),
    openUrl: url => {
      writer.field('open', url)
      writer.hint('Complete sign-in in the opened browser. If it runs on another machine,')
      writer.hint('re-run with --code and paste the redirect URL or authorization code.')
    },
  })
  writer.status('ok', '', 'Signed in and stored the openrouter session.')
  writer.hint('OpenRouter issued a permanent API key; it never expires and is refreshed as-is.')
  return 0
}

async function runXaiLogin(session: XaiOAuthSession, writer: CliWriter): Promise<number> {
  writer.heading('Sign in with SuperGrok / X Premium')
  const credential = await session.login((userCode, verificationUri) => {
    writer.line()
    writer.field('code', userCode)
    writer.field('open', verificationUri)
    writer.line()
    writer.hint('Enter the code at the URL above; this waits for xAI to confirm.')
  })
  writer.status('ok', '', 'Signed in and stored the xai session.')
  writer.field('expires', expiryText(credential.expires))
  return 0
}

async function runRadiusLogin(
  session: RadiusOAuthSession,
  writer: CliWriter,
  args: AuthArguments,
): Promise<number> {
  const gateway = session.resolveGateway(args.positional[1])
  writer.heading('Sign in to a Radius gateway')
  writer.field('gateway', gateway)
  const credential = args.flagsPresent.has('device')
    ? await session.login({
      gateway,
      method: 'device',
      onUserCode: (userCode, verificationUri) => {
        writer.line()
        writer.field('code', userCode)
        writer.field('open', verificationUri)
        writer.line()
        writer.hint('Enter the code at the URL above; this waits for the gateway to confirm.')
      },
    })
    : await session.login({ gateway })
  writer.status('ok', '', 'Signed in and stored the radius session.')
  writer.field('expires', expiryText(credential.expires))
  return 0
}

async function runRadiusStatus(session: RadiusOAuthSession, writer: CliWriter): Promise<number> {
  writer.heading('Radius session')
  const stored = await session.stored()
  if (stored) {
    writer.status('ok', '', 'Signed in (radius).')
    writer.field('gateway', stored.gateway)
    writer.field('expires', expiryText(stored.expires))
    return 0
  }
  writer.status('warn', '', 'Not signed in.')
  writer.hint(`run '${writer.command('xerxes auth login radius <gateway>')}' to authorize a gateway.`)
  return 1
}

/** Status for a plain credential file: stored → signed in, otherwise hint at login. */
async function runFlowStatus(
  heading: string,
  session: { stored(): Promise<{ access: string; expires: number } | undefined> },
  writer: CliWriter,
  loginCommand: string,
): Promise<number> {
  writer.heading(heading)
  const stored = await session.stored()
  if (stored) {
    writer.status('ok', '', 'Signed in.')
    writer.field('expires', expiryText(stored.expires))
    return 0
  }
  writer.status('warn', '', 'Not signed in.')
  writer.hint(`run '${writer.command(loginCommand)}' to authorize.`)
  return 1
}

async function runFlowLogout(
  provider: string,
  session: { logout(): Promise<boolean> },
  writer: CliWriter,
): Promise<number> {
  const removed = await session.logout()
  if (removed) {
    writer.status('ok', '', `Removed the stored ${provider} session.`)
    return 0
  }
  writer.status('warn', '', 'No stored session to remove.')
  return 0
}
