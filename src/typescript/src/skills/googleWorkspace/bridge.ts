// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  type CalendarCreateOptions,
  type CalendarListOptions,
  type GmailSendOptions,
  type GoogleSheetValues,
  type GoogleWorkspaceClient,
} from './client.js'
import {
  type GoogleAuthorizationRequest,
  type GoogleAuthorizationStatus,
  type GoogleWorkspaceOAuthClient,
  type GoogleWorkspaceToken,
} from './oauth.js'
import { googleWorkspaceSetupGuidance, type GoogleWorkspaceSetupGuidance } from './setup.js'

/** Structured, native replacement for the legacy external command-forwarding surface. */
export type GoogleWorkspaceBridgeCommand =
  | { readonly action: 'search'; readonly maxResults?: number; readonly query: string; readonly service: 'gmail'; readonly signal?: AbortSignal }
  | { readonly action: 'get'; readonly messageId: string; readonly service: 'gmail'; readonly signal?: AbortSignal }
  | { readonly action: 'send'; readonly options: GmailSendOptions; readonly service: 'gmail'; readonly signal?: AbortSignal }
  | { readonly action: 'reply'; readonly messageId: string; readonly options: { readonly body: string; readonly from?: string }; readonly service: 'gmail'; readonly signal?: AbortSignal }
  | { readonly action: 'labels'; readonly service: 'gmail'; readonly signal?: AbortSignal }
  | {
      readonly action: 'modify'
      readonly messageId: string
      readonly options: { readonly addLabelIds?: readonly string[]; readonly removeLabelIds?: readonly string[] }
      readonly service: 'gmail'
      readonly signal?: AbortSignal
    }
  | { readonly action: 'list'; readonly options?: CalendarListOptions; readonly service: 'calendar'; readonly signal?: AbortSignal }
  | { readonly action: 'create'; readonly options: CalendarCreateOptions; readonly service: 'calendar'; readonly signal?: AbortSignal }
  | { readonly action: 'delete'; readonly eventId: string; readonly options?: { readonly calendarId?: string }; readonly service: 'calendar'; readonly signal?: AbortSignal }
  | { readonly action: 'search'; readonly maxResults?: number; readonly query: string; readonly rawQuery?: boolean; readonly service: 'drive'; readonly signal?: AbortSignal }
  | { readonly action: 'list'; readonly pageSize?: number; readonly service: 'contacts'; readonly signal?: AbortSignal }
  | { readonly action: 'get'; readonly range: string; readonly service: 'sheets'; readonly signal?: AbortSignal; readonly spreadsheetId: string }
  | { readonly action: 'update'; readonly range: string; readonly service: 'sheets'; readonly signal?: AbortSignal; readonly spreadsheetId: string; readonly values: GoogleSheetValues }
  | { readonly action: 'append'; readonly range: string; readonly service: 'sheets'; readonly signal?: AbortSignal; readonly spreadsheetId: string; readonly values: GoogleSheetValues }
  | { readonly action: 'get'; readonly documentId: string; readonly service: 'docs'; readonly signal?: AbortSignal }
  | { readonly action: 'guidance'; readonly service: 'setup' }
  | { readonly action: 'status'; readonly service: 'setup' }
  | { readonly action: 'begin'; readonly openBrowser?: boolean; readonly service: 'setup' }
  | { readonly action: 'complete'; readonly callbackOrCode: string; readonly service: 'setup'; readonly signal?: AbortSignal }
  | { readonly action: 'revoke'; readonly service: 'setup'; readonly signal?: AbortSignal }

export interface GoogleWorkspaceBridgeOptions {
  readonly client: GoogleWorkspaceClient
  /** OAuth commands are intentionally unavailable until the host injects an explicit OAuth client. */
  readonly oauth?: GoogleWorkspaceOAuthClient
}

export type GoogleWorkspaceBridgeResult =
  | GoogleAuthorizationRequest
  | GoogleAuthorizationStatus
  | GoogleWorkspaceSetupGuidance
  | GoogleWorkspaceToken
  | boolean
  | object
  | readonly object[]
  | GoogleSheetValues

/**
 * Dispatch all supported Google Workspace actions directly to the native REST and
 * OAuth adapters. It never executes a command, reads environment credentials, or
 * forwards an access token to a child process.
 */
export class GoogleWorkspaceBridge {
  private readonly client: GoogleWorkspaceClient
  private readonly oauth: GoogleWorkspaceOAuthClient | undefined

  constructor(options: GoogleWorkspaceBridgeOptions) {
    this.client = options.client
    this.oauth = options.oauth
  }

  async dispatch(command: GoogleWorkspaceBridgeCommand): Promise<GoogleWorkspaceBridgeResult> {
    switch (command.service) {
      case 'gmail':
        return this.dispatchGmail(command)
      case 'calendar':
        return this.dispatchCalendar(command)
      case 'drive':
        return this.client.driveSearch(command.query, {
          ...(command.maxResults === undefined ? {} : { maxResults: command.maxResults }),
          ...(command.rawQuery === undefined ? {} : { rawQuery: command.rawQuery }),
          ...(command.signal === undefined ? {} : { signal: command.signal }),
        })
      case 'contacts':
        return this.client.contactsList({
          ...(command.pageSize === undefined ? {} : { pageSize: command.pageSize }),
          ...(command.signal === undefined ? {} : { signal: command.signal }),
        })
      case 'sheets':
        return this.dispatchSheets(command)
      case 'docs':
        return this.client.docsGet(command.documentId, command.signal)
      case 'setup':
        return this.dispatchSetup(command)
    }
  }

  private async dispatchGmail(command: Extract<GoogleWorkspaceBridgeCommand, { readonly service: 'gmail' }>): Promise<GoogleWorkspaceBridgeResult> {
    switch (command.action) {
      case 'search':
        return this.client.gmailSearch(command.query, {
          ...(command.maxResults === undefined ? {} : { maxResults: command.maxResults }),
          ...(command.signal === undefined ? {} : { signal: command.signal }),
        })
      case 'get':
        return this.client.gmailGet(command.messageId, command.signal)
      case 'send':
        return this.client.gmailSend(command.options, command.signal)
      case 'reply':
        return this.client.gmailReply(command.messageId, command.options, command.signal)
      case 'labels':
        return this.client.gmailLabels(command.signal)
      case 'modify':
        return this.client.gmailModify(command.messageId, command.options, command.signal)
    }
  }

  private async dispatchCalendar(command: Extract<GoogleWorkspaceBridgeCommand, { readonly service: 'calendar' }>): Promise<GoogleWorkspaceBridgeResult> {
    switch (command.action) {
      case 'list':
        return this.client.calendarList(command.options, command.signal)
      case 'create':
        return this.client.calendarCreate(command.options, command.signal)
      case 'delete':
        return this.client.calendarDelete(command.eventId, command.options, command.signal)
    }
  }

  private async dispatchSheets(command: Extract<GoogleWorkspaceBridgeCommand, { readonly service: 'sheets' }>): Promise<GoogleWorkspaceBridgeResult> {
    switch (command.action) {
      case 'get':
        return this.client.sheetsGet(command.spreadsheetId, command.range, command.signal)
      case 'update':
        return this.client.sheetsUpdate(command.spreadsheetId, command.range, command.values, command.signal)
      case 'append':
        return this.client.sheetsAppend(command.spreadsheetId, command.range, command.values, command.signal)
    }
  }

  private async dispatchSetup(command: Extract<GoogleWorkspaceBridgeCommand, { readonly service: 'setup' }>): Promise<GoogleWorkspaceBridgeResult> {
    if (command.action === 'guidance') return googleWorkspaceSetupGuidance()
    const oauth = this.requireOauth()
    switch (command.action) {
      case 'status':
        return oauth.authorizationStatus()
      case 'begin':
        return oauth.beginAuthorization(command.openBrowser === undefined ? {} : { openBrowser: command.openBrowser })
      case 'complete':
        return oauth.completeAuthorization(command.callbackOrCode, command.signal)
      case 'revoke':
        return oauth.revoke(command.signal)
    }
  }

  private requireOauth(): GoogleWorkspaceOAuthClient {
    if (!this.oauth) throw new Error('Google Workspace OAuth is not configured; inject an explicit GoogleWorkspaceOAuthClient')
    return this.oauth
  }
}
