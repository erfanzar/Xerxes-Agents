// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  requireSkillText,
  skillJsonArray,
  skillJsonObject,
  type SkillFetch,
  type SkillJsonObject,
} from '../http.js'

/** Google REST APIs used by the original bundled Google Workspace skill. */
export type GoogleWorkspaceApi = 'calendar' | 'docs' | 'drive' | 'gmail' | 'people' | 'sheets'

export interface GoogleWorkspaceEndpoints {
  readonly calendar: string
  readonly docs: string
  readonly drive: string
  readonly gmail: string
  readonly people: string
  readonly sheets: string
}

export const GOOGLE_WORKSPACE_ENDPOINTS: GoogleWorkspaceEndpoints = {
  calendar: 'https://www.googleapis.com/calendar/v3/',
  docs: 'https://docs.googleapis.com/v1/',
  drive: 'https://www.googleapis.com/drive/v3/',
  gmail: 'https://gmail.googleapis.com/gmail/v1/',
  people: 'https://people.googleapis.com/v1/',
  sheets: 'https://sheets.googleapis.com/v4/',
}

export type GoogleQueryValue = boolean | number | string | readonly string[] | undefined

/** A transport-neutral, authenticated Google REST request. */
export interface GoogleWorkspaceRequest {
  readonly api: GoogleWorkspaceApi
  readonly body?: unknown
  readonly method?: 'DELETE' | 'GET' | 'PATCH' | 'POST' | 'PUT'
  /** Must be a relative API path; absolute URLs are rejected to prevent credential exfiltration. */
  readonly path: string
  readonly query?: Readonly<Record<string, GoogleQueryValue>>
  readonly signal?: AbortSignal
}

/** Explicit source of a short-lived Google bearer token. */
export interface GoogleWorkspaceAccessTokenProvider {
  accessToken(signal?: AbortSignal): Promise<string>
}

export interface GoogleWorkspaceClientOptions {
  readonly endpoints?: Partial<GoogleWorkspaceEndpoints>
  readonly fetchImplementation: SkillFetch
  readonly now?: () => Date
  readonly tokenProvider: GoogleWorkspaceAccessTokenProvider
}

export interface GmailMessageSummary {
  readonly date: string
  readonly from: string
  readonly id: string
  readonly labels: readonly string[]
  readonly snippet: string
  readonly subject: string
  readonly threadId: string
  readonly to: string
}

export interface GmailMessage extends GmailMessageSummary {
  readonly body: string
}

export interface GmailSendOptions {
  readonly body: string
  readonly cc?: string
  readonly from?: string
  readonly html?: boolean
  readonly subject: string
  readonly threadId?: string
  readonly to: string
}

export interface GmailSendResult {
  readonly id: string
  readonly status: 'sent'
  readonly threadId: string
}

export interface GmailLabel {
  readonly id: string
  readonly name: string
  readonly type: string
}

export interface CalendarListOptions {
  readonly calendarId?: string
  readonly end?: string
  readonly maxResults?: number
  readonly start?: string
}

export interface GoogleCalendarEvent {
  readonly description: string
  readonly end: string
  readonly htmlLink: string
  readonly id: string
  readonly location: string
  readonly start: string
  readonly status: string
  readonly summary: string
}

export interface CalendarCreateOptions {
  readonly attendees?: readonly string[]
  readonly calendarId?: string
  readonly description?: string
  readonly end: string
  readonly location?: string
  readonly start: string
  readonly summary: string
}

export interface CalendarCreateResult {
  readonly htmlLink: string
  readonly id: string
  readonly status: 'created'
  readonly summary: string
}

export interface GoogleDriveFile {
  readonly id: string
  readonly mimeType: string
  readonly modifiedTime: string
  readonly name: string
  readonly webViewLink: string
}

export interface GoogleContact {
  readonly emails: readonly string[]
  readonly name: string
  readonly phones: readonly string[]
}

export type GoogleSheetValues = readonly (readonly unknown[])[]

export interface GoogleSheetUpdateResult {
  readonly updatedCells: number
  readonly updatedRange: string
}

export interface GoogleDocument {
  readonly body: string
  readonly documentId: string
  readonly title: string
}

/** Error surface deliberately omits response bodies and authorization headers. */
export class GoogleWorkspaceApiError extends Error {
  readonly api: GoogleWorkspaceApi
  readonly operation: string
  readonly status: number | undefined

  constructor(
    api: GoogleWorkspaceApi,
    operation: string,
    message: string,
    options: { readonly cause?: unknown; readonly status?: number } = {},
  ) {
    super(message, options)
    this.name = 'GoogleWorkspaceApiError'
    this.api = api
    this.operation = operation
    this.status = options.status
  }
}

/**
 * Native REST client for Gmail, Calendar, Drive, People, Sheets, and Docs.
 *
 * Authentication, fetch, and endpoints are all supplied by the host. No `gws`
 * executable, Python package, environment variable, or credential file is used.
 */
export class GoogleWorkspaceClient {
  private readonly endpoints: GoogleWorkspaceEndpoints
  private readonly fetchImplementation: SkillFetch
  private readonly now: () => Date
  private readonly tokenProvider: GoogleWorkspaceAccessTokenProvider

  constructor(options: GoogleWorkspaceClientOptions) {
    this.endpoints = normalizeEndpoints({ ...GOOGLE_WORKSPACE_ENDPOINTS, ...(options.endpoints ?? {}) })
    this.fetchImplementation = options.fetchImplementation
    this.now = options.now ?? (() => new Date())
    this.tokenProvider = options.tokenProvider
  }

  /** Execute a JSON Google REST request through the explicit authenticated transport. */
  async request(request: GoogleWorkspaceRequest): Promise<SkillJsonObject> {
    const method = request.method ?? (request.body === undefined ? 'GET' : 'POST')
    const url = googleWorkspaceRequestUrl(this.endpoints[request.api], request.path, request.query)
    const accessToken = await this.tokenProvider.accessToken(request.signal)
    if (!accessToken.trim()) throw new GoogleWorkspaceApiError(request.api, method, 'Google access-token provider returned an empty token')
    const headers: Record<string, string> = {
      Accept: 'application/json',
      Authorization: `Bearer ${accessToken}`,
    }
    const init: RequestInit = { headers, method, ...(request.signal === undefined ? {} : { signal: request.signal }) }
    if (request.body !== undefined) {
      headers['Content-Type'] = 'application/json'
      init.body = JSON.stringify(request.body)
    }

    let response: Response
    try {
      response = await this.fetchImplementation(url, init)
    } catch (error) {
      throw new GoogleWorkspaceApiError(request.api, method, 'Google Workspace API request failed', { cause: error })
    }
    if (!response.ok) {
      throw new GoogleWorkspaceApiError(
        request.api,
        method,
        `Google Workspace API request failed with HTTP ${response.status}`,
        { status: response.status },
      )
    }
    let text: string
    try {
      text = await response.text()
    } catch (error) {
      throw new GoogleWorkspaceApiError(request.api, method, 'Google Workspace API response could not be read', { cause: error })
    }
    if (!text.trim()) return {}
    try {
      return skillJsonObject(JSON.parse(text) as unknown, 'Google Workspace API response')
    } catch (error) {
      throw new GoogleWorkspaceApiError(request.api, method, 'Google Workspace API response was not valid JSON', { cause: error })
    }
  }

  /** Search Gmail and resolve each matching message's safe metadata. */
  async gmailSearch(query: string, options: { readonly maxResults?: number; readonly signal?: AbortSignal } = {}): Promise<readonly GmailMessageSummary[]> {
    const maxResults = boundedCount(options.maxResults ?? 10, 'maxResults')
    const response = await this.request({
      api: 'gmail',
      path: 'users/me/messages',
      query: { maxResults, q: requireSkillText(query, 'query') },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    const summaries: GmailMessageSummary[] = []
    for (const record of records(response.messages, 'Gmail message list')) {
      summaries.push(await this.gmailMetadata(requiredText(record.id, 'Gmail message id'), options.signal))
    }
    return summaries
  }

  /** Fetch a Gmail message's headers, labels, and decoded text/plain-or-HTML body. */
  async gmailGet(messageId: string, signal?: AbortSignal): Promise<GmailMessage> {
    const response = await this.request({
      api: 'gmail',
      path: `users/me/messages/${pathSegment(messageId, 'messageId')}`,
      query: { format: 'full' },
      ...(signal === undefined ? {} : { signal }),
    })
    return gmailMessage(response, true)
  }

  /** Send an RFC 5322 email directly through Gmail's REST API. */
  async gmailSend(options: GmailSendOptions, signal?: AbortSignal): Promise<GmailSendResult> {
    const raw = base64Url(encodeGmailMessage(options))
    const response = await this.request({
      api: 'gmail',
      body: {
        raw,
        ...(options.threadId === undefined ? {} : { threadId: requireSkillText(options.threadId, 'threadId') }),
      },
      method: 'POST',
      path: 'users/me/messages/send',
      ...(signal === undefined ? {} : { signal }),
    })
    return { id: requiredText(response.id, 'Gmail sent message id'), status: 'sent', threadId: optionalText(response.threadId) }
  }

  /** Reply to a Gmail message while preserving its thread and reference headers. */
  async gmailReply(
    messageId: string,
    options: { readonly body: string; readonly from?: string },
    signal?: AbortSignal,
  ): Promise<GmailSendResult> {
    const original = await this.request({
      api: 'gmail',
      path: `users/me/messages/${pathSegment(messageId, 'messageId')}`,
      query: { format: 'metadata', metadataHeaders: ['From', 'Subject', 'Message-ID'] },
      ...(signal === undefined ? {} : { signal }),
    })
    const headers = gmailHeaderMap(original)
    const subject = headers.get('subject') ?? ''
    const replySubject = /^re:/i.test(subject) ? subject : `Re: ${subject}`
    const replyTo = headers.get('from')
    if (!replyTo) throw new GoogleWorkspaceApiError('gmail', 'GET', 'Gmail reply source message did not include a From header')
    const reference = headers.get('message-id')
    const raw = base64Url(encodeGmailMessage({
      body: options.body,
      ...(options.from === undefined ? {} : { from: options.from }),
      ...(reference === undefined ? {} : { references: reference }),
      subject: replySubject,
      to: replyTo,
    }))
    const response = await this.request({
      api: 'gmail',
      body: { raw, threadId: requiredText(original.threadId, 'Gmail source threadId') },
      method: 'POST',
      path: 'users/me/messages/send',
      ...(signal === undefined ? {} : { signal }),
    })
    return { id: requiredText(response.id, 'Gmail sent message id'), status: 'sent', threadId: optionalText(response.threadId) }
  }

  /** List Gmail labels for the authorized user. */
  async gmailLabels(signal?: AbortSignal): Promise<readonly GmailLabel[]> {
    const response = await this.request({
      api: 'gmail',
      path: 'users/me/labels',
      ...(signal === undefined ? {} : { signal }),
    })
    return records(response.labels, 'Gmail labels').map(label => ({
      id: requiredText(label.id, 'Gmail label id'),
      name: requiredText(label.name, 'Gmail label name'),
      type: optionalText(label.type),
    }))
  }

  /** Add and/or remove label ids from a Gmail message. */
  async gmailModify(
    messageId: string,
    options: { readonly addLabelIds?: readonly string[]; readonly removeLabelIds?: readonly string[] },
    signal?: AbortSignal,
  ): Promise<{ readonly id: string; readonly labels: readonly string[] }> {
    const addLabelIds = optionalIdList(options.addLabelIds, 'addLabelIds')
    const removeLabelIds = optionalIdList(options.removeLabelIds, 'removeLabelIds')
    if (!addLabelIds.length && !removeLabelIds.length) throw new TypeError('provide at least one Gmail label to add or remove')
    const response = await this.request({
      api: 'gmail',
      body: {
        ...(addLabelIds.length ? { addLabelIds } : {}),
        ...(removeLabelIds.length ? { removeLabelIds } : {}),
      },
      method: 'POST',
      path: `users/me/messages/${pathSegment(messageId, 'messageId')}/modify`,
      ...(signal === undefined ? {} : { signal }),
    })
    return { id: requiredText(response.id, 'Gmail message id'), labels: strings(response.labelIds, 'Gmail labelIds') }
  }

  /** List upcoming Calendar events within the requested range. */
  async calendarList(options: CalendarListOptions = {}, signal?: AbortSignal): Promise<readonly GoogleCalendarEvent[]> {
    const start = withGoogleTimezone(options.start ?? this.now().toISOString())
    const end = withGoogleTimezone(options.end ?? new Date(this.now().getTime() + 7 * 24 * 60 * 60 * 1_000).toISOString())
    const maxResults = boundedCount(options.maxResults ?? 25, 'maxResults')
    const response = await this.request({
      api: 'calendar',
      path: `calendars/${pathSegment(options.calendarId ?? 'primary', 'calendarId')}/events`,
      query: { maxResults, orderBy: 'startTime', singleEvents: true, timeMax: end, timeMin: start },
      ...(signal === undefined ? {} : { signal }),
    })
    return records(response.items, 'Google Calendar events').map(calendarEvent)
  }

  /** Create a Google Calendar event. */
  async calendarCreate(options: CalendarCreateOptions, signal?: AbortSignal): Promise<CalendarCreateResult> {
    const attendees = options.attendees === undefined ? [] : optionalIdList(options.attendees, 'attendees')
    const response = await this.request({
      api: 'calendar',
      body: {
        ...(attendees.length ? { attendees: attendees.map(email => ({ email })) } : {}),
        ...(options.description === undefined ? {} : { description: requireSkillText(options.description, 'description') }),
        end: { dateTime: withGoogleTimezone(options.end) },
        ...(options.location === undefined ? {} : { location: requireSkillText(options.location, 'location') }),
        start: { dateTime: withGoogleTimezone(options.start) },
        summary: requireSkillText(options.summary, 'summary'),
      },
      method: 'POST',
      path: `calendars/${pathSegment(options.calendarId ?? 'primary', 'calendarId')}/events`,
      ...(signal === undefined ? {} : { signal }),
    })
    return {
      htmlLink: optionalText(response.htmlLink),
      id: requiredText(response.id, 'Calendar event id'),
      status: 'created',
      summary: optionalText(response.summary),
    }
  }

  /** Delete a Google Calendar event. */
  async calendarDelete(eventId: string, options: { readonly calendarId?: string } = {}, signal?: AbortSignal): Promise<{ readonly eventId: string; readonly status: 'deleted' }> {
    const normalizedEventId = requireSkillText(eventId, 'eventId')
    await this.request({
      api: 'calendar',
      method: 'DELETE',
      path: `calendars/${pathSegment(options.calendarId ?? 'primary', 'calendarId')}/events/${pathSegment(normalizedEventId, 'eventId')}`,
      ...(signal === undefined ? {} : { signal }),
    })
    return { eventId: normalizedEventId, status: 'deleted' }
  }

  /** Search Google Drive by escaped full-text query or an explicitly supplied raw Drive query. */
  async driveSearch(
    query: string,
    options: { readonly maxResults?: number; readonly rawQuery?: boolean; readonly signal?: AbortSignal } = {},
  ): Promise<readonly GoogleDriveFile[]> {
    const rawQuery = requireSkillText(query, 'query')
    const response = await this.request({
      api: 'drive',
      path: 'files',
      query: {
        fields: 'files(id,name,mimeType,modifiedTime,webViewLink)',
        pageSize: boundedCount(options.maxResults ?? 10, 'maxResults'),
        q: options.rawQuery === true ? rawQuery : `fullText contains '${driveQueryLiteral(rawQuery)}'`,
      },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    return records(response.files, 'Google Drive files').map(file => ({
      id: requiredText(file.id, 'Drive file id'),
      mimeType: optionalText(file.mimeType),
      modifiedTime: optionalText(file.modifiedTime),
      name: requiredText(file.name, 'Drive file name'),
      webViewLink: optionalText(file.webViewLink),
    }))
  }

  /** List People API contacts with their display names, email addresses, and phone numbers. */
  async contactsList(options: { readonly pageSize?: number; readonly signal?: AbortSignal } = {}): Promise<readonly GoogleContact[]> {
    const response = await this.request({
      api: 'people',
      path: 'people/me/connections',
      query: {
        pageSize: boundedCount(options.pageSize ?? 50, 'pageSize'),
        personFields: 'names,emailAddresses,phoneNumbers',
      },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    return records(response.connections, 'People connections').map(contact => {
      const name = records(contact.names, 'People names')[0]
      return {
        emails: records(contact.emailAddresses, 'People email addresses').map(value => optionalText(value.value)).filter(Boolean),
        name: name === undefined ? '' : optionalText(name.displayName),
        phones: records(contact.phoneNumbers, 'People phone numbers').map(value => optionalText(value.value)).filter(Boolean),
      }
    })
  }

  /** Read a rectangular range from a Google Sheet. */
  async sheetsGet(spreadsheetId: string, range: string, signal?: AbortSignal): Promise<GoogleSheetValues> {
    const response = await this.request({
      api: 'sheets',
      path: `spreadsheets/${pathSegment(spreadsheetId, 'spreadsheetId')}/values/${pathSegment(range, 'range')}`,
      ...(signal === undefined ? {} : { signal }),
    })
    return sheetValues(response.values, 'Google Sheets values')
  }

  /** Overwrite a Google Sheet range using USER_ENTERED semantics. */
  async sheetsUpdate(
    spreadsheetId: string,
    range: string,
    values: GoogleSheetValues,
    signal?: AbortSignal,
  ): Promise<GoogleSheetUpdateResult> {
    const response = await this.request({
      api: 'sheets',
      body: { values: normalizeSheetValues(values) },
      method: 'PUT',
      path: `spreadsheets/${pathSegment(spreadsheetId, 'spreadsheetId')}/values/${pathSegment(range, 'range')}`,
      query: { valueInputOption: 'USER_ENTERED' },
      ...(signal === undefined ? {} : { signal }),
    })
    return { updatedCells: optionalNumber(response.updatedCells), updatedRange: optionalText(response.updatedRange) }
  }

  /** Append rows to a Google Sheet using USER_ENTERED/INSERT_ROWS semantics. */
  async sheetsAppend(
    spreadsheetId: string,
    range: string,
    values: GoogleSheetValues,
    signal?: AbortSignal,
  ): Promise<{ readonly updatedCells: number }> {
    const response = await this.request({
      api: 'sheets',
      body: { values: normalizeSheetValues(values) },
      method: 'POST',
      path: `spreadsheets/${pathSegment(spreadsheetId, 'spreadsheetId')}/values/${pathSegment(range, 'range')}:append`,
      query: { insertDataOption: 'INSERT_ROWS', valueInputOption: 'USER_ENTERED' },
      ...(signal === undefined ? {} : { signal }),
    })
    const updates = optionalRecord(response.updates)
    return { updatedCells: optionalNumber(updates?.updatedCells) }
  }

  /** Fetch a Google Document and concatenate paragraph text in source order. */
  async docsGet(documentId: string, signal?: AbortSignal): Promise<GoogleDocument> {
    const response = await this.request({
      api: 'docs',
      path: `documents/${pathSegment(documentId, 'documentId')}`,
      ...(signal === undefined ? {} : { signal }),
    })
    return {
      body: extractGoogleDocumentText(response),
      documentId: optionalText(response.documentId) || requireSkillText(documentId, 'documentId'),
      title: optionalText(response.title),
    }
  }

  private async gmailMetadata(messageId: string, signal?: AbortSignal): Promise<GmailMessageSummary> {
    const response = await this.request({
      api: 'gmail',
      path: `users/me/messages/${pathSegment(messageId, 'messageId')}`,
      query: { format: 'metadata', metadataHeaders: ['From', 'To', 'Subject', 'Date'] },
      ...(signal === undefined ? {} : { signal }),
    })
    return gmailMessage(response, false)
  }
}

/** Build a safe absolute Google API request URL from a service base and relative path. */
export function googleWorkspaceRequestUrl(
  endpoint: string,
  path: string,
  query: Readonly<Record<string, GoogleQueryValue>> | undefined = undefined,
): URL {
  const base = normalizedEndpoint(endpoint)
  const relativePath = requireRelativePath(path)
  const url = new URL(relativePath, base)
  if (query !== undefined) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined) continue
      const queryKey = requireSkillText(key, 'Google API query key')
      if (Array.isArray(value)) {
        for (const item of value) url.searchParams.append(queryKey, requireSkillText(item, `Google API query ${queryKey}`))
      } else {
        url.searchParams.set(queryKey, String(value))
      }
    }
  }
  return url
}

/** Append a UTC timezone suffix to an ISO datetime that is missing an offset. */
export function withGoogleTimezone(value: string): string {
  const text = requireSkillText(value, 'datetime')
  if (!text.includes('T') || /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text)) return text
  return `${text}Z`
}

function normalizeEndpoints(endpoints: GoogleWorkspaceEndpoints): GoogleWorkspaceEndpoints {
  return {
    calendar: normalizedEndpoint(endpoints.calendar),
    docs: normalizedEndpoint(endpoints.docs),
    drive: normalizedEndpoint(endpoints.drive),
    gmail: normalizedEndpoint(endpoints.gmail),
    people: normalizedEndpoint(endpoints.people),
    sheets: normalizedEndpoint(endpoints.sheets),
  }
}

function normalizedEndpoint(value: string): string {
  const endpoint = requireSkillText(value, 'Google API endpoint')
  let url: URL
  try {
    url = new URL(endpoint)
  } catch (error) {
    throw new TypeError(`Google API endpoint must be an absolute URL: ${String(error)}`)
  }
  if (!/^https?:$/.test(url.protocol)) throw new TypeError('Google API endpoint must use HTTP or HTTPS')
  if (!url.pathname.endsWith('/')) url.pathname = `${url.pathname}/`
  return url.toString()
}

function requireRelativePath(value: string): string {
  const path = requireSkillText(value, 'Google API path')
  if (path.startsWith('/') || path.includes('://') || path.includes('?') || path.split('/').some(part => part === '..')) {
    throw new TypeError('Google API path must be a relative path without query strings or traversal')
  }
  return path
}

function pathSegment(value: string, name: string): string {
  return encodeURIComponent(requireSkillText(value, name))
}

function gmailMessage(value: SkillJsonObject, includeBody: boolean): GmailMessage {
  const headers = gmailHeaderMap(value)
  const summary: GmailMessageSummary = {
    date: headers.get('date') ?? '',
    from: headers.get('from') ?? '',
    id: requiredText(value.id, 'Gmail message id'),
    labels: strings(value.labelIds, 'Gmail labelIds'),
    snippet: optionalText(value.snippet),
    subject: headers.get('subject') ?? '',
    threadId: requiredText(value.threadId, 'Gmail threadId'),
    to: headers.get('to') ?? '',
  }
  return includeBody ? { ...summary, body: extractGmailBody(optionalRecord(value.payload)) } : { ...summary, body: '' }
}

function gmailHeaderMap(message: SkillJsonObject): Map<string, string> {
  const payload = optionalRecord(message.payload)
  const headers = records(payload?.headers, 'Gmail headers')
  const result = new Map<string, string>()
  for (const header of headers) {
    const name = optionalText(header.name)
    if (name) result.set(name.toLowerCase(), optionalText(header.value))
  }
  return result
}

function extractGmailBody(payload: SkillJsonObject | undefined): string {
  if (!payload) return ''
  const candidates: Array<{ readonly mimeType: string; readonly data: string }> = []
  collectGmailParts(payload, candidates)
  const preferred = candidates.find(part => part.mimeType === 'text/plain') ?? candidates.find(part => part.mimeType === 'text/html')
  return preferred === undefined ? '' : decodeBase64Url(preferred.data)
}

function collectGmailParts(payload: SkillJsonObject, candidates: Array<{ readonly mimeType: string; readonly data: string }>): void {
  const data = optionalRecord(payload.body)?.data
  const mimeType = optionalText(payload.mimeType)
  if (mimeType && (mimeType === 'text/plain' || mimeType === 'text/html') && typeof data === 'string' && data) {
    candidates.push({ data, mimeType })
  }
  for (const part of records(payload.parts, 'Gmail MIME parts')) collectGmailParts(part, candidates)
}

function encodeGmailMessage(options: GmailSendOptions & { readonly references?: string }): Uint8Array {
  const headers: string[] = [
    'MIME-Version: 1.0',
    `To: ${headerValue(options.to, 'to')}`,
    `Subject: ${headerValue(options.subject, 'subject')}`,
    `Content-Type: ${options.html === true ? 'text/html' : 'text/plain'}; charset=UTF-8`,
    'Content-Transfer-Encoding: 8bit',
  ]
  if (options.cc !== undefined) headers.push(`Cc: ${headerValue(options.cc, 'cc')}`)
  if (options.from !== undefined) headers.push(`From: ${headerValue(options.from, 'from')}`)
  if (options.references !== undefined) {
    const reference = headerValue(options.references, 'references')
    headers.push(`In-Reply-To: ${reference}`, `References: ${reference}`)
  }
  if (typeof options.body !== 'string') throw new TypeError('body must be a string')
  return new TextEncoder().encode(`${headers.join('\r\n')}\r\n\r\n${options.body}`)
}

function headerValue(value: string, name: string): string {
  const text = requireSkillText(value, name)
  if (/[\r\n]/.test(text)) throw new TypeError(`${name} must not contain newline characters`)
  return text
}

function base64Url(value: Uint8Array): string {
  return Buffer.from(value).toString('base64url')
}

function decodeBase64Url(value: string): string {
  if (!/^[A-Za-z0-9_\-=\s]*$/.test(value)) return ''
  try {
    return Buffer.from(value.replace(/\s+/g, ''), 'base64url').toString('utf8')
  } catch {
    return ''
  }
}

function calendarEvent(event: SkillJsonObject): GoogleCalendarEvent {
  const start = optionalRecord(event.start)
  const end = optionalRecord(event.end)
  return {
    description: optionalText(event.description),
    end: optionalText(end?.dateTime) || optionalText(end?.date),
    htmlLink: optionalText(event.htmlLink),
    id: requiredText(event.id, 'Calendar event id'),
    location: optionalText(event.location),
    start: optionalText(start?.dateTime) || optionalText(start?.date),
    status: optionalText(event.status),
    summary: optionalText(event.summary) || '(no title)',
  }
}

function driveQueryLiteral(value: string): string {
  return value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")
}

function extractGoogleDocumentText(document: SkillJsonObject): string {
  const body = optionalRecord(document.body)
  const text: string[] = []
  for (const content of records(body?.content, 'Google Document body content')) {
    const paragraph = optionalRecord(content.paragraph)
    for (const element of records(paragraph?.elements, 'Google Document paragraph elements')) {
      const textRun = optionalRecord(element.textRun)
      const value = optionalText(textRun?.content)
      if (value) text.push(value)
    }
  }
  return text.join('')
}

function sheetValues(value: unknown, name: string): GoogleSheetValues {
  if (value === undefined) return []
  const rows = skillJsonArray(value, name)
  return rows.map((row, index) => [...skillJsonArray(row, `${name} row ${index}`)])
}

function normalizeSheetValues(value: GoogleSheetValues): unknown[][] {
  if (!Array.isArray(value)) throw new TypeError('Google Sheet values must be an array of rows')
  return value.map((row, index) => {
    if (!Array.isArray(row)) throw new TypeError(`Google Sheet row ${index} must be an array`)
    return [...row]
  })
}

function boundedCount(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 1_000) throw new RangeError(`${name} must be an integer between 1 and 1000`)
  return value
}

function optionalIdList(value: readonly string[] | undefined, name: string): string[] {
  return (value ?? []).map(item => requireSkillText(item, name))
}

function records(value: unknown, context: string): SkillJsonObject[] {
  if (value === undefined || value === null) return []
  return skillJsonArray(value, context).map((item, index) => skillJsonObject(item, `${context} item ${index}`))
}

function strings(value: unknown, context: string): string[] {
  if (value === undefined || value === null) return []
  return skillJsonArray(value, context).map((item, index) => requiredText(item, `${context} item ${index}`))
}

function optionalRecord(value: unknown): SkillJsonObject | undefined {
  return value === null || typeof value !== 'object' || Array.isArray(value) ? undefined : value as SkillJsonObject
}

function optionalNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function optionalText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function requiredText(value: unknown, name: string): string {
  return requireSkillText(typeof value === 'string' ? value : '', name)
}
