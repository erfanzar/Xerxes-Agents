// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createConnection, type Socket } from 'node:net'
import { connect as connectTls, type TLSSocket } from 'node:tls'

import type { EmailSmtpSendRequest, EmailSmtpTransport } from './emailImap.js'

const DEFAULT_CONNECTION_TIMEOUT = 30_000

type SmtpSocket = Socket | TLSSocket

interface SmtpReply {
  readonly code: number
  readonly lines: readonly string[]
}

/** Native TCP/STARTTLS SMTP transport used when an embedding host does not inject one. */
export class BunSmtpTransport implements EmailSmtpTransport {
  private readonly connectionTimeout: number
  private readonly hostname: string

  constructor(options: BunSmtpTransportOptions = {}) {
    this.connectionTimeout = timeoutValue(options.connectionTimeout ?? DEFAULT_CONNECTION_TIMEOUT)
    this.hostname = smtpHostname(options.hostname ?? 'xerxes')
  }

  async send(request: EmailSmtpSendRequest): Promise<void> {
    const wire = await SmtpWire.connect(request.host, request.port, this.connectionTimeout)
    try {
      await wire.expectReply([220], 'server greeting')
      const hello = await wire.command(`EHLO ${this.hostname}`)
      if (hello.code < 200 || hello.code >= 300) {
        await wire.expectCommand(`HELO ${this.hostname}`, [250], 'HELO')
      }
      if (request.startTls) {
        await wire.expectCommand('STARTTLS', [220], 'STARTTLS')
        await wire.upgradeToTls(request.host)
        await wire.expectCommand(`EHLO ${this.hostname}`, [250], 'EHLO after STARTTLS')
      }
      if (request.authentication) {
        await authenticatePlain(wire, request.authentication.username, request.authentication.password)
      }
      await wire.expectCommand(`MAIL FROM:<${smtpEnvelopeAddress(request.from, 'from')}>`, [250], 'MAIL FROM')
      await wire.expectCommand(`RCPT TO:<${smtpEnvelopeAddress(request.to, 'to')}>`, [250, 251], 'RCPT TO')
      await wire.expectCommand('DATA', [354], 'DATA')
      await wire.write(dotStuff(request.mime) + '.\r\n')
      await wire.expectReply([250], 'message body')
      await wire.expectCommand('QUIT', [221], 'QUIT')
    } finally {
      wire.close()
    }
  }
}

export interface BunSmtpTransportOptions {
  /** Maximum time for connection, TLS negotiation, writes, and replies. */
  readonly connectionTimeout?: number
  /** SMTP hostname presented in EHLO/HELO. */
  readonly hostname?: string
}

class SmtpWire {
  private buffer = ''
  private readonly replies: SmtpReply[] = []
  private currentCode: number | undefined
  private currentLines: string[] = []
  private failure: Error | undefined
  private waiter: ((reply: SmtpReply) => void) | undefined
  private rejectWaiter: ((error: Error) => void) | undefined
  private socket: SmtpSocket

  private constructor(socket: SmtpSocket, private readonly timeout: number) {
    this.socket = socket
    this.attach(socket)
  }

  static async connect(host: string, port: number, timeout: number): Promise<SmtpWire> {
    const socket = createConnection({ host, port })
    const wire = new SmtpWire(socket, timeout)
    try {
      await waitForSocket(socket, 'connect', timeout, 'SMTP connection')
      return wire
    } catch (error) {
      wire.close()
      throw error
    }
  }

  async command(command: string): Promise<SmtpReply> {
    await this.write(command + '\r\n')
    return this.readReply()
  }

  async expectCommand(command: string, codes: readonly number[], operation: string): Promise<SmtpReply> {
    const reply = await this.command(command)
    return expectCode(reply, codes, operation)
  }

  async expectReply(codes: readonly number[], operation: string): Promise<SmtpReply> {
    return expectCode(await this.readReply(), codes, operation)
  }

  async write(value: string): Promise<void> {
    if (this.failure) throw this.failure
    await withTimeout(new Promise<void>((resolve, reject) => {
      this.socket.write(value, error => error == null ? resolve() : reject(error))
    }), this.timeout, 'SMTP write')
  }

  async upgradeToTls(host: string): Promise<void> {
    const prior = this.socket
    this.detach(prior)
    const secure = connectTls({ socket: prior, servername: host })
    try {
      await waitForSocket(secure, 'secureConnect', this.timeout, 'SMTP STARTTLS negotiation')
    } catch (error) {
      secure.destroy()
      throw error
    }
    this.socket = secure
    this.attach(secure)
  }

  close(): void {
    this.detach(this.socket)
    this.socket.destroy()
  }

  private attach(socket: SmtpSocket): void {
    socket.setEncoding('utf8')
    socket.on('data', this.onData)
    socket.once('error', this.onError)
    socket.once('close', this.onClose)
  }

  private detach(socket: SmtpSocket): void {
    socket.off('data', this.onData)
    socket.off('error', this.onError)
    socket.off('close', this.onClose)
  }

  private readonly onData = (chunk: unknown): void => {
    try {
      this.buffer += socketText(chunk)
      while (true) {
        const end = this.buffer.indexOf('\n')
        if (end < 0) return
        const line = this.buffer.slice(0, end).replace(/\r$/, '')
        this.buffer = this.buffer.slice(end + 1)
        this.acceptLine(line)
      }
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error('SMTP response parsing failed'))
    }
  }

  private readonly onError = (error: Error): void => {
    this.fail(new Error('SMTP socket failed', { cause: error }))
  }

  private readonly onClose = (): void => {
    this.fail(new Error('SMTP server closed the connection'))
  }

  private acceptLine(line: string): void {
    const match = /^(\d{3})([ -])(.*)$/.exec(line)
    if (!match) {
      this.fail(new Error('SMTP server sent an invalid response'))
      return
    }
    const code = Number.parseInt(match[1] ?? '', 10)
    const delimiter = match[2]
    const text = match[3] ?? ''
    if (this.currentCode === undefined) {
      this.currentCode = code
      this.currentLines = [text]
    } else if (this.currentCode === code) {
      this.currentLines.push(text)
    } else {
      this.fail(new Error('SMTP server changed response code in a multiline reply'))
      return
    }
    if (delimiter === '-') return
    const reply: SmtpReply = { code, lines: this.currentLines }
    this.currentCode = undefined
    this.currentLines = []
    if (this.waiter) {
      const resolve = this.waiter
      this.waiter = undefined
      this.rejectWaiter = undefined
      resolve(reply)
      return
    }
    this.replies.push(reply)
  }

  private async readReply(): Promise<SmtpReply> {
    if (this.failure) throw this.failure
    const reply = this.replies.shift()
    if (reply) return reply
    return withTimeout(new Promise<SmtpReply>((resolve, reject) => {
      this.waiter = resolve
      this.rejectWaiter = reject
    }), this.timeout, 'SMTP response')
  }

  private fail(error: Error): void {
    if (this.failure) return
    this.failure = error
    const reject = this.rejectWaiter
    this.waiter = undefined
    this.rejectWaiter = undefined
    reject?.(error)
  }
}

async function authenticatePlain(wire: SmtpWire, username: string, password: string): Promise<void> {
  const response = Buffer.from(`\u0000${username}\u0000${password}`, 'utf8').toString('base64')
  const reply = await wire.command(`AUTH PLAIN ${response}`)
  if (reply.code === 334) {
    await wire.write(response + '\r\n')
    await wire.expectReply([235], 'AUTH PLAIN')
    return
  }
  expectCode(reply, [235], 'AUTH PLAIN')
}

function expectCode(reply: SmtpReply, codes: readonly number[], operation: string): SmtpReply {
  if (codes.includes(reply.code)) return reply
  throw new Error(`SMTP server rejected ${operation} (${reply.code})`)
}

function smtpEnvelopeAddress(value: string, name: string): string {
  const matched = /<([^<>]+)>$/.exec(value.trim())
  const address = (matched?.[1] ?? value).trim()
  if (!address || /[\r\n<>]/.test(address)) {
    throw new TypeError(`SMTP ${name} address is invalid`)
  }
  return address
}

function dotStuff(value: string): string {
  const normalized = value.replace(/\r?\n/g, '\r\n')
  const body = normalized.replace(/(^|\r\n)\./g, '$1..')
  return body.endsWith('\r\n') ? body : body + '\r\n'
}

function smtpHostname(value: string): string {
  const hostname = value.trim()
  if (!hostname || /[\s\r\n]/.test(hostname)) {
    throw new TypeError('SMTP hostname must be a non-empty single token')
  }
  return hostname
}

function timeoutValue(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError('SMTP connectionTimeout must be a positive safe integer')
  }
  return value
}

function socketText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value instanceof Uint8Array) return new TextDecoder().decode(value)
  throw new TypeError('SMTP socket emitted an unsupported data chunk')
}

async function waitForSocket(
  socket: SmtpSocket,
  event: 'connect' | 'secureConnect',
  timeout: number,
  operation: string,
): Promise<void> {
  await withTimeout(new Promise<void>((resolve, reject) => {
    const cleanup = (): void => {
      socket.off(event, onReady)
      socket.off('error', onError)
    }
    const onReady = (): void => {
      cleanup()
      resolve()
    }
    const onError = (error: Error): void => {
      cleanup()
      reject(new Error(`${operation} failed`, { cause: error }))
    }
    socket.once(event, onReady)
    socket.once('error', onError)
  }), timeout, operation)
}

async function withTimeout<T>(promise: Promise<T>, timeout: number, operation: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${operation} timed out`)), timeout)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}
