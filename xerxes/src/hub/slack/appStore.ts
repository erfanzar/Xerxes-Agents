// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { lstatSync, readFileSync, type Stats } from 'node:fs'
import { mkdir, open, rename, unlink } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { record } from './wire.js'

const FILE = 'slack-app.json'
const MAX_BYTES = 4096

/** The Hub owner's own Slack app, saved from Desktop. Host-only; never returned to any client. */
export interface SavedSlackApp {
  readonly clientId: string
  readonly clientSecret: string
}

/** Client IDs and secrets as Slack issues them (the same rule SlackOAuthState enforces). */
export function slackClientValue(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9._-]{1,256}$/u.test(value)
}

const owned = (info: Stats): boolean => process.geteuid !== undefined && info.uid === process.geteuid()
const pathOf = (stateDir: string): string => join(resolve(stateDir), FILE)

/** Read once at Hub start. Missing means no saved app; anything unsafe or malformed fails closed. */
export function loadSlackApp(stateDir: string): SavedSlackApp | undefined {
  const path = pathOf(stateDir)
  let info: Stats
  try { info = lstatSync(path) } catch (error) {
    if (record(error)?.code === 'ENOENT') return undefined
    throw new Error('The saved Slack app settings could not be read.')
  }
  if (!info.isFile() || !owned(info) || (info.mode & 0o777) !== 0o600 || info.nlink !== 1 || info.size > MAX_BYTES) {
    throw new Error('The saved Slack app settings file is not private to the Hub user.')
  }
  let data: Record<string, unknown> | undefined
  try { data = record(JSON.parse(readFileSync(path, 'utf8'))) } catch { data = undefined }
  if (data?.version !== 1 || !slackClientValue(data.client_id) || !slackClientValue(data.client_secret)) {
    throw new Error('The saved Slack app settings are malformed.')
  }
  return { clientId: data.client_id, clientSecret: data.client_secret }
}

/** Atomic, owner-only (0600) write into the Hub state directory. */
export async function saveSlackApp(stateDir: string, app: SavedSlackApp): Promise<void> {
  if (!slackClientValue(app.clientId) || !slackClientValue(app.clientSecret)) throw new Error('Invalid Slack app credentials.')
  const dir = resolve(stateDir)
  await mkdir(dir, { recursive: true, mode: 0o700 })
  const temporary = join(dir, `.slack-app.${randomUUID()}.tmp`)
  const file = await open(temporary, 'wx', 0o600)
  try {
    try {
      await file.chmod(0o600)
      await file.writeFile(`${JSON.stringify({ version: 1, client_id: app.clientId, client_secret: app.clientSecret })}\n`)
      await file.sync()
    } finally { await file.close() }
    await rename(temporary, pathOf(dir))
  } catch (error) {
    await unlink(temporary).catch(() => {})
    throw error
  }
}

export async function removeSlackApp(stateDir: string): Promise<void> {
  try { await unlink(pathOf(stateDir)) } catch (error) { if (record(error)?.code !== 'ENOENT') throw error }
}
