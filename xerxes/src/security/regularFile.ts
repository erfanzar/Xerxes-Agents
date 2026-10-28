// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { closeSync, constants, fstatSync, ftruncateSync, lstatSync, openSync, type BigIntStats } from 'node:fs'
import { lstat, open, type FileHandle } from 'node:fs/promises'

function missing(error: unknown): boolean { return (error as NodeJS.ErrnoException).code === 'ENOENT' }
function refusal(path: string): NodeJS.ErrnoException {
  return Object.assign(new Error(`Refusing a symbolic link, special file, or replaced file: ${path}`), { code: 'ELOOP' })
}
function sameFile(left: BigIntStats, right: BigIntStats): boolean {
  return left.isFile() && right.isFile() && left.dev === right.dev && left.ino === right.ino
}
function guardedFlags(path: string, flags: number, before: BigIntStats | undefined): number {
  if (before && !before.isFile()) throw refusal(path)
  if (!before && !(flags & constants.O_CREAT)) throw Object.assign(new Error(`No such file: ${path}`), { code: 'ENOENT' })
  // Never truncate before identity verification. Exclusive creation closes the
  // missing-path race, including platforms without an O_NOFOLLOW constant.
  return (flags & ~constants.O_TRUNC) | (constants.O_NOFOLLOW ?? 0)
    | (before ? 0 : constants.O_EXCL)
}

/** Open only the regular file observed before open; verify the handle before any truncation. */
export function openRegularFileSync(path: string, flags: number, mode = 0o600): number {
  let before: BigIntStats | undefined
  try { before = lstatSync(path, { bigint: true }) } catch (error) { if (!missing(error)) throw error }
  const fd = openSync(path, guardedFlags(path, flags, before), mode)
  try {
    const opened = fstatSync(fd, { bigint: true })
    const after = lstatSync(path, { bigint: true })
    if (!sameFile(opened, after) || before && !sameFile(before, opened)) throw refusal(path)
    if (flags & constants.O_TRUNC) ftruncateSync(fd, 0)
    return fd
  } catch (error) { closeSync(fd); throw error }
}

/** Async counterpart; writes stay anchored to the verified handle if the name changes later. */
export async function openRegularFile(path: string, flags: number, mode = 0o600): Promise<FileHandle> {
  const before = await lstat(path, { bigint: true }).catch(error => { if (missing(error)) return undefined; throw error })
  const file = await open(path, guardedFlags(path, flags, before), mode)
  try {
    const opened = await file.stat({ bigint: true })
    const after = await lstat(path, { bigint: true })
    if (!sameFile(opened, after) || before && !sameFile(before, opened)) throw refusal(path)
    if (flags & constants.O_TRUNC) await file.truncate(0)
    return file
  } catch (error) { await file.close(); throw error }
}
