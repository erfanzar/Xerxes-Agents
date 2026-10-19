// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join } from 'node:path'

import { PluginSecretStore } from './pluginSecrets.js'

/** Terminal access for `xerxes plugins secret ...`; injected so tests need no TTY. */
export interface SecretCliIo {
  /** Read one value without echoing it (TTY) or from piped stdin. */
  readonly readSecret: (prompt: string) => Promise<string>
  readonly print: (line: string) => void
  readonly error: (line: string) => void
}

const USAGE = [
  'Usage:',
  '  xerxes plugins secret set <plugin> <NAME>    read the value without echo (or from piped stdin)',
  '  xerxes plugins secret unset <plugin> <NAME>',
  '  xerxes plugins secret list <plugin>          names only, never values',
].join('\n')

/** Read a line from the terminal with echo off; from stdin when it is not a TTY. */
export async function readHiddenLine(prompt: string): Promise<string> {
  const stdin = process.stdin
  if (!stdin.isTTY) {
    const text = await new Response(Bun.stdin.stream()).text()
    return text.replace(/\r?\n$/, '')
  }
  process.stderr.write(prompt)
  return new Promise<string>((resolve, reject) => {
    let value = ''
    stdin.setRawMode(true)
    stdin.resume()
    stdin.setEncoding('utf8')
    const done = (error?: Error) => {
      stdin.setRawMode(false)
      stdin.pause()
      stdin.off('data', onData)
      process.stderr.write('\n')
      if (error) reject(error)
      else resolve(value)
    }
    const onData = (chunk: string) => {
      for (const character of chunk) {
        if (character === '\r' || character === '\n') { done(); return }
        if (character === '\u0003') { done(new Error('Cancelled')); return }
        if (character === '\u007f' || character === '\b') value = value.slice(0, -1)
        else value += character
      }
    }
    stdin.on('data', onData)
  })
}

/**
 * `xerxes plugins secret set|unset|list`. Values are written to the encrypted
 * store under the Xerxes home and are never printed or logged; running plugin
 * tools read the store on each call, so no daemon restart is needed.
 */
export async function runPluginSecretCli(args: readonly string[], home: string, io: SecretCliIo, store = new PluginSecretStore(join(home, 'plugin-secrets.json'))): Promise<number> {
  const [group, action, plugin, name, ...rest] = args
  if (group !== 'secret' || !action || !plugin || rest.length) { io.error(USAGE); return 2 }
  try {
    if (action === 'list' && name === undefined) {
      const names = await store.names(plugin)
      io.print(names.length ? `Stored secrets of ${plugin} (values are never shown): ${names.join(', ')}` : `${plugin} has no stored secrets.`)
      return 0
    }
    if (!name) { io.error(USAGE); return 2 }
    if (action === 'unset') {
      io.print(await store.unset(plugin, name) ? `Removed secret ${name} of ${plugin}.` : `${plugin} had no stored secret ${name}.`)
      return 0
    }
    if (action === 'set') {
      const value = await io.readSecret(`Value for ${plugin} ${name} (input hidden): `)
      if (!value) { io.error('No value entered; nothing was stored.'); return 1 }
      await store.set(plugin, name, value)
      io.print(`Stored secret ${name} for ${plugin}. The plugin receives it as context.secrets.${name} if it declares it in permissions.secrets.`)
      return 0
    }
    io.error(USAGE)
    return 2
  } catch (error) {
    // Errors from the store never contain the value.
    io.error(error instanceof Error ? error.message : String(error))
    return 1
  }
}
