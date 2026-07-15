// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const ACP_PERMISSION_MODES = ['accept-all', 'auto', 'manual'] as const

export type AcpPermissionMode = (typeof ACP_PERMISSION_MODES)[number]

export interface AcpCommandOptions {
  readonly help: boolean
  readonly permissionMode: AcpPermissionMode
  readonly projectDirectory: string | undefined
  readonly writeRegistry: boolean
}

export const ACP_HELP = `Xerxes Agent Client Protocol server over stdio JSON-RPC.

Usage:
  xerxes acp [--project-dir <directory>] [--permission-mode <accept-all|auto|manual>]
  xerxes acp --write-registry
  xerxes-acp [--permission-mode <accept-all|auto|manual>]
  xerxes-acp --write-registry

Options:
  --write-registry               Write the ACP discovery manifest and exit.
  --permission-mode <mode>       Default tool-approval mode (default: accept-all).
  --project-dir <directory>      Workspace used to load local Xerxes configuration.
`

/** Parse the public ACP launcher flags without starting a provider-backed runtime. */
export function parseAcpCommandOptions(args: readonly string[]): AcpCommandOptions {
  let help = false
  let permissionMode: AcpPermissionMode = 'accept-all'
  let projectDirectory: string | undefined
  let writeRegistry = false

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--help' || argument === '-h') {
      help = true
      continue
    }
    if (argument === '--write-registry') {
      writeRegistry = true
      continue
    }
    if (argument === '--permission-mode') {
      const value = requiredOptionValue(args, index, argument)
      if (!isAcpPermissionMode(value)) {
        throw new Error(`--permission-mode must be one of: ${ACP_PERMISSION_MODES.join(', ')}`)
      }
      permissionMode = value
      index += 1
      continue
    }
    if (argument === '--project-dir') {
      projectDirectory = requiredOptionValue(args, index, argument)
      index += 1
      continue
    }
    throw new Error(`Unknown ACP option: ${argument}`)
  }

  return Object.freeze({ help, permissionMode, projectDirectory, writeRegistry })
}

function isAcpPermissionMode(value: string): value is AcpPermissionMode {
  return (ACP_PERMISSION_MODES as readonly string[]).includes(value)
}

function requiredOptionValue(args: readonly string[], index: number, option: string): string {
  const value = args[index + 1]
  if (!value || value.startsWith('-')) {
    throw new Error(`${option} requires a value`)
  }
  return value
}
