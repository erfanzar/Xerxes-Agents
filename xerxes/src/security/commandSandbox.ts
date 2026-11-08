// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The OS sandbox exec_command runs in when a run's safety rules allow it to run
 * without asking: Seatbelt (`sandbox-exec`) on macOS, bubblewrap (`bwrap`) on
 * Linux. Inside it a command may read anything, may write only the run's output
 * folder and temp folder, may not change the run's input files, and has no
 * network. Where neither runner works, there is no sandbox and commands ask
 * as before; nothing here ever runs a command unsandboxed on its own.
 */

/** Which OS mechanism enforces the policy. */
export type CommandSandboxRunner = 'seatbelt' | 'bubblewrap'

export interface CommandSandboxPolicy {
  /** Folders the command may write, as real absolute paths: the output folder and the run's temp folder. */
  readonly writable: readonly string[]
  /** Folders inside a writable folder it may not write (`.git`, `node_modules`). */
  readonly readOnlyFolders: readonly string[]
  /** Files inside a writable folder it may not change, delete or rename: the run's inputs. */
  readonly readOnlyFiles: readonly string[]
  /** The run's temp folder, given to the command as TMPDIR. */
  readonly tempFolder: string
}

/** A runner bound to one run's policy. */
export interface BoundCommandSandbox {
  readonly runner: CommandSandboxRunner
  /** The argv that runs `argv` inside the sandbox. */
  readonly wrap: (argv: readonly string[]) => string[]
  /** Variables the sandboxed command gets on top of the inherited environment. */
  readonly env: Readonly<Record<string, string>>
}

export interface CommandSandbox {
  readonly runner: CommandSandboxRunner
  bind(policy: CommandSandboxPolicy): BoundCommandSandbox
}

const SANDBOX_EXEC = '/usr/bin/sandbox-exec'

/**
 * System services a command may still reach. Everything else, including
 * LaunchServices and Apple events, is refused: those would let a command ask
 * another app (a browser, Finder) to act outside the sandbox.
 */
const SEATBELT_MACH_SERVICES = [
  'com.apple.system.opendirectoryd.libinfo',
  'com.apple.system.notification_center',
  'com.apple.system.logger',
  'com.apple.cfprefsd.daemon',
  'com.apple.cfprefsd.agent',
  'com.apple.trustd.agent',
  'com.apple.SecurityServer',
]

/** A Seatbelt string literal: backslashes and quotes escaped. */
function sbplString(text: string): string {
  return `"${text.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

/**
 * The Seatbelt profile for a policy. Later rules win, so the order is: allow
 * by default, refuse network and services, refuse every write, re-allow the
 * writable folders, then refuse the read-only folders and files inside them.
 */
export function seatbeltProfile(policy: CommandSandboxPolicy): string {
  const lines = [
    '(version 1)',
    '(allow default)',
    '(deny network*)',
    '(deny mach-lookup)',
    `(allow mach-lookup ${SEATBELT_MACH_SERVICES.map(name => `(global-name ${sbplString(name)})`).join(' ')})`,
    '(deny file-write*)',
    `(allow file-write* ${[
      ...policy.writable.map(path => `(subpath ${sbplString(path)})`),
      '(literal "/dev/null")', '(literal "/dev/stdout")', '(literal "/dev/stderr")', '(literal "/dev/tty")',
      '(regex #"^/dev/fd/")',
    ].join(' ')})`,
  ]
  if (policy.readOnlyFolders.length) {
    lines.push(`(deny file-write* ${policy.readOnlyFolders.map(path => `(subpath ${sbplString(path)})`).join(' ')})`)
  }
  if (policy.readOnlyFiles.length) {
    lines.push(`(deny file-write* ${policy.readOnlyFiles.map(path => `(literal ${sbplString(path)})`).join(' ')})`)
  }
  return lines.join('\n')
}

/**
 * bubblewrap arguments for a policy: the whole filesystem read-only, the
 * writable folders bound read-write on top, the read-only folders and files
 * bound read-only on top of those, and no network namespace access.
 */
export function bubblewrapArgs(policy: CommandSandboxPolicy, argv: readonly string[]): string[] {
  return [
    '--die-with-parent',
    '--unshare-net',
    '--ro-bind', '/', '/',
    '--dev', '/dev',
    '--proc', '/proc',
    ...policy.writable.flatMap(path => ['--bind', path, path]),
    ...policy.readOnlyFolders.flatMap(path => ['--ro-bind-try', path, path]),
    ...policy.readOnlyFiles.flatMap(path => ['--ro-bind', path, path]),
    '--',
    ...argv,
  ]
}

function sandboxEnv(policy: CommandSandboxPolicy): Readonly<Record<string, string>> {
  // Tools that keep caches under HOME (matplotlib, fontconfig) write here instead.
  return Object.freeze({ TMPDIR: policy.tempFolder, MPLCONFIGDIR: policy.tempFolder, XDG_CACHE_HOME: policy.tempFolder })
}

function seatbelt(): CommandSandbox {
  return {
    runner: 'seatbelt',
    bind: policy => {
      const profile = seatbeltProfile(policy)
      return { runner: 'seatbelt', env: sandboxEnv(policy), wrap: argv => [SANDBOX_EXEC, '-p', profile, ...argv] }
    },
  }
}

function bubblewrap(path: string): CommandSandbox {
  return {
    runner: 'bubblewrap',
    bind: policy => ({ runner: 'bubblewrap', env: sandboxEnv(policy), wrap: argv => [path, ...bubblewrapArgs(policy, argv)] }),
  }
}

/** How {@link findCommandSandbox} looks at the host; replaced in tests. */
export interface CommandSandboxHost {
  readonly platform: NodeJS.Platform
  which(command: string): string | null
  /** Exit code of a short probe command. */
  run(argv: readonly string[]): Promise<number>
}

export const SYSTEM_SANDBOX_HOST: CommandSandboxHost = {
  platform: process.platform,
  which: command => Bun.which(command),
  run: async argv => {
    const child = Bun.spawn([...argv], { stdin: 'ignore', stdout: 'ignore', stderr: 'ignore' })
    return child.exited
  },
}

/**
 * The working sandbox on this host, or the reason there is none. Each runner
 * must run a real probe command: an installed `bwrap` without user namespaces
 * fails here rather than at the first tool call.
 */
export async function findCommandSandbox(
  host: CommandSandboxHost = SYSTEM_SANDBOX_HOST,
): Promise<{ readonly sandbox: CommandSandbox } | { readonly sandbox?: undefined; readonly reason: string }> {
  if (host.platform === 'darwin') {
    const path = host.which(SANDBOX_EXEC) ?? host.which('sandbox-exec')
    if (path === null) return { reason: 'sandbox-exec is not available on this Mac' }
    const code = await host.run([path, '-p', '(version 1)(allow default)(deny network*)', '/usr/bin/true'])
    return code === 0 ? { sandbox: seatbelt() } : { reason: `sandbox-exec could not start a command (exit ${code})` }
  }
  if (host.platform === 'linux') {
    const path = host.which('bwrap')
    if (path === null) return { reason: 'bubblewrap (bwrap) is not installed; install it to run commands without asking' }
    const code = await host.run([path, '--ro-bind', '/', '/', '--unshare-net', '--die-with-parent', '--', '/bin/true'])
    return code === 0
      ? { sandbox: bubblewrap(path) }
      : { reason: `bubblewrap could not start a command (exit ${code}); unprivileged user namespaces may be disabled` }
  }
  return { reason: `no command sandbox is supported on ${host.platform}` }
}
