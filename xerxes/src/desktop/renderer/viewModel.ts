// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * View model for the desktop screens.
 *
 * Ported from the Claude Design canvas that specified this app, keeping its
 * fixture content verbatim — the copy on these screens is the design, and
 * paraphrasing it would quietly redesign the product. `buildView` is a pure
 * function of the UI state, which is what lets every screen render statically
 * before any of them is wired to the daemon.
 *
 * The colours resolve to the custom properties generated from the TUI palette
 * (see ../tokens.ts), so nothing here carries a literal hex.
 */

/* eslint-disable */

const DS = {
  working: 'var(--x-working)', done: 'var(--x-done)', failed: 'var(--x-failed)', needsInput: 'var(--x-needs)',
  activity: 'var(--x-activity)', structure: 'var(--x-structure)',
  needsInputText: 'var(--x-needs-text)', failedText: 'var(--x-failed-text)',
  hairline: 'var(--x-hairline)', prose: 'var(--x-prose)',
  workingCardBg: 'var(--x-working-bg)', needsInputCardBg: 'var(--x-needs-bg)', needsInputCardBorder: 'var(--x-needs-border)',
  doneCardBg: 'var(--x-done-bg)', doneCardBorder: 'var(--x-done-border)',
  failedCardBg: 'var(--x-failed-bg)', failedCardBorder: 'var(--x-failed-border)'
};

const skin = (state: any) => {
  if (state === 'needsInput') return { dot: DS.needsInput, ground: DS.needsInputCardBg, border: DS.needsInputCardBorder, text: DS.needsInputText };
  if (state === 'done') return { dot: DS.done, ground: DS.doneCardBg, border: DS.doneCardBorder, text: DS.prose };
  if (state === 'failed') return { dot: DS.failed, ground: DS.failedCardBg, border: DS.failedCardBorder, text: DS.failedText };
  return { dot: DS.working, ground: DS.workingCardBg, border: DS.hairline, text: DS.prose };
};

const lead = (n: any) => (n >= 2 ? ' ' + '·'.repeat(n) : '');

// ── The Derafsh Kaviani mark, verbatim from xerxes/src/ui/banner.ts ──────
// The Braille-pixel payload is kept intact: a generic ornament in its place
// would silently replace the product's visual identity.
const DERAFSH_ART = [
  '⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⣿⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠘⢿⣿⣷⣾⣿⣷⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠽⣿⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⢴⣿⡀⠀⠀⠀⠀⠀⣀⠀⠐⠿⠀⠐⡀⠀⠀⠀⠀⠀⢰⣿⡦',
  '⠀⢸⣟⣻⣿⡿⠿⢻⣿⣟⣻⣿⣿⣿⣿⡟⠿⢿⣿⣿⣿⡇⠀',
  '⠀⢸⣿⡽⣏⣿⣶⣄⠀⠀⠀⠤⡄⠀⠀⣠⣶⣿⣹⢿⣿⡇⠀',
  '⠀⢸⣿⡇⠹⣧⣬⣿⣷⡀⠀⠂⠁⢀⣾⣿⣤⣾⠏⢰⣿⡇⠀',
  '⠀⢸⡷⡇⠀⠈⠻⣿⣍⣿⡄⣉⣠⣿⣹⣿⠟⠁⠀⣼⣿⡇⠀',
  '⠀⢸⡿⢿⢠⣶⡄⢀⡉⢻⣿⠛⣿⡟⢩⡀⢠⣤⡄⠿⢿⡇⠀',
  '⠀⢸⡿⡿⠈⠉⠁⣀⣤⣾⣿⣶⣿⣧⣤⣀⠈⠋⠁⢿⣿⡇⠀',
  '⠀⢸⣿⡇⠀⣠⣾⣿⣤⡿⠁⠶⠈⢿⣼⣿⣷⣄⠀⢙⣿⡇⠀',
  '⠀⢸⣯⡁⣼⣏⣩⣿⠟⠀⢠⠒⡄⠀⠻⣿⣉⣻⣧⢸⣿⡇⠀',
  '⠀⢸⣿⣿⣧⠾⠋⣁⣀⡀⣀⣀⡀⢀⣀⡈⠛⢿⣿⣿⣿⡇⠀',
  '⣠⣼⠷⠾⣿⡿⠿⠿⠿⠷⢾⣿⡷⠾⢿⡿⠿⠿⠿⠷⢾⣧⣄',
  '⠙⠛⠀⠀⡾⠀⠀⠀⠀⠀⠀⣿⠀⠀⢸⡇⠀⠀⠀⠀⠈⠛⠃',
  '⠀⠀⠀⣀⡴⠀⠀⠀⠀⠀⢀⣿⡀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⡾⠋⠀⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠳⡄⠀⠀⠀⠀⠀⠀',
  '⠀⢀⡴⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⢠⡇⠀⠀⠀⠀⠀⠀',
  '⠀⠻⠁⠀⠀⠀⢀⡄⠀⠀⠀⣿⠀⠀⠰⡟⠀⠀⠀⠀⠰⡦⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠛⠀⠀⠀⠀⠀⠀⠀'
];

const WORDMARK_GLYPHS = {
  X: ['██╗  ██╗', '╚██╗██╔╝', ' ╚███╔╝ ', ' ██╔██╗ ', '██╔╝ ██╗', '╚═╝  ╚═╝'],
  E: ['███████╗', '██╔════╝', '█████╗  ', '██╔══╝  ', '███████╗', '╚══════╝'],
  R: ['██████╗ ', '██╔══██╗', '██████╔╝', '██╔══██╗', '██║  ██║', '╚═╝  ╚═╝'],
  S: ['███████╗', '██╔════╝', '███████╗', '╚════██║', '███████║', '╚══════╝']
};

// Dark-theme Derafsh stops. `azure` follows the brand token, which in the v3
// theme equals the lapis blue — so the static ramp dedupes before sampling,
// or the wordmark revisits its opening hue mid-word.
const DERAFSH_DARK = ['#6ea8fe', '#b39cf0', '#6ea8fe', '#9fb8d8'];
const DERAFSH_LIGHT = ['#1f64b5', '#7047b5', '#275e9e', '#466c91'];
const DERAFSH_FRAMES = 80;
const TAU = Math.PI * 2;
const mod = (v: any, d: any) => ((v % d) + d) % d;

const mixHex = (from: any, to: any, amount: any) => {
  const ch = (i: any) => {
    const a = parseInt(from.slice(1 + i * 2, 3 + i * 2), 16);
    const b = parseInt(to.slice(1 + i * 2, 3 + i * 2), 16);
    return Math.round(a + (b - a) * amount).toString(16).padStart(2, '0');
  };
  return '#' + ch(0) + ch(1) + ch(2);
};

const gradientColor = (palette: any, position: any) => {
  const scaled = mod(position, 1) * palette.length;
  const i = Math.floor(scaled) % palette.length;
  return mixHex(palette[i], palette[(i + 1) % palette.length], scaled - Math.floor(scaled));
};

// Only the colour phase travels; the payload and its cells stay fixed.
const wavePosition = (row: any, rowCount: any, frame: any) => {
  const phase = mod(frame, DERAFSH_FRAMES) / DERAFSH_FRAMES;
  const v = Math.max(0, Math.min(row, rowCount - 1)) / Math.max(1, rowCount - 1);
  return phase + v + Math.sin(TAU * (v * 1.35 - phase)) * 0.085 + Math.sin(TAU * (v * 2.7 + phase * 2)) * 0.025;
};

const gradientRamp = (palette: any, steps: any) => {
  const stops = palette.filter((c: any, i: any) => palette.indexOf(c) === i);
  const last = stops.length - 1;
  return Array.from({ length: steps }, (_, i) => {
    const scaled = (i / steps) * last;
    const low = Math.floor(scaled);
    return mixHex(stops[low], stops[Math.min(low + 1, last)], scaled - low);
  });
};

const NAV = [
  { head: 'WORK', id: 'session', label: 'Session', key: '⌘1', glyph: '❯' },
  { id: 'agents', label: 'Agents', key: 'F6', glyph: '●' },
  { id: 'sessions', label: 'Chats', key: '⌘K', glyph: '▸' },
  { id: 'diff', label: 'Diff', key: 'F7', glyph: '±' },
  { id: 'terminals', label: 'Terminals', key: 'F8', glyph: '·' },
  { head: 'MODEL', id: 'provider', label: 'Providers', key: '/provider', glyph: '·' },
  { id: 'model', label: 'Model', key: '⌘M', glyph: '◇' },
  { head: 'EXTEND', id: 'skills', label: 'Skills', key: '/skills', glyph: '·' },
  { id: 'tools', label: 'Tools', key: '/tools', glyph: '·' },
  { id: 'mcp', label: 'MCP & plugins', key: '/plugins', glyph: '·' },
  { id: 'agentdefs', label: 'Agent definitions', key: '.agents/', glyph: '·' },
  { id: 'cron', label: 'Scheduled work', key: '/cron', glyph: '·' },
  { id: 'channels', label: 'Channels', key: '·', glyph: '·' },
  { head: 'SYSTEM', id: 'permissions', label: 'Permissions', key: '/permissions', glyph: '·' },
  { id: 'memory', label: 'Memory', key: '·', glyph: '·' },
  { id: 'history', label: 'Session history', key: '/resume', glyph: '·' },
  { id: 'status', label: 'Doctor', key: '/status', glyph: '·' },
  { id: 'help', label: 'Help', key: '/help', glyph: '·' },
  { id: 'settings', label: 'Settings', key: '⌘,', glyph: '⚙' },
  { id: 'style', label: 'Style sheet', key: '⌘/', glyph: '✦' }
];

const r = (name: any, sub: any, right: any, opts: any) => ({ name, sub, right, ...(opts || {}) });

const TABLES = {
  skills: {
    title: '/skills', summary: '31 discovered · recursive SKILL.md', note: 'project overrides user overrides bundled',
    keys: [{ key: '⏎', label: 'invoke' }, { key: '/', label: 'filter' }, { key: 'r', label: 'rescan' }],
    rows: [
      r('cancel-safe-loop', '.agents/skills/', 'SKILL.md', { heading: 'PROJECT · 2', state: 'working' }),
      r('release-checklist', '.agents/skills/', 'SKILL.md', { state: 'done' }),
      r('himalaya', '~/.xerxes/skills/', 'SKILL.md', { heading: 'USER · 1', state: 'done' }),
      r('software-development', 'xerxes/skills/', 'SKILL.md', { heading: 'BUNDLED · 28', state: 'done' }),
      r('autoresearch', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('github', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('inference', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('architecture-diagram', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('manim-video', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('pallas-kernel', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('popular-web-designs', 'xerxes/skills/', 'SKILL.md', { state: 'done' }),
      r('evaluation', 'xerxes/skills/', 'SKILL.md', { state: 'done' })
    ],
    foot: 'The live /help catalogue is authoritative: installed plugins and project skills extend it at runtime.'
  },
};


const PROFILES = [
  { name: 'default', provider: 'anthropic', model: 'k3-256k', credential: 'ANTHROPIC_API_KEY', baseUrl: '—',
    state: 'working', badge: 'active', note: 'Selected on launch. /model changes the model without leaving the profile.' },
  { name: 'oss', provider: 'openai-compatible', model: 'gpt-oss-120b', credential: 'OPENAI_API_KEY',
    baseUrl: 'https://api.example-host.dev/v1', state: 'done', badge: 'ready',
    note: 'The compatible base URL is stored with the profile rather than in the environment.' },
  { name: 'local', provider: 'ollama', model: 'qwen3-coder-30b', credential: 'none needed',
    baseUrl: 'http://127.0.0.1:11434', state: 'failed', badge: 'unreachable',
    note: 'The host service is not running. Xerxes attaches to a backend; it never launches one.' }
];

const TOOL_ROWS = [
  { heading: 'REGISTERED BY DEFAULT · 8', name: 'file', sub: 'read, write, glob, patch', gate: 'path-checked', right: 'registered', state: 'done' },
  { name: 'data', sub: 'parse, query, transform', gate: 'workspace', right: 'registered', state: 'done' },
  { name: 'math', sub: 'evaluate, solve, units', gate: 'none', right: 'registered', state: 'done' },
  { name: 'process', sub: 'argv-only execution', gate: 'policy + sandbox', right: 'registered', state: 'done' },
  { name: 'system', sub: 'inspection only', gate: 'none', right: 'registered', state: 'done' },
  { name: 'web', sub: 'public HTTP', gate: 'URL checks', right: 'registered', state: 'done' },
  { name: 'ai', sub: 'local deterministic methods', gate: 'none', right: 'registered', state: 'done' },
  { name: 'workspace', sub: 'project-scoped resolution', gate: 'path-checked', right: 'registered', state: 'done' },
  { heading: 'OPT-IN · 3', name: 'coding', sub: 'destructive git and file actions', gate: 'explicit flag', right: 'opt-in', state: 'needsInput' },
  { name: 'skill_manage', sub: 'writes persistent user storage', gate: 'explicit flag', right: 'opt-in', state: 'needsInput' },
  { name: 'rl', sub: 'training and inference', gate: 'host backend', right: 'opt-in', state: 'needsInput' },
  { heading: 'NEEDS A HOST PORT · 7', name: 'browser', sub: 'CDP attach to a running browser', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'computer use', sub: 'privileged desktop session', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'media', sub: 'transcription, tts, vision', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'memory', sub: 'memory resolver', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'clarify', sub: 'question UI adapter', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'send_message', sub: 'channel dispatcher', gate: 'host port', right: 'unavailable', state: 'failed' },
  { name: 'history', sub: 'session index', gate: 'host port', right: 'unavailable', state: 'failed' }
];

const SERVERS = [
  { name: 'filesystem', transport: 'stdio', tools: '11 tools', handshake: '0.4s', bar: '28%', state: 'connected', st: 'done' },
  { name: 'github', transport: 'stdio · .cmd shim on Windows', tools: '24 tools', handshake: '1.1s', bar: '74%', state: 'connected', st: 'done' },
  { name: 'postgres', transport: 'stdio', tools: '0 tools', handshake: 'never', bar: '0%', state: 'idle', st: 'needsInput' }
];

const PLUGINS = [
  { name: 'xerxes-lint', sub: 'repo checks on every write' },
  { name: 'derafsh-banner', sub: 'boot mark animation' },
  { name: 'telegram-format', sub: 'channel message shaping' }
];

const TREE = [
  { branch: '', name: 'base.yaml', sub: 'root prompt and policy', tools: '—', policy: 'inherited by all', state: 'working' },
  { branch: '├─ ', name: 'coder.yaml', sub: 'implementation', tools: 'write, shell', policy: 'accept-all', state: 'done' },
  { branch: '│  └─ ', name: 'tester.yaml', sub: 'covers the change', tools: 'shell in project', policy: 'auto', state: 'done' },
  { branch: '├─ ', name: 'researcher.yaml', sub: 'evidence first', tools: 'read, web', policy: 'auto · read-only', state: 'done' },
  { branch: '├─ ', name: 'planner.yaml', sub: 'design only', tools: 'read', policy: 'plan · no writes', state: 'done' },
  { branch: '├─ ', name: 'reviewer.yaml', sub: 'audits the diff', tools: 'read', policy: 'manual', state: 'done' },
  { branch: '└─ ', name: 'objective.yaml', sub: 'iterative loop', tools: 'write, shell', policy: 'auto · gated', state: 'done' }
];

const JOBS = [
  { name: 'nightly-audit', schedule: '0 3 * * *', last: 'last 03:00 · ok · 4m 12s', next: 'next in 9h', state: 'done',
    marks: [{ at: '12.5%', s: 'done' }] },
  { name: 'pr-triage', schedule: '*/15 * * * *', last: 'last 4m ago · ok · 38s', next: 'next in 11m', state: 'working',
    marks: [{ at: '4%', s: 'done' }, { at: '18%', s: 'done' }, { at: '32%', s: 'done' }, { at: '46%', s: 'done' },
      { at: '60%', s: 'done' }, { at: '74%', s: 'done' }, { at: '88%', s: 'working' }] },
  { name: 'release-smoke', schedule: '0 9 * * 1', last: 'last Mon 09:00 · failed · exit 1', next: 'disabled', state: 'needsInput',
    marks: [{ at: '37.5%', s: 'failed' }] }
];

const MATRIX_MODES = [
  { name: 'code', fg: 'var(--x-mode-code)' },
  { name: 'researcher', fg: 'var(--x-working)' },
  { name: 'plan', fg: 'var(--x-structure)' },
  { name: 'objective', fg: 'var(--x-activity)' }
];

// yes / gated / no — three cell kinds, so a ceiling is readable at a glance.
const MATRIX = [
  { name: 'read files', cells: ['y', 'y', 'y', 'y'] },
  { name: 'write files', cells: ['y', 'n', 'n', 'g'] },
  { name: 'run commands', cells: ['g', 'n', 'n', 'g'] },
  { name: 'network', cells: ['y', 'y', 'g', 'y'] },
  { name: 'delegate', cells: ['y', 'y', 'y', 'y'] },
  { name: 'memory writes', cells: ['y', 'g', 'n', 'y'] }
];

const DENIALS = [
  { name: 'writes above the project root', sub: 'path traversal rejected at the boundary' },
  { name: 'credential file reads', sub: 'never reachable by any mode' },
  { name: 'network egress to unlisted hosts', sub: 'URL checks at the call boundary' },
  { name: 'rm -rf outside the workspace', sub: 'resolved against the active project' }
];

const TIERS = [
  { name: 'working', sub: 'what this turn is holding', count: '7', bar: '70%', voice: true, cap: 'capacity 10 — promotion is what empties it' },
  { name: 'episodic', sub: 'what happened, in order', count: '412', bar: '69%' },
  { name: 'semantic', sub: 'what is true about this project', count: '598', bar: '100%' },
  { name: 'procedural', sub: 'how things are done here', count: '167', bar: '28%' }
];

const WEIGHTS = [
  { label: 'semantic ·55', bar: '55%', color: 'var(--x-working)' },
  { label: 'bm25 ·30', bar: '30%', color: 'var(--x-activity)' },
  { label: 'recency ·15', bar: '15%', color: 'var(--x-structure)' }
];

const MEM_FILES = [
  { name: 'MEMORY.md', sub: 'workspace notes' },
  { name: 'USER.md', sub: 'durable preferences' },
  { name: 'IDENTITY.md', sub: 'notes on working identity' }
];

const TIMELINE = [
  { when: 'now', glyph: '●', name: 'live turn', sub: 'streaming · 4 tools · 3 agents', tokens: '18.4k', state: 'working' },
  { when: '1m ago', glyph: '◆', name: 'pre-compaction snapshot', sub: 'taken automatically before compaction', tokens: '18.4k', state: 'done' },
  { when: '4m ago', glyph: '◆', name: 'after the parity test', sub: 'manual /snapshot', tokens: '31.2k', state: 'done' },
  { when: '9m ago', glyph: '▸', name: 'compacted', sub: 'transcript summarised, history kept', tokens: '−12.8k', state: 'done' },
  { when: '12m ago', glyph: '◆', name: 'before the loop edit', sub: 'manual /snapshot', tokens: '12.1k', state: 'done' },
  { when: '22m ago', glyph: '▸', name: 'session opened', sub: '/new · project-scoped', tokens: '0', state: 'done' }
];

const OPS = [
  { name: '/resume', sub: 'reopen saved work' }, { name: '/branch', sub: 'fork the trunk' },
  { name: '/compact', sub: 'summarise, stay valid' }, { name: '/snapshot', sub: 'mark a point' },
  { name: '/rollback', sub: 'return to one' }, { name: 'xerxes export', sub: 'serialise for handoff' }
];

const CHECKS = [
  { heading: 'RUNTIME · 5', name: 'Bun runtime', sub: '1.3.12', right: 'ok', state: 'done' },
  { name: 'launcher on PATH', sub: '~/.local/bin/xerxes', right: 'ok', state: 'done' },
  { name: 'XERXES_HOME', sub: '~/.xerxes', right: 'ok', state: 'done' },
  { name: 'daemon socket', sub: 'project-scoped · v35', right: 'ok', state: 'done' },
  { name: 'TUI bundle', sub: 'xerxes/dist/ui/entry.js', right: 'ok', state: 'done' },
  { heading: 'PROVIDERS · 3', name: 'anthropic', sub: 'ANTHROPIC_API_KEY present', right: 'ok', state: 'done' },
  { name: 'openai', sub: 'OPENAI_API_KEY present', right: 'ok', state: 'done' },
  { name: 'ollama', sub: 'http://127.0.0.1:11434', right: 'warn', state: 'needsInput',
    line: 'Not running. The local profile will fail on its first turn — start the backend or pick another profile.' },
  { heading: 'WORKSPACE · 3', name: 'working tree', sub: '3 files changed', right: 'dirty', state: 'working' },
  { name: 'skills', sub: '31 discovered', right: 'ok', state: 'done' },
  { name: 'MCP servers', sub: '2 of 3 connected', right: 'ok', state: 'done' }
];

const CALLERS = [
  { name: 'OpenTUI', sub: 'React 19 terminal client', state: 'attached', st: 'working' },
  { name: 'one-shot CLI', sub: 'xerxes "prompt"', state: 'idle', st: 'done' },
  { name: 'ACP', sub: 'stdio · editor hosts', state: 'idle', st: 'done' },
  { name: 'Telegram', sub: '--token $TELEGRAM_BOT_TOKEN', state: 'running', st: 'working' },
  { name: 'HTTP API', sub: 'embeddable Bun handler', state: 'not listening', st: 'needsInput' }
];

const DAEMON_OWNS = [
  { text: 'provider streaming, retries and budgets' },
  { text: 'tool registry, permissions, sandbox routing' },
  { text: 'sessions, replay, compaction, snapshots, memory' },
  { text: 'agents, skills, MCP, channels, scheduling' },
  { text: 'audit events, ACP and embedded HTTP surfaces' }
];

const HELP_KEYS = [
  { key: '⏎', label: 'send, or queue while a turn runs' },
  { key: '⇥', label: 'cycle interaction mode' },
  { key: '←', label: 'background this chat' },
  { key: '→', label: 'attach the selected chat' },
  { key: '␣', label: 'peek without attaching' },
  { key: 'F6', label: 'agents panel' },
  { key: 'F7', label: 'diff review' },
  { key: 'F8', label: 'terminals' },
  { key: '⎋', label: 'interrupt the turn' },
  { key: '⌃C', label: 'quit' }
];

const HELP_COMMANDS = [
  { key: '/provider', label: 'create or switch provider profiles' },
  { key: '/model', label: 'choose a provider model' },
  { key: '/new', label: 'start a fresh session' },
  { key: '/resume', label: 'resume saved work' },
  { key: '/background', label: 'optionally instruct, then detach' },
  { key: '/agents', label: 'inspect sub-agents' },
  { key: '/terminals', label: 'inspect the shells Xerxes runs' },
  { key: '/skills', label: 'inspect discovered skills' },
  { key: '/tools', label: 'inspect the active tool registry' },
  { key: '/permissions', label: 'inspect or change policy' },
  { key: '/yolo', label: 'toggle accept-all execution' },
  { key: '/cron', label: 'manage scheduled work' },
  { key: '/status', label: 'runtime and session status' },
  { key: '/quit', label: 'exit' }
];

const TABLE_NOTES = {
  provider: [{ mark: '①', row: 'local', text: 'Xerxes attaches to services the host already runs — it never launches a backend or a browser process for you, so an unreachable profile says so here rather than failing on the first turn.' }],
  skills: [{ mark: '①', row: 'Precedence', text: 'Skills are SKILL.md bundles discovered recursively from project, user and bundled locations, in that order — a project skill shadows a bundled one of the same name.' }],
  tools: [{ mark: '①', row: 'Needs a host port', text: 'Browser, media, memory and channel access are explicit interfaces. Tests inject fakes; production injects a real adapter after a deliberate configuration choice.' }],
  mcp: [{ mark: '①', row: 'postgres', text: 'Configured but never handshaked reads as amber, not green: an idle server that a turn will ask for is a failure that has not happened yet.' }],
  agentdefs: [{ mark: '①', row: 'Inheritance', text: 'A definition inherits policy from its parent, so tightening base.yaml tightens every specialist rather than leaving one file behind.' }],
  cron: [{ mark: '①', row: 'release-smoke', text: 'Jobs run through the daemon turn loop under the same permission policy as the TUI. A job that fails twice disables itself rather than retrying forever.' }],
  channels: [{ mark: '①', row: 'api-server', text: 'The OpenAI-compatible surface is an embeddable handler, not a background command: it does not listen until a host supplies auth, CORS and rate-limit policy.' }],
  permissions: [{ mark: '①', row: 'Static denials', text: 'These are checked at the call boundary and cannot be granted by any mode, including accept-all. YOLO speeds up approvals; it does not widen the policy.' }],
  memory: [{ mark: '①', row: 'working', text: 'The working tier is capped at ten entries. Promotion is what moves anything important out of it — an unbounded working set is a context leak, not a memory.' }],
  history: [{ mark: '①', row: 'Snapshots', text: 'Compaction, branching and rollback all keep the transcript valid for the next provider request. That constraint is why they are operations rather than edits.' }],
  status: [{ mark: '①', row: 'ollama', text: 'Doctor reports what it can prove about this machine now. It never reports a check as passing that it did not actually run.' }],
  help: [{ mark: '①', row: 'Live catalogue', text: 'The help list is generated at runtime because installed plugins and project skills extend it. A hard-coded catalogue would be wrong on any machine but this one.' }]
};

const AGENTS = [
  { group: 'NEEDS INPUT · 1', state: 'needsInput', name: 'reviewer#2', task: 'Audit Synthetic Results',
    line: 'Asks: keep a synthetic result for the cancelled read, or drop the call?',
    budget: '12.4k tok · 6 tools', policy: 'manual · read-only ceiling · no shell', toolCount: 6,
    calls: [ { verb: 'read loop.ts', dur: '0.4s', state: 'done' }, { verb: 'read turnController.ts', dur: '0.6s', state: 'done' },
      { verb: 'grep synthetic', dur: '0.2s', state: 'done' }, { verb: 'read parity test', dur: '0.3s', state: 'done' },
      { verb: 'read gatewayAdapter.ts', dur: '0.5s', state: 'done' }, { verb: 'ask permission', dur: '—', state: 'needsInput' } ],
    files: ['xerxes/src/streaming/loop.ts', 'xerxes/src/ui/app/turnController.ts'] },
  { group: 'WORKING · 2', state: 'working', name: 'researcher#1', task: 'Survey Cancellation Repair',
    line: 'reading xerxes/src/streaming/loop.ts', budget: '8.2k tok · 3 tools',
    policy: 'auto · read-only ceiling · no shell', toolCount: 3,
    calls: [ { verb: 'grep cancel', dur: '0.3s', state: 'done' }, { verb: 'read loop.ts', dur: '0.4s', state: 'done' },
      { verb: 'read streaming/parsers', dur: '0.2s', state: 'working' } ],
    files: ['xerxes/src/streaming/loop.ts', 'xerxes/src/streaming/parsers.ts'] },
  { group: '', state: 'working', name: 'tester#1', task: 'Cover Interrupted Turns',
    line: 'bun test xerxes/test/streamingLoopParity.test.ts', budget: '5.1k tok · 4 tools',
    policy: 'auto · shell allowed in project', toolCount: 4,
    calls: [ { verb: 'read parity test', dur: '0.3s', state: 'done' }, { verb: 'write parity test', dur: '0.5s', state: 'done' },
      { verb: 'bash bun test', dur: '11.4s', state: 'working' }, { verb: 'read output tail', dur: '0.1s', state: 'working' } ],
    files: ['xerxes/test/streamingLoopParity.test.ts'] },
  { group: 'READY TO REVIEW · 1', state: 'done', name: 'planner#1', task: 'Stage The Loop Change',
    line: 'Result: three ordered edits; loop.ts lands last.', budget: '9.7k tok · 5 tools',
    policy: 'plan · read-only, no writes', toolCount: 5,
    calls: [ { verb: 'read AGENTS.md', dur: '0.2s', state: 'done' }, { verb: 'read loop.ts', dur: '0.4s', state: 'done' },
      { verb: 'read turnController.ts', dur: '0.6s', state: 'done' }, { verb: 'read parity test', dur: '0.3s', state: 'done' },
      { verb: 'write plan', dur: '0.2s', state: 'done' } ],
    files: ['.agents/projects/cancel-safe-loop.md'] },
  { group: 'FAILED · 1', state: 'failed', name: 'coder#3', task: 'Patch Steer Handling',
    line: 'Token budget exhausted after two retries.', budget: '31.0k tok · 11 tools',
    policy: 'accept-all · project scope', toolCount: 11,
    calls: [ { verb: 'read loop.ts', dur: '0.4s', state: 'done' }, { verb: 'edit loop.ts', dur: '0.9s', state: 'done' },
      { verb: 'bash bun run typecheck', dur: '18.2s', state: 'failed' }, { verb: 'edit loop.ts', dur: '0.8s', state: 'failed' } ],
    files: ['xerxes/src/streaming/loop.ts'] }
];

const SESSIONS = [
  { group: 'NEEDS INPUT · 1', state: 'needsInput', title: 'Rotate the release keys', age: '4m', line: 'Asks: which keychain holds the npm token?' },
  { group: 'WORKING · 1', state: 'working', title: 'Make the streaming loop cancel-safe', age: 'now', line: '└ reading xerxes/src/streaming/loop.ts', current: true },
  { group: 'READY TO REVIEW · 1', state: 'done', title: 'Map this repo', age: '22m', line: 'Result: 4 entry points, one dead channel adapter.' },
  { group: 'SAVED · 3', state: 'done', title: 'Telegram gateway retries', age: '2d', line: '' },
  { group: '', state: 'done', title: 'Compaction budget audit', age: '3d', line: '' },
  { group: '', state: 'done', title: 'Untitled chat', age: '5d', line: 'No exchange completed, so the daemon never titled it.' }
];

const TERMS = [
  { group: 'RUNNING · 1', state: 'working', cmd: 'bun test xerxes/test/streamingLoopParity.test.ts',
    meta: 'background · pid 48213 · 11.4s', detail: 'background command · started by tester#1 · pid 48213',
    tail: [ { text: 'bun test v1.3.12', c: 'meta' }, { text: '', c: 'meta' },
      { text: 'streamingLoopParity.test.ts:', c: 'prose' },
      { text: '  ✓ emits one synthetic result per cancelled call', c: 'done' },
      { text: '  ✓ keeps the transcript valid across a steer', c: 'done' },
      { text: '  ✓ repairs a tool turn interrupted mid-stream', c: 'done' },
      { text: '  › covers an interrupted turn with two open calls', c: 'working' } ] },
  { group: 'FAILED · 1', state: 'failed', cmd: 'bun run typecheck', meta: 'exit 2 · 18.2s ago',
    detail: 'background command · started by coder#3 · exit 2',
    tail: [ { text: 'xerxes/src/streaming/loop.ts:214:9 - error TS2532:', c: 'failed' },
      { text: "  Object is possibly 'undefined'.", c: 'prose' }, { text: '', c: 'meta' },
      { text: 'Found 1 error in 1 file.', c: 'failed' } ] },
  { group: 'INTERACTIVE · 1', state: 'needsInput', cmd: 'zsh', meta: 'pty · accepts input · idle 6m',
    detail: 'interactive pty · /Users/erfan/src/Xerxes-Agents · accepts input',
    tail: [ { text: '~/src/Xerxes-Agents ❯ git status --short', c: 'prose' },
      { text: ' M xerxes/src/streaming/loop.ts', c: 'meta' },
      { text: '?? xerxes/test/streamingLoopParity.test.ts', c: 'meta' },
      { text: '~/src/Xerxes-Agents ❯ ', c: 'prose' } ] },
  { group: 'SUCCEEDED · 2', state: 'done', cmd: 'bun run build:ui', meta: 'exit 0 · 3m ago',
    detail: 'background command · exit 0 · 6.1s',
    tail: [ { text: 'bundle xerxes/dist/ui/entry.js  1.42 MB', c: 'prose' }, { text: 'done in 6.1s', c: 'done' } ] },
  { group: '', state: 'done', cmd: 'git status --porcelain', meta: 'exit 0 · 3m ago', detail: 'background command · exit 0 · 0.1s',
    tail: [ { text: ' M xerxes/src/streaming/loop.ts', c: 'meta' } ] }
];

const FILES = [
  { path: 'xerxes/src/streaming/loop.ts', add: '+34', del: '−11',
    hunk: '@@ -206,9 +206,14 @@ async function* runToolTurn(',
    lines: [
      { no: 206, sign: ' ', text: '  for (const call of pending) {' },
      { no: 207, sign: ' ', text: '    if (!signal.aborted) {' },
      { no: 208, sign: '-', text: '      yield { type: "tool_result", id: call.id, result }' },
      { no: 209, sign: '-', text: '      continue' },
      { no: 209, sign: '+', text: '      yield toolResult(call, result)' },
      { no: 210, sign: '+', text: '      continue' },
      { no: 211, sign: ' ', text: '    }' },
      { no: 212, sign: ' ', text: '' },
      { no: 213, sign: '+', text: '    // Cancellation repair lives here and nowhere else: the loop' },
      { no: 214, sign: '+', text: '    // owns the tool turn, so it owns the synthetic close.' },
      { no: 215, sign: '+', text: '    yield syntheticResult(call, "cancelled")' },
      { no: 216, sign: ' ', text: '  }' }
    ] },
  { path: 'xerxes/src/ui/app/turnController.ts', add: '+8', del: '−2',
    hunk: '@@ -1118,7 +1118,9 @@ const settleSubagents = (',
    lines: [
      { no: 1118, sign: ' ', text: '  if (event.type === "tool_result") {' },
      { no: 1119, sign: '-', text: '    emitSynthetic(event.id)' },
      { no: 1120, sign: '+', text: '    // The loop already closed it; a second result invalidates' },
      { no: 1121, sign: '+', text: '    // the next provider request.' },
      { no: 1122, sign: ' ', text: '  }' }
    ] },
  { path: 'xerxes/test/streamingLoopParity.test.ts', add: '+61', del: '−0',
    hunk: '@@ -0,0 +1,61 @@',
    lines: [
      { no: 1, sign: '+', text: "it('emits one synthetic result per cancelled call', async () => {" },
      { no: 2, sign: '+', text: '  const events = await collect(runTurn(cancelAfter(1)))' },
      { no: 3, sign: '+', text: '  expect(results(events)).toHaveLength(1)' },
      { no: 4, sign: '+', text: '})' }
    ] }
];

const MODELS = [
  { name: 'anthropic/k3-256k', meta: '256k ctx · current' },
  { name: 'anthropic/k3-haiku', meta: '200k ctx' },
  { name: 'openai/gpt-oss-120b', meta: '128k ctx · compatible base URL' },
  { name: 'local/qwen3-coder-30b', meta: 'ollama · needs the backend running' }
];

const NOTES = {
  session: [
    { mark: '①', row: 'START WITH', text: 'Every chip carries its own consequence — counts, ages, file totals — so the choice is informed before the keypress. A chip with nothing true to say is not shown.' },
    { mark: '②', row: 'DELEGATED', text: 'Sub-agents stay inside their parent chat. Promoting them to top-level rows is what produced duplicate fleets in the rail.' },
    { mark: '③', row: 'Composer', text: 'The composer never degrades: at any window width it keeps the prompt, the queue state and the hints. Metadata goes first, input never.' }
  ],
  agents: [
    { mark: '①', row: 'Group order', text: 'Ranked by what you must do: unblock, monitor, review. A failed run sorts last — it has already spent its money, it does not get to spend your attention too.' },
    { mark: '②', row: 'Amber', text: 'Amber means a human is required, never emphasis. One blocked agent is findable from across the room precisely because nothing else on the screen is amber.' }
  ],
  sessions: [
    { mark: '①', row: 'Backgrounding', text: 'Detaching does not stop the turn. Sessions keep working while you are elsewhere, which is why the group captions count live state rather than files on disk.' },
    { mark: '②', row: 'Untitled chat', text: 'A chat holds the placeholder until the daemon writes a title after the first completed exchange. Seven identical rows were the reason the fallback shows its age.' }
  ],
  diff: [
    { mark: '①', row: 'File index', text: 'Paths clip from the left — the filename is the part you are looking for. The index is the first panel dropped when the window narrows.' },
    { mark: '②', row: 'Hunk header', text: 'Cyan appears only on hunk headers, so a fold boundary is never misread as a state dot. Rows keep the prose colour and let the sign and tint carry the change.' }
  ],
  terminals: [
    { mark: '①', row: 'INTERACTIVE', text: 'A shell you can type into is one someone has to come back to, so it wears amber — including the root shell in /etc that nobody remembers opening.' },
    { mark: '②', row: 'Output tail', text: 'The panel reads a mirror of the output, never the buffer the agent polls, so watching a run cannot change what the agent sees.' }
  ],
  model: [
    { mark: '①', row: 'ollama', text: 'An unreachable backend is marked at selection time rather than failing on the first turn. Xerxes never launches the host service for you.' }
  ],
  settings: [
    { mark: '①', row: 'Permission mode', text: 'Researcher and plan enforce read-only ceilings and a non-YOLO policy: changing the palette cannot silently retain write or command access.' }
  ],
  style: [
    { mark: '①', row: 'Six voices', text: 'The eight-step text ramp does the work colour usually does, so a screen can be fully legible while only two or three things are actually coloured.' },
    { mark: '②', row: 'Density order', text: 'Six one-line agents beat three legible ones when you are scanning for the amber dot, so cards collapse rather than reflow.' }
  ]
};



/**
 * Index a fixture list, stating once that these lists are non-empty.
 *
 * The screens select by index (`AGENTS[state.agent]`), and every fallback was
 * itself another index, so `noUncheckedIndexedAccess` flagged each one. A
 * throw here is the honest reading: an empty fixture list is a build mistake,
 * not a state the UI should try to render around.
 */
function pick<T>(list: readonly T[], index: number): T {
  const value = list[index] ?? list[0]
  if (value === undefined) throw new Error('desktop fixtures must not be empty')
  return value
}

/** Same guarantee for keyed fixture records. */
function lookup<T>(record: Readonly<Record<string, T>>, key: string): T | undefined {
  return record[key]
}

export interface DesktopState {
  screen: string; turn: string; agent: number; term: number; file: number; model: number;
  perm: string; mode: string; theme: string; frame: number; w: number;
}

export const INITIAL_STATE: DesktopState = { screen: 'session', turn: 'fanout', agent: 1, term: 0, file: 0, model: 0, perm: 'accept-all', mode: 'code',
    theme: 'dark', frame: 0, w: typeof window === 'undefined' ? 1600 : window.innerWidth };

export interface DesktopProps { callouts?: boolean; keyHints?: boolean; rail?: boolean }

/** Pure: the same state always produces the same screen. */
export function buildView(
  state: DesktopState,
  set: (patch: Partial<DesktopState>) => void,
  props: DesktopProps = {},
): Record<string, any> {
  const s_ = state;

    const s = s_;
    // The density contract, honoured rather than just documented: side panels
    // go before the centre pane is squeezed — notes first, then the rail.
    const theme = s.theme === 'auto'
      ? (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
      : s.theme;
    const derafsh = theme === 'light' ? DERAFSH_LIGHT : DERAFSH_DARK;
    const keyHints = props.keyHints ?? true;
    // Notes outrank the agents rail: the annotations are the point of the
    // artboard, so the rail is the panel that goes first.
    const callouts = (props.callouts ?? true) && s.w >= 900;
    const railOpen = (props.rail ?? true) && s.w >= (callouts ? 1230 : 1000);
    const on = (id: any) => s.screen === id;

    const nav = NAV.map(item => ({
      ...item,
      bg: on(item.id) ? 'var(--x-selected)' : 'transparent',
      mark: on(item.id) ? DS.working : 'transparent',
      fg: on(item.id) ? 'var(--x-strong)' : 'var(--x-menubar-fg)',
      dot: on(item.id) ? DS.working : 'var(--x-numeric)',
      pick: () => set({ screen: item.id })
    }));

    const agents = AGENTS.map((a, i) => {
      const sk = skin(a.state);
      return {
        ...a, heading: a.group, headFg: sk.dot, dot: sk.dot, ground: sk.ground, border: sk.border,
        rail: i === s.agent ? sk.dot : 'transparent', lineFg: sk.text,
        leader: lead(30 - a.task.length + 20), pick: () => set({ agent: i, screen: 'agents' })
      };
    });
    const agent = (() => {
      const a = pick(AGENTS, s.agent);
      const sk = skin(a.state);
      return { ...a, dot: sk.dot, lineFg: sk.text,
        calls: a.calls.map(c => ({ ...c, dot: skin(c.state).dot, leader: lead(22 - c.verb.length) })),
        files: a.files.map(path => ({ path })) };
    })();

    const railRows = AGENTS.filter(a => a.state === 'working' || a.state === 'needsInput').map(a => {
      const sk = skin(a.state);
      const i = AGENTS.indexOf(a);
      return { name: a.name + ' — ' + a.task, budget: a.budget, dot: sk.dot, ground: sk.ground,
        border: sk.border, rail: sk.dot, pick: () => set({ agent: i, screen: 'agents' }) };
    });

    const sessionRows = SESSIONS.map((x, i) => {
      const sk = skin(x.state);
      return { ...x, heading: x.group, headFg: sk.dot, dot: sk.dot, lineFg: x.current ? DS.activity : 'var(--x-meta)',
        bg: x.current ? 'var(--x-selected)' : 'transparent', leader: lead(52 - x.title.length),
        pick: () => set({ screen: 'session' }) };
    });

    const terms = TERMS.map((t, i) => {
      const sk = skin(t.state);
      return { ...t, heading: t.group, headFg: sk.dot, dot: sk.dot,
        bg: i === s.term ? 'var(--x-selected)' : 'transparent', pick: () => set({ term: i }) };
    });
    const term = pick(TERMS, s.term);
    const tailFg = { meta: 'var(--x-meta)', prose: 'var(--x-prose)', done: DS.done, failed: DS.failedText, working: DS.activity };
    const termTail = term.tail.map(l => ({ text: l.text || ' ', fg: lookup(tailFg, l.c) ?? 'var(--x-prose)' }));

    const file = pick(FILES, s.file);
    const diffFiles = FILES.map((f, i) => ({ ...f, bg: i === s.file ? 'var(--x-selected)' : 'transparent', pick: () => set({ file: i }) }));
    const diffLines = file.lines.map(l => ({
      no: l.sign === '+' ? l.no : l.sign === '-' ? l.no : l.no,
      sign: l.sign, text: l.text || ' ',
      bg: l.sign === '+' ? 'var(--x-diff-add)' : l.sign === '-' ? 'var(--x-diff-del)' : 'transparent',
      signFg: l.sign === '+' ? DS.done : l.sign === '-' ? DS.failed : 'var(--x-separator)',
      fg: l.sign === ' ' ? 'var(--x-diff-context)' : DS.prose
    }));

    const models = MODELS.map((m, i) => ({
      ...m, glyph: i === s.model ? '◆' : '◇',
      dot: i === s.model ? DS.working : 'var(--x-numeric)',
      border: i === s.model ? 'var(--x-focus)' : DS.hairline,
      bg: i === s.model ? 'var(--x-selected)' : 'var(--x-card)',
      leader: lead(40 - m.name.length), pick: () => set({ model: i })
    }));

    const perms = [
      { id: 'accept-all', name: 'accept-all', note: 'YOLO. Every tool runs; static policy denials still win.', key: '/yolo' },
      { id: 'auto', name: 'auto', note: 'Routes approvals by risk; writes and commands surface.', key: '1' },
      { id: 'manual', name: 'manual', note: 'Every privileged call waits on you.', key: '2' },
      { id: 'plan', name: 'plan', note: 'Read-only ceiling. Design only, no writes.', key: '3' }
    ].map(p => ({ ...p, glyph: p.id === s.perm ? '◆' : '◇', dot: p.id === s.perm ? DS.working : 'var(--x-numeric)',
      border: p.id === s.perm ? 'var(--x-focus)' : DS.hairline, bg: p.id === s.perm ? 'var(--x-selected)' : 'var(--x-card)',
      pick: () => set({ perm: p.id }) }));

    const MODE_NOTE = {
      code: 'Normal implementation. Code is deliberately un-hued — the default mode adds no colour to the screen at all.',
      researcher: 'Evidence-first research. Borrows the working blue, and enforces a read-only tool ceiling.',
      plan: 'Plan-only design. Borrows the structure teal; no writes, no commands.',
      objective: 'An iterative objective loop with verification gates. Borrows the activity violet.'
    };
    const MODE_DOT = { code: 'var(--x-mode-code)', researcher: DS.working, plan: DS.structure, objective: DS.activity };
    const modes = ['code', 'researcher', 'plan', 'objective'].map(id => ({
      id, name: id, dot: lookup(MODE_DOT, id),
      border: id === s.mode ? 'var(--x-focus)' : DS.hairline, bg: id === s.mode ? 'var(--x-selected)' : 'var(--x-card)',
      pick: () => set({ mode: id })
    }));

    const runtimeRows = [
      { label: 'Bun runtime', value: '1.3.12' },
      { label: 'Daemon protocol', value: 'v35 · unix socket' },
      { label: 'XERXES_HOME', value: '~/.xerxes' },
      { label: 'Skills discovered', value: '31 · project, user, bundled' },
      { label: 'MCP servers', value: '2 connected · 1 idle' },
      { label: 'Animations', value: 'on · XERXES_TUI_ANIMATIONS=1' }
    ].map(r => ({ ...r, leader: lead(34 - r.label.length) }));

    const voices = [
      { name: 'working', hex: DS.working, means: 'a turn is in flight' },
      { name: 'done', hex: DS.done, means: 'finished; nothing owed' },
      { name: 'failed', hex: DS.failed, means: 'over, and already paid for' },
      { name: 'needsInput', hex: DS.needsInput, means: 'a human is required' },
      { name: 'activity', hex: DS.activity, means: 'the latest live activity line' },
      { name: 'structure', hex: DS.structure, means: 'hunk headers, and nothing else' }
    ];
    const ramp = [
      { name: 'strong', hex: 'var(--x-strong)', sample: 'the thing you must read first' },
      { name: 'title', hex: 'var(--x-title)', sample: 'row titles and panel headings' },
      { name: 'prose', hex: 'var(--x-prose)', sample: 'answers, diff bodies, output' },
      { name: 'secondary', hex: 'var(--x-secondary)', sample: 'tool verbs and goals' },
      { name: 'meta', hex: 'var(--x-meta)', sample: 'thinking, ages, counts' },
      { name: 'numeric', hex: 'var(--x-numeric)', sample: '8.2k tok · 3 tools · 41s' },
      { name: 'caption', hex: 'var(--x-caption)', sample: 'GROUP CAPTIONS' },
      { name: 'separator', hex: 'var(--x-separator)', sample: '· between facts' }
    ];
    const glyphs = [
      { glyph: '●', job: 'state — always coloured, never bare' },
      { glyph: '✦', job: 'brand, and the cwd marker' },
      { glyph: '⏺', job: 'a tool call' },
      { glyph: '❯', job: 'you are typing' },
      { glyph: '└', job: 'the ledger line that closes a turn' },
      { glyph: '↳', job: 'soft wrap, so a wrap is never a second command' },
      { glyph: '▸ ▾', job: 'expandable, collapsed and open' },
      { glyph: '◆', job: 'interaction mode' }
    ];
    const density = [
      { n: '1st', what: 'secondary counts', at: '< 64 cols' },
      { n: '2nd', what: 'caption source labels', at: '< 76 cols' },
      { n: '3rd', what: 'card goal lines', at: '< 88 cols' },
      { n: '4th', what: 'side panels', at: '< 100 cols' }
    ];

    const isFanout = s.turn === 'fanout';
    const isApproval = s.turn === 'approval';
    const tableDef = lookup(TABLES, s.screen) ?? null;
    const table = !tableDef ? null : {
      ...tableDef,
      rows: tableDef.rows.map((row: any) => {
        const sk = skin(row.state || 'done');
        return { ...row, glyph: '●', dot: sk.dot, headFg: sk.dot, lineFg: sk.text,
          leader: lead(62 - String(row.name).length - String(row.sub || '').length) };
      })
    };
    const turnTabs = [
      { id: 'idle', label: 'idle', key: '/new' },
      { id: 'streaming', label: 'streaming', key: '⏎' },
      { id: 'fanout', label: 'fan-out', key: 'F6' },
      { id: 'approval', label: 'approval', key: '1–4' }
    ].map(t => ({ ...t, bg: t.id === s.turn ? 'var(--x-selected)' : 'transparent',
      fg: t.id === s.turn ? 'var(--x-strong)' : 'var(--x-secondary)', pick: () => set({ turn: t.id }) }));

    const tools = [
      { verb: 'read', arg: 'xerxes/src/streaming/loop.ts', dur: '0.4s', state: 'done' },
      { verb: 'grep', arg: 'synthetic · 12 hits', dur: '0.2s', state: 'done' },
      { verb: 'read', arg: 'xerxes/src/ui/app/turnController.ts', dur: '0.6s', state: 'done' },
      { verb: 'edit', arg: 'loop.ts · +34 −11', dur: isFanout ? '0.9s' : '—', state: isFanout ? 'done' : 'working' }
    ].map(t => ({ ...t, dot: skin(t.state).dot, leader: lead(46 - t.arg.length) }));

    const cards = AGENTS.slice(0, 3).map(a => {
      const sk = skin(a.state);
      return { name: a.name, task: a.task, line: a.line, budget: a.budget,
        dot: sk.dot, ground: sk.ground, border: sk.border, lineFg: sk.text };
    });

    const railIsInspector = s.screen === 'agents';

    return {
      callouts, keyHints, railOpen,
      mark1: callouts ? ' ①' : '', mark2: callouts ? ' ②' : '', mark3: callouts ? ' ③' : '',
      theme,
      artLines: DERAFSH_ART.map((text, row) => ({
        text, color: gradientColor(derafsh, wavePosition(row, DERAFSH_ART.length, s.frame))
      })),
      wordmark: (() => {
        const letters = [...'XERXES'];
        const ramp = gradientRamp(derafsh, letters.length);
        return Array.from({ length: 6 }, (_, row) => ({
          cells: letters.map((letter, i) => ({ text: pick(lookup(WORDMARK_GLYPHS, letter) ?? [], row), color: ramp[i] }))
        }));
      })(),
      nav, liveSessions: sessionRows.filter(x => x.state !== 'done' || x.current).slice(0, 3)
        .map(x => ({ ...x, meta: (x.line || '').replace(/^└ /, '') || x.age })),
      liveCount: '3 chats · 1 working', modeLabel: s.mode,
      windowTitle: 'Make the streaming loop cancel-safe',
      turnBadge: s.turn === 'idle' ? '● idle' : '● working · 41s',
      isSession: on('session'), isAgents: on('agents'), isSessions: on('sessions'), isDiff: on('diff'),
      isTerminals: on('terminals'), isModel: on('model'), isSettings: on('settings'), isStyle: on('style'),
      isIdle: s.turn === 'idle', isTranscript: s.turn !== 'idle', isFanout,
      sessionTitle: 'Make the streaming loop cancel-safe', sessionId: 'daemon:8f21c4',
      turnTabs, chips: [
        { key: '1', command: '/agents', label: 'check the fleet', consequence: '2 working · 1 needs you', dot: DS.needsInput },
        { key: '2', command: '/diff', label: 'review the working tree', consequence: '3 files · +103 −13', dot: DS.working },
        { key: '3', command: '', label: 'map this repo', consequence: 'entry points, hot paths, dead code', dot: DS.working }
      ],
      tools, cards, isApproval, isTable: s.screen === 'skills', table,
      isProvider: on('provider'), isTools: on('tools'), isMcp: on('mcp'), isAgentDefs: on('agentdefs'),
      isCron: on('cron'), isPermissions: on('permissions'), isMemory: on('memory'),
      isHistory: on('history'), isStatus: on('status'), isChannels: on('channels'), isHelp: on('help'),
      profiles: PROFILES.map(p => {
        const sk = skin(p.state);
        return { ...p, dot: sk.dot, ground: sk.ground, border: sk.border, noteFg: sk.text,
          credFg: p.state === 'failed' ? 'var(--x-numeric)' : 'var(--x-prose)' };
      }),
      toolRows: TOOL_ROWS.map(t => {
        const sk = skin(t.state);
        return { ...t, heading: t.heading || '', dot: sk.dot, stateFg: sk.dot };
      }),
      servers: SERVERS.map(v => {
        const sk = skin(v.st);
        return { ...v, dot: sk.dot, ground: sk.ground, border: sk.border };
      }),
      plugins: PLUGINS,
      tree: TREE.map(n => ({ ...n, dot: skin(n.state).dot,
        policyFg: /read-only|no writes/.test(n.policy) ? 'var(--x-structure)' : 'var(--x-numeric)' })),
      jobs: JOBS.map(j => {
        const sk = skin(j.state);
        return { ...j, dot: sk.dot, ground: sk.ground, border: sk.border,
          lastFg: j.state === 'needsInput' ? 'var(--x-needs-text)' : 'var(--x-numeric)',
          marks: j.marks.map(m => ({ at: m.at, color: skin(m.s).dot })) };
      }),
      hourTicks: ['00', '04', '08', '12', '16', '20', '24'].map(label => ({ label })),
      matrixModes: MATRIX_MODES,
      matrix: MATRIX.map(row => ({
        name: row.name,
        cells: row.cells.map(c => c === 'y'
          ? { glyph: '●', label: 'yes', fg: 'var(--x-done)' }
          : c === 'g'
            ? { glyph: '◐', label: 'gated', fg: 'var(--x-needs)' }
            : { glyph: '○', label: 'no', fg: 'var(--x-separator)' })
      })),
      denials: DENIALS, tiers: TIERS.map(t => ({ ...t, cap: t.cap || '',
        dot: t.voice ? 'var(--x-working)' : 'var(--x-separator)' })),
      weights: WEIGHTS, memFiles: MEM_FILES,
      timeline: TIMELINE.map(t => ({ ...t, dot: skin(t.state).dot,
        subFg: t.state === 'working' ? 'var(--x-activity)' : 'var(--x-meta)' })),
      ops: OPS,
      checks: CHECKS.map(c => {
        const sk = skin(c.state);
        return { ...c, heading: c.heading || '', line: c.line || '', lineFg: sk.text, dot: sk.dot,
          glyph: c.state === 'done' ? '✓' : c.state === 'needsInput' ? '!' : '·' };
      }),
      callers: CALLERS.map(c => {
        const sk = skin(c.st);
        return { ...c, dot: sk.dot, ground: sk.ground, border: sk.border };
      }),
      daemonOwns: DAEMON_OWNS, helpKeys: HELP_KEYS, helpCommands: HELP_COMMANDS,
      approvalOptions: [
        { key: '1', label: 'allow once', fg: 'var(--x-prose)' },
        { key: '2', label: 'allow for this session', fg: 'var(--x-prose)' },
        { key: '3', label: 'deny', fg: 'var(--x-prose)' },
        { key: '4', label: 'deny, and tell the agent why', fg: 'var(--x-needs-text)' }
      ],
      ledger: isFanout ? '4 tools · 3 agents · 34.8k tok · 2m 41s' : isApproval ? 'waiting on you · 3 tools · 14.6k tok' : '3 tools · 12.1k tok · 41s',
      composer: isApproval ? '' : isFanout ? 'take reviewer\'s question first — ' : 'and keep the parity test green',
      composerHint: isApproval ? 'blocked · answer the card above' : isFanout ? 'queued · sends when the turn settles' : 'steers the live turn',
      composerKeys: [
        { key: '⏎', label: 'send' }, { key: '⇥', label: 'mode' }, { key: '←', label: 'background' },
        { key: 'F6', label: 'agents' }, { key: 'F7', label: 'diff' }, { key: 'F8', label: 'terminals' }, { key: '⎋', label: 'interrupt' }
      ],
      agents, agent, agentSummary: '5 agents · 2 working · 1 needs you',
      agentKeys: [ { key: '↑↓', label: 'move' }, { key: '⏎', label: 'inspect' }, { key: '⎋', label: 'back' }, { key: 'r', label: 'retry' }, { key: 'F6', label: 'close' } ],
      railRows, railIsFleet: !railIsInspector, railIsInspector,
      railTitle: railIsInspector ? 'INSPECTOR' : 'AGENTS · 3',
      railKey: railIsInspector ? '⎋' : 'F6',
      railFoot: railIsInspector ? 'Esc returns to the list, Esc again to the main agent' : 'Enter on a row inspects one agent',
      sessionRows, sessionSummary: '3 chats · 1 working',
      sessionKeys: [ { key: '␣', label: 'peek' }, { key: '→', label: 'attach' }, { key: '←', label: 'background' }, { key: '⌫', label: 'delete' } ],
      diffFiles, diffLines, hunkHeader: file.hunk, diffSummary: '3 files · +103 −13',
      diffKeys: [ { key: 'j k', label: 'hunk' }, { key: '⇥', label: 'file' }, { key: 'f', label: 'fold' }, { key: 'F7', label: 'close' } ],
      terms, term, termTail, termSummary: '5 shells · 1 running · 1 interactive',
      termKeys: [ { key: 'i', label: 'send input' }, { key: 'c', label: 'interrupt' }, { key: 'k', label: 'kill' }, { key: 'K', label: 'force kill' }, { key: 'F8', label: 'close' } ],
      providers: [
        { name: 'anthropic', state: 'current', dot: DS.working, bg: 'var(--x-selected)' },
        { name: 'openai', state: 'ready', dot: DS.done, bg: 'transparent' },
        { name: 'google', state: 'ready', dot: DS.done, bg: 'transparent' },
        { name: 'ollama', state: 'offline', dot: DS.failed, bg: 'transparent' },
        { name: 'lmstudio', state: 'no key', dot: 'var(--x-numeric)', bg: 'transparent' }
      ],
      models, modelSummary: '4 models · 1 provider offline',
      modelKeys: [ { key: '↑↓', label: 'move' }, { key: '⏎', label: 'select' }, { key: '/', label: 'filter' }, { key: '⎋', label: 'cancel' } ],
      perms, modes, modeNote: lookup(MODE_NOTE, s.mode), runtimeRows,
      themes: [
        { id: 'auto', name: 'auto', note: 'OSC probe' },
        { id: 'dark', name: 'dark', note: 'Nocturne dark' },
        { id: 'light', name: 'light', note: 'Nocturne light' }
      ].map(t => ({ ...t, glyph: t.id === s.theme ? '◆' : '◇',
        dot: t.id === s.theme ? 'var(--x-working)' : 'var(--x-numeric)',
        border: t.id === s.theme ? 'var(--x-focus)' : 'var(--x-hairline)',
        bg: t.id === s.theme ? 'var(--x-selected)' : 'var(--x-card)',
        pick: () => set({ theme: t.id }) })),
      voices, ramp, glyphs, density,
      statusModel: pick(MODELS, s.model).name,
      permLabel: s.perm, permFg: s.perm === 'accept-all' ? DS.needsInput : 'var(--x-secondary)',
      contextLabel: '18.4k / 256k',
      statusRight: keyHints ? '⌘/ style sheet · ⌘, settings · /help' : '',
      notes: lookup(NOTES, s.screen) ?? lookup(TABLE_NOTES, s.screen) ?? []
    };
}
