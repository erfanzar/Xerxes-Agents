---
name: node-inspect-debugger
description: Debug Node.js via --inspect plus Chrome DevTools Protocol.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [debugging, nodejs, node-inspect, cdp, breakpoints]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/node-inspect-debugger/SKILL.md
---

# Node.js Inspect Debugger

## Overview

When `console.log` is not enough, drive Node's built-in V8 inspector from the
terminal: real breakpoints, step in/over/out, call-stack walking, scope
dumps, and expression evaluation in the paused frame.

Two approaches:

- **`node inspect`** — built-in, zero install, CLI REPL. Best for quick poking.
- **CDP via `chrome-remote-interface`** — scriptable; best when automating
  many breakpoints or collecting state across runs non-interactively.

Prefer `node inspect` first: always available and fast.

## When to Use

- A Node/Bun test fails and you need to see intermediate state
- A TUI or daemon child process misbehaves and you want pre-render state
- You need a closure value `console.log` cannot reach without patching
- Perf: attach to a running process for a CPU profile or heap snapshot

Do not use for problems `console.log` solves in under a minute.

## Quick Reference: `node inspect` REPL

```bash
node inspect path/to/script.js
# TypeScript via tsx
node --inspect-brk --import tsx path/to/script.ts
```

| Command | Action |
|---|---|
| `c` / `n` / `s` / `o` | continue / step over / step into / step out |
| `pause` | pause running code |
| `sb('file.js', 42)` | breakpoint at file.js line 42 |
| `sb('functionName')` | break when function is called |
| `cb('file.js', 42)` | clear breakpoint |
| `breakpoints` | list breakpoints |
| `bt` | backtrace (call stack) |
| `list(5)` | 5 lines of source around current position |
| `watch('expr')` / `watchers` | watched expressions |
| `repl` | REPL in current scope (Ctrl+C exits back to `debug>`) |
| `exec expr` | evaluate once |
| `restart` / `kill` / `.exit` | restart / kill / quit |

In `repl` sub-mode any JS expression works, including locals and closure
variables.

## Attaching to a Running Process

```bash
# 1. Enable the inspector on an existing process
kill -USR1 <pid>       # Node prints: Debugger listening on ws://127.0.0.1:9229/<uuid>
# 2. Attach
node inspect -p <pid>
# or by URL
node inspect ws://127.0.0.1:9229/<uuid>
```

Start with the inspector from the beginning:

```bash
node --inspect script.js             # listen on 127.0.0.1:9229, keep running
node --inspect-brk script.js         # listen AND pause on first line
node --inspect=0.0.0.0:9230 x.js     # custom host:port (avoid 0.0.0.0; see Pitfalls)
```

## Programmatic CDP (scripting from the terminal)

Install `chrome-remote-interface` to a throwaway location so the project
stays clean, start the target with `node --inspect-brk=9229 target.js`,
then run a driver:

```javascript
// /tmp/cdp-debug.js
const CDP = require('chrome-remote-interface');
(async () => {
  const client = await CDP({ port: 9229 });
  const { Debugger, Runtime } = client;
  Debugger.paused(async ({ callFrames, reason }) => {
    const top = callFrames[0];
    console.log(`PAUSED: ${reason} @ ${top.url}:${top.location.lineNumber + 1}`);
    for (const scope of top.scopeChain) {
      if (scope.type === 'local' || scope.type === 'closure') {
        const { result } = await Runtime.getProperties({
          objectId: scope.object.objectId, ownProperties: true,
        });
        for (const p of result) {
          console.log(`  ${scope.type}.${p.name} =`, p.value?.value ?? p.value?.description);
        }
      }
    }
    await Debugger.resume();
  });
  await Runtime.enable();
  await Debugger.enable();
  await Debugger.setBreakpointByUrl({ urlRegex: '.*app\\.js$', lineNumber: 119, columnNumber: 0 });
  await Runtime.runIfWaitingForDebugger();
})();
```

```bash
mkdir -p /tmp/cdp-tools && cd /tmp/cdp-tools && npm i chrome-remote-interface
NODE_PATH=/tmp/cdp-tools/node_modules node /tmp/cdp-debug.js
```

Swap `Debugger` for `HeapProfiler` / `Profiler` for heap snapshots and CPU
profiles; write the result to `/tmp` and open it in Chrome DevTools.

## Running Tests Under the Debugger

```bash
node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/foo.test.ts
```

In another terminal: `node inspect -p <pid>`, then `sb('src/foo.ts', 42)`,
`cont`. Use `--no-file-parallelism` (vitest) or `--runInBand` (jest) so only
one worker exists.

## Common Pitfalls

1. **TS line numbers.** Breakpoints hit emitted JS, not `.ts`. Break in built
   output, or use CDP clients that follow sourcemaps (`node
   --enable-source-maps`). The `node inspect` CLI does not.
2. **`--inspect` vs `--inspect-brk`.** Plain `--inspect` races past an early
   breakpoint if you attach late; use `--inspect-brk` to pause on first line.
3. **Port collisions.** Default is 9229. Use `--inspect=0` and read the real
   URL from `curl -s http://127.0.0.1:9229/json/list`.
4. **Child processes.** Inspecting a parent does not inspect children.
   Propagate with `NODE_OPTIONS='--inspect-brk'`.
5. **Paused targets.** Quitting `node inspect` while the target is paused
   leaves it paused: `cont` first or `kill` the target.
6. **Interactive REPLs.** Run `node inspect` through a PTY-capable
   shell/terminal tool session; one-shot non-PTY calls cannot step
   interactively.
7. **Security.** `--inspect` exposes arbitrary code execution. Bind to
   `127.0.0.1` unless on an isolated network.

## Verification Checklist

- [ ] `curl -s http://127.0.0.1:9229/json/list` returns exactly the target
- [ ] First breakpoint actually hits (else you missed `--inspect-brk` or attached late)
- [ ] Source listing at pause shows the right file (mismatch = sourcemap issue)
- [ ] `exec process.pid` in `repl` returns the PID you meant to attach to

## One-Shot Recipes

"Why is this variable undefined at line X?": `node --inspect-brk script.js`,
`sb('script.js', X)`, `cont`, then `repl` and inspect the variable.

"What is the call path into this function?": `sb('suspectFn')`, `cont`, `bt`.

"This async chain hangs — where?": start with plain `--inspect`, let it hang,
then `pause` and `bt` to see the stuck frame.

---

Adapted from the `node-inspect-debugger` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent.
