---
name: authoring-tool-plugins
description: Build native Xerxes tool plugins with Bun tests and concrete usage examples in Plugin Creator mode.
---

# Native tool plugin authoring

1. Understand the requested tool and inspect the destination. Preserve existing files.
2. Create `.xerxes/plugins/<name>/index.ts`, `index.test.ts` and `README.md`.
3. Export `register(registry)` using the structural API below. Tool names use at most 57 letters, digits, underscores or hyphens. Register all tools and commands under one plugin name.
4. Prefer **structured tools**: an object with `description`, a JSON-schema `parameters` object (`type: "object"`) and `execute(input, context)`. The model sees that description and schema as `plugin_<name>`, so describe each parameter. `context.signal` aborts when the turn is cancelled; pass it to `fetch` or check it in long loops. A bare function is still accepted (positional `{ "args": [...] }`), but the model has to guess its arguments.
5. Validate every argument anyway — the schema guides the model, it does not guarantee input. Return serializable results. Keep registration free of filesystem, network and process side effects.
6. Optional slash commands: `registry.registerCommand(name, (args, context) => ..., { description })`. `args` is the text after the command; `context.cwd` is the session's directory. Names must not collide with built-in commands; return a string to show it.
7. Test normal input, empty input and invalid input with `bun test <path>/index.test.ts`. For external operations use injected ports and cover errors and cancellation. Do not require live credentials for tests.
8. Document the install command, a model request, the tool's arguments, expected result, the reload/disable/uninstall commands and prerequisites. Only claim verification actually performed.

```ts
interface ToolContext {
  readonly signal?: AbortSignal
  readonly sessionId?: string
  /** Declared secrets the user has set; see "Execution, permissions and secrets". */
  readonly secrets?: Readonly<Record<string, string>>
}
interface Permissions { network?: string[]; env?: string[]; filesystem?: string[]; secrets?: string[] }
interface Registry {
  registerPlugin(meta: { name: string; description?: string; version?: string; permissions?: Permissions }): void
  registerTool(name: string, tool: {
    description: string
    parameters: Record<string, unknown>
    timeoutMs?: number
    execute(input: Record<string, unknown>, context: ToolContext): unknown
  }, meta?: undefined, pluginName?: string): void
  registerCommand(name: string, handler: (args: string, context: { cwd?: string }) => unknown,
    options?: { description?: string }, meta?: undefined, pluginName?: string): void
}
export function register(registry: Registry): void {
  registry.registerPlugin({ name: 'greetings', description: 'Greeting helpers', version: '1.0.0' })
  registry.registerTool('greet', {
    description: 'Greet a person by name',
    parameters: { type: 'object', properties: { name: { type: 'string', description: 'Who to greet' } }, required: ['name'] },
    execute(input) {
      if (typeof input.name !== 'string' || !input.name.trim()) throw new Error('name must be a non-empty string')
      return { greeting: `Hello, ${input.name}!` }
    },
  }, undefined, 'greetings')
  registry.registerCommand('greet-me', args => `Hello, ${args.trim() || 'you'}!`, { description: 'Print a greeting' }, undefined, 'greetings')
}
```

After reviewing and testing, the user can run:

```text
/plugins install "/absolute/path/to/.xerxes/plugins/greetings/index.ts"
/plugins inspect greetings
```

Ask the model: “Greet Ada with the greetings plugin.” It calls `plugin_greet` with `{"name": "Ada"}`; expected result: `{"greeting":"Hello, Ada!"}`. `/greet-me Ada` prints `Hello, Ada!`.

From inside a task, `PluginManageTool` (`{"action": "install", "path": "...", "scope": "temporary"}`) installs it after the user approves; the tools are callable from your next step, and delegated agents with plugin access see them too. Temporary installs vanish when Xerxes restarts; use `"scope": "permanent"` only when the user wants the tool kept.

Lifecycle: `/plugins reload greetings` starts a fresh plugin process with the edited code (including edited helper files) without restarting Xerxes; a failed reload keeps the previous working version. `/plugins disable greetings` refuses new calls; `/plugins enable greetings` re-enables them. `/plugins uninstall greetings` removes it from Xerxes; the source files stay in place. Hooks, providers and channels require an embedding host and are not supported by managed installation.

## Execution, permissions and secrets

Each managed plugin runs in its own `bun --no-env-file` child process; tool and command calls cross JSON IPC, so arguments and results must be JSON-serializable. Each call has a time limit (default 60 000 ms; set `timeoutMs` on a structured tool, at most one hour). A timed-out call, or a cancelled call that ignores `context.signal` for 2 s, kills the process; the next call starts a fresh one. A crash never takes Xerxes down.

Declare what the plugin needs:

```ts
registry.registerPlugin({
  name: 'weather', description: 'Forecasts', version: '1.0.0',
  permissions: {
    network: ['api.weather.example', '*.cdn.example'], // hostnames fetch may reach
    env: ['TZ'],                                       // variables copied into the process
    filesystem: ['/tmp/weather-cache'],                // displayed only
    secrets: ['WEATHER_API_KEY'],                      // delivered as context.secrets
  },
})
```

What is actually enforced — say this honestly in READMEs:

- `env`: enforced. The process starts with an empty environment (no `.env` loading) and receives only declared variables.
- `network`: a guard rail on `fetch` only. `node:http`, `node:net`, `Bun.fetch` and child processes are not intercepted.
- `filesystem`: declared and shown to the user, not enforced.
- The plugin runs with the user's privileges. This is **not** a security sandbox.

Secrets: read them from `context.secrets.WEATHER_API_KEY` inside `execute`; only declared names that the user has set are present. The user sets a value in a terminal with `xerxes plugins secret set weather WEATHER_API_KEY` (input hidden; stored encrypted, mode 0600). `/plugins secret list weather` shows names and set/unset only; `/plugins secret unset weather WEATHER_API_KEY` removes one. Never ask for a secret in chat, never log it, never put it in code or tests.

## Installing, sharing and project plugins

- `PluginManageTool` install runs `index.test.ts` (or `<name>.test.ts` / `<name>.test.js`) beside the module first and refuses when it fails or is missing; `skip_tests: true` is shown to the user in the approval request. `/plugins install` by the user does not run tests.
- `/plugins export <name> <new-directory>` copies the plugin source, tests and README (never secrets, `.env*`, `.git` or `node_modules`) and refuses to overwrite.
- `/plugins install <https-or-ssh-git-url>` shallow-clones into `~/.xerxes/plugins/sources/` and prints a preview (files with line counts, the `permissions` text found in the source, README head). Nothing runs until `/plugins install --confirm <token>`. `file://`, `http://` and `git://` URLs are refused.
- A folder `.xerxes/plugins/<name>/index.ts` in a workspace is a project plugin: it loads only for that workspace, only after `/plugins trust <name>`, and editing any file in the folder withdraws trust until it is trusted again. Project plugins may register tools, not slash commands.
- When a turn runs the same command shape three or more times, Xerxes suggests `/plugin-creator`; it never creates anything by itself.
