# Xerxes Agents for VS Code

Xerxes is a multi-agent coding runtime. This extension is a Xerxes chat in the VS Code sidebar: tasks with their agents, workflows, goals, approvals and questions, working on the folder you have open. Files open in the editor, changes in VS Code's diff, and Git in Source Control.

The Bun runtime ships inside the extension, so there is nothing else to install.

## Getting started

1. Open a folder.
2. Click the Xerxes icon in the activity bar.
3. Choose a provider in **Settings** (the gear in the view's title bar): an API key, a local endpoint, or a ChatGPT or Claude Code sign-in.

Click the task title to switch to an earlier task. Use **+** in the message box to add files as context.

Your tasks, settings and providers are shared with the Xerxes desktop app and the `xerxes` terminal UI on the same machine.

## Commands

| Command | What it does |
| --- | --- |
| **Xerxes: Send Selection to Xerxes** (`⌘⌥X` / `Ctrl+Alt+X`, or right-click) | Adds the selected code, with its file and lines, to the message box. |
| **Xerxes: New Task** | Starts a new task. |
| **Xerxes: Task History** | Switches to an earlier task in this folder. |
| **Xerxes: Search Message History** | Searches every task's messages. |
| **Xerxes: Command Palette** | Opens Xerxes' own command palette. |
| **Xerxes: Settings** | Providers, models and approvals. |
| **Xerxes: Activity** / **Xerxes: Usage** | Running agents and background work; token and cost use. |
| **Xerxes: Export Transcript** | Exports the current task as Markdown. |

## Remote machines

Open a folder on another machine with **VS Code Remote-SSH**. The extension then runs on that machine, and Xerxes works on its files there.

## Notifications

When the Xerxes view is hidden, VS Code tells you when a task finishes or when Xerxes needs your approval or has a question.

## License

Apache-2.0. The bundled Bun runtime is MIT-licensed; see `runtime/Bun-LICENSE.md`.
