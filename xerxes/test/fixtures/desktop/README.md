# Native desktop layout verification

From the repository root, run:

```sh
bun xerxes/scripts/previewDesktop.ts
```

This opens the production React shell in an isolated Electron window with a deterministic bridge and a temporary user profile. It does not attach to project daemons, read credentials, or mutate saved sessions. The launcher prints the fixture PID and artifact directory. Quit this window normally when finished.

The application menu selects Welcome, Populated, Streaming, Reconnecting, Error, and Onboarding states, and Narrow (720×700), Normal (1200×820), or Wide (1600×1000) sizes. Cmd+Alt+1 through Cmd+Alt+6 also select the six scenarios. Use the native zoom action for a maximized window.

The populated state includes forty sessions, eight agents, a long goal, sixty models, 167 skills, deep file paths, and a 200-line diff. Exercise the real shell controls: paste and send a multiline draft; expand the goal and agent rows; filter/select models and skills; select/attach a file; open Changes; resize the dividers with keyboard arrows; switch sessions and workspaces. Scenario changes preserve the mounted shell's layout preferences.

The bridge only supplies fixture responses for layout verification. This is not a live provider, SSH, credential, scheduler, or filesystem integration test. Separately build the app with `bun run --cwd xerxes build:desktop`, launch the packaged app, and inspect real saved sessions and file browsing against its existing daemon.
