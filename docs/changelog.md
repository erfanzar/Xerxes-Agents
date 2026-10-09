# Changelog

Selected highlights from the project's git history, grouped by major theme. For the full history, run `git log --oneline --no-merges`.

The current native work is the Bun/TypeScript cutover. Historical entries below intentionally
describe the earlier implementation and are not current setup instructions.

---

## 0.6.46 — 2026-10-09

- Claude Code models call tools natively. Xerxes declares its tools to `claude -p` over a loopback MCP endpoint, so a call is a real tool_use and the API itself ends the reply there. With the text protocol, a reply ended only when the model wrote the block close, and `claude -p` has no stop sequences: one subagent wrote 1,336 calls and 2,673 blank lines in a single 23-minute reply, and two more ran away the same way. Xerxes still runs every tool itself, through its own permissions and policy; Claude Code runs with `--permission-mode dontAsk` and never executes them. If Claude Code cannot reach the endpoint, that request falls back to the text protocol.

## 0.6.45 — 2026-10-09

- An attached image reaches the model. A detailed render pasted as a 7.6 MB PNG was over the API's 5 MB image limit, and the model saw only a placeholder. An image still over the limit after scaling to 2000px is now sent as a JPEG small enough to fit.

## 0.6.44 — 2026-10-09

- Claude Code keeps its prompt cache. A task with a pasted screenshot fell to a 26% cache hit: every round re-wrote the whole conversation (500K tokens) and read back only the system prompt. Two causes, both measured through `claude -p`:
  - Claude Code resizes an image wider than 2000px itself, and a resized image stops the cache from extending. Pasted and dropped images are now scaled to 2000px on their longest side before they are attached; the API scales them to 1568px anyway.
  - The API looks back only 20 blocks from a cache marker. Each tool result was its own block, so a step with 19 or more calls re-wrote everything on the next round. One step's results now share a block.

## 0.6.43 — 2026-10-09

- An attached image shows whole above your message, at a readable size, instead of as a small cropped thumbnail inside the bubble.

## 0.6.42 — 2026-10-09

- An image you attach shows in your message, in the desktop app and VS Code: when you send it, and when the task reloads. History keeps a screenshot up to 256 KB inline; a larger one shows as "[image omitted: N KB]" after a reload, while the model always received the full image.

## 0.6.41 — 2026-10-09

- A monitor reaction runs until its watch ends, not 60 seconds. A reacting watch without an explicit timeout, including "tell me when this command finishes", cancelled its reaction turn after 60 seconds ("Reaction deadline exceeded"), so on a large-context model the turn was stopped before its first round finished. A reaction is a turn, bounded by the turn's own limits, Stop and the watch's lifetime; an explicit timeout can now be up to the watch's 24 hours instead of 2 minutes.

## 0.6.40 — 2026-10-09

- Paste or drop images into the message box, in the desktop app and VS Code. A screenshot shows as a thumbnail you can remove and is sent with the message; the runtime already accepted images, but the message box never sent any. While a step is running, images wait for it to finish, because a steer carries text only.
- The VS Code release build keeps each verified Bun download, so a dropped connection costs one file instead of the whole build.

## 0.6.39 — 2026-10-08

- VS Code: the runtime update notice is readable. It used the theme's warning fill without the text colour paired with it, so light text sat on a pale yellow; it now uses the editor widget surface and text, with the warning colour as its edge.

## 0.6.38 — 2026-10-08

- VS Code shows when the runtime it is talking to is older than the extension. The runtime is shared, so reloading a window reconnects to the one an earlier install launched, and an update that was waiting on running work was invisible: a 0.6.33 runtime kept serving after 0.6.37 was installed, and none of the fixes in between ran. The view now says the update is pending, what it waits for, and offers Restart now.

## 0.6.37 — 2026-10-08

- Compaction works for OpenRouter models with a vendor prefix (`mistralai/mistral-large-4-0`) on every path — `/compact`, the check before a turn, and subagents — not only mid-turn. The summary request asked the bare provider registry which provider owns the model id, only to choose a reasoning hint, and `mistralai` is an OpenRouter vendor, not a provider; every compaction failed with "unknown provider prefix 'mistralai'". The client is already routed by the session's profile, so an id the registry cannot place now goes without the hint.

## 0.6.36 — 2026-10-08

- Compaction waits for the threshold the context meter shows. It reserved the model's whole output ceiling before applying the threshold, and for a model whose ceiling is half its window (Mistral Large 4: 262K of 524K) that compacted a task at 40% full, every round. The threshold is now a share of the window, and each round's `max_tokens` is fitted to the room the window has left, so a long prompt no longer makes a request that exceeds the window. Only a `max_tokens` you pin is still reserved.

## 0.6.35 — 2026-10-08

- Compaction during a turn reaches the provider the task uses. It resolved the provider implicitly, found none on that path, and fell back to the default connection, which read an OpenRouter vendor (`mistralai/mistral-large-4-0`) as a provider prefix and failed with "unknown provider prefix 'mistralai'". The turn now hands compaction its own session.

## 0.6.34 — 2026-10-08

- Compaction is visible: the composer shows its progress ("Compacting: summary request 2…") with the working ring, and says the task will take messages when it finishes. It used to show nothing while the task was busy, and a send went nowhere.
- VS Code: the working and compacting status line lines up with the message box in a wide view instead of drifting into a centred column.
- A request stays alive while the runtime keeps reporting progress. A multi-minute `/compact` failed with "rpc timeout: slash (120000ms)" and dropped the connection, which cancelled the compaction; a runtime that goes quiet still times out at the deadline.
- VS Code: one conversation column — replies, your messages, status, to-dos and the message box share its width and edges, full width in a sidebar and a centred column in a wide view, instead of 760px replies beside full-width everything else.
- VS Code: a wide view (a full window or wide editor tab) shows the task's activity beside the conversation — status, steps, context, to-dos, goal and agents.

## 0.6.33 — 2026-10-08

- Claude Code compaction finishes. Claude Code's output cap covers thinking as well as the reply, so a compaction summary's 2,048-token cap left opus nothing to write after it thought, and compaction failed with "summary did not finish (length)". With thinking on, the model's own reported output limit now stands, as the Anthropic path raises its cap past the thinking budget; with thinking off, the cap is exact.

## 0.6.32 — 2026-10-08

- Compaction aims at what the context meter will measure. The meter scales its estimate by each task's measured calibration and adds the system prompt and tools; compaction was given the raw budget, so a task calibrated at 3.8x read 206% full while compaction judged it to fit, shed nothing, and every turn stopped with "Context remains above the automatic compaction threshold".

## 0.6.31 — 2026-10-07

Claude Code replies that end where their tool calls end.

- Claude Code is asked to put a step's calls in one `<tool_calls>` block and close it, and the reply ends at the close — what the API's stop sequence does, which `claude -p` lacks. One reply had run on past its calls for 135,461 tokens and 1,176 reads of one file, which filled the context and broke the task.
- A request is one model response. When a response hits the output limit, Claude Code quietly asks the model to continue; Xerxes no longer takes that continuation as part of the reply.

## 0.6.30 — 2026-10-06

- Compaction makes room when a single tool round is bigger than the model's window. One Claude Code task held 1,176 calls in one round (1.25M tokens); summarizing could not split the round from its calls, so `/compact` answered "Nothing to compact" and every turn overflowed. The oldest tool results now give way to a one-line note ("result omitted to fit the context window"), each call keeps its result, and the newest results and every user message stay as they were — 1.25M tokens down to about 96K for that task.
- A compaction that made room without a summary is no longer discarded for having no summary.

## 0.6.29 — 2026-10-06

Claude Code tasks that compact before they overflow, and long histories you can scroll back through.

- Claude Code reports each model's window on every reply; Xerxes now uses it, so a plain alias such as `opus` compacts ahead of time instead of running into "Prompt is too long". A size you set yourself still wins.
- When the conversation does not fit and automatic compaction cannot make room, the stop message says why instead of dropping the reason.
- A failed run notice names its failure ("Monitor reaction stopped before completion: …") instead of its id.
- "Show earlier conversation" keeps loading until something new appears: a stretch of tool calls folds into one line, so a page of them used to change nothing on screen. A task that fits the view loads earlier history when it opens, so there is something to scroll back to.
- The folded tool line counts every call (Claude Code reuses call ids) and says when one file was read over and over ("Read 1 file 1,091 times").
- Scrolling up while a reply streams stops following the newest text at once.
- VS Code: the view's files carry a build stamp, so an updated extension never shows a cached old view.

## 0.6.28 — 2026-10-06

- VS Code: the to-dos and queue bar sits a step above the message box instead of touching it, with the box's corners and its text in line with the message.
- VS Code: Settings has Agent intelligence (the light, balanced and smart tiers) between Models & Providers and Permissions.

## 0.6.27 — 2026-10-06

- The agents card's batch header spins while its agents run, instead of showing a still icon above spinning rows. It stops with the rest when reduced motion is on.

## 0.6.26 — 2026-10-05

Xerxes in VS Code: a chat in the sidebar, on the Visual Studio Marketplace.

- The **Xerxes Agents** extension (`erfanzar.xerxes-agents`) is a chat view like Claude Code's or Codex's: the task title is a searchable history, the message box holds the add-files, approval, plan, model and context controls, and the colours and fonts follow your VS Code theme.
- Files open in the editor, a file's changes in VS Code's diff, and Git in Source Control; **+** adds files through VS Code's file picker. New task, history and settings are the view's title-bar buttons; activity and usage are in its menu. Settings is one page: providers, models and approvals.
- The Bun runtime ships inside the extension, one package per platform (macOS, Linux and Windows), and under Remote-SSH it runs on the remote machine.
- If another copy of the extension is installed (an earlier build under a different ID), its buttons no longer just appear twice: the extension names it and offers to uninstall it.
- `bun run publish:vscode` builds every platform, publishes to the Marketplace and attaches the packages to the GitHub release.

## 0.6.25 — 2026-10-03

SSH tasks that keep running through your Mac when the link drops, and task switching without reloads.

- A task running through the Mac's provider is re-linked automatically after a dropped SSH link or an app restart, even mid-turn, and the desktop keeps retrying until it is back. The running turn waits (about 40 s of retries) and continues instead of failing. The server still discards a dropped binding; only an explicit re-link of the same provider and model restores it, and a revoke still stops at once.
- The badge shows "Reconnecting to this Mac…" while it restores the link instead of "access ended", and names the provider.
- OpenRouter: thinking off sends an off effort only when the model reports one. Models that report nothing got `effort: none` and failed with 400 on routes taking only low/high/max.
- Switching between tasks of an SSH workspace while one runs brings forward the view already showing the task instead of opening a new view, a new SSH connection and a full reload each time.

## 0.6.24 — 2026-10-03

SSH providers that work as expected, Claude Code installed automatically, and a cleaner Providers list.

- On an SSH workspace, ChatGPT/Codex and Claude Code are marked "runs on this Mac"; choosing one moves the task onto the Mac's sign-in instead of the host's empty copy, and renewals keep that choice. The list shows the provider a task really uses ("via this Mac") and the host's own default separately.
- A provider added or changed on the Mac reaches a connected SSH host within a second, and opening a task re-syncs as a backstop.
- Choosing Claude Code installs it through Bun when it is missing; a copy Xerxes installed is updated at most daily (Anthropic's own installer updates itself). Set XERXES_AUTO_INSTALL_CLAUDE_CODE=0 to turn this off.
- The Providers list is a set of cards: name and status, model, one Use action, and quiet edit/delete tools.
- Settings and other dialogs are solid instead of see-through.
- The welcome ring keeps moving instead of stopping after nine seconds, and a frame dropped while the window was hidden can no longer stop it for good.

## 0.6.23 — 2026-10-03

Your providers work on SSH workspaces automatically, and you can always see when a task runs through your Mac.

- Keyed providers: a host that never chose a provider now starts on the Mac's active one (or the first copied one that works) instead of an empty model list. A key rotated on the Mac replaces the copy on the host; the host's own model and limits stay.
- ChatGPT/Codex and Claude Code: their logins are not copied (a copy would sign one machine out, or live in the Mac's keychain). New SSH tasks automatically send their model requests through the Mac's own sign-in, renewed before the eight-hour limit and resumed after a reconnect. Tasks already run on the host's providers are left alone. Claude Code can now use this route, running on the Mac as a model only with its tools off.
- A composer badge shows "via this Mac" while a task runs through your Mac, pulses while a request is in flight, and turns into "Mac access ended" when the connection is gone. Hover for the provider, model, request count and last error; click to inspect or stop it.

## 0.6.22 — 2026-10-03

Check for Updates works again.

- Check for Updates… and the automatic update prompt never reached the window: the check ran in the background and its answer was dropped, so clicking it showed nothing and a new version was never offered. Both now show, and a desktop build without the update channel says so instead of staying silent.
- Installed 0.6.21 or earlier? Install this version once from the release page; later updates are offered in the app.

## 0.6.21 — 2026-10-03

Remote workspaces set up on servers without curl.

- Remote setup installs Bun itself instead of running Bun's curl-only installer. It picks the official build for the server (Linux or macOS, x64 or ARM, musl, CPUs without AVX2, Rosetta), downloads it with curl or wget, checks it against Bun's published SHA-256 sums, and unpacks it with unzip or bsdtar.
- When something essential is missing, the connection error names the one tool to install.

## 0.6.20 — 2026-10-03

Remote workspaces on a non-standard SSH port.

- The remote workspace form has a Port field, and an SSH host may be written as `user@host:port`; `/machine add` in the terminal accepts the same. Folder browsing, setup, the tunnel and runtime updates all use the port.
- Connections still use key-based login and a known host key. To set up a password-only server once, run `ssh-copy-id -p <port> user@host`.

## 0.6.19 — 2026-10-02

Reliability release, round 2: 55 more bugs, each confirmed by an independent check and fixed with a test that fails without the fix, including two regressions from 0.6.18.

- Messages and sessions: a message sent between goal rounds is no longer dropped. Renaming, saving, compressing or undoing another chat no longer switches this window to it. Unsaved edits in the Files editor are no longer lost when the panel closes or changes. Session cost now counts cached tokens.
- Approvals and questions: a hidden approval card can be brought back instead of leaving the turn waiting forever. Approvals from goals resumed after an update, monitor reactions and schedules can be answered, and an answer clears the card in every window.
- Channels: Telegram turns can ask and be answered (/approve, /deny, /answer, or a plain reply; in a group, address the bot, e.g. /approve@YourBot). /stop reaches a running turn, long replies are sent in full, context survives restarts, and group commands addressed to the bot work.
- Startup and extensions: a broken managed plugin no longer stops the runtime from starting. Reloading skills or plugins no longer hides a running turn, skill commands run in the session's workspace, and block-scalar skill descriptions parse.
- Schedules: one running job no longer blocks the others, cancelling a one-shot run no longer reschedules it, and follow-ups no longer time out while queued behind your own turn.
- Terminal UI: a prompt sent while switching sessions goes to the right one, a pending approval shows when you switch to its tab, and /undo removes the undone exchange.
- Providers: Codex turns retry a dropped connection and handle its final event. A stored Anthropic subscription token is no longer sent to other Anthropic-compatible providers. Unknown finish reasons are no longer fatal, and parallel agents no longer race on OAuth refreshes.
- Context: compaction and limits follow the profile the session runs on, images are no longer counted as huge text, and a token-capped goal is not blocked by one transient failure.
- This week's features: finished workflow agents are not delivered twice, the memory guard no longer over-counts shared memory on Linux or reports a reused process id as stopped, and structured agent results keep every field.

## 0.6.18 — 2026-10-02

Reliability release: 46 bugs found by a whole-codebase review, each confirmed by an independent check, fixed with a test that fails without the fix.

- Messages and sessions: a turn that was running when the app quit, crashed or updated is recovered on reopen, including a brand-new session's first turn. /undo and /retry after a compaction no longer wipe the session. Opening a chat mid-turn, or reconnecting, no longer hides, duplicates or reorders messages. A save conflict no longer leaves a window on a hidden empty session. A damaged pre-compaction archive no longer makes a session unopenable.
- Goals: after a runtime update every armed goal continues, not only the session you reopen. A steer sent while a turn is saving is applied. A forced restart no longer drops the goal.
- Agents and workflows: agents are not lost or merged after a restart, and retries keep their settings and count correctly. Reconnecting an MCP server or changing LSP settings no longer kills running agents. Workflows no longer fail at a hidden 100-agent cap, and an agent's background commands end with it.
- Claude Code: tool calls are no longer cut at a literal "</function>" inside their arguments, a reply is no longer swallowed after markup mentioned in prose, and a call cut off by the output limit is not run half-written.
- App and SSH: windows can be closed, reloading a window keeps its own session, and windows restore on Windows and Linux. A dead remote runtime is restarted instead of reconnecting forever, and connecting no longer overwrites the server's own provider settings.
- Tools: large command output keeps its beginning and its final error lines. An agent's own file append no longer blocks its next edit, and agents no longer evict each other's read-before-edit records.

## 0.6.17 — 2026-10-02

- Memory guard: an agent command, with everything it starts, is stopped once it uses more than a limit (default: half the computer's memory; GPU memory counts on macOS). The agent is told why and how to run less at once. Set the limit, or turn it off, in Settings → General.
- Desktop: opening a chat shows your latest message. A chat opened on its newest 100 actions, and a goal or workflow session can run hundreds of tool actions after the last thing you said, so the conversation looked lost. It now pages back until your last message is on screen; the button reads "Show earlier conversation".

## 0.6.16 — 2026-10-01

- Claude Code: a reply rejected with "The model's tool call could not be parsed" is retried (twice at most) with a note telling the model it used the native <invoke> syntax and must call tools as <function=NAME>{json}</function>. Sonnet slips into the native form in long, tool-heavy agent runs, and each slip used to fail the whole agent. Text already shown is not repeated.

## 0.6.15 — 2026-09-30

- Agents: a reply with no text and no readable tool call no longer ends an agent as "completed without a final response". After tool work, the agent is asked again (twice at most) to call a tool in the required form or give its answer. Workflow agents on Claude Code hit this after large tool results.
- Settings → Agent intelligence: point the light, balanced and smart tiers at any provider, any model (from the provider's list or typed in) and any reasoning effort that model supports, and choose which tier new agents use.
- Updates: the app checks for a new version every five minutes. "Check for Updates…" now says when you are up to date instead of showing nothing, and "Not now" keeps a version from reappearing until you ask or relaunch.

## 0.6.14 — 2026-09-30

- Compaction: a long session's summary no longer grows until compaction cannot make room. Each pass used to keep the previous summary word for word and add the new one after it; a goal session reached a 25K-token summary against a 2K budget and its turns failed with "Automatic compaction could not make room". Once the carried summary and the new one pass twice the budget, they are folded into one summary. The full history stays in the pre-compaction archive.

## 0.6.13 — 2026-09-30

- Tools: a large file can be read and edited. ReadFile refused any file over 256 KB, even a 50-line window of it, and since an edit needs a read first, such a file could not be edited at all. Windows of any file up to 32 MB now read normally; only a whole-file read (limit=-1) keeps the 256 KB limit.
- Agents ask with clickable choices. AskUserQuestionTool takes `options` (recommended first); the desktop shows them as buttons with number keys plus a field for your own answer, and the question text renders as Markdown. The daemon used to drop options from the plain question form.
- Desktop: a question's options are single-choice; a second click moves the choice instead of sending both.

## 0.6.12 — 2026-09-30

- Desktop: a new task no longer shows the task it replaced. After switching, a window could keep asking the runtime about the previous task, so the new one showed that task's goal, background jobs and model time. It now always uses the task the runtime says it opened, ignores answers meant for the task it left, and a finished turn's totals only reach the window still showing that task.
- Desktop: the side panel's tabs no longer overlap when it is narrow. They keep their width and scroll, the open tab stays in view, and only an edge with hidden tabs fades.

## 0.6.11 — 2026-09-30

- Desktop: the side panel's edge runs unbroken around its rounded corners, and the resize divider beside it no longer draws a second line through the gaps above and below the panel (it still highlights on hover).

## 0.6.10 — 2026-09-29

- Desktop: a workflow agent's full output shows in the inspector. Agents in a workflow used to be saved with only their first 2,000 characters; they now keep up to 16,000 like any agent, and only the oldest agents of a very large run are trimmed. Trimmed output is labelled as such.
- Desktop: an agent's structured result (a workflow schema answer) shows as labelled fields instead of one line of JSON, and its output is pretty-printed.
- Desktop: workflow phase headings line up with their agents, and the inspector's "All activity" link points back.

## 0.6.9 — 2026-09-29

- Desktop: the Files panel lists dot folders and files (.github, .vscode, .env.example…); only .git and .DS_Store stay hidden.
- Desktop: a rendered Markdown file fills the preview and scrolls as one page; it was capped at 380px, which cut long files off at the first big code block.
- Desktop: with a transparent background, the side panel's corners no longer show a sliver of bare desktop beside the conversation.

## 0.6.8 — 2026-09-29

- Desktop: "Restart workspace runtime" works while the runtime is busy. The popover names what is still running and offers Restart now; an active goal picks up again on the new runtime (older runtimes are shut down and the goal is resumed).
- Desktop: a clearer icon for the Agents chip and card, and the Tasks / Agents switch is a proper segmented control instead of a bent underline.

## 0.6.7 — 2026-09-29

- Desktop: the app updates itself from GitHub releases, with your permission.
  It checks at startup and every six hours (and from Xerxes Agents › Check for
  Updates…), shows the new version and its release notes, and asks: Install and
  restart, Not now, or Skip this version. On yes it downloads the installer,
  checks it against GitHub's published SHA-256, verifies the app's signature,
  and swaps it in when the app quits, then relaunches. Development builds and
  unwritable Applications folders get the release page instead.
- Desktop: reopening the app during a long turn no longer hides what that turn
  already did. The conversation only catches up when a turn ends, so a goal
  round that ran for hours looked like it had lost its last several steps
  until it finished; the runtime now serves the running turn's messages to a
  client that attaches mid-turn.

- Agents: an **Agents** chip beside Plan in the composer (and `/delegate`):
  **Off** works alone, **Auto** (default) fans out when work splits and asks
  once before a run of more than about ten agents whether to be budget-minded or
  thorough, **Eager** delegates whenever work can split, with a verify step.
- Agents: the model gets a playbook for choosing between working alone, one
  agent, a batch and a Workflow — reviews, sweeps, migrations, research,
  claim-checking, competing debugging theories and benchmark matrices each have
  a shape, and the costs of fanning out are spelled out.
- Workflows report their cost at published prices; the run's card shows it.
- Git: **Review** runs large changes as a find-then-verify workflow (a reviewer
  per slice, a skeptic per finding) and reports how many findings survived.
  **Create PR** runs the project's checks and writes a sectioned description:
  summary, changes by area, behaviour changes, how it was tested, risks and
  rollback, follow-ups.

## 0.6.6 — 2026-09-29

- Desktop: agents and workflows stay visible in the conversation. They used to
  fold away with the tool calls around them; now only the tool calls fold.
- Desktop: the agents card follows Claude Code's shape — "Running 4 agents…"
  over a tree, one branch per agent with its model, tool uses, tokens and
  time, and under it what it is doing now or "Done (15 tool uses · 23.4K
  tokens · 1m 13s)"; "+N more tool uses" opens its latest calls in place.
  Workflows keep their phases, progress bar and dot grid for large runs.
- Desktop: workflow runs get their own card in the Activity panel, by phase;
  the Agents list keeps the agents started outside a workflow.
- Agents' live activity reads as a command line ("Running env
  JAX_PLATFORMS=cpu,mps python bench.py") instead of "cmd=env, args=…".

## 0.6.5 — 2026-09-29

- Runtime: an update no longer waits forever behind an armed goal. It holds
  the next goal round, installs when the current one ends, and the fresh
  runtime re-arms the same goal and carries on. 0.6.3 had stopped restarts
  while a goal was armed, so a long-running goal kept the old runtime (and
  new tools such as Workflow) indefinitely. The sidebar says "Update after
  this goal round" meanwhile.
- Desktop: a steer leaves "Queued messages" as soon as the model receives it.
  It used to stay there until the turn ended, which in a goal round can be
  hours after the model had already answered it.

## 0.6.4 — 2026-09-29

- Agents: a `Workflow` tool. The model writes a short script with `agent()`,
  `parallel()`, `pipeline()` and `phase()` to fan work out to as many
  subagents as the job needs (reviews, audits, migrations, find-then-verify),
  picking a model per agent and getting structured results back. The script
  runs in its own process without provider keys; stopping the task stops the
  script and every agent it started.
- Agents: the model delegates on its own initiative and matches the model to
  the job, using a new prompt section that lists the models each configured
  provider reports (with published prices) so mechanical work can go to
  cheaper, faster models.
- Desktop: the in-chat agents card is rebuilt for one agent to thousands. A
  workflow shows its name, phases, each agent's model, what it is doing right
  now and its clock; large phases draw as a dot grid with rows only for agents
  still running or failed. Spawn batches are no longer cut at 24 in the card.
- TUI: a Workflow row shows its run's phases, tallies and live or failed agents.
- Runtime: compaction now fires on the provider's real prompt size. The context
  estimate ran about 1.8x low on code-heavy Claude transcripts, so a 1M window the
  meter put near half full was rejected as full and auto-compaction never started.
  Each round's reported prompt tokens now calibrate the estimate, which is kept on
  the session for the pre-turn check and the context meter.
- Runtime: a long goal round can overflow and compact more than once. The
  compact-and-retry was spent once per turn, so the second overflow in an
  hours-long round blocked the goal.
- Runtime: Stop takes effect at once. A tool that ignored the stop (a
  `check_command` waiting up to a minute on a background build) held the turn
  open; interruptible tools are no longer awaited after a stop, and
  `check_command`'s wait ends on it.
- Desktop: Escape needs a second press within two seconds to stop a running task.
  One stray press (reaching for a screenshot shortcut) cancelled the task and every
  agent it had spawned; it now shows "Press esc again to stop". The Stop button is
  still one click.
- Desktop: connecting to an SSH workspace copies this Mac's key-based provider
  profiles (OpenRouter, Z.ai, Kimi and the like) to the host, so they work there
  without re-entering keys. The host's own profiles and selection are kept;
  Claude Code and ChatGPT sign-ins stay per machine.
- Providers: the built-in Claude Code profile is now `claude-code` instead of
  `cc`. Saved profiles and selections migrate on load, and `cc` still resolves
  for older sessions, `/provider cc` and `cc/<model>` ids.

## 0.6.3 — 2026-09-29

- Desktop: much lower energy use. Every workspace page a window has opened kept
  listing all saved and live sessions every five seconds, even when hidden; with
  several workspaces open that polling dominated the runtime's CPU. Hidden or covered
  pages now skip it and catch up when shown, an unfocused window refreshes its lists
  every thirty seconds, and background output, agent and run pollers pause while hidden.
- Runtime: an idle-only runtime update no longer restarts while a goal is armed; a
  restart silently disarmed the goal and stopped autonomous work at its next round.

## 0.6.2 — 2026-09-28

- Usage: OpenRouter keys show credit left, spend today, this week, this month and
  all time. A new "By model" section totals the last 30 days per model from the
  runtime's own record of every task and subagent round, priced at published prices
  ("price unknown" when none exists).
- Providers: the model is optional when adding a provider. Left blank, the runtime
  asks the provider which models the key can use and starts on the first; an edit
  keeps the saved model.
- Desktop: command and skill suggestions, menus and pickers stay opaque over a
  transparent or image background.
- Desktop: the rail's Touched card expands in place, and clicking a file opens
  Session edits on that file's diff.

## 0.6.1 — 2026-09-28

- Desktop: fresh macOS installs open with the transparent window, system blur, an
  opaque chat column and see-through side panels; other platforms keep solid surfaces.
- README: a real screenshot of the desktop app replaces a design render that never
  loaded on GitHub.

## 0.6.0 — 2026-09-28

Desktop application
- Claude-style layout: list sidebar, one reading column, a composer with the
  repository bar (branch, diff, Review, Create PR), model, effort and a context ring.
- Dotted agent status orbs for tools, reasoning, writing and waiting, on one shared
  low-rate timer that stops when the window is unfocused, covered or settled.
- Files open and edit in place with rendered Markdown, save conflicts and revert;
  read, write and edit tool rows show their content inline.
- Themes and backgrounds (Xerxes default, Claude colours), working blur, a transparent
  mode, copy buttons on replies, chat deletion and Claude-sized type.
- Lower energy use: covered workspace views are throttled, stream updates are
  batched, and looping animations pause while unfocused.
- Running chats reattach after an update or reconnect instead of opening a new task.

Runtime
- Leased clients' turns and armed goals keep running while the desktop is away
  (a locked laptop, a dropped SSH link) and reattach on return.
- Goal rounds lost to transient provider failures retry after 1, 3 and 10 minutes
  instead of blocking the goal.
- Session recovery never splices journal entries from older compaction generations
  onto a compacted session; a live session that loses a save conflict is preserved
  under `sessions/divergent/` before the saved copy is reloaded.
- Claude Code: streams stay alive while a large tool call is written, tag-form tool
  calls parse, the system prompt is passed by file and the CLI is kept current.
- Goal evidence can cite the latest successful tool call; reasoning effort reports
  "default" rather than a false "off"; outdated SSH runtimes update when idle.

Terminal UI
- Welcome screen, transparent-mode panels, accurate status and turn durations;
  diff panel seeks land after layout.

## 0.5.0 — 2026-09-20

- Add a selectable transparent TUI background alongside Chrome styling.
- Reuse supported local providers for SSH tasks with one scoped setup approval;
  credentials and refresh stay local, with explicit expiry and revocation.
- Keep approved local models discoverable and usable by delegated agents.
- Continue finished spawned agents through SendMessageTool with stable identity,
  saved history, and ordered follow-up input.
- Persist model, mode, effort and permission choices atomically; preserve prior
  settings on save failures and restore task policy after daemon restart.
- Improve task navigation, history, output, cancellation and recovery behavior.
- Correct empty optional provider fields in model inventory.
- Avoid Linux watcher failures caused by unrelated Unix sockets by polling scoped
  file metadata; report unsupported Windows PTY runtimes without orphaned panels.


- Reconcile restored active-agent statuses with workers owned by this daemon.
- Show provider response waits explicitly and avoid generic working verbs during silence.
- Deliver compaction and retry progress before the first model output, and retain terminal errors from silent attempts.

- Show the compaction spinner for automatic and mid-turn summaries, preserve
  lifecycle statuses through the TUI adapter, and clear the indicator on failure.

- Bound compaction input chunks independently of the model context window and
  retry timeouts with smaller chunks, including mid-turn compaction.
- Identify automatic compaction failures explicitly while retaining the original conversation.

## 0.4.5 — 2026-09-09

- Explicit model selections now take precedence over intelligence tier hints when
  spawning agents; working-tree isolation accepts an explicit HEAD reference.

- Fixed saving custom agents whose names are derived from their filenames,
  including agents created by project setup.
- The custom-agent editor now lists and edits nested Markdown specialists and
  preserves valid declared names that differ from their filenames.
- Agent drafts without a discovery description now fail validation instead of
  saving successfully and silently disappearing from the runtime catalog.
- Background task creation now exposes, validates, and forwards worktree settings.
- Cancelling SSH setup returns promptly even if SSH ignores termination.
- A tunnel drop during renderer handoff no longer launches a TUI on a dead tunnel.

## 0.4.4 — 2026-09-09

- Added bounded SSH reconnects, manual retry, connection progress, and recovery of
  the last recorded remote session.
- Redesigned the custom-agent browser and added model-generated, editable drafts.
- Enabled automatic workspace snapshots before model turns and shell commands.
- Fixed automatic compaction checks inside long-running tool loops, bounded
  oversized compaction requests, and preserved compaction state across saves.
- Kept status RPCs responsive during compaction and extended the manual
  compaction timeout.

- Interrupted commands retain partial stdout/stderr and report cancellation with
  its available reason, instead of mislabeling valid arguments as a validation error.
- Added `/activity`: a unified panel for session shells/watches and workspace
  schedules, with elapsed times, output/details, stop/pause controls and brief
  completion/failure badges. Lifecycle events replace composer status polling.
- Bound provider profiles to conversations and inherited subagents. Model picker
  selection applies the provider and model together; GPT models cannot be routed
  to Kimi subscriptions. HTML provider error pages now show concise messages.

- Added live blue shell/watcher counts beside the composer settings, including
  while idle. Click a count to inspect background work; completed work clears.

- Remote workspaces now render locally over a private SSH-forwarded daemon
  socket. Tunnel failures return to the original local workspace.
- F7 collects changes on the daemon host and includes navigable untracked-file
  previews, with empty-file and binary-file handling.

- Fixed inline swarm status lists splitting comma-containing titles into phantom
  queued agents; saved transcripts reconcile titles with actual agent records.
- Added `TaskOutputTool` character pagination (`offset`, optional `limit`, up to
  8,000 characters). Follow the returned next offset to read complete saved
  reports without rerunning agents. For new work on a completed agent, use
  `AgentTool` with `resume` and `prompt`; `SendMessageTool` targets running agents.

## 0.4.3 — 2026-09-07

- Redesigned operational TUI dialogs, forms, empty states, and capability guides.
- Added SSH config host discovery, remote project folder browsing, and automatic
  installation and updates of managed remote Xerxes builds.
- Added Plugin Creator mode with native TypeScript plugin examples and Bun tests.
- Exposed custom-agent editing and skill/plugin controls, with project specialist
  discovery and Claude-style agent workflows.
- Improved goal configuration, provider/model discovery, reasoning visibility,
  session restoration, and attachment handling.
- Fixed bang-command output and exit codes; command results now receive a model
  follow-up and remain available when restoring the conversation.

## 0.4.0 — 2026-09-03

- Published the native distribution under the scoped npm identity
  [`@xsimurgh/xerxes-agents`](https://www.npmjs.com/package/@xsimurgh/xerxes-agents).
- Added Claude Code compatibility for project instructions, `/init`, shell mode, memory capture,
  hooks, project MCP, headless output formats, model fallback, and injected skills.
- Added timeout-driven command backgrounding and persistent PTY sessions with live terminal
  inspection and control from the TUI.
- Added a live spawned-agent roster with animated activity cubes, elapsed time, tool summaries,
  completion state, and restoration after session reattachment.
- Improved Agent View attachment, session-tab switching, reasoning-model pinning, provider-profile
  capability resolution, and full tool-argument summaries after resume.
- Corrected cumulative LLM/TTFT accounting and rejected fabricated Codex throughput samples from
  tool-only terminal events.
- Added provider-stream retries, fresh instruction-file overlays, and complete daemon-capability
  mapping across the TUI and desktop renderer.

## 0.3.0 — Native Bun/TypeScript migration

- Added the native runtime, CLI, daemon, session, streaming, API, and OpenTUI client paths under the
  Bun workspace.
- Replaced the handwritten RST/Sphinx documentation branch with Markdown sources and generated
  TypeScript API pages.
- Added native examples, installer, CI/release paths, and focused contract/parity coverage.
- Completed the Bun-only cutover: retired Python source, tests, tooling, playground paths,
  virtual-environment artifacts, caches, and distributions are removed; the frozen install,
  typecheck, full native test suites, build, release-package check, and Bun-only guard pass.

Project identity has shifted over time: originally `eLLM`, renamed to `AgentX`, then `Calute`, and now **`xerxes-agent`** with the native `xerxes` command.

---

## 0.2.6 — 2026-06-23

- Added the bundled `bug-bounty-hunter` multi-iteration swarm repair skill.
- Bumped the package/runtime version to 0.2.6.

## 0.2.5 — 2026-06-22

- Added the bundled `eternal-army` swarm-orchestration skill.
- Fixed skill discovery refresh for source checkouts and resilient frontmatter parsing.
- Tightened `/steer` delivery so queued guidance reaches the next provider request or is saved for the next turn.
- Bumped the package/runtime version to 0.2.5.

## 0.2.4 — 2026-06-19

- Replaced static/clamped tool-result handling with project-memory spillover plus agent-written summaries.
- Added context-window provisioning before provider calls and after tool batches.
- Tightened DeepScan so subagents save full findings to project memory and return only compact pointers.
- Bumped the package/runtime version to 0.2.4.

## 0.2.3 — 2026-06-19

- Fixed bare `/provider` with zero saved profiles so it opens the add-profile picker instead of returning dead-end text.
- Bumped the package/runtime version to 0.2.3.

## 0.2.2 — 2026-06-19

- Added the managed `~/.xerxes-venv` installer flow and terminal alias setup.
- Updated `xerxes update --force` to target the managed venv when present.
- Bumped the package/runtime version to 0.2.2.

## 0.2.1 — 2026-06-19

- Bumped the package/runtime version to 0.2.1.

## 0.2.0 — 2026-04-16

### Project rename and restructure

- **Renamed Python module** `xerxes_agent` → `xerxes` ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Distribution package stays `xerxes-agent` for PyPI compatibility.
- All build config (pyproject, hatch hook, pytest.ini, CI, Dockerfile, pre-commit) repointed to the new module.
- Orphan `src/python/xerxes_agent/` shim directory removed.
- Metric `xerxes_agent_switches_total` renamed to `xerxes_switches_total`.

### Security hardening

- **Fixed sandbox pickle-escape vulnerability.** Child-to-parent IPC in both `docker_backend` and `subprocess_backend` now uses JSON instead of pickle. The old symmetric pickle design was an arbitrary-code-execution vector via `__reduce__` (a malicious sandboxed tool could craft a payload that would execute `os.system(...)` in the parent process upon deserialization).
- **Fixed `math_tools.Calculator` eval-sandbox bypass.** Replaced `eval(expr, {"__builtins__": {}})` with an AST-whitelist evaluator that accepts only numeric literals, the arithmetic operators `+ - * / // % **`, unary `+/-`, the functions `sin cos tan log sqrt abs pow exp`, and the constants `pi` and `e`. The previous pattern was bypassable via `().__class__.__bases__[0].__subclasses__()`.
- Both fixes verified with negative proof-of-concept tests.

### Feature work

- CLI feature buildout and scrollable scrollback ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Agent orchestration, hand-off, and planning features ([a816780](https://github.com/erfanzar/Xerxes/commit/a816780)).
- Markdown rendering and CLI enhancements ([edfcfa9](https://github.com/erfanzar/Xerxes/commit/edfcfa9)).
- `Cmd-Enter` submit in the TUI ([7e88262](https://github.com/erfanzar/Xerxes/commit/7e88262)).
- Hardened tool routing and TUI behavior ([20e143b](https://github.com/erfanzar/Xerxes/commit/20e143b)).

### Removed

- **Chainlit UI package.** The `xerxes/ui/` subpackage wrapping Chainlit was removed; UI now belongs outside the core framework. Removed `chainlit` from optional dependencies, eliminated the `ui`/`research`/`full` extras' Chainlit dependencies, and stripped `create_ui` methods from `Xerxes`, `Cortex`, `CortexAgent`, `CortexTask`, `TaskCreator`, and `DynamicCortex`.
- Outdated `docs/api_docs/calute.rst` and other stale Sphinx references.

### Test fixes

- 15 previously-failing tests restored to passing. All referenced the old `xerxes_agent.*` module via `mock.patch` / `monkeypatch.setattr` strings that static-analysis couldn't catch. Rewritten to use `xerxes.*`.
- pytest coverage target corrected from nonexistent `src/python/xerxes_agent` to the real `src/python/xerxes`.

---

## 0.2.0 prerelease — early 2026

- **Modular runtime and TUI** ([88e9ac8](https://github.com/erfanzar/Xerxes/commit/88e9ac8)) — released as "Calute 0.2.0".
- Enhanced LLM functionality: reasoning-token extraction, sampling parameter overrides, compat shims for OpenAI-wire-compatible providers ([dcdbfdf](https://github.com/erfanzar/Xerxes/commit/dcdbfdf)).
- Comprehensive docstrings added across the codebase ([d2067be](https://github.com/erfanzar/Xerxes/commit/d2067be)).

---

## 0.1.x series

### 0.1.2 ([bda33b9](https://github.com/erfanzar/Xerxes/commit/bda33b9))

- Modified `create_ui` parameter signature (since removed).

### 0.1.1 ([659cf05](https://github.com/erfanzar/Xerxes/commit/659cf05))

- Added `create_ui` method across `Xerxes` / `Cortex` / `CortexAgent` / `CortexTask` (since removed in 0.2.0).
- Gradio-based UI components added ([6194f5f](https://github.com/erfanzar/Xerxes/commit/6194f5f)) — later replaced by Chainlit, now removed.

### 0.1.0 core

- **MCP (Model Context Protocol) integration** ([474d7f7](https://github.com/erfanzar/Xerxes/commit/474d7f7), [8d1afaf](https://github.com/erfanzar/Xerxes/commit/8d1afaf)) — stdio + SSE + streamable-HTTP transports; `MCPClient`, `MCPManager`, `MCPTool`, `MCPResource` public API.

---

## 0.0.x series (Calute era)

- **Architecture overhaul with new LLM abstraction layer** ([7779931](https://github.com/erfanzar/Xerxes/commit/7779931)) — introduced `BaseLLM` as the abstract provider interface.
- **Migrated from custom chat types to OpenAI proxy types** ([9b4fbf1](https://github.com/erfanzar/Xerxes/commit/9b4fbf1)); improved API server streaming.
- **API server (FastAPI)** added in 0.0.19 ([0535858](https://github.com/erfanzar/Xerxes/commit/0535858)).
- **EnhancedMemoryStore** and general helper assistant ([cf05bec](https://github.com/erfanzar/Xerxes/commit/cf05bec)).
- Comprehensive tools suite: web search, code execution, browser tools, data tools, math tools, etc. ([45edc91](https://github.com/erfanzar/Xerxes/commit/45edc91), [061a762](https://github.com/erfanzar/Xerxes/commit/061a762)).
- Advanced delegation engine + memory management system ([02255dc](https://github.com/erfanzar/Xerxes/commit/02255dc)).
- `reinvoke_after_function` option for re-prompting LLM after tool execution ([d54b9ec](https://github.com/erfanzar/Xerxes/commit/d54b9ec)).
- Python 3.13 support ([9909837](https://github.com/erfanzar/Xerxes/commit/9909837)).
- XerxesConfig hierarchical config system; `/context`, `/cost`, `/compact` slash commands.

---

## Historical (pre-Xerxes: eLLM / AgentX era)

Deep history predating the current multi-agent framework shape. Highlights:

- eLLM → AgentX rename with project restructure ([8ecb339](https://github.com/erfanzar/Xerxes/commit/8ecb339)).
- Early Gradio GUI experiments (later deprecated).
- Initial Ollama backend support ([38654fc](https://github.com/erfanzar/Xerxes/commit/38654fc)).
- Original RAG work, LLM quantization support, WebSocket server experiments.

These commits are preserved for archaeological purposes but don't reflect current API or design.

---

## How to read this

Commit SHAs link to the GitHub commit if the repository is public. Dates are approximate (inferred from tag history and commit clustering, not author dates).

For a precise diff between any two points in history:

```bash
git log --oneline <from>..<to>
git diff --stat <from>..<to>
```

For a live changelog of the most recent work:

```bash
git log --oneline --no-merges -20
```
