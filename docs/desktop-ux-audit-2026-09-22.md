# Desktop app — UI/UX audit, 2026-09-22

Two passes, cross-checked against each other:

1. **Live**: the real `src/desktop/renderer` bundle was booted in a browser against a scripted fake preload (`tmp-files/ux-harness/`), and every panel, modal, rail, page and card was driven by hand. Findings below marked _(observed)_ were reproduced on screen.

2. **Source**: 12 parallel auditors read the renderer across one dimension each; every finding was then handed to an independent agent whose job was to refute it. 57 of ~72 survived.


Severity is the verifier's, not the finder's.

> **Status: fixed.** Everything below was worked through in the same session that found it.
> What was deliberately left, and why, is listed at the end of this note.
> `bun run check` passes; `bun test ./test` is 4173 pass / 0 fail.

**Not fixed, on purpose:**
- **`isPlanReview` keyword heuristic** (`store.ts:636`) — detecting a protocol concept by matching the English words "plan" and "approve" in display text is wrong, but fixing it properly needs a typed field on the wire, which is a daemon change.
- **Deleting a task** — absent end to end; there is no `session.delete` RPC to call.
- **Syntax highlighting** — code blocks now carry their language and a copy button, but actual highlighting needs a new dependency, which is a call for the repo owner.
- **Paste/drop into the composer, `@`-path completion, and rewind-and-retry** — the three opportunities at the end of this note. They are features, not defects.
- **Tab bodies still mount conditionally** (`App.tsx`), so switching tabs still re-collapses disclosures and re-pins the transcript. Jump-to-latest now makes the scroll part recoverable; persisting per-tab scroll and collapse state is a larger change.

---



---

# Observed live in the running renderer

These were reproduced on screen in the harness, not inferred from source.

### Bare `1`/`2`/`3` grant tool permissions whenever focus is not a text field
`xerxes/src/desktop/renderer/App.tsx:1844-1849`

The guard is `target.tagName === 'INPUT' || 'TEXTAREA'`. Verified by dispatching a real bubbling `keydown` from a focused `button.studio-nav` and from a focused `<select>`: in both cases the pending `rm -rf build/ && bun run build` approval was granted (`allow_once`) with no modifier, no confirmation and no undo. After clicking any chrome control, focus sits on a button — so a stray digit approves whatever is pending.

The mirror image is also true: the composer is auto-focused on session entry and after every insert, so in the ordinary type-then-approve flow the advertised `1/2/3` silently do nothing and just append a digit to the draft. The same hotkey is both unreliable and unsafe depending on focus state the user cannot see.

**Fix** — move focus into the approval card on arrival (also fixes the silent `role="alertdialog"`), and require a modifier or an explicitly focused card for the digits.

### The dictation error toast renders underneath the sidebar
`xerxes/src/desktop/renderer/atelier.css:129,215`

`.dictation-error` is `position:absolute; right:0; width:320px` anchored to the Dictate button at x≈316-377, so it extends left to x≈57 — into the sidebar's 0-240 band. `.side` has `backdrop-filter: blur(24px)`, which creates a stacking context that paints over the toast's `z-index:20`. Verified: `document.elementFromPoint()` at the toast's centre returns `nav.side__list`. On screen the user sees a white sliver reading "draft is" and cannot reach "Dismiss".

**Fix** — clamp the toast inside the chat column, or portal it to `document.body` above the chrome.

### A floating approval card gives the transcript a horizontal scrollbar
`xerxes/src/desktop/renderer/app.css:531`

`.approval` is `margin: 0 20px 10px` on a full-width child of `.stream` (`padding: 28px 32px`, `overflow-x: auto`). Measured: `scrollWidth 890` vs `clientWidth 850` — exactly the 40px of horizontal margin. A scrollbar appears under the card in both themes.

**Fix** — `max-width: calc(100% - 40px)` on `.approval`, or drop the horizontal margins.

### Plan-review options 3+ are mouse-only, and the descriptions are positional
`xerxes/src/desktop/renderer/App.tsx:1345,1386-1396`

`approveOption` is picked by regex `/approve|accept|start/i` over the **display text**, falling back to `options[0]`. The remaining options are rendered with `index === 0 ? <kbd>2</kbd> : null`, and the key handler only binds `'1'` and `'2'`. Observed with `["Approve and run","Revise the plan","Cancel"]`: Cancel had no key badge. Worse, `opt__desc` hard-codes "— tell the agent what to revise" onto whichever option happens to be first — so a daemon sending `["Approve","Cancel","Revise"]` labels **Cancel** as the revise action, and if no option matches the regex, `options[0]` becomes the styled-primary "approve" bound to `1`.

**Fix** — drive the primary action and the per-option descriptions off a typed field on the wire, not off English substrings; number every option.

### `isPlanReview` detects a protocol concept with an English keyword match
`xerxes/src/desktop/renderer/store.ts:636-639`

`haystack.includes('plan') && (includes('approve') || includes('accept'))` over the question's display text. A plan review phrased "Ready to proceed with these steps?" is not recognised; an ordinary question mentioning both words is misclassified and captures the agent's prose as the session plan. Breaks entirely for non-English phrasing.

### The first-run checklist appears with nothing left to do
`xerxes/src/desktop/renderer/Setup.tsx:14-20`

Shown whenever `localStorage[SETUP_KEY] !== 'done'`, independent of `setupReadiness()`. It is also gated on `!snap.noWorkspace`, so the genuine first run (no folder yet) never sees it — you pick a folder, and *then* get a three-step "set up before your first task" modal whose three steps are already ✓. Observed.

**Fix** — skip it when `state.ready` is true on first evaluation, or show it before the workspace gate, not after.

### Plan tab empty state points at a control that does not exist
`xerxes/src/desktop/renderer/Workspaces.tsx:119`

"Toggle the ⏸ plan chip in the composer" — but the header plan chip is `display:none` (atelier.css:439) and the composer's control is labelled **"Work directly"**. Observed.

### Scheduled jobs: raw cron, prompt-as-title, dangling separator
`xerxes/src/desktop/renderer/DesktopPanels.tsx:744-749`

The heading is `job.prompt` (a whole instruction, not `job.name`); the schedule renders as the raw `0 2 * * *`; and `{schedule} · {timezone} · {execution_state}` leaves a trailing "·" whenever the state is empty. Observed. The same dangling separator appears in the Activity rail ("agent · running ·").

### Duplicate control, developer text, and a blank tab in Settings
Observed: Models & Providers shows two buttons that both fetch models ("fetch models" and "↻ Fetch models"); the MCP tab leads with its error before explaining what the panel is; the **Language servers** tab rendered completely empty (`LspPanel.tsx` has no render branch between "no view yet" and "error"); interface font size offers only `11 | 12 | 13` with no larger option; and there is no About/version, no shortcut reference and no settings search anywhere.

### No delete for a session, anywhere in the app
The sidebar context menu is Open / Rename… / Copy id / Export md (`App.tsx:526-540`). The omission is deliberate and commented (no wire capability backs it) — but end to end, a user accumulating hundreds of sessions has no way to remove one.

### macOS menu bar is missing platform basics
`xerxes/src/desktop/main.ts` — no Settings (⌘,) item in the app menu, no Help menu, and File offers only New Window / Open Workspace / Close. ⌘, *is* bound in the renderer (`App.tsx:1771`), so the menu simply does not advertise it.

---


# Source audit findings

## HIGH

### [approvals] Cancelling a turn while an approval or question card is up strands the card: nothing clears `approval`/`question` on cancel or `turn_end` (store.ts:1015-1019, 3379-3417), the cards have no dismiss control (App.tsx:1157-1162, 1315-1317), and the daemon has already force-rejected the request, so every remaining button returns ok:false and store.ts:1196-1198 prints 'approval refused'/'answer refused' while keeping the card — leaving the header badge stuck on 'needs input' (App.tsx:824) until the user switches sessions or a new request happens to overwrite it

**`xerxes/src/desktop/renderer/App.tsx:1157`**

*Impact* — User sees 'Bash — approval required', decides not to run it, presses Esc (or clicks Stop). The turn dies but the red pulsing approval card stays pinned to the transcript forever. Clicking Deny — the obvious way to get rid of it — prints 'approval refused' in the feed and the card is still there. The header badge stays stuck on 'needs input', which also suppresses the 'Working'/'idle' state for every later turn. The only escape is navigating to another session (store.ts:2601 patches `approval: null`) and back.

*Fix* — Two halves. (a) Renderer: clear `approval`/`question` in `cancel()` and in the `turn_end` handler, and add a ghost 'Dismiss' button to `ApprovalCard` and `QuestionCard` that clears the local card without sending a decision. (b) Daemon: have `endTurn`/`finishPermission`/`finishQuestion` emit `approval_response` / `question_response` with the forced outcome so every attached surface clears in sync — today only the explicit `permissionResponse` RPC path emits it (server.ts:8900-8903).

<details><summary>Evidence & verification</summary>

```
`ApprovalCard` has exactly three controls and no close/dismiss:
```
1157:      <div className="approval__row">
1158:        <button ... onClick={() => store.approve(approval.id, 'allow_once')}>Allow once <kbd>1</kbd></button>
1159:        <button ... onClick={() => store.approve(approval.id, 'allow_session')}>This session <kbd>2</kbd></button>
1160:        <button ... onClick={() => store.approve(approval.id, 'deny')}>Deny <kbd>3</kbd></button>
```
Escape while a card is up cancels the turn (App.tsx:1837-1840 `if (snap.turnActive) { event.preventDefault(); store.cancel() }`), and `store.cancel()` (store.ts:1015-1019) only calls `turn.cancel` — it never patches `approval: null`. The `turn_end` handler (store.ts:3371-3407) never clears `approval` either. Daemon-side the abort resolves the request locally and emits nothing: `daemon/interactions.ts:214-221` `endTurn` does `this.permissions.delete(requestId); pending.resolve('reject')`, and `finishPermission` (interactions.ts:285-293) / `finishQuestion` (295-303) likewise resolve with no client event. A later click then hits `respondPermission`, which returns `false` for the now-unknown id (interactions.ts:162-165), and the renderer deliberately keeps the card: store.ts:1193-1201 — "Clear only on confirmation … dropping the card would strand it" → pushes 'approval refused' and leaves `approval` set.
```

**Verifier:** Verified end to end. App.tsx:1148-1165 ApprovalCard has only allow_once/allow_session/deny — no dismiss; QuestionCard (App.tsx:1315-1317) and PlanReviewCard likewise. App.tsx:913-956 renders the card on `snap.approval !== null` alone, never gated on turnActive, so it outlives the turn. store.ts:1015-1019 cancel() only calls turn.cancel; the turn_end handler (store.ts:3379-3417) patches turnActive/failed/queue/metricPhase and never approval/question; the only nulling sites are store.ts:1200 (successful response) and 2568/2621 (session (re)initialization). There is no `approval_response` case in the renderer event switch at all (grep across src/ hits only PROTOCOL.md, bridge/wireEvents.ts, daemon/server.ts:8900), so the daemon-side half of the proposed fix would not work without adding a renderer handler. Daemon side confirmed: runtime.ts:618 cancelTurn aborts the controller whose signal reaches streaming/loop.ts:1043 -> interactions.ts:251 abort -> finishPermission('reject'), deleting the id with no client event; interactions.ts:212-222 endTurn does the same for permissions and finishQuestion('') for questions. A later click then hits respondPermission -> false (interactions.ts:162-165), server.ts:8887-8904 returns {ok:false} and emits nothing, and store.ts:1193-1201 intentionally keeps the card and pushes 'approval refused'. That comment documents a different scenario (id owned by another connection), not the post-cancel one, so this is not a documented design decision for this path. App.tsx:824+844-853 confirms needsInput outranks turnActive, pinning the header badge at 'needs input' for later turns. Two overstatements in the original claim, not enough to refute: the Stop button still renders while turnActive (App.tsx:851), and the card is not literally permanent — navigating to another session and back clears it (store.ts:2621) and a fresh approval_request/question_request overwrites it. Reachability caveat: the approval flavor needs permission mode 'ask' (repo default is accept-all), but the question/plan-review flavor is reachable in the default config, where Esc on a plan-review card is a very plausible 'dismiss this' gesture.
</details>

### [approvals] ApprovalCard renders one fallback-generated line and nothing else: the daemon ships the full tool `inputs` on every approval_request (turnRunner.ts:1550) but store.ts:3330-3344 and types.ts:181-188 discard it — so under the shipped accept-all default the only approvals users ever see (send_message, cron/schedule create, remote trigger) display e.g. `send_message(telegram)` with recipient, message body and schedule hidden

**`xerxes/src/desktop/renderer/store.ts:3322`**

*Impact* — Asked to authorize an Edit, the user sees literally `Edit: /Users/erfan/.../server.ts` — no diff, no old string, no new string, no line count, and no indication of the cwd the path is relative to. Approving a file mutation blind is the single highest-stakes click in the product and the UI gives the user strictly less information than `git status` would. Same for `Write` (no content), `Agent`/`SpawnAgents` (prompt truncated to 60 chars at permissions.ts:183), and any unrecognised tool, which falls through to `${name}(${firstValue.slice(0,60)})` (permissions.ts:202-203).

*Fix* — Add `inputs?: Readonly<Record<string, unknown>>` and `cwd?: string` to the `Approval` type and carry `payload.inputs` through store.ts:3328. In `ApprovalCard`, branch on `approval.toolName`: for Edit/MultiEdit render the existing `DiffPreview` component from `old_string`/`new_string`; for Write render a capped content preview; for Bash render the command in a mono block with the resolved `workdir`; for everything else render a collapsed `<details>` with pretty-printed JSON. Always render the working directory as a subtitle line.

<details><summary>Evidence & verification</summary>

```
The daemon puts the complete argument object on the wire — `daemon/server.ts:9669-9679`:
```
    case 'permission_request':
      return { type: 'approval_request', payload: { id, request_id, name, action, tool_name, description: event.request.description, inputs: event.request.inputs } }
```
The renderer parses only four scalars and never reads `inputs`:
```
3322:      case 'approval_request': {
3325:        const description = str(payload.description) || `${str(payload.name)} ${str(JSON.stringify(payload.arguments ?? ''))}`.trim()
3328:        this.patch({ approval: { id, action, ...toolCallId, ...toolName } })
```
and `types.ts:181-189` has no field for it. The description itself is deliberately thin — `streaming/permissions.ts:170-172` returns `Edit: ${file_path}` for Edit and `Write to: ${file_path}` for Write, with no old/new content. The card renders only that string (App.tsx:1156 `<pre className="approval__desc">{approval.description}</pre>`). There is no other surface: `permission_request` is yielded at `streaming/loop.ts:1042`, strictly *before* `tool_start` at line 1101, so the tool-call block the approval refers to does not exist in the transcript yet.
```

**Verifier:** Verified end-to-end. createPermissionRequest (streaming/loop.ts:1664-1670) attaches the full argument object as `inputs`; turnRunner.ts:1550-1561 forwards it on the approval_request payload; wireEvents.ts:226-238 deliberately preserves `optionalRecordField(payload,'inputs')`; daemon/server.ts:9645-9682 passes the payload through untouched. The renderer then drops it: store.ts:3330-3344 destructures only id/action/description/tool_call_id/tool_name, types.ts:181-188 has no inputs field, and App.tsx:1148-1165 renders a single `<pre>{approval.description}</pre>` plus three buttons — no other renderer file consumes approval data. permission_request is yielded at loop.ts:1042 before tool_start at :1101, and tool_call_id is never even in the payload, so the inline-grouping path at App.tsx:917 never matches and no tool block exists to fall back on. Corrections to the finding: the daemon cite is wrong (the mapping is turnRunner.ts:1550, not server.ts:9669) and the store line is 3330, not 3322. The Edit/Write framing is also misleading for the shipped default — DEFAULT_PERMISSION_MODE = 'accept-all' (permissions.ts:14) means Edit/Write never prompt unless the user opts into manual/plan. But that makes the impact worse, not better: ALWAYS_APPROVAL_TOOLS (permissions.ts:37) forces send_message, RemoteTriggerTool, ScheduleCronTool, manage_schedule create/update/resume/run and CreatorForgeTool define/undefine to prompt even under accept-all, and none of those names match any branch of permissionDescription, so every approval a default-config user ever sees renders via the generic fallback at permissions.ts:202-203. For send_message (schema sendMessage.ts:86-93: platform/recipient/text/files) the card reads `send_message(telegram)` — recipient, body and attachments are present in payload.inputs and thrown away. No comment anywhere documents this as intentional; the protocol validator going out of its way to preserve `inputs` is evidence of the opposite.
</details>

### [layout] Agents and Skills & tools pages break their two-column split at 900-1150px windows: fallbacks at atelier.css:202/321 are @media (window) not @container (pane), and minWidth 760 (main.ts:143) makes them unreachable dead code — at a 960px window the detail pane is ~60px wide (the run-filters/rail part of the claim is false: UnifiedRuns renders only in the ~780px schedules sheet)

**`xerxes/src/desktop/renderer/atelier.css:202`**

*Impact* — `DesktopPage` (Agents / Skills & tools / Artifacts) renders inside the `.chat` column, which is 318-600px in any normal three-pane layout — but its responsive rules key off `@media(max-width:700px)`, i.e. the OS window. On a 900px window with the sidebar and rail open, the chat is 318px, the media query never fires, and the Skills catalog gets a fixed 230px nav + 20px gap + 12px/20px section padding, leaving roughly 36px for the plugin/skill detail text — one or two characters per line. The Agents page is the same with a 200px nav and 26px section padding (~66px of content). The panes are already `container-type:inline-size`, so the fix is a one-word change that was simply missed.

*Fix* — Convert these to container queries against the pane that actually holds them: change `@media(max-width:700px)` at atelier.css:202 and atelier.css:319 to `@container(max-width:700px)` (the `.atelier .chat` container declared at atelier.css:437 is the nearest inline-size container for both `.desktop-page` and `.catalog-page`). Do the same for `@media(max-width:700px){.run-filters{grid-template-columns:1fr}}` at atelier.css:724, since `UnifiedRuns` (UnifiedRuns.tsx:34) renders inside the 260px rail where three equal columns give ~75px each.

<details><summary>Evidence & verification</summary>

```
atelier.css:186  .desktop-page .studio-split{grid-template-columns:200px minmax(0,1fr);min-height:0}
atelier.css:202  @media(max-width:700px){.desktop-page .studio-split{grid-template-columns:1fr}
atelier.css:302  .catalog-page .catalog-browser{gap:20px;margin-top:12px;grid-template-columns:230px minmax(0,1fr)}
atelier.css:321  .catalog-page .catalog-browser{grid-template-columns:minmax(0,1fr);gap:12px}   /* inside @media(max-width:700px) at line 319 */
App.tsx:885  {page && <DesktopPage panel={page} snap={snap} />}    // rendered INSIDE <main className="chat">
atelier.css:437  .atelier .chat{container-type:inline-size; ...}
```

**Verifier:** Verified line-by-line. atelier.css:186/202 and :302/319-321 contain exactly the cited rules, and DesktopPage (DesktopPanels.tsx:112) renders only at App.tsx:885 inside <main className="chat"> as an in-flow flex child (.desktop-page.studio-sheet, atelier.css:181 — not a fixed overlay), so both pages are bounded by the chat column while their responsive fallbacks key off the OS window. The math checks out and is worse than claimed: with layout.tsx:11 defaults (sidebar 240 / inspector 340) the rail auto-opens whenever window >= 900 (App.tsx:145), giving chat = 900-240-340-2 = 318px exactly as claimed; but both panels also sit inside .studio-form, which atelier.css:187 pads 20px 32px, so the Skills detail pane is ~4px (not 36px) and the Agents detail pane ~2px (not 66px). Stronger still: main.ts:143 sets minWidth 760, so @media(max-width:700px) can never fire at default zoom — the fallbacks at atelier.css:202, :321 and app.css:113 are unreachable dead code. No mitigation exists: narrow mode (App.tsx:140) only hides the sidebar below 850px, the .desktop-rail overrides (atelier.css:195) cover the rail copy only, and navigating to a page leaves the rail open (App.tsx:155). No comment marks this intentional, and the file already uses @container correctly at :406 and :516. ONE SUB-CLAIM IS REFUTED: the run-filters addendum (atelier.css:724) is wrong — UnifiedRuns is reachable only via SchedulesPanel (DesktopPanels.tsx:613), which renders only inside DesktopSheet (DesktopPanels.tsx:201), a ~780px modal; DesktopRail handles only files/review/activity, so run-filters never lives in the 260px rail. Severity stays high: at a 960px window (exactly half a 1080p display, a routine size for a coding assistant beside an editor) the Skills & tools and Agents pages render their detail text in roughly 60px, i.e. unreadable, and the pages are effectively unusable across the entire ~900-1150px band.
</details>

### [navigation] Opening Changes (or an expanded Files/Activity rail) sets data-review/data-context-full, and atelier.css:492/:524 display:none the entire .chat column — deleting the composer, ApprovalCard, QuestionCard and the only "needs input" badge (App.tsx:850, whose own comment calls it the last surviving signal). With OS notifications suppressed while focused (notify.ts:70) and no waiting state on sidebar rows (store.ts:131), a mid-review AskUser question is invisible AND unanswerable until the user happens to close the rail; approvals are invisible but still fire blind via the global 1/2/3 binding (App.tsx:1844-1851).

**`xerxes/src/desktop/renderer/atelier.css:492`**

*Impact* — The Changes review and the expanded Files/Activity rail are full-width takeovers implemented by hiding `.chat`. Everything that asks the user for a decision lives inside `.chat`: the composer, ApprovalCard, QuestionCard, the `needs input` badge and the Stop button. Review a diff while the agent runs, an approval lands, and there is zero signal anywhere — the topbar's only persistent status is the shell/watcher count (DesktopPanels.tsx:1390), and the OS notification is suppressed because the window is focused (notify.ts:70). The turn just sits blocked until the user happens to close the rail. Worse, `contextRequiresFullWidth` (App.tsx:146) is derived from window width, so merely dragging the window narrower than sidebar+inspector+320px with the Activity rail open makes the conversation and its approval prompt vanish with no user action at all.

*Fix* — Stop hiding `.chat` wholesale. Either (a) keep a persistent needs-input strip in the topbar — render the `needs input` badge + Stop + an "Answer" button from `snap.approval || snap.question` in Topbar (App.tsx:197-208), which survives every takeover since `.top` is outside `app__body`; or (b) better, force `rail`/`filesExpanded` back to a split layout whenever `snap.approval || snap.question` is non-null, so the takeover auto-yields to the decision. Also drop the automatic width-driven takeover: when `contextRequiresFullWidth` flips true, collapse the rail to an overlay drawer instead of deleting the conversation.

<details><summary>Evidence & verification</summary>

```
.atelier .app__body[data-review]>.chat{display:none}   (line 492)
.atelier .app__body[data-context-full]>.chat{display:none}   (line 524)

App.tsx:166  <div className="app__body" data-context-full={rail && rail !== 'review' && (filesExpanded || contextRequiresFullWidth) || undefined} data-review={rail === "review" || undefined}>
App.tsx:894  <Composer snap={snap} />   // inside <main className="chat">
App.tsx:850  <span className="badge badge--need">needs input…</span>   // inside .chat__head
App.tsx:945/951 <ApprovalCard …/> , 954 <QuestionCard …/>   // inside <Stream> inside .chat
main/notify.ts:70  if (state.anyWindowFocused) return null
```

**Verifier:** Verified line-by-line. atelier.css:492 (.atelier .app__body[data-review]>.chat{display:none}) and atelier.css:524 (same rule for [data-context-full]) are exactly as quoted, driven by App.tsx:166, and nothing overrides them (.atelier .chat at :437 has lower specificity; atelier.css loads after app.css per index.html:13-14). Every decision surface lives inside .chat: Composer (App.tsx:894), ApprovalCard (:945/:951), QuestionCard (:952-954) and the sole "needs input" badge (:850). That badge's own comment at :845-847 states the header is "the only place the signal survives" on other tabs — these two CSS rules delete exactly that fallback, so the code contradicts its documented intent. No alternate cue exists: sidebar dots fold "waiting" into "working" (store.ts:131), the topbar carries only shell/watcher counts (DesktopPanels.tsx:1349+), and notify.ts:70 suppresses the OS notification while any window is focused. Two corrections to the claim: (1) approvals ARE still answerable — GlobalKeys App.tsx:1844-1851 binds 1/2/3 outside .chat — but blind, with the description unreadable; questions are the genuinely unanswerable case, since QuestionCard has no keyboard path. (2) The width-driven takeover only fires once railChoice is set (App.tsx:144-146); with the untouched default, narrowing sets rail=null and the chat survives, so it requires one prior click on Activity/Files. The full-width review layout itself looks deliberate (comment at App.tsx:145), so the correct fix is a needs-input affordance that survives outside .chat (or auto-yielding the takeover when snap.approval||snap.question is non-null), not abolishing the takeover.
</details>

### [settings] New-task modal labels the permission mode "in this workspace" but it is session-pinned (server.ts:5114, runtime.ts:1371) and startTask (store.ts:1814-1834) never forwards it, so a task started from a `manual`-pinned session silently opens at the daemon default accept-all (runtime.ts:1218-1225,1506) with the modal still showing `manual`

**`xerxes/src/desktop/renderer/App.tsx:429`**

*Impact* — A user who pinned `manual` in the current session opens ⌘K → New task and reads "approvals in this workspace: manual". They hit Start. The daemon opens a fresh session at its own default (accept-all), and the agent starts running shell commands and writes with no approval cards at all. The label is wrong on both axes: wrong scope word ("workspace" for a session pin) and a value that belongs to the session they are leaving, not the one they are about to create.

*Fix* — Carry the mode across: in `startTask` (store.ts:1796), right after `beginFreshTask` resolves and alongside the existing `/model` slash, issue `slash { session_key: this.sessionKey, command: `/permissions ${previousMode}` }` when the user's pinned mode is non-default. Better: replace the read-only fieldnote with a real selector in the modal — `PERMISSION_MODES` already exists (Overlays.tsx:866) — so the mode is chosen for the task being created. Either way change the copy from "in this workspace" to "for this task".

<details><summary>Evidence & verification</summary>

```
App.tsx:429 — `approvals in this workspace: {snap.permissionMode || 'daemon policy'} — <u ...>change</u>`
store.ts:2329 — `.call('slash', { command: `/permissions ${mode}`, session_key: sessionKey })`  // pinned PER SESSION
store.ts:1796-1815 `startTask()` — forwards only `/model ${model}` (1803) and `set_plan_mode` (1808) to the fresh session. There is no `/permissions` call anywhere in startTask/beginFreshTask.
store.ts:2615 — `...(str(result.permission_mode) ? { permissionMode: str(result.permission_mode) } : {})` // initialize overwrites the snapshot with the NEW session's mode
```

**Verifier:** Confirmed at every link. App.tsx:428-431 renders the literal copy "approvals in this workspace: {snap.permissionMode || 'daemon policy'}" and it is visible (atelier.css:159 styles .atelier .taskmodal .fieldnote rather than hiding it). The mode is session-scoped, not workspace-scoped: store.ts:2347 sends `/permissions <mode>` with a session_key; server.ts:5114-5118 comments "Scoped to this session so a second one keeps its own trust level" and calls runtime.setSessionPermissionMode, whose implementation (runtime.ts:1355-1378) sets session.permissionMode with permissionPinned:true on that single session; the daemon-wide runtime.reload fallback at server.ts:5120 only fires when no session exists, which is never the desktop path. A fresh session is unpinned, so runtime.ts:1218-1225 applies runtimeSettings.permission_mode, defaulting to "accept-all" (runtime.ts:1506-1507). store.ts startTask (1814-1834) forwards only `/model` (1821) and set_plan_mode (1826), and beginFreshTask (1160-1176) passes only session_key + agent_id to initialize — no /permissions anywhere. store.ts:2624-2627 then overwrites snapshot.permissionMode with the NEW session's mode, with a comment that itself states "/permissions pins the mode per session", confirming the label's "in this workspace" wording contradicts the codebase's own documented model. No later CSS rule, code path, or comment handles this; the state is trivially reachable (pin manual, hit New task). Impact is a stricter pin silently dropping to accept-all while the modal asserts the strict value, with no other trustworthy indicator (App.tsx:1161 hardcodes "session policy: ask"). Only nit in the original report: the PERMISSION_MODES reference at Overlays.tsx:866 was not independently verified; the mechanism does not depend on it.
</details>


## MEDIUM

### [a11y] ApprovalCard prints "1 / 2 / 3" but App.tsx:1846 drops those keys whenever focus is in an INPUT/TEXTAREA — and nothing ever moves focus off the composer (focused at 1444/1447/1448/1520, never blurred by send()), so the advertised shortcuts are dead in the default post-submit state; the role="alertdialog" card (1150) also has no focus move and no aria-live, so screen readers announce nothing (same for QuestionCard, guard 1230/1244)

**`xerxes/src/desktop/renderer/App.tsx:1846`**

*Impact* — The single most blocking event in the product — the agent stopped and is waiting for you — is silent and its own documented shortcuts do not work in the state users are actually in. With the caret in the composer (where the app puts it), pressing 1 types "1" into the draft instead of approving; the user sees "1 / 2 / 3" printed right there and watches it fail. For a screen-reader user it is worse: role="alertdialog" is not auto-announced the way role="alert" is — it expects focus to be moved into it — so nothing at all is spoken, and the turn just appears to hang forever. Same silence applies to QuestionCard (App.tsx:1259, role="form") whose number-key handler bails on the same test at App.tsx:1230.

*Fix* — When snap.approval transitions null -> set, store document.activeElement, focus the "Allow once" button (App.tsx:1158) via a ref, and restore the stored element after store.approve resolves. That alone fixes the shortcuts, because focus then sits on a BUTTON and the App.tsx:1846 guard no longer trips. Additionally wrap the card in a visually-hidden assertive announcer (or add aria-live="assertive" to the .approval root) carrying "Approval required: <toolName>", and either add aria-modal="true" + a focus trap or downgrade role to "group" — an inline alertdialog nested in the transcript is role misuse. Do the same focus move for QuestionCard.

<details><summary>Evidence & verification</summary>

```
GlobalKeys: `// Approval keys work unless a field has the focus.` / `if (!snap.approval) return` / `const target = event.target as HTMLElement | null` / `if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return` / `if (event.key === '1') store.approve(...)`. The card itself prints the shortcut: App.tsx:1154 `<span className="approval__keys">1 / 2 / 3</span>`, and is declared `role="alertdialog" aria-label="Tool approval"` (App.tsx:1150) with no aria-modal, no ref, and no focus() anywhere — grep for focus() in App.tsx returns only 230/235/248/475/502/1444/1447/1448/1520, none of them the approval. Meanwhile the composer textarea is focused programmatically on session open and after every hint/context insert (App.tsx:1444, 1447, 1448, 1520).
```

**Verifier:** Confirmed line-by-line. App.tsx:1846 is exactly `if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return`, sitting above the only 1/2/3 approval handlers (1847-1849); grep for `allow_once` in the renderer returns just App.tsx:1158 (the button) and 1847 (the guarded key), so there is no second code path that rescues the shortcut. The card prints the shortcut at App.tsx:1154 and on each button as <kbd>1/2/3</kbd> (1158-1160), and is role="alertdialog" (1150) with no ref, no focus() and no aria-modal — the only focus() calls in App.tsx are 230/235/248/475/477/502/1444/1447/1448/1520, none on the approval. The composer textarea is focused on session change (1444), on context/command insert (1447-1448) and on hint pick (1520), and send() (1468-1480) never blurs, so after the user hits Enter the caret is still in the TEXTAREA when the approval arrives mid-turn — the guard trips in the dominant flow and '1' types into the draft. Same pattern for QuestionCard (guard at 1230/1244, role="form" at 1259) and PlanReviewCard (1354). The a11y half also holds: role="alertdialog" with no focus move and no aria-live is not announced, and main/notify.ts:70 suppresses the OS ping whenever a window is focused — i.e. in exactly this state. Two things argue against 'high': the user is never actually blocked (the three buttons are visible and clickable), and App.tsx:849 renders a pulsing `needs input` badge in the chat header (the comment at 845 states this is deliberate so the signal survives on other tabs), so 'the turn appears to hang forever' is overstated for sighted users. Note too that the guard is deliberate and must be kept — the fix is to move focus onto the Allow-once button (a BUTTON, which the guard does not exclude), not to delete the guard. Broken advertised affordance plus a real screen-reader gap, but with a working mouse path and a visible header signal: medium.
</details>

### [a11y] Command palette and model picker have no listbox/option roles at all (Overlays.tsx:1497-1509, 1031-1052) and neither they nor the composer hints link their input to the list via aria-activedescendant, so arrow-key selection is invisible to assistive tech on all three command surfaces; the composer additionally traps both Tab and Shift+Tab because App.tsx:1591 matches e.key === 'Tab' without checking e.shiftKey

**`xerxes/src/desktop/renderer/App.tsx:1549`**

*Impact* — A screen-reader user presses ⌘K, hears "Type a command, edit text", arrows down five times and hears nothing at all, then presses Enter and an unknown command executes — switching model, provider, or submitting /compact. Identical failure typing "/" in the composer: the hints strip is announced only if they hunt for it, arrowing is silent, and Tab is hijacked so they can never reach the options list to read it. These three surfaces are the app's entire command vocabulary.

*Fix* — Make each input a real combobox: on the composer textarea and the two palette inputs add role="combobox", aria-expanded={hints !== null}, aria-controls={listId}, aria-activedescendant={`${listId}-${index}`}, and give each option a matching id. Wrap the palette/model lists in role="listbox" with role="option" + aria-selected on each row (they are already buttons, so keep them focusable but set tabIndex={-1}). Keep the Tab-completes behaviour but add a documented escape hatch (e.g. Shift+Tab leaves the hints).

<details><summary>Evidence & verification</summary>

```
Composer hints: `<div className="hints__list" role="listbox" aria-label="Command and skill hints" ref={hintsRef}>` with `<button role="option" aria-selected={index === hints.index}>` (App.tsx:1555-1556), while focus stays in the textarea (App.tsx:1574-1605) which has only `aria-label="Message"` — no role="combobox", no aria-expanded, no aria-controls, no aria-activedescendant. `grep -rn "activedescendant|role=\"combobox\"|aria-controls|aria-owns|aria-labelledby" *.tsx` returns exactly ONE hit in the whole renderer: App.tsx:198 (the sidebar toggle). The textarea also swallows Tab to complete (App.tsx:1591 `else if (e.key === 'Tab' && hints && hints.items[hints.index])`), so the options cannot be reached by Tab either. Command palette is worse — Overlays.tsx:1497-1509 is a plain `<div className="palette__list">` of buttons whose cursor is only `className={`prow${index === cursor ? ' is-sel' : ''}`}`, and Enter runs `filtered[cursor]` (Overlays.tsx:1468) with the footer promising `<kbd>↑↓</kbd> select` (Overlays.tsx:1519). ModelPicker repeats it at Overlays.tsx:1041 (`' is-hover'`).
```

**Verifier:** All cited lines say what is claimed, verified by reading them.

CONFIRMED:
- App.tsx:1549-1556 — the hints strip IS `role="listbox"` with `role="option"` + `aria-selected={index === hints.index}` on each row. So the list itself is annotated; what is missing is the link from the input.
- App.tsx:1574-1577 — the textarea carries only `aria-label="Message"`: no role="combobox", no aria-expanded, no aria-controls, no aria-activedescendant. Focus never leaves it (pickHint at App.tsx:1520 explicitly calls `ref.current?.focus()`), and arrowing only mutates React state (App.tsx:1585-1589) plus a visual `scrollIntoView` (App.tsx:1525-1528). Nothing an AT can observe changes, so arrowing is silent.
- App.tsx:1591 — `e.key === 'Tab'` is matched without checking `e.shiftKey`, so BOTH Tab and Shift+Tab are consumed into pickHint while hints are open. The claim understated this: it is a bidirectional keyboard trap out of the composer while the strip is open (escapable only via Escape/Enter/completion).
- Overlays.tsx:1497-1509 — the palette list is a bare `<div className="palette__list">`; the cursor is purely `' is-sel'` CSS. Overlays.tsx:1465-1469 runs `filtered[cursor]` on Enter while focus is in the input, and Overlays.tsx:1519 advertises `<kbd>↑↓</kbd> select`. Same shape in ModelPicker (Overlays.tsx:1031-1052, `' is-hover'`, cursor scroll at Overlays.tsx:992).
- Their grep is accurate: for {aria-activedescendant, role="combobox", aria-controls, aria-owns, aria-labelledby} the whole renderer has exactly one hit, App.tsx:198 (sidebar toggle).

This is NOT a deliberate documented decision. The surrounding comments (App.tsx:1496-1497, 1523, 1586, Overlays.tsx:1012-1014, 1449-1450, 1020) document debounce, scroll, Escape-propagation and palette-lifetime intent — none mention accessibility. The codebase clearly does care about a11y elsewhere (aria-live/role=status at App.tsx:249, 887, 1570; a full `role="treeitem"`/aria-level/aria-selected tree at WorkspaceFileTree.tsx:55), which makes these three surfaces an omission, not a stance.

OVERSTATED, hence the downgrade:
1. Title says "no combobox semantics anywhere" — the composer hints already have listbox/option/aria-selected; only the input→list association is missing there. The palette and model picker are the ones with no roles at all.
2. Palette impact: those rows are native `<button>`s and the palette does NOT intercept Tab, and Overlays.tsx:1466 (`if (event.target instanceof HTMLButtonElement) return`) deliberately lets a focused button handle its own Enter. So a screen-reader user CAN Tab into the palette list and hear each command — the failure is the arrow path being silent and Enter-on-input firing an unannounced cursor, not total unreachability. Only the composer is genuinely unreachable-by-Tab.
3. Zero impact on the sighted keyboard user: the visual cursor, scroll-into-view and the advertised Tab-completes contract (footer text at App.tsx:1566) all work as designed.

Net: a real, specific, fixable a11y defect on the app's three command surfaces, with no functional impact on sighted users — medium, not high.
</details>

### [a11y] Every tab strip and segmented control (chat tabs App.tsx:876-880, sidebar Sessions/Agents App.tsx:618, Settings nav Overlays.tsx:52, Theme Overlays.tsx:215-217, font size Overlays.tsx:224) encodes selection only in an `is-on`/`is-selected` CSS class — no aria-pressed/aria-selected/role=tab anywhere, and `.tab.is-on` (app.css:881) and `.seg button.is-on` (app.css:1129) differ only by colour, so the current view, settings pane, theme and font size are unreadable to screen readers, to forced-colors users, and to the AX-tree inspection the repo's own parity testing relies on (docs/desktop-parity.md:206,251)

**`xerxes/src/desktop/renderer/App.tsx:876`**

*Impact* — A screen-reader user in Settings hears "General button, Models & Providers button, Agent presets button…" with no way to know which pane is showing, and after clicking gets no confirmation that anything changed. Same for Theme (System/Dark/Light) and font size: the control reports no state, so the user cannot tell what the app is currently set to. In the chat header they cannot tell whether they are looking at Conversation, Plan or Log. Windows High Contrast / forced-colors users lose the same information, because `.tab.is-on` is distinguished by colour plus a border-bottom tint only.

*Fix* — Add `aria-pressed={snap.tab === 'activity'}` to the three chat tabs (cheapest, matches the existing DesktopPanels.tsx:127 convention and its atelier.css:193 styling), or promote them to a full tablist: role="tablist" on `.tabs`, role="tab" + aria-selected + aria-controls on each button, role="tabpanel" on the rendered pane (App.tsx:888-891), with roving tabIndex and Left/Right arrow handling. Apply aria-pressed to Overlays.tsx:52, 215-217, 224 and App.tsx:618 the same way.

<details><summary>Evidence & verification</summary>

```
Primary workspace tabs: `<button className={`tab${snap.tab === 'activity' ? ' is-on' : ''}`} onClick={() => store.setTab('activity')}>Conversation</button>` and siblings at 877 (Plan & todos) and 880 (Log) — no role="tab", no aria-selected, no aria-current, and no role="tablist" on the `.tabs` wrapper (App.tsx:875). Same pattern: sidebar Sessions/Agents segment App.tsx:618 (`className={page !== 'agents' ? 'is-selected' : ''}`), Settings navigation Overlays.tsx:52 (`className={`mtab${snap.settingsTab === tab.id ? ' is-on' : ''}`}`), Theme segment Overlays.tsx:215-217, font-size segment Overlays.tsx:224. `grep -rn "aria-current|aria-selected|role=\"tab\"|role=\"radio\"" *.tsx` finds only App.tsx:1556, DesktopPanels.tsx:1024 and WorkspaceFileTree.tsx:55. The codebase already knows the right pattern — DesktopPanels.tsx:127 builds the rail nav with `aria-pressed={panel === value}` and atelier.css:193 even styles `[aria-pressed=true]`.
```

**Verifier:** Every cited line is verbatim accurate. App.tsx:875-881 shows `.tabs` with no role and three `<button className={`tab${snap.tab === 'x' ? ' is-on' : ''}`}>`; App.tsx:618 uses only `className={page !== 'agents' ? 'is-selected' : ''}`; Overlays.tsx:52, 215-217, 224 are class-only too. An exhaustive grep for aria-current|aria-selected|role="tab|role="radio|aria-pressed across the renderer yields only 6 hits (App.tsx:1277,1556,1645; DesktopPanels.tsx:127,1024; OutputViewer.tsx:34; WorkspaceFileTree.tsx:55) — none on a tab strip or segment, so no later code path mitigates it. No aria-live region compensates: the only ones (App.tsx:249,887,934,1570; DesktopPanels.tsx:1390) announce connection/turn status. No comment marks this as intentional, and the codebase demonstrably knows the correct pattern (DesktopPanels.tsx:127 aria-pressed, styled at atelier.css:193). The forced-colors sub-claim also holds: app.css:874-881 gives `.tab` `border-bottom: 2px solid transparent` and `.tab.is-on` differs only by `color` + `border-bottom-color`; app.css:1129 `.seg button.is-on` differs only by background/color; there is no forced-colors block in app.css or atelier.css (only prefers-contrast:more at atelier.css:573, which restores no selection cue). An angle the finding missed strengthens it: docs/desktop-parity.md:206,251 and docs/ssh-setup-recovery-2026-09-20.md:15 show the author verifies this app through the native macOS accessibility tree, which cannot read a CSS-class-only selected state — so this also degrades the project's own acceptance-testing loop. I lowered severity from high to medium because nothing is functionally broken or unreachable for a sighted user (selection is visible via colour plus a 2px accent underline); this is a WCAG 4.1.2 state-exposure gap in an app that otherwise has real a11y investment (29 aria-labels in App.tsx, a global :focus-visible ring at atelier.css:42), so it is categorically less severe than defects that blank the window or hide controls outright.
</details>

### [copy] One background process is named two ways in adjacent UI — "Restart workspace runtime" (App.tsx:252) calls store.restartDaemon() — and the version-mismatch popover contradicts itself in three lines: heading "App and runtime versions differ" (App.tsx:250), body "restart the daemon" (buildInfo.ts:69), button "Restart workspace runtime"; the disconnected state carries five different labels (App.tsx:218, :887, :1051, :1534, TerminalsPanel.tsx:46), and the composer's imperative "Connect to a daemon first…" (App.tsx:1534) names an action no control offers — the adjacent controls all say "Retry" (App.tsx:887, :1056) — and it is the only prominent text on the blank screen shown while connection==='connecting' with an empty transcript.

**`xerxes/src/desktop/renderer/App.tsx:1534`**

*Impact* — A user who loses connection sees "Runtime offline" in the top bar, "Connecting to the shared daemon" on the error screen, "Daemon offline" in Terminals (TerminalsPanel.tsx:46) and "Connect to a daemon first…" in the composer — four labels for one state, two nouns for one process, so it reads like four separate subsystems are broken. Worse, the composer placeholder is an imperative for an action that does not exist: there is no connect-to-daemon control anywhere; the process autostarts and store.retryConnection() is the only affordance. The user stares at an instruction they cannot follow. " · daemon did not report" likewise gives the user a plumbing status instead of telling them which permission mode is active.

*Fix* — Pick one user-facing noun — "runtime" is already the friendlier half — and purge "daemon" from every string: Settings row 'Daemon' → 'Xerxes runtime'; Offline h1 → 'Reconnecting to the Xerxes runtime'; App.tsx:1534 → 'Reconnecting — messages resume automatically' (state, not an order the user can't obey), or 'Runtime offline — Retry' with the retry button inline. Overlays.tsx:883 → `Permission mode${current ? '' : ' · unknown'}` and render the unknown case as 'Not reported yet — reconnecting'. Add a lint rule (grep for /daemon|RPC|socket|worktree|fold\b/ in .tsx string literals) so the word cannot come back.

<details><summary>Evidence & verification</summary>

```
App.tsx:1534 composer placeholder: `? 'Connect to a daemon first…'`. Same file, RuntimeStatus uses the other word for the same thing — App.tsx:218 `'Updating runtime…' : … 'Runtime offline'`, :250 `<strong>App and runtime versions differ</strong>`, :252 `Restart workspace runtime`. Then the full-screen error goes back to the first word — App.tsx:1051 `'Connecting to the shared daemon'`, :1055 `The terminal, TUI and desktop app share a daemon across workspaces.`, sidebar empty state App.tsx:677 `'Connecting to the shared daemon…'`. Settings has a top-level row literally titled `<div className="row__t">Daemon</div>` (Overlays.tsx:239) with subtitle `'connected · shared across workspaces' : 'offline — retrying with backoff'`, while the Permissions pane says `The daemon evaluates every tool call against its policy` (Overlays.tsx:879) and labels the mode `Permission mode{current ? ` · daemon reports: ${current}` : ' · daemon did not report'}` (Overlays.tsx:883). 20 user-visible occurrences of "daemon" total across App/Overlays/TerminalsPanel/Workspaces.
```

**Verifier:** Every cited string verified verbatim: App.tsx:1534 is exactly `? 'Connect to a daemon first…'`; App.tsx:218 'Runtime offline', :250 'App and runtime versions differ', :252 'Restart workspace runtime', :1051, :1055, :677, Overlays.tsx:239/879/883 and TerminalsPanel.tsx:46 all match. The two nouns provably name ONE process: the "Restart workspace runtime" button at App.tsx:252 calls store.restartDaemon(), and "Runtime offline" is driven by snap.connection — the same daemon socket the Offline screen calls "the shared daemon". The leak is wider than reported (buildInfo.ts:47,55,69; Overlays.tsx:60,168,283,433,911,1067,1288,1513,1522; App.tsx:429,639; Workspaces.tsx:63,178; ChannelsPanel.tsx:45,54; AgentInspector.tsx:24; desktopRpc.ts:7,21). No comment anywhere documents the split as deliberate. HOWEVER the load-bearing sub-claim "there is no connect-to-daemon control anywhere... the user stares at an instruction they cannot follow" is FALSE and I refute it: App.tsx:1056 renders "↻ Retry now" inside the Offline card, which App.tsx:925 places in .stream directly above the Composer (App.tsx:894); App.tsx:887 renders a connection-status banner with "Retry now" immediately above the composer whenever blocks exist; Overlays.tsx:244 adds a "reconnect" button in Settings. A working recovery control is adjacent to the placeholder in every reachable state, so the defect is a verb mismatch ("Connect" vs the button's "Retry"), not an unreachable action — which is why severity drops from high to medium (clarity/trust cost in an error path, no broken capability, no data loss). Two things the original missed that I verified and that sharpen it: (a) the version-mismatch popover contradicts itself within three lines — heading "App and runtime versions differ" (App.tsx:250), body from buildInfo.ts:69 "App and daemon builds differ — restart the daemon, and quit and relaunch the app.", button "Restart workspace runtime" (App.tsx:252) — so the instruction names a noun that appears on no button, the one place the split has real actionability cost; (b) connection==='connecting' with an empty transcript renders a totally blank stream (offline is false so no Offline card, `empty` requires online so no Welcome, the banner needs blocks.length>0), leaving "Connect to a daemon first…" as the most prominent text while the app is already connecting.
</details>

### [copy] Provider profile editor labels the full catalog count ("N discovered") but renders only the first 24 (Overlays.tsx:698/701) with no filter or overflow marker — and since CachedModelEditor is the renderer's only caller of saveModelCapabilities, per-model context/max-output overrides are unreachable for every model past #24

**`xerxes/src/desktop/renderer/Overlays.tsx:698`**

*Impact* — OpenRouter and OpenAI-compatible endpoints return hundreds of models. The label proudly says "Models · 327 discovered", the list shows 24, and the other 303 are unreachable — there is no scroll-to-load, no 'show all', no search box. A user who just fetched a catalog to select a specific model simply cannot select it from this panel; the only escape is guessing the id into the free-text Model field at Overlays.tsx:693. The label actively lies about what is on screen, so the user assumes the fetch failed rather than that the UI truncated.

*Fix* — Make the label describe what is rendered and add a way to reach the rest: `Models · showing {shown} of {profileModels.length}` plus a filter input above the list (`value={modelFilter}` filtering on `cached.id`), and raise/remove the cap once filtered (`profileModels.filter(m => m.id.includes(modelFilter)).slice(0, modelFilter ? 200 : 24)`) with a trailing `{rest > 0 && <div className="row__s">{rest} more — type to filter</div>}`. Apply the same 'showing X of Y' treatment to RemoteProviders.tsx:26.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:698 `<label>Models · {modelsLoading ? 'fetching…' : `${profileModels.length} discovered`}</label>` and three lines later Overlays.tsx:701 `{profileModels.slice(0, 24).map(cached => (`. Nothing renders a remainder count; contrast Overlays.tsx:567 which at least appends `{group.choices.length > 4 ? ', …' : ''}`, and StructuredResult.tsx:14 which appends `{entries.length-50} more entries in Raw details`. RemoteProviders.tsx:26 has the same silent cut: `.filter(profile=>profile.supported).slice(0,32)`.
```

**Verifier:** Lines confirmed verbatim: Overlays.tsx:698 prints `${profileModels.length} discovered`, Overlays.tsx:701 renders `profileModels.slice(0, 24)`. No upstream cap makes the label honest — store.ts:1270/338-366 dedupes but never truncates, and daemon/server.ts:4423 -> discoverModelCatalog (modelDiscovery.ts:101) bounds only response BYTES (1 MiB), never count. CSS does not compensate: app.css:1172-1175 gives the list max-height:180px/overflow:auto, so the user scrolls to a visible bottom at item 24 while the header claims the full count; atelier.css has no .provform rule. No filter, no "show more", no remainder text exists in Overlays.tsx:696-724. The strongest consequence the claimant missed: CachedModelEditor (Overlays.tsx:752) is the ONLY caller of store.saveModelCapabilities (store.ts:1290) in the whole renderer, and it only renders inside the sliced list — so context-limit/max-output overrides are permanently unreachable for models at index >=24, and typing the id into the free-text Model field (Overlays.tsx:693) does not surface that editor. However I refute two supporting claims, which is why severity drops from high to medium: (1) RemoteProviders.tsx:26 is NOT an analogous render cut — `.slice(0,32)` only seeds the default checked set via setSelected; line 54 renders `review.profiles.map(...)`, so every profile is visible and selectable. (2) The user is not stranded on model selection: ModelPicker (Overlays.tsx:1022-1035) renders `group.choices.map` uncapped behind a "Search models..." input for the active profile's catalog, and the free-text Model input sits three lines above the truncated list. docs/feature-gap-implementation-status.md:1016 ("Only a bounded slice of the catalog is rendered") documents the TUI agent-config field (Up/Down, F5), not this desktop panel, so it does not excuse this cap. Fix: label the rendered count ("showing 24 of 327"), add a filter input over the list, and lift the cap when filtered.
</details>

### [copy] Settings > General's "Plan this session" row is write-only: it hardcodes setPlanMode(true) and never reads snap.planMode (Overlays.tsx:293-299), so it always reads "enable now" even when plan mode is armed, cannot disable it, and gives no feedback — under a subtitle at :210 claiming "nothing here is per-task" in a card that also prints the session id

**`xerxes/src/desktop/renderer/Overlays.tsx:295`**

*Impact* — The subtitle tells the user this pane has no per-conversation settings, two rows down from a per-conversation setting and a row that prints the current session's id. And the plan-mode control is write-only: if plan mode is already on (the chat header chip at App.tsx:833 toggles it independently), Settings still says "enable now" and clicking is a no-op that gives no feedback. A user who opens Settings to check whether plan mode is armed before a risky task gets no answer, and a user who clicks thinks they changed something when they didn't.

*Fix* — Delete the false subtitle or narrow it to the rows it covers ('Theme and font apply to every window'). Convert the plan row to the same switch pattern as Stream thinking: `<button className={`switch${snap.planMode ? ' is-on' : ''}`} role="switch" aria-checked={snap.planMode} aria-label="Plan before changes" onClick={() => store.setPlanMode(!snap.planMode)} />`, and retitle it 'Plan before changes' so it stops fighting the session/task vocabulary. Give the Session row a copy button instead of a bare uncopyable uuid.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:210 `<p className="modal__sub">Applies immediately; nothing here is per-task.</p>`. In the same card: Overlays.tsx:227-231 a row titled `Session` showing `{snap.currentId ? `${snap.currentId} · ${snap.model || 'model unset'}` : 'no session yet'}`, and Overlays.tsx:293-299 `<div className="row__t">Plan this session</div>` … `<button className="chipbtn" onClick={() => store.setPlanMode(true)}>enable now</button>`. Every other stateful row in this card is a real switch bound to live state (`aria-checked={snap.streamThinking}` at Overlays.tsx:286, `aria-checked={notifications}` at :272), but this one hardcodes `true` and never reads `snap.planMode` — GeneralCard receives `snap` (Overlays.tsx:196) and uses `snap.currentId`, `snap.connection`, `snap.streamThinking`, but not `snap.planMode`, which does exist (store.ts:1569 `this.setPlanMode(!this.frame.planMode)`).
```

**Verifier:** Confirmed by reading the file. Overlays.tsx:210 is verbatim `<p className="modal__sub">Applies immediately; nothing here is per-task.</p>`, and Overlays.tsx:293-299 renders a `Plan this session` row whose only control is `<button className="chipbtn" onClick={() => store.setPlanMode(true)}>enable now</button>` — hardcoded `true`, rendered unconditionally, never reading `snap.planMode`. A grep of the renderer shows GeneralCard (Overlays.tsx:196-302) never references planMode, while `snap.planMode` exists and is kept live (store.ts:266, :756, :3308 from status_update.plan_mode) and is consumed everywhere else (App.tsx:838, :1644, Workspaces.tsx:115, Overlays.tsx:1357 palette). The two sibling rows in the same card DO bind state (aria-checked={snap.streamThinking} at :286, aria-checked={notifications} at :275), so the divergence is not a house pattern. store.setPlanMode (store.ts:1565-1584) ends in `this.patch({ planMode: next })`, which is a no-op when plan mode is already on, and the label is static — clicking while armed changes nothing on screen and offers no feedback, and there is no way to disable from Settings. Refutation attempts all failed: (a) "task != session" is wrong — this file uses task as a synonym for session ('New task' at :1408, '/compact — compact this task now' at :1371, 'start a task first — a preset binds to an existing session' at :398) and setPlanMode dispatches with session_key (store.ts:1568), so the subtitle is contradicted by the row two below it and by the Session row at :229-236; (b) you cannot read the live state behind the modal — app.css:1078-1081 .backdrop is fixed inset:0 rgba(0,0,0,.6) and the dialog is aria-modal (Overlays.tsx:46), so the composer chip at App.tsx:1644 is dimmed and inert, and the header chip at App.tsx:838 is the one atelier.css already hides; (c) no comment anywhere documents a deliberate one-way action, and store.ts:1564 calls plan mode a persistent "read-only ceiling", not a fire-and-forget action like reconnect/manage. Severity stays medium rather than high: the click is idempotent and harmless, plan mode is correctly armed when off, and state-aware toggles do exist in the composer and command palette — the damage is misinformation at the exact place a user goes to verify a safety ceiling.
</details>

### [dead-ui] ChangesTab is dead code: snap.tab can never be 'changes' (no setTab('changes') anywhere), so the per-file undo / 'Undo all' / 'Keep all' controls and the changes.undo RPC are unreachable - the reachable ReviewPanel offers only snapshot restore, never a one-click undo of the agent's edits

**`xerxes/src/desktop/renderer/App.tsx:889`**

*Impact* — In an agentic coding tool, the user can never revert what the agent just wrote from the UI. ~170 lines of finished diff-review UI (per-file hunks, +/- totals, "Keep all", "Undo all", per-file "undo") plus a working daemon RPC (`changes.undo`) ship in the binary and are impossible to reach by any click or keystroke. The only recovery paths left are the `/undo` slash command (discoverable only inside the command palette, which is itself undiscoverable — see the next finding) or git.

*Fix* — Pick one and delete the other. Either (a) add a fourth tab button next to Conversation/Plan/Log — `<button className={...} onClick={() => store.setTab('changes')}>Changes{totals.adds||totals.dels ? <span className="pillcount">+{totals.adds} −{totals.dels}</span> : null}</button>` — which also gives App.tsx:818-821's `totals` useMemo (currently computed and never read) a consumer; or (b) port the three controls into ReviewPanel's toolbar at DesktopPanels.tsx:989 (`Undo all` -> store.undoChanges(null), `Keep all` -> store.ackChanges(), and a per-file `undo` in the file nav at DesktopPanels.tsx:1019), then delete ChangesTab, the 'changes' member of WorkspaceTab (types.ts:210) and App.tsx:889.

<details><summary>Evidence & verification</summary>

```
App.tsx:889  {snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}

The tab bar right above it only emits three tabs (App.tsx:876-880):
  onClick={() => store.setTab('activity')}  /  store.setTab('plan')  /  store.setTab('log')

Grep for every setTab call in the renderer: App.tsx:876, 877, 880 ('activity'|'plan'|'log'), App.tsx:1007 ('plan'), App.tsx:1676 ('plan'), DesktopPanels.tsx:454 (setTab(name) over the same three-name rail list). 'changes' is declared in types.ts:210 as part of WorkspaceTab but is never passed anywhere.

Consequence chain — ChangesTab (Workspaces.tsx:22) is the ONLY caller of the undo/keep actions:
  Workspaces.tsx:52   <button className="btn" onClick={() => store.ackChanges()}>Keep all</button>
  Workspaces.tsx:56   onClick={() => { void store.undoChanges(null) }}   // "Undo all"
  Workspaces.tsx:85   onClick={() => { void store.undoChanges(file.path) }}  // per-file "undo"
and store.ts:1906 `async undoChanges(path: string | null)` / store.ts:3451 `ackChanges()` have no other caller in any .tsx file.

The reachable Changes surface (ReviewPanel, opened by Topbar App.tsx:205 -> open('review')) has no undo at all — its toolbar is DesktopPanels.tsx:989-1010: "Snapshots & restore", "Refresh changes", "Load more new files".
```

**Verifier:** Confirmed at source. App.tsx:889 renders ChangesTab only when snap.tab === 'changes', and that state is unreachable: store.setTab (store.ts:1592) is the sole writer of `tab`, and every call site passes 'activity' | 'plan' | 'log' (App.tsx:876, 877, 880, 1007, 1676; DesktopPanels.tsx:454 is an unrelated local useState). The other writes are patch({tab:'activity'}) at store.ts:1968/3561 and the initial 'activity' at 762. ChangesTab (Workspaces.tsx:22) is the only caller of store.undoChanges (store.ts:1906) and store.ackChanges (store.ts:3459), so 'Keep all', 'Undo all' and per-file 'undo' plus the changes.undo RPC ship unreachable, and App.tsx:818-821's totals useMemo is computed for a component that never mounts. Git confirms an unfinished migration rather than a design decision: fc487d0d removed the `setTab('changes')` tab button and added the topbar open('review') button, but left the branch, component and the 'changes' member of WorkspaceTab (types.ts:210) behind, and store.ts:2580's comment still reasons about keeping the user on the Changes tab. BUT the impact claim is overstated: reverting agent edits IS reachable from the UI. The desktop spawns the CLI daemon, and cli.ts:1126 sets autoSnapshotTurns: true, so server.ts:1340 captures a snapshot before every turn; ReviewPanel's first toolbar button (DesktopPanels.tsx:990) opens SnapshotsPanel (1415), which does a confirmed per-file snapshot.restoreFile (1515-1520). The text fallback is also mis-named in the claim: it is `/undo-edits <path|--all> --confirm` (server.ts:5400, bridge/commands.ts:136), and it is surfaced by the composer's live slash completions (App.tsx:1496-1512), not only by the command palette. So: dead code plus a missing one-click undo affordance, not an impossible revert. Fix stands as proposed - preferably (b): add 'Undo all' / 'Keep all' to the ReviewPanel toolbar at DesktopPanels.tsx:989 and a per-file undo in the file nav at 1019, then delete ChangesTab, App.tsx:889 and the 'changes' member of WorkspaceTab.
</details>

### [dead-ui] The ⌘K command palette has no visible affordance: atelier.css:474 hides all four composer keyboard hints, and App.tsx:1777 is the only palette trigger (no Topbar button, no Electron menu item at main.ts:694). Impact is narrower than claimed — slash commands (/compact, /undo) are reachable by typing `/` in the composer (hints.ts:24 + App.tsx:1540), session search and model switching have visible buttons, and esc/stop has the Stop button at App.tsx:856 — so the real loss is palette discoverability plus ⇧⏎-newline, which has no alternative cue.

**`xerxes/src/desktop/renderer/atelier.css:474`**

*Impact* — The command palette — which is the sole UI for /compact, every daemon slash command incl. /undo, switching provider, switching model without the chip, and "Search sessions & messages" — has zero visible affordance in the running app. So do "⇧⏎ for newline" and "esc stops the turn". A new user has no way to learn any of them; they only exist in source.

*Fix* — Delete atelier.css:474 and style the hints instead (they already sit in the flex row defined on atelier.css:473 — e.g. `.atelier .composer__hints>span{display:inline-flex;align-items:center;gap:4px;color:var(--x-meta)}`). At minimum keep the ⌘K and esc spans. Independently, add a palette trigger to the Topbar next to the search button at App.tsx:202 — `<button title="Command palette (⌘K)" aria-label="Command palette" onClick={() => store.togglePalette()}><Icon name="tools" /></button>` — so the feature does not depend on a CSS rule for its discoverability.

<details><summary>Evidence & verification</summary>

```
atelier.css:474  .atelier .composer__hints>span{display:none}
(atelier.css is the last stylesheet in index.html:12-14, so it wins.)

The direct `<span>` children of `.composer__hints` are exactly the four keyboard hints (App.tsx:1653-1656):
  1653  <span><kbd>⏎</kbd> {snap.turnActive ? 'queue' : 'send'}</span>
  1654  <span><kbd>⇧⏎</kbd> newline</span>
  1655  <span><kbd>esc</kbd> stop / clear</span>
  1656  <span><kbd>⌘K</kbd> palette</span>
The siblings that survive are all `<button>` (Permissions, Plan/Work, workspace chip), so the rule removes every hint and nothing else.

And ⌘K is the only entry point: grep for togglePalette/openPalette across all .tsx gives one hit — App.tsx:1778, inside GlobalKeys. The Topbar (App.tsx:197-207) offers Search, BackgroundIndicator, Changes, Files — no palette. CommandPalette (Overlays.tsx:1342) is mounted only by `{snap.paletteOpen && ...}` at App.tsx:178.
```

**Verifier:** CSS FACT — CONFIRMED. atelier.css:474 is exactly `.atelier .composer__hints>span{display:none}`. `.atelier` is on the root div unconditionally (App.tsx:163: `className={`app atelier${...}`}`), and atelier.css is the last of three stylesheets in renderer/index.html:12-14, so nothing overrides it (app.css:605 only sets `.composer__hints`, never the spans). The four direct `<span>` children at App.tsx:1653-1656 (⏎ send/queue, ⇧⏎ newline, esc stop/clear, ⌘K palette) are the only direct spans; the surviving siblings are all `<button className="cchip">`. So all four hints are invisible. There is no comment anywhere near atelier.css:438-480 documenting the intent, and no test or doc referencing it — this is not a documented design decision.

PALETTE ENTRY POINT — CONFIRMED. Grepping `paletteOpen|togglePalette|openPalette` across the entire desktop tree (renderer/, main.ts, preload.ts, main/) yields exactly one caller: App.tsx:1777-1779 inside GlobalKeys (`meta && key==='k'`). Topbar (App.tsx:196-207) has only Search, BackgroundIndicator, Changes, Files. The Electron application menu (main.ts:694-702) is `appMenu`, a File menu containing only New Window / Open Workspace… / Close, plus stock `editMenu`/`viewMenu`/`windowMenu` roles — no palette item, no ⌘K accelerator. So ⌘K really is the sole entry point and it really is undocumented on screen.

IMPACT — MATERIALLY OVERSTATED, hence the downgrade. The claim that the palette is "the sole UI for /compact, every daemon slash command incl. /undo, switching provider, switching model without the chip, and Search sessions" is wrong on every item I checked:
  • Slash commands are NOT palette-only. hints.ts:24-30 `wantsHints()` returns true for a bare `/` (text starts with `/`, contains no whitespace). App.tsx:1498-1512 debounces `store.completeText(draft)` and App.tsx:1540-1560 renders a `.hints` listbox headed "Commands & skills" with the daemon's own catalog, pickable by mouse or keyboard. store.ts:907-964 then routes any `/`-prefixed draft to the daemon's `slash` RPC. Typing `/` in the composer — the universal convention — exposes /compact, /undo and the whole catalog.
  • Session search has a visible Topbar button (App.tsx:202, `Search sessions` → `store.openSessionSearch()`).
  • Model/reasoning switching has a visible composer chip (App.tsx:1612-1620, `.chipanchor > .cchip`, styled and not hidden — atelier.css:472 explicitly styles `.atelier .composer__bar .cchip`).
  • "esc stops the turn" has a visible alternative: the Stop button at App.tsx:851 and 856 inside `.chat__state`, which atelier.css:522 keeps visible (the hide rules at 439-440 target `.chat__head>.hchip`/`.chipanchor` and non-status badges only).

What genuinely survives: the ⌘K command palette — the only fuzzy-jump surface over sessions plus commands — has zero visible affordance anywhere in the app, and ⇧⏎-for-newline is the one hint with no alternative cue at all (a user wanting a newline presses ⏎ and sends the message instead). That is a real discoverability defect worth fixing, but it is an onboarding/polish gap, not a capability the user cannot otherwise reach. Medium, not high.

FIX (revised): the proposed CSS restoration is fine, but the durable fix is the Topbar trigger, since discoverability should not hinge on a skin rule — add next to App.tsx:202 a `<button title="Command palette (⌘K)" aria-label="Command palette" onClick={() => store.togglePalette()}><Icon name="tools" /></button>`, and add a matching View-menu item with a `CmdOrCtrl+K` accelerator in the template at main.ts:694 so the shortcut is self-documenting in the menu bar. Separately, un-hide at minimum the ⇧⏎ span, which has no other affordance.
</details>

### [dead-ui] ApprovalCard unconditionally advertises 1/2/3 but never takes focus, so the shortcut only fires when focus happens to be off the composer textarea — after the normal type→send flow it appends a digit to the draft instead, and when focus is on body a stray digit approves a tool with no confirmation

**`xerxes/src/desktop/renderer/App.tsx:1846`**

*Impact* — In the normal flow (type in the composer -> ⏎ -> agent runs -> approval lands) focus is still in the textarea, so pressing 1 does not approve — it appends "1" to the user's next draft. The reviewer stares at a card that says "1 / 2 / 3", presses 1, nothing happens, and silently corrupts the draft. Keyboard-only and screen-reader users additionally get no focus moved to the alertdialog at all.

*Fix* — Move focus to the card when an approval appears and keep the shortcut usable: in ApprovalCard add `const ref = useRef<HTMLDivElement>(null); useEffect(() => { ref.current?.querySelector('button')?.focus() }, [approval.id])` on the `.approval` div (App.tsx:1150). That both satisfies the alertdialog role and takes focus off the textarea so the App.tsx:1846 guard stops firing. Apply the same to QuestionCard (App.tsx:1201). If you would rather not steal focus, instead render the `1 / 2 / 3` hint conditionally on `document.activeElement` not being a field, so the app never promises a key it will not honour.

<details><summary>Evidence & verification</summary>

```
App.tsx:1154  <span className="approval__keys">1 / 2 / 3</span>
App.tsx:1158-1160  <button ...>Allow once <kbd>1</kbd></button> / This session <kbd>2</kbd> / Deny <kbd>3</kbd>

The handler that would honour them bails whenever a text field owns focus:
App.tsx:1846  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
App.tsx:1847-1849  if (event.key === '1') store.approve(...)  ...

The composer IS a textarea (App.tsx:1574-1576 `<textarea ref={ref} className="composer__input" ...>`) and it is explicitly focused on session entry (App.tsx:1444 `ref.current?.focus()`) and after context/command insertion (App.tsx:1447, 1448). Nothing ever blurs it: grep for `.blur()` across the whole renderer returns zero hits. ApprovalCard (App.tsx:1148-1165) has no focus management either — no `useDialogFocus`, no `autoFocus`, despite `role="alertdialog"`.

QuestionCard carries the identical dead hint: App.tsx:1230 and App.tsx:1244 use the same guard plus `|| target.tagName === 'BUTTON'`.
```

**Verifier:** Every cited line checks out. App.tsx:1846 is `if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return`, placed before the 1/2/3 approve branches (1847-1849) and returning without preventDefault, so the keystroke falls through to the focused field. ApprovalCard (App.tsx:1148-1165) is a hook-free component with role="alertdialog" that unconditionally renders the "1 / 2 / 3" hint (1154) plus three <kbd> badges (1158-1160) and has no ref/autoFocus/useDialogFocus — even though dialogFocus.ts exists and is used by Settings, SearchPanel, the DesktopPanels sheet and Setup. app.css:540 styles .approval__keys visibly and atelier.css has no .approval override, so the hint genuinely renders. The composer textarea (App.tsx:1574-1576) is never disabled (the disabled props in App.tsx cover the send button at 1629, not .composer__input), is explicitly focused at 1444/1447/1448/1520, and grep for `.blur()` across the whole renderer returns zero hits — so in the type→⏎→approval-lands flow the textarea still owns focus and pressing 1 appends a digit to the draft. Not a documented tradeoff: the only comment (1845) documents the guard, and QuestionCard's own comment at 1289-1290 ("no kbd badge on later questions would advertise a dead key") shows the codebase explicitly rejects advertising unhonoured keys. Two things the claim overstates, hence medium not high: (a) the shortcut is not universally dead — clicking any non-focusable area (the transcript, the card padding) drops focus to body and then 1/2/3 do fire, so the behaviour is unpredictable rather than absent, and the inverse hazard is real (with focus on body a stray "1" fires allow_once with no confirmation); (b) the three buttons remain visible and clickable, so nobody is blocked from approving — the cost is a broken keyboard promise, a corrupted draft, and an alertdialog that never takes focus for keyboard/screen-reader users.
</details>

### [dead-ui] Search hits (SearchPanel.tsx:49,114) and command-palette session rows (Overlays.tsx:1404) stay disabled mid-turn on a comment's claim that "openSession silently no-ops mid-turn", which store.ts:1088-1093 stopped being true on 2026-09-19 — it now opens the session in a second window, which the sidebar (App.tsx:707) already advertises and uses

**`xerxes/src/desktop/renderer/SearchPanel.tsx:49`**

*Impact* — Exactly when a long task is running — the moment you want to go read what a previous session decided — full-text search greys out every hit ("finish or stop the running task before switching sessions") and the palette drops all session rows, while the sidebar next to it happily opens the same session in a second window. Two surfaces refuse an operation the third performs.

*Fix* — Delete `locked` from SearchPanel.tsx:47, the early return at :49 and `disabled={locked}` at :114, and set the title to the sidebar's wording ("Open this session while the current task continues"). In Overlays.tsx:1404, drop `!snap.turnActive` from the session-row gate (keep `snap.connection === 'online'`) and change the 'new' entry to stay gated, since openTaskModal (store.ts:1780-1781) really does refuse mid-turn.

<details><summary>Evidence & verification</summary>

```
SearchPanel.tsx:47-52
  const locked = snap.turnActive
  const open = (sessionId: string): void => {
    if (locked) return // openSession silently no-ops mid-turn — same as the sidebar
    ...
SearchPanel.tsx:114  disabled={locked}   // on every result row

That comment is stale. store.ts:1086-1093 openSessionNow:
  if (this.frame.turnActive) {
    if (id === this.frame.currentId) return false
    if (!this.bridge.openWorkspaceWindow) throw new Error(...)
    await this.bridge.openWorkspaceWindow(row?.cwd || this.frame.cwd, id)
Mid-turn it opens the session in its own window. The sidebar already relies on this: SessionCell has no `disabled` (App.tsx:707-711) and its tooltip reads "Open this session while the current task continues" (App.tsx:709, 697-698).

The command palette repeats the same stale assumption — Overlays.tsx:1402-1404 comments "openSession silently no-ops mid-turn" and gates the whole session-row block behind `if (!snap.turnActive && snap.connection === 'online')`, so mid-turn the palette lists no sessions at all.
```

**Verifier:** Verified line-by-line. SearchPanel.tsx:48-52 and :114 say exactly what is claimed, including the comment "openSession silently no-ops mid-turn". store.ts:1088-1093 refutes that premise: mid-turn, openSessionNow no-ops ONLY when id === currentId; for any other id it calls bridge.openWorkspaceWindow(row?.cwd || frame.cwd, id) and opens the session in a second window. That path is fully wired (preload.ts:125-128 -> IPC 'desktop:new-window' -> main.ts:617), and the sidebar deliberately relies on it: SessionCell (App.tsx:690-711) has no disabled prop and its tooltip/doc-comment read "Open this session while the current task continues". The state is reachable: store.ts:2227 openSessionSearch() has no turnActive guard, so the panel opens mid-turn and every hit renders greyed out with "finish or stop the running task before switching sessions". Overlays.tsx:1402-1404 repeats the same stale comment and gates the whole session-row block behind !snap.turnActive, so the command palette lists zero sessions mid-turn. Git history rules out deliberate design: the mid-turn window branch landed in a45e83ae (2026-09-19) while SearchPanel.tsx was last touched 2026-09-13 (fc487d0d) — the comment describes pre-9/19 store behavior. The fix's carve-out is also correct: store.ts:1798-1799 openTaskModal() genuinely early-returns when turnActive, so the palette's 'new' entry should stay gated. One caveat the fix must absorb: search hits carry no cwd field, so row is undefined for sessions outside frame.sessions/live and store.ts:1092 falls back to frame.cwd, opening the window scoped to the wrong workspace — the same mis-scoping already happens out-of-turn at store.ts:1094, so it is not grounds to keep the lock, but the hit shape should carry cwd. Severity is not overstated, though the blast radius is friction and inconsistency, not data loss.
</details>

### [dead-ui] ⌘N (App.tsx:1783) runs newChat() — it silently no-ops mid-turn with no press-time feedback (store.ts:1161 bare return), and it opens a bare session rather than the New-task modal that Overlays.tsx:1409, App.tsx:310 and store.ts:313 all advertise as the ⌘N surface

**`xerxes/src/desktop/renderer/App.tsx:1783`**

*Impact* — Press ⌘N while the agent is working — the single most likely moment to want a second task — and absolutely nothing happens: no window, no toast, no disabled cue, because the keyboard path has no visual counterpart. And when it does work it does not do what the palette says it does.

*Fix* — Make the shortcut agree with its own documentation and never fail silently: at App.tsx:1783 call `store.openTaskModal()` instead of `store.newChat()` (it is already guarded at store.ts:1780-1781), and give the refusal a voice — in openTaskModal's guard, push a notice instead of a bare `return`, e.g. `this.builder.push('notification', { severity: 'info', message: 'Finish or stop the running task, or open a new window (⌘⇧N), to start another task.' }); this.notify()`. Same treatment for beginFreshTask's early return at store.ts:1161.

<details><summary>Evidence & verification</summary>

```
App.tsx:1781-1784
  if (meta && event.key.toLowerCase() === 'n') {
    event.preventDefault()
    store.newChat()
    return
  }
store.newChat() -> store.ts:1160-1161 beginFreshTask:
  if (this.openingSession || this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)
No patch, no notification, no error block — the promise is dropped by `void this.beginFreshTask()` at the newChat body.

The sidebar button for the same action IS gated (App.tsx:621 `disabled={!online || snap.turnActive}`) and its tooltip still advertises the key: App.tsx:623 `... (⌘N)`.

Separately, the palette lists `{ id: 'new', icon: '＋', label: 'New task', hint: '⌘N', run: () => store.openTaskModal() }` (Overlays.tsx:1405-1411) — so the app tells the user ⌘N opens the New-task modal (preset, model, plan-first), while ⌘N actually creates a bare blank session with none of those choices.
```

**Verifier:** All cited lines read and confirmed. App.tsx:1781-1784 calls store.newChat() with no turnActive/offline guard; store.ts:1160-1161 beginFreshTask returns Promise.resolve(false) with no error patch and no notification, and newChat drops the promise (store.ts:1143-1144), so the keypress is unobservable. The other two surfaces for the same action DO cue the refusal — App.tsx:621 disables the sidebar button, and Overlays.tsx:1403-1404 hides the palette entry behind an explicit comment saying "newChat too — offering them as runnable actions would just close the palette over a dead click" — which proves the silence is a missed surface, not a policy. The divergence half is stronger than claimed: Overlays.tsx:1405-1411 (hint '⌘N' -> openTaskModal), App.tsx:310 ("⌘N: describe the outcome, optionally arm the plan ceiling, start") and store.ts:313 ("The ⌘N new-task modal (mockup 18)") all declare ⌘N as the task modal, while the handler starts a bare session with no preset/model/plan-first. openTaskModal is already guarded (store.ts:1798-1799), so the proposed swap is safe. No CmdOrCtrl+N accelerator exists in the main process (only main.ts:697 New Window ⌘⇧N and 698 ⌘⇧O), so nothing downstream rescues it, and no comment defends the current binding. Sole mitigation: the greyed sidebar button carrying a "⌘ N" kbd chip is statically visible, so "absolutely nothing happens" is mildly overstated — there is a passive cue, just no press-triggered feedback. Severity stays medium: no data loss and ⌘⇧N is a workaround, but the app's most-used shortcut contradicts its own documentation and fails silently at the exact moment users reach for it.
</details>

### [errors] First-run setup step 2 (Setup.tsx:70) hard-codes "Connecting to the shared runtime…" for every non-online connection because setupReadiness.ts:7 ignores snap.error — so authentication/validation/wrong-workspace failures, which store.ts:2435 deliberately never retries, are shown as an in-progress connection whose only button (Retry) is futile by connectionFailure.ts:4's own contract, while the Offline card carrying the real message (App.tsx:1046-1057) is dimmed and click-blocked under the z-index:150 backdrop (atelier.css:131); escapable only via "Later"/Escape.

**`xerxes/src/desktop/renderer/Setup.tsx:70`**

*Impact* — A brand-new user whose daemon refuses to bind (bad runtime path, socket permission denied, protocol/build mismatch, workspace rejected) sits on a step-2 spinner message that claims it is still connecting. The actual reason — `snap.error`, rendered by the Offline card (App.tsx:1052) and the connection banner (App.tsx:887) — is painted underneath a `z-index:150` opaque backdrop, so it is literally unreachable. "Retry connection" re-runs the same rejected call with no visible change, and "Start working →" stays disabled forever. The only way out of the app's own onboarding is to press Escape/"Later" and discover the error by accident.

*Fix* — Pass `snap.error` and `connectionFailureKind(snap.error)` into the step-2 body: render the real message plus a kind-specific next step (transport → "Retry connection"; configuration → "Open workspace settings"; session → "Start a new session here"). Only show "Connecting to the shared runtime…" while `snap.connection === 'connecting'`; when it is `'offline'` say "Could not reach the runtime" and show the reason. Also un-disable "Start working →" once the workspace exists so onboarding is never a hard trap.

<details><summary>Evidence & verification</summary>

```
Setup.tsx:70  <p>{state.runtime ? 'Connected and ready.' : 'Connecting to the shared runtime…'}</p>
Setup.tsx:72  {!state.runtime && (<button onClick={() => store.retryConnection()}>Retry connection</button>)}
Setup.tsx:91  <button className="studio-primary" disabled={!state.ready} onClick={finish}>Start working →</button>
setupReadiness.ts: runtime: snap.connection === 'online'  // the ONLY input; snap.error is never read
atelier.css:131  .setup-backdrop{position:fixed;inset:0;z-index:150;background:#0008;display:grid;place-items:center;padding:24px}
```

**Verifier:** All cited lines read and verified. Setup.tsx:70 is verbatim; setupReadiness.ts:7,9 derives `runtime`/`ready` solely from `snap.connection === 'online'` and never reads `snap.error`, so every non-online state — including a permanently rejected one — renders the same "Connecting to the shared runtime…" line with "Start working →" disabled (Setup.tsx:91) and the model/permissions buttons disabled too (Setup.tsx:79,88). The state is reachable on first run: once a workspace is saved, store.ts:827-831 bypasses the gate and initializeLive's failure path (store.ts:854-860 → wentOffline, store.ts:2396-2418) sets `error` + `connection:'offline'` while Setup.tsx:14 `shown` stays true, and the card sits over the Offline diagnosis (App.tsx:925,1046-1057) under .setup-backdrop z-index:150 (atelier.css:131). Three parts of the claim are overstated and I corrected them: (a) the backdrop is #0008 (~53% alpha), so the error underneath is dimmed/click-blocked and occluded by the 520px card, not "literally unreachable"; (b) it is not a hard trap — Setup.tsx:58 renders a visible "Later" button and Setup.tsx:39 binds Escape to finish(); (c) for `transport` failures the copy is defensible because store.ts:2435-2436 genuinely auto-retries every 5s (HEARTBEAT_MS=5_000). The surviving, sharper defect: for `configuration`/`session` kinds (connectionFailure.ts:6-7 — authentication/unauthorized/validation error/agent-preset/wrong-workspace) beat() deliberately stops retrying, yet step 2 still claims it is connecting and its only affordance re-issues a call that connectionFailure.ts:4's own comment says "cannot repair its input" — and the state flips connecting→offline with identical text, so the retry produces no visible change. Misleading-but-escapable onboarding copy that hides an actionable error: medium, not high.
</details>

### [errors] App.tsx:887's inline offline banner is the one connection surface that never checks connectionFailureKind: it claims "Connection lost. Retrying…" even for configuration/session rejections that store.ts:2435's beat() explicitly refuses to retry, and its Retry button resends the same resume_session_id through a self-heal path (store.ts:884-894) that only covers transcript_generation conflicts

**`xerxes/src/desktop/renderer/App.tsx:887`**

*Impact* — When initialize is rejected for a `configuration` or `session` reason (auth/credentials/validation/agent-preset/"belongs to another workspace"), the banner asserts "Connection lost. Retrying…" while `beat()` explicitly refuses to retry — the user watches a false progress claim indefinitely. Both surfaces then offer exactly one action, Retry, which `retryConnection()` fires with the same `resume_session_id` (store.ts:2362-2364) and which connectionFailure.ts's own header comment says cannot possibly repair the input. There is no "Open settings", no "Choose another folder", no "Start a new session here" — a dead end with a decoy button.

*Fix* — Branch both surfaces on `connectionFailureKind(snap.error)`. Only say "Retrying…" for `'transport'`; for `'configuration'` say "The runtime rejected this workspace" and offer "Open workspace settings" / "Open provider settings"; for `'session'` say the session lives elsewhere and offer "Start a new session here" (reuse the self-heal already written at store.ts:848-850, which mints a fresh `sessionKey`) plus "Open its workspace". Disable or relabel "Retry now" for the non-transport kinds.

<details><summary>Evidence & verification</summary>

```
App.tsx:887  <strong>{snap.connection === 'connecting' ? 'Reconnecting…' : 'Connection lost. Retrying…'}</strong>{snap.error && <span>{snap.error}</span>}<button onClick={() => store.retryConnection()}>Retry now</button>
store.ts:2417 (beat)  if (connectionFailureKind(this.frame.error) !== 'transport') return   // no auto-retry
connectionFailure.ts:4  /** A rejected RPC proves the transport worked; retrying it cannot repair its input. */
App.tsx:1051-1056 (Offline)  'Session belongs to another workspace' : 'Could not open this workspace' … <button className="btn" onClick={() => store.retryConnection()}>↻ Retry now</button>   // the only action offered
```

**Verifier:** Core mechanism verified. App.tsx:887 is exactly as quoted and is the ONLY offline surface that never consults connectionFailureKind, while store.ts:2435 in beat() returns early for any non-'transport' error, so "Connection lost. Retrying…" is a false progress claim that persists indefinitely on auth/credentials/validation/agent-preset/wrong-workspace rejections. The claim that Retry cannot help is also correct and stronger than stated: retryConnection (store.ts:2373-2389) resends the same resume_session_id through initializeSelfHealing, which (store.ts:884-894) only heals transcript_generation/divergent-append conflicts — the session self-heal the proposed fix cites at store.ts:847-850 lives in initializeLive and is unreachable from Retry. That the same file branches on connectionFailureKind at lines 218, 677 and 1047 shows the banner's omission is an oversight, not a documented design decision. However the IMPACT is substantially overstated: (a) the full-screen Offline at App.tsx:1046-1058 already branches its headline per kind and prints the error, so it does not assert "Retrying…"; (b) "no Open settings, no Choose another folder, a dead end" is false — RuntimeStatus at App.tsx:252 renders an "Open workspace settings" button whenever connection !== 'online' and no daemonWarning, its label at App.tsx:218 already reads "Workspace needs attention" for non-transport kinds, and App.tsx:683 keeps that chip plus Settings and Workspace/folder buttons in the sidebar footer at all times; (c) the banner displays snap.error beside the claim, so the cause is visible. Scope is one inline banner's copy, not a systemic dead end — hence medium, not high.
</details>

### [errors] Composer is dead and uncancellable during the daemon's pre-launch window: Stop is gated on snap.turnActive (App.tsx:851/856/1839, Overlays.tsx:1361) so the cancel path the daemon explicitly built for this window (server.ts:9713-9727, 9975-9990) is unreachable, while submitTrackedTurn awaits an unbounded git snapshot + LLM compaction (server.ts:9967-9973) after turn.submit already returned ok:true (server.ts:2847) — render Stop on `turnActive || submissionPending` and time-box preparingSubmissions (store.ts:916)

**`xerxes/src/desktop/renderer/store.ts:916`**

*Impact* — `turn.submit` resolving `ok:true` is not the turn starting. The flag is cleared only by `turn_begin` (store.ts:3053), an error-severity notification (:3106), `turn_end` (:3355), a transport drop (:2379), or the submit call itself throwing. If the daemon accepts the submit and then stalls before `turn_begin` — session evicted, snapshot prep wedged, provider handshake hanging with no error event — the composer stays disabled at "Sending… preparing the task" forever. Enter is a no-op (App.tsx:1478), the send button is disabled, and there is no Stop because `turnActive` is false. The user's only escape is quitting the app or forcing a reconnect.

*Fix* — Time-box the pending state. Start a timer alongside `preparingSubmissions.set` (say 20s); on expiry patch a dismissible notice — "Still waiting for the runtime to start this turn" — with "Cancel" (calls `turn.cancel` and clears `preparingSubmissions`) and "Retry". Also render Stop whenever `snap.turnActive || snap.submissionPending` so there is always one control that ends the wait.

<details><summary>Evidence & verification</summary>

```
store.ts:916  this.preparingSubmissions.set(sessionKey, optimistic)
store.ts:3574 (frozen)  submissionPending: this.preparingSubmissions.has(String(value.sessionKey ?? this.sessionKey))
App.tsx:1569  {snap.submissionPending && !snap.turnActive && <div className="streamstatus composer-status" role="status">Sending… preparing the task</div>}
App.tsx:1629  disabled={!ready || !draft.trim() || snap.submissionPending}
App.tsx:1478 (send)  if (!draft.trim() || snap.connection !== 'online' || store.getSnapshot().submissionPending || sendingDraft.current) return
App.tsx:851/856  Stop is rendered only inside `snap.turnActive ? …` — it does not exist while merely pending.
```

**Verifier:** Verified at source. store.ts:916 is the claimed `preparingSubmissions.set`; the `finally` at :933 comments that the flag is deliberately held past RPC acceptance until turn_begin/turn_end. The 120s RPC deadline (main/daemon.ts:34,492) cannot save it because daemon/server.ts:2847 returns {ok:true} immediately after a fire-and-forget `void this.submitTrackedTurn(...)`. The pre-launch window is unbounded real work: submitTrackedTurn (server.ts:9967-9973) awaits captureTurnSnapshot (a git snapshot, :1340-1361, no timeout) then autoCompactIfDue (an LLM call) before turn_begin. Crucially, the daemon ALREADY implements cancel for this window — cancelTrackedTurn (:9713-9727) comments "Retain the stop intent while server-side setup (notably compaction) is still awaiting and no runtime controller exists yet", and :9975-9990 emits turn_end{cancelled,unstarted} because "Without this the client waits forever". The desktop renderer simply cannot invoke it: every Stop is gated on snap.turnActive (App.tsx:851, 856, 1837-1839; Overlays.tsx:1360-1362), while store.cancel() (store.ts:1015) calls turn.cancel unconditionally and would work. No timer sweeps the map (beat() at :2421 does nothing relevant). This is not just a missing timeout — it is a wired-up server capability with no client control surface, and it also degrades the ordinary path: during a routine 10-30s auto-compaction the composer is fully disabled behind an uninformative "Sending… preparing the task" with no Stop and no progress. One overstatement to correct: escape is not only quit/reconnect — opening a different session rebinds sessionKey (store.ts:2566,2575) and frees the composer there, though the stalled session re-bricks on return.
</details>

### [errors] loadModels (store.ts:1236-1244, not :1221) reports discovery failures only as a transcript notice — the ModelPicker retry (Overlays.tsx:1065), Settings › Models fetch (Overlays.tsx:577,583) and the New-Task modal all keep saying "no models discovered yet" with no error and no loading state, while TaskModal's lone error slot (App.tsx:433) shows an unrelated stale snap.error because openTaskModal never clears it — even though the sibling loadProviderModels (store.ts:1253-1287) already does per-surface loading+warning state

**`xerxes/src/desktop/renderer/store.ts:1221`**

*Impact* — The comment at store.ts:1219 states the intent exactly — distinguish "no profile" from "provider refused" from "offline" — and then routes that sentence into the transcript, behind the picker popover and behind the New-Task modal's opaque backdrop. The picker's own "↻ fetch from the daemon" retry can therefore fail on every click while the panel keeps saying "no models discovered yet"; a retry affordance with no failure feedback at the point of retry. The TaskModal's one error slot reads `snap.error`, which `loadModels` never writes — and because `openTaskModal` (store.ts:1780-1784) does not clear it, that slot instead shows a stale, unrelated message from some earlier `fail()` (a failed rename, a refused provider switch), since `snap.error` is only cleared on an online transition (store.ts:2375) or on pressing Start (store.ts:1798).

*Fix* — Add `modelsError: string | null` to the snapshot, set it in both `loadModels` branches, clear it on success. Render it inside ModelPicker above the "↻ fetch from the daemon" button and in the TaskModal's Model field, each with a "Manage providers…" link. Independently, clear `error: null` in `openTaskModal` so the modal can never display a stale failure from another flow, and scope modal errors to local component state rather than the global sticky `snap.error`.

<details><summary>Evidence & verification</summary>

```
store.ts:1217-1231 (loadModels)
  if (result.ok === false) {
    // 'no profile', 'provider refused' and 'offline' are not the same state as 'zero models' — say which one happened.
    this.builder.push('notification', { severity: 'error', message: str(result.error) || 'model discovery failed' })
    this.notify(); return }
  …
  .catch(error => this.fail(error))

Callers, all of them overlays: store.ts:1784 (openTaskModal), Overlays.tsx:1065 (ModelPicker retry button), Overlays.tsx:463 (ModelsCard).
Overlays.tsx:1057-1069  {groups.length === 0 && (<div className="mgroup__cap">… 'no models discovered yet'</div>)}{snap.models.length === 0 && (<button className="mrow" onClick={() => store.loadModels(true)}>↻ fetch from the daemon</button>)}  // no error state at all
App.tsx:433 (TaskModal)  <div role="status" className="taskmodal__error">{snap.error}</div>  // wired to the global sticky error, which loadModels never sets
```

**Verifier:** VERIFIED, with three citation corrections.

Confirmed by reading the files:
- The quoted code is real but lives at store.ts:1231-1249, NOT 1221 (line 1221 is `} else {` inside `answerQuestion`, and `loadModels` *begins* at 1231, so the evidence range "1217-1231" is also wrong). `loadModels`'s failure branch is store.ts:1236-1244: the comment "'no profile', 'provider refused' and 'offline' are not the same state as 'zero models' — say which one happened" sits directly above `this.builder.push('notification', {severity:'error', message: str(result.error) || 'model discovery failed'})`, and `.catch(error => this.fail(error))` at 1249. Neither branch writes any snapshot field a picker/modal could read; there is no `modelsError` and no `modelsLoading` anywhere (grep: zero hits in the renderer).
- Every UI entry point is an overlay that covers the transcript: Overlays.tsx:1065 (ModelPicker "↻ fetch from the daemon"), Overlays.tsx:577 and :583 (Settings › Models "fetch models" / "↻ Fetch models"), Overlays.tsx:463 (RemoteProviders `changed` callback), store.ts:1802 (`openTaskModal`). Overlays.tsx:1057-1069 has no error branch at all — after a failed retry the panel still reads "no models discovered yet", and the popover is `position:fixed` anchored over the composer (Overlays.tsx:937-947), i.e. directly over the tail of the stream where the new notice block lands (blocks.ts:306-316 appends it at the end).
- TaskModal's only error slot is `snap.error` (App.tsx:433); `openTaskModal` (store.ts:1797-1802) patches `taskModalOpen` and calls `this.loadModels()` without clearing `error`, and `loadModels` never sets `error`. So that red slot can only ever show an unrelated leftover from `fail()` (store.ts:3565-3573, which sets `error` AND pushes a transcript notice).
- Decisively NOT a deliberate design: the sibling `loadProviderModels` (store.ts:1253-1287) does exactly the proposed fix — per-profile `providerModelLoading` + `providerModelWarnings`, rendered in place at Overlays.tsx:630, 698, 712, 718-721 ("fetching…", warning text). `loadModels` is the odd one out.

Corrections that trim the claim (not enough to refute):
1. The TaskModal/Settings backdrop is `rgba(0,0,0,0.6)` (app.css:1079-1082), i.e. 60% dim, not "opaque"; the ModelPicker's backdrop is fully transparent (`.backdrop--clear`, app.css:1083). The notice is dimmed/occluded rather than literally unrenderable.
2. `snap.error` is cleared in more places than claimed — also at store.ts:761 (blank/new session frame) and store.ts:2598 (every session open), not just the online transition and Start. The stale-error window is "within the current session after a non-connection `fail()`", not permanent.
3. The picker footer already has a "Manage providers…" link (Overlays.tsx:1073), so part of the proposed fix exists; but Settings › Models has no error surface either, so the escape hatch leads to another silent dead end.
4. `!model` disables Start (App.tsx:436), but `model` seeds from `snap.model` (App.tsx:321-323), so the modal is only hard-blocked when no model is set at all (fresh install / broken profile) — precisely the case where discovery is most likely to fail.

Severity stays medium: no data loss and the message does land in the transcript, but a retry button that can fail silently on every click, on the exact surface a new user hits when their provider profile is wrong, is a real dead end.
</details>

### [keyboard] Approval hot-keys 1/2/3 leak through the modal overlays that lack a keydown stopPropagation (Settings, New task, model/reasoning/context menus, the context rail) — the `dialog[open]` guard at App.tsx:1769 matches only OutputViewer's native dialog, and the focus guard at 1846 misses BUTTON, which dialogFocus.ts:13 focuses by default, so one digit silently grants a session-wide permission or denies a tool call the user cannot see, with no undo

**`xerxes/src/desktop/renderer/App.tsx:1769`**

*Impact* — An approval is pending. The user opens the Project files sheet (or Settings, or the ⌘K palette) to check something before deciding, arrows through the file tree — focus is now on a `<button>` (WorkspaceFileTree.tsx:59-65), which GlobalKeys does not guard — and types "3" as part of a path or a filter. The pending tool call is denied instantly, underneath the sheet, with no confirmation and no undo. The same happens with "1" (allow once) and "2" (allow for the whole session), so a stray keystroke can silently grant a tool blanket session permission.

*Fix* — Stop gating on `dialog[open]`. Gate on the store's own overlay flags — the effect at App.tsx:1856 already depends on every one of them: `if (snap.paletteOpen || snap.settingsOpen || snap.searchOpen || snap.taskModalOpen || snap.pickerOpen || snap.reasoningPickerOpen || snap.modelMenuOpen || snap.contextMenuOpen || snap.wsMenuOpen || snap.sessionMenu || snap.panel) return` before the number branch. Widen the focus guard to match App.tsx:1230 (`BUTTON`, `isContentEditable`, `[role="dialog"]` ancestor). And make deny non-instant: require ⇧3 or a second press within 2s for the irreversible option, matching how the card's own Deny button is styled `btn--danger` (App.tsx:1160).

<details><summary>Evidence & verification</summary>

```
App.tsx:1769  `if (event.defaultPrevented || document.querySelector('dialog[open]')) return`
App.tsx:1846  `if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return`
App.tsx:1849  `else if (event.key === '3') store.approve(snap.approval.id, 'deny')`
The ONLY native `<dialog>` in the whole renderer is OutputViewer.tsx:41 (`createPortal(<dialog ref={dialog} className="output-dialog"`). Every other overlay is a plain div: SettingsModal is `<div className="modal" role="dialog">` (Overlays.tsx:46), CommandPalette is `<div className="palette" role="dialog">` (Overlays.tsx:1479), DesktopSheet is `<div className="studio-backdrop">` (DesktopPanels.tsx:174). So the guard on 1769 matches none of them. Note the sibling QuestionCard handler is strictly safer — App.tsx:1230 guards `INPUT || TEXTAREA || BUTTON || isContentEditable` — the weaker guard sits on the destructive path.
```

**Verifier:** The mechanism is real and verified, but the claim's headline ("every overlay") and its cited scenario are both wrong, so it needs restating and a lower severity.

VERIFIED: App.tsx:1769 reads exactly `if (event.defaultPrevented || document.querySelector('dialog[open]')) return`, and `grep -rn "<dialog"` across /Users/erfan/Documents/Projects/Xerxes-Agents/xerxes/src/desktop/renderer/ returns one hit, OutputViewer.tsx:41. App.tsx:1846 guards only INPUT/TEXTAREA while the sibling QuestionCard handler at App.tsx:1230 guards INPUT/TEXTAREA/BUTTON/isContentEditable — the weaker guard is on the destructive branch (1847-1849). dialogFocus.ts:13 focuses `focusable()[0]`, which is a <button> in every non-input dialog, so "a BUTTON has focus under an overlay" is the default state. store.ts:1179-1199 fires `permission_response` immediately, App.tsx:1147-1165 has no confirmation, and grep for `revoke`/`allow_session` finds no UI anywhere to undo an accidental allow_session grant.

REFUTED PARTS: (a) The exact scenario cited — Project files sheet + WorkspaceFileTree buttons — cannot happen. DesktopPanels.tsx:179-182 attaches `onKeyDown={(event) => { event.stopPropagation(); if (event.key === 'Escape') close() }}` to the `.studio-backdrop` wrapper, swallowing ALL keys, not just Escape. The sheet renders inside the React root (main.tsx:23-25, `createRoot(document.getElementById('root'))`, no portal), so React's delegated bubble listener runs at #root and stopPropagation halts the native event before the window listener in GlobalKeys. Every DesktopSheet panel (files, settings sheets, review, extensions, schedules) is immune. (b) "types 3 as part of a path or a filter" is self-refuting — filter/path typing targets an <input>, already guarded at 1846. (c) CommandPalette (Overlays.tsx:1479) and SearchPanel focus an <input>, so ⌘K/search are guarded too.

STILL LEAKING: SettingsModal (Overlays.tsx:44-46, plain `.backdrop` + `.modal` with no keydown handler), TaskModal (App.tsx:336-337, same shape), ModelPicker/ReasoningPicker/ContextMenu (Overlays.tsx:1014, 1116, 1168, 1271 stop propagation for Escape ONLY), and DesktopRail (DesktopPanels.tsx:126-134, no key handler at all). The two full-screen modals fully cover the approval card, so a stray digit there is invisible and unrecoverable.

SEVERITY: medium, not high. It needs three things to coincide — a daemon approval arriving while a full-screen modal/menu is open, focus resting on a button (the default), and a stray digit press outside any text field. The most-used overlay is immune, so the everyday exposure the claim describes does not exist. The consequence is still genuine: "2" grants blanket session permission with no revoke path in the UI, and "3" denies with no undo.

The proposed fix is sound; note it should also cover the DesktopRail (which is a sibling, not a snap flag — it lives in local `rail` state at App.tsx:143/171 and is not in the dep list at App.tsx:1856), and the `snap.panel` flag named in the fix does not exist (the sheet is local `panel` state at App.tsx:181).
</details>

### [keyboard] No find-in-transcript at all: the bare `editMenu` role (main.ts:701) contributes no Find item, findInPage is never called in the main process, and the renderer's only shortcut handler (App.tsx:1770) binds no 'f' — meta+F silently does nothing; the sole search is cross-session FTS that discards hit.messageIndex and is disabled mid-turn (SearchPanel.tsx:48-51)

**`xerxes/src/desktop/main.ts:701`**

*Impact* — This is an agentic coding tool whose transcripts routinely run thousands of lines of tool output, diffs and file dumps. ⌘F — the most reflexive shortcut in any text-bearing app — does absolutely nothing, with no feedback that it did nothing. To find the filename the agent mentioned forty tool calls ago, the user must scroll the whole transcript by mouse wheel. The Edit menu offers no clue that Find is unavailable.

*Fix* — Add an Edit submenu instead of the bare `editMenu` role: spread the stock roles, then append `{ label: 'Find…', accelerator: 'CmdOrCtrl+F' }`, `{ label: 'Find Next', accelerator: 'CmdOrCtrl+G' }`, `{ label: 'Find Previous', accelerator: 'CmdOrCtrl+Shift+G' }`. Each sends an IPC message to the focused surface's `contents`; implement a small find bar in the renderer that calls back into `contents.findInPage(query, { findNext, forward })` and `stopFindInPage('clearSelection')` on Escape. Give the existing SearchPanel ⌘⇧F so "find in this transcript" and "find across sessions" are adjacent, discoverable shortcuts.

<details><summary>Evidence & verification</summary>

```
main.ts:701  `{ role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },` — Electron's `editMenu` role template contains undo/redo/cut/copy/paste/delete/selectAll/speech and no Find item. `grep -rn "findInPage" main.ts main/*.ts` returns nothing — `webContents.findInPage` is never called. `grep` for any `'f'` key handler across the renderer returns nothing: the only `f`-adjacent binding is the specialist-generate `g` at DesktopPanels.tsx:290. The only search that exists is server-side session search (SearchPanel.tsx:5, "Session search (⌘K → 'Search sessions & messages…')"), which has no accelerator of its own and searches *sessions*, not the transcript on screen.
```

**Verifier:** Verified directly. main.ts:694-701 builds the only application menu and uses the bare `{ role: 'editMenu' }` role, whose stock template has no Find item. grep across main.ts, main/*.ts and preload.ts finds zero occurrences of findInPage/stopFindInPage, zero 'context-menu' handlers, zero before-input-event hooks and zero globalShortcut registrations. In the renderer the only key handler is GlobalKeys (App.tsx:1763-1810), binding meta+',', meta+k, meta+n and Escape — no 'f' branch exists, so meta+F is silently swallowed with no UI. The strongest refutation — that cross-session search covers it — fails on reading SearchPanel.tsx: open() at :47-51 calls store.openSession(hit.sessionId) and discards hit.messageIndex, so it never scrolls to the match; `locked = snap.turnActive` (:48) disables every hit row while a turn runs; it requires >=2 chars and queries the daemon's persisted FTS (server.ts:2256) rather than the live transcript on screen; and it has no accelerator, reachable only via the palette row at Overlays.tsx:1377. Severity corrected high -> medium: nothing is broken, mis-rendered or misleading — this is an unbuilt feature with a partial, awkward alternative — but it is a real daily-use gap in a transcript-heavy tool, and webContents.findInPage makes the fix cheap.
</details>

### [layout] The inspector divider can push `inspectorWidth` past the `contextRequiresFullWidth` threshold (App.tsx:146) that display:none's the divider itself (atelier.css:526) — the oversized width is then persisted to localStorage with no UI able to shrink it, so the Activity/Files rail permanently covers the conversation at that window size (the chat itself is one click back via the rail's x)

**`xerxes/src/desktop/renderer/atelier.css:526`**

*Impact* — On a 1000px-wide window with the sidebar (240) and Activity rail (340) open, dragging the inspector divider from 340 to 441 crosses `windowWidth < 240 + 441 + 320`. The instant it crosses, the whole conversation — transcript, header AND composer — is display:none'd, the rail snaps from 441px to ~759px, and the drag handle the user is holding is display:none'd too. On pointer-up there is no handle left to drag back. Closing the rail and reopening it does not help (inspectorWidth is still 441, so data-context-full re-applies immediately), and the value is persisted to localStorage. The only escapes are resizing the OS window or hiding the sidebar — neither of which the user has any reason to connect to "I dragged a divider 100px".

*Fix* — Stop deriving `data-context-full` from the *dragged* width. Either (a) clamp the divider's `max` to `windowWidth - sidebarWidth - 320` in App.tsx:170 so the drag can never cross the threshold, or (b) keep the divider rendered when `data-context-full` is set — delete the `.panel-divider:last-of-type{display:none}` rule at atelier.css:526 and instead pin the divider to the rail's left edge so the user can always drag back. Also give it a reset affordance (double-click the divider → `setLayout({ inspectorWidth: 340 })`), since the bad value survives a relaunch.

<details><summary>Evidence & verification</summary>

```
atelier.css:524-526
  .atelier .app__body[data-context-full]>.chat{display:none}
  .atelier .app__body[data-context-full]>.desktop-rail{flex:1;width:auto}
  .atelier .app__body[data-context-full]>.panel-divider:last-of-type{display:none}
App.tsx:146  const contextRequiresFullWidth = windowWidth < (focused ? 0 : layout.sidebarWidth) + layout.inspectorWidth + 320
App.tsx:166  <div className="app__body" data-context-full={rail && rail !== 'review' && (filesExpanded || contextRequiresFullWidth) || undefined} ...>
App.tsx:170  {rail && rail !== 'review' && <PanelDivider label="Resize inspector" value={layout.inspectorWidth} min={260} max={520} reverse onChange={inspectorWidth => setLayout({ inspectorWidth })} />}
layout.tsx:25  window.localStorage.setItem(key, JSON.stringify(next))
```

**Verifier:** Verified in source, but the impact is dramatized. Confirmed: atelier.css:524-526 reads exactly as quoted and is never overridden (grep: the rule appears once); `.panel-divider:last-of-type` really does resolve to the inspector divider, since app__body's only div children are the two dividers (sidebar is <aside class="side"> App.tsx:615, chat is <main class="chat"> App.tsx:827, rail is <aside class="desktop-rail"> DesktopPanels.tsx:126); App.tsx:146 derives contextRequiresFullWidth from the very value the divider at App.tsx:170 writes (min 260 / max 520); layout.tsx:23-27 persists it to localStorage; main.ts:143 sets minWidth 760, so the 850-1080px window band is reachable (at a 960px half-screen window the default 340px inspector has only ~60px of drag headroom); App.tsx:171 passes toggleFilesExpanded only when !contextRequiresFullWidth, so the "Restore conversation" button (DesktopPanels.tsx:127) that rescues the filesExpanded case is absent on this path; no setLayout caller anywhere resets the width, and the hidden divider cannot be focused for its Arrow/Home keyboard path (layout.tsx:41). Refuted parts: (1) the conversation is NOT lost — DesktopPanels.tsx:127 always renders the "Close task context" x which calls setRail(null), and data-context-full is gated on `rail` being truthy (App.tsx:166), so transcript, header and composer return in one click; (2) the divider keeps pointer capture while display:none (it stays connected to the document, layout.tsx:38), so dragging back inside the same gesture still works — the trap only closes on pointer-up. The durable defect is the stuck width, not a deleted conversation: the only control that can shrink inspectorWidth is display:none'd exactly when it is needed, and the bad value survives relaunch, so the rail thereafter always covers the chat at that window size; escape requires resizing the OS window or hiding the sidebar. Real and worth fixing (clamp the divider max to windowWidth - sidebarWidth - 320 at App.tsx:170), but one-click recoverable for the alarming part and only reachable in an 850-1080px window.
</details>

### [layout] Opening "Changes" hides the transcript and composer at every window width (atelier.css:492), and it is the only route to working-tree diffs — the in-chat changes tab (App.tsx:889) has no setter, so diff and conversation can never be on screen together

**`xerxes/src/desktop/renderer/atelier.css:492`**

*Impact* — `data-review` hides `.chat` unconditionally — at 900px and at 2560px alike. Files and Activity dock beside the conversation from the same three-button rail nav (DesktopPanels.tsx:127), but Changes, reached by the identical-looking third button, silently deletes the transcript and the composer. The core loop of a coding assistant — read the diff, then type "revert that hunk" — is impossible: the user must close review, retype from memory, and reopen it. App.tsx:170 also skips the PanelDivider for review, so even on a 2560px monitor the diff view cannot be narrowed and its file list gets `minmax(160px,26%)` ≈ 600px of empty gutter.

*Fix* — Treat review like the other rail panels: delete the `[data-review]>.chat{display:none}` rule at atelier.css:492, let `.desktop-rail--review` use `flex:0 0 var(--inspector-width)` with a wider max (e.g. bump the App.tsx:170 `max` to 900 when `rail === 'review'`), and render the PanelDivider for review by dropping the `rail !== 'review'` guard at App.tsx:170. Keep the existing expand button (DesktopPanels.tsx:127) as the opt-in full-width mode so full-screen review stays available but stops being mandatory.

<details><summary>Evidence & verification</summary>

```
atelier.css:492-493
  .atelier .app__body[data-review]>.chat{display:none}
  .atelier .desktop-rail.desktop-rail--review{flex:1;width:auto;min-width:0}
App.tsx:166  data-review={rail === "review" || undefined}
App.tsx:170  {rail && rail !== 'review' && <PanelDivider label="Resize inspector" ... />}   // no divider in review
App.tsx:205  <button title="Working tree changes" onClick={() => open('review')}><Icon name="changes" /><span>Changes</span></button>
App.tsx:894  <Composer snap={snap} />   // child of <main className="chat">
```

**Verifier:** Verified line-by-line. atelier.css:492 `.atelier .app__body[data-review]>.chat{display:none}` is real, unconditional (no enclosing @media/@container; the only nearby query at 406 closes at 408) and always active since `.atelier` is on the root element (App.tsx:163). data-review is set purely by `rail === 'review'` (App.tsx:166), and all three entry points (App.tsx:205, DesktopPanels.tsx:224, Workspaces.tsx:24) funnel through navigate -> setRail('review') (App.tsx:155), so the state is trivially reachable and has no sheet variant. Composer is a child of <main className="chat"> (App.tsx:827, 894), so it is hidden too. The asymmetry is real: the identical `display:none` for files/activity (atelier.css:524) is opt-in via the expand toggle (DesktopPanels.tsx:127, rendered only for files/activity) or forced only when the window is actually too narrow (contextRequiresFullWidth, App.tsx:146); review gets no toggle, no divider (App.tsx:170 guard confirmed) and no width condition. Not a documented decision: git blame points to fc487d0d whose message and DESIGN.md say nothing about full-screen review, while the DesktopRail doc comment at DesktopPanels.tsx:120 states the opposite intent ("the conversation and draft remain interactive"). I also killed the one plausible refutation: the in-chat fallback at App.tsx:889 (`snap.tab === 'changes'`) is dead code -- the tab bar (App.tsx:876-881) offers only Conversation/Plan/Log and `setTab('changes')` has zero callers -- so there is no route to working-tree diffs that keeps the composer. Two overstatements corrected: (a) display:none preserves the React subtree and the store-backed draft, so typed text is NOT lost on close/reopen -- the cost is that diff and transcript are mutually exclusive, not data loss; (b) the proposed fix's "keep the existing expand button as opt-in full-width" does not work as written, since that button is not rendered for review and App.tsx:171 withholds toggleFilesExpanded when contextRequiresFullWidth. Severity stays medium: it breaks the read-diff-then-instruct loop and cannot be worked around, but it is recoverable in one click with no lost work.
</details>

### [navigation] A full-page destination (Agents / Skills &amp; tools / Artifacts) offers no in-page close, and Escape there silently cancels the running turn — DesktopPage's header (DesktopPanels.tsx:114) has no close button, App.tsx:874 hides the workspace tab bar while `page` is set, and `page` is Shell-local state absent from the GlobalKeys Escape ladder, so Escape falls through to `store.cancel()` (App.tsx:1837). The only exit is the sidebar's `Sessions` button, which renders as already-selected for the extensions/artifacts pages (App.tsx:618). Does NOT apply to the rail (default-open by App.tsx:149 and already has a × at DesktopPanels.tsx:127) or the sheet (owns Escape at DesktopPanels.tsx:150-161).

**`xerxes/src/desktop/renderer/App.tsx:1837`**

*Impact* — Three of the four shell surfaces answer Escape differently. A sheet closes. A DesktopPage (Agents / Skills & tools / Artifacts) and the rail do nothing — the key falls through to GlobalKeys and kills the running agent turn. So the user browsing the skills catalog mid-run presses Esc to "go back" and instead cancels the work. And because DesktopPage has no close button either, Escape is the natural thing to try. The root cause is architectural: `panel`/`page`/`rail` are Shell-local useState, so neither GlobalKeys nor the store's overlay bookkeeping can see them.

*Fix* — Move `panel`/`page`/`rail` into the store next to `paletteOpen`/`settingsOpen` (store.ts:773-790) and insert them in the GlobalKeys Escape ladder immediately above the `turnActive` branch, innermost-first (sheet → page → rail → cancel). Delete DesktopSheet's bespoke capture listener once the ladder owns it. Independently, give DesktopPage the same `× / Back to conversation` button its sibling DesktopRail already has (DesktopPanels.tsx:127).

<details><summary>Evidence & verification</summary>

```
App.tsx:1786-1841 GlobalKeys Escape ladder: paletteOpen, searchOpen, taskModalOpen, settingsOpen, pickerOpen, reasoningPickerOpen, modelMenuOpen, contextMenuOpen, wsMenuOpen, sessionMenu → then `if (snap.turnActive) { event.preventDefault(); store.cancel() }`. `panel`, `page` and `rail` are absent from the ladder and from its dep array (line 1856).

App.tsx:135  const [panel, setPanel] = useState<DesktopPanel>(null)
App.tsx:142  const [railChoice, setRail] = useState<…>(undefined)
App.tsx:150  const [page, setPage] = useState<'agents'|'extensions'|'artifacts'|null>(null)

DesktopPanels.tsx:112-117 DesktopPage — header is `<div><h2>…</h2><p>…</p></div>`, no close button, no keydown handler.
DesktopPanels.tsx:150-157 DesktopSheet — the ONLY surface that handles Escape, via a document capture listener with stopImmediatePropagation.
```

**Verifier:** Verified at the cited lines. App.tsx:1786-1841 is exactly the ladder described and ends in `if (snap.turnActive) { preventDefault(); store.cancel() }`; `page` is Shell-local useState (App.tsx:150) and is in neither the ladder nor the dep array (1856). store.ts:1015 `cancel()` fires `turn.cancel` with no confirmation. DesktopPanels.tsx:114 DesktopPage's header truly has no close button, and App.tsx:874 hides the workspace tab bar while a page is open, removing the "Conversation" tab as a back route. The Composer stays mounted (App.tsx:894) and App.tsx:1595-1603 deliberately lets Escape bubble when no hints are open, so the key does reach GlobalKeys. HOWEVER two thirds of the claim are wrong: (1) the rail is default-open, not an overlay (App.tsx:149 defaults railChoice to 'activity' whenever a workspace exists), so adding it to the Escape ladder as proposed would break the documented `Esc stop` contract (App.tsx:1763) in the ordinary case; and it already has an explicit close button (DesktopPanels.tsx:127 `aria-label="Close task context"`). (2) DesktopSheet already handles Escape (150-161), so only one of four surfaces misbehaves, not three. Also, the page is not a trap: the sidebar stays mounted and App.tsx:618 `Sessions` calls open(null) — though that button renders with `is-selected` whenever page !== 'agents', so it looks already-active while Skills & tools or Artifacts is open. Impact is an interrupted turn (draft preserved per the App.tsx:1602 comment, work resendable), not data loss, and Esc=stop is documented — so medium, not high, and the fix is DesktopPage-local rather than the proposed store migration.
</details>

### [navigation] ⌘N is bound to store.newChat() (App.tsx:1783) even though TaskModal's own docstring (App.tsx:309) and the palette row's "⌘N" hint (Overlays.tsx:1410) both promise it opens the new-task modal — leaving the objective/plan-ceiling/preset/model surface reachable from exactly one palette row (its only caller)

**`xerxes/src/desktop/renderer/Overlays.tsx:1409`**

*Impact* — Two things fire from one advertised shortcut: the palette row labelled "New task ⌘N" opens TaskModal (objective field, plan-ceiling toggle, worktree naming — App.tsx:315-400), while actually pressing ⌘N skips it and immediately creates a bare session. A user who learns ⌘N from the palette never sees the modal again, so a whole surface is effectively orphaned. Separately, the ⌘K palette — the only keyboard route in the app — can open Settings tabs but not a single one of the nine `DesktopPanel` destinations, so Changes, Files, Activity, Artifacts, Skills & tools, Scheduled jobs, Snapshots and Workspace are mouse-only.

*Fix* — Pick one ⌘N behaviour: either point the GlobalKeys handler at `store.openTaskModal()` and keep the sidebar button on `newChat()` (labelled "New session"), or drop the '⌘N' hint from the palette row and give TaskModal its own accelerator (⇧⌘N is free; ⌘⇧N is already New Window at main.ts:687, so use ⌘T). Then add one palette action per DesktopPanel value — they all funnel through the single `useDesktopNavigation()` callback once that state lives in the store, e.g. `{ id: 'nav:review', label: 'Changes…', run: () => store.navigate('review') }`.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:1405-1411  { id: 'new', icon: '＋', label: 'New task', hint: '⌘N', run: () => store.openTaskModal() }
App.tsx:1781-1784  if (meta && event.key.toLowerCase() === 'n') { event.preventDefault(); store.newChat(); return }
App.tsx:622-623  onClick={() => { open(null); store.newChat() }}  title="…(⌘N)"
grep openTaskModal → Overlays.tsx:1410 and store.ts:1780 only — one caller.
Palette action list (Overlays.tsx:1351-1441): plan, stop, /goal, /compact, settings×5 tabs, session-search, providers, models, new task, session rows, daemon slash commands. No 'review', 'files', 'activity', 'artifacts', 'agents', 'extensions', 'schedules', 'workspace' or 'snapshots'.
```

**Verifier:** Confirmed at source, and stronger than reported on the first half. Overlays.tsx:1405-1411 is verbatim as quoted: the palette row labelled "New task" with hint "⌘N" runs store.openTaskModal(). App.tsx:1781-1784 binds the real ⌘N to store.newChat(), which via beginFreshTask (store.ts:1143,1158) re-keys straight into a bare session without ever mounting TaskModal. grep for openTaskModal across src/desktop/ returns only the definition (store.ts:1798) and that single palette caller, so TaskModal genuinely has one entry point. The reporter missed the decisive evidence: TaskModal's own docstring at App.tsx:307-313 states "⌘N: describe the outcome, optionally arm the plan ceiling, start," and GlobalKeys' docstring at App.tsx:1762 says "⌘N new task" — so this is a violated documented contract, not a deliberate design split. No menu accelerator compensates (main.ts:697-698 defines only CmdOrCtrl+Shift+N = New Window and CmdOrCtrl+Shift+O). The sidebar button is self-consistent ("New session" + <kbd>⌘ N</kbd> → newChat, App.tsx:620-623), so the lie is localized to the palette hint and the modal's stated contract. The second half is factually true but overstated and does not belong in the same finding: DesktopPanels.tsx:42-52 confirms the nine-value union and none appear in the palette action list (Overlays.tsx:1351-1441), but all destinations are real focusable <button>s in the sidebar (App.tsx:625-627) and topbar (App.tsx:205-207), so they are tab-reachable, not "mouse-only" — it is a missing accelerator, i.e. an enhancement. The proposed fix for it is also heavier than claimed: panel state is local useState in Shell (App.tsx:135,150-159) and is intentionally reset on cwd/sessionKey change (App.tsx:160), so lifting it into the store is a real refactor. Severity therefore drops from high to medium: the shortcut misleads and an entire configured-start surface hangs off one palette row, but nothing is lost or corrupted since newChat does produce a usable session.
</details>

### [navigation] Sheets (.studio-backdrop, atelier.css:69, z-index 120) outrank the command palette (app.css:1239, z-50) and the Settings backdrop (app.css:1081, z-40), and `panel` is component state (App.tsx:135) no store overlay can close — so Cmd-K / Cmd-, over Scheduled jobs / Workspace / Snapshots mounts a focused, typable palette or Settings dialog beneath the sheet's scrim (Enter can blind-run /compact), every mouse click closes the sheet instead (DesktopPanels.tsx:176-178), and Escape is captured by the sheet (DesktopPanels.tsx:150-157) — it self-recovers on a second Escape, but dialogFocus.ts:24 then yanks focus out of the revealed palette.

**`xerxes/src/desktop/renderer/atelier.css:69`**

*Impact* — With Scheduled jobs / Workspace / Snapshots open, ⌘K mounts the palette *underneath* the sheet's translucent scrim: it is dimly visible, the keyboard still drives it (focus moves to its input), but every mouse click lands on the backdrop and closes the sheet instead of running the command, and Escape is swallowed by the sheet's capture listener so it never closes the palette. ⌘, does the same with Settings. The reverse stacking also exists by design: the Changes review is a full-width takeover, and its own toolbar button pops a modal sheet on top of it — two nested "full" surfaces with two different close gestures.

*Fix* — Put every overlay on one z-scale in tokens.ts and make sheets peers of the palette/settings, not superiors (sheet 50, palette 60, settings 60, picker 70, setup 150). Once `panel` is store state, have `openSettings`/`togglePalette`/`openPicker` patch `panel: null` the same way they patch each other (store.ts:1612), so two competing surfaces can never coexist. Replace ReviewPanel's `open('snapshots')` (DesktopPanels.tsx:990) with an in-rail view swap so restore stays inside the review instead of stacking a modal on a takeover.

<details><summary>Evidence & verification</summary>

```
atelier.css:69  .studio-backdrop{position:fixed;inset:0;z-index:120;…background:#0005}
app.css:1239  .palette { … z-index: 50; }
app.css:1081  .backdrop { … z-index: 40; }   // Settings modal
DesktopPanels.tsx:150-157  document.addEventListener('keydown', key, true) → Escape: preventDefault + stopImmediatePropagation + close()
DesktopPanels.tsx:176-178  onMouseDown={(event) => { if (event.target === event.currentTarget) close() }}
store.ts:1612-1618, 1709-1733 — openPicker/openModelMenu/openSettings patch every *other* store overlay closed; `panel` is not store state, so they cannot close a sheet.
DesktopPanels.tsx:990  <button onClick={() => open('snapshots')}>Snapshots & restore</button>   // opens a modal sheet on top of the full-width review takeover
```

**Verifier:** Verified at source. atelier.css:69 literally reads `.studio-backdrop{position:fixed;inset:0;z-index:120;...background:#0005}`; app.css:1239 gives `.palette` z-index 50 and app.css:1081 gives the Settings `.backdrop` z-index 40. A full grep of z-index in both stylesheets shows no later override (atelier.css:65/263/579 restyle .palette but never its z-index), and the palette is NOT portaled into .picker-layer (z-1000) — PickerLayer (Overlays.tsx:925) wraps only ModelMenu/ModelPicker/ReasoningPicker (App.tsx:1621-1625), while App.tsx:178 mounts CommandPalette as a bare sibling of DesktopSheet (App.tsx:182) inside the same .app container, so the 120-vs-50 comparison is real. The state is easily reachable: `panel` is plain useState (App.tsx:135), so store.togglePalette (store.ts:1620) and store.openSettings cannot clear it, and sheets are opened from the sidebar (App.tsx:627 Scheduled jobs, App.tsx:683 Workspace), the chat header (App.tsx:829), the composer chip (App.tsx:1652) and the review toolbar (DesktopPanels.tsx:990 open('snapshots')). GlobalKeys (App.tsx:1769-1780) still fires Cmd-K / Cmd-, over a sheet — it bails only on defaultPrevented or a native `dialog[open]`, and the sheet is a role="dialog" div; DesktopPanels.tsx:150-157 stops propagation only for Escape. The sheet's full-viewport scrim then eats every mouse click (DesktopPanels.tsx:176-178 closes the sheet), while the palette's input is focused and typable even though it is hidden (Overlays.tsx:1349 + dialogFocus.ts:13; the sheet's trap at dialogFocus.ts:14-23 intercepts Tab only), so Enter can blind-run an action such as /compact or the plan-mode toggle (Overlays.tsx:1356-1371). One part of the claim is overstated and I corrected it in the title: Escape is not a dead end — the sheet's capture handler closes the sheet, revealing the palette, so a second Escape dismisses it (with the side effect that dialogFocus.ts:24 then steals focus back out of the palette input). Because the situation self-recovers with one Escape or one click and destroys no data, this is a medium, not a high.
</details>

### [perf] Stream events are never coalesced: store.ts:3095-3100 calls notify() per text_part, notify() (3619) rebuilds the frame and blocks.ts:370 re-copies the whole committed array, and App.tsx:115's single root useSyncExternalStore re-renders the entire unvirtualized, unmemoized transcript once per provider delta (preload.ts:100-113 delivers one IPC macrotask each, so React cannot batch)

**`xerxes/src/desktop/renderer/store.ts:3091`**

*Impact* — At a normal 40-80 tok/s stream the renderer does 40-80 full Shell renders per second, each one allocating a fresh copy of the entire committed-block array. At 5000 blocks that is ~400k element copies/sec plus 5000 BlockView invocations per token — the window drops to single-digit FPS, the composer stops echoing keystrokes, and Stop/approval clicks queue behind the render backlog. This is the single largest cause of the app feeling slower than the TUI on long sessions.

*Fix* — Coalesce emit(): have notify()/patch() mark dirty and schedule one flush per animation frame (`if (!this.flushQueued) { this.flushQueued = true; requestAnimationFrame(() => { this.flushQueued = false; this.rebuildFrame(); this.emit() }) }`), with a synchronous flush before any await in submit()/approve() so tests and user actions still see the latest frame. Separately, stop copying the committed fold: cache the concat in BlockBuilder and invalidate it only when `this.blocks` actually changes (a `blocksRevision` counter), so a token only rebuilds the small `live` tail.

<details><summary>Evidence & verification</summary>

```
store.ts:3087-3093 `case 'text_part': { this.setMetricPhase('llm'); if (...) this.agentText += payload.text; this.builder.push(type, payload); this.notify(); break }` — and notify() (store.ts:3591-3598) does `this.frame = this.frozen({ ...this.frame, blocks: this.builder.snapshot(this.frame.turnActive), log: this.logRing, ... }); this.emit()`, while blocks.ts:370 returns `[...this.blocks, ...live]`. preload.ts:100-112 delivers one IPC message per event (`ipcRenderer.on(EVENT_CHANNEL, listener)` -> handler), so each token is its own task: React 19 cannot batch across them. App.tsx:115 `const snap = useSyncExternalStore(store.subscribe, store.getSnapshot)` is the ONLY subscription, at the root, and the only memo() in the entire renderer is markdown.tsx:207 and Execution.tsx:49.
```

**Verifier:** Confirmed at source, with the cited lines slightly off. store.ts:3095-3100 is the real `case 'text_part'` (the claim said 3091, which is a refreshFleet comment); it calls this.notify() per delta. notify() at store.ts:3619-3627 (not 3591) rebuilds the whole frame via builder.snapshot() and emit() (3634) fires listeners synchronously; patch() (3606) does the same, even for the 1s turnSeconds tick. blocks.ts:358-371 returns `[...this.blocks, ...live]` with no cache. No coalescing exists at any of the four layers: daemon/turnRunner.ts:1533 maps each provider text delta to one text_part, main/ipc.ts:39-42 sends one `daemon:event` IPC message per event, preload.ts:100-113 dispatches each on its own ipcRenderer.on callback (separate macrotasks, so React 19 cannot batch), and App.tsx:115 useSyncExternalStore(store.subscribe, ...) is the sole subscription at the root with store.subscribe (807) unwrapped. Grep for throttle/debounce/requestAnimationFrame/startTransition/useDeferredValue in the renderer finds only the slash-hint debounce and SearchPanel, so no mitigation exists. Stream (App.tsx:901-919) also re-filters all blocks and runs findIndex over them per event, and BlockView (App.tsx:1083) is not memoized; there is no virtualization. However the impact is materially overstated: the only two memo boundaries in the renderer (markdown.tsx:207 Markdown, Execution.tsx:49 ToolCallRow) are exactly the expensive leaves, so committed prose is not re-parsed and tool rows bail out; copying 5000 array pointers 80x/sec is microseconds, not a frame-rate cause, so the "400k element copies/sec" evidence is the weak part of the argument; 5000 blocks is unrealistic since blocks are folded runs (blocks.ts:203-235) and history pages are capped at 100 (store.ts:2529, historyPage.ts:82); and the composer keeps its draft in local useState (App.tsx:1428), so the "composer stops echoing keystrokes" claim is second-order inference. No profile or FPS measurement was taken, and no comment in the code frames the immediate emit as a deliberate latency decision. Real architectural defect that worsens with transcript length; the rAF-coalesce plus cached committed fold is the correct fix, but it is a degradation, not a break.
</details>

### [perf] activityFleetRows re-runs detailOf (JSON.parse + re-stringify) over every tool output in the transcript on every store notify, as a provably idempotent no-op, and discards the result entirely when the session has no subagents

**`xerxes/src/desktop/renderer/AgentRoster.tsx:27`**

*Impact* — `tool.output` is already a pretty-printed JSON string (blocks.ts:242,254), so this re-parses and re-serialises every Read/Bash/Grep result in the transcript on every single render — i.e. per streamed token. A session with 200 tool calls averaging 20 KB of output means ~4 MB of JSON.parse + JSON.stringify per token, on the main thread. The whole map is then discarded unless an `agents` block exists, so in the common no-subagent session it is 100% wasted work and the primary reason long sessions become unusable rather than merely slow.

*Fix* — Pass the already-computed string as the second argument — `toolFailureText({ error: tool.error }, tool.output)` — so `detailOf` never runs; bail out early when the transcript contains no `agents` block (`if (!blocks.some(b => b.kind === 'agents')) return rows`); and wrap the call at App.tsx:1688 in `useMemo(..., [snap.fleet, snap.blocks])`. Also replace the `result.findIndex` inside the member loop (AgentRoster.tsx:32) with a prebuilt id/title index to drop the O(members x rows) inner scan.

<details><summary>Evidence & verification</summary>

```
AgentRoster.tsx:22-27 `export function activityFleetRows(rows, blocks) { const result = [...rows]; const failures = new Map(); for (const block of blocks) { ... if (block.kind === 'tools') for (const tool of block.items) failures.set(tool.id, toolFailureText({error:tool.error,result:tool.output})) ...`. blocks.ts:459 `export function toolFailureText(payload, output = detailOf(payload.return_value ?? payload.result ?? payload.output))` — a default parameter, so it is evaluated unconditionally; blocks.ts:53-58 `detailOf` does `JSON.stringify(JSON.parse(trimmed), null, 2)`. This is called from App.tsx:1688 `const fleet = activityFleetRows(snap.fleet, snap.blocks)` inside ActivityDetails, which App.tsx:150 mounts by default (`const rail = railChoice === undefined ? (!snap.noWorkspace && !contextRequiresFullWidth ? 'activity' : null) : railChoice`).
```

**Verifier:** Every cited line checks out verbatim. AgentRoster.tsx:27 calls toolFailureText({error,result:tool.output}) omitting the second argument, and blocks.ts:459 declares that argument with a default of detailOf(...), which JS evaluates at call time before the body runs — so blocks.ts:53-58 (value.trim() + JSON.stringify(JSON.parse(trimmed),null,2)) executes for every tool item unconditionally. tool.output was already produced by detailOf at blocks.ts:242/254, so this second pass is provably idempotent (I checked both branches: pretty JSON re-parses to the identical string; non-JSON returns value unchanged) — i.e. pure waste, and the proposed fix of passing tool.output as the second argument is exactly semantics-preserving (the payload.permitted branch is already dead from this call site since AgentRoster never passes it). The failures Map is read only at line 30 inside the block.kind === 'agents' branch, so in the common no-subagent session it is built and thrown away. Reachability confirmed: App.tsx:150 defaults rail to 'activity' whenever a workspace exists and the window is wide enough, DesktopPanels.tsx:131 renders activityDetails for that panel, App.tsx:1688 calls activityFleetRows with no useMemo, and neither ActivityDetails nor DesktopRail is wrapped in memo() (only ToolCallRow at Execution.tsx:49 and Markdown at markdown.tsx:207 are). store.ts:3634 emit() has no throttling and store.ts:3192 notifies per stream event, so this genuinely runs per streamed delta. blocks.ts has no cap on retained blocks and no truncation of tool output. Where the claim overstates: I benchmarked the exact detailOf body under V8 (node 22) against a 200-tool transcript — 3.9 ms/render for a realistic mixed text/JSON corpus (~3.9 MB) and 13.4 ms/render for an all-dense-JSON worst case (~5.7 MB). That is real jank plus multi-MB-per-render GC churn, but it is not the claimed order-of-magnitude catastrophe and there is no evidence it is 'the primary reason long sessions become unusable' — the unmemoized full-transcript render around it costs more. The secondary result.findIndex claim at line 32 is real but bounded by member count and is minor. Confirmed as a wasteful-work defect worth fixing exactly as proposed, at medium rather than high severity.
</details>

### [settings] Composer "Permissions ▾" chip (App.tsx:1636-1642) renders a constant label — the live snap.permissionMode is exposed only via the title tooltip (and is often '' → "not configured"), it carries no mode-derived styling, and the ▾ opens the 8-tab settings modal instead of an anchored menu, unlike the adjacent model and plan chips which both render live state

**`xerxes/src/desktop/renderer/App.tsx:1636`**

*Impact* — The single most consequential piece of session state — whether the next `exec_command` runs unattended or stops for approval — is invisible. The chip reads the same static word "Permissions" whether the session is `accept-all` or `manual`; you only learn the mode by hovering for a tooltip. And the ▾ affordance promises a dropdown but ejects you into an 8-tab modal, so changing a mode mid-task costs a modal open, a tab read, a click, and an Escape — versus one click for the model chip right beside it.

*Fix* — Render the value in the label (`Permissions · {snap.permissionMode || 'default'}`) and tint it when it is `accept-all`. Replace the `openSettings('permissions')` handler with a small anchored popover built from the existing `PERMISSION_MODES` array (Overlays.tsx:866-871) plus a "More…" row that opens the settings tab — the exact pattern `ModelMenu`/`PickerLayer` already implement for the model chip.

<details><summary>Evidence & verification</summary>

```
App.tsx:1636-1642 —
`<button className="cchip" title={`Permissions: ${snap.permissionMode || 'not configured'} — open settings`} onClick={() => store.openSettings('permissions')}>`
`  <Icon name="shield" size={14} /> Permissions <span className="c">▾</span>`
The sibling model chip (App.tsx:1613-1620) renders the live value: `{snap.model ? bareModelName(snap.model) : 'model'}{snap.reasoningEffort ...}` and opens an anchored popover via `store.toggleModelMenu()`.
```

**Verifier:** Verified at source. App.tsx:1636-1642 is exactly as quoted: className is the constant "cchip" (no mode-derived class), the label is the literal word "Permissions" followed by a "▾" caret, snap.permissionMode appears only inside title=, and onClick is store.openSettings('permissions') — the 8-tab modal (Overlays.tsx:33,68), not an anchored menu. No CSS rescues or hides it: app.css:612-623 and atelier.css:60-61 style .cchip normally, and atelier.css:474 hides only .composer__hints>span (the kbd hints), so the chip renders as an uninformative constant. grep for permissionMode in the whole renderer yields only Overlays.tsx:874 (settings modal), App.tsx:429 (a fieldnote that exists only inside the new-task modal) and App.tsx:1638 (the tooltip) — so during an active session the mode has no on-screen textual surface at all. It is worse than claimed: store.ts:800 initializes permissionMode to '' and it is only filled opportunistically (store.ts:2331-2338, 2627, 3325), so the tooltip often reads "not configured"; and ApprovalCard (App.tsx:1161) prints the hardcoded "session policy: ask", actively contradicting the real mode rather than compensating. The inconsistency is real and local: the model chip (App.tsx:1611-1620) renders its value and opens ModelMenu through chipanchor/PickerLayer, and the plan chip (App.tsx:1644-1651) renders live state with .is-on styling (app.css:622) — the permissions chip is the only chip in that row showing nothing. Not a documented decision: there is no intent comment near App.tsx:1636, though this file/store comments intent heavily elsewhere (e.g. store.ts:1572-1574). Other entry points (Overlays.tsx:1374 palette, Setup.tsx:88) also just open the same modal, and main/menus.ts has no permission entry, so there is no cheaper path. The proposed fix is straight reuse of PERMISSION_MODES (Overlays.tsx:866-871) plus store.setPermissionMode (store.ts:2342). I lower severity to medium: the mode is still reachable in one click and the daemon still raises an approval card when it asks, so no capability is lost — the defect is invisible state plus a caret that misrepresents a modal as a dropdown.
</details>

### [settings] Provider-profile Delete (Overlays.tsx:540) and LSP "Remove server" (LspPanel.tsx:96) hard-delete on one unconfirmed click, unlike every other destructive action in the app — and LSP Remove is a pixel-identical btn--ghost sitting next to Cancel

**`xerxes/src/desktop/renderer/Overlays.tsx:540`**

*Impact* — A provider profile holds the API key the user typed into the password field (Overlays.tsx:675-690, which then renders as `leave blank to keep the stored key` — i.e. the key is never shown again). One stray click on `Delete` writes it out of `~/.xerxes/profiles.json` with no confirm and no undo; the user has to go find the key again. In the LSP editor, `Remove server` and `Cancel` are the same ghost button, same size, adjacent — a misclick destroys a hand-entered executable path, argv JSON and env JSON with no prompt.

*Fix* — Wrap Overlays.tsx:540 in `window.confirm(`Delete provider profile ${provider.name}? Its stored API key cannot be recovered.`)`, matching the preset delete at line 408. For LspPanel.tsx:96, give Remove `className="btn chipbtn--danger"`, push it to the opposite end of the `.lsp-actions` row from Cancel, and add the same confirm naming the server.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:536-541 — `<button className="pcard__del" disabled={snap.turnActive} title={`delete ${provider.name}`} onClick={() => store.deleteProvider(provider.name)}>Delete</button>`
store.ts:1394-1416 `deleteProvider` — goes straight to `.call('provider_delete', { name })`; no guard beyond `row.active`/`turnActive`.
LspPanel.tsx:96 — `<button className="btn btn--ghost" type="button" onClick={() => void run('lsp.settings.save', { ..., action: 'remove' })}>Remove server</button>` sitting in the same flex row as, and visually identical to, the `btn--ghost` Cancel button.
Meanwhile the LESS destructive actions do confirm: Overlays.tsx:408 `if (window.confirm(`Delete agent preset ${row.id}? Running sessions are unaffected.`))` and TerminalsPanel.tsx:152 `if (window.confirm(`Kill terminal ...`))`.
```

**Verifier:** Verified line by line. Overlays.tsx:536-541 wires `pcard__del` straight to `store.deleteProvider(name)`; store.ts:1398-1419 only guards on `row.active`/`turnActive` before `.call('provider_delete')`; daemon/server.ts:3788 calls `this.profileStore.delete(name)`, which at bridge/profiles.ts:224-235 does `delete document.profiles[name]` plus an atomic rewrite of ~/.xerxes/profiles.json — no backup, no tombstone, no undo in the success notification. The stored API key is write-only in the UI (Overlays.tsx:684 renders the placeholder `leave blank to keep the stored key`), so it is unrecoverable once the profile is gone. LspPanel.tsx:96 confirmed: `[Save server (btn)][Cancel (btn--ghost)][Remove server (btn--ghost)]` in one `.lsp-actions` flex row (atelier.css:708, gap 8px) — Cancel and Remove share the identical class and sit adjacent. Refutation attempts all failed: `.pcard__del` (app.css:1161-1165) does tint with --x-failed-text, but the LSP Remove has no danger styling at all even though `.btn--danger` exists unused at app.css:556; the TUI equivalent (ui/app/slash/commands/integrations.ts:223-233) also skips a confirm but requires typing `/providers remove <name>`, which is explicit intent unlike a card-row click; and no comment documents the omission — store.ts:1397 only says the active profile must be switched away first. The inconsistency is real: DesktopPanels.tsx:1217 confirms removing a saved workspace while noting remote files are kept, DesktopPanels.tsx:813 builds an inline schedule-confirmation group, Overlays.tsx:408 confirms preset delete, TerminalsPanel.tsx:152 confirms killing a terminal. One correction to the claim: LSP Remove destroys the previously *saved* command/argv/env, not what is currently typed — LspPanel.tsx:92-95 renders those fields blank on edit because the saved values are private. That makes the loss worse, since the destroyed values were never shown. Downgraded to medium because both live behind a deliberately opened settings modal and the loss is a re-obtainable credential plus config fields, not user work product.
</details>

### [states] Session switch and new-task reuse `connection: 'connecting'` (store.ts:1106, 1164), so App.tsx:887 paints a false "Reconnecting…" banner over the previous session's still-rendered transcript — with a "Retry now" button that store.ts:2374 makes a no-op during navigation — while the composer is blocked and mislabeled "Connect to a daemon first…" (App.tsx:1531)

**`xerxes/src/desktop/renderer/App.tsx:887`**

*Impact* — Every session switch and every ⌘N shows a red-flag banner reading "Reconnecting…" with a "Retry now" button, the composer greys out and says "Connect to a daemon first…", and the transcript of the session you just LEFT stays fully painted underneath — for a full daemon round trip (resume of a large session is not instant). The user cannot tell a deliberate navigation from a dropped daemon, cannot tell whether the session they clicked is loading or failed to load, and if they type they get a dead composer. There is no skeleton, no "Opening <title>…", and no visual link to the row they clicked.

*Fix* — Add a distinct `navigating: 'session' | 'fresh' | null` field to the snapshot, set in `openSessionNow`/`beginFreshTask` instead of (or alongside) `connection: 'connecting'`. Gate App.tsx:887 on `!snap.navigating` so the reconnect banner never fires for a user-initiated open. In `Stream`, when `snap.navigating` is set, clear/blur the outgoing transcript and render a transcript skeleton captioned "Opening <row.title>…" (the row title is already known at store.ts:1074 via `row`). Keep the composer enabled-but-queuing, or at minimum change the placeholder to "Opening session…" rather than "Connect to a daemon first…".

<details><summary>Evidence & verification</summary>

```
store.ts:1092 (openSessionNow) and store.ts:1146 (beginFreshTask) both do `this.patch({ connection: 'connecting' })` before awaiting `initialize`. Nothing else is patched — blocks/currentTitle still hold the OLD session. Meanwhile:
App.tsx:887  `{snap.connection !== 'online' && snap.blocks.length > 0 && <div className="connection-status" role="status" aria-live="polite"><strong>{snap.connection === 'connecting' ? 'Reconnecting…' : 'Connection lost. Retrying…'}</strong>…<button onClick={() => store.retryConnection()}>Retry now</button></div>}`
App.tsx:830  `{snap.currentTitle || (snap.connection === 'online' ? 'New task' : 'Not connected')}`
App.tsx:1533 `snap.connection !== 'online' ? 'Connect to a daemon first…'` (composer placeholder), and App.tsx:1478 hard-blocks send on the same condition.
App.tsx:909  `const empty = snap.connection === 'online' && …` — so the Welcome/empty path is also suppressed during the whole window.
```

**Verifier:** Verified line by line. store.ts:1106 (openSessionNow) and store.ts:1164 (beginFreshTask) both patch `connection: 'connecting'` before awaiting `initialize`, and nothing clears the outgoing transcript — blocks/currentTitle are only replaced in applyInitializedSession, which patches `connection: 'online'` at store.ts:2596-2597 in the same batch. App.tsx:887 is verbatim as quoted and fires on exactly `connection !== 'online' && blocks.length > 0`, which is precisely the navigation window. atelier.css:531-533 styles `.connection-status` rather than hiding it, so unlike the chat__head chips it really does paint. App.tsx:909 gates `empty` on `connection === 'online'`, suppressing Welcome. App.tsx:1531/1629 disable send and swap the placeholder to "Connect to a daemon first…". No sidebar row (App.tsx:711) shows any pending state. The window is not negligible: daemon/server.ts:9262+ resolves the resume, and the fresh-task branch awaits runtime.flushSessions() (a disk flush of every session) before eviction. STRONGER THAN CLAIMED: store.ts:2374 makes retryConnection() early-return while openingSession/openingFreshTask is set, so the "Retry now" button the banner offers is completely inert for the whole time it is on screen. Two overstatements corrected: App.tsx:830's "Not connected" fallback does not actually fire (currentTitle still holds the old title, and .chat__title is hidden by atelier.css anyway), and the textarea is not disabled — only send is blocked, with send() at App.tsx:1478 silently early-returning on Enter; the draft is preserved so nothing is lost. Severity lowered to medium rather than high: the state is transient and self-healing, the session does open correctly, and no work is lost — but it misreports a deliberate navigation as a connection failure on the single most common action in the app and hands the user a dead recovery button.
</details>

### [states] Artifacts panel's changed-file list is live-events-only: snap.changes is written solely by foldChange from the 'tool_call' event (store.ts:3188/3445) and adoptHistory (store.ts:2493) never re-folds restored history, so every resumed or relaunched session shows the absolute empty state "No files changed in this session yet." (DesktopPanels.tsx:225) while the git-backed Changes rail (workspace.diff, DesktopPanels.tsx:958-970) simultaneously lists those same files as dirty

**`xerxes/src/desktop/renderer/DesktopPanels.tsx:225`**

*Impact* — Resume any session that edited 20 files, open Artifacts (a top-level sidebar destination, App.tsx:626), and it flatly says nothing was changed — while the Changes rail (`workspace.diff`, DesktopPanels.tsx:969) simultaneously shows all those files dirty. Two surfaces in the same app contradict each other, and the empty state is indistinguishable from a genuinely clean session. The export button above it stays enabled, so it reads as "the session is empty", not "we lost the fold".

*Fix* — In `adoptHistory` (store.ts:2475), after `builder.reset(...)`, walk the restored history actions/tool rows and replay `foldChange` for every `editStatsOf` hit so `this.changes` is rebuilt from persisted history (the per-row stats already exist in blocks.ts). Until that lands, distinguish the states: render "Edits made before this session was reopened are not folded here — see Changes" when `snap.turnCount > 0 && !snap.changes.length`, instead of the absolute "No files changed in this session yet." (Separately: `snap.tab === 'changes'` at App.tsx:889 is unreachable — nothing ever calls `store.setTab('changes')` — so `ChangesTab` in Workspaces.tsx is dead code.)

<details><summary>Evidence & verification</summary>

```
DesktopPanels.tsx:225 `{!snap.changes.length && <p className="studio-muted">No files changed in this session yet.</p>}` under the page header that promises "Files changed in this session and its exportable transcript." (DesktopPanels.tsx:114).
`snap.changes` is written from exactly one place: store.ts:3170 `case 'tool_call': { const stats = editStatsOf(…); if (stats) this.foldChange(stats, …) }` — the LIVE event stream.
On resume, store.ts:2559/2563 call `adoptHistory(session)` which only does `this.builder.reset(historyBlocks(…))` (store.ts:2487) and store.ts:2560 calls `resetWorkspaceFolds()`, whose body does `this.changes.clear()` (store.ts:3459) and patches `changes: []` (store.ts:3483). Nothing re-folds edits out of the restored history, even though blocks.ts:224/241 already computes `editStatsOf` per tool row when replaying.
```

**Verifier:** Confirmed by reading every cited line. DesktopPanels.tsx:225 renders the absolute empty state "No files changed in this session yet." whenever snap.changes is empty, under a header (DesktopPanels.tsx:114) that promises "Files changed in this session and its exportable transcript." snap.changes is populated from exactly one place — store.ts:3445 inside foldChange, whose only caller is store.ts:3188 in `case 'tool_call'` of the live event switch; the only other writes are the `[]` initial (store.ts:765), the `[]` in resetWorkspaceFolds (store.ts:3512) and an undo filter (store.ts:1923). adoptHistory (store.ts:2493) rebuilds only this.builder from historyBlocks() and never touches this.changes, and the resume path (store.ts:2577-2578) additionally calls resetWorkspaceFolds() which does this.changes.clear() (store.ts:3477). So after any resume/app relaunch the Artifacts panel is unconditionally empty. The contradiction is real and not handled elsewhere: ReviewPanel (DesktopPanels.tsx:958-970) fetches `workspace.diff` from the daemon, git-backed and independent of snap.changes, so the Changes rail lists the dirty files at the same moment Artifacts denies any changed. Artifacts is a top-level sidebar destination (App.tsx:626), so this is on the main navigation. No comment anywhere documents this as intentional. The proposed fix is feasible: blocksFromStoredMessages replays stored tool_calls with their arguments (blocks.ts:536, 556-560) and already computes editStatsOf per row (blocks.ts:224), so the fold can be rebuilt from persisted history. The parenthetical also checks out — WorkspaceTab includes 'changes' (types.ts:210) but store.setTab is only ever called with 'activity'/'plan' (App.tsx:876,877,1007,1676), so App.tsx:889's ChangesTab branch is unreachable dead code. Two corrections: (a) resetWorkspaceFolds is not the root cause — on a cold resume this.changes is already empty, the defect is purely the missing re-fold; (b) severity is medium, not high: nothing crashes or is misrouted, the git-backed Changes rail stays correct and the panel's Export transcript action still works, so the damage is a lying empty state on a secondary panel rather than lost or wrong work.
</details>

### [states] Activity rail's 3s poll shares one busy flag with its first load, so an unreserved ~40px "Working…" row flashes into layout every 3 seconds (DesktopPanels.tsx:914, atelier.css:103) and the "Stop watcher" button at DesktopPanels.tsx:899 is re-disabled on each poll, silently dropping clicks

**`xerxes/src/desktop/renderer/DesktopPanels.tsx:875`**

*Impact* — The default right-hand rail permanently jitters: every ~3 s a ~40px spinner row appears at the top of Activity, pushing the background-jobs list, terminals button and context disclosure down, then removes it and snaps them back. Clicking anything in that rail is a moving target. `role="status"` also makes VoiceOver announce "Working…" every three seconds forever. This happens on an idle session with nothing running.

*Fix* — Split first-load from refresh. Give ActivityPanel its own `const [loaded, setLoaded] = useState(false)` gate (it already has one at DesktopPanels.tsx:858) and render `<Feedback busy={request.busy && !loaded} error={request.error} />`. For background polls, either drop the spinner entirely or reserve its height (`min-height` on `.studio-progress` container) and drop `role="status"` on the refresh path. Same pattern applies to the polling loop in AgentInspector.tsx:27.

<details><summary>Evidence & verification</summary>

```
DesktopPanels.tsx:863-876 `const load = async () => { … await request.run(async () => { const result = await request.call('background.activity') … }); if (active) { … timer = setTimeout(() => void load(), 3000) } }` — the poll reuses the same `useRequest` handle as the initial load.
`useRequest.run` (DesktopPanels.tsx:68-85) sets `setBusy(true)` on entry and `setBusy(pending.current > 0)` in `finally`.
DesktopPanels.tsx:914 renders `<Feedback {...request} />` ABOVE the list, and Feedback (DesktopPanels.tsx:93-98) emits `<div className="studio-progress" role="status"><span className="studio-spinner" />Working…</div>`.
atelier.css:103 `.studio-progress{display:flex;align-items:center;gap:9px;padding:12px 20px;…}` — an in-flow block, not absolutely positioned.
The rail is open by default: App.tsx:149 `const rail = railChoice === undefined ? (!snap.noWorkspace && !contextRequiresFullWidth ? 'activity' : null) : railChoice`.
```

**Verifier:** Confirmed at source. DesktopPanels.tsx:863-877 re-arms a 3s setTimeout after every completion and routes the poll through the same useRequest handle as the first load; useRequest.run (68-85) sets busy=true on entry and clears it in finally with no first-load/refresh split; line 914 renders <Feedback {...request} /> as the first child of .studio-form.activity-panel, above the list, idle card and utilities. atelier.css:103 is the ONLY .studio-progress rule in either CSS file (app.css has none): display:flex;padding:12px 20px — in flow, unpositioned, no reserved height, not hidden, so the ~36-40px row genuinely enters and leaves layout each cycle. main/ipc.ts:23-33 shows background.activity is a real daemon socket RPC, so busy is true for an actual round trip, and App.tsx:149 makes the activity rail the default, so this is the resting state of the app. Two impacts are overstated and one part of the fix is wrong: (a) the busy window is one RPC round trip (low ms), so "clicking anything in that rail is a moving target" does not hold — it is a flash, not a sustained shift; (b) the role="status" node is inserted together with its text rather than mutated in place, which most screen readers do not announce, so "VoiceOver announces Working… every three seconds forever" is unverified; (c) AgentInspector.tsx:16-31 renders no Feedback and no busy indicator at all and deliberately stops polling once state.priority >= 2, so the "same pattern" claim there is false. The author also missed the sharper consequence: line 899 gates the "Stop watcher" button on the same shared request.busy, so that control is re-disabled every 3s and a click landing in the window is dropped. Net: real, reachable, default-on, unhandled — but a recurring flicker plus a briefly-dead button, not a high-severity break, so medium.
</details>

### [states] Desktop model picker has no loading or error state: store.ts:1231 `loadModels` patches no `modelsLoading`/`modelsError` and has no in-flight guard (unlike `loadProviderModels` at store.ts:1256), so Overlays.tsx:1056-1069 shows "no models discovered yet" + a retry row for the entire fetch (up to the 8s MODEL_DISCOVERY_TIMEOUT_MS); clicking retry launches a genuinely concurrent second probe (fetch_models is in CONCURRENT_DISPATCH_METHODS, server.ts:397), and on failure the reason is written only into the transcript (store.ts:1241, blocks.ts:276) behind the popover while the picker's copy stays wrong — the TUI already solves this (ui/opentui/modelPicker.tsx:1115)

**`xerxes/src/desktop/renderer/store.ts:1213`**

*Impact* — `fetch_models` probes the live provider, so on a cold app it can take seconds. For that whole time the picker asserts you have no models and offers a retry — the user clicks it, which fires a second concurrent `fetch_models` (no guard), and if discovery then fails the explanation is written into the chat transcript hidden behind the picker's own backdrop, so the picker just keeps saying "no models discovered yet" with no reason. Loading, empty and failed are all the same pixel.

*Fix* — Add `modelsLoading: boolean` and `modelsError: string | null` to the snapshot; set loading in `loadModels` before the call and bail early when already in flight. In Overlays.tsx:1057-1069 branch three ways: loading → a skeleton/"asking <active profile>…" row with the retry button hidden; error → `snap.modelsError` inline with a Retry; empty → today's copy. Do not route model-discovery failures into the transcript while an overlay owns the screen.

<details><summary>Evidence & verification</summary>

```
store.ts:1612-1615 `openPicker(): void { this.patch({ pickerOpen: true, … }); this.loadModels() }` — the picker is painted synchronously, the fetch is fire-and-forget.
store.ts:1213-1214 `loadModels(force = false): void { if (!force && this.frame.models.length) return; void this.bridge.call('fetch_models', {for_model_selection:true})…` — no `modelsLoading` flag is ever patched, and there is no in-flight guard (contrast `loadProviderModels` at store.ts:1237, which does guard on `providerModelLoading`).
Overlays.tsx:1057-1069 therefore renders `snap.models.length ? 'no match' : 'no models discovered yet'` and a `<button … onClick={() => store.loadModels(true)}>↻ fetch from the daemon</button>`.
On failure, store.ts:1221-1225 pushes the reason as a transcript `notification` block — which renders in the Stream, behind `backdrop backdrop--clear` (Overlays.tsx:1021).
```

**Verifier:** Verified in source, with corrected line numbers. store.ts:1231-1249 `loadModels(force=false)` fires `fetch_models` with no loading flag and no in-flight guard (the snapshot type at store.ts:229 has only `models`; the sibling `loadProviderModels` at store.ts:1255-1257 DOES guard on `providerModelLoading`, proving the pattern exists and was skipped here). `openPicker` (store.ts:1620-1623), `togglePicker` (1718-1721), `togglePalette` (1631) and `openTaskModal` (1802) all paint synchronously and call it fire-and-forget. `models` is written in exactly one place (store.ts:1247) and starts `[]` (store.ts:729) with no hydration from the daemon snapshot, so first open on a cold window really is empty. Overlays.tsx:1056-1069 then renders "no models discovered yet" plus an always-visible `↻ fetch from the daemon` row for the entire in-flight window. Latency is real: fetchModels (server.ts:4423) calls discoverModelCatalog bounded by MODEL_DISCOVERY_TIMEOUT_MS = 8_000 (modelDiscovery.ts:19). The double-fetch is worse than claimed: `fetch_models` is listed in CONCURRENT_DISPATCH_METHODS (server.ts:397), so the daemon intentionally runs both probes concurrently and the later-resolving one wins the patch. Failure text goes only to a transcript `notification` (store.ts:1236-1244, rendered solely by blocks.ts:276) — there is no toast/banner in App.tsx — so the picker keeps asserting "no models discovered yet" with the real reason ("No active provider profile is configured", server.ts:4448) elsewhere. Refutation attempts failed: the comments argue the other way — server.ts:355-361 and codexAuth.ts:59 both name "the model picker sat on 'discovering models…'" as a bug they fixed, and the TUI picker implements that loading state (ui/opentui/modelPicker.tsx:1115,1289,1354, tested at ui/__tests__/modelPicker.test.tsx:404); the desktop renderer just never got it. One overstatement trimmed: `.backdrop--clear` is transparent (app.css:1083) and `.modelpop` is a 420px popover anchored above the composer chip (app.css:629-631), so the error notification is misplaced behind/around the popover rather than fully hidden. Severity stays medium: short-lived lie on the happy path, but a permanent dead end with no stated reason when discovery fails.
</details>

### [transcript] The `changes` workspace tab is dead code: `store.setTab('changes')` is never called anywhere, so ChangesTab (session-scoped diff fold with per-file undo / Keep all) plus the `totals` memo at App.tsx:818 are unreachable — but the impact is narrower than claimed, since the working-tree diff and `/undo-edits` still cover the core need

**`xerxes/src/desktop/renderer/App.tsx:889`**

*Impact* — In an agentic coding tool, the one thing a user must see after every turn — what the agent changed in my files — is unavailable from the conversation. The transcript shows only `+3 −1` counts (Execution.tsx:60), and expanding the tool row shows the edit as escaped JSON (`"new_string": "line1\nline2..."`) inside the `Raw details` disclosure at Execution.tsx:86. The Topbar `Changes` button (App.tsx:205) opens `open('review')`, a different surface (git working-tree diff), not this per-turn fold — so `Keep all` and `Undo all` for the agent's own recorded edits are permanently inaccessible.

*Fix* — Two parts. (a) Re-add the tab: in App.tsx:875-881 add `<button className={`tab${snap.tab === 'changes' ? ' is-on' : ''}`} onClick={() => store.setTab('changes')}>Changes{totals.adds || totals.dels ? <span className="pillcount"><span className="add">+{totals.adds}</span> <span className="del">−{totals.dels}</span></span> : null}</button>` — this also consumes the dead `totals` memo. (b) Render the diff inline in the transcript: in Execution.tsx `ExecutionDetails`, branch on `isEditTool(item.name)` (already exported from blocks.ts:94) and render the old_string/new_string hunk with the existing `DiffPreview` (DiffPreview.tsx:6) instead of dumping raw JSON, so an edit row expands to a diff, not a blob.

<details><summary>Evidence & verification</summary>

```
App.tsx:889 renders it: `{snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}`
App.tsx:875-881 renders only three tab buttons: `store.setTab('activity')` / `store.setTab('plan')` / `store.setTab('log')` — no 'changes' button.
A repo-wide grep for `setTab(` in src/desktop returns only those three plus App.tsx:1007 and App.tsx:1676 (both `'plan'`), and DesktopPanels.tsx:454 which is the extension-catalog's own local `setTab`. `store.setTab('changes')` is called from nowhere.
types.ts:210 `export type WorkspaceTab = 'activity' | 'changes' | 'plan' | 'log'` — the state exists and is unreachable.
Workspaces.tsx:38 describes what is lost: "File edits the agents make land here as reviewable diffs — per-file +/− and the exact hunks", with `Keep all` (line 52) and `Undo all` (line 58).
store.ts:3417-3438 `foldChange` already builds real `DiffLine[]` hunks per edit into `snap.changes`.
App.tsx:818-821 `const totals = useMemo(...)` over `snap.changes` — `totals` appears exactly once in the whole 1858-line file, i.e. it is computed and never rendered, the leftover of a removed header chip.
```

**Verifier:** VERIFIED (factual core): App.tsx:889 does render `{snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}`, and the tab strip at App.tsx:875-881 has exactly three buttons (`activity`, `plan`, `log`). A repo-wide grep for `setTab` across src/desktop returns only App.tsx:876/877/880, App.tsx:1007 and App.tsx:1676 (both `'plan'`), DesktopPanels.tsx:426/454 (the extensions panel's own local useState), and the definition at store.ts:1592. `tab` is initialised to `'activity'` (store.ts:762, 3561) and reset to `'activity'` (store.ts:1968); there is no persisted/URL/menu/palette path that can produce `'changes'`. types.ts:210 keeps `'changes'` in WorkspaceTab. `totals` (App.tsx:818-821) appears exactly once in the file — computed, never rendered. So ChangesTab (Workspaces.tsx:22), its `Keep all` (store.ackChanges, Workspaces.tsx:52 — the ONLY caller in the codebase), `Undo all` (line 56) and per-file `undo` (line 85) are genuinely unreachable UI, and the transcript itself shows edits only as `+n −n` (Execution.tsx:60) with escaped JSON under `Raw details` (Execution.tsx:86) — confirmed by reading ExecutionDetails, which has no edit-tool branch.

PARTIALLY REFUTED (impact overstated, hence severity down from high to medium): (1) "what the agent changed in my files is unavailable" is false. The Topbar `Changes` button (App.tsx:205 → `open('review')`) mounts ReviewPanel (DesktopPanels.tsx:130/209, impl at 959-1019), which calls `workspace.diff` (daemon server.ts:2085 → collectGitDiff, includes untracked) and renders a per-file file list plus real hunk lines — the actual on-disk result of the agent's edits. Additionally DesktopPanels.tsx:224 (Artifacts panel) iterates `snap.changes` itself, showing `path · +adds −dels` with a `Review changes →` button that deep-links `open('review', file.path)` into that same hunk view. So the session's own recorded change set IS surfaced and IS clickable to a diff. (2) "`Undo all` is permanently inaccessible" is false: `undo-edits` is a daemon slash command (`src/daemon/server.ts:477` in HANDLED_CANONICAL_COMMANDS, handler at server.ts:5400-5404, catalogued in src/bridge/commands.ts:136), and the desktop palette enumerates the daemon catalog and prefills every command (Overlays.tsx:1429-1441, snap.commands from `commands.catalog`), so `/undo-edits --all --confirm` reaches store's `bridge.call('slash', ...)` path and runs the same `changes.undo` RPC as the dead button.

What actually survives as a defect: dead unreachable code, the loss of the session-scoped fold (the git diff conflates the user's own uncommitted edits and misses edits in a non-git cwd, since collectGitDiff is git-only), no inline diff on edit rows in the transcript, and undo demoted from a one-click confirmed button to a typed slash with a `--confirm` flag nobody discovers. That is a real product gap and dead state worth fixing, but not "the must-see surface is unavailable" — medium, not high. The proposed fix (re-add the tab button consuming `totals`; branch ExecutionDetails on `isEditTool`, which is indeed exported at blocks.ts:94, and render via DiffPreview) is sound.
</details>

### [transcript] App.tsx:888-891 mount the Conversation/Plan/Log tab bodies conditionally, so any tab switch unmounts Stream: transcriptScroll.ts:10,43-47 re-pins to the transcript tail and every mount-local disclosure (App.tsx:1066-1067,1107; Execution.tsx:50,54,67; OutputViewer.tsx:22-23) re-collapses, with no saved state to restore — fix requires persisting scrollTop/following per session key, not just the `hidden` pattern from line 886, since atelier.css:179 display:none drops scrollTop

**`xerxes/src/desktop/renderer/App.tsx:888`**

*Impact* — You are 300 lines up in a long turn, reading a failed test's output with three tool rows expanded. You click `Plan & todos` to check the checklist, click `Conversation` again — and you are back at the bottom of the transcript with every row re-collapsed, with no history of what you had open. On a 200-block session that is a minute of re-scrolling and re-clicking, every single time. The same loss happens when the store opens a different session and returns.

*Fix* — Mirror the `hidden` pattern already used at App.tsx:886: render all three tab bodies unconditionally inside their own wrappers and toggle `hidden={snap.tab !== 'activity'}` etc. If `Stream` must stay conditional, lift the disclosure state into the store keyed by `${sessionKey}:${block.id}` (ActivityGroup, ToolCallRow, thinkrow) and persist `element.scrollTop` per session in `useTranscriptScroll` so a remount restores the saved offset instead of unconditionally calling `follow()`.

<details><summary>Evidence & verification</summary>

```
App.tsx:888-891 mounts the tabs conditionally: `{snap.tab === 'activity' && <Stream snap={snap} />}` / `{snap.tab === 'plan' && ...}` / `{snap.tab === 'log' && ...}`.
The correct pattern is already used one line above for desktop pages — App.tsx:886: `<div className="conversation-surface" hidden={page !== null}>` — which keeps the subtree mounted.
Everything that dies on unmount is component-local `useState`/uncontrolled `<details>`: ActivityGroup App.tsx:1066 `const [expanded, setExpanded] = useState(false)`; thinking row App.tsx:1107 `<details className="thinkrow">` (uncontrolled, comment on 1105 claims "the user's toggle survives re-renders" — it survives re-renders, not unmounts); ToolCallRow Execution.tsx:50 `useState(false)`; ExecutionDetails Execution.tsx:67 `useState(false)`; OutputViewer.tsx:22-23 wrap/expanded.
Scroll: transcriptScroll.ts:8-11 `const ref = useRef(...)`, `const following = useRef(true)`, then useLayoutEffect line 43-47 `if (following.current && element.clientHeight > 0) { element.scrollTop = element.scrollHeight }` — a fresh mount starts `following === true` and snaps to the bottom.
```

**Verifier:** Confirmed at source. App.tsx:888-891 mount the three tab bodies conditionally (`{snap.tab === 'activity' && <Stream snap={snap} />}`), with no keep-alive and no comment documenting the choice — notable in a file that comments intent densely around it (903-904, 911-912, 928-930, 1104-1105). transcriptScroll.ts:10 `following = useRef(true)` plus the dep-array-less useLayoutEffect (36-75) calling `follow()` (43-47, `element.scrollTop = element.scrollHeight`) means a remount unconditionally snaps to the tail. Every disclosure is mount-local: App.tsx:1066-1067 useState, App.tsx:1107 uncontrolled `<details className="thinkrow">` (its 1105 comment says the toggle survives re-renders — it does not survive unmount), Execution.tsx:54 uncontrolled `<details>` with the :50 `inspected` latch, Execution.tsx:67, OutputViewer.tsx:22-23. No CSS or code path rescues any of it. Two corrections: (1) Execution.tsx:50 is `inspected`, a one-way render latch, not an expanded toggle — the actual open state is the uncontrolled DOM `<details open>`; same outcome, different mechanism. (2) The proposed primary fix is wrong as written: atelier.css:179 makes `[hidden]` `display:none!important`, and Chromium resets scrollTop on display:none, so mirroring line 886 would restore every disclosure but land the user at scrollTop 0 — the top of a long transcript, worse than the tail. Only the fallback (persist scrollTop and the `following` flag per session key inside useTranscriptScroll) fixes the scroll half; the same caveat means the existing page-navigation path at line 886 already loses transcript scroll today. Severity downgraded to medium: repeatable and unavoidable, but purely ergonomic — no data loss, nothing unrecoverable, and the common case of a user pinned at the tail during a live turn loses nothing.
</details>

### [transcript] Transcript auto-follow can only be re-armed by scrolling back into a 64px band (transcriptScroll.ts:60); the hook exports no `following` flag or `scrollToLatest` (:76), so there is no jump-to-latest control and — because :41 re-pins only on session change — submitting a new prompt while scrolled up renders your own message and the whole reply off-screen

**`xerxes/src/desktop/renderer/transcriptScroll.ts:60`**

*Impact* — During a long turn the agent keeps appending blocks, so the bottom of the document is a moving target. The moment you scroll up to read a grep result, following is off and there is no way back to live except dragging the scrollbar to a bottom that keeps receding — and nothing tells you that output is still arriving below. Users end up hammering End or waiting for the turn to finish before they can read anything.

*Fix* — Return the pinned state from the hook — expose `following` via a `useSyncExternalStore`-backed boolean and a `scrollToLatest()` that sets `following.current = true; element.scrollTop = element.scrollHeight`. In `Stream` (App.tsx:932), render a sticky pill at the bottom of `.stream` when `!following && (snap.turnActive || newBlocksSince)`: `Jump to latest` with a count of blocks added since unpinning. Bind it to the End key as well.

<details><summary>Evidence & verification</summary>

```
transcriptScroll.ts:60 is the only way to re-pin: `following.current = element.scrollHeight - element.scrollTop - element.clientHeight < 64` — it fires only from a real `scroll` event, and only inside a 64px band at the bottom.
The hook's public surface is transcriptScroll.ts:76 `return { ref, loadOlder }` — it exposes no `following` flag and no `scrollToLatest`, so the UI cannot even know it is unpinned.
Grep for `latest|Jump|scrollIntoView|scrollTo` across App.tsx and transcriptScroll.ts returns only App.tsx:154 (a rail scroll reset) and the internal `element.scrollTop` writes. Grep for `jump|latest|follow` across app.css and atelier.css returns one unrelated comment (app.css:55). There is no button.
app.css:321 `.stream { ... overflow-anchor: none; }` also disables the browser's own anchoring fallback.
```

**Verifier:** CONFIRMED with corrections. transcriptScroll.ts:60 is verbatim as claimed (`following.current = element.scrollHeight - element.scrollTop - element.clientHeight < 64`) and fires only from the passive `scroll` listener registered at :66. `following.current` is written in exactly three places — :19 (loadOlder sets false), :41 (session-identity change sets true), :60 — so the 64px band is the only re-pin path during a turn. Line 76 `return { ref, loadOlder }` exposes neither the flag nor a scroll method, so no consumer can know it is unpinned. App.tsx:902 keys the hook on `${snap.currentId}:${snap.sessionOpenRevision}`, and sessionOpenRevision bumps only on session open/resume (store.ts:1120), never on submit. Grep across App.tsx, app.css and atelier.css finds no jump/latest/follow control. No mitigating code path, no comment documenting this as intentional.

TWO CLAIMED IMPACTS ARE OVERSTATED: (1) "nothing tells you that output is still arriving" is false — App.tsx:1570 renders a persistent `.streamstatus` pill ("Acting… m:ss" with a pulsing dot, app.css:404-408) inside the Composer, outside the `.stream` scroll container, so it is always visible; app.css:403's own comment says it is "the turn clock where the eye is while scrolling." (2) "no way back except dragging the scrollbar to a bottom that keeps receding" is overstated — a wheel scroll to the bottom does re-pin on entering the 64px band, and because `overflow-anchor: none` (app.css:321) content growth does not push the viewport, so the travel distance is finite. "Hammering End" is speculative: focus normally sits in the composer textarea where End is a caret move.

ONE POINT THE CLAIM MISSED, SHARPER THAN WHAT IT ARGUED: since :41 is the only re-pin, submitting a new prompt while scrolled up leaves `following` false — the user's own message and the entire reply render off-screen with no signal other than the composer clock.

Severity corrected to medium: a real missing affordance plus a genuinely disorienting unpinned-submit case, but nothing is unreachable and a live-turn indicator exists, so it is friction rather than broken function. The proposed fix (expose `following` + `scrollToLatest()`, render a sticky pill in `.stream`) is correct; it should additionally set `following.current = true` on submit.
</details>

### [transcript] The transcript has no find at all — no Cmd+F in GlobalKeys (App.tsx:1766-1786), no findInPage in main (only a bare {role:'editMenu'} at main.ts:699) — and the cross-session search that does exist throws the match away: SearchPanel.tsx:46 passes only hit.sessionId to store.openSession (store.ts:1055 accepts no anchor), which hydrates just the last 100 entries (store.ts:1108), so a deep hit opens a session that does not even contain the matched message, unhighlighted; collapsed ActivityGroups (App.tsx:1070) also keep every tool call out of the DOM, defeating any future find.

**`xerxes/src/desktop/renderer/App.tsx:1776`**

*Impact* — The transcript is where users spend 95% of their time and it is the only large text surface in the app with no search at all. "Which turn touched authMiddleware?" is answered by scrolling. And the one search that works cross-session drops you at the bottom of the matched session with no highlight and no scroll — you know the message exists somewhere in 400 blocks and you are back to scrolling.

*Fix* — (a) Add ⌘F in `GlobalKeys` (App.tsx:1766) opening an in-transcript find bar: filter/highlight over `snap.blocks` text with next/previous that calls `scrollIntoView` on the matching block, and force-open any ActivityGroup containing a hit (this needs the store-backed expansion state from finding 2). (b) In SearchPanel.tsx:46, change the signature to `open(hit)` and pass the match through — `store.openSession(hit.sessionId, { anchorMessage: hit.messageId })` — then have `Stream` scroll to and flash that block after hydration, reusing the `data-history-anchor` attribute already emitted at App.tsx:943.

<details><summary>Evidence & verification</summary>

```
App.tsx:1765-1785 `GlobalKeys` binds exactly three shortcuts: `if (meta && event.key === ',')` → settings, `if (meta && event.key.toLowerCase() === 'k')` → palette, `if (meta && event.key.toLowerCase() === 'n')` → new chat. No `f`.
Grep for `findInPage|Find` across src/desktop/main.ts and src/desktop/main/*.ts returns nothing, and there is no menus.ts (the main/ directory holds contextNavigation, daemon, ipc, machines, notify, providerForwarding, remote, remoteResume, spawn, voice, windowRecovery, windowRoutes, windowState, workspaceNavigation, workspaceSettings) — so Electron's native find is not wired either.
SearchPanel.tsx:2-9 says each hit is "one matched message — role, excerpt, session title, age", but SearchPanel.tsx:46-50 discards which message: `const open = (sessionId: string): void => { if (locked) return; store.closeSessionSearch(); void store.openSession(sessionId) }`. Called from line 63 `if (hit) open(hit.sessionId)` — `hit` has the match, only `sessionId` is passed.
Compounding it: App.tsx:1070 `{(expanded || inspected) && blocks.map(...)}` — a collapsed ActivityGroup keeps its children out of the DOM entirely, so even a future `findInPage` would miss every tool call and reasoning trail.
```

**Verifier:** Every cited line says what is claimed. App.tsx:1766-1786 binds only Cmd+, / Cmd+K / Cmd+N (plus Escape dismissal and 1/2/3 approval keys) — I read GlobalKeys to its end at line 1858 and there is no 'f' branch. main.ts:694-700 is the only menu template and uses bare {role:'editMenu'}, which in Electron contains no Find item; grep for findInPage|stopFindInPage|CommandOrControl+F across main.ts, main/**, and the entire renderer returns zero hits, so native find is not wired. SearchPanel.tsx:46-50 is exactly `const open = (sessionId: string) => { ... store.openSession(sessionId) }`, called at line 63 and line 116 with `hit.sessionId`; the hit's messageIndex (populated at store.ts:470) is used only as a React key and dropped. store.ts:1055 confirms `openSession(id: string)` has no anchor parameter. App.tsx:1070 confirms `{(expanded || inspected) && blocks.map(...)}` so collapsed ActivityGroup children are out of the DOM. App.tsx:943 confirms data-history-anchor exists, consumed only by transcriptScroll.ts:22,31 for history-prepend retention. Impact is understated, not overstated: store.ts:1108 opens a session with history_limit:100, so a hit from deep in a long session opens a transcript that does not contain the matched message at all — the user must repeatedly click "Load 100 older actions" (App.tsx:938) and scroll, with no highlight. And per already-confirmed finding 1 the "Session log" export button is CSS-hidden, so there is no escape hatch to grep the text externally. No comment anywhere frames this as intentional; SearchPanel.tsx:4-9 documents a hit as "one matched message", which the open() implementation contradicts.
</details>


## LOW

### [a11y] The renderer has no persistent live region or sr-only utility, so every status announcement (App.tsx:1570, 1569, 887 and ~50 peers) is inserted already containing its text and removed on completion — unreliable to announce on Chromium/VoiceOver and, for turn completion, announced by nothing at all, while the transcript (App.tsx:932) is an unlabelled non-log region

**`xerxes/src/desktop/renderer/App.tsx:1570`**

*Impact* — Nothing in this app is reliably spoken: NVDA and VoiceOver only announce live regions that were already in the accessibility tree when their content changed, so "Acting…", "Sending… preparing the task", "Connection lost. Retrying…", "Working…" and every panel error are inserted-with-content and routinely dropped. Turn completion is announced by nothing at all — the region simply disappears. The one region that does persist reports the fixed string "Activity" instead of the running count, so a screen-reader user can never learn that three shells are running or that context compaction started.

*Fix* — Mount one persistent announcer at the App root (`<div className="sr-only" role="status" aria-live="polite" />` plus an assertive twin) and write strings into it from the store on state transitions — turn started/finished/failed, approval required, question asked, connection lost/restored. Keep the visual chips as they are but mark them aria-hidden so the text is not doubled. On DesktopPanels.tsx:1390, drop aria-label="Activity" (the inner text is a better name) or move aria-live to an inner span that is not the labelled element.

<details><summary>Evidence & verification</summary>

```
`{snap.turnActive && <div className="streamstatus composer-status" role="status" aria-live="polite">{snap.networkRetrying ? 'Retrying connection…' : 'Acting…'} {turnDurOf(snap.turnSeconds)}</div>}` — the node does not exist before the turn starts and is unmounted when it ends. Identical conditional-mount pattern for every other announcement: App.tsx:1569, 887, 934; Overlays.tsx:161 `{busy && <p role="status">…}`; DesktopPanels.tsx:94, 612, 673, 1229, 1263; ContextInspector.tsx:66. The only permanently mounted live region in the renderer is DesktopPanels.tsx:1390 — `<button title="Background activity" aria-label="Activity" className="studio-background" onClick={…} aria-live="polite">` — whose aria-label masks the very text that changes inside it ("2 shells running", the compaction spinner label). The transcript itself (App.tsx:932 `<div className={`stream…`} ref={ref}>`) has no role="log", no aria-live and no accessible name.
```

**Verifier:** The line is verbatim as quoted and the mount pattern is real: App.tsx:1570, 1569, 887, 934 and ~50 other role="status"/role="alert" nodes across the renderer are all inserted into the DOM with their text already present and removed when the condition clears. Grep confirms no sr-only/visually-hidden utility exists in app.css or atelier.css, so there is no persistent-announcer infrastructure anywhere, and App.tsx:932's transcript container has no role="log", no aria-live and no accessible name. No documented decision exempts the renderer (docs/code-standards.md:40 only constrains the OpenTUI), and the renderer clearly attempts a11y elsewhere, so this is intent-not-executed. However two sub-claims are wrong and force a downgrade. (1) DesktopPanels.tsx:1390 is characterised backwards: that button is unconditionally mounted from App.tsx:203 inside Topbar and its inner span mutates in place, making it the one place the pattern is done correctly; aria-atomic is absent so the default false means the changed descendant text ("2 shells running", the compaction label) is what is announced — aria-label sets the button's name, not the live announcement, so the proposed fix there would regress it. (2) "announce nothing"/"nothing is reliably spoken" overstates the mechanism: this is Electron/Chromium only, and Blink posts a live-region-created notification for newly inserted regions which NVDA generally does announce; the defect is unreliability concentrated on macOS/VoiceOver, not universal silence. Also App.tsx:1569's missing aria-live is a non-issue since role="status" implies polite. What survives is narrower: turn start/end/failure and every panel error have no announcement channel that outlives the node, so turn completion in particular is announced by nothing at all (no OS-notification fallback exists in main/notify.ts either).
</details>

### [a11y] TaskModal (App.tsx:337) is the renderer's only aria-modal dialog with no focus trap and no focus restore — every other one uses useDialogFocus or hand-rolls it — so Tab walks into the background the dialog told AT does not exist and closing drops focus to body; its inline "change" permission affordance (App.tsx:430) is a bare click-only &lt;u&gt;. Mitigated: Escape still closes it (App.tsx:1797) and ⌘K → "Permissions settings…" (Overlays.tsx:1374) still reaches the same screen.

**`xerxes/src/desktop/renderer/App.tsx:337`**

*Impact* — Tab from the last button in the dialog walks out into the sidebar, session list and composer behind the dialog — controls that aria-modal="true" has told the screen reader do not exist, so the user is typing into an interface their AT cannot describe, with no way back except reverse-tabbing. Closing the modal drops focus to <body>, so the next Tab restarts from the top of the window. And the "change" affordance that takes you from a new task to the permission settings — the one place a user decides whether this task can run commands unattended — is reachable by mouse only: keyboard and screen-reader users cannot activate it at all.

*Fix* — Call `useDialogFocus(ref)` on the modal root (it already gives trap + restore + initial focus, and its `closest('[role="dialog"]')` lookup works unchanged here). Replace the <u> with `<button type="button" className="linkbtn">change</button>` styled to look like the current underline. While there, associate the bare `<label>` elements at App.tsx:343, 388 and 421 with their controls via htmlFor/id so the objective textarea has a real name instead of falling back to its placeholder.

<details><summary>Evidence & verification</summary>

```
`<div className="modal taskmodal" role="dialog" aria-modal="true" aria-label="New task">` — TaskModal is absent from `grep -rn "useDialogFocus" .`, which lists only Overlays.tsx:42, Overlays.tsx:1349, SearchPanel.tsx:34 and DesktopPanels.tsx:148. It relies on bare `autoFocus` (App.tsx:356, 393). Inside it, App.tsx:430: `<u style={{ cursor: 'pointer' }} onClick={() => { store.closeTaskModal(); store.openSettings('permissions') }}>change</u>` — no role, no tabIndex, no key handler. It is the only <u>-as-button in the renderer (`grep -rn "<u " *.tsx` = 1 hit).
```

**Verifier:** Code facts verified verbatim. App.tsx:337 is `<div className="modal taskmodal" role="dialog" aria-modal="true" aria-label="New task">` with no ref and no focus management; dialogFocus.ts:7 (`useDialogFocus`, which does trap + restore + initial focus) is wired into Overlays.tsx:42/1349, SearchPanel.tsx:34 and DesktopPanels.tsx:148, and Setup.tsx:36-48 hand-rolls the identical trap+restore — TaskModal is the only aria-modal dialog in the renderer with neither, and the component doc comment at App.tsx:308-313 discusses workspace/model semantics only, so this is an oversight rather than a documented choice. App.tsx:430 is verbatim a bare `<u style={{cursor:'pointer'}} onClick={...}>change</u>` with no role/tabIndex/keydown; `grep -rn "<u " *.tsx` returns exactly that one hit and no `u` selector exists in app.css or atelier.css. The `.backdrop` (app.css:1079) has no onClick and nothing marks the background inert or aria-hidden, so background controls genuinely stay in the tab order behind a dialog that claims aria-modal. TWO IMPACT CLAIMS ARE OVERSTATED, which is why I lowered severity: (1) "no way back except reverse-tabbing" is wrong — App.tsx:1797-1801 closes the task modal on Escape, and the early-return guard at App.tsx:1769 only bails on a native `dialog[open]` element, which this div is not, so Escape works even after focus has walked out; (2) "keyboard and screen-reader users cannot activate it at all" is wrong about the destination — Overlays.tsx:1374 registers a `Permissions settings…` command palette entry calling `store.openSettings('permissions')`, and both ⌘K (App.tsx:1776) and ⌘, (App.tsx:1771) still fire while the task modal is open, so the permissions screen is one keystroke away. The inline affordance is mouse-only; the capability is not lost. Net: a real, cheap-to-fix a11y/ergonomics defect (escaped tab order, focus dropped to body on close, one mouse-only control) with working fallbacks and no lost functionality. The proposed fix is sound — `useDialogFocus(ref)` needs no change since its `closest('[role="dialog"]')` lookup resolves to this same node.
</details>

### [a11y] WorkspaceMenu (App.tsx:271) is a role="menu" with zero valid owned children (plain buttons at 274/299 plus two divs), no focus entry/restore/arrow keys, and its sole trigger (DesktopPanels.tsx:1191 open(null)) unmounts itself as the menu opens — so AT reports an empty menu and focus is thrown back to the sheet's opener; not a lockout, since the sidebar (App.tsx:659-667) exposes the same enterWorkspace list to the keyboard

**`xerxes/src/desktop/renderer/App.tsx:271`**

*Impact* — Activating "Recent workspaces" by keyboard destroys the pressed button, so focus lands on <body> while a menu appears elsewhere on screen; reaching it means Tab-crawling the whole window from the top, and the menu never announces that it opened. A screen reader that does honour role="menu" reports zero items, because plain buttons are not valid owned children of a menu — so the list of workspaces reads as empty. Switching workspace, a daily action, is effectively mouse-only.

*Fix* — Give WorkspaceMenu the SessionMenu treatment: a ref with initial focus on the first row, previous-focus restore on unmount, local Escape + Arrow/Home/End roving, and role="menuitem" on the rows at App.tsx:274 and 299. Since the two menus now diverge only in content, extract SessionMenu's useLayoutEffect (App.tsx:469-478) and keydown block (App.tsx:494-504) into a shared `usePopupMenu(ref)` hook and call it from both.

<details><summary>Evidence & verification</summary>

```
`<div className="wsmenu" role="menu" aria-label="Workspaces">` whose children are plain `<button className={`wsrow…`}>` (App.tsx:274-286, 299) — no role="menuitem", no ref, no focus(), no keydown handler; Escape is only handled far away in GlobalKeys (App.tsx:1827). It is opened from DesktopPanels.tsx:1189-1193: `onClick={() => { open(null); store.toggleWorkspaceMenu() }}` — `open(null)` unmounts the sheet that holds the very button being pressed. Contrast the sibling SessionMenu, which does all of it correctly: focus entry `element.querySelector<HTMLElement>('button,input')?.focus()` (App.tsx:475), restore on unmount (App.tsx:477), roving ArrowUp/Down/Home/End (App.tsx:497-503), `role="menuitem"` on each row (App.tsx:527, 530, 534, 537).
```

**Verifier:** Mechanics verified: App.tsx:271 is `<div className="wsmenu" role="menu" aria-label="Workspaces">` whose children are plain `<button className="wsrow…">` (274-286) and `<button className="menu__item">` (299) with no role="menuitem", plus non-interactive divs at 272 and 298; there is no ref, no focus() on entry, no focus restore, no Arrow/Home/End handler — unlike SessionMenu (App.tsx:469-478, 494-504, 527/530/534/537). DesktopPanels.tsx:1189-1193 is the only trigger (grep toggleWorkspaceMenu returns one hit) and sits inside WorkspacePanel, rendered by DesktopSheet (DesktopPanels.tsx:211), so open(null) -> App.tsx:157 setPanel(null) does unmount the pressed button while .wsmenu appears at fixed top:44px (app.css:720-725). BUT two of the three impact claims are false. (1) "focus lands on <body>" is wrong: DesktopSheet calls useDialogFocus(ref) (DesktopPanels.tsx:148) and dialogFocus.ts:24 restores focus to the still-connected opener on unmount, i.e. the sidebar Workspace button (App.tsx:683). (2) "Switching workspace is effectively mouse-only" is wrong: the sidebar renders the same groupByWorkspace list (App.tsx:611 vs 266) as ordinary Tab-reachable buttons calling the identical store.enterWorkspace(group.cwd) (App.tsx:659-667), and App.tsx:679 offers Add folder. (3) Escape does work from anywhere via the window-level handler at App.tsx:1827-1831, not just "far away" in theory. What remains real: an aria-required-children violation (a menu whose owned children are buttons/divs, so strict AT announces zero items) and a focus displacement on open, on a redundant secondary surface that a keyboard user never needs. That is polish, not a capability loss — low, not medium. The proposed shared usePopupMenu(ref) extraction remains the correct fix.
</details>

### [copy] Sidebar mixes three nouns for one object — button label "New session" (App.tsx:624) contradicts its own tooltip "Start a new task…" (App.tsx:623), under a "Sessions" header (:618), over rows titled "New task" (:595) and an empty state calling them "chats" (:677); the workspace menu's unguarded plural (App.tsx:267) also prints "1 tasks"

**`xerxes/src/desktop/renderer/App.tsx:624`**

*Impact* — The user cannot build a mental model of what the app's primary noun is. "New session" (sidebar) and "New task" (⌘N modal) look like two different creation flows and invite the question of which one to use; the sidebar header says "Sessions" but the workspace switcher counts "tasks" and the empty state calls them "chats". Screen-reader users get the worst of it — the accessible name of the ⌘N button is "New session" while its tooltip and the dialog it opens both say task. And the count renders "1 tasks" for the extremely common single-session workspace.

*Fix* — Pick one noun for the top-level object and sweep every literal. "Task" fits an agentic coding tool (it has an outcome and it finishes); then: App.tsx:199/624 → 'New task', :618 → 'Tasks', :677 → 'No tasks yet — they live inside the workspace folder', DesktopPanels.tsx:223 → 'Export task transcript', :225 → 'No files changed in this task yet', App.tsx:504 'Rename session' → 'Rename task'. Fix the count with the ternary the codebase already uses elsewhere (App.tsx:757, :974): `` `${n} task${n === 1 ? '' : 's'}` ``. Do the same for the other unguarded plurals: Overlays.tsx:178 `{status.tools} tools`, AgentRoster.tsx:61 `{info.toolCount} tools`, ContextInspector.tsx:67 `{summary.count} entries`.

<details><summary>Evidence & verification</summary>

```
One button disagrees with itself: App.tsx:623 `title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N)` : 'Start a fresh session (⌘N)'}` on App.tsx:624 `<span>New session</span>`. Its sibling segment is App.tsx:618 `>Sessions</button>`. The window title says App.tsx:199 `{snap.currentTitle || 'New session'}` while the chat header directly under it says App.tsx:830 `{snap.currentTitle || (snap.connection === 'online' ? 'New task' : 'Not connected')}` and the sidebar row for that very session uses App.tsx:595 `title: snap.currentTitle || 'New task'`. The ⌘N modal is App.tsx:339 `<h2 className="modal__title">New task</h2>`. The workspace menu counts them as App.tsx:267 `` `${groups.find(g => g.cwd === cwd)?.rows.length ?? 0} tasks` `` — which also prints "1 tasks". A fourth word appears in the prose: App.tsx:264 comment-adjacent copy and App.tsx:677 `'No tasks yet — your chats live inside the workspace folder'`. Same split in the side panels: DesktopPanels.tsx:126 `aria-label="Task context"` / :127 `aria-label="Close task context"` vs DesktopPanels.tsx:223 `Export session transcript` and :225 `No files changed in this session yet.`
```

**Verifier:** Core claim verified at the cited lines, but two of its evidence items are wrong and the impact is overstated, so I downgrade to low.

VERIFIED (read directly):
- App.tsx:623-624 — one button truly disagrees with itself: `title={homeLabel(snap) ? \`Start a new task in ${homeLabel(snap)} (⌘N)\` : 'Start a fresh session (⌘N)'}` on a `<span>New session</span>` label.
- App.tsx:618 segment reads `Sessions`; App.tsx:199 `{snap.currentTitle || 'New session'}`; App.tsx:595 sidebar row placeholder `'New task'`; App.tsx:677 empty state `'No tasks yet — your chats live inside the workspace folder'`; App.tsx:504 `aria-label={renaming ? 'Rename session' : 'Session actions'}`. All visible, all in the same sidebar column. This is a genuine three-noun split (session / task / chat) inside one pane.
- App.tsx:267 `\`${groups.find(g => g.cwd === cwd)?.rows.length ?? 0} tasks\`` — unguarded plural, reachable for any non-current workspace group with one row, so "1 tasks" really renders. The codebase already has the guarded idiom at App.tsx:757 and :974 (`${n} agent${n === 1 ? '' : 's'}`), so this is an inconsistency with its own convention, not a missing capability. Overlays.tsx:178, AgentRoster.tsx:61, ContextInspector.tsx:67 are likewise unguarded.
- DesktopPanels.tsx:126/127 `aria-label="Task context"` / `"Close task context"` vs :223 `Export session transcript` and :225 `No files changed in this session yet.` — same panel, both nouns.

REFUTED PARTS of the evidence:
1. "The window title says New session while the chat header directly under it says New task" — false as rendered. App.tsx:163 applies the `atelier` class unconditionally, and atelier.css contains `.atelier .chat__title,.atelier .chat__id{display:none}`. App.tsx:830 never paints; that adjacency does not exist on screen (and the string is already covered by confirmed finding #1's dead-UI set).
2. "The ⌘N button ... the dialog it opens both say task" — false. App.tsx:622 `onClick={() => { open(null); store.newChat() }}` and GlobalKeys App.tsx:1781-1784 (`meta && key === 'n'` → `store.newChat()`) go to `store.newChat()` → `beginFreshTask()` (store.ts:1143), which never sets `taskModalOpen`. The "New task" modal (App.tsx:337-339) is opened only by `store.openTaskModal()` from the command palette (Overlays.tsx:1410). So "New session" and "New task" are in fact two genuinely different creation flows (blank rebind vs objective+plan-first+preset+model form) — the impact paragraph's "two labels for one flow, which one do I use" framing is inverted.

Impact is copy polish, not capability: nothing is blocked or misread consequentially, and the only hard defect is the "1 tasks" plural. Hence low, not medium. (Incidental: Overlays.tsx:1407-1410 labels the palette entry 'New task' with hint '⌘N' while ⌘N runs a different action — a separate, sharper bug the claim did not notice.)
</details>

### [copy] Log tab's only explanation (Workspaces.tsx:178) is written in implementation nouns — "wire order", "this ring" — and the Changes footnote (:63) re-exposes the raw tool names blocks.ts:41 verbOf() humanizes everywhere else

**`xerxes/src/desktop/renderer/Workspaces.tsx:63`**

*Impact* — These are the only explanatory sentences in their panels, and neither is decodable without having read the source. "Folded" is an internal verb for the block accumulator; "this ring" is the ring-buffer implementation; "wire order" is the protocol's ordering guarantee; `FileEditTool` / `WriteFile` are TypeScript class names the user never sees anywhere else in the product. The Changes note is trying to reassure the user that the diff is read-only and it instead reads as a stack trace in prose. The Log empty state is the user's only chance to learn what the tab is for, and it spends all of it on implementation nouns.

*Fix* — Rewrite both in the reader's terms. Workspaces.tsx:63 → 'Built from the edits this task made. Read-only — nothing here rewrites your files.' Workspaces.tsx:178 → 'A live record of what the runtime did, newest last — one line per step. Start a task and it fills in (last 400 steps kept).' Keep <code> for things the user can actually type (paths, commands), never for internal symbol names.

<details><summary>Evidence & verification</summary>

```
Workspaces.tsx:62-64, the note under every diff in the Changes tab: `Folded live from <code>FileEditTool</code> / <code>WriteFile</code> calls — the daemon owns the worktree; nothing here rewrites it.` And Workspaces.tsx:178, the entire empty state of the Log tab: `<p>Every daemon event — wire order, one line each — as it arrived. Streaming a turn fills this ring (last 400 events).</p>` under `<h1>Event stream</h1>`.
```

**Verifier:** Both strings exist verbatim at Workspaces.tsx:63 and :178, neither is CSS-suppressed (app.css:909 styles .changes__note at 9.5px mono; app.css:894-901 styles .tabempty; atelier.css has no rules for .changes/.tabempty/.loglist), and both states are reachable — Log is a first-class tab (App.tsx:880) and store.ts:3479 resetWorkspaceFolds clears logRing on every session switch, so the empty state is routine. "ring"/"400" literally name logRing/LOG_CAP (store.ts:641,3472), confirming the implementation-leak read. Two pieces of the evidence are wrong, though: FileEditTool/WriteFile are tool wire names (tools/fileTools.ts:70,181), not TypeScript class names, and the user DOES see them raw in the approval card (App.tsx:1153 renders approval.toolName from payload.tool_name, store.ts:3334) — so "never sees anywhere else" is false. The sharper inconsistency is that blocks.ts:41 verbOf() deliberately maps FileEditTool -> "edit" for the transcript, and line 63 undoes that humanization. The Changes half is also overstated: line 38 is a second explanatory sentence in the same panel, every actionable control there already has plain-language labels/tooltips (:52, :55, :84), and line 63 is a 9.5px mono footnote whose loss costs the user nothing they can act on. And "daemon" is pervasive user-facing vocabulary across the product (App.tsx:677,1051,1534; ChannelsPanel.tsx:45; TerminalsPanel.tsx:56), so these two strings are a slice of systemic copy drift rather than an isolated defect. What survives is the Log half: line 178 is the entire answer to "what is this tab for" on a top-level tab, and it spends it on "wire order" and "this ring". Copy-only, no behavior change, one of two instances near-cosmetic => low, not medium.
</details>

### [dead-ui] Escape chain (App.tsx:1822) lets the non-modal "Token breakdown" expander (DesktopPanels.tsx:948) swallow the first Escape ahead of turn-cancel (App.tsx:1837); since nothing auto-closes contextMenuOpen and closing the rail does not reset it, that press can be entirely feedback-free — while the real ContextMenu dialog (Overlays.tsx:1160) ships unmounted

**`xerxes/src/desktop/renderer/App.tsx:1822`**

*Impact* — Expand "Token breakdown" once, leave it open (nothing closes it), then hit Escape to stop a runaway agent: the breakdown silently collapses and the turn keeps running. The user has to notice nothing happened and press Escape again. Meanwhile a finished, styled context popover ships dead in the bundle.

*Fix* — Move the `snap.contextMenuOpen` branch below the `snap.turnActive` branch in App.tsx (or drop it entirely — a `<details>`-style inline expander should not claim Escape), and rename the flag to `contextBreakdownOpen` so it stops reading like a modal. Then either mount ContextMenu (Overlays.tsx:1160) behind its own flag — it is the better presentation of the same data, with the % meter DesktopPanels.tsx:949-953 lacks — or delete it along with the now-unused `compactTokens` call sites in it.

<details><summary>Evidence & verification</summary>

```
App.tsx:1822-1826 (GlobalKeys Escape chain)
  if (snap.contextMenuOpen) { event.preventDefault(); store.closeContextMenu(); return }
...and only eleven lines later:
App.tsx:1837-1840
  if (snap.turnActive) { event.preventDefault(); store.cancel() }

The flag no longer belongs to a modal. Its one consumer is a plain inline expander inside the Activity rail:
  DesktopPanels.tsx:948  <button className="context-breakdown-toggle" aria-expanded={snap.contextMenuOpen} onClick={() => store.toggleContextMenu()}>Token breakdown</button>
  DesktopPanels.tsx:949-953  {snap.contextMenuOpen && (... <dl className="context-breakdown"> ...)}
and store.ts:1740-1743 toggleContextMenu just flips it; nothing auto-closes it (no click-outside, no blur), so it stays true until the user clicks the toggle again.

The dialog the flag was written for is orphaned: Overlays.tsx:1160 `export function ContextMenu({ snap, onClose })` — a 60-line `role="dialog"` context-usage popover with a meter and per-bucket swatches — has zero `<ContextMenu` mount sites anywhere in the renderer (it is the only never-mounted component in the whole tree).
```

**Verifier:** All cited lines verified. App.tsx:1822-1826 places the `snap.contextMenuOpen` branch (with an early `return`) ahead of the `snap.turnActive` cancel branch at App.tsx:1837-1840, and the GlobalKeys preamble (App.tsx:1769) only bails for `defaultPrevented`/open native `<dialog>`, so the branch is live. The flag's only consumer is a non-modal inline expander: DesktopPanels.tsx:948-953, a `.context-breakdown-toggle` button plus a `<dl>` inside `<details className="activity-context">` in `ActivityPanel` — no dialog role, no backdrop, no focus trap. store.ts:1740-1777 only flips the flag; the sole resets are `resetInFlightRequests()` (store.ts:3491-3499, reconnect) and the session-switch fold (store.ts:3527-3537), so closing the rail leaves it `true` with the toggle unmounted. `ActivityPanel` renders in both `DesktopSheet` (DesktopPanels.tsx:205) and the non-modal `DesktopRail` (DesktopPanels.tsx:131, mounted App.tsx:171); the sheet's capture-phase `stopImmediatePropagation` Escape handler (DesktopPanels.tsx:149-161) blocks the window listener, but the rail has no Escape handler at all, so the bug is reachable there. The composer's Escape (App.tsx:1595-1603) only stops propagation for hints and its comment confirms global Escape is meant to stop the turn. CSS does not hide the toggle (atelier.css:364-370). Overlays.tsx:1160 `ContextMenu` (role="dialog", % meter, swatches, own Escape at 1165-1171) has no mount site or import anywhere in src/ or test/ — dead code, and atelier.css:383 styles `.atelier .session-diagnostics .ctxpop`, a selector that can never match since App.tsx:566 `session-diagnostics` never renders a ctxpop. Only correction: impact is one swallowed Escape (the branch clears the flag, so the second press cancels), and it requires the user to have expanded both the "Conversation usage" details and the Token breakdown; that narrowness puts it at low rather than medium, though the rail-closed case swallows Escape with no visual feedback at all.
</details>

### [keyboard] main.ts:694-702 ships the app's only menu with no Preferences item, no Help menu, and no New Session item, so ⌘, is the one binding labeled nowhere and there is no consolidated shortcut sheet — though Settings itself is always reachable via the sidebar button (App.tsx:683) and ⌘K palette (Overlays.tsx:1372), and ⌘N/1/2/3/tab are labeled at their own controls

**`xerxes/src/desktop/main.ts:694`**

*Impact* — Half the app's real bindings are undiscoverable: ⌘, (settings), 1/2/3 (approvals — only visible while a card happens to be on screen), Tab (hint completion), `g` (generate specialist), ⌘⇧O (open workspace). A macOS user looking for Settings checks the app menu first, finds nothing, and concludes the app has no preferences. The missing Help menu also removes the system Help-search field, which is macOS's built-in menu-command finder. There is no ⌘/ or "Keyboard Shortcuts" panel to fall back on.

*Fix* — Define one exported binding table (id, label, accelerator, scope) and drive both surfaces from it. In main.ts, replace the bare `appMenu` role with a spread that inserts `{ label: 'Settings…', accelerator: 'CmdOrCtrl+,' }` after About; add `{ label: 'New Session', accelerator: 'CmdOrCtrl+N' }` as the first File item; add a `{ role: 'help', submenu: [...] }` containing `{ label: 'Keyboard Shortcuts', accelerator: 'CmdOrCtrl+/' }`. In the renderer, render that same table as a new DesktopSheet panel (DesktopPanels.tsx already has the `names` map at :162 to slot it into) so ⌘/ opens a live, always-accurate cheat sheet.

<details><summary>Evidence & verification</summary>

```
The entire application menu is 9 lines, main.ts:694-702:
```
Menu.setApplicationMenu(Menu.buildFromTemplate([
  ...(process.platform === 'darwin' ? [{ role: 'appMenu' as const }] : []),
  { label: 'File', submenu: [
    { label: 'New Window', accelerator: 'CmdOrCtrl+Shift+N', ... },
    { label: 'Open Workspace in New Window…', accelerator: 'CmdOrCtrl+Shift+O', ... },
    { type: 'separator' }, { role: 'close' },
  ] },
  { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
]))
```
Electron's `appMenu` role expands to About/Services/Hide/HideOthers/Unhide/Quit — it contains no Preferences item. So ⌘, (bound at App.tsx:1771 → `store.openSettings()`) appears nowhere in the menu bar. There is no Help menu at all. ⌘N (App.tsx:1781) has no File item, while ⌘⇧N is spent on New Window. The only shortcut documentation that ships anywhere is four chips in the composer footer (App.tsx:1653-1656: ⏎, ⇧⏎, esc, ⌘K) and three in the palette footer (Overlays.tsx:1519-1521).
```

**Verifier:** The code claim is literally accurate — main.ts:694-702 is the app's only menu template, `appMenu` expands to About/Services/Hide/Quit with no Preferences item, there is no Help role anywhere in src/desktop, and ⌘, is bound only in the renderer (App.tsx:1772). But the impact is substantially overstated on two counts. (1) Settings is not undiscoverable: App.tsx:683 renders a permanently visible labeled "Settings" button in the sidebar footer (atelier.css:52-55 styles it visible, no hide rule), and Overlays.tsx:1372-1376 lists "Settings…" plus four deep-links in the ⌘K palette, with ⌘K itself advertised at App.tsx:1656. (2) "The only shortcut documentation is four composer chips and three palette chips" is false — App.tsx:624 prints ⌘N on the New session button, App.tsx:1158-1160 prints 1/2/3 on the approval buttons, App.tsx:1384/1395/1414 on question cards, App.tsx:1566 documents tab/↑↓/esc in the hints popup; these are point-of-use labels, a better pattern than a menu item. ⌘⇧O is in fact in the File menu, contradicting its listing as an undiscoverable binding. What survives is narrow: no Help menu (hence no macOS Help-search over commands), no consolidated shortcut sheet, and ⌘, is the single binding with no label at its own control. That is macOS menu-convention polish affecting discoverability of one accelerator, not lost capability — low, not high.
</details>

### [keyboard] ⌘N/Ctrl+N is a silent no-op whenever a turn is active or the app is offline — App.tsx:1781-1784 calls store.newChat() ungated while store.ts:1161 returns false without patching any state, so unlike the greyed sidebar button (App.tsx:621) and the hidden palette rows (Overlays.tsx:1401) the keyboard path gives no feedback, and none at all when the sidebar is hidden (atelier.css:422)

**`xerxes/src/desktop/renderer/App.tsx:1783`**

*Impact* — The core agentic workflow is "kick off a long task, then start another one while it runs." Exactly then, ⌘N does nothing at all — no toast, no error, no flash — so the app reads as frozen or as having dropped the keystroke. Combined with the palette hiding both "New task" and every "Task: …" row behind the same `!snap.turnActive` gate (Overlays.tsx:1400-1419), there is *no* keyboard route to any other session while a turn is active.

*Fix* — Make ⌘N mid-turn do what `openSessionNow` already does for session switching: call `store.bridge.openWorkspaceWindow(snap.cwd)` to open a fresh surface instead of returning false. If that is out of scope, at minimum have `beginFreshTask` patch a transient message (`{ error: 'Finish or stop the running task to start a new one' }`) on the refusal path at store.ts:1143 so the keystroke produces feedback. Separately, drop the `!snap.turnActive` gate on the palette's `session:` rows so ⌘K can still navigate mid-turn, routing each to `openWorkspaceWindow`.

<details><summary>Evidence & verification</summary>

```
App.tsx:1781-1784 fires unconditionally:
```
if (meta && event.key.toLowerCase() === 'n') { event.preventDefault(); store.newChat() }
```
but store.ts:1143 bails out with no user-visible state change: `if (this.openingSession || this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)`. The mouse path is honest — App.tsx:621 `disabled={!online || snap.turnActive}` greys the New session button. The palette hides the action entirely, Overlays.tsx comment at :1402: "openSession silently no-ops mid-turn; newChat too — offering them as runnable actions would just close the palette over a dead click." Meanwhile store.ts:1077-1079 already implements the right answer for the sibling case: mid-turn `openSession` calls `this.bridge.openWorkspaceWindow(...)` to spawn a second window rather than refusing.
```

**Verifier:** Core claim verified in source, with two corrections. App.tsx:1781-1784 does call store.newChat() unconditionally on Cmd/Ctrl+N, and the refusal path is real but sits at store.ts:1161 (not 1143): `if (this.openingSession || this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)` with no patch() before the return, so the keystroke produces no state change at all — mid-turn AND while offline (the offline case is unmentioned in the claim). The mouse surfaces are honestly gated (App.tsx:621 disabled={!online || snap.turnActive}; store.openTaskModal early-returns at store.ts:1799), and the Overlays.tsx:1401-1403 comment documents only the palette decision, not the keyboard one — so the silent keystroke is an oversight of a known fact, not a documented design. The impact is overstated on two points: (1) a mid-turn keyboard route does exist — main.ts:697 registers File > New Window at CmdOrCtrl+Shift+N calling createWorkspaceWindow(), which works during a turn (it opens a workspace-less window per main.ts:247, clumsier, but the claim's "no keyboard route" is false); (2) the greyed sidebar button renders a literal <kbd>⌘ N</kbd> beside it (App.tsx:624), so whenever the sidebar is visible the user has an adjacent explanation — that mitigation only fails in focus/narrow mode (atelier.css:422 .atelier--focus .side{display:none}, App.tsx:141), a persisted layout state, so it is reachable but not the default. No capability is lost: refusing a second in-window task mid-turn is a consistent deliberate decision across all mouse surfaces. This is a keyboard/mouse parity and feedback gap, not a workflow blocker, hence low rather than medium.
</details>

### [keyboard] The transcript scroller (App.tsx:932) has no tabIndex, no role="log"/aria-live, and no scroll shortcuts in GlobalKeys — so there is no named or discoverable keyboard path to page it (the only routes are Shift+Tab into a nested &lt;details&gt; then native Page/Home/End, or Chromium's auto-focusable-scroller fallback when the transcript has no focusable children), and streamed output is never announced to screen readers

**`xerxes/src/desktop/renderer/App.tsx:932`**

*Impact* — With focus in the composer, PageUp/PageDown/Home/End/Space are swallowed by the textarea; with focus anywhere else they hit the document scroller, which does not scroll. So the transcript is mouse-wheel-only. Reviewing a 500-line diff or a long test output the agent just printed forces the user off the keyboard entirely — in a tool whose entire premise is that you keep your hands on the keys. Screen-reader and keyboard-only users cannot reach the transcript content at all, since an unfocusable div with no landmark role is not in the tab order.

*Fix* — Give the stream element `tabIndex={0}` and `role="log" aria-label="Transcript" aria-live="polite"` at App.tsx:932 so it enters the tab order and Page/Home/End work natively once focused. Add to GlobalKeys a focus-independent pair — ⌘↑/⌘↓ scrolling the stream ref to top/bottom, and ⌥PgUp/⌥PgDn paging it — plus a ⌘L (or ⇧Esc) that toggles focus between the composer and the transcript, so there is a named way back and forth.

<details><summary>Evidence & verification</summary>

```
`.stream` is the scroll container — app.css:321 `.stream { flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-anchor: none; padding: 16px 20px; }` (atelier.css:441 confirms `overflow:auto`) — but it is rendered with no `tabIndex`, App.tsx:932: `<div className={\`stream${empty && !failedCard && !snap.question ? ' stream--welcome' : ''}\`} ref={ref}>`. A grep for `tabIndex` across App.tsx finds exactly one hit, App.tsx:247, on the unrelated runtime popover. Meanwhile the composer textarea claims and reclaims focus (App.tsx:1444 and :1520 `ref.current?.focus()`). There is no PageUp/PageDown/Home/End/⌘↑/⌘↓ handler anywhere in GlobalKeys (App.tsx:1764-1856) or in transcriptScroll.ts.
```

**Verifier:** The mechanical claims all check out, but the impact is materially overstated on two counts, so it survives only in weakened form.

VERIFIED: App.tsx:932 is verbatim `<div className={`stream${empty && !failedCard && !snap.question ? ' stream--welcome' : ''}`} ref={ref}>` — no tabIndex, no role, no onKeyDown. `grep tabIndex App.tsx` returns exactly one hit (App.tsx:247, tabIndex={-1} on the runtime popover). app.css:321 and atelier.css:441 confirm .stream owns the scroll. GlobalKeys (App.tsx:1763-1856) handles only Cmd+comma / Cmd+K / Cmd+N / Escape / 1-2-3 — no Page/Home/End/Cmd-arrow. transcriptScroll.ts contains zero keyboard handling (wheel-driven follow/unfollow only). main.ts registers exactly two accelerators (:697-698), neither scroll-related. The composer's onKeyDown (App.tsx:1583-1603) intercepts only Enter, Arrows-with-hints, Tab-with-hints and Escape, so Page/Home/End fall through to the textarea's native caret handling and are consumed. There is no role="log" anywhere; aria-live appears on the welcome-notices div (App.tsx:934) which renders only in the EMPTY branch, so the populated transcript announces nothing. Sibling scrollers in the same codebase DO carry tabIndex={0} + role/aria-label (DesktopPanels.tsx:850, 1033, 1085; markdown.tsx:142), so this is an omission rather than a house convention — that strengthens the finding.

OVERSTATED #1 — "cannot be scrolled by keyboard at all" is false. package.json:41 pins electron ^44.0.0 (bun.lock: electron@44.0.0) => Chromium ~144. (a) Chromium 130+ ships keyboard-focusable scrollers: a scroll container with no keyboard-focusable descendants is added to the tab order automatically, which covers the plain-prose transcript. (b) When the transcript does contain focusable children (App.tsx has 6 <details>; <summary> is focusable, plus the history-pager button at :936), Shift+Tab from the composer textarea lands focus inside .stream — nothing focusable sits between <Stream/> (:888) and the textarea except ComposerTaskSummary — and Page/Home/End then natively scroll the focused element's nearest scrollable ancestor, i.e. .stream. Tab is preventDefault'd only while hints are open (:1591), so that route is always available. So the correct statement is "no named or discoverable keyboard path," not "no keyboard path."

OVERSTATED #2 — "screen-reader users cannot reach the transcript content at all" is plainly wrong. NVDA/VoiceOver read static text through the virtual/browse cursor regardless of tab order; a non-focusable div is fully readable. The genuine a11y defect is narrower: no role="log" and no aria-live on the populated transcript, so streamed agent output is never announced.

Additionally, transcriptScroll.ts auto-pins to the tail (follow() in the useLayoutEffect), so the highest-frequency case — watching live output — needs no scrolling at all. The pain is confined to scrolled-back review of long diffs/test output, which is real but narrower than "forces the user off the keyboard entirely."

Fix still stands and is cheap: add tabIndex={0} role="log" aria-label="Transcript" aria-live="polite" at App.tsx:932 (matching DesktopPanels.tsx:1033's existing pattern), and add a focus-independent Cmd+Up/Cmd+Down + Alt+PgUp/PgDn pair to GlobalKeys operating on the transcriptScroll ref, plus a shortcut to toggle focus between composer and transcript.
</details>

### [layout] .wsmenu (app.css:720) sets neither left nor right and App.tsx:271 gives it no inline position, so the 340px Workspaces dropdown resolves to its static position at x=0 and opens in the window's top-left corner over the sidebar; its top:44px was sized for the base 36px .top (app.css:141) and now overlaps the 48px atelier topbar (atelier.css:415), and the .wschip anchor it was designed for (app.css:711) is never rendered in any component

**`xerxes/src/desktop/renderer/app.css:720`**

*Impact* — Unlike SessionMenu (App.tsx:473), which clamps itself with `Math.max(8, Math.min(menu.x, window.innerWidth - rect.width - 8))`, `.wsmenu` declares neither `left` nor `right` and gets no inline position. As a fixed box with `left:auto` it falls back to its static position, which for an abspos child of the `.app` column flex container is the container's start corner — x ≈ 0. So clicking "Recent workspaces" in the Workspace panel pops a 340px menu flush against the left edge of the window, nowhere near the button that opened it, covering the sidebar. `top:44px` was also written for the old 36px topbar; the atelier topbar is 48px, so the menu tucks 4px *under* the title bar.

*Fix* — Give WorkspaceMenu the same treatment SessionMenu already has: capture the trigger's `getBoundingClientRect()` when `toggleWorkspaceMenu` fires, store x/y in the snapshot, and set `style={{left, top}}` on App.tsx:271 clamped into the viewport. Minimum viable fix if that is too invasive: add `left:50%; transform:translateX(-50%); top:56px; max-width:calc(100vw - 24px)` to app.css:720 so it is at least centered and below the real 48px topbar.

<details><summary>Evidence & verification</summary>

```
app.css:720-725
  .wsmenu {
    position: fixed; top: 44px; z-index: 41; width: 340px;
    background: var(--x-screen); border: 1px solid var(--x-hairline);
    border-radius: var(--r-sm); padding: 4px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
  }
App.tsx:271  <div className="wsmenu" role="menu" aria-label="Workspaces">   // no style prop, no ref
DesktopPanels.tsx:1190-1193  onClick={() => { open(null); store.toggleWorkspaceMenu() }}  // "Recent workspaces" button
atelier.css:415  .atelier .top{height:48px;flex:0 0 48px;...}
```

**Verifier:** Verified and confirmed. app.css:720-725 declares `.wsmenu { position: fixed; top: 44px; z-index: 41; width: 340px }` with neither `left` nor `right`; `grep -rn "wsmenu"` across src/desktop returns only this rule and App.tsx:271, so nothing overrides it and there is no attribute-selector fallback in either stylesheet. App.tsx:271 renders the div with no `style` and no `ref`, unlike SessionMenu (App.tsx:471) which clamps itself via getBoundingClientRect. `.app` (app.css:155) is a plain column flex container with no transform/filter/contain, so the fixed box's containing block is the viewport and `left:auto` falls back to the static position. I reproduced the exact cascade in a served HTML page and measured it: computed `left` is "0px", rect.x = 0, rect.y = 44, topbar height 48. The stale offset is also explained precisely: base app.css:141 is `.top { flex: 0 0 36px }` (44 = 36 + 8 gap), and atelier.css:415 raises it to 48px. Additional supporting evidence the author missed: `.wschip` (app.css:711-719), the topbar chip this dropdown was designed to anchor under, is never rendered anywhere -- grep finds it only in CSS, zero TSX -- and the sole trigger (DesktopPanels.tsx:1189-1193) calls open(null) first, closing the panel the button lives in, so there is no visible anchor at all when the menu appears. One sub-claim is wrong: "tucks 4px under the title bar" is false on stacking -- `.top` has no z-index and is not positioned, so the z-index:41 menu paints over it; the defect is a 4px overlap, not occlusion. Severity downgraded from medium to low because the menu still renders fully on-screen, readable, and dismissable by backdrop click and Escape (App.tsx:1827-1830); nothing is clipped, unreachable, or functionally blocked, and the only entry point is a secondary button inside Workspace settings.
</details>

### [layout] Transcript column and composer dock never share a centreline: .stream's 32px padding plus a one-sided 8px scrollbar gutter vs .composer-wrap's 24px leaves the dock's centre 4px right at every width (measured), and 24px wider with an 8px left overhang and 16px right overhang once the pane drops below the 870px cap; the only padding compaction (atelier.css:517) is a window-width @media even though .chat is already container-type:inline-size

**`xerxes/src/desktop/renderer/atelier.css:517`**

*Impact* — Both columns cap at `--conversation-width:870px` and centre with `margin:0 auto`, but they centre inside different boxes: the stream's content box is `chat − 64 − 8` (32px padding each side plus the one-sided `scrollbar-gutter:stable` reserve) while the composer's is `chat − 48`. The composer dock is therefore 16px wider than the transcript column and its centre sits 4px to the right of it — a permanent, visible misalignment of the two most-looked-at edges in the app. Worse, the only rule that equalises the paddings fires on `@media(max-width:1000px)` — the OS window — while the chat pane's width is set independently by the sidebar and inspector. So the misalignment appears and disappears based on how wide the *window* is, not how wide the *conversation* is: a 618px chat inside a 1200px window is misaligned, while the same 618px chat inside a 980px window is not.

*Fix* — Give both surfaces one inset token and one gutter. Set `--stream-inset:32px` on `.atelier .chat`, use `padding:28px var(--stream-inset)` on `.atelier .stream` (atelier.css:441) and `padding:12px var(--stream-inset) 10px` on `.atelier .composer-wrap` (atelier.css:464), change `scrollbar-gutter:stable` to `stable both-edges` so the reserve is symmetric, and move the compaction at atelier.css:517 to `@container(max-width:700px){.atelier .chat{--stream-inset:16px}}` so it tracks the pane instead of the window.

<details><summary>Evidence & verification</summary>

```
atelier.css:441  .atelier .stream{padding:28px 32px;min-height:0;overflow:auto;font:15px/1.65 var(--sans);scrollbar-gutter:stable}
atelier.css:446  .atelier .stream__col{width:100%;min-width:0;max-width:var(--conversation-width);gap:16px;overflow-wrap:anywhere}
atelier.css:464  .atelier .composer-wrap{flex:none;position:relative;padding:12px 24px 10px;background:var(--x-screen)}
atelier.css:465  .atelier .composer-dock{width:100%;max-width:var(--conversation-width);margin:0 auto;...}
atelier.css:517  @media(max-width:1000px){...;.atelier .composer-wrap{padding:10px 16px}.atelier .stream{padding:20px 16px}}
```

**Verifier:** Core defect confirmed by measuring the shipped app.css+atelier.css in a real Chromium. .stream (atelier.css:441, padding 28px 32px, overflow:auto, scrollbar-gutter:stable) and .composer-wrap (atelier.css:464, padding 12px 24px 10px) are siblings in <main className="chat"> (App.tsx:827/888/894/1542), so they share a parent width; .stream__col keeps margin:0 auto from app.css:322 and both cap at --conversation-width:870px (atelier.css:413). app.css:147 ::-webkit-scrollbar{width:8px} forces a classic space-taking scrollbar, so the stream content box is chat-64-8 while the composer's is chat-48. Measured dock-minus-column edges: chat 950 in a 1400 window -> left +4, right +4, centre +4; chat 700 in a 1400 window -> column 628 vs dock 652 (dock 24px wider), left -8, right +16; chat 700 in a 980 window -> left 0, right +8, centre +4. Nothing later overrides these rules and there is no comment documenting the asymmetry (the block comment at atelier.css:411 in fact says "a bounded reading column", i.e. one column). The window-keyed @media(max-width:1000px) at atelier.css:517 is real and is the odd rule out, since .atelier .chat is container-type:inline-size (atelier.css:437) and the adjacent rule at 516 already uses @container. TWO CLAIMS ARE WRONG AND I CORRECTED THEM: the dock is 24px wider, not 16px, and it overhangs to the LEFT by 8px in that case; and the misalignment never disappears -- the one-sided scrollbar gutter keeps the centres 4px apart at every width, so the author's own counter-example (618px chat in a 980px window) is still misaligned (right edges 8px apart). The proposed fix is also partly unworkable: @container(max-width:700px){.atelier .chat{--stream-inset:16px}} cannot style .chat from its own container query; the rule must target .stream/.composer-wrap. Severity stays low: purely visual, 4px in the common wide-pane case, no effect on what the user can do or understand.
</details>

### [navigation] Sidebar lights two destinations at once: App.tsx:618 marks "Sessions" selected whenever page !== 'agents', so opening Skills &amp; tools or Artifacts underlines Sessions (atelier.css:428) while the matching studio-nav row is also highlighted (atelier.css:49) — fix by testing page === null. (The claimed "no exit in focus mode" is false: the Topbar toggle at App.tsx:198 restores the sidebar, and five other call sites plus the session-change effect at App.tsx:160 call open(null).)

**`xerxes/src/desktop/renderer/App.tsx:618`**

*Impact* — Open Skills & tools or Artifacts and the sidebar highlights TWO destinations simultaneously: the "Sessions" segment (because the test is `page !== 'agents'`) and the studio-nav row you actually opened. The segment is the strongest selected-state in the sidebar, so the chrome insists you are looking at Sessions while the conversation is replaced by a catalog. And since that mislabeled segment is also the only control that calls `open(null)`, toggling the sidebar off (Topbar, App.tsx:198) while a page is open strands the user on the page — `.atelier--focus .side{display:none}` removes the only exit, the page has no close button, and Escape cancels the turn instead.

*Fix* — Make the segment reflect the real state: `className={page === null ? 'is-selected' : ''}` for Sessions, and make the Skills/Artifacts rows toggles (`onClick={() => open(page === 'extensions' ? null : 'extensions')}`) so clicking a lit row returns to the conversation. Add a "← Conversation" button to DesktopPage's header (DesktopPanels.tsx:114) so the exit does not depend on the sidebar being visible.

<details><summary>Evidence & verification</summary>

```
App.tsx:618  <div className="studio-segment"><button className={page !== 'agents' ? 'is-selected' : ''} onClick={() => open(null)}>Sessions</button><button className={page === 'agents' ? 'is-selected' : ''} onClick={() => open('agents')}>Agents</button></div>
App.tsx:625  <button className={`studio-nav${page === "extensions" ? " is-selected" : ""}`} … onClick={() => open('extensions')}>Skills & tools</button>
App.tsx:626  <button className={`studio-nav${page === "artifacts" ? " is-selected" : ""}`} … onClick={() => open('artifacts')}>Artifacts</button>
App.tsx:885  {page && <DesktopPage panel={page} snap={snap} />}   // inside <main className="chat">
atelier.css:422  .atelier--focus .side{display:none}
```

**Verifier:** Line 618 says exactly what is quoted and the dual-selection is real: with page==='extensions'|'artifacts', the Sessions segment test (page !== 'agents') is true while the studio-nav row also gets is-selected, and both render visibly (atelier.css:428 underline+title color; atelier.css:49 accent+accent-soft background). DesktopPage (DesktopPanels.tsx:114-115) genuinely has no close control and App.tsx:885 makes it a full takeover of the chat column. But the headline's second half — the strand/no-exit claim — is refuted. atelier.css:422 hides only .side; the sidebar toggle lives in the Topbar (.top, App.tsx:198) and is an unconditional boolean flip (App.tsx:167), so the same button that hid the sidebar brings it straight back. The segment is also not the only open(null): App.tsx:622 (New session), DesktopPanels.tsx:380, 465, 531, 555 all return to the conversation, and App.tsx:160's effect resets page on any cwd/sessionKey change (e.g. resuming a session from the command palette). What remains is a chrome state-display inconsistency that misinforms but never blocks: low, not medium.
</details>

### [perf] Composer's auto-grow layout effect (App.tsx:1461-1467) has `[draft]` in its deps, so every keystroke disconnects and reconstructs a ResizeObserver and re-adds a window resize listener, and runs `grow()` two to three times (inline at App.tsx:1582, in the effect, and again from the new observer's initial delivery) — each one a forced synchronous layout of an unvirtualized, uncontained transcript. Redundant work causing typing lag during streaming, not dropped characters; the per-keystroke sessionStorage write (App.tsx:1445 → drafts.ts:24,46-49) is deliberate draft durability and negligible by comparison. Fix: keep a `[draft]` effect that only calls `grow()`, and move the observer plus window listener into a `[]` effect driven by a `grow` ref — do not set the deps to `[]`, which would break auto-grow for Dictation (App.tsx:1608) and `xerxes:add-context` (App.tsx:1447).

**`xerxes/src/desktop/renderer/App.tsx:1467`**

*Impact* — Each character typed triggers: two forced reflows (inline grow + layout-effect grow), a ResizeObserver disconnect/construct/observe cycle whose first delivery forces a third layout, and a synchronous sessionStorage write of the entire draft. Stacked on top of the per-token whole-app re-render, typing a long prompt while the agent is streaming visibly drops characters — the worst possible latency in a coding tool, because steering mid-turn is the core interaction.

*Fix* — Change the layout-effect deps to `[]` and hold `grow` in a ref so the observer is created once; the inline `grow()` in onChange already covers the per-keystroke resize. Debounce the draft persistence: keep `transitionDraft` on `[key, workspace, snap.currentId]` only, and persist `draft` from a separate effect with a ~250 ms timer plus a flush on blur, unmount and submit (the in-memory `memory` map in drafts.ts already guarantees no data loss between flushes).

<details><summary>Evidence & verification</summary>

```
App.tsx:1461-1467 `useLayoutEffect(() => { grow(); const observer = new ResizeObserver(grow); if (ref.current?.parentElement) observer.observe(ref.current.parentElement); window.addEventListener('resize', grow); return () => { observer.disconnect(); window.removeEventListener('resize', grow) } }, [draft])` — deps include `draft`. `grow` (App.tsx:1452-1458) does `el.style.height = 'auto'` then reads `el.scrollHeight`, a forced reflow, and App.tsx:1582 already calls `grow()` inline in onChange. Separately App.tsx:1435-1444 `useEffect(() => { ... transitionDraft(draftSession.current, next, draft) ... }, [key, workspace, snap.currentId, draft])` and drafts.ts:46-49 `writeDraft` does a synchronous `storage.setItem(key, text)` on window.sessionStorage.
```

**Verifier:** The code claims are literally accurate — I read each line. App.tsx:1461-1467 really does build and tear down a ResizeObserver plus a window resize listener on every keystroke (deps `[draft]`), and `grow()` (App.tsx:1453-1459) forces a synchronous layout by writing `style.height='auto'` then reading `scrollHeight`. Because App.tsx:1582's onChange already calls `grow()` inline, and a freshly-`observe()`d ResizeObserver always delivers an initial callback, a single character produces roughly three forced layouts. That cost is real and not cheap: the transcript is not virtualized (App.tsx:901-909 maps all of `snap.blocks`), there is no React.memo anywhere in App.tsx, and neither app.css nor atelier.css puts `content-visibility` or layout `contain` on `.stream` (only `overscroll-behavior:contain`), so each forced layout relayouts the entire session DOM. The disconnect/reconstruct churn can also produce "ResizeObserver loop" notifications. However, two parts of the claim are overstated and I have downgraded severity accordingly. First, typing does NOT trigger a whole-app re-render: `draft` is local useState inside Composer (App.tsx:1428), so only the Composer subtree re-renders; the root `useSyncExternalStore` (App.tsx:115) with no memoization is what re-renders the whole unvirtualized tree per streamed token, and that separate, larger defect dominates the profile here. Second, the synchronous sessionStorage write is a small-string `setItem` (drafts.ts:46-49) and is the deliberate, commented durability mechanism (drafts.ts:39 "Window-scoped drafts survive renderer reloads without becoming shared history"; App.tsx:1425 "drafts belong to the durable session") — debouncing it is an optimization, not a bug fix, and it is not a plausible cause of input loss. "Visibly drops characters" is asserted rather than measured; main-thread jank causes lag and event coalescing, not discarded keystrokes. Finally, the proposed fix is partly wrong as written: setting the layout-effect deps to `[]` would regress auto-grow, because Dictation's onText (App.tsx:1608) and the `xerxes:add-context` listener (App.tsx:1447) set the draft without calling `grow()`, and today only the `[draft]` layout effect resizes the textarea for them. The correct shape is to keep a `[draft]` effect that only calls `grow()` and move the observer/window listener into a separate `[]` effect driven by a `grow` ref. The finding survives as a genuine but low-severity redundant-work defect, not a medium-severity input-loss bug.
</details>

### [settings] The New-task wizard has one entrance (a single ⌘K row, Overlays.tsx:1410) that advertises hint '⌘N', but ⌘N (App.tsx:1783) and the sidebar's New-session button (App.tsx:622) both call newChat() and skip the wizard — so it is undiscoverable; the settings it offers (model, plan-first, preset) all remain reachable afterwards from composer chips and Settings → Agents 'Use here'

**`xerxes/src/desktop/renderer/App.tsx:622`**

*Impact* — The TaskModal is the only place to pick an agent preset (App.tsx:370-385), pick a model (420-426), and set plan-first (406-418) for a new session — and it can only be reached by opening ⌘K and finding one row. The big "New session" button and ⌘N both drop you into a bare session where the agent preset silently inherits whatever the daemon defaults to. Worse, that palette row prints the hint `⌘N`, so a user who learns the shortcut from the palette will press ⌘N forever and never see the modal again, with no idea they lost preset/model/plan selection.

*Fix* — Point both entry points at the configured flow: `onClick={() => { open(null); store.openTaskModal() }}` at App.tsx:622 and `store.openTaskModal()` at App.tsx:1783 (keep `newChat()` on ⌥⌘N or a "blank session" item in the sidebar's overflow). If the bare path must stay primary, remove the `hint: '⌘N'` from Overlays.tsx:1410 so the palette stops advertising a shortcut that does something else.

<details><summary>Evidence & verification</summary>

```
App.tsx:619-624 — sidebar primary button: `onClick={() => { open(null); store.newChat() }}` with `<kbd>⌘ N</kbd>`
App.tsx:1781-1784 — `if (meta && event.key.toLowerCase() === 'n') { ... store.newChat() }`
store.ts:1143-1145 — `newChat(): void { void this.beginFreshTask() }`  // no preset, no model, no plan flag
Overlays.tsx:1410 — the ONLY caller of the modal: `{ id: 'new', icon: '＋', label: 'New task', hint: '⌘N', run: () => store.openTaskModal() }`
```

**Verifier:** Mechanics verified: App.tsx:622 and App.tsx:1783 both call store.newChat() (store.ts:1143-1145 → beginFreshTask() with no preset), and Overlays.tsx:1410 is the sole caller of store.openTaskModal() in the entire renderer while printing hint '⌘N'. No Electron menu duplicates it (grep over main.ts and main/*.ts is empty). However the stated impact is overstated and partly false: model is selectable at any time from the composer chip (App.tsx:1610-1618 → ModelMenu/ModelPicker, not hidden by atelier.css:60/472), plan-first is a persistent composer chip (App.tsx:1643-1650) plus a palette row (Overlays.tsx:1357) and a settings row (Overlays.tsx:298), and the agent preset can be bound to an already-created bare session via the 'Use here' button (Overlays.tsx:394-403 → store.selectAgentPreset), whose surrounding comment (Overlays.tsx:362-365) documents that a preset binds to an existing session and is refused mid-turn — exactly the state a newChat() session is in. So nothing is silently lost or irreversible; what remains is that the start wizard has a single entrance (one ⌘K row) whose advertised shortcut opens a different, wizard-less path, making the wizard undiscoverable. That is a labeling/discoverability defect, not a capability loss.
</details>

### [settings] Settings ▸ General "Plan this session" row is stateless and write-only: it always reads "enable now" and fires setPlanMode(true) regardless of snap.planMode (Overlays.tsx:293-298), unlike the three role="switch" rows directly above it (254-292), so Settings never shows whether the plan-mode ceiling is armed and clicking it while armed is a silent no-op RPC

**`xerxes/src/desktop/renderer/Overlays.tsx:293`**

*Impact* — Plan mode is the read-only ceiling that stops the agent mutating the repo, and it is exposed in four places with four behaviours: composer chip (toggle), palette (toggle, Overlays.tsx:1357), TaskModal (switch, App.tsx:411), and this row. This row alone is one-way — when plan mode is already on it still reads "enable now" and clicking it fires a no-op `setPlanMode(true)`, so a user who came to Settings to turn the ceiling OFF finds no control and concludes it is stuck. It also sits directly under a subtitle promising "nothing here is per-task", while being the one genuinely per-session control on the tab.

*Fix* — Replace the chipbtn with the same `role="switch"` markup the neighbouring rows use, bound to `snap.planMode` with `onClick={() => store.setPlanMode(!snap.planMode)}`, and disable it with an explanatory title when `snap.currentId === ''`. Reword Overlays.tsx:210 to "Theme, notifications and startup apply app-wide; the rows below marked 'this session' do not." — or move the plan row out of General entirely and let the composer chip own it.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:210 — `<p className="modal__sub">Applies immediately; nothing here is per-task.</p>`
Overlays.tsx:293-299 —
`  <div className="row__t">Plan this session</div>`
`  <div className="row__s">Review a plan before making changes</div>`
`  <button className="chipbtn" onClick={() => store.setPlanMode(true)}>enable now</button>`
The label is unconditional — no `snap.planMode` read anywhere in this row. Compare the composer chip (App.tsx:1643-1650) which is a proper toggle: `aria-pressed={snap.planMode}` / `onClick={() => store.togglePlanMode()}` / label flips `Plan first` ↔ `Work directly`. Every other stateful row in this card uses `role="switch"` + `aria-checked` (lines 258-291).
```

**Verifier:** Verified in the file. Overlays.tsx:295-298 is exactly `<div className="row__t">Plan this session</div>` / `<div className="row__s">Review a plan before making changes</div>` / `<button className="chipbtn" onClick={() => store.setPlanMode(true)}>enable now</button>`. grep for `planMode` in Overlays.tsx returns only lines 1357 and 1442 (the command palette), never GeneralCard — confirming the row reads no state. The three sibling rows (Launch at login 254-266, Notifications 267-279, Stream thinking 280-292) all use role="switch" + aria-checked bound to state, so this is inconsistency rather than a documented pattern; no comment in GeneralCard justifies it. store.setPlanMode (store.ts:1565) unconditionally calls set_plan_mode {enabled:true}, so a click while plan mode is already on is a redundant round-trip with no visible change. Not CSS-hidden: `.atelier .chipbtn` (atelier.css:327) and `.atelier .row__t` (275) style it normally, so the row does render. TWO parts of the claim fail. (1) Impact is overstated: plan mode is never stuck on. The composer chip (App.tsx:1643-1650) is a correct toggle with aria-pressed={snap.planMode} and a label flipping 'Plan first' / 'Work directly'; its class `cchip` is styled by atelier.css:60 and :472, not hidden, and it sits permanently above the input. ⌘K also offers 'Exit plan mode' (Overlays.tsx:1357). A user cannot reach a state with no way to disable it. (2) The 'contradicts the subtitle' angle is wrong: modal__sub (line 210) says 'nothing here is per-task', and plan mode is per-session, not per-task — the Session row (231-238) and Daemon row (239-247) are session/app-scoped too, so the subtitle is literally accurate. What survives is a real but low-impact defect: one of four plan-mode surfaces misreports state. The markup half of the proposed fix stands — role="switch" + aria-checked={snap.planMode} + onClick={() => store.setPlanMode(!snap.planMode)}, disabled with an explanatory title when snap.currentId === '' since setPlanMode round-trips a session_key — but the subtitle rewrite is unnecessary.
</details>

### [states] Settings → Models has no provider loading state: Overlays.tsx:488 gates "No saved provider profiles" only on providerError, so the first open (and every "Retry loading providers" click, since store.ts:2287 clears the error while providers is still []) falsely tells a configured user they have no profiles until provider_list resolves; separately, setSettingsTab('models') re-fires the three RPCs openSettings already issued, with no in-flight guard

**`xerxes/src/desktop/renderer/Overlays.tsx:488`**

*Impact* — A user with four configured providers opens Preferences and, for one render pass, is told they have none and invited to add one — then the list pops in and the whole card shifts. Clicking the Models tab from any other tab issues a redundant `provider_list` + `provider_types` + `runtime.status` round trip that was already in flight from `openSettings`, and because there is no guard the two responses race to patch the same `providers` array.

*Fix* — Add `providersLoading` to the snapshot, set it at the top of `loadProviders` and clear it in the `Promise.all` handler; early-return when it is already true so the tab click cannot double-fetch. In Overlays.tsx:488, render a two-row skeleton while `snap.providersLoading && !snap.providers.length`, and only show "No saved provider profiles" once a load has completed.

<details><summary>Evidence & verification</summary>

```
Overlays.tsx:488-496 `{snap.providers.length === 0 && !snap.providerError && (<div className="row">… <div className="row__t">No saved provider profiles</div><div className="row__s">Add a provider to configure a model.</div> …)}` — the only guard is `providerError`; there is no loading flag anywhere in the provider state (store.ts:791-798 lists `providers/providerError/providerSwitching/providerSwitchError/providerModels/providerModelLoading/providerModelWarnings/providerTypes` — no `providersLoading`).
store.ts:2268-2321 `loadProviders()` fires `provider_list`, `provider_types` and `runtime.status`, patching only on resolution, with no in-flight guard.
store.ts:1578-1584 `openSettings()` already calls `this.loadModels(); void this.loadProviders(); void this.loadAgentPresets()`. store.ts:1594-1598 `setSettingsTab()` calls `this.loadModels(); void this.loadProviders()` again, and Overlays.tsx:53 wires every tab button to `store.setSettingsTab(tab.id)`.
```

**Verifier:** Anchor is exact: Overlays.tsx:488-496 gates the "No saved provider profiles" empty state only on `!snap.providerError`, and there is no loading flag anywhere (grep for `providersLoading` returns zero hits across xerxes/src and xerxes/test). store.ts:791 `providers: []` is the only initializer and nothing ever resets it; store.ts:2286 `loadProviders()` patches `providerError: ''` then only writes `providers` at 2334 inside `Promise.all`, so from the first `openSettings` (store.ts:1596-1603) until the `provider_list` round trip resolves, a user with configured profiles is told they have none and invited to add one, with the card shifting when the list arrives. The claim missed a second, more visible instance: because 2287 clears `providerError` at the top of the retry, clicking "Retry loading providers" (Overlays.tsx:486) swaps the error card for the false "No saved provider profiles" row for the whole retry. Not a deliberate decision: the same file guards the identical pattern correctly at Overlays.tsx:1310 (`!snap.reasoningLoading`) and 712 (`modelsLoading`), and desktopShell.test.tsx:338 shows the author only handled the error-vs-empty case. Double-fetch is real too: openSettings fires loadProviders, then setSettingsTab('models') (store.ts:1609-1614), wired to every tab button at Overlays.tsx:52, fires provider_list + provider_types + runtime.status again with no in-flight guard. Downgraded to low because two parts of the claimed impact are overstated: the racing responses cannot corrupt anything (both resolve to equivalent rows and patch is idempotent, so the cost is three wasted RPCs), the flash lasts one round trip, self-corrects, recurs only once per app launch, and blocks nothing. Claim's store.ts line cites are also ~15 lines stale (openSettings is 1596 not 1578, loadProviders 2286 not 2268), though the code substance matches.
</details>

### [states] Sidebar search has no no-match empty state: App.tsx:676 gates the message on groups.length===0 only, so a zero-match query shows bare remembered-workspace headers with no message (workspaceGroups.ts:52-54 emits zero-row groups), or — with no remembered folders, e.g. remote mode — the misleading "No tasks yet" copy

**`xerxes/src/desktop/renderer/App.tsx:676`**

*Impact* — Type a query that matches nothing into the sidebar's "Search sessions…" box and one of two wrong things happens. If you have no remembered folders, the sidebar asserts "No tasks yet — your chats live inside the workspace folder" — reading as data loss rather than a filter miss. If you do have remembered folders (the normal case), `groups.length > 0`, so you instead get a column of empty folder headers with no message at all: no "no matches", no hint that a filter is active, no way to tell a filter miss from a failed session load.

*Fix* — Compute `const filtering = needle.length > 0` and branch: when `filtering && rows.length === 0`, render `No sessions match "{filter}"` with a Clear button that resets `setFilter('')`; when `!filtering`, keep the existing copy. Also suppress the zero-row remembered-workspace groups while a filter is active (pass `[]` as `remembered` to `groupByWorkspace` when `needle` is non-empty) so the empty headers do not swallow the message.

<details><summary>Evidence & verification</summary>

```
App.tsx:606-611 `const rows = [...snap.live…, ...snap.sessions…, ...(currentRow ? [currentRow] : [])].filter(match)` then `const groups = groupByWorkspace(rows, snap.cwd, snap.workspaceDirectories)`.
App.tsx:676-678 `{groups.length === 0 && (<div className="side__empty">{… : online ? 'No tasks yet — your chats live inside the workspace folder' : …}</div>)}` — the condition is purely `groups.length === 0`; the `filter` state (App.tsx:575) is never consulted.
workspaceGroups.ts:52-54 `for (const cwd of remembered) { if (!groups.has(cwd)) { order.push(cwd); groups.set(cwd, { cwd, rows: [] }) } }` — remembered workspaces are emitted as groups with zero rows.
```

**Verifier:** All three cited lines say what is claimed. App.tsx:606-611 filters rows by the search needle before grouping; App.tsx:676-678 gates the empty-state message on `groups.length === 0` alone and never consults `filter`/`needle`; workspaceGroups.ts:52-54 pushes remembered workspaces as zero-row groups that are emitted unconditionally by the map at lines 55-57. `.wgroup__cap` (app.css:216-224, atelier.css:50) styles those headers visibly with a divider, so a filter miss in the normal local case (main.ts:408 + workspaceSettings.ts:50 mean any user who has opened a folder has remembered dirs) renders a column of bare folder headers and no message at all. In remote mode `desktop:workspaces` returns [] (main.ts:408), so the same miss yields "No tasks yet — your chats live inside the workspace folder", which reads as missing data rather than a filter miss. No CSS, comment, or later code path handles it, and the codebase already ships a proper no-match empty state elsewhere (DesktopPanels.tsx:552, App.tsx:1550), so this is inconsistency rather than intent. Severity lowered to low: the claim's "no hint that a filter is active" is overstated — the typed query stays visible in the `.side__search` input (App.tsx:628-635) immediately above the list, recovery is clearing one field, and nothing the user can do is blocked or lost. The proposed fix (branch on `needle.length > 0`, show `No sessions match "…"` with a Clear button, and pass `[]` as `remembered` while filtering so empty headers do not swallow the message) is correct and cheap.
</details>

### [transcript] Tool results that are JSON objects/arrays without a top-level stdout/output string (glob, list_dir, memory, browser, skill and MCP tools) render via StructuredResult (Execution.tsx:78) and get no Copy output, Wrap lines, line count or full-screen dialog — those live only in the OutputViewer branch (OutputViewer.tsx:34-41), and Execution.tsx:83 offers Copy command only for shell tools; the labelled `Raw details` fallback (Execution.tsx:86) holds the complete text but is capped at 320px by app.css:400 with no copy button, leaving drag-selection as the only way to copy. Shell, read, grep, write and edit are unaffected.

**`xerxes/src/desktop/renderer/Execution.tsx:78`**

*Impact* — You cannot copy a grep hit list, a file read, or an MCP tool's result out of the transcript — the only path is manual text selection across a nested `<dl>` of disclosures, and for anything deeper than 4 levels or wider than 50 entries the data is not on screen at all. The fallback the UI points you at is a 320px-tall `<pre>` with no copy button either. For a tool whose users paste output into issues and commit messages all day, this is a dead end.

*Fix* — Always wrap the structured branch in the same chrome: render `<OutputViewer text={item.output} label="Result" />` alongside (or as the fallback tab of) `StructuredResult`, so `Copy output` / `Wrap lines` / expand exist for every tool. Add a `CopyButton` for `item.output` to `execution__actions` (Execution.tsx:82) unconditionally, not just when `view.command` is truthy. Note the codebase has two clipboard paths and they disagree: App.tsx:446 documents `execCommand` as the only one that works ("the renderer has no clipboard capability on file://", and main.ts:169 does `window.loadFile(...)`), while Execution.tsx:47 and OutputViewer.tsx:32 use `navigator.clipboard.writeText`. Pick one — route every copy button through `copyText` (App.tsx:447) or delete the stale comment after verifying the async API.

<details><summary>Evidence & verification</summary>

```
Execution.tsx:71 `const structured = view.stdout === null && output === item.output ? structuredOutput(item.output) : null`
Execution.tsx:78 `{output && (structured ? <>...<StructuredResult value={structured}/></> : <OutputViewer text={output} />)}` — the `OutputViewer` branch is the only one with copy: OutputViewer.tsx:35 `<button disabled={!text} onClick={copy}>{copied || 'Copy output'}</button>`, plus `Wrap lines` (line 34), the line count (line 39) and the expand dialog (line 41).
The StructuredResult branch truncates: StructuredResult.tsx:13 `if (depth >= 4) return <span className="result-muted">{entries.length} entries · available in Raw details</span>` and line 14 `entries.slice(0,50)` with `{entries.length-50} more entries in Raw details`.
The escape hatch it names is Execution.tsx:86 `<details className="execution__raw">...<strong>Result</strong><pre>{item.output || '(no result)'}</pre></details>`, and app.css:400 caps it: `.execution__raw pre { max-height:320px; overflow:auto; }` — with no copy button.
Execution.tsx:82-85 confirms the actions row only ever holds one button, for shell tools: `{view.command && <CopyButton text={view.command} label="Copy command" />}` followed by a blank line.
blocks.ts:242 shows why this branch dominates: `detailOf` JSON.stringifies any non-string `return_value`, so read/grep/glob/web-fetch/MCP results all land here.
```

**Verifier:** The core structural claim survives. Execution.tsx:71 computes `structured` and :78 branches: the StructuredResult path renders only a `Result` label plus an Expand/Compact toggle, while OutputViewer.tsx:34-41 is the sole carrier of Wrap lines, Copy output, the line count and the expand dialog. Execution.tsx:82-85 gates the only other copy button behind `view.command`, so non-shell tools get no copy at all. app.css:400 caps the `Raw details` fallback at 320px with no copy button. The branch is reachable and common: executors/toolRegistry.ts:432 `serializeToolResult` JSON.stringifies every non-string tool return, and blocks.ts:53-62 `detailOf` re-parses and pretty-prints it, so tools typed `Promise<JsonObject>`/`Promise<string[]>` (globFiles fileTools.ts:521, listDirectory :468, plus the memory/AI/browser/skill/MCP families) all land in StructuredResult. However several load-bearing details of the claim are wrong or overstated, which lowers severity: (1) read and grep do NOT land here — readFile (fileTools.ts:268) and grepFiles (:560) both return `Promise<string>` of non-JSON text and take the OutputViewer branch with a working copy button, as do edit/write/bash; (2) the stated mechanism is wrong — streaming/events.ts:44 declares `result: string` and wireEvents.ts:315-325 rejects non-strings, so `return_value` is always a string on the wire and blocks.ts:242 never stringifies an object; (3) not "silent" — StructuredResult.tsx:13,14 both print "available in Raw details"/"more entries in Raw details", and Execution.tsx:86 does render the complete `item.output`, so no data is off screen; (4) main.ts:701 registers `{ role: 'editMenu' }`, so select-plus-Cmd+C is a real (if awkward) escape hatch; (5) OutputViewer.tsx:14 `readableOutput` unwraps top-level `stdout`/`output` string keys, so JSON-object results carrying an `output` field also reach OutputViewer — the affected set is narrower than "any JSON object". Net: a genuine, verified affordance inconsistency with a working manual workaround and no data loss, not a dead end.
</details>


---

# Xerxes Desktop — UI/UX audit synthesis

## Themes

### T1. The `atelier.css` skin layer is load-bearing, and it deletes working features
**Root cause:** `atelier.css` is a later "skin" loaded last (`renderer/index.html:12-14`) and `.atelier` is applied unconditionally (`App.tsx:163`), so it can silently override anything in `app.css`. It tidies the base markup with blanket `display:none` selectors that are written against *classes*, not against semantics, and nothing in the test suite asserts that a named control has a non-zero box. Every rule is a decision to un-ship a feature taken by a stylesheet author who didn't know what was in the box.

**Blast radius:** `atelier.css:439-440,520` erase FleetChip + JobsChip (`App.tsx:744-813`), the plan-mode header chip, the agent-preset chip and the "Session log" export button. `atelier.css:474` erases all four composer key hints (`App.tsx:1653-1656`), which is the only on-screen documentation of ⌘K, ⇧⏎, and esc-to-stop; combined with the fact that `App.tsx:1777` is the *sole* palette trigger (no Topbar button, no menu item at `main.ts:694-702`), the command palette ships with zero affordance. `atelier.css:492` and `:524` go furthest: they `display:none` the entire `.chat` column when the Changes review or an expanded rail is open, which deletes the composer (`App.tsx:894`), ApprovalCard (`App.tsx:945`), QuestionCard (`App.tsx:954`) and the "needs input" badge whose own comment (`App.tsx:845-847`) calls it the last surviving signal. With `notify.ts:70` suppressing OS notifications while the window is focused and `store.ts:131` folding "waiting" into "working" on sidebar dots, a mid-review AskUser question is both invisible and unanswerable.

**Smallest change:** delete the four `display:none` rules and replace them with explicit styling; where a takeover is genuinely wanted (review), keep `.chat` in the DOM and collapse it to a persistent decision strip instead. Then add one DOM smoke test over the pty/Electron harness asserting `getBoundingClientRect().height > 0` for a named list of controls (palette trigger, export, stop, approval buttons, composer) — the class of bug is only cheap to prevent, never to find.

---

### T2. Shell surface state lives in `useState`, so nothing global can reason about it
**Root cause:** `panel` (`App.tsx:135`), `railChoice` (`App.tsx:142`) and `page` (`App.tsx:150`) are component-local, while every other overlay (`paletteOpen`, `settingsOpen`, `pickerOpen`, …) lives in the store. The store's overlay bookkeeping — which carefully patches every *other* overlay closed on open (`store.ts:1620`) — is structurally blind to half the shell. So is the Escape ladder (`App.tsx:1822-1841`), whose dep array at `:1856` cannot list them.

**Blast radius:** Escape over a DesktopPage (Agents / Skills & tools / Artifacts) falls through to `store.cancel()` and silently kills the running turn (`App.tsx:1837`), and `DesktopPage` has no close button (`DesktopPanels.tsx:114`) while `App.tsx:874` hides the tab bar — so Escape is the natural thing to try. `DesktopSheet` had to hand-roll its own capture listener (`DesktopPanels.tsx:150-161`) precisely because the ladder can't see it. The z-scale is inverted — sheets sit at `z-index:120` (`atelier.css:69`) above the palette at `50` (`app.css:1239`) and Settings at `40` (`app.css:1081`) — so ⌘K over Scheduled jobs mounts a focused, typable palette *under* the scrim where Enter can blind-run `/compact`. The 1/2/3 approval keys leak through every overlay that lacks a keydown stop, because the guard at `App.tsx:1769` tests `dialog[open]` and the renderer's only native `<dialog>` is `OutputViewer.tsx:41`. The sidebar lights two destinations at once because `App.tsx:618` tests `page !== 'agents'` instead of `page === null`. And the palette can navigate to five Settings tabs but not one of the nine `DesktopPanel` destinations (`Overlays.tsx:1351-1441`).

**Smallest change:** move `panel`/`page`/`rail` into the store next to `paletteOpen` (`store.ts:773-790`), derive one ordered `overlayStack` from it, and drive four things off that single list: the Escape ladder (innermost-first, `turnActive` last), mutual exclusion on open, the z-index scale in `tokens.ts`, and the global-key guard (`if (overlayStack.length) return` before the number branch). Delete `DesktopPanels.tsx:150-161` afterwards.

---

### T3. The keyboard contract is advertised in three unsynchronized places and honored in none of them
**Root cause:** there is no binding table. Shortcuts exist as (a) JSX `<kbd>` labels, (b) palette `hint` strings, (c) `GlobalKeys` branches — three independent literals — and the app has no focus management to make (c) reachable.

**Blast radius:** ApprovalCard prints "1 / 2 / 3" (`App.tsx:1154`) but `App.tsx:1846` bails whenever an INPUT/TEXTAREA has focus, and the composer is focused programmatically on session entry and after every insert (`App.tsx:1444,1447,1448,1520`) with zero `.blur()` calls anywhere in the renderer — so in the dominant type→⏎→approval flow, pressing `1` appends a digit to your next draft. The card is `role="alertdialog"` with no focus move, so screen readers announce nothing. ⌘N (`App.tsx:1783`) runs `newChat()` while the palette row labelled "New task ⌘N" (`Overlays.tsx:1405-1411`), TaskModal's own docstring (`App.tsx:309`) and `store.ts:313` all say it opens the task modal; mid-turn it is a bare `return` (`store.ts:1161`) with no feedback while the mouse path is honestly disabled (`App.tsx:621`). The Electron menu (`main.ts:694-702`) has no Preferences item, no Help menu and no Find. `⌘F` does nothing — `findInPage` is called nowhere in `main.ts`/`main/*.ts` — in an app whose primary surface is thousands of lines of tool output. The transcript scroller (`App.tsx:932`) has no `tabIndex` and no `role="log"`.

**Smallest change:** one exported `BINDINGS` table (`id, label, accel, scope`) that generates the Electron menu template, the palette hints, the on-screen `<kbd>` chips and the `GlobalKeys` switch, so a label can never disagree with a handler again. Separately, four lines in ApprovalCard/QuestionCard: `useEffect(() => ref.current?.querySelector('button')?.focus(), [approval.id])` — that single change makes 1/2/3 live, satisfies the alertdialog role, and moves focus off the textarea so the guard at `:1846` stops firing.

---

### T4. The approval card knows less than the wire does
**Root cause:** the renderer's `Approval` type is four scalars (`types.ts:181-188`) and `store.ts:3330-3344` parses only those, discarding `payload.inputs` — which the daemon deliberately attaches (`loop.ts:1664`), forwards (`turnRunner.ts:1550`) and the protocol validator goes out of its way to preserve (`wireEvents.ts:226-238`). The fallback description (`store.ts:3330`) then re-stringifies `arguments`, and `payload.reason` is never read at all.

**Blast radius:** under the shipped `accept-all` default (`permissions.ts:14`), the only approvals a user ever sees are the `ALWAYS_APPROVAL_TOOLS` (`permissions.ts:37`) — `send_message`, cron/schedule create, remote trigger — and none of them match a branch of `permissionDescription`, so they all render through the generic fallback at `permissions.ts:202-203`: `send_message(telegram)`, with recipient, body and schedule present in `payload.inputs` and thrown away. `permission_request` is yielded at `loop.ts:1042` *before* `tool_start` at `:1101` and carries no `tool_call_id`, so there is no transcript row to fall back to. `App.tsx:1161` then hardcodes "session policy: ask" regardless of `snap.permissionMode`, and the composer's Permissions chip (`App.tsx:1636-1642`) renders a constant word with the live mode only in a tooltip. Worse, cancelling a turn strands the card forever: nothing clears `approval`/`question` in `cancel()` (`store.ts:1015-1019`) or `turn_end` (`store.ts:3379-3417`), the daemon has already force-rejected it (`interactions.ts:212-222`), and every remaining button returns `ok:false` → "approval refused" while the card and the header badge stay pinned.

**Smallest change:** add `inputs` and `cwd` to `Approval`, carry `payload.inputs` through `store.ts:3330`, and branch ApprovalCard by `toolName` — `DiffPreview` (`DiffPreview.tsx:6`, today only wired to `DesktopPanels.tsx:1508` and `WorkspaceReview.tsx:29`) for edits, a mono block + workdir for Bash, a `<details>` of pretty JSON otherwise. In the same patch, clear `approval`/`question` in `cancel()` and `turn_end`, and add a ghost Dismiss.

---

### T5. Async surfaces model two states where there are four
**Root cause:** there is no convention. `loadProviderModels` (`store.ts:1255-1287`) implements per-surface `loading` + `warnings` correctly; its sibling `loadModels` (`store.ts:1231-1249`) implements neither, has no in-flight guard, and routes failures into a *transcript notification* that renders behind the popover that triggered them. Nobody enforces which pattern wins.

**Blast radius:** the model picker asserts "no models discovered yet" for the whole 8s `MODEL_DISCOVERY_TIMEOUT_MS` window and after every failure; its own retry button (`Overlays.tsx:1065`) can fail on every click with no feedback, and `fetch_models` is in `CONCURRENT_DISPATCH_METHODS` (`server.ts:397`) so each click really does race a second probe. Settings → Models says "No saved provider profiles" to a configured user until `provider_list` resolves (`Overlays.tsx:488`), and the Retry button clears the error first (`store.ts:2287`) so retrying shows the *false empty state* for its whole duration. `submissionPending` (`store.ts:916`) is held past `ok:true` across an unbounded git snapshot + LLM compaction (`server.ts:9967-9973`) with no timer and — because Stop is gated on `turnActive` (`App.tsx:851,856`) — no control at all, even though the daemon explicitly built a cancel path for exactly that window (`server.ts:9713-9727,9975-9990`). The Activity rail's 3s poll shares one busy flag with its first load (`DesktopPanels.tsx:863-877`), so an unreserved 40px "Working…" row enters layout every three seconds and re-disables "Stop watcher" (`:899`). Session switching reuses `connection:'connecting'` (`store.ts:1106,1164`), so every navigation paints "Reconnecting…" over the *previous* session's transcript with a Retry button that `store.ts:2374` makes inert. And the sidebar's no-match state (`App.tsx:676`) gates on `groups.length === 0`, never on the filter.

**Smallest change:** one `AsyncState<T>` = `{status:'idle'|'loading'|'ready'|'error', data, error}` in the snapshot per fetch surface, a rule that every consumer renders four branches, and a `navigating:'session'|'fresh'|null` field distinct from `connection` so deliberate navigation stops impersonating a dropped daemon.

---

### T6. Failure copy is written from the daemon's point of view, and nothing contains a failure
**Root cause:** `connectionFailureKind()` exists (`connectionFailure.ts:4`) and is consulted in three of five connection surfaces; the other two hardcode progress. Plus there is no React error boundary anywhere in the renderer (verified: zero `componentDidCatch`/`getDerivedStateFromError`), so `DesktopPanels.tsx:445,460` calling `records()` during render blanks the window to black on an unexpected daemon shape.

**Blast radius:** `App.tsx:887` claims "Connection lost. Retrying…" for configuration/session rejections that `beat()` (`store.ts:2435`) explicitly refuses to retry. First-run Setup step 2 (`Setup.tsx:70`) hardcodes "Connecting to the shared runtime…" because `setupReadiness.ts:7` never reads `snap.error`, so an auth/validation failure looks like a spinner, the Retry is futile by `connectionFailure.ts:4`'s own contract, and the Offline card carrying the real reason sits under a `z-index:150` backdrop (`atelier.css:131`). One process has two names — "Restart workspace runtime" (`App.tsx:252`) calls `restartDaemon()`, and the version popover contradicts itself in three lines (heading `App.tsx:250`, body `buildInfo.ts:69`, button `App.tsx:252`) — with the disconnected state carrying five labels across `App.tsx:218,887,1051,1534` and `TerminalsPanel.tsx:46`, one of which ("Connect to a daemon first…") is an imperative for a verb no button offers. The Log tab's entire explanation is "wire order… this ring" (`Workspaces.tsx:178`); the Changes footnote re-exposes `FileEditTool`/`WriteFile` that `blocks.ts:41 verbOf()` humanizes everywhere else. One object has three nouns (session/task/chat) inside a single sidebar column (`App.tsx:595,618,624,677`), with an unguarded `"1 tasks"` at `App.tsx:267`.

**Smallest change:** (a) a root `<ErrorBoundary>` that renders the error + a reload button instead of a black window — ~30 lines, the cheapest high-severity fix in this report; (b) branch every connection surface on `connectionFailureKind` and only say "Retrying…" for `'transport'`; (c) pick one user-facing noun for the process ("runtime") and one for the object ("task"), then add a grep-based lint over `.tsx` string literals for `/daemon|wire order|ring|fold\b|worktree/`.

---

### T7. Features were migrated without deleting the old branch or wiring the new entry
**Root cause:** partial refactors (`fc487d0d` removed the Changes tab button and added the topbar `open('review')`) leave the old state member, component and render branch behind, and nothing detects a union member with no producer.

**Blast radius:** `snap.tab === 'changes'` (`App.tsx:889`) is unreachable — `store.setTab` is only ever called with `'activity'|'plan'|'log'` — so ChangesTab (`Workspaces.tsx:22`), its per-file `undo` (`:85`), "Undo all" (`:56`), "Keep all" (`:52`), the `changes.undo` RPC and the `totals` memo (`App.tsx:818-821`) all ship unreachable, while `'changes'` stays in `WorkspaceTab` (`types.ts:210`). `ContextMenu` (`Overlays.tsx:1160`) — a finished 60-line role=dialog popover with a usage meter — has zero mount sites, while its old flag now drives a plain inline expander (`DesktopPanels.tsx:948`) that still swallows the first Escape ahead of turn-cancel. `snap.changes` is written only from the live `tool_call` event (`store.ts:3188/3445`) and `adoptHistory` (`store.ts:2493`) never re-folds restored history, so any resumed session shows "No files changed in this session yet." (`DesktopPanels.tsx:225`) while the git-backed Changes rail simultaneously lists those files as dirty.

**Smallest change:** a test that asserts every member of each state union has at least one producer (a grep-level assertion is enough), then for each orphan: wire it or delete it. Re-fold `editStatsOf` over restored history in `adoptHistory` — `blocks.ts:224` already computes it per replayed row.

---

### T8. Layout responds to the OS window; content lives in panes
**Root cause:** the panes are already `container-type:inline-size` (`atelier.css:437`), but every responsive fallback is an `@media` on window width, and `main.ts:143` sets `minWidth:760`, which makes the `max-width:700px` breakpoints unreachable dead code at default zoom.

**Blast radius:** at a 960px window (half a 1080p display — the canonical size for a coding assistant beside an editor) the chat column is 318px, so Skills & tools gets a fixed 230px nav + gaps + `.studio-form` 32px padding leaving ~4px for detail text, and Agents ~2px (`atelier.css:186,202,302,319-321`). The inspector divider can push `inspectorWidth` past the `contextRequiresFullWidth` threshold (`App.tsx:146`) that `display:none`s the divider itself (`atelier.css:526`), and the bad value is persisted (`layout.tsx:25`), so the rail permanently covers the chat at that window size with no control able to shrink it. `.wsmenu` (`app.css:720`) sets neither `left` nor `right` and `App.tsx:271` gives it no inline position, so the 340px workspace dropdown resolves to x=0 in the window corner — and the `.wschip` anchor it was designed for is never rendered by any component. The transcript column and composer dock never share a centreline (32px + one-sided scrollbar gutter vs 24px).

**Smallest change:** swap `@media(max-width:700px)` → `@container(max-width:700px)` at `atelier.css:202` and `:321` (a one-word change that un-breaks the 900–1150px band), clamp the divider `max` to `windowWidth - sidebarWidth - 320` at `App.tsx:170`, and give both stream and composer one `--stream-inset` token with `scrollbar-gutter:stable both-edges`.

---

### T9. The transcript is a prose renderer in a tool whose medium is code
**Root cause:** the transcript was built for chat blocks and never specialized for code. `markdown.tsx:122` captures `data-lang` on `<pre>` and nothing consumes it; there is no highlighting library in `package.json` at all; `Execution.tsx:78` sends structured results down a branch with no copy/wrap/expand chrome (those live only in `OutputViewer.tsx:34-41`), and the `Raw details` fallback it points at (`Execution.tsx:86`) is capped at 320px by `app.css:400` with no copy button.

**Blast radius:** no syntax highlighting and no copy button anywhere in the transcript; glob/list_dir/memory/browser/MCP results are uncopyable except by drag-selection; edit rows show `+3 −1` (`Execution.tsx:60`) and escaped JSON, never a diff. Switching to Plan or Log unmounts `Stream` (`App.tsx:888-891`), so `transcriptScroll.ts:41-47` re-pins to the tail and every disclosure (`App.tsx:1066,1107`; `Execution.tsx:54,67`; `OutputViewer.tsx:22-23`) re-collapses with no saved state. Auto-follow can only be re-armed inside a 64px band (`transcriptScroll.ts:60`) and the hook exports neither the flag nor a `scrollToLatest` (`:76`), so submitting while scrolled up renders your own message and the whole reply off-screen. Cross-session search discards the match: `SearchPanel.tsx:46` passes only `hit.sessionId`, and `store.ts:1108` hydrates the last 100 entries, so a deep hit opens a session that doesn't contain the matched message.

**Smallest change:** three surgical additions rather than a rewrite — a `CopyButton` + `data-lang` badge on `.md__code` and on `execution__actions` unconditionally; branch `ExecutionDetails` on `isEditTool` (already exported, `blocks.ts:94`) into `DiffPreview`; and render all three tab bodies with persisted `scrollTop`/`following` per session key instead of conditional mounts.

---

### T10. Selection and status exist only as CSS classes; async status exists only as mounted nodes
**Root cause:** the renderer expresses state through class names (`is-on`, `is-selected`) and expresses announcements through conditionally-mounted `role="status"` nodes. Both are invisible to assistive tech — and to the project's own macOS AX-tree parity testing (`docs/desktop-parity.md:206,251`).

**Blast radius:** no `aria-pressed`/`aria-selected`/`role="tab"` on the chat tabs (`App.tsx:876-880`), sidebar segment (`:618`), Settings nav (`Overlays.tsx:52`), theme or font-size segments (`:215-217,224`) — even though `DesktopPanels.tsx:127` already does it correctly and `atelier.css:193` already styles `[aria-pressed=true]`. Command palette and model picker have no listbox/option roles at all, and no surface uses `aria-activedescendant` (one hit in the whole renderer, `App.tsx:198`); the composer additionally traps Shift+Tab because `App.tsx:1591` matches `e.key === 'Tab'` without checking `shiftKey`. Every announcement is inserted already containing its text (`App.tsx:1570,1569,887`), so turn completion is announced by nothing. `TaskModal` (`App.tsx:337`) is the only `aria-modal` dialog with no `useDialogFocus` (`dialogFocus.ts:7`), and its permissions "change" affordance is a bare `<u>` (`App.tsx:430`) — the only one in the renderer.

**Smallest change:** a mechanical `aria-pressed` sweep across the five class-only selection groups (matching the `DesktopPanels.tsx:127` precedent), one persistent `sr-only` announcer pair at the App root written from the store on turn/approval/connection transitions, and `useDialogFocus(ref)` on `App.tsx:337`.

---

## Shortlist — ranked by impact ÷ cost

| # | Fix | Files | Cost | Why it ranks here |
|---|---|---|---|---|
| 1 | Delete the blanket `display:none` rules and style instead | `atelier.css:439-440,474,520` | ~1h + 1 test | One CSS diff un-ships-back FleetChip, JobsChip, plan chip, preset chip, transcript export, and all four keyboard hints incl. the only ⌘K/⇧⏎/esc documentation. Nothing else in this report restores as much per line. |
| 2 | Root `<ErrorBoundary>` with error text + reload | new file, `App.tsx:163` | ~30 lines | Today an unexpected daemon shape at `DesktopPanels.tsx:445,460` blanks the window to black with no recovery. Cheapest severe-failure containment available. |
| 3 | Focus the approval/question card on arrival | `App.tsx:1150,1201` | ~6 lines | Simultaneously makes the advertised 1/2/3 keys work, fixes the silent `role="alertdialog"`, and removes the draft-corruption path at `App.tsx:1846`. |
| 4 | Clear `approval`/`question` on `cancel()` + `turn_end`; add Dismiss | `store.ts:1015,3379`, `App.tsx:1157` | ~15 lines | Un-wedges a card that currently survives its own turn forever and pins the header badge to "needs input" for all subsequent turns. |
| 5 | `@media` → `@container` for the page/catalog breakpoints | `atelier.css:202,321` | one word ×2 | Un-breaks Agents and Skills & tools across the entire ~900–1150px window band, where detail text currently renders in ~4px. |
| 6 | Unify ⌘N and give refusals a voice | `App.tsx:1783`, `store.ts:1161,1798` | ~10 lines | Kills four findings at once: the shortcut/palette/docstring divergence, the orphaned task wizard, and the feedback-free mid-turn no-op. |
| 7 | Render Stop on `turnActive \|\| submissionPending`; 20s timer on `preparingSubmissions` | `App.tsx:851,856`, `store.ts:916` | ~20 lines | Removes the only state that fully bricks the composer, and finally reaches the cancel path the daemon already built at `server.ts:9713-9727`. |
| 8 | Un-gate mid-turn session navigation | `SearchPanel.tsx:48-51,114`; `Overlays.tsx:1404` | delete ~5 lines | The gate rests on a comment that `store.ts:1088-1093` made false on 2026-09-19; the sidebar already does the thing these two surfaces refuse. |
| 9 | Carry `payload.inputs` + `cwd` into `Approval`; per-tool preview | `store.ts:3330`, `types.ts:181`, `App.tsx:1148` | half a day | The highest-stakes click in the product currently shows `send_message(telegram)` while the recipient and body sit unread on the wire. |
| 10 | `navigating` snapshot field, distinct from `connection` | `store.ts:1106,1164`, `App.tsx:887,909,1531` | ~25 lines | Stops every session switch and ⌘N from impersonating a dropped daemon with a Retry button that `store.ts:2374` makes inert. |

**Just below the line** (same ratio band, slightly more cost): persistent needs-input strip in the Topbar so a decision survives `atelier.css:492/524`; `modelsLoading`/`modelsError` + in-flight guard on `loadModels` (`store.ts:1231`), which is onboarding-critical; the `aria-pressed` sweep (T10); `window.confirm` on provider delete (`Overlays.tsx:540`) and LSP "Remove server" (`LspPanel.tsx:96`), the only two unconfirmed hard-deletes in an app that confirms killing a terminal; `rAF`-coalesced `emit()` (`store.ts:3619`) plus passing `tool.output` as the second argument at `AgentRoster.tsx:27` so `detailOf`'s default param (`blocks.ts:459`) stops re-parsing the whole transcript per token.

---

## Three opportunities

### 1. Make the transcript a code surface, not a chat log
The app renders code the way a messaging client does. There is no highlighting dependency in `package.json` at all; `markdown.tsx:122` already extracts the fence language into `data-lang` and then drops it; there is no copy button anywhere in the transcript; `Execution.tsx:86` dumps tool results as a 320px-capped `<pre>`; and edits — the single most important thing an agent produces — render as `+3 −1` plus escaped JSON, even though `DiffPreview.tsx:6` exists and is wired only to snapshot restore (`DesktopPanels.tsx:1508`) and managed-workspace review (`WorkspaceReview.tsx:29`). On top of that there is no find: `main.ts:701` ships a bare `{role:'editMenu'}`, `findInPage` is called nowhere in the main process, and the cross-session FTS that does exist throws the match away (`SearchPanel.tsx:46`). A coding-agent desktop app's transcript should be highlightable, copyable, diff-rendering and searchable; all four are absent and all four have their inputs already in hand (`data-lang`, `item.output`, `isEditTool` at `blocks.ts:94`, `hit.messageIndex`).

### 2. Let the composer accept the things developers actually have
There are **zero** `onPaste`, `onDrop`, `clipboardData` or `DataTransfer` handlers in the entire renderer — you cannot paste a screenshot of a broken UI, drag in a log file, or drop a stack trace file. And there is no `@`-path completion: `hints.ts:24` fires only for `/`-prefixed drafts, so file references must be typed from memory — despite the daemon plainly understanding `@"path"` syntax, which `DesktopPanels.tsx:1080` emits from the Files sheet one file at a time via a modal that hides the conversation. The app already has a full workspace file tree (`WorkspaceFileTree.tsx:55`) and a `complete` RPC (`store.completeText`, `App.tsx:1506`). Adding paste/drop and extending `wantsHints` to `@` would turn two multi-click modal detours into typing.

### 3. Ship rewind-and-retry — the daemon already does the hard part
`cli.ts:1126` sets `autoSnapshotTurns:true` and `server.ts:1340` captures a git snapshot before **every** turn; `SnapshotsPanel` (`DesktopPanels.tsx:1415`) can already restore individual files (`:1515-1520`). None of that is connected to the transcript: there is no "revert to this turn", no "edit this message and retry", and the session-scoped undo UI that does exist — per-file `undo`, "Undo all", "Keep all" (`Workspaces.tsx:52,56,85`) with a live `changes.undo` RPC — is unreachable because nothing calls `store.setTab('changes')` (`App.tsx:889`). The canonical agentic-coding loop is *run → dislike the result → rewind to the prompt → rephrase*, and here it costs a slash command (`/undo-edits <path> --confirm`) plus a manual retype. Attach each turn's snapshot id to its user-message block, put a "Restore & retry" control on that row, and the feature is ~90% already built below the UI line.
