# Shading: layout reference notes (Lor 0.5.16)

Observed from screen captures of the running Lor app on 2026-11-13 (main
window 1800×1130 pt, captured at 1567×984, so capture px ≈ 0.87 pt) plus the
user's own screenshots of a chat, the artifact panel, the floating rail, the ⋯
menu and the terminal panel. These are measurements of what is on screen;
nothing was taken from Lor's bundle (code, CSS or icons). Colours are read
off screenshots and are approximate (±#050505).

## Window and panes
- One flat ground everywhere (≈ #1b1b1b dark). Sidebar and main pane share
  it; a single 1px hairline separates them. No header hairline in the main pane.
- Sidebar ≈ 243 pt wide (211 px in capture), resizable (AX "Resize sidebar").
- The right section pane, when open, is a raised card (≈ #1f1f1f, radius
  ≈ 10, inset ≈ 8 pt from the window's top, right and bottom edges), not a
  full-height column.
- Between the chat column and the section card there is a scroll-position
  marker strip: a short tick per message, the current one a longer "—".

## Sidebar (top to bottom)
- Row 1 (in the title bar, after the traffic lights): back, forward arrows;
  right-aligned search and sidebar-toggle ("Close sidebar") icons. Muted,
  ~14 pt icons.
- Nav rows, ~30 pt tall, 8–10 pt radius, icon + 13 pt label:
  New Chat (+), Omni Agent, Compute. Selected row = slightly lighter fill
  (≈ #262626), no border, no accent bar.
- "Recents": 11 pt muted label with a folder-plus ("New group") icon at its
  right. Hidden controls in the same header (AX): window popup ("All time"),
  order popup ("Needs you"), "Mark all as read", "New chat".
- Session rows: title (13 pt, ellipsis) with age ("11h") right-aligned in
  muted 11 pt. No status dots in the idle list. On hover: Pin, Archive,
  Move chat icon buttons.
- Bottom block: update pill ("Update to Lor 0.5.17", refresh icon; filled
  ≈ #2a2a2a when an update is ready — earlier capture read it as a faint
  teal tint), Feedback, Settings. Same ~30 pt row style as the nav rows.

## Main pane
- Header: task title as plain 13 pt text / faint chip top-left; "···"
  (Chat actions) at the right edge of the chat column. Nothing else.
- Empty state: blank canvas; only the composer.
- Chat column ≈ 745 pt (648 px capture), centred in the chat pane.
- User turn: full-column-width block, ≈ #262626, radius ≈ 10. The current
  prompt sticks at the top while scrolling; content fades out beneath it.
- Replies: plain prose, 13 pt, line height ≈ 19 pt (capture), paragraph gap
  ≈ 14 pt. Section labels are bold 13 pt paragraphs, not headings.
  Bullets indent ≈ 28 pt; ordered-list numbers flush. Inline code: mono 11 pt
  chip, ≈ #2a2a2a, radius ≈ 4. File paths render as clickable links.
- Turn meta: "Today · 6:11 AM" and a collapsible "Worked for 28s"; hook runs
  as a collapsed row ("lor.agent_hooks · 3 events"); "Load older messages".
- End of turn: "Files changed 1 +74 −25" card directly above the composer
  (gap ≈ 10 pt), ≈ #1f1f1f, hairline border, radius ≈ 10. Header 11 pt muted
  with green/red counts; rows: doc icon + path, +/− right-aligned.
- Floating rail (user screenshot): slim, rounded ≈ 14, lighter fill, no
  border; collapse, agent, layers (artifacts), files, terminal icons.

## Composer
- Same width as the chat column, ≈ #232323, radius ≈ 12, no visible border,
  ≈ 72 pt tall.
- Row 1: "+" at left, placeholder "Message"; right: model name 11 pt light
  grey ("Claude Opus 5.5"), effort fainter ("High ⌄"), mic icon, round send
  button (≈ 20 pt, grey fill ≈ #6b6b6b, dark ↑).
- Row 2, 11 pt muted, icon + text: laptop "Erfan's MacBook Pro", folder with
  full path, branch "main"; a busy spinner at the far right while working.

## Popovers and menus (shared style)
- Bg ≈ #262626, hairline border, radius ≈ 8–10, rows ≈ 24 pt, 12–13 pt
  label, hover/selected row = rounded fill ≈ #333. Submenus open with ▸ and
  the Right arrow key; Escape closes.
- Chat ⋯ menu: Pin P, Rename R, Mark as unread U · Fork F, Move to… M ·
  Copy ▸ (Session ID I, Link L, Transcript as Markdown T), Open in ▸
  (New window W, Browser B), Export as ZIP X · Mute notifications N ·
  Archive A. Icon left, grey single-letter shortcut right, hairline groups.
- Model/effort picker: one popover from the composer chip, anchored above
  and right-aligned to it, ≈ 175 pt wide. Sections with 11 pt muted headers:
  Effort (Low, Medium, High ✓, Xhigh, Max — check right-aligned) · Speed
  (Fast) · Model (Claude Opus 5.5 ▸ submenu).
- "+" menu: anchored above "+", ≈ 170 pt wide, overlays the Files changed
  card: Attach files (paperclip), Skills ▸ (sparkle), Context files ▸ (doc).

## Section panel
- Breadcrumb 11 pt muted, last crumb white: "Sections › Terminal",
  "Sections › Files › <file>", "Sections › Artifacts › <title>"; right icons:
  split, expand ⤢, collapse ».
- Files: two columns in the card. Left tree ≈ 210 pt: "Search in files"
  field + icons (search, hide ignored, collapse all, refresh); group
  "⌄ Changed files 12"; rows ≈ 22 pt: type icon (React atom blue, # purple
  CSS, {} yellow JSON, doc for md) + name coloured by status (blue modified,
  green added), circled letter badge right (M blue, A green). Right: path bar
  (sidebar toggle, doc icon, muted dir + white filename, ···), body preview;
  "Loading." muted centred while it loads.
- Terminal: tab pills (green dot, title, ×) and "+"; split/search/clear/⋯;
  mono prompt with cyan path, timestamp right.
- Artifacts (user screenshot): title, kind, size, edited time; pin, source
  toggle, ⋯.

## Type and colour
- System sans throughout, 13 pt UI, 11 pt meta; mono only for code, paths in
  the Files tree, and the terminal.
- Greys only; colour appears only in diff +/−, file-status badges, file-type
  icons and terminal output.

## More menus and popovers
- All popovers: bg ≈ #232323, radius ≈ 10, no border, rows ≈ 24 pt, 11 pt
  light-grey labels, hover/selected fill ≈ #2a2a2a radius ≈ 6, section headers
  10 pt muted, hairline between sections, ✓ right-aligned for the current value.
- "+" › Skills: submenu ≈ 240×316 to the right; inline "Search skills" (no
  field chrome); rows = name + muted description on one line, ellipsis;
  footer gear "Settings".
- "+" › Context files: same shape; "Search context files"; rows like
  "AGENTS.md", "~claude:CLAUDE.md" (scope prefix).
- Model submenu: ≈ 172×318, bottom-aligned with the effort popover; "Search
  models"; rows = name + context size muted ("Claude Opus 5.5 1M High ✓");
  footer "+ Add Models" and a refresh icon above a hairline.
- Recents header (visible on hover): "Recents ⌄", window popup "All"
  ("Show chats from": Last 8 hours / 2 days / 7 days / 2 weeks / All time),
  sort popup ("Sort by": Needs you / Activity / Last opened / Your last
  message), Mark all as read, +, New group.
- Right-click on a chat row = the same list as the ⋯ menu, at the cursor.
- Rename is inline in the sidebar row (title selected, Enter/Escape).
- Move to… is a modal (≈ 313×188, radius ≈ 12, padding ≈ 20) centred over the
  chat column: title, one-line subtitle, option rows (Pinned, Recents,
  New group…), × close, no footer buttons. App behind is blurred + dimmed.

## Search palette (⌘K)
- ≈ 640 pt wide, top-centred, radius ≈ 10, app behind blurred + dimmed.
- Input row ≈ 44 pt (magnifier, "Search chats, messages and commands"),
  hairline; group label "Recent"/"Commands" 10 pt; rows ≈ 34 pt with icon,
  title, muted age or shortcut, ↵ on the selected row.
- ">" switches to commands (New Chat ⌘N, New chat pane right ⌘D, …).
- Footer keycaps: ↑↓ navigate · ↵ open · esc close · > commands.

## Other pages
- Omni Agent: title chip "Observer", "Start a conversation." muted at the top
  of the column, composer at the bottom (row 2 shows machine + folder, no
  branch outside git).
- Compute: title 15 pt, search field + segmented "Clusters | My allocations",
  centred two-line empty state (no card, no icon).
- Floating rail with no panel open: ≈ 30 pt wide card at the top right
  (radius ≈ 10), icons «, agent, artifacts, file, copy, terminal at ≈ 26 pt
  pitch; chat ⋯ sits just left of it.

## Settings
- Settings replaces the sidebar: "← Back", "Search Settings" field, then
  General, Appearance, Organization, Billing, Models, Context, Sandboxes,
  Connectors, Channels, Automations, Notifications, Privacy, Voice Input,
  Keyboard Shortcuts, Usage, Archive, Extensions (≈ 30 pt rows, no group
  headers, selected fill ≈ #262626 radius ≈ 6). Escape leaves Settings.
- Page: title 14 pt semibold; content column ≈ 690 pt centred (Usage,
  Archive, Extensions use a wider ≈ 1070 pt column).
- Section label 12 pt regular (not uppercase) above each card; optional
  right-aligned action on the same line ("Add server", "New automation",
  "Using defaults").
- Card: ≈ #1f1f1f, radius ≈ 10, no border in dark (hairline ≈ #e6e6e6 in
  light); rows ≈ 50 pt, padding ≈ 11 pt, title 12 pt + description 10 pt
  muted (≈ #8a8a8a), hairline between rows; control right-aligned.
- Controls:
  - Plain muted text actions ("Sign out", "Open", "Create team").
  - Dropdown = value text + ⌄, no chrome; as a field: ≈ #151515, radius 6.
  - Segmented = muted labels, selected pill ≈ #2e2e2e radius 6, no track.
  - Toggle ≈ 28×16: on = white track + black knob (light: black + white);
    off = grey track (≈ #4a4a4a–#5a5a5a), knob left.
  - Number field ≈ 68 pt (≈ 110 for tokens), ≈ #151515, radius 6, value
    right-aligned, unit muted inside. Stepper = − field +.
  - Buttons: primary = white pill, black text; secondary = ghost pill
    (hairline or ≈ #262626 fill, radius ≈ 10, 10–11 pt); round "+" icon button.
  - Keycaps ≈ 18 pt, ≈ #141414, radius 4.
  - Leading icon tile ≈ 28 pt (≈ #141414 radius 6) for connectors/channels.
  - Status: green dot ≈ 5 pt; badge pill with tinted fill + 1px border
    ("Gets alerts"); progress bar 4 pt, white fill on ≈ #2e2e2e.
  - Empty state = bordered card with one muted paragraph, no illustration.
- Usage page: filter chips, four stat cards (radius ≈ 8), single-series blue
  bar chart with hairline grid, table with right-aligned numerics.
- Focused pane: 1 px blue (≈ #4a8fd8) outline around the content pane.

## Keyboard shortcuts (defaults)
Search ⌘K · Omni Agent ⌘⇧O · Settings ⌘, · Feedback ⌘⇧Y · Back ⌘[ ·
Forward ⌘] · New Chat ⌘N · Next/previous recent task ⌘⌥] / ⌘⌥[ · Switch task
⌃Tab / ⌃⇧Tab · Next/previous sidebar task ⌘⇧] / ⌘⇧[ (or ⌘⌥↓/↑) · Next/previous
unread ⌥⇧↓/↑ · Toggle sidebar ⌘B · Zoom ⌘= ⌘- ⌘0 · Focus next/previous area
F6 / ⇧F6 · Find ⌘F · Open sidebar task 1–9 ⌘1–9 · New window ⌘⇧N · Close
window ⌘⇧W · Paste plain ⌘⇧V · Toggle full view of section ⌘⇧X · Back from
Settings Esc · Search Settings /.

## Light theme
Same layout and sizes; ground #fff, sidebar ≈ #f7f7f7, hairline ≈ #ececec,
selected row ≈ #ececec, text ≈ #1a1a1a / muted ≈ #6b6b6b, user bubble and
code chips ≈ #f2f2f2. Cards, composer and rail become white with a hairline
or soft shadow instead of a lighter fill. Update row tint ≈ #e6eef2.

## Narrow widths (window 820×900 pt)
- The sidebar keeps its full ≈ 243 pt width; it does not collapse on its own
  (⌘B toggles it).
- The chat column fills the rest with ≈ 12 pt gutters; the rail stays at the
  top right, and the scroll ticks sit just left of it.
- The sticky user bubble, Files changed card and composer shrink to the column.
- Composer row 1 is unchanged. Row 2 chips truncate in the middle
  ("Erfan's …Book Pro", "/Users/erfan/Documen…jects/Xerxes-Agents", "m…").
- Sidebar titles truncate at the end with a fade before the age.
- The header title chip and ⋯ stay on one row.

## Not yet observed
Pressed and disabled states beyond those noted above (disabled = dimmed, as
on the forward arrow), notifications as delivered, error states. Triggering
those would mean sending a message or breaking something in Lor, so I left
them out.
