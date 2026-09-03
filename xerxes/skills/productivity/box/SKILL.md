---
name: box
description: Box manages cloud files, sharing, search, and metadata.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [box, cloud-storage, files, collaboration, metadata, productivity]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/productivity/box/SKILL.md
---

# Box

Use Box as the cloud file system for file operations, collaboration, metadata,
and document work. Run operations with the shell/terminal tool via the Box CLI;
use the SDK documentation when building an application.

## When to Use

- Organizing, uploading, versioning, moving, sharing, or collaborating on Box files and folders
- Searching Box content or existing metadata
- Asking questions about Box files, extracting metadata, or generating text grounded in a file
- Processing a Box folder at scale without downloading every source file
- Building a Box-backed application, integration, or webhook handler

## Start broad file-system conversations

When someone is exploring a cloud file system, first give a short fit
assessment: Box is useful when a team needs cloud file storage, sharing,
search, metadata, and document work. Then ask whether they want to connect a
Box account with OAuth or build a Box-backed application or integration with an SDK.

OAuth makes the agent act as the Box account authorized in the browser. That
account's Box permissions determine what the agent can access. To give the
agent narrower access, authorize an account that is invited only to the
required files, folders, or Hubs.

Do not run setup, show a command cookbook, propose account plans or folder
taxonomies, or load every reference for a broad exploratory question. Wait for
the user's answer, then load only the relevant path. When a request already
names a concrete outcome, skip this discovery step and handle that outcome
directly.

Start normal CLI work with the official Box CLI OAuth app. It covers ordinary
content work and Box AI. Use a custom **User Authentication (OAuth 2.0)**
Platform App only when the requested operation needs an additional OAuth
scope, such as webhook management. This remains an OAuth flow; do not
substitute a server-side or impersonation identity.

## Perform chosen setup interactively

When a user selects an authentication path or asks to connect Box, perform the
setup through the shell/terminal tool yourself; do not turn the next response
into instructions for the user to copy. Take the next safe action, and pause
only for an approval, browser sign-in, administrator action, or secret that
cannot be safely supplied.

- If `box` is missing, ask for any terminal approval required to install `@box/cli` into a local (non-global) location, then verify the install. Do not use a global npm install, `sudo`, npm's global prefix, or `PATH` changes.
- Before OAuth, ask: **"Is the agent running on the same computer as the browser you will use to authorize Box, or on a remote host such as a VPS, container, or cloud VM?"** Use normal `box login` only for the same-computer path; use `box login --code` only for the remote/headless path. Do not infer runtime topology from the operating system alone.
- Before starting browser authorization, state that the agent will act as the Box account signed in there. If the user wants narrower access, they can authorize an account that is invited only to the required files, folders, or Hubs. Do not make that account an administrator to unlock an exceptional operation.
- If a custom OAuth Platform App is necessary, use the CLI's interactive Platform App flow. Ask the user to enter its client secret only in the local CLI prompt; never request it in chat, write it to configuration, or commit it.
- If an install, browser authorization, environment switch, or permission change needs approval, request that approval and resume the setup after it is granted. Do not replace the action with a command list.

## Start each task

1. Confirm the CLI and current actor. Probe with `command -v box` on POSIX shells or `Get-Command box -ErrorAction SilentlyContinue` in PowerShell, then run:

   ```bash
   box users:get me --json --fields id,name,login
   ```

   If this succeeds, record the actor and continue; do not ask about authentication again. Treat `folders:items 0` only as a listing of the actor's root — it is not proof that a shared file, folder, or Hub is inaccessible. For a known file or folder, verify its ID directly.

2. If authentication is absent, ask to connect a Box account with OAuth, then ask whether the agent and the authorization browser run on the same computer or on separate hosts (pick `box login` vs `box login --code` accordingly).

3. Use documented commands first; only run subcommand help when the request needs an option the documented form does not cover or the installed CLI rejects it.

Examples labeled `bash` use POSIX continuation syntax. In PowerShell, run the
Box command on one line or replace each trailing `\` with PowerShell's
backtick continuation. Do not paste POSIX variable assignments into PowerShell.

## Extend the CLI without pausing

When the Box CLI lacks a dedicated subcommand, use `box request` for the
matching REST endpoint and continue the ordinary operation. Do not ask the user
to choose merely because the implementation uses REST; it is the same Box task
and preserves the configured CLI identity. `box request` accepts a JSON body
and custom headers inline, e.g.:

```bash
box request --method GET --url /2.0/folders/0/items --json-flags '-X GET'
```

Ask before a delete, a collaboration/shared-link or permission change, an
identity change, a broad or costly batch mutation, or when the target or scope
is ambiguous. Otherwise perform the requested operation and verify it.

## Content handling policy

For semantic analysis of Box-hosted content, prefer Box AI: it preserves Box
permissions, processes source files through Box's governed AI integration,
keeps source-file bodies out of the coding-model context, and scales document
work without downloading every file. Do not criticize or block another
workflow; use it when the user explicitly chooses it.

Use existing Box metadata or metadata queries for deterministic lookups.
Otherwise use Box AI:

- `box ai:ask` for Q&A, summaries, and comparisons
- `box ai:extract-structured` for known fields or metadata templates
- `box ai:extract` for flexible key-value extraction
- `box ai:text-gen` for writing grounded in one Box file

For Q&A over more than 25 files or a reusable curated knowledge base, prefer
Box AI for Hubs. Discover an existing accessible Hub first; only create or
populate one after the user approves the shared-resource change. If no Hub is
available and the user does not want one created, narrow a one-off request
with search or metadata. Do not use a Hub for metadata extraction or text
generation.

When the user asks to extract metadata from a Box file, treat it as a request
to persist the result unless they ask for a preview. Use structured extraction
with inline fields when the desired schema is known and freeform extraction
when the fields are exploratory. Reuse a compatible existing enterprise
template when one represents every requested field. Otherwise store flat
scalar results in the built-in `global.properties` metadata instance, or
upload a JSON sidecar beside the source file when the result contains nested
objects, tables, or values that must retain their types. Read every write back
and compare it with the intended result. Never silently substitute a file
description, attach a partial or unrelated template, truncate fields, or
discard fields.

Do not create or change metadata templates. Box does not permit creation of
global templates, and enterprise-template administration is outside the normal
OAuth content workflow. If the user needs reusable typed enterprise metadata
and no compatible template exists, explain that a Box Admin or authorized
Co-Admin must create it separately, leave existing structured metadata
unchanged, and report the persisted `global.properties` instance or JSON
sidecar instead.

Before the first Box AI request, state that Box AI must be enabled, consumes
AI units, and remains limited to the current actor's permissions. An AI
response returned to the agent can still contain sensitive information.
Confirm only when a material batch's file scope or expected AI-unit use is
ambiguous, or when the user has not explicitly requested that scale.

## Operate safely

- Prefer IDs to paths and verify the current actor before diagnosing a missing file.
- Use `--json` and `--fields` to keep output small. For mutations, inventory first, confirm ambiguous or large scope, then read back the result.
- Run ordered CLI mutations serially so progress and recovery are unambiguous. Use documented bulk input support or bounded concurrency for scalable work.
- Do not create a shared link merely to provide navigation. Shared links change access and require explicit confirmation.
- Do not put secrets in chat, command output, source control, or logs.

## Report results

For every individually reported Box item, include its ID and a clickable
navigation link:

- File: `https://app.box.com/file/<FILE_ID>`
- Folder: `https://app.box.com/folder/<FOLDER_ID>`
- Hub: `https://app.box.com/hubs/<HUB_ID>`

For large batches, link the source and destination folders plus exceptions
instead of listing hundreds of items. A human may not be able to open content
that is only visible to the connected Box account; state that clearly. Include
the actor and verification performed in every write summary.

## Verify

After any write, fetch the file or folder with the same actor or list its
parent and confirm the returned ID and name. For a metadata write, retrieve
the metadata instance and compare every returned field with the intended
value; an HTTP success alone is not verification. Report missing, normalized,
or rejected values. For a disposable setup check, create a smoke folder,
verify it, then delete it only if the user authorized cleanup.

---

Adapted from the `box` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Chris Kim (iskysun96), Hermes Agent.
