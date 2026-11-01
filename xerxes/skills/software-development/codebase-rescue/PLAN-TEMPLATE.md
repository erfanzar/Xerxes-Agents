# Rescue plan: <project name>

Status: <survey | in progress | paused | done>. Last updated: <date>.

## Goal

- What "fixed" means for the user:
- Behaviour that must keep working:
- Off limits without approval (public APIs, schemas, formats, flags, config, deployment):

## Baseline (exact commands and results)

| Check | Command | Result |
|---|---|---|
| Install / build | | |
| Start | | |
| Tests | | pass / fail / skip, duration |
| Type checker | | error count |
| Linter | | error count |

Known-broken features:

## Findings

| # | Class (dangerous / broken / blocking / structural / cosmetic) | Location (`file:line`) | Problem | Proposed fix |
|---|---|---|---|---|

## Slices (in order)

| # | Slice | Target shape (module, interface, seam) | Move (in place / strangler fig / rewrite) | Safety net | Status |
|---|---|---|---|---|---|

## Decisions

- <date>: <decision and the reason, especially anything the user approved or rejected>

## Progress log

- <date>: <slice>, <what changed>, tests <before -> after>, type errors <before -> after>
