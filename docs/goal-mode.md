# Goal mode

A goal is a durable objective attached to one session, plus the authority to keep
working on it without a person typing between attempts.

It replaces an earlier mechanism that inferred the same thing from prose: the
objective guard read the assistant's final message, looked for English phrases
like "verified" or "blocked by", and pushed a reminder back into the running turn
when it did not find one. That approach fails in three ways that this one does
not — it cannot tell a claim from a description of a claim, it cannot be steered
between attempts because everything happens inside one physical turn, and it
leaves nothing on disk, so a crash loses the entire run.

The design is modelled on DeepSeek Harness's goal subsystem
(github.com/deepseek-ai/deepseek-harness, MIT). Xerxes is Apache-2.0, so what is
shared is the design — phases, compare-and-set refs, whole-snapshot change
events, process-local activation, round attribution, the authority split — and
not the code.

## What a goal is

| Field | Meaning |
| --- | --- |
| `objective` | What the person asked for, in their words |
| `phase` | `active`, `paused`, `blocked`, or `complete` — durable |
| `blockedReason` | `{ code, message }`, present exactly while blocked |
| `maxGoalRounds` | The total number of automatic rounds this goal may spend |
| `roundsStarted` | How many it has spent |
| `activation` | `armed` or `disarmed` — **process-local, never persisted** |

The last row is the important one. **Phase** answers *what happened to the
objective*; **activation** answers *may this process keep working on it right
now*. They are deliberately separate, and only phase survives a restart. A
session resumed in a fresh process comes back `disarmed`, so reopening old
history can never silently restart autonomous work that nobody re-authorised.

## How a round runs

1. A turn ends and the session goes idle.
2. The daemon asks the round driver whether the goal wants another round.
3. If it does, the round is reserved in the durable log — the number is spent
   before the prompt exists, so a crash between the two costs one round rather
   than running one twice.
4. The round is submitted as **a real turn**: its own `turn_begin` / `turn_end`,
   its own auto-compaction check, its own entry in the transcript.

Point 4 is what makes a long run readable and steerable. The provider receives
the full `<goal_round>` brief; the transcript shows one line
(`Goal round 3/8 — <objective>`), so a person can follow twenty rounds without
scrolling past the same block twenty times.

A round stops being admitted when the goal is not `active`, is not `armed`, has a
human message waiting, or has spent its budget. Spending the budget is recorded
as `blocked` with code `round-limit` rather than silently ceasing, because a goal
that reads `active` while nothing advances it is indistinguishable from a bug.

The run also stops on the first round that fails or produces nothing, recorded as
`round-failed` or `round-produced-nothing`. This is not theoretical: against an
out-of-quota provider an early build spent all 24 rounds in nine seconds and
wrote nothing into the transcript but its own prompts. Note that the decisive
signal is the error notification, not the absence of output — the runtime renders
a provider failure as assistant text, so "did any text arrive" called every one
of those 403s a productive round.

## Who may do what

| Operation | Human turn | The goal's own current round | Subagent |
| --- | --- | --- | --- |
| `get_goal` | yes | yes | no |
| create / edit / pause / resume | yes | no | no |
| complete / blocked | yes | yes | no |

An automatic round may conclude the goal it belongs to; it may not redefine it.
"Its own current round" is exact — a turn opened for an earlier round, or under a
revision the goal has since moved past, carries no concluding authority, because
that is precisely when a completion claim is least trustworthy.

`blocked` is refused before the third consecutive round, so difficulty is not
mistaken for a blocker. That threshold is mechanical because it is a count.

Completion is not. An earlier build gated `complete` on mechanically detected
"verification evidence" — a command from a hardcoded list of names like `test`,
`build`, `lint`. A live run failed on it: the model wrote the file, proved it
with `cmp` (exit 0), was refused because `cmp` is not on the list, and deleted
its own correct output to start over. A whitelist cannot enumerate how a thing is
checked, and punishing a correct proof is worse than no gate. The requirement now
lives in the policy prompt and in the closing brief, where it can be about the
work rather than about which binary was invoked.

When an automatic round concludes a goal, the result carries a closing brief and
the model gets one more inference to address the person. The alternative — ending
the turn the instant the tool call lands — means the run's last visible act is a
tool call, and the person has to reconstruct the outcome from the transcript.

## Concurrency

Every mutation names the exact revision it expects (`{ goal_id, revision }`) and
is refused if the goal has moved. A refusal comes back as a tool *result*, not a
thrown turn: a stale revision is information the model should re-read and retry
against.

Every change event carries the complete post-mutation snapshot, so a partial or
reordered write cannot leave a half-applied goal. The replay fold is strict — it
rejects revision gaps, non-consecutive rounds, rounds past the cap, rounds
against a non-active goal, and creates over live work — because the log lives in
session metadata, which is a file that survives crashes and can be edited by
hand. When the log reaches its size bound it is *compacted* into the snapshot it
folds to, never truncated: cutting the head would produce a log
indistinguishable from a tampered one.

## Using it

```
/goal                                  show the current goal
/goal <objective>                      create and arm one
/goal edit <objective>                 change the objective in place
/goal pause                            stop automatic work, keep the goal
/goal resume                           re-arm it
/goal clear                            drop it, keeping a tombstone
```

Only those exact words are subcommands. `/goal clear the backlog` creates a goal,
because objectives are prose and prose starts with ordinary words.

Interrupting an automatic round pauses the goal, so it stays visible and
resumable. Interrupting your own turn disarms without changing the phase.

## Where it lives

| File | Role |
| --- | --- |
| `src/runtime/goalDomain.ts` | Event-sourced state, strict fold, activation |
| `src/runtime/goalTools.ts` | `get_goal` / `create_goal` / `update_goal`, authority, wrap-up |
| `src/runtime/goalRoundDriver.ts` | Round admission and the round prompt |
| `src/daemon/goalCommand.ts` | `/goal`, shared by every client surface |
| `src/daemon/server.ts` | The idle loop that submits rounds as turns |
| `src/agents/default/*.yaml` | Which modes see which goal tools |

One note on that last row, because it cost a whole live run: registering a tool
is not the same as exposing it. The default agent declares an explicit `tools:`
allow-list, so a tool absent from that list is invisible to the model however
correctly it was registered. The first live goal run spent its entire budget
replying "objective complete, the update_goal tool is unavailable".
