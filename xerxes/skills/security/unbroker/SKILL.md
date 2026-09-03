---
name: unbroker
description: Consent-gated workflow to find and remove a person's data from data-broker and people-search sites.
version: 1.0.0
author: SHL0MS (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [privacy, data-broker, opt-out, ccpa, gdpr, security]
source: https://github.com/NousResearch/hermes-agent/tree/main/optional-skills/security/unbroker
---

# Unbroker

Find where a person's personal information (name, addresses, phone,
email, relatives) is exposed on data brokers and people-search sites,
then remove it — automatically where possible, with a consolidated list
of human steps only where a site demands a CAPTCHA, government ID, or
phone call.

This skill does **not** defeat anti-bot systems, does **not** act on
anyone without recorded consent, and does **not** remove public records
(voter/property/court) or accounts the person controls.

## When to Use

- "Remove my data from data brokers / people-search sites."
- "Opt me out of Spokeo/Whitepages/etc."
- "Clean up my exposure after a doxxing incident."
- "Set up recurring privacy monitoring" (brokers re-list people).

## Hard limits (never overridden)

1. **Recorded consent first.** Before any scan or submission, confirm
   the subject is the operator or someone who gave recorded consent,
   and record: full name, aliases, emails/phones, city/state, prior
   locations, and the exact fields the operator allows disclosing
   (`disclosure_fields`). Never disclose fields beyond that list.
2. **No CAPTCHA circumvention.** Soft/managed challenges a normal
   browser passes are fine to complete; anything requiring a solver
   service or fingerprint spoofing becomes a human task instead.
3. **`confirmed_removed` only after a verifying re-scan** — an opt-out
   confirmation email is not proof.
4. **One consolidated human-task digest at the end of the run**, not
   interruptions at every blocked site.

## Procedure

### Phase 1 — Intake (the only required human touchpoint)

Collect the subject's details and consent, choose an autonomy level
(`assisted` = confirm each submission; `full` = the intake consent is
standing authorization for standard opt-outs), and create a ledger
file, e.g. `~/.xerxes/unbroker/<subject>.json`, recording every case as
`{broker, listing_url, state, evidence, disclosed, next_recheck_at}`.

### Phase 2 — Discover (read-only, parallel-safe)

Crawl every broker for the subject's listings before acting on any.
Most people-search sites render results as static HTML readable with
the web fetch tool; escalate to browser automation only for JS-only
sites. Record a verdict per broker: `found` / `not_found` /
`indirect_exposure` / `blocked` (anti-bot-gated sites get `blocked` and
move to the human-task list — do not fight them).

Useful free levers discovered here: the **California Data Broker
Registry** (one CA DROP request legally forces deletion from all
registered brokers) and ownership clusters.

### Phase 3 — Reduce (cluster and order)

Collapse findings into an action plan:

- **Ownership clusters are one action, not N** — a parent removal
  (e.g. Intelius/PeopleConnect) often clears many child sites
  (Truthfinder, Instant Checkmate, US Search). Order `found` cases
  **cluster-parents first**.
- Prefer **deletion over suppression** where a broker offers both —
  with one permanent exception: **PeopleConnect deletion wipes existing
  suppressions** and does not stop public-records re-listing, so use
  suppress-and-maintain there.
- Batch parallel `found` probes via subagents for large runs, but
  re-verify key `found` URLs yourself before trusting subagent
  self-reports.

### Phase 4 — Delete (sequential, irreversible)

For each case, parents first:

1. Open the broker's opt-out page (search "<broker> opt out" if the
   listing URL does not link it).
2. Fill the form with exactly the approved `disclosure_fields` — never
   more. If a broker demands more than planned mid-flow, stop that case
   and queue it as a human task.
3. For email-based opt-outs (CCPA requests): send from the operator's
   own address, then fetch the verification link from their webmail or
   inbox (the `himalaya` skill can read the inbox). Treat verification
   links skeptically — check the domain matches the broker before
   clicking.
4. Record state + evidence in the ledger: `submitted →
   verification_pending → confirmed_removed`, with `next_recheck_at`
   (brokers re-list; 30–90 days is typical).

### Phase 5 — Verify and report

Re-scan confirmed removals to confirm they stay gone, then deliver:

- Final status per broker (removed / suppressed / human task / blocked)
- The consolidated human-task digest (CAPTCHA, ID-upload, phone-call
  cases) with direct links
- A re-scan schedule for recurring monitoring

## Pitfalls

- A successful email send is not proof of delivery; the re-scan is the
  real confirmation.
- Never submit third-party/indirect records without fresh confirmation
  from the operator — only the subject's own direct listings qualify
  for standing consent.
- Sites sometimes reject the whole submission after a "correct" answer
  to a simple arithmetic question — that is bot fingerprinting; stop
  that case rather than retrying.
- Do not pivot into investigating other people whose names appear in
  listings — out of scope.

## Verification

- [ ] Consent and `disclosure_fields` were recorded before any action
- [ ] Every `confirmed_removed` is backed by a post-removal re-scan
- [ ] The ledger reflects each case's current state and recheck date
- [ ] The human-task digest covers every blocked or gated case

---

Adapted from the `unbroker` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright SHL0MS and Hermes Agent contributors.
