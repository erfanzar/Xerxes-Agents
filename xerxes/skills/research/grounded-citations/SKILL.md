---
name: grounded-citations
description: Ground answers and documents in cited, verifiable sources using a ledger-backed numbered citation system.
version: 1.1.0
author: Hermes Agent + Teknium (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [research, citations, grounding, sources, web, reports]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/research/grounded-citations/SKILL.md
---

# Grounded Citations

Every claim taken from an outside source gets an inline numbered citation and
a `Sources:` list. A citation ledger owns the `url → [n]` mapping so the
numbers and URLs come from retrieval, never from memory — the agent only ever
emits small integers the ledger handed it. The ledger is a plain JSON file the
agent maintains with the file write/read tools; there is no bundled script.

Ledger location: default to `~/.xerxes/cache/citations/ledger.json`, or a
task-local path for isolated work. Format:

```json
{ "sources": [ { "id": 1, "url": "https://example.com/a", "title": "A", "quotes": [] } ], "next": 2 }
```

## When to Use

Use whenever an answer or artifact rests on information you fetched rather
than knew:

- Research, comparisons, news summaries, "what is the current state of X"
- Any deliverable written to disk that quotes, paraphrases, or reports outside
  facts — reports, briefs, docs, decks, wiki pages
- Fact-finding where the user will want to check your work
- Multi-source synthesis where conflicting sources must be attributed

Skip inline citations when retrieval is incidental — a quick syntax or version
lookup mid-coding, casual conversation, creative writing. Mention a URL only
if the user would plausibly want the link.

For academic papers, feed this workflow from the `arxiv` skill; for
high-stakes fact-checking, pair with the `blocked-page-recovery` skill when a
source will not fetch directly.

## Procedure

**1. Reset the ledger** at the start of a task that will produce a grounded
answer or document. Skip the reset when continuing work whose ids are already
in a draft — reusing the ledger keeps numbering stable.

**2. Register every source at retrieval time.** After each web search or web
fetch, append an entry with the next id and read back the assigned `[n]`. Do
this *before* writing prose. Registering later, from memory, is the failure
mode this skill exists to prevent.

**3. Write cite-while-drafting.** Place the bracketed id(s) immediately after
each sentence the source supports:

```
Ice floats because it is less dense than liquid water.[1][2]
```

- No space before the bracket; each id in its own brackets.
- Max 3 ids per sentence. Cite per sentence, not one dump at the end.
- Only ids the ledger returned. Never invent an id or a URL.
- Claims from your own knowledge get no citation.
- Conflicting sources: present both readings, each with its own id.
- Quote exact figures, dates, and names as the source states them; flag gaps
  explicitly ("no source found for X") instead of smoothing them over.

**4. Append the Sources block.** Generate `## Sources` mechanically from the
ledger with the file write tool — one line per cited id, `[n] Title — URL`.
Never retype URLs by hand. For non-markdown targets, place footnotes in docx,
endnotes in PDF/LaTeX, a Sources slide in decks, and per-page source lists in
wiki output.

**5. Verify before delivering.** Re-read the draft with the file read tool and
check: every `[n]` exists in the ledger; the Sources block lists exactly the
cited ids with the ledger's URLs; source-bearing sentences are not left
uncited. Fix and re-check.

**6. Chat answers** follow the same steps in the reply: register sources,
cite inline, end with the rendered `Sources:` list.

## Fact-Checking Mode

For work where the reader must be able to check the chain — medical, legal,
financial, safety, disputed claims, or when the user asks for fact-checking:

**a. Attach a verbatim quote per source.** Save the extracted page text to a
file and store, in the ledger entry, the sentence(s) carrying each claim. The
quote must appear verbatim in the saved page text — copy-paste from the fetched
text, never retype or paraphrase. A paraphrase or misremembered figure cannot
masquerade as evidence.

**b. Flag model-knowledge claims with `[unverified]`.** A load-bearing claim
you could not source gets an explicit marker instead of a citation:

```
The refactor likely predates the 2.0 release.[unverified]
```

The goal is declared provenance for every claim, not a citation on every
sentence. A fact-check deliverable dominated by `[unverified]` markers should
say so in its summary.

**c. Cross-check disputed facts against a second independent source.** When
sources disagree, cite both readings with their own ids and quotes, and say
which you weight and why.

**d. Render an evidence block** for the deliverable: beneath each source's
URL, print its stored quotes so the reader sees claim → source → exact
supporting text with nothing taken on faith.

## Pitfalls

- **Registering after writing.** The ledger must be populated from tool
  output, not reconstructed from the draft.
- **Renumbering mid-task.** Ids are ledger identities; never hand-edit them in
  a draft. Reset only between tasks.
- **Retyping URLs into the Sources block.** Always regenerate from the ledger.
- **Citing a search snippet as if you read the page.** Fetch the page first
  when the claim needs the body.
- **Over-citing.** Three ids on a sentence is the ceiling.
- **Citing the ledger in code/config artifacts.** Source comments belong in
  prose deliverables, not generated code.
- **Quoting from a snippet instead of the page.** Evidence quotes must come
  from the extracted page text.
- **Using `[unverified]` as an escape hatch.** It marks the rare claim that
  genuinely cannot be sourced.
- **Hand-editing the Sources block.** Regenerate it from the ledger instead.

## Verification

- [ ] Every `[n]` in the draft exists in the ledger.
- [ ] The Sources block lists exactly the cited ids with the ledger's URLs.
- [ ] Every evidence quote appears verbatim in the saved page text.
- [ ] The cited share of source-bearing sentences meets the task's bar; read
      remaining warnings before delivering.

---

Adapted from the `grounded-citations` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent + Teknium.
